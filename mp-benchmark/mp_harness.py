"""
MP Shear-Modulus Rescue Harness
===============================
Port of experiment_harness_v2.py (the Borg/HEA harness) to the Materials
Project shear-modulus benchmark. The INVARIANT is preserved exactly:
every arm is identical except candidate GENERATION.

WHAT CHANGED FROM THE BORG HARNESS, AND WHY
  1. Oracle / admission come from mp_oracle (RF on Magpie features, VRH
     physics channel, distance-bounded admission at the pre-registered
     90th-percentile cutoff). Replaces canonical_oracle + admission +
     novelty_metric.
  2. TWO PARALLEL REPRESENTATIONS per point (the key structural change):
       - a Magpie FEATURE VECTOR (132-d) used by the GP surrogate, the
         oracle, and the distance/admission gate;
       - a COMPOSITION DICT {element: fraction} used by the digest arm's
         clustering and the mutation arm's perturbation.
     In Borg the feature vector CONTAINED the element fractions, so one
     object served both roles. Magpie features do not expose element
     fractions, so composition is threaded ALONGSIDE the vector. This
     keeps the digest and mutation arms operating on real elements —
     identical in meaning to Borg — preserving cross-benchmark
     equivalence. (Decision: Option 1.)
  3. objective is shear modulus G (GPa); "HV" text is parameterised.
  4. budget ~20 (from the rescuable-stagnation probe), not 50.

REUSED UNCHANGED (imported, not reimplemented):
  - exploration_digest.build_digest / render_digest  (the single source
    of truth guaranteeing identical information to LLM and digest arms)
  - the five-arm structure, stagnation detector, instrumentation schema,
    escape attribution, and return-dict shape (so the existing analysis
    layer attaches with minimal change).
"""

import numpy as np
import torch
import warnings

from gpytorch.mlls        import ExactMarginalLogLikelihood
from botorch.models       import SingleTaskGP
from botorch.fit          import fit_gpytorch_mll
from botorch.acquisition  import ExpectedImprovement

from mp_oracle import (
    MPOracle, FEATURE_COLS, composition_to_vector, admit_candidate,
    channel_b_vrh,
)
from exploration_digest import build_digest, render_digest

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

OBJECTIVE_NAME = 'G'          # shear modulus, GPa (parameterises digest text)
LLM_OVERSAMPLE = 4


# ══════════════════════════════════════════════════════════════════════════
# composition helpers (the parallel representation)
# ══════════════════════════════════════════════════════════════════════════

def dominant_cluster_comp(comp, thresh=0.01):
    """
    Cluster key from a COMPOSITION DICT: the two most-abundant elements,
    sorted, e.g. 'Fe-Ni'. This is the Magpie-space analogue of Borg's
    dominant_cluster, but reads the real composition rather than feature
    columns — so clustering means the same thing on both benchmarks.
    """
    parts = [(e, f) for e, f in comp.items() if f > thresh]
    parts.sort(key=lambda p: -p[1])
    return "-".join(sorted(e for e, _ in parts[:2]))


def build_digest_from_comps(obs_comps, obs_y, max_clusters=8):
    """
    Build THE exploration digest from composition dicts + objective values.
    Mirrors exploration_digest.build_digest but keyed on real compositions
    (Borg's build_digest reads feature columns; here we pass compositions).
    Returns the SAME dict shape, so render_digest and the digest arm read
    it identically.
    """
    clusters = {}
    for comp, y in zip(obs_comps, obs_y):
        key = dominant_cluster_comp(comp)
        if key not in clusters:
            clusters[key] = {'n': 0, 'best': -np.inf}
        clusters[key]['n']   += 1
        clusters[key]['best'] = max(clusters[key]['best'], float(y))
    ordered = sorted(clusters.items(), key=lambda kv: -kv[1]['n'])
    visible = dict(ordered[:max_clusters])
    seen = set()
    for key in visible:
        seen.update(key.split('-'))
    return {'visible': visible, 'n_hidden': max(0, len(clusters)-len(visible)),
            'n_total': len(clusters), 'seen_elements': seen}


def render_digest_G(digest):
    """render_digest with the objective label parameterised to G (GPa)."""
    lines = [f"  {key:<10} : {v['n']:>3} queries, best {OBJECTIVE_NAME} {v['best']:.0f}"
             for key, v in digest['visible'].items()]
    if digest['n_hidden']:
        lines.append(f"  (+{digest['n_hidden']} smaller clusters)")
    return "\n".join(lines)


def _formula_string(comp, thresh=0.01):
    parts = [(e, f) for e, f in comp.items() if f > thresh]
    parts.sort(key=lambda p: -p[1])
    return " ".join(f"{e}{f:.3f}" for e, f in parts)


# ══════════════════════════════════════════════════════════════════════════
# GP surrogate — identical to Borg (operates on Magpie feature vectors)
# ══════════════════════════════════════════════════════════════════════════

class GPSurrogate:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def fit(self, X, y):
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        tX = torch.tensor(Xs); tY = torch.tensor(ys).unsqueeze(-1)
        self.model = SingleTaskGP(tX, tY)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)     # uncapped, as in Borg (cap under-converges)
        self.model.eval()

    def ei_scores(self, X_cand, best_f):
        Xs = self.scaler_X.transform(X_cand)
        tX = torch.tensor(Xs)
        best_f_s = float(self.scaler_y.transform([[best_f]])[0][0])
        EI = ExpectedImprovement(model=self.model,
                                 best_f=torch.tensor([[best_f_s]]))
        with torch.no_grad():
            return EI(tX.unsqueeze(1)).numpy()

    def predict_point(self, x):
        Xs = self.scaler_X.transform(x.reshape(1, -1))
        tX = torch.tensor(Xs)
        with torch.no_grad():
            post = self.model.posterior(tX)
            mu_s = float(post.mean.numpy().flatten()[0])
            sd_s = float(post.variance.sqrt().numpy().flatten()[0])
        mu = float(self.scaler_y.inverse_transform([[mu_s]])[0][0])
        sd = sd_s * float(self.scaler_y.scale_[0])
        return mu, sd


# ══════════════════════════════════════════════════════════════════════════
# GENERATORS — the only thing that differs between arms.
# Each returns a list of raw composition dicts; admission is shared.
# ══════════════════════════════════════════════════════════════════════════

def generate_random(n_target, elements, rng, n_elem_range=(3, 6), oversample=4):
    comps = []
    for _ in range(n_target * oversample):
        k = int(rng.integers(n_elem_range[0], n_elem_range[1] + 1))
        els = rng.choice(elements, size=min(k, len(elements)), replace=False)
        fr = rng.dirichlet(np.ones(len(els)))
        comps.append({e: float(f) for e, f in zip(els, fr)})
    return comps


def generate_mutation(n_target, obs_comps, obs_y, elements, rng,
                      top_k=3, sigma=0.05, oversample=4, n_elem_min=3):
    """
    Gaussian perturbation of the top-k incumbents' COMPOSITIONS (not
    feature vectors). Same operation as Borg's mutation, on real element
    fractions threaded alongside the Magpie vector.
    """
    top_idx = np.argsort(obs_y)[::-1][:top_k]
    comps = []
    for _ in range(n_target * oversample):
        parent = obs_comps[int(rng.choice(top_idx))]
        els = list(parent.keys())
        vals = np.array([parent[e] for e in els], dtype=float)
        child = np.clip(vals + rng.normal(0, sigma, size=len(vals)), 0, None)
        if child.sum() == 0:
            continue
        child /= child.sum()
        comp = {els[i]: float(child[i]) for i in range(len(els))
                if child[i] > 0.01}
        if len(comp) < n_elem_min:
            continue
        tot = sum(comp.values())
        comps.append({k: v / tot for k, v in comp.items()})
    return comps


def _anchor_sample(anchor, elements, rng, n_elem_range):
    anchor = [e for e in anchor if e in elements]
    k = int(rng.integers(n_elem_range[0], n_elem_range[1] + 1))
    others = [e for e in elements if e not in anchor]
    n_more = max(0, k - len(anchor))
    extra = list(rng.choice(others, size=min(n_more, len(others)),
                            replace=False)) if n_more and others else []
    els = anchor + extra
    if len(els) < n_elem_range[0]:
        return None
    fr = rng.dirichlet(np.ones(len(els)))
    return {e: float(f) for e, f in zip(els, fr)}


def generate_digest_guided(n_target, obs_comps, obs_y, elements, rng,
                           n_elem_range=(3, 6), oversample=4):
    """
    PRIMARY causal control ('digest' arm). Identical logic to Borg:
    rank clusters by query count, uniformly sample the least-explored
    half, anchor a random composition there. Reads ONLY counts; ignores
    best-G. Uses the SAME digest object the LLM arm sees.
    """
    digest = build_digest_from_comps(obs_comps, obs_y)
    visible = digest['visible']
    keys = sorted(visible.keys(), key=lambda k: visible[k]['n'])
    remaining = keys[:max(1, len(keys) // 2)] if len(keys) >= 2 else keys
    unseen = [e for e in elements if e not in digest['seen_elements']]

    comps = []
    for _ in range(n_target * oversample):
        options = list(remaining) + (['__unseen__'] if unseen else [])
        if not options:
            anchor = list(rng.choice(elements, size=2, replace=False))
        else:
            pick = options[int(rng.integers(0, len(options)))]
            if pick == '__unseen__':
                anchor = list(rng.choice(unseen, size=min(2, len(unseen)),
                                         replace=False))
            else:
                anchor = pick.split('-')
        c = _anchor_sample(anchor, elements, rng, n_elem_range)
        if c is not None:
            comps.append(c)
    return comps


def generate_digest_ucb(n_target, obs_comps, obs_y, elements, rng,
                        n_elem_range=(3, 6), oversample=4, beta=None):
    """SUPPLEMENTARY value-aware control ('digest_ucb'). UCB1 over clusters
    using best-G + exploration bonus. beta from data (G spread)."""
    digest = build_digest_from_comps(obs_comps, obs_y)
    visible = digest['visible']
    if beta is None:
        beta = float(np.std(obs_y)) if len(obs_y) > 1 else 1.0
    N_total = max(int(sum(v['n'] for v in visible.values())), 1)
    scored = [(key, v['best'] + beta*np.sqrt(np.log(max(N_total,2))/max(v['n'],1)))
              for key, v in visible.items()]
    unseen = [e for e in elements if e not in digest['seen_elements']]

    comps = []
    for _ in range(n_target * oversample):
        if unseen and rng.random() < 0.5:
            anchor = list(rng.choice(unseen, size=min(2, len(unseen)),
                                     replace=False))
        elif scored:
            anchor = max(scored, key=lambda s: s[1])[0].split('-')
        else:
            anchor = list(rng.choice(elements, size=2, replace=False))
        c = _anchor_sample(anchor, elements, rng, n_elem_range)
        if c is not None:
            comps.append(c)
    return comps


def generate_llm(obs_comps, obs_y, elements, stagnation_count, iteration,
                 budget, intervention_log, n_request=12):
    """LLM arm. Delegates to mp_llm_proposal (built separately). Returns
    raw composition dicts; admission is harness-side."""
    try:
        from mp_llm_proposal import llm_propose_compositions
        return llm_propose_compositions(
            obs_comps=obs_comps, obs_y=obs_y, available_elements=elements,
            stagnation_count=stagnation_count, iteration=iteration,
            budget=budget, intervention_log=intervention_log,
            n_request=n_request)
    except Exception as e:
        print(f"    [llm] proposal failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════
# stagnation detector — identical to Borg
# ══════════════════════════════════════════════════════════════════════════

def is_stagnating(best_history, window=5, threshold=0.02):
    if len(best_history) < window + 1:
        return False
    recent, previous = best_history[-1], best_history[-(window + 1)]
    return (recent - previous) / (abs(previous) + 1e-12) < threshold


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP — one arm, one seed
# ══════════════════════════════════════════════════════════════════════════

def run_arm(arm, oracle, pool_vectors, pool_comps, elements,
            dedup_tol, n_initial=8, n_iterations=20, inject_n=3,
            inject_cooldown=3, stagnation_window=5, stagnation_thresh=0.02,
            random_seed=42):
    """
    Run one arm for one seed on the MP benchmark.

    pool_vectors : (N, 132) Magpie feature matrix of the candidate pool
    pool_comps   : list of N composition dicts, aligned with pool_vectors
    oracle       : MPOracle for this split (carries admission_threshold)
    dedup_tol    : near-duplicate distance in scaled feature space

    Two representations are threaded in lockstep: obs_X (feature vectors,
    for GP/oracle) and obs_comps (composition dicts, for digest/mutation).
    """
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)

    surrogate = GPSurrogate()
    pool_X = [v for v in pool_vectors]      # list so we can append injected
    pool_c = list(pool_comps)
    queried = set()

    # scaled reference for admission dedup (observed ∪ pool ∪ batch)
    def scaled(mat):
        return oracle.scaler.transform(np.asarray(mat))

    # ── initial sampling ──────────────────────────────────────────────────
    init_idx = rng.choice(len(pool_X), size=min(n_initial, len(pool_X)),
                          replace=False)
    obs_X, obs_y, obs_comps = [], [], []
    for idx in init_idx:
        obs_X.append(pool_X[idx])
        obs_y.append(oracle.query(pool_X[idx]))
        obs_comps.append(pool_c[idx])
        queried.add(idx)
    obs_X = np.array(obs_X); obs_y = np.array(obs_y)

    best = float(obs_y.max()); best_hist = [best]

    inject_events, candidate_tags, intervention_log = [], {}, []
    calibration_log, trajectory_log, stagnation_trace = [], [], []
    stag_counter, last_inject, terminated_early = 0, -inject_cooldown, False

    for it in range(1, n_iterations + 1):
        surrogate.fit(obs_X, obs_y)
        stagnating  = is_stagnating(best_hist, stagnation_window, stagnation_thresh)
        cooldown_ok = (it - last_inject) >= inject_cooldown

        stagnation_trace.append({
            'iter': it, 'stagnating': bool(stagnating),
            'cooldown_ok': bool(cooldown_ok), 'stag_counter': stag_counter,
            'injection_fired': bool(arm != 'none' and stagnating and cooldown_ok)})

        # ── injection ────────────────────────────────────────────────────
        if arm != 'none' and stagnating and cooldown_ok:
            import time as _time
            _t0 = _time.perf_counter()          # generation cost (point 10)
            if arm == 'random':
                raw = generate_random(inject_n, elements, rng)
            elif arm == 'mutation':
                raw = generate_mutation(inject_n, obs_comps, obs_y, elements, rng)
            elif arm == 'digest':
                raw = generate_digest_guided(inject_n, obs_comps, obs_y, elements, rng)
            elif arm == 'digest_ucb':
                raw = generate_digest_ucb(inject_n, obs_comps, obs_y, elements, rng)
            elif arm == 'llm':
                raw = generate_llm(obs_comps, obs_y, elements, stag_counter,
                                   it, n_iterations, intervention_log,
                                   n_request=inject_n * LLM_OVERSAMPLE)
            else:
                raw = []
            _gen_seconds = _time.perf_counter() - _t0

            # digest information density (minor issue a): with a 59-element
            # vocabulary the 8-cluster truncation hides proportionally more
            # history than on Borg. Record what the digest/LLM arms could NOT
            # see, so the identical-information claim is quantified, not assumed.
            _dg = build_digest_from_comps(obs_comps, obs_y)
            _digest_n_hidden = _dg['n_hidden']
            _digest_n_total  = _dg['n_total']

            # shared admission: reference = observed ∪ current pool ∪ TRAIN.
            # Including the oracle's TRAINING compositions is essential: it
            # blocks an arm (the LLM in particular, which knows the materials
            # literature) from proposing a near-copy of a training compound
            # that the RF would then score with inflated, memorised accuracy.
            # The Borg harness deduped against the full dataset for exactly
            # this reason; omitting the train split here would specifically
            # advantage the hypothesis arm. oracle._Xtr_s is already scaled.
            ref_scaled = np.vstack([scaled(np.vstack([obs_X] + [np.array(pool_X)])),
                                    oracle._Xtr_s])
            admitted, rejects = [], {}
            for comp in raw:
                res = admit_candidate(comp, FEATURE_COLS, oracle,
                                      ref_scaled, dedup_tol)
                if res.admitted:
                    admitted.append((res, comp))
                    ref_scaled = np.vstack([ref_scaled,
                                            oracle.scaler.transform(res.vec.reshape(1,-1))])
                    if len(admitted) >= inject_n:
                        break
                else:
                    rejects[res.reason] = rejects.get(res.reason, 0) + 1

            last_inject = it
            eid = len(inject_events); start = len(pool_X)
            for j, (res, comp) in enumerate(admitted):
                pool_X.append(res.vec); pool_c.append(comp)
                candidate_tags[start + j] = {
                    'event': eid, 'distance': res.distance,
                    'channel_b': channel_b_vrh(comp),
                    'queried': False, 'queried_at': None,
                    'improved': False, 'y': None}

            inject_events.append({
                'event': eid, 'iteration': it, 'arm': arm,
                'stag_at_trigger': stag_counter, 'n_proposed': len(raw),
                'n_admitted': len(admitted), 'rejects': rejects,
                'gen_seconds': _gen_seconds,               # point 10
                'digest_n_hidden': _digest_n_hidden,       # minor issue a
                'digest_n_total': _digest_n_total,
                'incumbent_before': best,
                'pool_indices': list(range(start, len(pool_X))),
                'vectors': [res.vec.tolist() for res, _ in admitted],
                'compositions': [comp for _, comp in admitted],
                'escaped': False, 'escape_latency': None})

        # ── acquisition ──────────────────────────────────────────────────
        unq = np.array([i for i in range(len(pool_X)) if i not in queried])
        if len(unq) == 0:
            terminated_early = True; break
        X_cand = np.array([pool_X[i] for i in unq])
        ei = surrogate.ei_scores(X_cand, best)
        pick = int(unq[int(np.argmax(ei))])

        # ── EI diagnostics for injected-but-unqueried candidates ─────────
        inj_diag = None
        inj_mask = np.array([idx in candidate_tags for idx in unq])
        if inj_mask.any():
            inj_pos = np.where(inj_mask)[0]
            best_inj_pos = inj_pos[int(np.argmax(ei[inj_pos]))]
            best_inj_idx = int(unq[best_inj_pos])
            mu_i, sd_i = surrogate.predict_point(pool_X[best_inj_idx])
            inj_diag = {
                'best_injected_pool_idx': best_inj_idx,
                'event': candidate_tags[best_inj_idx]['event'],
                'ei': float(ei[best_inj_pos]),
                'ei_of_pick': float(ei[int(np.argmax(ei))]),
                'ei_rank': int(np.sum(ei > ei[best_inj_pos]) + 1),
                'n_candidates': int(len(unq)),
                'mu': mu_i, 'sigma': sd_i,
                'picked_an_injected': bool(pick in candidate_tags)}

        mu, sd = surrogate.predict_point(pool_X[pick])
        x_new = pool_X[pick]; comp_new = pool_c[pick]
        y_new = oracle.query(x_new)
        queried.add(pick)
        calibration_log.append({'iter': it, 'mu': mu, 'sigma': sd, 'y': y_new})

        trajectory_log.append({
            'iter': it, 'pool_idx': pick,
            'is_injected': bool(pick in candidate_tags),
            'formula': _formula_string(comp_new), 'hv': float(y_new),
            'gp_mu': mu, 'gp_sigma': sd, 'inj_diag': inj_diag})

        prev_best = best
        obs_X = np.vstack([obs_X, x_new]); obs_y = np.append(obs_y, y_new)
        obs_comps.append(comp_new)
        best = float(obs_y.max()); best_hist.append(best)

        stag_counter = 0 if best > prev_best + 1e-9 else stag_counter + 1

        if pick in candidate_tags:
            tag = candidate_tags[pick]
            tag['queried'] = True; tag['queried_at'] = it; tag['y'] = y_new
            improved = y_new > prev_best + 1e-9
            tag['improved'] = improved
            ev = inject_events[tag['event']]
            if improved and y_new > ev['incumbent_before'] + 1e-9 and not ev['escaped']:
                ev['escaped'] = True; ev['escape_latency'] = it - ev['iteration']

    # ── aggregate metrics (identical definitions to Borg) ────────────────
    tags = list(candidate_tags.values())
    nc = len(tags); nq = sum(1 for t in tags if t['queried'])
    nh = sum(1 for t in tags if t['improved'])
    ne = len(inject_events); nesc = sum(1 for e in inject_events if e['escaped'])
    lat = [e['escape_latency'] for e in inject_events if e['escaped']]

    return {
        'arm': arm, 'seed': random_seed, 'best_history': best_hist,
        'final_best': best, 'terminated_early': terminated_early,
        'inject_events': inject_events, 'candidate_tags': tags,
        'intervention_log': intervention_log, 'calibration_log': calibration_log,
        'trajectory_log': trajectory_log, 'stagnation_trace': stagnation_trace,
        'n_injection_events': ne, 'n_injected_candidates': nc,
        'pickup_rate': (nq/nc) if nc else np.nan,
        'per_candidate_hit_rate': (nh/nc) if nc else np.nan,
        'per_event_rescue_success': (nesc/ne) if ne else np.nan,
        'median_escape_latency': float(np.median(lat)) if lat else np.nan,
        'escape_censored_frac': (1 - nesc/ne) if ne else np.nan,
        'mean_distance_injected': float(np.mean([t['distance'] for t in tags])) if tags else np.nan,
        'observed_y': obs_y}
