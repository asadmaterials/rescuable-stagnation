"""
Symmetric Injection Experiment Harness — v2
============================================
Rewritten for the Wave-1 rigor fixes. Changes from v1:

  A1/A5/D2 — ALL arms' candidates pass through admission.admit_candidate()
             (one shared path: descriptors, nan, simplex, Mahalanobis
             dedup, soft-stability ANNOTATION — no stability gating).
  A3       — hit tracking is PER-CANDIDATE over the whole run:
             every injected candidate is tagged with its event; when
             (if ever) it is queried, we record whether it beat the
             incumbent AT THAT MOMENT. Metrics:
               per_candidate_hit_rate, per_event_rescue_success,
               pickup_rate.
  A4       — escape latency is ATTRIBUTED: latency = iterations from
             trigger until one of THAT EVENT'S OWN candidates is queried
             and beats the incumbent-at-trigger. Never-escaping events
             are recorded as censored, and reported as median + censored
             fraction (never a mean over successes).
  A6       — true consecutive-stagnation counter passed to the LLM
             (was previously a constant equal to the window size).
  A2       — novelty = Mahalanobis distance in descriptor space
             (novelty_metric.py), reference = observed ∪ current pool.
             Legacy composition-Euclidean recorded as secondary.
  C2(rec)  — GP predictive mean/σ recorded at every pick for post-hoc
             calibration analysis (analysis happens in Wave 2).
  B3       — pool-exhaustion guard: terminated_early recorded; runner
             asserts budget ≤ pool size for the 'none' worst case.
  D1/D3/D4 — working-dataset constant, dead code removed, DESCRIPTOR_COLS
             single source of truth.

INVARIANT (unchanged): every arm identical except candidate GENERATION.
"""

import numpy as np
import pandas as pd
import torch
import warnings

from gpytorch.mlls        import ExactMarginalLogLikelihood
from botorch.models       import SingleTaskGP
from botorch.fit          import fit_gpytorch_mll
from botorch.acquisition  import ExpectedImprovement

from canonical_oracle import (
    CanonicalOracle, make_splits, get_feature_cols,
    get_composition_cols, DESCRIPTOR_COLS, load_working_dataset,
)
from admission      import admit_candidate
from novelty_metric import (
    descriptor_covariance, mahalanobis_novelty,
    composition_euclidean, default_min_novelty,
)

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

# D1 — single source of truth for the canonical dataset path
WORKING_DATA = '../data/processed/hea_hardness_working.csv'

# A2-new — all injection arms oversample by this factor before admission,
# so realized injection counts are equalized in expectation across arms.
LLM_OVERSAMPLE = 4


def _formula_string(x, feature_cols, comp_cols, thresh=0.01):
    """Readable formula from a feature vector, e.g. 'Cr0.25 Fe0.25 Ni0.25 Co0.25'."""
    parts = [(c, float(x[feature_cols.index(c)])) for c in comp_cols
             if x[feature_cols.index(c)] > thresh]
    parts.sort(key=lambda p: -p[1])
    return " ".join(f"{c}{v:.3f}" for c, v in parts)


# ═══════════════════════════════════════════════════════════════════════════════
# GP SURROGATE (identical across arms)
# ═══════════════════════════════════════════════════════════════════════════════

class GPSurrogate:
    """Standard single-task GP surrogate. Identical across all arms."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model    = None

    def fit(self, X, y):
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        tX = torch.tensor(Xs, dtype=torch.float64)
        tY = torch.tensor(ys, dtype=torch.float64).unsqueeze(-1)
        self.model = SingleTaskGP(tX, tY)
        mll        = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        # An earlier maxiter=50 cap (added for sandbox speed) was verified to
        # leave the MLL materially under-converged at the larger training
        # sizes the loop reaches late in a run: capped-vs-uncapped predictions
        # diverged by 30-50+ HV (comparable to the oracle MAE of 78), which
        # would alter acquisition decisions. Cap removed; fits run to
        # convergence. (Sweep showed maxiter<500 under-converges.)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def ei_scores(self, X_cand, best_f):
        Xs        = self.scaler_X.transform(X_cand)
        tX        = torch.tensor(Xs, dtype=torch.float64)
        best_f_s  = float(self.scaler_y.transform([[best_f]])[0][0])
        EI        = ExpectedImprovement(model=self.model,
                                        best_f=torch.tensor([[best_f_s]]))
        with torch.no_grad():
            return EI(tX.unsqueeze(1)).numpy()

    def predict_point(self, x):
        """Predictive (mean, sigma) in original HV units — for calibration."""
        Xs = self.scaler_X.transform(x.reshape(1, -1))
        tX = torch.tensor(Xs, dtype=torch.float64)
        with torch.no_grad():
            post = self.model.posterior(tX)
            mu_s = float(post.mean.numpy().flatten()[0])
            sd_s = float(post.variance.sqrt().numpy().flatten()[0])
        mu = float(self.scaler_y.inverse_transform([[mu_s]])[0][0])
        sd = sd_s * float(self.scaler_y.scale_[0])
        return mu, sd


# ═══════════════════════════════════════════════════════════════════════════════
# CANDIDATE GENERATORS — the ONLY thing that differs between arms.
# Each returns a list of raw composition dicts; admission is shared.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_random(n_target, available_elements, rng, n_elem_range=(4, 6),
                    oversample=4):
    """Random valid compositions (raw dicts; admission filters later)."""
    comps = []
    for _ in range(n_target * oversample):
        k     = rng.integers(n_elem_range[0], n_elem_range[1] + 1)
        elems = rng.choice(available_elements, size=k, replace=False)
        fracs = rng.dirichlet(np.ones(k))
        comps.append({e: float(f) for e, f in zip(elems, fracs)})
    return comps


def generate_digest_ucb(n_target, obs_X, obs_y, feature_cols,
                        available_elements, rng, stagnation_count=0,
                        iteration=0, budget=0, n_elem_range=(4, 6),
                        oversample=4, beta=None):
    """
    VALUE-AWARE information-matched control ('digest_ucb' arm).

    Receives the SAME inputs as the LLM arm — the per-cluster exploration
    digest INCLUDING best-HV per cluster (not just coverage), the top-5
    compositions, best-so-far, stagnation depth, and remaining budget —
    and consumes them with a FIXED, PRE-REGISTERED mechanical policy
    instead of open-ended reasoning.

    HONEST SCOPE OF THE COMPARISON (stated in methods, not overclaimed):
        LLM arm       = same inputs + open-ended policy
        this arm      = same inputs + ONE fixed policy (chosen a priori)
        'digest' arm  = coverage-only information + inverse-frequency rule
        random arm    = no information
        mutation arm  = local information (top-k only)

    So this arm licenses "the LLM beats THIS pre-registered mechanical
    policy", not "reasoning beats no-reasoning in general". The policy is
    a standard, respectable heuristic (UCB1 over composition clusters),
    deliberately not a strawman, and it is FIXED BEFORE the run so it
    cannot be tuned to the result.

    PRE-REGISTERED RULE (UCB1 over dominant-element clusters):
        score(c) = best_HV(c) + beta * sqrt(ln(N_total) / n_queries(c))
      - clusters are the same top-2-element keys the LLM's digest uses,
        and the same max_lines=8 truncation is applied, so this arm sees
        EXACTLY the digest the LLM sees (not a richer, untruncated view);
      - unobserved elements form an implicit "unseen" option that receives
        an infinite bonus (must-try), matching UCB1's optimism;
      - beta is set from the data (not hand-tuned): the observed HV spread,
        so exploitation and exploration terms are commensurable.
    Compositions are then sampled within the selected cluster: its two
    defining elements are always included, remaining elements drawn at
    random from the available set.
    """
    from exploration_digest import build_digest

    comp_cols = get_composition_cols(feature_cols)
    # Same shared digest builder as the LLM and 'digest' arms
    digest  = build_digest(obs_X, obs_y, feature_cols, comp_cols)
    visible = digest['visible']

    # ── beta from data: HV spread makes the two UCB terms commensurable ──
    if beta is None:
        beta = float(np.std(obs_y)) if len(obs_y) > 1 else 1.0

    N_total = max(int(sum(v['n'] for v in visible.values())), 1)

    # ── Score visible clusters by UCB1 ───────────────────────────────────
    scored = []
    for key, v in visible.items():
        bonus = beta * np.sqrt(np.log(max(N_total, 2)) / max(v['n'], 1))
        scored.append((key, v['best'] + bonus))

    # ── Elements never seen form an optimistic "unseen" option ───────────
    unseen = [e for e in available_elements
              if e not in digest['seen_elements']]

    comps = []
    for _ in range(n_target * oversample):
        # UCB1 optimism: untried options first, else the top-scoring cluster
        if unseen and rng.random() < 0.5:
            anchor = list(rng.choice(unseen, size=min(2, len(unseen)),
                                     replace=False))
        elif scored:
            best_key = max(scored, key=lambda s: s[1])[0]
            anchor   = best_key.split('-')
        else:
            anchor = list(rng.choice(available_elements, size=2, replace=False))

        # Sample a composition anchored on the selected cluster
        k      = int(rng.integers(n_elem_range[0], n_elem_range[1] + 1))
        anchor = [e for e in anchor if e in available_elements]
        others = [e for e in available_elements if e not in anchor]
        n_more = max(0, k - len(anchor))
        extra  = list(rng.choice(others, size=min(n_more, len(others)),
                                 replace=False)) if n_more else []
        elems  = anchor + extra
        if len(elems) < n_elem_range[0]:
            continue
        fracs = rng.dirichlet(np.ones(len(elems)))
        comps.append({e: float(f) for e, f in zip(elems, fracs)})
    return comps


def generate_mutation(n_target, obs_X, obs_y, feature_cols, rng,
                      top_k=3, sigma=0.05, oversample=4):
    """Gaussian perturbations of current top-k (raw dicts)."""
    comp_cols = get_composition_cols(feature_cols)
    comp_idx  = [feature_cols.index(c) for c in comp_cols]
    top_idx   = np.argsort(obs_y)[::-1][:top_k]

    comps = []
    for _ in range(n_target * oversample):
        parent = obs_X[rng.choice(top_idx)][comp_idx].copy()
        child  = np.clip(parent + rng.normal(0, sigma, size=len(parent)), 0, None)
        if child.sum() == 0:
            continue
        child /= child.sum()
        comp   = {comp_cols[i]: float(child[i])
                  for i in range(len(comp_cols)) if child[i] > 0.01}
        if len(comp) < 4:
            continue
        tot  = sum(comp.values())
        comps.append({k: v / tot for k, v in comp.items()})
    return comps


def generate_digest_guided(n_target, obs_X, obs_y, feature_cols,
                           available_elements, rng, n_elem_range=(4, 6),
                           oversample=4):
    """
    PRIMARY causal control ('digest' arm).

    INFORMATION: identical to the LLM arm's. Both consume the digest built
    by exploration_digest.build_digest() — the SAME object, same clusters,
    same MAX_CLUSTERS truncation, containing per-cluster query counts AND
    best-HV. The LLM receives it rendered as text; this arm reads its fields.
    Identical information is guaranteed by construction (one shared builder),
    not by two implementations agreeing.

    RULE (fully specified, no thresholds to tune):
        Rank clusters by cumulative query count.
        Uniformly sample from the least-explored half of the ranked clusters.
        Sample a random composition anchored on the chosen cluster.
      Never-seen elements are included as an additional least-explored
      option, so the rule is not confined to already-visited clusters.

    The arm READS ONLY the query counts and IGNORES the best-HV values,
    and performs no optimization. This is deliberate and is the entire
    experimental contrast:

        "The information available to both arms was identical.
         The difference is purely what they did with it."

    Any optimization inside this rule (e.g. scoring clusters by best-HV
    with an exploration bonus) would change the question from
    "does information access explain the effect?" to
    "is the LLM better than another optimizer?" — that different question
    is asked separately by the SUPPLEMENTARY 'digest_ucb' arm.

    Note on the one choice made: "least-explored half" is a median split of
    the ranked clusters. It is not tuned and was fixed before any results
    existed; median is the neutral split (it privileges no particular
    degree of aggression). This is parameter-free, not choice-free.
    """
    from exploration_digest import build_digest

    comp_cols = get_composition_cols(feature_cols)
    digest    = build_digest(obs_X, obs_y, feature_cols, comp_cols)
    visible   = digest['visible']

    # ── Rank by query count; keep the least-explored HALF ────────────────
    keys = sorted(visible.keys(), key=lambda k: visible[k]['n'])
    if len(keys) >= 2:
        half      = max(1, len(keys) // 2)
        remaining = keys[:half]          # least-explored half of the ranking
    else:
        remaining = keys

    # Never-seen elements = the least-explored region available at all
    unseen = [e for e in available_elements
              if e not in digest['seen_elements']]

    comps = []
    for _ in range(n_target * oversample):
        options = list(remaining) + (['__unseen__'] if unseen else [])
        if not options:
            anchor = list(rng.choice(available_elements, size=2, replace=False))
        else:
            pick = options[int(rng.integers(0, len(options)))]   # uniform
            if pick == '__unseen__':
                anchor = list(rng.choice(unseen, size=min(2, len(unseen)),
                                         replace=False))
            else:
                anchor = pick.split('-')

        anchor = [e for e in anchor if e in available_elements]
        k      = int(rng.integers(n_elem_range[0], n_elem_range[1] + 1))
        others = [e for e in available_elements if e not in anchor]
        n_more = max(0, k - len(anchor))
        extra  = list(rng.choice(others, size=min(n_more, len(others)),
                                 replace=False)) if n_more else []
        elems  = anchor + extra
        if len(elems) < n_elem_range[0]:
            continue
        fracs = rng.dirichlet(np.ones(len(elems)))
        comps.append({e: float(f) for e, f in zip(elems, fracs)})
    return comps


def generate_llm(obs_X, obs_y, feature_cols, available_elements,
                 stagnation_count, iteration, budget, intervention_log,
                 n_request=12):
    """
    LLM-reasoned compositions (raw dicts). The hypothesis arm.
    Returns raw composition dicts; admission is applied by the harness
    exactly as for every other arm (A1 fix — no separate LLM gate).

    A2-new: n_request lets the harness ask for more candidates than
    inject_n so the LLM arm has the same admission headroom (oversample)
    as random/mutation, equalizing realized injection counts.
    """
    try:
        from llm_proposal import llm_propose_compositions
        return llm_propose_compositions(
            observed_X         = obs_X,
            observed_y         = obs_y,
            feature_cols       = feature_cols,
            available_elements = available_elements,
            stagnation_count   = stagnation_count,
            iteration          = iteration,
            budget             = budget,
            intervention_log   = intervention_log,
            n_request          = n_request,
        )
    except Exception as e:
        print(f"    [llm] proposal failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# STAGNATION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def is_stagnating(best_history, window=5, threshold=0.02):
    if len(best_history) < window + 1:
        return False
    recent, previous = best_history[-1], best_history[-(window + 1)]
    return (recent - previous) / (abs(previous) + 1e-12) < threshold


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — one arm, one seed
# ═══════════════════════════════════════════════════════════════════════════════

def run_arm(
    arm:                str,
    oracle:             CanonicalOracle,
    candidate_pool:     np.ndarray,
    feature_cols:       list,
    available_elements: list,
    cov_inv:            np.ndarray,
    min_novelty:        float,
    dataset_X:          np.ndarray = None,
    n_initial:          int   = 12,
    n_iterations:       int   = 50,
    inject_n:           int   = 3,
    inject_cooldown:    int   = 3,
    stagnation_window:  int   = 5,
    stagnation_thresh:  float = 0.02,
    random_seed:        int   = 42,
) -> dict:
    """
    Run one experimental arm for one seed. arm ∈ {none,random,mutation,llm}.

    dataset_X : full working-dataset feature matrix. Injected candidates
        must be novel with respect to EVERY measured alloy in it — not
        just observed ∪ pool. This closes a leakage/memorization pathway
        (fix #2): an LLM could otherwise propose a training-set alloy
        near-verbatim, which admission would accept and the RF (trained
        on that exact alloy) would score with inflated accuracy, biasing
        the hypothesis arm. Deduping against the whole dataset forces
        genuinely novel rescue candidates for all arms identically.

    Returns full per-candidate instrumentation for mechanism analysis.
    """
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)

    surrogate = GPSurrogate()
    pool      = candidate_pool.copy()
    queried   = set()

    # ── Initial random sampling ───────────────────────────────────────────
    init_idx = rng.choice(len(pool), size=min(n_initial, len(pool)),
                          replace=False)
    obs_X, obs_y = [], []
    for idx in init_idx:
        obs_X.append(pool[idx])
        obs_y.append(oracle.query(pool[idx]))
        queried.add(idx)
    obs_X = np.array(obs_X)
    obs_y = np.array(obs_y)

    best      = float(obs_y.max())
    best_hist = [best]

    # ── Instrumentation state ─────────────────────────────────────────────
    inject_events    = []     # one dict per injection event
    candidate_tags   = {}     # pool_index -> {'event': eid, ...}
    intervention_log = []     # raw LLM traces (llm arm only)
    calibration_log  = []     # (mu, sigma, y) at every pick — for C2
    trajectory_log   = []     # Finding 1: full queried path (x, formula, hv)
    stagnation_trace = []     # Finding 4: detector-fired / cooldown-blocked
    stag_counter     = 0      # A6: TRUE consecutive stagnation duration
    last_inject      = -inject_cooldown
    terminated_early = False

    comp_cols = get_composition_cols(feature_cols)   # for formula strings

    for it in range(1, n_iterations + 1):
        surrogate.fit(obs_X, obs_y)

        stagnating  = is_stagnating(best_hist, stagnation_window,
                                    stagnation_thresh)
        cooldown_ok = (it - last_inject) >= inject_cooldown

        # Finding 4: log the detector state every iteration — including
        # iterations where it fired but cooldown blocked injection. This is
        # the raw material for any trigger-sensitivity analysis (attack #5).
        stagnation_trace.append({
            'iter'           : it,
            'stagnating'     : bool(stagnating),
            'cooldown_ok'    : bool(cooldown_ok),
            'stag_counter'   : stag_counter,
            'injection_fired': bool(arm != 'none' and stagnating and cooldown_ok),
        })

        # ── Injection (arms differ ONLY in generation; admission shared) ──
        # C2-new: an event is logged whenever injection is TRIGGERED,
        #         regardless of how many candidates survive admission —
        #         so rescue-success denominators count attempts, not just
        #         successful injections, for every arm identically.
        # A2-new: all arms oversample and the harness caps admissions at
        #         inject_n, equalizing realized injection counts in
        #         expectation (the LLM is asked for extra candidates).
        if arm != 'none' and stagnating and cooldown_ok:

            if arm == 'random':
                raw = generate_random(inject_n, available_elements, rng)
            elif arm == 'mutation':
                raw = generate_mutation(inject_n, obs_X, obs_y,
                                        feature_cols, rng)
            elif arm == 'digest':
                raw = generate_digest_guided(inject_n, obs_X, obs_y,
                                             feature_cols, available_elements,
                                             rng)
            elif arm == 'digest_ucb':
                raw = generate_digest_ucb(inject_n, obs_X, obs_y,
                                          feature_cols, available_elements, rng,
                                          stagnation_count=stag_counter,
                                          iteration=it, budget=n_iterations)
            elif arm == 'llm':
                # A2-new: request oversample×inject_n candidates so the LLM
                # arm has the same admission headroom as the other arms.
                raw = generate_llm(obs_X, obs_y, feature_cols,
                                   available_elements, stag_counter,
                                   it, n_iterations, intervention_log,
                                   n_request=inject_n * LLM_OVERSAMPLE)
            else:
                raw = []

            # Shared admission (A1/A5/D2): reference = observed ∪ pool ∪
            # full working dataset (fix #2 — leakage-safe: injected
            # candidates must be novel vs every measured alloy, including
            # the oracle's training set).
            parts = [obs_X]
            if len(pool):
                parts.append(pool)
            if dataset_X is not None and len(dataset_X):
                parts.append(dataset_X)
            reference = np.vstack(parts)
            admitted, rejects = [], {'nan_descriptor': 0, 'simplex_range': 0,
                                     'simplex_sum': 0, 'near_duplicate': 0}
            for comp in raw:
                res = admit_candidate(comp, feature_cols, reference,
                                      cov_inv, min_novelty)
                if res.admitted:
                    admitted.append(res)
                    reference = np.vstack([reference, res.vec])
                    if len(admitted) >= inject_n:
                        break
                else:
                    rejects[res.reason] = rejects.get(res.reason, 0) + 1

            # C2-new: log the event UNCONDITIONALLY (attempt-based).
            last_inject = it
            eid   = len(inject_events)
            start = len(pool)

            if admitted:
                vecs = np.array([r.vec for r in admitted])
                pool = np.vstack([pool, vecs])
                for j, r in enumerate(admitted):
                    pidx = start + j
                    candidate_tags[pidx] = {
                        'event'        : eid,
                        'novelty_maha' : r.novelty,
                        # A5-new: both novelty metrics against the SAME
                        # reference (observed ∪ pool ∪ batch) for a fair
                        # legacy comparison.
                        'novelty_eucl' : composition_euclidean(
                                             r.vec, reference, feature_cols),
                        'stability'    : r.stability,
                        'queried'      : False,
                        'queried_at'   : None,
                        'improved'     : False,
                        'hv'           : None,
                    }

            inject_events.append({
                'event'            : eid,
                'iteration'        : it,
                'arm'              : arm,
                'stag_at_trigger'  : stag_counter,
                'n_proposed'       : len(raw),
                'n_admitted'       : len(admitted),   # A2-new covariate
                'rejects'          : rejects,
                'incumbent_before' : best,
                'pool_indices'     : list(range(start, len(pool))),
                'vectors'          : [r.vec.tolist() for r in admitted],
                'escaped'          : False,
                'escape_latency'   : None,
            })

        # ── Acquisition over unqueried pool ──────────────────────────────
        unq = np.array([i for i in range(len(pool)) if i not in queried])
        if len(unq) == 0:
            terminated_early = True
            break
        X_cand = pool[unq]
        ei     = surrogate.ei_scores(X_cand, best)
        pick   = int(unq[int(np.argmax(ei))])

        # ── Finding 3: EI diagnostics for injected-but-unqueried candidates ─
        # Answers "why weren't the LLM's candidates picked?" — were they
        # low-EI, or near-misses? Record the best injected candidate still
        # in play this step, its EI rank, and the GP's view of it.
        inj_diag = None
        injected_unq_mask = np.array([idx in candidate_tags for idx in unq])
        if injected_unq_mask.any():
            inj_positions = np.where(injected_unq_mask)[0]
            best_inj_pos  = inj_positions[int(np.argmax(ei[inj_positions]))]
            best_inj_idx  = int(unq[best_inj_pos])
            # EI rank among all candidates (1 = would be picked next)
            ei_rank       = int(np.sum(ei > ei[best_inj_pos]) + 1)
            mu_i, sd_i    = surrogate.predict_point(pool[best_inj_idx])
            inj_diag = {
                'best_injected_pool_idx' : best_inj_idx,
                'event'                  : candidate_tags[best_inj_idx]['event'],
                'ei'                     : float(ei[best_inj_pos]),
                'ei_of_pick'             : float(ei[int(np.argmax(ei))]),
                'ei_rank'                : ei_rank,
                'n_candidates'           : int(len(unq)),
                'mu'                     : mu_i,
                'sigma'                  : sd_i,
                'picked_an_injected'     : bool(pick in candidate_tags),
            }

        # ── Record GP prediction at pick (C2 calibration data) ───────────
        mu, sd = surrogate.predict_point(pool[pick])

        # ── Query oracle ──────────────────────────────────────────────────
        x_new = pool[pick]
        hv    = oracle.query(x_new)
        queried.add(pick)
        calibration_log.append({'iter': it, 'mu': mu, 'sigma': sd, 'y': hv})

        # ── Finding 1: full queried trajectory (what, not just how good) ──
        trajectory_log.append({
            'iter'        : it,
            'pool_idx'    : pick,
            'is_injected' : bool(pick in candidate_tags),
            'formula'     : _formula_string(x_new, feature_cols, comp_cols),
            'hv'          : float(hv),
            'gp_mu'       : mu,
            'gp_sigma'    : sd,
            'inj_diag'    : inj_diag,   # Finding 3, attached per-step
        })

        prev_best = best
        obs_X = np.vstack([obs_X, x_new])
        obs_y = np.append(obs_y, hv)
        best  = float(obs_y.max())
        best_hist.append(best)

        # ── A6: true consecutive stagnation counter ───────────────────────
        if best > prev_best + 1e-9:
            stag_counter = 0
        else:
            stag_counter += 1

        # ── A3/A4: per-candidate hit + attributed escape bookkeeping ─────
        if pick in candidate_tags:
            tag = candidate_tags[pick]
            tag['queried']    = True
            tag['queried_at'] = it
            tag['hv']         = hv
            improved          = hv > prev_best + 1e-9
            tag['improved']   = improved

            ev = inject_events[tag['event']]
            # attributed escape: this event's OWN candidate beat the
            # incumbent that was in place when the event fired
            if improved and hv > ev['incumbent_before'] + 1e-9 \
               and not ev['escaped']:
                ev['escaped']        = True
                ev['escape_latency'] = it - ev['iteration']

    # ── Aggregate mechanism metrics (A3/A4 definitions) ──────────────────
    tags = list(candidate_tags.values())
    n_candidates = len(tags)
    n_queried    = sum(1 for t in tags if t['queried'])
    n_hits       = sum(1 for t in tags if t['improved'])

    per_candidate_hit_rate   = (n_hits / n_candidates) if n_candidates else np.nan
    pickup_rate              = (n_queried / n_candidates) if n_candidates else np.nan
    n_events                 = len(inject_events)
    n_escaped                = sum(1 for e in inject_events if e['escaped'])
    per_event_rescue_success = (n_escaped / n_events) if n_events else np.nan

    latencies      = [e['escape_latency'] for e in inject_events if e['escaped']]
    median_latency = float(np.median(latencies)) if latencies else np.nan
    censored_frac  = (1 - n_escaped / n_events) if n_events else np.nan

    return {
        'arm'                      : arm,
        'seed'                     : random_seed,
        'best_history'             : best_hist,
        'final_best'               : best,
        'terminated_early'         : terminated_early,
        # events + per-candidate detail (for analysis layer / traces)
        'inject_events'            : inject_events,
        'candidate_tags'           : tags,
        'intervention_log'         : intervention_log,
        'calibration_log'          : calibration_log,
        'trajectory_log'           : trajectory_log,      # Finding 1
        'stagnation_trace'         : stagnation_trace,    # Finding 4
        # headline mechanism metrics
        'n_injection_events'       : n_events,
        'n_injected_candidates'    : n_candidates,
        'pickup_rate'              : pickup_rate,
        'per_candidate_hit_rate'   : per_candidate_hit_rate,
        'per_event_rescue_success' : per_event_rescue_success,
        'median_escape_latency'    : median_latency,
        'escape_censored_frac'     : censored_frac,
        'mean_novelty_maha'        : float(np.mean([t['novelty_maha'] for t in tags])) if tags else np.nan,
        'mean_stability_injected'  : float(np.mean([t['stability'] for t in tags])) if tags else np.nan,
        'observed_y'               : obs_y,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST — non-LLM arms, working dataset (D1 fixed)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df = load_working_dataset(WORKING_DATA)

    feature_cols       = get_feature_cols(df)
    available_elements = get_composition_cols(feature_cols)
    dataset_X          = df[feature_cols].values

    cov_inv     = descriptor_covariance(dataset_X, feature_cols)
    min_novelty = default_min_novelty(dataset_X, feature_cols, cov_inv)
    print(f"Dedup threshold (Mahalanobis, 5th pct distinct-NN): {min_novelty:.4f}")

    splits = make_splits(df, random_seed=42)
    oracle = CanonicalOracle(splits['train'], feature_cols)
    base_pool = np.vstack([splits['val'][feature_cols].values,
                           splits['hidden'][feature_cols].values])

    # B3 guard: 'none' arm worst case must fit in the pool
    assert 12 + 40 <= len(base_pool), "budget exceeds pool for 'none' arm"

    print(f"Base pool: {len(base_pool)} (val+hidden)\n")
    for arm in ['none', 'random', 'mutation']:
        res = run_arm(
            arm=arm, oracle=oracle, candidate_pool=base_pool,
            feature_cols=feature_cols, available_elements=available_elements,
            cov_inv=cov_inv, min_novelty=min_novelty, dataset_X=dataset_X,
            n_initial=12, n_iterations=40, inject_n=3, random_seed=42,
        )
        print(f"  {arm:>9}  final={res['final_best']:.1f}  "
              f"events={res['n_injection_events']}  "
              f"cands={res['n_injected_candidates']}  "
              f"pickup={res['pickup_rate'] if not np.isnan(res['pickup_rate']) else 0:.2f}  "
              f"cand_hit={res['per_candidate_hit_rate'] if not np.isnan(res['per_candidate_hit_rate']) else 0:.2f}  "
              f"rescue={res['per_event_rescue_success'] if not np.isnan(res['per_event_rescue_success']) else 0:.2f}  "
              f"lat={res['median_escape_latency'] if not np.isnan(res['median_escape_latency']) else float('nan')}  "
              f"cens={res['escape_censored_frac'] if not np.isnan(res['escape_censored_frac']) else 0:.2f}")
