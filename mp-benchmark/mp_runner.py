"""
MP Shear-Modulus Rescue — Multi-Split Runner
============================================
Port of four_arm_runner_v2.py to the MP benchmark. Produces output in the
SAME shape the existing analysis layer consumes, so statistics/calibration/
dual-channel/rescue-frequency analyses attach with only "HV"->"G" changes.

KEY MP-SPECIFIC BEHAVIOUR
  - FEATURIZE ONCE, CACHE. Magpie featurization of 1,827 compositions takes
    ~7 min. It is done a single time and the feature matrix is cached to
    disk; every split/seed reuses it. (Borg had cheap descriptors and did
    not need this.)
  - COMPOSITION THREADING. Each pool row carries both its Magpie vector and
    its composition dict (recovered from the formula), so the digest and
    mutation arms operate on real elements (Option 1).
  - PER-SPLIT ADMISSION THRESHOLD. The distance cutoff is the 90th
    percentile of that split's pool-to-train distances, recomputed inside
    MPOracle for each split (NOT a global constant) — the pre-registered
    rule, robust to split-dependent geometry.
  - PER-SPLIT DEDUP TOLERANCE from the split's own near-neighbour scale.

Everything else — the five arms, the fail-fast LLM guard, the config
snapshot, per-run detail persistence, LLM-trace persistence — mirrors the
Borg runner.
"""

import os
import json
import datetime
import inspect
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import mp_oracle as MO
import mp_harness as H
import mp_llm_proposal as LP
from mp_oracle import MPOracle, FEATURE_COLS
from pymatgen.core import Composition
from sklearn.metrics import pairwise_distances

FEATURE_CACHE = 'mp_features_cache.parquet'


# ══════════════════════════════════════════════════════════════════════════
# data prep — featurize once, cache, recover compositions
# ══════════════════════════════════════════════════════════════════════════

def prepare_dataset():
    """Load, featurize (cached), and attach composition dicts."""
    if os.path.exists(FEATURE_CACHE):
        print(f"  loading cached features from {FEATURE_CACHE}")
        df = pd.read_parquet(FEATURE_CACHE)
    else:
        print("  featurizing (one-time, ~7 min)...")
        df = MO.featurize(MO.load_mp_dataset())
        df.drop(columns=['comp_obj'], errors='ignore').to_parquet(FEATURE_CACHE)
        print(f"  cached features to {FEATURE_CACHE}")

    # composition dicts, recovered once from the formula column
    comps = [{str(e): float(a)
              for e, a in Composition(f).fractional_composition.items()}
             for f in df['formula']]
    return df, comps


def make_splits(n_rows, seed):
    """60/40 train/pool split (pool = everything not used to train the
    oracle). Matches the screening/threshold analysis split style."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    cut = int(0.6 * n_rows)
    return idx[:cut], idx[cut:]        # train_idx, pool_idx


def dedup_tolerance(oracle, pool_X):
    """Per-split near-duplicate tolerance: a small fraction of the median
    nearest-neighbour distance among pool points in scaled space."""
    Xs = oracle.scaler.transform(pool_X[:min(len(pool_X), 400)])
    D = pairwise_distances(Xs)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return float(np.median(nn) * 0.25)


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════

def run_experiment(
    arms         = ('none', 'random', 'mutation', 'digest', 'llm'),
    n_splits     = 5,          # raised from 3 (reviewer pt 1): more
                               # independent oracles => the conclusions are
                               # sampled over train/test partitions, not just
                               # BO randomness. 5 x 7 = 35 paired runs/arm.
    n_bo_seeds   = 7,
    n_initial    = 8,
    n_iterations = 20,
    inject_n     = 3,
    save_dir     = 'results/mp_shear_v1',
    force_mock_llm = False,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/llm_traces', exist_ok=True)
    os.makedirs(f'{save_dir}/run_detail', exist_ok=True)

    df, comps = prepare_dataset()
    X_all = df[FEATURE_COLS].values
    y_all = df['G'].values
    elements = MO.dataset_elements(df)
    print(f"  dataset: {len(df)} compositions, {len(elements)} elements, "
          f"{len(FEATURE_COLS)} features")

    # ── config snapshot (pre-registration) ───────────────────────────────
    _sig = inspect.signature(H.run_arm).parameters
    config = {
        'timestamp'        : datetime.datetime.now().isoformat(),
        'benchmark'        : 'Materials Project shear modulus (metallic, >=3 elem)',
        'objective'        : 'maximize shear modulus G (GPa)',
        'arms'             : list(arms),
        'n_splits'         : n_splits, 'n_bo_seeds': n_bo_seeds,
        'n_initial'        : n_initial, 'n_iterations': n_iterations,
        'inject_n'         : inject_n, 'llm_oversample': H.LLM_OVERSAMPLE,
        'stagnation_window': _sig['stagnation_window'].default,
        'stagnation_thresh': _sig['stagnation_thresh'].default,
        'inject_cooldown'  : _sig['inject_cooldown'].default,
        'llm_model'        : LP.LLM_MODEL, 'llm_temperature': LP.LLM_TEMPERATURE,
        'llm_max_tokens'   : LP.LLM_MAX_TOKENS,
        'n_compositions'   : len(df), 'n_elements': len(elements),
        'gp_fit'           : 'uncapped (fit_gpytorch_mll to convergence)',
        'admission_rule'   : ('distance-bounded: Euclidean-to-nearest-train in '
                              'scaled Magpie space <= 90th percentile of pool '
                              'distances (recomputed PER SPLIT); then near-'
                              'duplicate dedup. Pre-registered from error-vs-'
                              'distance analysis (error flat within this bound).'),
        'physics_channel'  : 'Voigt-Reuss-Hill G from elemental moduli (Channel B)',
        'digest_rule'      : ('PRIMARY control: shared 8-cluster digest (counts + '
                              'best-G); rank by count; sample least-explored half; '
                              'reads counts only, no optimization.'),
        'representation'   : ('two parallel: Magpie vector (GP/oracle/distance) + '
                              'composition dict (digest clustering, mutation).'),
    }
    with open(f'{save_dir}/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── LLM mode ──────────────────────────────────────────────────────────
    llm_mode = 'real'
    if 'llm' in arms:
        if force_mock_llm or not os.environ.get('ANTHROPIC_API_KEY', '').startswith('sk-ant-'):
            llm_mode = 'MOCK'
            _install_mock_llm()
            print("  ⚠ LLM arm in MOCK mode — placeholder numbers.\n")
        else:
            print("  ✓ LLM arm using REAL API key.\n")

    results = {arm: [] for arm in arms}
    oracle_ranks = []        # per-split Spearman(RF, true G) on the pool

    for sp in range(n_splits):
        tr_idx, pool_idx = make_splits(len(df), seed=sp)
        oracle = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=sp)
        pool_X = X_all[pool_idx]
        pool_comps = [comps[i] for i in pool_idx]
        dtol = dedup_tolerance(oracle, pool_X)

        # per-split oracle ranking quality (reviewer pt 3): Spearman of the
        # RF prediction vs true G on this split's POOL (held-out from train).
        # Collected across splits -> mean and CI, so the 0.85 headline is
        # shown to be typical, not a single lucky number.
        from scipy.stats import spearmanr
        pool_pred = oracle.query_batch(pool_X)
        pool_true = y_all[pool_idx]
        split_rank_rho = float(spearmanr(pool_true, pool_pred).correlation)
        oracle_ranks.append(split_rank_rho)

        # pool ceiling (for logging / sanity)
        ceil = float(oracle.query_batch(pool_X).max())
        print(f"  Split {sp}: pool={len(pool_X)}  ceiling(G)={ceil:.1f}  "
              f"admit_thresh={oracle.admission_threshold:.2f}  dedup={dtol:.3f}")

        # llm first on split 0 so the fail-fast guard trips early
        arm_order = (['llm'] + [a for a in arms if a != 'llm']
                     if (sp == 0 and 'llm' in arms) else list(arms))

        for arm in arm_order:
            for seed in range(n_bo_seeds):
                res = H.run_arm(
                    arm=arm, oracle=oracle, pool_vectors=pool_X,
                    pool_comps=pool_comps, elements=elements, dedup_tol=dtol,
                    n_initial=n_initial, n_iterations=n_iterations,
                    inject_n=inject_n, random_seed=seed)
                res['split_seed'] = sp
                if res['terminated_early']:
                    raise RuntimeError(f"pool exhausted: {arm} split{sp} seed{seed}")
                results[arm].append(res)

                # persist LLM traces
                if arm == 'llm' and res['intervention_log']:
                    with open(f'{save_dir}/llm_traces/split{sp}_seed{seed}.json',
                              'w', encoding='utf-8') as f:
                        json.dump(_clean(res['intervention_log']), f, indent=2,
                                  ensure_ascii=False)

                # fail-fast: real LLM must produce usable candidates on first run
                if (arm == 'llm' and llm_mode == 'real' and sp == 0 and seed == 0):
                    got = any(r.get('raw_candidates') for r in res['intervention_log'])
                    fired = len(res['intervention_log']) > 0
                    if fired and not got:
                        raise RuntimeError(
                            "REAL LLM produced zero usable candidates on the first "
                            "run — key invalid/expired/rate-limited or response format "
                            "changed. Aborting before a wasted full run.")

                # per-run detail for the analysis layer
                detail = {
                    'arm': arm, 'split_seed': sp, 'bo_seed': seed,
                    'final_best': res['final_best'],
                    'best_history': res['best_history'],
                    'calibration_log': res['calibration_log'],
                    'candidate_tags': res['candidate_tags'],
                    'inject_events': res['inject_events'],
                    'trajectory_log': res['trajectory_log'],
                    'stagnation_trace': res['stagnation_trace']}
                with open(f'{save_dir}/run_detail/{arm}_split{sp}_seed{seed}.json',
                          'w', encoding='utf-8') as f:
                    json.dump(_clean(detail), f, ensure_ascii=False)

        for arm in arms:
            finals = [r['final_best'] for r in results[arm] if r['split_seed'] == sp]
            print(f"    {arm:>10}: final {np.mean(finals):.1f} ± {np.std(finals):.1f}")

    summary = _summarize(results, arms, llm_mode)
    summary['oracle_ranking'] = _rank_ci(oracle_ranks)          # reviewer pt 3
    summary['admission_by_arm'] = _admission_summary(results)   # reviewer pt 2
    summary['cost_by_arm'] = _cost_summary(results)             # reviewer pt 10
    with open(f'{save_dir}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(_clean(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {save_dir}/  (llm_mode={llm_mode})")
    return {'results': results, 'summary': summary, 'llm_mode': llm_mode}


# ══════════════════════════════════════════════════════════════════════════
# summarize + stats (paired, aligned on (split,seed), Holm) — as Borg
# ══════════════════════════════════════════════════════════════════════════

def _paired(results, a, b):
    def keyed(arm):
        d = {}
        for r in results[arm]:
            d[(r['split_seed'], r['seed'])] = r['final_best']
        return d
    A, B = keyed(a), keyed(b)
    keys = sorted(set(A) & set(B))
    return np.array([A[k] for k in keys]), np.array([B[k] for k in keys]), keys


def _holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = max(min(1.0, (m - rank) * p), prev)
        out[name] = adj; prev = adj
    return out


def _rank_ci(rhos):
    """Mean and 95% CI of per-split oracle Spearman (reviewer pt 3)."""
    if not rhos:
        return {}
    a = np.array(rhos)
    return {'per_split': [float(r) for r in a], 'mean': float(a.mean()),
            'std': float(a.std()),
            'ci95_lo': float(a.mean() - 1.96*a.std()/np.sqrt(len(a))),
            'ci95_hi': float(a.mean() + 1.96*a.std()/np.sqrt(len(a)))}


def _admission_summary(results):
    """
    Per-arm admission outcome breakdown (reviewer pt 2 / the distance-gate
    bias check). For each arm: total proposed, total admitted, and rejects
    by reason (unfeaturizable / beyond_reliable_region / near_duplicate).
    If the LLM arm is rejected far more often than mutation — especially for
    'beyond_reliable_region' — the distance gate is asymmetrically clipping
    exploration, and any result is conditional on that. This makes the gate's
    effect a REPORTED quantity rather than an assumed-neutral one.
    """
    out = {}
    for arm, runs in results.items():
        if arm == 'none':
            continue
        proposed = admitted = 0
        reasons = {}
        for r in runs:
            for ev in r['inject_events']:
                proposed += ev.get('n_proposed', 0)
                admitted += ev.get('n_admitted', 0)
                for k, v in (ev.get('rejects') or {}).items():
                    reasons[k] = reasons.get(k, 0) + v
        out[arm] = {'proposed': proposed, 'admitted': admitted,
                    'admit_rate': (admitted/proposed) if proposed else None,
                    'rejects_by_reason': reasons}
    return out


def _cost_summary(results):
    """Per-arm generation time per intervention (reviewer pt 10)."""
    out = {}
    for arm, runs in results.items():
        if arm == 'none':
            continue
        times = [ev.get('gen_seconds') for r in runs
                 for ev in r['inject_events'] if ev.get('gen_seconds') is not None]
        out[arm] = {'n_events': len(times),
                    'mean_gen_seconds': float(np.mean(times)) if times else None,
                    'median_gen_seconds': float(np.median(times)) if times else None,
                    'total_gen_seconds': float(np.sum(times)) if times else None}
    return out


def _summarize(results, arms, llm_mode):
    from scipy.stats import wilcoxon
    summary = {'llm_mode': llm_mode,
               'claim_scope': 'rescue-on-benchmark; NOT a discovery claim',
               'objective': 'shear modulus G (GPa)', 'arms': {}, 'tests': {}}
    for arm in arms:
        finals = np.array([r['final_best'] for r in results[arm]])
        summary['arms'][arm] = {
            'n_runs': len(finals), 'final_mean': float(finals.mean()),
            'final_std': float(finals.std()),
            'final_ci95': float(1.96*finals.std()/np.sqrt(len(finals))),
            'pickup_rate': _nanmean([r['pickup_rate'] for r in results[arm]]),
            'rescue_success': _nanmean([r['per_event_rescue_success'] for r in results[arm]]),
            'finals': finals.tolist()}
    if 'llm' in arms:
        raw = {}
        for ctrl in arms:
            if ctrl == 'llm':
                continue
            lf, cf, _ = _paired(results, 'llm', ctrl)
            diffs = lf - cf
            if np.allclose(diffs, 0):
                p = 1.0
            else:
                try: _, p = wilcoxon(lf, cf)
                except ValueError: p = 1.0
            summary['tests'][f'llm_vs_{ctrl}'] = {
                'median_diff': float(np.median(diffs)), 'wilcoxon_p': float(p),
                'llm_wins': int(np.sum(diffs > 0)), 'ties': int(np.sum(diffs == 0)),
                'ctrl_wins': int(np.sum(diffs < 0))}
            raw[f'llm_vs_{ctrl}'] = p if not np.isnan(p) else 1.0
        for k, ap in _holm(raw).items():
            summary['tests'][k]['holm_p'] = ap
    return summary


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float('nan')


def _clean(o):
    if isinstance(o, dict):  return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [_clean(v) for v in o]
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    return o


def _install_mock_llm():
    """Labelled mock for pipeline testing without an API key."""
    def mock(obs_comps, obs_y, available_elements, stagnation_count,
             iteration, budget, intervention_log, n_request=3):
        rng = np.random.default_rng(iteration + int(obs_y.sum()) % 1000)
        out = []
        for _ in range(n_request):
            k = rng.integers(3, 6)
            els = list(rng.choice(available_elements, size=k, replace=False))
            fr = rng.dirichlet(np.ones(k))
            out.append({e: float(f) for e, f in zip(els, fr)})
        intervention_log.append({'iteration': iteration,
                                 'stagnation_count': stagnation_count,
                                 'reasoning': 'MOCK', 'raw_candidates': out})
        return out
    LP.llm_propose_compositions = mock


if __name__ == '__main__':
    out = run_experiment(force_mock_llm=False)
    s = out['summary']
    print("\n" + "="*60)
    print(f"  SUMMARY  [LLM={out['llm_mode']}]  objective: shear modulus G")
    print("="*60)
    for arm in s['arms']:
        a = s['arms'][arm]
        print(f"  {arm:<11} G={a['final_mean']:.1f}±{a['final_ci95']:.1f}")
    if s['tests']:
        print("\n  Paired tests (llm vs controls), Holm-corrected:")
        for k, t in s['tests'].items():
            print(f"    {k:<16} Δ={t['median_diff']:+.1f}  "
                  f"p={t['wilcoxon_p']:.4f} holm={t.get('holm_p',float('nan')):.4f}  "
                  f"(W{t['llm_wins']}/T{t['ties']}/L{t['ctrl_wins']})")
