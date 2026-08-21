"""
Four-Arm Experiment Runner — v2 (multi-split)
=============================================
Wave-1 rewrite. Changes from v1:

  B1 — Replication over DATA SPLITS, not just BO seeds. Previously all
       seeds shared one train/val/hidden split and one oracle, so the
       statistics supported "LLM wins on THIS split" not "in general".
       Now: N_SPLITS split seeds × N_BO_SEEDS BO seeds. One oracle per
       split; arms within a split share it (pairing preserved). A result
       is credible only if it holds across splits.

  B2 — Multiple-comparison correction + effect sizes. Paired Wilcoxon
       (llm vs each control) within each split, Holm-Bonferroni across
       the three comparisons, rank-biserial effect size, and a bootstrap
       95% CI on the median paired difference.

  B3 — Pool-exhaustion guard asserted before running; terminated_early
       from any run aborts silent padding.

  D1 — canonical dataset path imported from the harness (single source).

Mechanism metrics (per-candidate, from harness v2) are aggregated per
arm across all splits×seeds. LLM traces are persisted per run for the
Wave-2 qualitative analysis.

The LLM arm auto-detects a real API key; without one it runs in a clearly
labelled MOCK mode whose numbers are placeholders, not results.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from canonical_oracle import make_splits, CanonicalOracle, get_feature_cols, get_composition_cols
from novelty_metric   import descriptor_covariance, default_min_novelty
import experiment_harness_v2 as H

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK LLM (labelled; used only when no real API key)
# ═══════════════════════════════════════════════════════════════════════════════

def install_mock_llm():
    import llm_proposal
    def mock_propose(observed_X, observed_y, feature_cols, available_elements,
                     stagnation_count, iteration, budget, intervention_log,
                     n_request=3):
        rng  = np.random.default_rng(iteration + int(observed_y.sum()) % 1000)
        base = [
            {'Mo':0.2,'Nb':0.2,'Ta':0.2,'Ti':0.2,'V':0.2},
            {'Al':0.2,'Cr':0.2,'Fe':0.2,'Mo':0.2,'Ti':0.2},
            {'Co':0.2,'Cr':0.2,'Fe':0.2,'Ni':0.2,'Nb':0.2},
            {'Cr':0.2,'Fe':0.2,'Nb':0.2,'Ti':0.2,'V':0.2},
            {'Al':0.2,'Co':0.2,'Cr':0.2,'Ni':0.2,'Ti':0.2},
        ]
        out = []
        for i in range(n_request):
            b = base[i % len(base)]
            j = {k: max(0.01, v + rng.normal(0, 0.03)) for k, v in b.items()}
            t = sum(j.values()); out.append({k: v/t for k, v in j.items()})
        intervention_log.append({'iteration': iteration,
                                 'stagnation_count': stagnation_count,
                                 'reasoning': 'MOCK placeholder',
                                 'raw_candidates': out})
        return out
    llm_proposal.llm_propose_compositions = mock_propose


def real_llm_available():
    return os.environ.get('ANTHROPIC_API_KEY', '').startswith('sk-ant-')


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS (B2)
# ═══════════════════════════════════════════════════════════════════════════════

def rank_biserial(diffs):
    """Effect size for paired samples: (wins - losses) / n_nonzero."""
    nz = diffs[np.abs(diffs) > 1e-12]
    if len(nz) == 0:
        return 0.0
    return float((np.sum(nz > 0) - np.sum(nz < 0)) / len(nz))


def bootstrap_median_ci(diffs, n_boot=5000, seed=0):
    rng  = np.random.default_rng(seed)
    meds = [np.median(rng.choice(diffs, size=len(diffs), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def holm_bonferroni(pvals: dict) -> dict:
    """Return Holm-adjusted p-values for a dict {name: p}."""
    items  = sorted(pvals.items(), key=lambda kv: kv[1])
    m      = len(items)
    out    = {}
    prev   = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev)      # enforce monotonicity
        out[name] = adj
        prev = adj
    return out


def paired_tests(llm_finals, ctrl_finals):
    diffs = llm_finals - ctrl_finals
    if np.allclose(diffs, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(llm_finals, ctrl_finals)
        except ValueError:
            p = np.nan
    lo, hi = bootstrap_median_ci(diffs)
    return {
        'median_diff'   : float(np.median(diffs)),
        'median_ci95'   : [lo, hi],
        'wilcoxon_p'    : float(p),
        'rank_biserial' : rank_biserial(diffs),
        'llm_wins'      : int(np.sum(diffs > 0)),
        'ties'          : int(np.sum(np.abs(diffs) <= 1e-12)),
        'ctrl_wins'     : int(np.sum(diffs < 0)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    arms         = ('none', 'random', 'mutation', 'digest', 'llm'),
    n_splits     = 3,
    n_bo_seeds   = 7,
    n_initial    = 12,
    n_iterations = 50,
    inject_n     = 3,
    save_dir     = '../results/four_arm_v2',
    force_mock_llm = False,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/figures', exist_ok=True)
    os.makedirs(f'{save_dir}/llm_traces', exist_ok=True)

    # Finding 2: snapshot the full run configuration so methods-writing and
    # reproducibility never require code archaeology. Written before the run.
    import datetime, llm_proposal, inspect
    # Pull stagnation/cooldown defaults from run_arm's actual signature so
    # the config can never silently drift from what the harness uses.
    _sig = inspect.signature(H.run_arm).parameters
    config = {
        'timestamp'         : datetime.datetime.now().isoformat(),
        'arms'              : list(arms),
        'n_splits'          : n_splits,
        'n_bo_seeds'        : n_bo_seeds,
        'n_initial'         : n_initial,
        'n_iterations'      : n_iterations,
        'inject_n'          : inject_n,
        'llm_oversample'    : H.LLM_OVERSAMPLE,
        'stagnation_window' : _sig['stagnation_window'].default,
        'stagnation_thresh' : _sig['stagnation_thresh'].default,
        'inject_cooldown'   : _sig['inject_cooldown'].default,
        'llm_model'         : llm_proposal.LLM_MODEL,
        'llm_temperature'   : llm_proposal.LLM_TEMPERATURE,
        'llm_max_tokens'    : llm_proposal.LLM_MAX_TOKENS,
        'working_data'      : H.WORKING_DATA,
        'gp_fit'            : 'uncapped (fit_gpytorch_mll to convergence)',
        'dedup_metric'      : 'mahalanobis-descriptor; ref = obs ∪ pool ∪ dataset',
        'digest_rule'       : ('PRIMARY causal control, pre-specified, ZERO hyperparameters: '
                               'read the same 8-cluster digest the LLM sees (counts + best-HV); '
                               'rank clusters by query count; discard clusters at/above the '
                               'MEDIAN count (parameter-free split); choose uniformly among the '
                               'remainder (plus never-seen elements); sample random composition '
                               'anchored there. Performs NO optimization — isolates information '
                               'access from reasoning.'),
        'digest_ucb_note'   : ('SUPPLEMENTARY ablation, not in the main run. Asks a DIFFERENT '
                               'question: can the LLM beat a classical heuristic built from the '
                               'same information? Enable explicitly by adding digest_ucb to arms.'),
        'digest_ucb_rule'   : ('UCB1 over '
                               'dominant-element clusters; score(c)=best_HV(c)+beta*'
                               'sqrt(ln(N)/n_c); beta=std(observed HV) (data-set, not '
                               'hand-tuned); unseen elements get optimistic priority; '
                               'sees the SAME 8-cluster-truncated digest as the LLM arm.'),
        'arm_information'   : {
            'none'      : 'n/a',
            'random'    : 'none',
            'digest'    : 'same as llm: counts + best-HV (8-truncated); rule uses counts only',
            'digest_ucb': 'same as llm: per-cluster best-HV + counts (8-truncated), '
                          'top-5, best-so-far, stagnation depth, budget',
            'mutation'  : 'top-k incumbents (local)',
            'llm'       : 'per-cluster best-HV + counts (8-truncated), top-5, '
                          '15 recent, best-so-far, stagnation depth, budget',
        },
    }
    with open(f'{save_dir}/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    from canonical_oracle import load_working_dataset
    df = load_working_dataset(H.WORKING_DATA)
    feature_cols       = get_feature_cols(df)
    available_elements = get_composition_cols(feature_cols)
    dataset_X          = df[feature_cols].values

    # Fixed novelty scale across ALL splits/seeds/arms (fairness)
    cov_inv     = descriptor_covariance(dataset_X, feature_cols)
    min_novelty = default_min_novelty(dataset_X, feature_cols, cov_inv)

    # LLM mode
    llm_mode = 'real'
    if 'llm' in arms:
        if force_mock_llm or not real_llm_available():
            install_mock_llm(); llm_mode = 'MOCK'
            print("  ⚠ LLM arm in MOCK mode — placeholder numbers.\n")
        else:
            print("  ✓ LLM arm using REAL API key.\n")

    # results[arm] = list over (split, seed) of run dicts
    results = {arm: [] for arm in arms}

    for split_seed in range(n_splits):
        splits = make_splits(df, random_seed=split_seed)
        oracle = CanonicalOracle(splits['train'], feature_cols)
        base_pool = np.vstack([splits['val'][feature_cols].values,
                               splits['hidden'][feature_cols].values])

        # B3 guard: 'none' worst case must fit the pool
        assert n_initial + n_iterations <= len(base_pool), \
            f"budget {n_initial+n_iterations} exceeds pool {len(base_pool)} " \
            f"on split {split_seed}"

        # Order arms so 'llm' runs FIRST on the first split — this makes the
        # real-LLM fail-fast guard (below) trigger within the first minutes,
        # not after the other arms' seeds have already burned time.
        arm_order = (['llm'] + [a for a in arms if a != 'llm']
                     if (split_seed == 0 and 'llm' in arms) else list(arms))
        print(f"  Split {split_seed}: pool={len(base_pool)}")
        for arm in arm_order:
            for bo_seed in range(n_bo_seeds):
                res = H.run_arm(
                    arm=arm, oracle=oracle, candidate_pool=base_pool,
                    feature_cols=feature_cols, available_elements=available_elements,
                    cov_inv=cov_inv, min_novelty=min_novelty, dataset_X=dataset_X,
                    n_initial=n_initial, n_iterations=n_iterations,
                    inject_n=inject_n, random_seed=bo_seed,
                )
                res['split_seed'] = split_seed
                if res['terminated_early']:
                    raise RuntimeError(
                        f"Run terminated early (pool exhausted): "
                        f"arm={arm} split={split_seed} seed={bo_seed}")
                results[arm].append(res)

                # Persist LLM traces
                if arm == 'llm' and res['intervention_log']:
                    fn = f'{save_dir}/llm_traces/split{split_seed}_seed{bo_seed}.json'
                    with open(fn, 'w', encoding='utf-8') as f:
                        json.dump(_clean(res['intervention_log']), f, indent=2, ensure_ascii=False)

                # Finding 2: fail-fast guard. In REAL mode, verify the LLM
                # actually returned usable candidates on its first invocation.
                # A set-but-invalid/expired/rate-limited key would otherwise
                # let the whole (multi-hour) run complete labelled REAL while
                # the LLM arm silently degenerated to no-injection.
                if (arm == 'llm' and llm_mode == 'real'
                        and split_seed == 0 and bo_seed == 0):
                    got_real = any(r.get('raw_candidates')
                                   for r in res['intervention_log'])
                    triggered = len(res['intervention_log']) > 0
                    if triggered and not got_real:
                        raise RuntimeError(
                            "REAL LLM mode but the first llm run produced zero "
                            "usable candidates across all injection events — "
                            "key may be invalid/expired/rate-limited, or the "
                            "response format changed. Run preflight_api_test.py "
                            "to diagnose before rerunning. Aborting to avoid a "
                            "wasted full run.")

                # Persist per-run detail needed by the Wave-2 analysis layer:
                # calibration (mu/sigma/y per pick), injected-candidate tags
                # (novelty/stability/hit), best_history, and convergence.
                detail = {
                    'arm'             : arm,
                    'split_seed'      : split_seed,
                    'bo_seed'         : bo_seed,
                    'final_best'      : res['final_best'],
                    'best_history'    : res['best_history'],
                    'calibration_log' : res['calibration_log'],
                    'candidate_tags'  : res['candidate_tags'],
                    'inject_events'   : res['inject_events'],
                    'trajectory_log'  : res['trajectory_log'],     # Finding 1
                    'stagnation_trace': res['stagnation_trace'],   # Finding 4
                }
                os.makedirs(f'{save_dir}/run_detail', exist_ok=True)
                dfn = f'{save_dir}/run_detail/{arm}_split{split_seed}_seed{bo_seed}.json'
                with open(dfn, 'w', encoding='utf-8') as f:
                    json.dump(_clean(detail), f, ensure_ascii=False)

        # progress line
        for arm in arms:
            finals = [r['final_best'] for r in results[arm]
                      if r['split_seed'] == split_seed]
            print(f"    {arm:>9}: final {np.mean(finals):.1f} ± {np.std(finals):.1f}")

    summary = _summarize(results, arms, llm_mode)
    _save(results, summary, arms, save_dir, llm_mode)
    _plot(results, arms, n_iterations, save_dir, llm_mode)
    return {'results': results, 'summary': summary, 'llm_mode': llm_mode}


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(obj):
    """JSON-safe: drop numpy, keep primitives."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float('nan')


def _paired_finals(results, arm_a, arm_b):
    """
    B1-new: build paired final-HV vectors for two arms, aligned EXPLICITLY
    on (split_seed, seed) keys rather than relying on list order. Raises
    if the two arms were not run on the same set of (split,seed) cells,
    so a mis-paired test can never pass silently.
    """
    def keyed(arm):
        d = {}
        for r in results[arm]:
            key = (r['split_seed'], r['seed'])
            if key in d:
                raise ValueError(f"duplicate (split,seed) {key} for arm {arm}")
            d[key] = r['final_best']
        return d

    A, B = keyed(arm_a), keyed(arm_b)
    if set(A.keys()) != set(B.keys()):
        raise ValueError(
            f"arms '{arm_a}' and '{arm_b}' have mismatched (split,seed) "
            f"cells — cannot pair. "
            f"only in {arm_a}: {set(A)-set(B)}; only in {arm_b}: {set(B)-set(A)}")

    keys = sorted(A.keys())
    return (np.array([A[k] for k in keys]),
            np.array([B[k] for k in keys]),
            keys)


def _per_split_signs(results, arm_a, arm_b):
    """
    B2-new: per-split median paired difference (arm_a − arm_b) and its
    sign, so we can report cross-split direction consistency alongside
    the pooled test.
    """
    a, b, keys = _paired_finals(results, arm_a, arm_b)
    diffs = a - b
    by_split = {}
    for (split, _seed), d in zip(keys, diffs):
        by_split.setdefault(split, []).append(d)
    out = {}
    for split, ds in by_split.items():
        med = float(np.median(ds))
        out[split] = {'median_diff': med,
                      'sign': int(np.sign(med)),
                      'n': len(ds)}
    signs = [v['sign'] for v in out.values()]
    consistent = len(set(s for s in signs if s != 0)) <= 1
    return {'per_split': out, 'direction_consistent': bool(consistent)}


def _summarize(results, arms, llm_mode):
    summary = {'llm_mode': llm_mode,
               'claim_scope': 'rescue-on-benchmark; NOT a discovery claim',
               'arms': {}, 'tests': {}}

    for arm in arms:
        runs   = results[arm]
        finals = np.array([r['final_best'] for r in runs])
        summary['arms'][arm] = {
            'n_runs'        : len(runs),
            'final_mean'    : float(finals.mean()),
            'final_std'     : float(finals.std()),
            'final_ci95'    : float(1.96*finals.std()/np.sqrt(len(finals))),
            'pickup_rate'   : _nanmean([r['pickup_rate'] for r in runs]),
            'cand_hit_rate' : _nanmean([r['per_candidate_hit_rate'] for r in runs]),
            'rescue_success': _nanmean([r['per_event_rescue_success'] for r in runs]),
            'escape_latency': _nanmean([r['median_escape_latency'] for r in runs]),
            'censored_frac' : _nanmean([r['escape_censored_frac'] for r in runs]),
            'novelty_maha'  : _nanmean([r['mean_novelty_maha'] for r in runs]),
            'stability_inj' : _nanmean([r['mean_stability_injected'] for r in runs]),
            'finals'        : finals.tolist(),
        }

    # Paired tests: llm vs each control, EXPLICITLY aligned on (split,seed).
    if 'llm' in arms:
        raw_p = {}
        for ctrl in arms:
            if ctrl == 'llm':
                continue
            llm_f, ctrl_f, _keys = _paired_finals(results, 'llm', ctrl)
            t = paired_tests(llm_f, ctrl_f)
            # B2-new: attach per-split direction consistency
            t.update(_per_split_signs(results, 'llm', ctrl))
            summary['tests'][f'llm_vs_{ctrl}'] = t
            raw_p[f'llm_vs_{ctrl}'] = t['wilcoxon_p']
        # Holm across the comparisons (B3-new: guard NaN)
        raw_p_clean = {k: (v if not np.isnan(v) else 1.0)
                       for k, v in raw_p.items()}
        adj = holm_bonferroni(raw_p_clean)
        for k, ap in adj.items():
            summary['tests'][k]['holm_p'] = ap

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE + PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def _save(results, summary, arms, save_dir, llm_mode):
    rows = []
    for arm in arms:
        a = summary['arms'][arm]
        tag = ' (MOCK)' if arm == 'llm' and llm_mode == 'MOCK' else ''
        rows.append({
            'arm'           : arm + tag,
            'final_HV'      : f"{a['final_mean']:.1f} ± {a['final_ci95']:.1f}",
            'pickup'        : f"{a['pickup_rate']:.2f}"    if not np.isnan(a['pickup_rate']) else '—',
            'cand_hit'      : f"{a['cand_hit_rate']:.2f}"  if not np.isnan(a['cand_hit_rate']) else '—',
            'rescue_succ'   : f"{a['rescue_success']:.2f}" if not np.isnan(a['rescue_success']) else '—',
            'escape_lat'    : f"{a['escape_latency']:.1f}" if not np.isnan(a['escape_latency']) else '—',
            'novelty'       : f"{a['novelty_maha']:.3f}"   if not np.isnan(a['novelty_maha']) else '—',
        })
    pd.DataFrame(rows).to_csv(f'{save_dir}/summary_table.csv', index=False, encoding='utf-8')
    with open(f'{save_dir}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(_clean(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {save_dir}/")


def _plot(results, arms, n_iterations, save_dir, llm_mode):
    colors = {'none':'#888','random':'#2196F3','mutation':'#4CAF50',
              'digest':'#FF9800','digest_ucb':'#9C27B0','llm':'#F44336'}
    L = n_iterations + 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm in arms:
        curves = np.array([
            np.array(r['best_history'][:L].copy() +
                     [r['best_history'][-1]]*(L-len(r['best_history'][:L])))
            if len(r['best_history']) < L else np.array(r['best_history'][:L])
            for r in results[arm]
        ])
        mean = curves.mean(axis=0)
        ci   = 1.96*curves.std(axis=0)/np.sqrt(len(curves))
        x    = np.arange(L)
        lab  = arm + (' (MOCK)' if arm=='llm' and llm_mode=='MOCK' else '')
        ax.plot(x, mean, color=colors.get(arm,'#000'), lw=2.4, label=lab, zorder=3)
        ax.fill_between(x, mean-ci, mean+ci, color=colors.get(arm,'#000'),
                        alpha=0.15, zorder=2)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Best HV (mean ± 95% CI)')
    title = 'Stagnation rescue on the Borg as-cast HEA benchmark'
    if llm_mode == 'MOCK':
        title += '  [LLM=MOCK]'
    ax.set_title(title, fontsize=12)
    ax.text(0.5, 1.015, 'rescue study — not a discovery claim',
            transform=ax.transAxes, ha='center', fontsize=8, color='#666')
    ax.legend(fontsize=9, title='Injection source'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figures/four_arm_convergence.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Convergence plot → {save_dir}/figures/four_arm_convergence.png")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    out = run_experiment(
        arms=('none','random','mutation','digest','llm'),
        n_splits=3, n_bo_seeds=7, n_iterations=50,
        force_mock_llm=False,   # REAL run — requires ANTHROPIC_API_KEY.
                                # Set True only to dry-run the pipeline on mock data.
    )
    s = out['summary']
    print("\n" + "="*70)
    print("  SUMMARY  " + ("[LLM=MOCK]" if out['llm_mode']=='MOCK' else "[LLM=REAL]"))
    print("  claim scope: rescue-on-benchmark; NOT discovery")
    print("="*70)
    print(f"  {'arm':<10}{'final HV':>15}{'pickup':>8}{'hit':>7}{'rescue':>8}{'lat':>6}{'novelty':>9}")
    for arm in ('none','random','mutation','digest','llm'):
        a = s['arms'][arm]
        def f(x, p='.2f'): return format(x, p) if not np.isnan(x) else '—'
        print(f"  {arm:<10}{a['final_mean']:>9.1f}±{a['final_ci95']:>4.1f}"
              f"{f(a['pickup_rate']):>8}{f(a['cand_hit_rate']):>7}"
              f"{f(a['rescue_success']):>8}{f(a['escape_latency'],'.1f'):>6}"
              f"{f(a['novelty_maha'],'.3f'):>9}")
    print("\n  Paired tests (llm vs controls), Holm-corrected:")
    for k, t in s['tests'].items():
        print(f"    {k:<16} Δ={t['median_diff']:+.1f} "
              f"CI[{t['median_ci95'][0]:+.1f},{t['median_ci95'][1]:+.1f}]  "
              f"p={t['wilcoxon_p']:.4f} holm={t['holm_p']:.4f}  "
              f"r={t['rank_biserial']:+.2f}  "
              f"(W{t['llm_wins']}/T{t['ties']}/L{t['ctrl_wins']})")
    if out['llm_mode']=='MOCK':
        print("\n  ⚠ LLM numbers are MOCK placeholders. Set ANTHROPIC_API_KEY,")
        print("    rerun with force_mock_llm=False for real results.")