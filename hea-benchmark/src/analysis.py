"""
Wave-2 Analysis Layer
=====================
Consumes the output of four_arm_runner_v2 and produces the finished
analyses and figures for the paper. Does NOT touch the optimization
loop — pure post-hoc analysis, so it runs identically on the mock run
(for validation now) and on the real run (once the API key is set).

Four components, mapping to the issue register:

  B2 — Statistical report: consolidated table of final-HV comparisons
       with Holm-corrected p, rank-biserial effect size, bootstrap CIs,
       plus per-split direction-consistency check (does llm's sign vs
       each control hold across all splits? — the B1 credibility test).

  C2 — GP calibration (on ACQUISITION-SELECTED points; these are
       argmax-EI picks, a biased slice favouring high-σ points, so
       coverage here is not general calibration): pooled residuals z=(y-μ)/σ from
       every pick, empirical ±1σ/±2σ coverage, and NLL. Tells us whether
       the surrogate's uncertainty (which drives EI) is trustworthy.

  E3 — Dual-channel oracle-reliability: annotate each arm's queried
       points and injected candidates with SSH proxy + RF confidence
       (distance-to-train). Reports, per arm: mean confidence of the
       region each arm's search operated in, and channel agreement
       (Spearman) on each arm's top candidates. This is the promised
       "characterize where the oracle is trustworthy" analysis.

  D6 — LLM trace analysis: consumes persisted intervention logs. Per
       rescue event: did the LLM's proposal get admitted, picked, and
       did it escape? Aggregates reasoning themes and links reasoning to
       outcome (which is the qualitative core of the paper's discussion).
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_run(save_dir):
    """Load summary + all per-run detail + traces from a run directory."""
    summary = json.load(open(f'{save_dir}/summary.json', encoding='utf-8'))

    details = []
    for fn in sorted(glob.glob(f'{save_dir}/run_detail/*.json')):
        details.append(json.load(open(fn, encoding='utf-8')))

    traces = []
    for fn in sorted(glob.glob(f'{save_dir}/llm_traces/*.json')):
        traces.append({'file': os.path.basename(fn),
                       'events': json.load(open(fn, encoding='utf-8'))})

    return summary, details, traces


# ═══════════════════════════════════════════════════════════════════════════════
# B2 — STATISTICAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def statistical_report(summary, details, save_dir):
    """
    Consolidated stats table + per-split direction-consistency check.
    The direction-consistency check is the B1 credibility test: a claim
    is split-robust only if llm's sign vs a control is the same across
    all splits.
    """
    lines = ["STATISTICAL REPORT", "="*60]
    lines.append(f"LLM mode: {summary['llm_mode']}")
    lines.append(f"Claim scope: {summary['claim_scope']}")
    lines.append("")

    # Headline table
    lines.append("Final HV by arm (mean ± 95% CI over all splits×seeds):")
    for arm, a in summary['arms'].items():
        lines.append(f"  {arm:<10}: {a['final_mean']:7.1f} ± {a['final_ci95']:.1f}"
                     f"   (n={a['n_runs']})")
    lines.append("")

    # Paired tests (already Holm-corrected in runner)
    lines.append("LLM vs controls (paired, Holm-corrected):")
    for k, t in summary['tests'].items():
        sig = "significant" if t.get('holm_p', 1) < 0.05 else "n.s."
        lines.append(
            f"  {k:<16}: Δmedian={t['median_diff']:+6.1f} HV  "
            f"CI[{t['median_ci95'][0]:+.1f},{t['median_ci95'][1]:+.1f}]  "
            f"holm_p={t.get('holm_p',float('nan')):.4f} ({sig})  "
            f"effect r={t['rank_biserial']:+.2f}")
    lines.append("")

    # E2: contextualize every arm difference against the oracle's own
    # out-of-sample error, so a statistically-significant gap that is
    # SMALLER than oracle MAE is read honestly, not oversold.
    ORACLE_OOS_MAE = 78.0   # honest hidden-set MAE of the canonical oracle
    lines.append(f"Oracle-error context (E2): canonical oracle OOS MAE "
                 f"= {ORACLE_OOS_MAE:.0f} HV.")
    for k, t in summary['tests'].items():
        d = abs(t['median_diff'])
        rel = ("EXCEEDS" if d > ORACLE_OOS_MAE else
               "within" if d > 0.5*ORACLE_OOS_MAE else "well within")
        lines.append(f"  {k:<16}: |Δ|={d:.1f} HV {rel} oracle MAE — "
                     + ("physically meaningful margin" if d > ORACLE_OOS_MAE
                        else "interpret as benchmark-optimization behaviour, "
                             "not a claim of physical hardness difference"))
    lines.append("")

    # B1 direction-consistency across splits
    lines.append("Per-split direction check (B1 credibility):")
    by = {}
    for d in details:
        by.setdefault((d['arm'], d['split_seed']), []).append(d['final_best'])
    splits = sorted({d['split_seed'] for d in details})
    for ctrl in [a for a in summary['arms'] if a != 'llm']:
        signs = []
        for sp in splits:
            llm_mean  = np.mean(by.get(('llm', sp), [np.nan]))
            ctrl_mean = np.mean(by.get((ctrl, sp), [np.nan]))
            signs.append(int(np.sign(llm_mean - ctrl_mean)))
        consistent = len(set(signs)) == 1
        lines.append(f"  llm vs {ctrl:<9}: per-split signs {signs}  "
                     f"{'CONSISTENT' if consistent else 'INCONSISTENT'}")

    lines.append("Control-sensitivity note (E1): the mutation arm uses "
                 "sigma=0.05, top_k=3. These are fixed heuristics and the "
                 "mutation control's strength depends on them; this is a "
                 "stated limitation. A sigma sweep is planned as a "
                 "robustness check (Wave 3) but the reported conclusions do "
                 "not assume it.")
    lines.append("")
    report = "\n".join(lines)
    open(f'{save_dir}/statistical_report.txt', 'w', encoding='utf-8').write(report)
    print(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# C2 — GP CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibration_analysis(details, save_dir):
    """
    Pool all (mu, sigma, y) from every pick across all runs; compute
    standardized residuals and coverage. If the GP is well-calibrated,
    ~68% of |z|<1 and ~95% of |z|<2.
    """
    mu = np.array([c['mu'] for d in details for c in d['calibration_log']])
    sg = np.array([c['sigma'] for d in details for c in d['calibration_log']])
    y  = np.array([c['y'] for d in details for c in d['calibration_log']])

    sg_safe = np.maximum(sg, 1e-6)
    z = (y - mu) / sg_safe

    cov1 = float(np.mean(np.abs(z) < 1))
    cov2 = float(np.mean(np.abs(z) < 2))
    nll  = float(np.mean(0.5*np.log(2*np.pi*sg_safe**2) + (y-mu)**2/(2*sg_safe**2)))

    out = {
        'n_predictions'  : int(len(z)),
        'coverage_1sigma': cov1, 'target_1sigma': 0.683,
        'coverage_2sigma': cov2, 'target_2sigma': 0.954,
        'mean_nll'       : nll,
        'z_mean'         : float(z.mean()),
        'z_std'          : float(z.std()),
    }

    # Calibration plot: reliability of predictive intervals
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(z, bins=40, density=True, alpha=0.6, color='steelblue',
               label='standardized residuals')
    xs = np.linspace(-4, 4, 200)
    ax[0].plot(xs, norm.pdf(xs), 'r-', lw=2, label='N(0,1) ideal')
    ax[0].set_xlabel('z = (y − μ)/σ'); ax[0].set_ylabel('density')
    ax[0].set_title(f'GP calibration on acquisition-selected points  (n={len(z)})')
    ax[0].legend(fontsize=9)

    # Coverage vs nominal
    noms = np.linspace(0.05, 0.95, 19)
    emp  = [float(np.mean(np.abs(z) < norm.ppf(0.5 + n/2))) for n in noms]
    ax[1].plot([0,1],[0,1],'k--',alpha=0.5,label='perfect')
    ax[1].plot(noms, emp, 'o-', color='darkorange', label='empirical')
    ax[1].set_xlabel('nominal coverage'); ax[1].set_ylabel('empirical coverage')
    ax[1].set_title('Reliability diagram'); ax[1].legend(fontsize=9)
    plt.tight_layout()
    fig.suptitle('Borg as-cast HEA benchmark — rescue study (not a discovery claim)',
                 fontsize=8, y=1.02, color='#666')
    plt.savefig(f'{save_dir}/figures/gp_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

    out['note'] = ('Calibration computed on acquisition-selected (argmax-EI) '
                   'points only, not a uniform sample of input space; EI '
                   'favours high-uncertainty points, so coverage may appear '
                   'conservative. Interpret as calibration-where-it-acts.')
    json.dump(out, open(f'{save_dir}/calibration.json','w',encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\nGP CALIBRATION (n={out['n_predictions']} predictions):")
    print(f"  ±1σ coverage: {cov1:.3f} (target 0.683)")
    print(f"  ±2σ coverage: {cov2:.3f} (target 0.954)")
    print(f"  mean NLL: {nll:.3f}   z: {out['z_mean']:+.2f} ± {out['z_std']:.2f}")
    verdict = ("well-calibrated" if 0.60 < cov1 < 0.76 else
               "OVER-confident (σ too small)" if cov1 < 0.60 else
               "UNDER-confident (σ too large)")
    print(f"  → {verdict}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# E3 — DUAL-CHANNEL ORACLE-RELIABILITY ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════

def dual_channel_analysis(details, save_dir):
    """
    Per arm: mean RF-confidence (distance-to-train) of the injected
    candidates, and SSH-proxy vs RF-HV Spearman agreement on them.
    Characterizes whether each arm searched in oracle-trustworthy
    territory and whether the physics channel corroborates its picks.
    """
    import pandas as pd
    from canonical_oracle import make_splits, CanonicalOracle, get_feature_cols, get_composition_cols
    from ss_strengthening import compute_ss_proxy
    from experiment_harness_v2 import WORKING_DATA

    from canonical_oracle import load_working_dataset
    df = load_working_dataset(WORKING_DATA)
    fc = get_feature_cols(df); comp_cols = get_composition_cols(fc)

    oracles = {}
    for sp in sorted({d['split_seed'] for d in details}):
        splits = make_splits(df, random_seed=sp)
        oracles[sp] = CanonicalOracle(splits['train'], fc)

    def vec_to_comp(vec):
        return {c: vec[fc.index(c)] for c in comp_cols if vec[fc.index(c)] > 1e-6}

    per_arm = {}
    for d in details:
        arm = d['arm']
        if arm == 'none':
            continue
        oracle = oracles[d['split_seed']]
        A = per_arm.setdefault(arm, {'conf': [], 'rf': [], 'ssh': []})
        for ev in d['inject_events']:
            for vec in ev.get('vectors', []):
                v   = np.array(vec)
                A['conf'].append(oracle.confidence(v))
                A['rf'].append(oracle.query(v))
                ssh = compute_ss_proxy(vec_to_comp(v))
                A['ssh'].append(ssh if ssh is not None else np.nan)

    out = {}
    print("\nDUAL-CHANNEL ORACLE-RELIABILITY (E3):")
    print(f"  {'arm':<10}{'n_inj':>7}{'mean_conf':>11}{'rf_ssh_rho':>12}")
    for arm, A in per_arm.items():
        conf = np.array(A['conf']); rf = np.array(A['rf']); ssh = np.array(A['ssh'])
        mask = ~np.isnan(ssh)
        if mask.sum() >= 3:
            rho, _ = spearmanr(rf[mask], ssh[mask])
        else:
            rho = np.nan
        out[arm] = {
            'n_injected'   : int(len(conf)),
            'mean_conf'    : float(np.mean(conf)) if len(conf) else float('nan'),
            'rf_ssh_spearman': float(rho),
            'ssh_valid_frac': float(mask.mean()) if len(ssh) else float('nan'),
        }
        print(f"  {arm:<10}{len(conf):>7}{out[arm]['mean_conf']:>11.3f}"
              f"{out[arm]['rf_ssh_spearman']:>12.3f}")

    print("  (higher mean_conf = searched FURTHER from training data =")
    print("   oracle predictions there are less reliable; interpret with care)")
    json.dump(out, open(f'{save_dir}/dual_channel.json','w',encoding='utf-8'), indent=2, ensure_ascii=False)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# D6 — LLM TRACE / QUALITATIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def trace_analysis(traces, details, save_dir):
    """
    Link LLM reasoning to outcome. For each rescue event: what did the
    LLM claim, how many candidates were admitted, did any escape.
    Aggregates reasoning-mechanism keywords and their success rate.
    """
    if not traces:
        print("\nLLM TRACE ANALYSIS: no traces (non-LLM run or no key).")
        return {}

    # Gather llm run details keyed by (split,seed)
    llm_details = {(d['split_seed'], d['bo_seed']): d
                   for d in details if d['arm'] == 'llm'}

    events = []
    for tr in traces:
        # filename split{S}_seed{B}.json
        base = tr['file'].replace('.json','')
        sp   = int(base.split('split')[1].split('_')[0])
        bo   = int(base.split('seed')[1])
        det  = llm_details.get((sp, bo))
        ev_escaped = {}
        if det:
            for e in det['inject_events']:
                ev_escaped[e['iteration']] = e['escaped']
        for rec in tr['events']:
            it = rec.get('iteration')
            events.append({
                'split'     : sp, 'seed': bo, 'iteration': it,
                'stagnation': rec.get('stagnation_count'),
                'reasoning' : (rec.get('reasoning') or '')[:200],
                'mechanism' : (rec.get('mechanism') or ''),
                'n_raw'     : len(rec.get('raw_candidates', [])),
                'escaped'   : ev_escaped.get(it, None),
            })

    edf = pd.DataFrame(events)
    edf.to_csv(f'{save_dir}/llm_events.csv', index=False)

    n_events   = len(edf)
    n_escaped  = int(edf['escaped'].sum()) if 'escaped' in edf and edf['escaped'].notna().any() else 0
    escape_rate = n_escaped / n_events if n_events else float('nan')

    print(f"\nLLM TRACE ANALYSIS ({n_events} rescue events):")
    print(f"  events that escaped: {n_escaped} ({escape_rate:.1%})")
    if 'mechanism' in edf and edf['mechanism'].notna().any():
        print("  mechanisms cited (top):")
        for mech, cnt in edf['mechanism'].value_counts().head(5).items():
            sub = edf[edf['mechanism']==mech]
            esc = sub['escaped'].sum() if sub['escaped'].notna().any() else 0
            print(f"    '{mech[:45]}': {cnt}×  (escaped {int(esc)})")

    return {'n_events': n_events, 'escape_rate': escape_rate,
            'events_csv': f'{save_dir}/llm_events.csv'}


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def rescue_frequency_diagnostic(details, save_dir):
    """
    Rescue-frequency diagnostic — answers the reviewer question
    "does the LLM only help because it gets called more often?"
    WITHOUT requiring a rerun, using data already recorded during the run
    (stagnation_trace + inject_events).

    Why this matters: rescue frequency is NOT purely a treatment — it is
    partly an OUTCOME. Triggering depends on whether the loop is improving,
    and injection changes whether the loop improves. An arm whose candidates
    are good escapes stagnation and stops triggering; an arm whose candidates
    are poor stays stuck and keeps triggering. So event counts can differ
    across arms at a FIXED trigger setting, and an arm that receives more
    injections gets more shots on goal independent of candidate quality.

    Reports, per arm:
      events_per_run     — injection events actually fired (attempt-based)
      admitted_per_run   — candidates that survived admission (shots on goal)
      detector_fired     — iterations the stagnation detector fired
      cooldown_blocked   — detector fired but cooldown suppressed injection
      admit_ratio        — admitted / proposed (dedup burden per arm)

    INTERPRETATION (decide before looking):
      - If events/admitted are COMPARABLE across arms → the "called more
        often" critique is answered with one sentence, no ablation needed.
      - If they DIVERGE materially → the confound is live, and the clean
        answer is a fixed-injection-schedule variant (inject every N
        iterations regardless of stagnation), which decouples frequency
        from performance entirely. A stagnation-threshold sweep does NOT
        answer this, because it changes the trigger identically for all
        arms and so re-runs the same coupling at three settings.
    """
    by_arm = {}
    for d in details:
        arm = d['arm']
        a = by_arm.setdefault(arm, {'events': [], 'admitted': [], 'fired': [],
                                    'blocked': [], 'proposed': []})
        events = d.get('inject_events', [])
        a['events'].append(len(events))
        a['admitted'].append(sum(e.get('n_admitted', 0) for e in events))
        a['proposed'].append(sum(e.get('n_proposed', 0) for e in events))
        trace = d.get('stagnation_trace', [])
        a['fired'].append(sum(1 for s in trace if s.get('stagnating')))
        a['blocked'].append(sum(1 for s in trace
                                if s.get('stagnating') and not s.get('cooldown_ok')))

    out = {}
    print("\n  Rescue-frequency diagnostic (is the LLM just called more often?):")
    print(f"    {'arm':<10}{'events/run':>12}{'admitted/run':>14}"
          f"{'fired':>8}{'blocked':>9}{'admit_ratio':>13}")
    for arm, a in by_arm.items():
        ev   = float(np.mean(a['events']))
        adm  = float(np.mean(a['admitted']))
        prop = float(np.mean(a['proposed']))
        ratio = (adm / prop) if prop > 0 else float('nan')
        out[arm] = {
            'events_per_run'   : ev,
            'admitted_per_run' : adm,
            'proposed_per_run' : prop,
            'admit_ratio'      : ratio,
            'detector_fired'   : float(np.mean(a['fired'])),
            'cooldown_blocked' : float(np.mean(a['blocked'])),
        }
        rs = f"{ratio:.2f}" if not np.isnan(ratio) else '—'
        print(f"    {arm:<10}{ev:>12.1f}{adm:>14.1f}"
              f"{np.mean(a['fired']):>8.1f}{np.mean(a['blocked']):>9.1f}{rs:>13}")

    # Flag divergence among injection arms (exclude 'none' — it never injects)
    inj = {k: v for k, v in out.items() if k != 'none'}
    if len(inj) >= 2:
        evs = [v['events_per_run'] for v in inj.values()]
        adm = [v['admitted_per_run'] for v in inj.values()]
        ev_spread  = (max(evs) - min(evs)) / max(np.mean(evs), 1e-9)
        adm_spread = (max(adm) - min(adm)) / max(np.mean(adm), 1e-9)
        out['_divergence'] = {'events_rel_spread': float(ev_spread),
                              'admitted_rel_spread': float(adm_spread)}
        print(f"\n    Relative spread across injection arms: "
              f"events {ev_spread:.1%}, admitted {adm_spread:.1%}")
        if ev_spread < 0.15 and adm_spread < 0.20:
            verdict = ("COMPARABLE — arms received similar rescue opportunity; "
                       "the 'called more often' critique can be answered by "
                       "reporting these numbers. No frequency ablation needed.")
        else:
            verdict = ("DIVERGENT — arms did NOT receive comparable rescue "
                       "opportunity. Frequency is confounded with performance; "
                       "run the fixed-injection-schedule variant (inject every "
                       "N iterations regardless of stagnation) to decouple.")
        out['_verdict'] = verdict
        print(f"    → {verdict}")

    json.dump(out, open(f'{save_dir}/rescue_frequency.json','w',encoding='utf-8'), indent=2, ensure_ascii=False)
    return out


def run_full_analysis(save_dir):
    print("="*64)
    print(f"  WAVE-2 ANALYSIS: {save_dir}")
    print("="*64)
    summary, details, traces = load_run(save_dir)

    statistical_report(summary, details, save_dir)
    rescue_frequency_diagnostic(details, save_dir)
    calibration_analysis(details, save_dir)
    dual_channel_analysis(details, save_dir)
    trace_analysis(traces, details, save_dir)

    print("\n" + "="*64)
    print("  Wave-2 analysis complete. Artifacts in", save_dir)
    print("="*64)


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else '../results/four_arm_v2_smoke'
    run_full_analysis(d)