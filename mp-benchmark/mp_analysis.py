"""
MP Shear-Modulus Rescue — Analysis Layer
========================================
Reads results/mp_shear_v1/ and produces every table/figure the paper needs.
NO re-running: every quantity below is computed from data the run already
recorded (run_detail/*.json, llm_traces/*.json, summary.json).

SECTIONS (each answers a specific reviewer point or design question):
  1. Headline stats     — paired diffs, effect sizes, bootstrap CIs,
                          per-split direction consistency
  2. GP calibration     — +/-1sigma, +/-2sigma coverage, NLL, z-stats
                          (reviewer pt 9). NOTE: calibration on ACQUISITION-
                          SELECTED points, which argmax-EI biases toward
                          high-sigma regions — reported as such, not as
                          general calibration.
  3. Admission by arm   — proposed/admitted/reject-reason per arm
                          (reviewer pt 2 / distance-gate bias). Answers
                          "did the gate suppress the LLM?"
  4. Rescuable events   — how many stagnation events were actually rescuable
                          (gap-to-ceiling > 1 MAE at the moment of stall).
                          Tells you how hard the hypothesis was really tested.
  5. Region overlap     — do mutation and the LLM propose the SAME regions?
                          (reviewer pt 5). If yes, an LLM lead is local search
                          in disguise.
  6. Cost-benefit       — per-arm generation time vs. benefit (reviewer pt 10)
  7. Outlier traces     — surface the LLM runs that beat the field, with the
                          model's own reasoning, to judge "higher ceiling"
                          vs. noise.

Usage:
    python mp_analysis.py                       # results/mp_shear_v1
    python mp_analysis.py results/mp_shear_v1
"""

import os
import sys
import json
import glob
import warnings
import numpy as np

warnings.filterwarnings('ignore')

ORACLE_MAE = 11.1         # GPa, from oracle diagnostics (per-split ~12-13)
OBJ = 'G'                  # objective label


# ══════════════════════════════════════════════════════════════════════════
# loading
# ══════════════════════════════════════════════════════════════════════════

def load(save_dir):
    summary = json.load(open(f'{save_dir}/summary.json', encoding='utf-8'))
    details = []
    for fn in sorted(glob.glob(f'{save_dir}/run_detail/*.json')):
        details.append(json.load(open(fn, encoding='utf-8')))
    traces = {}
    for fn in sorted(glob.glob(f'{save_dir}/llm_traces/*.json')):
        traces[os.path.basename(fn)] = json.load(open(fn, encoding='utf-8'))
    return summary, details, traces


def _by_arm(details):
    d = {}
    for r in details:
        d.setdefault(r['arm'], []).append(r)
    return d


def _paired(details, a, b):
    """Aligned final values on (split_seed, bo_seed)."""
    def keyed(arm):
        return {(r['split_seed'], r['bo_seed']): r['final_best']
                for r in details if r['arm'] == arm}
    A, B = keyed(a), keyed(b)
    keys = sorted(set(A) & set(B))
    return (np.array([A[k] for k in keys]),
            np.array([B[k] for k in keys]), keys)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/d) if d else float('nan')


# ══════════════════════════════════════════════════════════════════════════
# 1. headline stats
# ══════════════════════════════════════════════════════════════════════════

def headline(summary, details, out):
    from scipy.stats import wilcoxon
    print("\n" + "="*70)
    print("  1. HEADLINE — paired LLM-vs-control, effect sizes, CIs")
    print("="*70)
    print(f"  {'arm':<10}{'mean '+OBJ:>10}{'CI95':>8}{'std':>8}")
    for arm, a in summary['arms'].items():
        print(f"  {arm:<10}{a['final_mean']:>10.1f}{a['final_ci95']:>8.1f}"
              f"{a['final_std']:>8.1f}")

    print(f"\n  {'comparison':<18}{'Δmed':>7}{'boot CI95':>16}{'r':>7}"
          f"{'holm_p':>9}{'W/T/L':>10}{'splits':>9}")
    res = {}
    for ctrl in summary['arms']:
        if ctrl == 'llm':
            continue
        lf, cf, keys = _paired(details, 'llm', ctrl)
        diffs = lf - cf
        # rank-biserial effect size
        nz = diffs[diffs != 0]
        if len(nz):
            r = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz)
        else:
            r = 0.0
        # bootstrap CI on median diff
        boot = [np.median(np.random.choice(diffs, len(diffs), replace=True))
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # per-split direction
        splits = sorted(set(k[0] for k in keys))
        signs = []
        for s in splits:
            sd = np.array([lf[i]-cf[i] for i,k in enumerate(keys) if k[0]==s])
            signs.append(int(np.sign(np.median(sd))))
        consistent = len(set(s for s in signs if s != 0)) <= 1
        holm = summary['tests'][f'llm_vs_{ctrl}'].get('holm_p', float('nan'))
        t = summary['tests'][f'llm_vs_{ctrl}']
        print(f"  llm_vs_{ctrl:<11}{np.median(diffs):>7.1f}"
              f"  [{lo:>5.1f},{hi:>5.1f}]{r:>7.2f}{holm:>9.4f}"
              f"  W{t['llm_wins']}/T{t['ties']}/L{t['ctrl_wins']}"
              f"{'CONS' if consistent else 'incon':>9}")
        res[f'llm_vs_{ctrl}'] = {'median_diff': float(np.median(diffs)),
            'boot_ci': [float(lo), float(hi)], 'rank_biserial': float(r),
            'per_split_signs': signs, 'direction_consistent': consistent}
    out['headline'] = res
    print("\n  Interpretation gate (E2): |Δ| < 1 oracle MAE (13 GPa) → read as")
    print("  optimization behaviour, NOT a physical shear-modulus claim.")


# ══════════════════════════════════════════════════════════════════════════
# 2. GP calibration (reviewer pt 9)
# ══════════════════════════════════════════════════════════════════════════

def calibration(details, out):
    print("\n" + "="*70)
    print("  2. GP CALIBRATION  (on acquisition-selected points)")
    print("="*70)
    mu, sd, y = [], [], []
    for r in details:
        for c in r.get('calibration_log', []):
            if c['sigma'] and c['sigma'] > 0:
                mu.append(c['mu']); sd.append(c['sigma']); y.append(c['y'])
    mu, sd, y = np.array(mu), np.array(sd), np.array(y)
    if len(mu) == 0:
        print("  no calibration data"); return
    z = (y - mu) / sd
    cov1 = float(np.mean(np.abs(z) <= 1))
    cov2 = float(np.mean(np.abs(z) <= 2))
    nll = float(np.mean(0.5*np.log(2*np.pi*sd**2) + 0.5*z**2))
    print(f"  n predictions      : {len(mu)}")
    print(f"  ±1σ coverage       : {cov1:.3f}   (target 0.683)")
    print(f"  ±2σ coverage       : {cov2:.3f}   (target 0.954)")
    print(f"  mean NLL           : {nll:.3f}")
    print(f"  z mean ± std       : {z.mean():+.3f} ± {z.std():.3f}")
    verdict = ("well-calibrated" if abs(cov1-0.683) < 0.08
               else "over-confident" if cov1 < 0.683 else "under-confident")
    print(f"  → {verdict}")
    print("  NOTE: these are argmax-EI-selected points; EI over-samples high-σ")
    print("  regions, so this is calibration WHERE THE BO QUERIED, not general.")
    out['calibration'] = {'n': len(mu), 'cov_1sigma': cov1, 'cov_2sigma': cov2,
                          'nll': nll, 'z_mean': float(z.mean()),
                          'z_std': float(z.std()), 'verdict': verdict}


# ══════════════════════════════════════════════════════════════════════════
# 3. admission by arm (reviewer pt 2)  — from summary, restated as the check
# ══════════════════════════════════════════════════════════════════════════

def admission(summary, out):
    print("\n" + "="*70)
    print("  3. ADMISSION BY ARM  (did the distance gate suppress the LLM?)")
    print("="*70)
    ab = summary.get('admission_by_arm', {})
    print(f"  {'arm':<10}{'proposed':>10}{'admitted':>10}{'admit%':>8}"
          f"{'beyond_region':>15}{'near_dup':>10}")
    llm_rate = ab.get('llm', {}).get('admit_rate')
    for arm, d in ab.items():
        rej = d.get('rejects_by_reason', {})
        print(f"  {arm:<10}{d['proposed']:>10}{d['admitted']:>10}"
              f"{100*d['admit_rate']:>7.1f}%{rej.get('beyond_reliable_region',0):>15}"
              f"{rej.get('near_duplicate',0):>10}")
    print()
    if llm_rate is not None:
        others = [d['admit_rate'] for a, d in ab.items() if a != 'llm']
        if llm_rate >= max(others):
            print("  → LLM had the HIGHEST admit rate: the gate did NOT suppress")
            print("    it. The null is not an artifact of clipped exploration —")
            print("    the LLM got a fair (indeed the least-constrained) test.")
        else:
            print("  → LLM admit rate is BELOW some controls: report this; the")
            print("    gate may partially account for any LLM under-performance.")
    out['admission'] = ab


# ══════════════════════════════════════════════════════════════════════════
# 4. rescuable events actually encountered
# ══════════════════════════════════════════════════════════════════════════

def rescuable(details, out):
    print("\n" + "="*70)
    print("  4. RESCUABLE EVENTS  (was the hypothesis actually tested hard?)")
    print("="*70)
    # TRUE per-split pool ceilings (max RF prediction over each split's pool),
    # printed by the runner at launch. Using the true ceiling instead of the
    # per-run trajectory maximum is essential: the trajectory proxy is
    # circular (a run that stalls early never SEES the pool's high points, so
    # its own max is low, so headroom looks small, so saturation is
    # manufactured). The true ceiling measures headroom that actually existed.
    POOL_CEILINGS = {0: 157.0, 1: 212.5, 2: 146.7, 3: 128.0, 4: 135.4}

    by = _by_arm(details)
    print(f"  {'arm':<10}{'events/run':>12}{'rescuable/run':>15}{'resc_frac':>11}")
    rows = {}
    for arm, runs in by.items():
        if arm == 'none':
            continue
        ev_counts, resc_counts = [], []
        for r in runs:
            ceil = POOL_CEILINGS.get(r['split_seed'])
            if ceil is None:
                traj = r.get('trajectory_log', [])
                ceil = max([t['hv'] for t in traj], default=max(r['best_history']))
            n_ev = n_resc = 0
            for ev in r.get('inject_events', []):
                n_ev += 1
                inc = ev.get('incumbent_before', r['best_history'][-1])
                if (ceil - inc) >= ORACLE_MAE:
                    n_resc += 1
            ev_counts.append(n_ev); resc_counts.append(n_resc)
        me, mr = np.mean(ev_counts), np.mean(resc_counts)
        frac = mr/me if me else 0
        rows[arm] = {'events_per_run': float(me), 'rescuable_per_run': float(mr),
                     'rescuable_frac': float(frac)}
        print(f"  {arm:<10}{me:>12.1f}{mr:>15.1f}{frac:>10.0%}")
    print("\n  Ceiling = TRUE per-split pool max (not trajectory proxy). If")
    print("  rescuable/run is still low, the loop genuinely reached near the")
    print("  ceiling before stalling and the null is underpowered by design.")
    print("  If it is now HIGH, the earlier low count was a proxy artifact and")
    print("  the hypothesis WAS tested — the null stands on its own.")
    out['rescuable'] = rows


# ══════════════════════════════════════════════════════════════════════════
# 5. mutation / LLM region overlap (reviewer pt 5)
# ══════════════════════════════════════════════════════════════════════════

def region_overlap(details, out):
    print("\n" + "="*70)
    print("  5. REGION OVERLAP  (is an LLM lead just local search?)")
    print("="*70)

    def top2(comp):
        parts = sorted(((e, f) for e, f in comp.items() if f > 0.01),
                       key=lambda p: -p[1])
        return "-".join(sorted(e for e, _ in parts[:2]))

    def clusters(arm):
        cl = {}
        for r in details:
            if r['arm'] != arm:
                continue
            for ev in r.get('inject_events', []):
                for comp in ev.get('compositions', []):
                    k = top2(comp); cl[k] = cl.get(k, 0) + 1
        return cl

    mut, llm = clusters('mutation'), clusters('llm')
    if not mut or not llm:
        print("  insufficient injected-composition data"); return
    shared = set(mut) & set(llm)
    only_llm = set(llm) - set(mut)
    j = len(shared) / len(set(mut) | set(llm))
    print(f"  distinct 2-element regions — mutation: {len(mut)}, llm: {len(llm)}")
    print(f"  shared regions: {len(shared)}   Jaccard: {j:.2f}")
    print(f"  regions the LLM explored that mutation NEVER did: {len(only_llm)}")
    if only_llm:
        ex = sorted(only_llm, key=lambda k: -llm[k])[:8]
        print(f"    e.g. {', '.join(ex)}")
    print()
    if j > 0.6:
        print("  → High overlap: the LLM largely searched the same regions as")
        print("    mutation. Any LLM edge is hard to distinguish from local search.")
    else:
        print("  → Low overlap: the LLM explored genuinely different regions than")
        print("    mutation. Its behaviour is not reducible to local perturbation.")
    out['region_overlap'] = {'mutation_regions': len(mut), 'llm_regions': len(llm),
        'shared': len(shared), 'jaccard': float(j),
        'llm_only_examples': sorted(only_llm, key=lambda k: -llm[k])[:12]}


# ══════════════════════════════════════════════════════════════════════════
# 6. cost-benefit (reviewer pt 10)
# ══════════════════════════════════════════════════════════════════════════

def cost_benefit(summary, out):
    print("\n" + "="*70)
    print("  6. COST-BENEFIT  (generation cost per arm)")
    print("="*70)
    cb = summary.get('cost_by_arm', {})
    arms = summary['arms']
    base = arms['none']['final_mean'] if 'none' in arms else 0
    print(f"  {'arm':<10}{'sec/event':>12}{'total sec':>12}{'mean '+OBJ:>10}"
          f"{'Δ vs none':>11}")
    for arm, c in cb.items():
        m = c.get('mean_gen_seconds')
        tot = c.get('total_gen_seconds')
        fm = arms[arm]['final_mean']
        print(f"  {arm:<10}{(m if m else 0):>12.4f}{(tot if tot else 0):>12.1f}"
              f"{fm:>10.1f}{fm-base:>+11.1f}")
    # headline ratio
    if 'llm' in cb and 'digest' in cb:
        lm = cb['llm'].get('mean_gen_seconds') or 0
        dm = cb['digest'].get('mean_gen_seconds') or 1e-9
        print(f"\n  LLM is ~{lm/dm:,.0f}× slower per event than the digest control,")
        print("  which reads the SAME information. If the benefit is null-to-marginal,")
        print("  this cost asymmetry is itself a central practical finding.")
    out['cost_benefit'] = cb


# ══════════════════════════════════════════════════════════════════════════
# 7. outlier traces  — "higher ceiling" or noise?
# ══════════════════════════════════════════════════════════════════════════

def outlier_traces(details, traces, out):
    print("\n" + "="*70)
    print("  7. LLM OUTLIER RUNS  (did reasoning find the high-G points?)")
    print("="*70)
    llm = [r for r in details if r['arm'] == 'llm']
    if not llm:
        print("  no llm runs"); return
    finals = [(r['final_best'], r['split_seed'], r['bo_seed']) for r in llm]
    finals.sort(reverse=True)
    others_max = max((r['final_best'] for r in details if r['arm'] != 'llm'),
                     default=0)
    print(f"  best non-LLM final anywhere: {others_max:.1f}")
    print(f"  top LLM runs:")
    hits = []
    for fb, sp, seed in finals[:5]:
        flag = "  <-- exceeds all non-LLM" if fb > others_max else ""
        print(f"    split{sp} seed{seed}: {OBJ}={fb:.1f}{flag}")
        if fb > others_max:
            hits.append((sp, seed, fb))
    # surface the reasoning of the runs that beat the field
    if hits:
        print("\n  Reasoning from the field-beating run(s):")
        for sp, seed, fb in hits[:2]:
            key = f'split{sp}_seed{seed}.json'
            tr = traces.get(key, [])
            # find the event whose reasoning preceded the peak
            for rec in tr:
                if rec.get('reasoning'):
                    print(f"    [split{sp} seed{seed}, {OBJ}={fb:.1f}]")
                    print(f"    reasoning: {rec['reasoning'][:240]}")
                    if rec.get('mechanism'):
                        print(f"    mechanism: {rec['mechanism'][:160]}")
                    break
        print("\n  Judge: if the reasoning names the right chemistry (high-VED,")
        print("  refractory, strong-bond) and the peak followed, 'higher ceiling'")
        print("  is a supported claim. If reasoning is generic, treat as variance.")
    else:
        print("\n  No LLM run exceeded the best non-LLM final — the higher mean is")
        print("  driven by distribution shape, not unique discoveries.")
    out['outliers'] = {'best_non_llm': float(others_max),
                       'llm_field_beating': [{'split': s, 'seed': se, OBJ: f}
                                             for s, se, f in hits]}


# ══════════════════════════════════════════════════════════════════════════
def main(save_dir='results/mp_shear_v1'):
    print("="*70)
    print(f"  MP SHEAR-MODULUS RESCUE — FULL ANALYSIS  [{save_dir}]")
    print("="*70)
    summary, details, traces = load(save_dir)
    out = {}
    headline(summary, details, out)
    calibration(details, out)
    admission(summary, out)
    rescuable(details, out)
    region_overlap(details, out)
    cost_benefit(summary, out)
    outlier_traces(details, traces, out)
    # oracle ranking (from summary)
    orank = summary.get('oracle_ranking', {})
    if orank:
        print("\n" + "="*70)
        print("  ORACLE RANKING ACROSS SPLITS")
        print("="*70)
        print(f"  Spearman(RF, true {OBJ}) = {orank['mean']:.3f} "
              f"[{orank['ci95_lo']:.3f}, {orank['ci95_hi']:.3f}]  "
              f"(per split: {[round(r,3) for r in orank['per_split']]})")
        out['oracle_ranking'] = orank
    json.dump(out, open(f'{save_dir}/analysis_report.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f"\n  Full analysis JSON → {save_dir}/analysis_report.json")
    print("="*70)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/mp_shear_v1')