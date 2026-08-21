"""
Rescue-effectiveness analysis (post-hoc, no re-run).
====================================================
For every rescue (injection) event, measures whether the incumbent actually
improved in the k iterations AFTER the event, using data already in the run
JSONs (inject_events + best_history). Aggregates per arm.

It PRINTS THE DISTRIBUTION FIRST. Only if there is meaningful spread does a
figure earn its place; if post-rescue gains are essentially all zero, the
honest output is a one-sentence result, not a plot. The script tells you which.

Usage:
    python rescue_effectiveness.py results/mp_shear_v1
    python rescue_effectiveness.py results/mp_shear_v1 --k 5 --figure
"""
import os, sys, json, glob
import numpy as np

ARM_COLORS = {'random': '#c9a227', 'mutation': '#d1711f',
              'digest': '#2e7d32', 'llm': '#1f4e79'}


def collect(results_dir, k):
    """Return {arm: [post_rescue_gain, ...]} pooled over all runs.
    Gain = best_history[min(t+k, last)] - best_history[t] at each event iter t."""
    per_arm = {}
    files = (glob.glob(f'{results_dir}/run_detail/*.json')
             or glob.glob(f'{results_dir}/*.json'))
    n_files = 0
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        arm = d.get('arm')
        bh = d.get('best_history')
        evs = d.get('inject_events')
        if not arm or arm == 'none' or not bh or not evs:
            continue
        n_files += 1
        last = len(bh) - 1
        for ev in evs:
            t = ev.get('iteration')
            if t is None or t > last:
                continue
            t2 = min(t + k, last)
            gain = bh[t2] - bh[t]
            per_arm.setdefault(arm, []).append(gain)
    return per_arm, n_files


def summarize(per_arm, k):
    print("=" * 66)
    print(f"  POST-RESCUE GAIN OVER k={k} ITERATIONS  (incumbent after \u2212 before)")
    print("=" * 66)
    order = [a for a in ['random', 'mutation', 'digest', 'llm'] if a in per_arm]
    rows = []
    for a in order:
        g = np.array(per_arm[a], dtype=float)
        n = len(g)
        mean = g.mean() if n else 0.0
        frac_pos = float((g > 1e-9).mean()) if n else 0.0
        # bootstrap 95% CI on the mean
        if n:
            bs = [np.random.choice(g, n, replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = 0.0
        rows.append((a, n, mean, lo, hi, frac_pos))
        print(f"  {a:9s}  n={n:4d}  mean gain={mean:6.3f} GPa "
              f"[95% CI {lo:6.3f}, {hi:6.3f}]  improved={100*frac_pos:5.1f}% of events")
    print("=" * 66)
    # decision heuristic
    all_g = np.concatenate([np.array(per_arm[a], dtype=float) for a in order]) if order else np.array([])
    overall_pos = float((all_g > 1e-9).mean()) if all_g.size else 0.0
    spread = all_g.std() if all_g.size else 0.0
    print(f"  overall: {100*overall_pos:.1f}% of rescue events improved the incumbent; "
          f"gain SD = {spread:.3f} GPa")
    if overall_pos < 0.05 or spread < 0.5:
        print("  VERDICT: post-rescue gains are essentially zero across arms.")
        print("  \u2192 A figure would be four flat bars at ~0. Report as ONE SENTENCE instead:")
        print("    e.g. 'Across all arms, fewer than X% of rescue events produced any")
        print("    incumbent improvement, and no arm had a mean post-rescue gain")
        print("    distinguishable from zero (Table N) \u2014 no injection strategy")
        print("    actually rescues on this benchmark.'")
    else:
        print("  VERDICT: there is meaningful spread \u2014 a figure is justified.")
        print("  \u2192 Re-run with --figure to draw it.")
    print("=" * 66)
    return rows


def make_figure(rows, k, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    arms = [r[0] for r in rows]
    means = [r[2] for r in rows]
    los = [r[2] - r[3] for r in rows]
    his = [r[4] - r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(arms, means, color=[ARM_COLORS.get(a, '#1f4e79') for a in arms],
                  width=0.62, edgecolor='white', zorder=2)
    ax.errorbar(range(len(arms)), means, yerr=[los, his], fmt='none',
                ecolor='#222222', elinewidth=1.3, capsize=4, zorder=3)
    ax.axhline(0, ls='-', lw=0.8, color='#888888')
    ax.set_ylabel(f'mean incumbent gain,\n{k} iters after rescue (GPa)')
    top = max(r[4] for r in rows)
    ax.set_ylim(min(0, min(r[3] for r in rows)) - 0.6, top * 1.35)
    ax.set_title('Post-rescue incumbent gain by strategy\n'
                 '(95% percentile-bootstrap CI)', fontsize=9.5, pad=8)
    # '% improved' and n label centred inside the upper part of each bar
    for i, (r, b) in enumerate(zip(rows, bars)):
        ax.text(i, b.get_height() * 0.5,
                f'{100*r[5]:.0f}% improved\nn={r[1]}',
                ha='center', va='center', fontsize=7.0, color='white', weight='bold')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outdir}/figA_rescue_effectiveness.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote figA_rescue_effectiveness.pdf / .png in {outdir}/")


def main():
    args = sys.argv[1:]
    results_dir = next((a for a in args if not a.startswith('--')), 'results/mp_shear_v1')
    k = 5
    if '--k' in args:
        k = int(args[args.index('--k') + 1])
    want_fig = '--figure' in args
    outdir = 'figures'; os.makedirs(outdir, exist_ok=True)

    for kk in ([k] if want_fig else sorted({3, 5, k})):
        per_arm, n_files = collect(results_dir, kk)
        if not per_arm:
            print(f"  no usable events found in {results_dir} "
                  f"(need inject_events + best_history in run JSONs)")
            return
        print(f"\n[{n_files} run files, k={kk}]")
        rows = summarize(per_arm, kk)
        if want_fig and kk == k:
            make_figure(rows, kk, outdir)


if __name__ == '__main__':
    main()
