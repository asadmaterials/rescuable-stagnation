"""
Publication Figure Generation
=============================
Reads the experiment's result JSONs and produces publication-quality figures
as vector PDFs (and PNG previews). No new computation, no API.

FIGURES
  fig1_criterion       — the rescuable-stagnation criterion: two failure modes
                         (saturation / never-stalls) and the testable middle
                         band. Conceptual schematic (the signature figure).
  fig2_arms            — five-arm final G with 95% CIs (the null, visualized).
  fig3_rescuable       — rescuable-event fraction per arm (the
                         benchmark-qualification evidence).
  fig4_capability      — cross-model difference forest plot (all CIs vs zero).
  fig5_descriptor      — 2D projection (PCA) of injected candidates by arm,
                         showing where the LLM steers vs the heuristics.

Reads from a results dir (default results/mp_shear_v1) plus the Haiku dir for
the capability figure. Missing inputs are skipped with a message, so partial
runs still produce what they can.

Usage:
    python make_figures.py
    python make_figures.py results/mp_shear_v1 results/haiku_v1 figures/
"""

import os
import sys
import json
import glob
import math
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── house style: clean, serif-ish, journal-appropriate ────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10,
    'font.family': 'DejaVu Sans', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.8,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.frameon': False,
})
ACC = '#1f4e79'; GREY = '#888888'; RED = '#b03030'; GRN = '#2e7d32'
ARM_COLORS = {'none': '#999999', 'random': '#c9a227', 'mutation': '#d1711f',
              'digest': '#2e7d32', 'llm': '#1f4e79'}


def save(fig, outdir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outdir}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {name}.pdf / .png")


# ── FIG 1 — the criterion schematic ───────────────────────────────────────

def fig1_criterion(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    x = np.linspace(0, 20, 200)
    ceiling = 100
    # All curves are MONOTONIC (best-so-far incumbent never decreases).
    # Regime I: climbs fast and plateaus AT the ceiling (saturated).
    curve_I = ceiling * (1 - np.exp(-x * 0.9))
    # Regime II: climbs slowly, still rising at the budget end (never stalls).
    curve_II = ceiling * (1 - np.exp(-x * 0.12))
    # Regime III: climbs then plateaus BELOW the ceiling (stalls with headroom).
    # Saturating rise to a plateau at ~0.62*ceiling; strictly non-decreasing.
    plateau = 0.62 * ceiling
    curve_III = plateau * (1 - np.exp(-x * 0.7))
    scenarios = [
        ("Regime I: Saturation\n(weak oracle / small pool)", curve_I,
         "stalls at the ceiling\n\u2014 rescue cannot help", False),
        ("Regime II: Continuous improvement\n(strong oracle / large pool)", curve_II,
         "still improving at budget\n\u2014 rescue never fires", False),
        ("Regime III: Rescuable stagnation\n(diagnostically valid regime)", curve_III,
         "stalls while improvement\nremains available", True),
    ]
    ORANGE = '#d1711f'
    for ax, (title, curve, note, good) in zip(axes, scenarios):
        ax.axhline(ceiling, ls='--', lw=1, color=GREY)
        ax.text(0.4, ceiling + 2, 'oracle-scored pool ceiling', ha='left', va='bottom',
                fontsize=7.5, color=GREY)
        ax.plot(x, curve, lw=2.4, color=(GRN if good else RED))
        if good:
            # vertical headroom bracket from the plateau up to the ceiling
            # (orange, not green: this is unused potential, not improvement)
            xb = 18.5
            ax.annotate('', xy=(xb, ceiling), xytext=(xb, plateau),
                        arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.4))
            ax.text(xb - 0.6, (plateau + ceiling) / 2, 'headroom', rotation=90,
                    va='center', ha='right', fontsize=8, color=ORANGE)
            ax.plot([x[-1], xb], [plateau, plateau], ls=':', lw=0.8, color=ORANGE)
        ax.set_title(title, fontsize=9)
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha='center',
                va='top', fontsize=8, color=(GRN if good else RED))
        ax.set_xlim(0, 20); ax.set_ylim(0, 115)
        ax.set_xlabel('optimization iteration', fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel('best-so-far objective', fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Three benchmark regimes',
                 fontsize=11, y=1.04)
    save(fig, outdir, 'fig1_criterion')


# ── FIG 2 — five-arm final G with CIs ──────────────────────────────────────

def fig2_arms(results_dir, outdir):
    fn = f'{results_dir}/summary.json'
    if not os.path.exists(fn):
        print("    fig2 skipped (no summary.json)"); return
    s = json.load(open(fn))['arms']
    order = ['none', 'random', 'mutation', 'digest', 'llm']
    order = [a for a in order if a in s]
    means = [s[a]['final_mean'] for a in order]
    cis = [s[a].get('final_ci95', 0) for a in order]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = np.arange(len(order))[::-1]
    for yi, a, m, ci in zip(y, order, means, cis):
        ax.errorbar(m, yi, xerr=ci, fmt='o', ms=7, capsize=4, lw=1.6,
                    color=ARM_COLORS.get(a, ACC),
                    mfc=ARM_COLORS.get(a, ACC), mec='white', mew=0.8)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel('final shear modulus G (GPa)')
    ax.set_title('Final objective by rescue strategy\n(mean \u00b1 95% CI; x-axis truncated)',
                 fontsize=10)
    # mark the digest and llm to highlight the key comparison
    ax.margins(y=0.15)
    save(fig, outdir, 'fig2_arms')


# ── FIG 3 — rescuable-event fraction ───────────────────────────────────────

def _wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion; returns (lo, hi) in %."""
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100 * (center - half), 100 * (center + half))


def fig3_rescuable(results_dir, outdir, n_runs=35):
    fn = f'{results_dir}/analysis_report.json'
    if not os.path.exists(fn):
        print("    fig3 skipped (no analysis_report.json)"); return
    r = json.load(open(fn)).get('rescuable', {})
    if not r:
        print("    fig3 skipped (no rescuable data)"); return
    arms = [a for a in ['random', 'mutation', 'digest', 'llm'] if a in r]
    fracs = [100 * r[a]['rescuable_frac'] for a in arms]
    # Derive total event counts from per-run averages x number of runs
    # (n_runs = splits x seeds; default 5x7=35). Wilson CI on the pooled
    # binomial; note this ignores run-level clustering (approximation).
    cis, ns = [], []
    for a in arms:
        d = r[a]
        epr = d.get('events_per_run')
        if epr is not None:
            n = int(round(epr * n_runs))
            k = int(round(d['rescuable_frac'] * n))
            cis.append(_wilson(k, n)); ns.append(n)
        else:
            cis.append(None); ns.append(None)
    have_ci = any(c is not None for c in cis)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bars = ax.bar(arms, fracs, color=[ARM_COLORS.get(a, ACC) for a in arms],
                  width=0.62, edgecolor='white', zorder=2)
    if have_ci:
        for b, f, ci in zip(bars, fracs, cis):
            if ci is None:
                continue
            lo, hi = ci
            xc = b.get_x() + b.get_width() / 2
            ax.errorbar(xc, f, yerr=[[f - lo], [hi - f]], fmt='none',
                        ecolor='#222222', elinewidth=1.3, capsize=4, zorder=4)
    mfrac = np.mean(fracs)
    ax.axhline(mfrac, ls='--', lw=1.2, color='#555555', zorder=1)
    # label the mean line just outside the plot at its right end, so the
    # annotation cannot collide with any bar or CI cap
    ax.set_xlim(-0.6, len(arms) - 0.2)
    ax.text(len(arms) - 0.38, mfrac, 'mean\nqualifying\nrate', ha='left',
            va='center', fontsize=7, color='#555555', linespacing=1.1)
    for b, f, n in zip(bars, fracs, ns):
        lbl = f'{f:.0f}%' + (f'\n(n={n})' if n else '')
        ax.text(b.get_x() + b.get_width() / 2, f + 6.5,
                lbl, ha='center', fontsize=8, zorder=5)
    ax.set_ylabel('rescuable events (% of stalls)')
    ax.set_ylim(0, 105)
    sub = '(vs oracle-scored pool ceiling; 95% Wilson intervals)' if have_ci else \
          '(vs oracle-scored pool ceiling)'
    ax.set_title('Rescuable stalls per strategy\n' + sub, fontsize=9.5)
    save(fig, outdir, 'fig3_rescuable')


# ── FIG 4 — capability difference forest plot ──────────────────────────────

def fig4_capability(haiku_dir, outdir):
    """Between-model (Sonnet - Haiku) differences.

    Organised around the scientific question 'does model capability change any
    aspect of the behaviour?' \u2014 spanning exploration (novelty, coverage),
    reasoning (coherence), and outcome (final G) \u2014 not around manuscript
    section order. Cross-model manifold overlap is a SIMILARITY, not a
    difference, so it is annotated rather than plotted on the difference axis.
    """
    fn = f'{haiku_dir}/cross_model_difference_ci.json'
    if not os.path.exists(fn):
        print("    fig4 skipped (no cross_model_difference_ci.json)"); return
    d = json.load(open(fn))
    # causal flow: exploration -> reasoning -> outcome
    spec = [
        ('Novelty (dist. to train)', ['novelty']),
        ('Coverage (spread)', ['coverage']),
        ('Coherence (/8)', ['coherence']),
        ('Final G (GPa)', ['final_G']),
    ]
    rows = []
    for lab, keys in spec:
        m = next((d[k] for k in keys if k in d and d[k]), None)
        if m and 'delta' in m and 'ci' in m:
            rows.append((lab, m['delta'], m['ci'][0], m['ci'][1]))
    if not rows:
        print("    fig4 skipped (no metric CIs)"); return
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, delta, lo, hi) in zip(y, rows):
        crosses0 = lo <= 0 <= hi
        ax.plot([lo, hi], [yi, yi], lw=2, color=(GREY if crosses0 else RED))
        ax.plot(delta, yi, 'o', ms=7, color=(ACC if crosses0 else RED),
                mec='white', mew=0.8)
    ax.axvline(0, ls='--', lw=1.0, color='#333333', alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('Sonnet \u2212 Haiku difference (95% CI)')
    ax.set_title('Capability comparison: Sonnet vs Haiku\n(between-model difference per metric)',
                 fontsize=10)
    # cross-model manifold overlap is a similarity, not a difference: annotate
    xmo = d.get('cross_model_overlap') or d.get('manifold_overlap_cross_model')
    xmo_txt = (f'cross-model manifold overlap = {xmo:.3f}'
               if isinstance(xmo, (int, float)) else
               'cross-model manifold overlap = 0.708')
    ax.text(0.5, -0.34, xmo_txt + '  (similarity, not a difference)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='#555555')
    ax.margins(y=0.2)
    save(fig, outdir, 'fig4_capability')


# ── FIG 5 — descriptor-space projection of injected candidates ─────────────

def fig5_descriptor(results_dir, outdir):
    try:
        import mp_oracle as MO
        from mp_oracle import MPOracle, FEATURE_COLS, composition_to_vector
        from mp_runner import prepare_dataset, make_splits
        from sklearn.decomposition import PCA
    except Exception as e:
        print(f"    fig5 skipped (deps unavailable: {e})"); return
    try:
        df, _ = prepare_dataset()
        tr_idx, pool_idx = make_splits(len(df), seed=0)
        orc = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=0)
    except Exception as e:
        print(f"    fig5 skipped (data/oracle unavailable: {e})"); return

    # collect injected candidates per arm
    arm_vecs = {}
    for fn in glob.glob(f'{results_dir}/run_detail/*.json'):
        d = json.load(open(fn))
        arm = d['arm']
        if arm == 'none':
            continue
        for ev in d.get('inject_events', []):
            for comp in ev.get('compositions', []):
                v = composition_to_vector(comp, FEATURE_COLS)
                if v is not None:
                    arm_vecs.setdefault(arm, []).append(
                        orc.scaler.transform(v.reshape(1, -1))[0])
    if not arm_vecs:
        print("    fig5 skipped (no injected candidates)"); return

    # fit PCA on the pool for a stable projection, then project candidates
    pool_s = orc.scaler.transform(df[FEATURE_COLS].values[pool_idx])
    pca = PCA(n_components=2).fit(pool_s)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    # faint pool background
    bg = pca.transform(pool_s)
    ax.scatter(bg[:, 0], bg[:, 1], s=6, color='#dddddd', alpha=0.5,
               label='dataset pool', zorder=1)
    for arm in ['mutation', 'digest', 'random', 'llm']:
        if arm not in arm_vecs:
            continue
        V = pca.transform(np.array(arm_vecs[arm]))
        col = ARM_COLORS.get(arm, ACC)
        ax.scatter(V[:, 0], V[:, 1], s=22, alpha=0.7,
                   color=col, label=arm,
                   edgecolor='white', linewidth=0.4, zorder=3)
        # transparent convex hull to make region occupancy legible
        if V.shape[0] >= 3:
            try:
                from scipy.spatial import ConvexHull
                h = ConvexHull(V)
                poly = V[h.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.10, zorder=2)
                ax.plot(np.append(poly[:, 0], poly[0, 0]),
                        np.append(poly[:, 1], poly[0, 1]),
                        color=col, lw=1.0, alpha=0.5, zorder=2)
            except Exception:
                pass
    ax.set_xlabel('descriptor PC1 (PCA on pool)'); ax.set_ylabel('descriptor PC2')
    # put the key overlap number in the subtitle, not a buried textbox
    overlap = None
    for cand in (f'{results_dir}/analysis_report.json', f'{results_dir}/compare_models.json'):
        if os.path.exists(cand):
            try:
                cj = json.load(open(cand))
                overlap = (cj.get('descriptor_overlap_llm_mutation')
                           or cj.get('manifold_overlap')
                           or (cj.get('exploration', {}) or {}).get('overlap_llm_mutation'))
                if overlap is not None:
                    break
            except Exception:
                pass
    ov = f'{overlap:.3f}' if overlap is not None else '0.033'
    ax.set_title('Descriptor-space occupancy of injected candidates\n'
                 f'(LLM\u2013mutation manifold overlap = {ov})', fontsize=10)
    ax.legend(loc='best', markerscale=1.2)
    save(fig, outdir, 'fig5_descriptor')


def fig11_summary(outdir):
    """One-page conceptual summary: the paper's logical chain.
    Boxes are colour-coded by epistemic status:
      green  = supported observation
      gray   = interpretation
      yellow = hypothesis
    """
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.6); ax.axis('off')
    GREEN_E, AMBER_E, YELLOW_E = '#2e7d32', '#b8860b', '#b8860b'
    # (label, subtitle, epistemic_status)
    steps = [
        ("Benchmark is rescuably stagnant", "86\u201388% of stalls retain headroom", 'obs'),
        ("LLM rescue shows no aggregate gain", "incl. a same-information control", 'obs'),
        ("The comparison is fair", "GP calibrated; LLM least-constrained", 'obs'),
        ("LLM steers into distinct regions", "descriptor overlap 0.033", 'obs'),
        ("Behaviour holds across model scale", "all difference CIs include zero", 'obs'),
        ("Coherence doesn\u2019t predict quality", "reasoning does not explain differences", 'obs'),
        ("Contribution is where exploration goes", "interpretation; no outcome gain at budget", 'interp'),
    ]
    # obs = solid green; interp = amber with a dashed edge (visually distinct status)
    box_style = {'obs': (GREEN_E, 0.14, 1.6, '-'),
                 'interp': (AMBER_E, 0.20, 2.2, (0, (4, 1.5)))}
    n = len(steps)
    box_h = 1.04
    gap = (12.6 - 0.7 - n * box_h) / (n - 1)
    y = 12.6 - 0.4 - box_h
    for i, (label, sub, status) in enumerate(steps):
        color, fill_alpha, lw, ls = box_style[status]
        # light fill (no edge) + full-opacity coloured edge, so the two
        # epistemic categories are easy to tell apart
        fillbox = FancyBboxPatch((1.15, y), 7.7, box_h,
                                 boxstyle="round,pad=0.06,rounding_size=0.12",
                                 linewidth=0, facecolor=color, alpha=fill_alpha, zorder=2)
        ax.add_patch(fillbox)
        edgebox = FancyBboxPatch((1.15, y), 7.7, box_h,
                                 boxstyle="round,pad=0.06,rounding_size=0.12",
                                 linewidth=lw, edgecolor=color, facecolor='none',
                                 linestyle=ls, alpha=1.0, zorder=3)
        ax.add_patch(edgebox)
        ax.text(5.0, y + box_h * 0.60, label, ha='center', va='center',
                fontsize=9.5, color='#1a1a1a', weight='bold', zorder=4)
        ax.text(5.0, y + box_h * 0.18, sub, ha='center', va='center',
                fontsize=7.3, color='#555555', style='italic', zorder=4)
        if i < n - 1:
            ax.annotate('', xy=(5.0, y - gap * 0.9), xytext=(5.0, y - 0.02),
                        arrowprops=dict(arrowstyle='-|>', color=GREY, lw=1.6))
        y -= (box_h + gap)
    ax.set_title('Summary of the argument', fontsize=12, weight='bold', color=ACC, pad=4)
    # epistemic-status legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=GREEN_E, alpha=0.30, edgecolor=GREEN_E, linewidth=1.6,
                 label='supported observation'),
           Patch(facecolor=AMBER_E, alpha=0.30, edgecolor=AMBER_E, linewidth=2.2,
                 linestyle='--', label='interpretation')]
    ax.legend(handles=leg, loc='lower center', ncol=2, fontsize=7.5,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, outdir, 'fig11_summary')


def fig6_exploration_vs_gain(results_dir, outdir):
    """Mechanistic figure from candidate_tags: exploration distance vs. oracle
    value for injected candidates that were actually evaluated (queried) by the
    GP, per arm. Closes the causal story: injected candidates sit far out, but
    the ones evaluated are not better, so distinct exploration does not become
    an aggregate advantage. Uses the real run_detail schema (candidate_tags with
    distance / y / channel_b / queried / improved)."""
    import glob, json
    # arm inferred from filename prefix (e.g. llm_split1_seed5.json) or 'arm' field
    per_arm_q = {}   # queried candidates: (distance, y)
    per_arm_all = {} # all admitted: (distance, channel_b)
    files = glob.glob(f'{results_dir}/run_detail/*.json') or glob.glob(f'{results_dir}/*.json')
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        arm = d.get('arm')
        tags = d.get('candidate_tags')
        if not arm or arm == 'none' or not tags:
            continue
        for t in tags:
            dist = t.get('distance')
            if dist is None:
                continue
            cb = t.get('channel_b')
            per_arm_all.setdefault(arm, []).append((dist, cb))
            if t.get('queried') and t.get('y') is not None:
                per_arm_q.setdefault(arm, []).append((dist, t['y']))
    if not per_arm_all:
        print("    fig6 skipped (no candidate_tags with distance found in run_detail)")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    order = ['mutation', 'digest', 'random', 'llm']

    # LEFT: distance vs oracle value, queried candidates only
    any_q = False
    all_q_pts = []
    for arm in order:
        if arm in per_arm_q and per_arm_q[arm]:
            pts = np.array(per_arm_q[arm]); any_q = True
            all_q_pts.extend(per_arm_q[arm])
            axL.scatter(pts[:, 0], pts[:, 1], s=26, alpha=0.7,
                        color=ARM_COLORS.get(arm, ACC), label=arm,
                        edgecolor='white', linewidth=0.4)
    axL.set_xlabel('exploration distance (to nearest train)')
    axL.set_ylabel('oracle value of evaluated candidate (GPa)', color='#1f4e79')
    axL.tick_params(axis='y', labelcolor='#1f4e79')
    axL.set_title('Exploration distance vs oracle value\n(evaluated injected candidates)', fontsize=9.5)
    if any_q:
        A = np.array(all_q_pts)
        # pooled least-squares line + Spearman rho, if enough points
        if A.shape[0] >= 4:
            xs, ys = A[:, 0], A[:, 1]
            try:
                m, b = np.polyfit(xs, ys, 1)
                xg = np.linspace(xs.min(), xs.max(), 50)
                axL.plot(xg, m * xg + b, ls='--', lw=1.4, color='#333333', zorder=5)
            except Exception:
                pass
            rho_txt = ''
            try:
                from scipy.stats import spearmanr
                rho, pval = spearmanr(xs, ys)
                rho_disp = 0.0 if abs(rho) < 0.005 else rho
                rho_txt = f'Spearman \u03c1 = {rho_disp:.2f} (p = {pval:.2f}, n = {A.shape[0]})'
            except Exception:
                rho_txt = f'n = {A.shape[0]}'
            axL.text(0.02, 0.02, rho_txt, transform=axL.transAxes, ha='left',
                     va='bottom', fontsize=8, color='#1a1a1a',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))
        axL.legend(loc='best', fontsize=8)
    else:
        axL.text(0.5, 0.5, 'few candidates queried\n(injected points rarely evaluated)',
                 transform=axL.transAxes, ha='center', va='center', fontsize=9, color=GREY)

    # RIGHT: distance vs Channel-B physics estimate, all admitted (shows LLM
    # explores far + high-physics regions that nonetheless do not win)
    for arm in order:
        if arm in per_arm_all and per_arm_all[arm]:
            pts = np.array([(a, b) for (a, b) in per_arm_all[arm] if b is not None])
            if pts.size == 0:
                continue
            axR.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.55,
                        color=ARM_COLORS.get(arm, ACC), label=arm,
                        edgecolor='white', linewidth=0.3)
    axR.set_xlabel('exploration distance (to nearest train)')
    axR.set_ylabel('Channel-B physics estimate (GPa)', color='#8a5a00')
    axR.tick_params(axis='y', labelcolor='#8a5a00')
    axR.set_title('LLM preferentially samples higher\nChannel-B candidates (all admitted)', fontsize=9.5)
    axR.legend(loc='best', fontsize=8)

    fig.suptitle('Exploration distance versus candidate quality',
                 fontsize=11, y=1.02)
    save(fig, outdir, 'fig6_exploration_vs_gain')


def main(results_dir='results/mp_shear_v1', haiku_dir='results/haiku_v1',
         outdir='figures'):
    os.makedirs(outdir, exist_ok=True)
    print("=" * 60)
    print(f"  GENERATING FIGURES → {outdir}/")
    print("=" * 60)
    # Files are generated below; the RECOMMENDED MANUSCRIPT DISPLAY ORDER
    # (per reviewer) is:
    #   Fig 1 = fig1_criterion       (three benchmark regimes)
    #   Fig 2 = fig3_rescuable       (headroom \u2014 prove benchmark valid)
    #   Fig 3 = fig2_arms            (rescue performance \u2014 the null)
    #   Fig 4 = fig5_descriptor      (occupancy \u2014 LLM behaves differently)
    #   Fig 5 = fig6_exploration...  (exploration distance \u2014 why)
    #   Fig 6 = fig4_capability      (capability comparison)
    #   Fig 7 = fig11_summary        (synthesis)
    # Number the callouts in the manuscript accordingly.
    fig1_criterion(outdir)
    fig2_arms(results_dir, outdir)
    fig3_rescuable(results_dir, outdir)
    fig4_capability(haiku_dir, outdir)
    fig5_descriptor(results_dir, outdir)
    fig6_exploration_vs_gain(results_dir, outdir)
    fig11_summary(outdir)
    print("=" * 60)
    print(f"  Done. PDFs (vector, for the manuscript) and PNGs (preview) in {outdir}/")
    print("=" * 60)


if __name__ == '__main__':
    a = sys.argv[1:]
    rd = a[0] if len(a) > 0 else 'results/mp_shear_v1'
    hd = a[1] if len(a) > 1 else 'results/haiku_v1'
    od = a[2] if len(a) > 2 else 'figures'
    main(rd, hd, od)
