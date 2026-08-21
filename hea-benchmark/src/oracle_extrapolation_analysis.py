"""
Oracle-Extrapolation Analysis (post-run)
========================================
Tests the central interpretive claim of the experiment:

    The RF oracle systematically UNDER-REWARDS compositions far from its
    training data, so any arm that explores genuinely novel chemistry is
    penalised by construction — independent of proposal quality.

Evidence assembled here, all from data already on disk (NO rerun needed):

  (1) For every composition the LLM proposed (from llm_traces/), compute
      BOTH channels:
        Channel A — the RF oracle's HV prediction (what the experiment scored)
        Channel B — the Toda-Caraballo SSH physics proxy (never fitted to
                    the data, extrapolates by construction)
      plus the distance-to-training extrapolation indicator.

  (2) Compare against the arm that WON (mutation): its candidates are
      reconstructed from run_detail/ candidate tags.

  (3) Test the key relationship: does RF-predicted HV fall as
      distance-from-training rises, while the physics proxy does not?
      If yes, the RF is regressing to the mean on novel chemistry — the
      benchmark cannot reward exploration.

IMPORTANT SCOPE / HONESTY NOTES
  - Channel B returns None for alloys containing interstitials (B, C):
    they are out-of-model for a substitutional SSH theory. Those
    compositions are reported separately, NOT silently dropped — a large
    fraction of LLM proposals contain B, and excluding them without
    saying so would bias the comparison.
  - Channel B is a RANKING quantity (the Z constant cancels). It is NOT
    calibrated hardness. Never compare its absolute value to HV.
  - This analysis CANNOT prove the LLM's alloys are actually harder. It
    can only show the two channels disagree in a direction that matches
    the extrapolation hypothesis. Real validation needs experiment/DFT.

Usage (from src/):
    python oracle_extrapolation_analysis.py
    python oracle_extrapolation_analysis.py ../results/four_arm_v2
"""

import os
import sys
import json
import glob
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from canonical_oracle import (
    load_working_dataset, make_splits, CanonicalOracle,
    get_feature_cols, get_composition_cols,
)
from ss_strengthening import compute_ss_proxy, INTERSTITIALS
from data_pipeline import compute_descriptors


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def comp_to_vector(comp, feature_cols, comp_cols):
    """Composition dict -> model feature vector (compositions + descriptors)."""
    total = sum(comp.values())
    if total <= 0:
        return None
    comp = {k: v / total for k, v in comp.items()}
    try:
        desc = compute_descriptors(comp)
    except Exception:
        return None
    vec = np.zeros(len(feature_cols))
    for i, c in enumerate(feature_cols):
        if c in comp_cols:
            vec[i] = comp.get(c, 0.0)
        else:
            v = desc.get(c, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            vec[i] = v
    # match the working dataset's omega clip convention
    if 'omega' in feature_cols:
        j = feature_cols.index('omega')
        vec[j] = min(vec[j], 50.0)
    return vec


def load_llm_proposals(save_dir):
    """Every composition the LLM proposed, across all traces."""
    props = []
    for fn in sorted(glob.glob(f'{save_dir}/llm_traces/*.json')):
        try:
            events = json.load(open(fn, encoding='utf-8'))
        except Exception as e:
            print(f"  ! could not read {os.path.basename(fn)}: {e}")
            continue
        for ev in events:
            for comp in ev.get('raw_candidates', []):
                if isinstance(comp, dict) and comp:
                    props.append({
                        'comp': comp,
                        'iteration': ev.get('iteration'),
                        'source': os.path.basename(fn),
                    })
    return props


def load_arm_injected_vectors(save_dir, arm):
    """Feature vectors of candidates actually injected by a given arm."""
    vecs = []
    for fn in sorted(glob.glob(f'{save_dir}/run_detail/{arm}_*.json')):
        try:
            d = json.load(open(fn, encoding='utf-8'))
        except Exception:
            continue
        for ev in d.get('inject_events', []):
            for v in ev.get('vectors', []):
                vecs.append(np.array(v, dtype=float))
    return vecs


def spearman(a, b):
    """Spearman rho without scipy (rank-transform then Pearson)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else float('nan')


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main(save_dir='../results/four_arm_v2'):
    print("=" * 70)
    print("  ORACLE-EXTRAPOLATION ANALYSIS")
    print("  Does the RF penalise novel chemistry that the physics rates highly?")
    print("=" * 70)

    df = load_working_dataset()
    fc = get_feature_cols(df)
    cc = get_composition_cols(fc)

    # Oracle from split 0 (the same construction the run used)
    splits = make_splits(df, random_seed=0)
    oracle = CanonicalOracle(splits['train'], fc)

    # ── 1. LLM proposals: both channels ───────────────────────────────────
    props = load_llm_proposals(save_dir)
    print(f"\n  LLM proposals found: {len(props)}")
    if not props:
        print("  ! No traces found — check the path.")
        return

    rows, n_interstitial, n_bad = [], 0, 0
    for p in props:
        comp = p['comp']
        vec  = comp_to_vector(comp, fc, cc)
        if vec is None:
            n_bad += 1
            continue
        rf   = oracle.query(vec)
        dist = oracle.confidence(vec)          # distance-based extrapolation indicator
        ss   = compute_ss_proxy(comp)          # None if B/C present
        if ss is None:
            n_interstitial += 1
        rows.append({'comp': comp, 'rf': rf, 'dist': dist, 'ss': ss})

    print(f"    unusable (descriptor failure): {n_bad}")
    print(f"    out-of-model for physics (contain B/C): {n_interstitial}"
          f"  [{100*n_interstitial/max(len(rows),1):.0f}% of scored]")

    ok = [r for r in rows if r['ss'] is not None]
    print(f"    scored by BOTH channels: {len(ok)}")

    # ── 2. Reference: what mutation actually injected ─────────────────────
    mut = load_arm_injected_vectors(save_dir, 'mutation')
    if mut:
        mut_rf   = np.array([oracle.query(v) for v in mut])
        mut_dist = np.array([oracle.confidence(v) for v in mut])
    else:
        mut_rf = mut_dist = np.array([])

    llm_rf   = np.array([r['rf'] for r in rows])
    llm_dist = np.array([r['dist'] for r in rows])

    print("\n" + "-" * 70)
    print("  CHANNEL A (RF oracle) — what the experiment actually scored")
    print("-" * 70)
    print(f"    {'source':<22}{'n':>6}{'mean HV':>10}{'max HV':>9}{'mean dist':>11}")
    print(f"    {'LLM proposals':<22}{len(llm_rf):>6}{llm_rf.mean():>10.1f}"
          f"{llm_rf.max():>9.1f}{llm_dist.mean():>11.2f}")
    if len(mut_rf):
        print(f"    {'mutation injected':<22}{len(mut_rf):>6}{mut_rf.mean():>10.1f}"
              f"{mut_rf.max():>9.1f}{mut_dist.mean():>11.2f}")
    train_hv = splits['train']['HV'].values
    print(f"    {'(training set)':<22}{len(train_hv):>6}{train_hv.mean():>10.1f}"
          f"{train_hv.max():>9.1f}{0.0:>11.2f}")

    # ── 3. THE KEY TEST ───────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  THE KEY TEST: does RF prediction decay with distance from training,")
    print("  while the physics proxy does not?")
    print("-" * 70)

    rho_rf_dist = spearman(llm_dist, llm_rf)
    print(f"    Spearman(distance, RF-predicted HV)      = {rho_rf_dist:+.3f}")
    if len(ok) >= 5:
        ss_v   = np.array([r['ss']   for r in ok])
        d_v    = np.array([r['dist'] for r in ok])
        rf_v   = np.array([r['rf']   for r in ok])
        rho_ss_dist = spearman(d_v, ss_v)
        rho_rf_ss   = spearman(rf_v, ss_v)
        print(f"    Spearman(distance, physics SS proxy)     = {rho_ss_dist:+.3f}")
        print(f"    Spearman(RF HV,   physics SS proxy)      = {rho_rf_ss:+.3f}")
        print()
        print("    Interpretation:")
        if rho_rf_dist < -0.2 and rho_ss_dist > rho_rf_dist + 0.2:
            print("      → RF predictions FALL as candidates get further from training,")
            print("        while the physics proxy does not follow that pattern.")
            print("        Consistent with the RF regressing toward the mean on novel")
            print("        chemistry: the benchmark penalises exploration by construction.")
        elif rho_rf_dist < -0.2:
            print("      → RF predictions fall with distance, but the physics proxy")
            print("        moves similarly. Cannot cleanly separate 'oracle blindness'")
            print("        from 'these alloys are genuinely unremarkable'.")
        else:
            print("      → No strong distance/RF decay in these proposals. The")
            print("        extrapolation-penalty story is NOT supported; the null")
            print("        should be read at face value.")

    # ── 4. Physics-top vs RF-top: do the channels disagree on WHICH? ──────
    if len(ok) >= 10:
        print("\n" + "-" * 70)
        print("  DISAGREEMENT: top candidates by each channel")
        print("-" * 70)
        by_ss = sorted(ok, key=lambda r: -r['ss'])[:5]
        by_rf = sorted(ok, key=lambda r: -r['rf'])[:5]

        def fmt(c):
            return " ".join(f"{k}{v:.2f}" for k, v in
                            sorted(c.items(), key=lambda kv: -kv[1]))

        print("    Top 5 by PHYSICS (SSH proxy):")
        for r in by_ss:
            print(f"      ss={r['ss']:8.1f}  rf_HV={r['rf']:6.1f}  "
                  f"dist={r['dist']:5.2f}  {fmt(r['comp'])[:52]}")
        print("    Top 5 by RF ORACLE:")
        for r in by_rf:
            print(f"      rf_HV={r['rf']:6.1f}  ss={r['ss']:8.1f}  "
                  f"dist={r['dist']:5.2f}  {fmt(r['comp'])[:52]}")

        ss_top_rf = np.mean([r['rf'] for r in by_ss])
        print(f"\n    Physics-favoured candidates score {ss_top_rf:.1f} HV on the RF")
        print(f"    (training-set mean is {train_hv.mean():.1f} HV) — if these sit near")
        print(f"    the training mean despite high physics scores, that is the")
        print(f"    regression-to-the-mean signature.")

    # ── 5. Save ───────────────────────────────────────────────────────────
    out = {
        'n_llm_proposals'        : len(props),
        'n_scored'               : len(rows),
        'n_interstitial_excluded': n_interstitial,
        'n_both_channels'        : len(ok),
        'llm_rf_mean'            : float(llm_rf.mean()),
        'llm_dist_mean'          : float(llm_dist.mean()),
        'mutation_rf_mean'       : float(mut_rf.mean()) if len(mut_rf) else None,
        'mutation_dist_mean'     : float(mut_dist.mean()) if len(mut_dist) else None,
        'train_hv_mean'          : float(train_hv.mean()),
        'spearman_dist_vs_rf'    : rho_rf_dist,
        'caveat'                 : ('Channel B is a ranking proxy, not calibrated HV. '
                                    'Interstitial (B/C) alloys are out-of-model and '
                                    'excluded from physics comparisons but reported. '
                                    'This shows channel DISAGREEMENT, not that the '
                                    'LLM proposals are truly harder.'),
    }
    with open(f'{save_dir}/oracle_extrapolation.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {save_dir}/oracle_extrapolation.json")
    print("=" * 70)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '../results/four_arm_v2')
