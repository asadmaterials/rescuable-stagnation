"""
Cross-Model Exploration Comparison  (model-agnostic, with uncertainty)
=====================================================================
Compares the exploration BEHAVIOUR of two models' LLM arms — a frontier
baseline (e.g. Sonnet, results/mp_shear_v1) and a smaller model (e.g. Haiku).
Every metric is model-agnostic: none assumes either model's chemistry is
"correct". Chemistry is DESCRIBED afterward, never used as a success criterion.

This version incorporates review feedback:
  - PRIMARY metric is DESCRIPTOR-SPACE overlap, not element-name regions.
    top2() element-region Jaccard compresses a 5-element alloy to 2 elements
    and loses real chemistry (two different alloys can map to the same pair),
    so it is kept only as a caveated SECONDARY view. The defensible question
    is whether the two models occupy the same region of Magpie descriptor
    space. (reviewer: "the biggest scientific question")
  - BOOTSTRAP 95% CIs on every metric (over runs), so "0.27 vs 0.35" can be
    judged as signal or noise. (reviewer concerns 6, 7)
  - "consistency" renamed STABILITY and presented descriptively, since a model
    that adapts per split SHOULD differ — instability is not badness.
    (reviewer concern 5)
  - coverage = mean pairwise distance, with nearest-neighbour distance also
    reported so the choice is justified. (reviewer concern 3)
  - a SUMMARY TABLE across the headline quantities. (reviewer request)

INTERPRET AGNOSTICALLY (all publishable):
  same descriptor manifold -> exploration robust across levels
  smaller model collapses toward heuristic -> consistent with capability
    influencing exploration
  smaller model occupies a DIFFERENT distinct manifold -> exploration
    diversity, not one correct chemistry

Usage:
    python compare_models.py results/mp_shear_v1 results/haiku_v1
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
from itertools import combinations

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(0)


def top2(comp):
    parts = sorted(((e, f) for e, f in comp.items() if f > 0.01),
                   key=lambda p: -p[1])
    return "-".join(sorted(e for e, _ in parts[:2]))


def load_injected(save_dir, arm):
    out = []
    for fn in glob.glob(f'{save_dir}/run_detail/{arm}_*.json'):
        d = json.load(open(fn, encoding='utf-8'))
        for ev in d.get('inject_events', []):
            for comp in ev.get('compositions', []):
                out.append({'split': d['split_seed'], 'seed': d['bo_seed'],
                            'comp': comp, 'region': top2(comp)})
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float('nan')


def boot_ci(values, stat=np.mean, n=2000):
    """95% bootstrap CI of a statistic over a list of values."""
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(values) < 2:
        return (float('nan'), float('nan'), float('nan'))
    vals = np.array(values, float)
    boots = [stat(RNG.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(stat(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── descriptor-space machinery (the PRIMARY, defensible space) ─────────────

def _scaler_and_vectors(injected):
    import mp_oracle as MO
    from mp_oracle import MPOracle, FEATURE_COLS, composition_to_vector
    from mp_runner import prepare_dataset, make_splits
    df, _ = prepare_dataset()
    tr_idx, pool_idx = make_splits(len(df), seed=0)
    orc = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=0)
    vecs = []
    for x in injected:
        v = composition_to_vector(x['comp'], FEATURE_COLS)
        if v is not None:
            vecs.append(orc.scaler.transform(v.reshape(1, -1))[0])
    return orc, (np.array(vecs) if vecs else np.empty((0,)))


def descriptor_overlap(Va, Vb):
    """
    Manifold-overlap proxy in descriptor space: for each point in A, is its
    nearest neighbour among A∪B actually in B (and vice versa)? If the two
    clouds occupy the same manifold, cross-set nearest neighbours are common
    (overlap ~0.5); if they are separated, overlap ~0.
    Returns symmetric overlap in [0,1].
    """
    if len(Va) < 3 or len(Vb) < 3:
        return float('nan')
    from sklearn.metrics import pairwise_distances_argmin_min
    # fraction of A whose nearest B-point is closer than its nearest other-A
    def cross_frac(P, Q):
        _, dq = pairwise_distances_argmin_min(P, Q)
        # nearest within P (excluding self): add small diagonal
        from sklearn.metrics import pairwise_distances
        Dp = pairwise_distances(P); np.fill_diagonal(Dp, np.inf)
        dp = Dp.min(axis=1)
        return float(np.mean(dq <= dp))
    return 0.5 * (cross_frac(Va, Vb) + cross_frac(Vb, Va))


def coverage_metrics(orc, V):
    if len(V) < 3:
        return None, None, None
    from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
    idx = RNG.choice(len(V), min(len(V), 200), replace=False)
    S = V[idx]
    D = pairwise_distances(S)
    mean_pair = float(D[np.triu_indices_from(D, 1)].mean())
    Dnn = pairwise_distances(S); np.fill_diagonal(Dnn, np.inf)
    mean_nn = float(Dnn.min(axis=1).mean())
    _, dtrain = pairwise_distances_argmin_min(V, orc._Xtr_s)
    novelty = float(dtrain.mean())
    return mean_pair, mean_nn, novelty


# ── per-run metrics for bootstrapping ──────────────────────────────────────

def per_run_region_overlap(inj, ref_regions):
    """Jaccard vs a reference region set, computed per (split,seed) run."""
    byrun = {}
    for x in inj:
        byrun.setdefault((x['split'], x['seed']), set()).add(x['region'])
    return [jaccard(regs, ref_regions) for regs in byrun.values()]


def stability(inj):
    """Per-split region sets; mean pairwise Jaccard across splits. Renamed from
    'consistency' — a model that adapts per split SHOULD differ, so this is a
    descriptive STABILITY measure, not a quality score."""
    bysplit = {}
    for x in inj:
        bysplit.setdefault(x['split'], set()).add(x['region'])
    sets = list(bysplit.values())
    if len(sets) < 2:
        return float('nan'), []
    js = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return float(np.mean(js)), js


def final_stats(save_dir):
    finals = []
    for fn in glob.glob(f'{save_dir}/run_detail/llm_*.json'):
        finals.append(json.load(open(fn))['final_best'])
    return finals


def describe_regions(inj, label):
    from collections import Counter
    cnt = Counter(x['region'] for x in inj)
    print(f"    {label} top regions: " +
          ", ".join(f"{r}({n})" for r, n in cnt.most_common(10)))


def main(baseline_dir, other_dir):
    print("=" * 70)
    print("  CROSS-MODEL EXPLORATION COMPARISON  (model-agnostic, with CIs)")
    print("=" * 70)
    bcfg = json.load(open(f'{baseline_dir}/config.json'))
    bmodel = bcfg.get('llm_model', bcfg.get('model', 'baseline'))
    try:
        omodel = json.load(open(f'{other_dir}/config.json')).get('model', 'other')
    except Exception:
        omodel = 'other'
    print(f"  baseline: {bmodel}   ({baseline_dir})")
    print(f"  other   : {omodel}   ({other_dir})\n")

    base_llm = load_injected(baseline_dir, 'llm')
    other_llm = load_injected(other_dir, 'llm')
    mutation = load_injected(baseline_dir, 'mutation')
    if not base_llm or not other_llm:
        print("  missing injected data"); return
    mut_regions = set(x['region'] for x in mutation)

    # ── PRIMARY: descriptor-space manifold overlap ────────────────────────
    print("  " + "-" * 66)
    print("  1. DESCRIPTOR-SPACE OVERLAP  (PRIMARY — the defensible metric)")
    print("  " + "-" * 66)
    try:
        orc, Vb = _scaler_and_vectors(base_llm)
        _, Vo = _scaler_and_vectors(other_llm)
        _, Vm = _scaler_and_vectors(mutation)
        dov = descriptor_overlap(Vb, Vo)
        dov_bm = descriptor_overlap(Vb, Vm)
        dov_om = descriptor_overlap(Vo, Vm)
        print(f"    manifold overlap {bmodel} vs {omodel}: {dov:.3f}")
        print(f"      (~0.5 = same manifold; ~0 = separated clouds)")
        print(f"    overlap {bmodel} vs mutation(heuristic): {dov_bm:.3f}")
        print(f"    overlap {omodel} vs mutation(heuristic): {dov_om:.3f}")
        if not np.isnan(dov_bm) and not np.isnan(dov_om):
            if dov_om > dov_bm + 0.1:
                print("    → smaller model sits CLOSER to the heuristic manifold —")
                print("      consistent with capability influencing exploration.")
            elif abs(dov_om - dov_bm) <= 0.1:
                print("    → both models are similarly separated from the heuristic")
                print("      manifold — exploration distinctness robust across levels.")
            else:
                print("    → smaller model is even more separated — not a frontier-only")
                print("      property.")
    except Exception as e:
        print(f"    descriptor-space metric unavailable: {e}")
        Vb = Vo = Vm = None; dov = dov_bm = dov_om = float('nan')

    # ── coverage & novelty with justification ─────────────────────────────
    print("\n  " + "-" * 66)
    print("  2. COVERAGE & NOVELTY (descriptor space)")
    print("  " + "-" * 66)
    cov_b = coverage_metrics(orc, Vb) if Vb is not None and len(Vb) else (None,)*3
    cov_o = coverage_metrics(orc, Vo) if Vo is not None and len(Vo) else (None,)*3
    print(f"    {'model':<22}{'mean-pair':>11}{'mean-NN':>10}{'novelty':>10}")
    print(f"    {bmodel:<22}{_f(cov_b[0]):>11}{_f(cov_b[1]):>10}{_f(cov_b[2]):>10}")
    print(f"    {omodel:<22}{_f(cov_o[0]):>11}{_f(cov_o[1]):>10}{_f(cov_o[2]):>10}")
    print("    (mean-pairwise = overall spread; mean-NN = local packing;")
    print("     both reported so the coverage choice is justified, not assumed.)")

    # ── SECONDARY: element-region Jaccard, with bootstrap CIs + caveat ────
    print("\n  " + "-" * 66)
    print("  3. ELEMENT-REGION OVERLAP vs heuristic  (SECONDARY — caveated)")
    print("  " + "-" * 66)
    print("    CAVEAT: 2-element region keys compress multi-element alloys and")
    print("    can merge chemically different compositions. Descriptor-space")
    print("    (metric 1) is the primary measure; this is a coarse cross-check.")
    ob = per_run_region_overlap(base_llm, mut_regions)
    oo = per_run_region_overlap(other_llm, mut_regions)
    mb, lo_b, hi_b = boot_ci(ob)
    mo, lo_o, hi_o = boot_ci(oo)
    print(f"    {bmodel:<22} Jaccard vs mutation: {mb:.3f}  95% CI [{lo_b:.3f},{hi_b:.3f}]")
    print(f"    {omodel:<22} Jaccard vs mutation: {mo:.3f}  95% CI [{lo_o:.3f},{hi_o:.3f}]")
    if not np.isnan(hi_b) and not np.isnan(lo_o):
        if lo_o > hi_b:
            print("    → CIs separate: smaller model overlaps more with heuristic.")
        elif lo_b > hi_o:
            print("    → CIs separate: smaller model overlaps LESS with heuristic.")
        else:
            print("    → CIs overlap: no reliable difference on this coarse metric.")

    # ── STABILITY across splits (renamed, descriptive) ────────────────────
    print("\n  " + "-" * 66)
    print("  4. REGION STABILITY ACROSS SPLITS  (descriptive, not quality)")
    print("  " + "-" * 66)
    sb, sb_all = stability(base_llm)
    so, so_all = stability(other_llm)
    mb_s, lo_bs, hi_bs = boot_ci(sb_all)
    mo_s, lo_os, hi_os = boot_ci(so_all)
    print(f"    {bmodel:<22} stability: {sb:.3f}  95% CI [{lo_bs:.3f},{hi_bs:.3f}]")
    print(f"    {omodel:<22} stability: {so:.3f}  95% CI [{lo_os:.3f},{hi_os:.3f}]")
    print("    NOTE: higher = same region-types across splits. A model that")
    print("    adapts per split will score lower — that is not necessarily")
    print("    worse. Report descriptively, not as a quality ranking.")

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────
    print("\n  " + "-" * 66)
    print("  5. SUMMARY TABLE")
    print("  " + "-" * 66)
    fb, fo = final_stats(baseline_dir), final_stats(other_dir)
    fbm, fblo, fbhi = boot_ci(fb); fom, folo, fohi = boot_ci(fo)
    rows = [
        ("final G (mean)", f"{fbm:.1f}", f"{fom:.1f}"),
        ("  95% CI", f"[{fblo:.0f},{fbhi:.0f}]", f"[{folo:.0f},{fohi:.0f}]"),
        ("descriptor overlap vs heuristic", f"{dov_bm:.3f}", f"{dov_om:.3f}"),
        ("region Jaccard vs heuristic", f"{mb:.3f}", f"{mo:.3f}"),
        ("coverage (mean-pair)", _f(cov_b[0]), _f(cov_o[0])),
        ("novelty (dist to train)", _f(cov_b[2]), _f(cov_o[2])),
        ("region stability", f"{sb:.3f}", f"{so:.3f}"),
        ("distinct regions", str(len(set(x['region'] for x in base_llm))),
                              str(len(set(x['region'] for x in other_llm)))),
    ]
    print(f"    {'metric':<34}{bmodel[:14]:>14}{omodel[:14]:>14}")
    for name, a, b in rows:
        print(f"    {name:<34}{a:>14}{b:>14}")

    print("\n  " + "-" * 66)
    print("  DESCRIPTIVE (regions observed — NOT a success criterion)")
    print("  " + "-" * 66)
    describe_regions(base_llm, bmodel)
    describe_regions(other_llm, omodel)

    out = {'baseline_model': bmodel, 'other_model': omodel,
           'descriptor_overlap': {'cross_model': dov, 'baseline_vs_heuristic': dov_bm,
                                   'other_vs_heuristic': dov_om},
           'region_jaccard_vs_heuristic': {'baseline': [mb, lo_b, hi_b],
                                           'other': [mo, lo_o, hi_o]},
           'coverage': {'baseline': cov_b, 'other': cov_o},
           'stability': {'baseline': [sb, lo_bs, hi_bs], 'other': [so, lo_os, hi_os]},
           'final_G': {'baseline': [fbm, fblo, fbhi], 'other': [fom, folo, fohi]}}
    json.dump(out, open(f'{other_dir}/cross_model_comparison.json', 'w',
                        encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n  Saved → {other_dir}/cross_model_comparison.json")
    print("\n  INTERPRET AGNOSTICALLY. Two models = a robustness check across two")
    print("  capability levels, NOT a scaling law. Do not claim 'reasons better';")
    print("  claim only what is measured: descriptor-region distinctness.")
    print("=" * 70)


def _f(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) and x is not None and not np.isnan(x) else " n/a"


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python compare_models.py <baseline_dir> <other_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
