"""
Cross-Model Difference CIs  (Sonnet vs Haiku)
=============================================
Turns "nearly identical" into a statistical statement. For each comparison
metric it reports the between-model difference and a 95% bootstrap CI, so the
paper can say "no evidence of a practically meaningful difference (Δ, 95% CI
[.,.])" instead of "they look close".

Two resampling schemes, matched to each metric's unit of analysis:

  PAIRED over (split, seed) runs — for run-level metrics where both models were
  run on the SAME seeds (final G, coherence per trace grouped by run). We
  resample matched run-pairs and recompute the difference, so the CI reflects
  the paired design (identical seeds → the shared BO scaffold cancels).

  UNPAIRED over injected candidates — for descriptor-space aggregates
  (coverage, novelty) and region metrics, which are computed over pooled
  injected candidates, not per matched run. We bootstrap each model's pooled
  statistic independently and difference them.

Reports, per metric:  value_A, value_B, Δ=A−B, 95% CI on Δ, and whether the CI
includes 0 ("no reliable difference").

Usage:
    python compare_models_ci.py results/mp_shear_v1 results/haiku_v1
"""

import os
import sys
import json
import glob
import warnings
import numpy as np

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(0)
NBOOT = 5000


# ── loaders ────────────────────────────────────────────────────────────────

def finals_by_run(save_dir):
    """{(split,seed): final_best} for the llm arm."""
    d = {}
    for fn in glob.glob(f'{save_dir}/run_detail/llm_*.json'):
        r = json.load(open(fn, encoding='utf-8'))
        d[(r['split_seed'], r['bo_seed'])] = r['final_best']
    return d


def coherence_by_run(save_dir):
    """{(split,seed): mean coherence over that run's traces}, from the
    reasoning_coherence.json produced by the scorer."""
    fn = f'{save_dir}/reasoning_coherence.json'
    if not os.path.exists(fn):
        return {}
    data = json.load(open(fn, encoding='utf-8'))
    by = {}
    for e in data.get('events', []):
        by.setdefault((e['split'], e['seed']), []).append(e['mean_total'])
    return {k: float(np.mean(v)) for k, v in by.items()}


def injected_vectors(save_dir):
    """Scaled Magpie vectors of all llm-injected candidates (baseline scaler)."""
    import mp_oracle as MO
    from mp_oracle import MPOracle, FEATURE_COLS, composition_to_vector
    from mp_runner import prepare_dataset, make_splits
    df, _ = prepare_dataset()
    tr_idx, pool_idx = make_splits(len(df), seed=0)
    orc = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=0)
    vecs = []
    for fn in glob.glob(f'{save_dir}/run_detail/llm_*.json'):
        d = json.load(open(fn, encoding='utf-8'))
        for ev in d.get('inject_events', []):
            for comp in ev.get('compositions', []):
                v = composition_to_vector(comp, FEATURE_COLS)
                if v is not None:
                    vecs.append(orc.scaler.transform(v.reshape(1, -1))[0])
    return orc, (np.array(vecs) if vecs else np.empty((0,)))


# ── paired bootstrap over matched runs ─────────────────────────────────────

def paired_diff_ci(A, B, label, unit="GPa"):
    keys = sorted(set(A) & set(B))
    if len(keys) < 3:
        print(f"    {label}: too few matched runs"); return None
    a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
    diffs = a - b
    obs = float(diffs.mean())
    boots = [float(RNG.choice(diffs, len(diffs), replace=True).mean())
             for _ in range(NBOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    incl0 = lo <= 0 <= hi
    print(f"    {label:<26} A={a.mean():.3f}  B={b.mean():.3f}  "
          f"Δ={obs:+.3f} {unit}  95% CI [{lo:+.3f},{hi:+.3f}]  "
          f"{'(includes 0)' if incl0 else '(excludes 0)'}")
    return {'A': float(a.mean()), 'B': float(b.mean()), 'delta': obs,
            'ci': [float(lo), float(hi)], 'includes_zero': bool(incl0),
            'n_pairs': len(keys)}


# ── unpaired bootstrap over pooled candidates ──────────────────────────────

def unpaired_diff_ci(Va, Vb, statfn, label, unit=""):
    if len(Va) < 5 or len(Vb) < 5:
        print(f"    {label}: too few candidates"); return None
    sa, sb = statfn(Va), statfn(Vb)
    boots = []
    for _ in range(NBOOT):
        ra = Va[RNG.integers(0, len(Va), len(Va))]
        rb = Vb[RNG.integers(0, len(Vb), len(Vb))]
        boots.append(statfn(ra) - statfn(rb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    incl0 = lo <= 0 <= hi
    print(f"    {label:<26} A={sa:.3f}  B={sb:.3f}  "
          f"Δ={sa-sb:+.3f} {unit}  95% CI [{lo:+.3f},{hi:+.3f}]  "
          f"{'(includes 0)' if incl0 else '(excludes 0)'}")
    return {'A': float(sa), 'B': float(sb), 'delta': float(sa-sb),
            'ci': [float(lo), float(hi)], 'includes_zero': bool(incl0)}


def _coverage(V):
    from sklearn.metrics import pairwise_distances
    idx = RNG.choice(len(V), min(len(V), 200), replace=False)
    S = V[idx]; D = pairwise_distances(S)
    return float(D[np.triu_indices_from(D, 1)].mean())


def main(dir_a, dir_b):
    print("=" * 70)
    print("  CROSS-MODEL DIFFERENCE CIs  (paired where seeds match)")
    print("=" * 70)
    ma = json.load(open(f'{dir_a}/config.json')).get('llm_model',
         json.load(open(f'{dir_a}/config.json')).get('model', 'A'))
    mb = json.load(open(f'{dir_b}/config.json')).get('model', 'B')
    print(f"  A = {ma}   ({dir_a})")
    print(f"  B = {mb}   ({dir_b})")
    print(f"  bootstrap resamples: {NBOOT}\n")
    out = {'model_A': ma, 'model_B': mb}

    print("  " + "-" * 66)
    print("  PAIRED over matched (split, seed) runs")
    print("  " + "-" * 66)
    out['final_G'] = paired_diff_ci(finals_by_run(dir_a), finals_by_run(dir_b),
                                    "final G", "GPa")
    ca, cb = coherence_by_run(dir_a), coherence_by_run(dir_b)
    if ca and cb:
        out['coherence'] = paired_diff_ci(ca, cb, "rationale coherence", "/8")
    else:
        print("    rationale coherence: run reasoning_coherence.py on BOTH dirs first")

    print("\n  " + "-" * 66)
    print("  UNPAIRED over pooled injected candidates (descriptor space)")
    print("  " + "-" * 66)
    try:
        orc, Va = injected_vectors(dir_a)
        _, Vb = injected_vectors(dir_b)
        from sklearn.metrics import pairwise_distances_argmin_min
        def novelty(V):
            _, d = pairwise_distances_argmin_min(V, orc._Xtr_s); return float(d.mean())
        out['coverage'] = unpaired_diff_ci(Va, Vb, _coverage, "coverage (spread)")
        out['novelty'] = unpaired_diff_ci(Va, Vb, novelty, "novelty (dist to train)")
    except Exception as e:
        print(f"    descriptor metrics unavailable: {e}")

    json.dump(out, open(f'{dir_b}/cross_model_difference_ci.json', 'w',
                        encoding='utf-8'), indent=2, ensure_ascii=False)
    print("\n  " + "-" * 66)
    print("  READING: a CI that INCLUDES 0 supports 'no evidence of a")
    print("  practically meaningful difference' on that metric. Report the")
    print("  Δ and CI, not just the point estimates. All differences here are")
    print("  between two Claude capability levels — not a universal claim.")
    print(f"\n  Saved → {dir_b}/cross_model_difference_ci.json")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python compare_models_ci.py <dir_A> <dir_B>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
