"""
Oracle Diagnostics for the MP Shear-Modulus Benchmark
=====================================================
Runs on the cached CSV (mp_shear_metallic.csv) — NO API calls, fast,
connection-independent.


Usage:
    python oracle_diagnostics.py                 # expects mp_shear_metallic.csv
    python oracle_diagnostics.py path/to.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition


def featurize(df):
    df = df.copy()
    df['comp_obj'] = df['formula'].apply(Composition)
    ep = ElementProperty.from_preset('magpie')
    df = ep.featurize_dataframe(df, 'comp_obj', ignore_errors=True)
    cols = ep.feature_labels()
    df = df.dropna(subset=cols).reset_index(drop=True)
    return df, cols


def main(path='mp_shear_metallic.csv'):
    print("=" * 70)
    print("  ORACLE DIAGNOSTICS — is R2=0.43 good enough to build on?")
    print("=" * 70)

    df = pd.read_csv(path)
    # guard: apply the same physical filter in case an old CSV slipped in
    df = df[(df['G'] > 0) & (df['G'] < 600)].reset_index(drop=True)
    print(f"  compositions: {len(df)}   G: {df['G'].min():.0f}-{df['G'].max():.0f} "
          f"mean {df['G'].mean():.0f} GPa")

    print("\n  Featurizing (Magpie)...")
    df, cols = featurize(df)
    X = df[cols].values
    y = df['G'].values

    # Consistent split across all tests
    Xtr_i, Xte_i = train_test_split(np.arange(len(df)), test_size=0.3,
                                    random_state=0)
    sc = StandardScaler().fit(X[Xtr_i])
    Xtr, Xte = sc.transform(X[Xtr_i]), sc.transform(X[Xte_i])
    ytr, yte = y[Xtr_i], y[Xte_i]

    # ── Q1 + Q2: model comparison, R2 AND rank correlation ────────────────
    print("\n  " + "-" * 66)
    print("  Q1/Q2 — value accuracy (R2/MAE) AND ranking (Spearman)")
    print("  " + "-" * 66)
    print(f"    {'model':<20}{'R2':>8}{'MAE':>9}{'Spearman':>11}")

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=300,
                        max_features='sqrt', random_state=0, n_jobs=-1),
        'GradientBoost': GradientBoostingRegressor(n_estimators=400,
                        max_depth=3, learning_rate=0.05, subsample=0.8,
                        random_state=0),
    }
    best = None
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        r2 = r2_score(yte, p)
        mae = mean_absolute_error(yte, p)
        rho = spearmanr(yte, p).correlation
        print(f"    {name:<20}{r2:>8.3f}{mae:>9.1f}{rho:>11.3f}")
        if best is None or rho > best[1]:
            best = (name, rho, m, p)
    print("\n    Interpretation: for a BENCHMARK OBJECTIVE, Spearman is the")
    print("    number that matters — it says the oracle ranks good candidates")
    print("    above bad ones. Spearman >= 0.7 is a discriminating landscape")
    print("    even when R2 is modest; the search can still be scored fairly.")

    # ── Q3: THE decisive test — is error coupled to distance? ─────────────
    print("\n  " + "-" * 66)
    print("  Q3 — ERROR vs DISTANCE FROM TRAINING (the confound check)")
    print("  " + "-" * 66)
    # distance = min Euclidean distance (scaled features) to any train point
    from sklearn.metrics import pairwise_distances_argmin_min
    _, dmin = pairwise_distances_argmin_min(Xte, Xtr)
    abs_err = np.abs(best[3] - yte)

    # correlation of error with distance
    rho_ed = spearmanr(dmin, abs_err).correlation
    print(f"    Using best-ranking model: {best[0]}")
    print(f"    Spearman(distance_to_train, |error|) = {rho_ed:+.3f}")

    # error in distance quartiles
    q = np.quantile(dmin, [0.25, 0.5, 0.75])
    bins = np.digitize(dmin, q)
    print(f"\n    {'distance quartile':<22}{'n':>6}{'mean |error| GPa':>18}")
    for b in range(4):
        m = bins == b
        label = ['Q1 (closest)', 'Q2', 'Q3', 'Q4 (farthest)'][b]
        print(f"    {label:<22}{m.sum():>6}{abs_err[m].mean():>18.1f}")

    print("\n    Interpretation:")
    if rho_ed > 0.25:
        print("      ⚠ ERROR RISES WITH DISTANCE. Exploratory arms (LLM, digest)")
        print("        will be scored more noisily than local arms (mutation) —")
        print("        the Borg confound in a new dress. This must be addressed:")
        print("        report it explicitly, restrict candidate novelty, or use")
        print("        the dual-channel check to flag high-distance proposals.")
    elif rho_ed > 0.1:
        print("      ~ Mild distance coupling. Tolerable, but report it and lean")
        print("        on the physics channel to cross-check distant proposals.")
    else:
        print("      ✓ Error is essentially FLAT in distance. The oracle judges")
        print("        near and far candidates comparably — no distance confound.")
        print("        R2=0.43 is then acceptable as a benchmark objective: the")
        print("        noise is uniform, shared by all arms, and cancels in the")
        print("        paired comparison.")

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n  " + "=" * 66)
    print("  VERDICT")
    print("  " + "=" * 66)
    ok_rank = best[1] >= 0.7
    ok_dist = rho_ed <= 0.25
    if ok_rank and ok_dist:
        print("  BUILD. Ranking is discriminating and error is not distance-")
        print("  coupled. Report R2 as a limitation; the oracle is a fair judge.")
    elif ok_rank and not ok_dist:
        print("  BUILD WITH CARE. Ranking is fine but error rises with distance.")
        print("  Address the confound before trusting exploratory-arm results.")
    else:
        print("  IMPROVE FIRST. Ranking is too weak for a discriminating")
        print("  benchmark; try more features or accept structure-aware inputs.")
    print(f"    best-ranking model: {best[0]} (Spearman {best[1]:.3f})")
    print(f"    error-distance coupling: {rho_ed:+.3f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mp_shear_metallic.csv')
