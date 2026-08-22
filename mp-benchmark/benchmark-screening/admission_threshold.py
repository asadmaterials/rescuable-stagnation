"""
Admission Distance Threshold
============================

Usage:
    python admission_threshold.py        # expects mp_shear_metallic.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, pairwise_distances_argmin_min
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition


def main(path='mp_shear_metallic.csv'):
    print("=" * 68)
    print("  ADMISSION DISTANCE THRESHOLD")
    print("  Keep candidates in the oracle's flat-error region")
    print("=" * 68)

    df = pd.read_csv(path)
    df = df[(df['G'] > 0) & (df['G'] < 600)].reset_index(drop=True)

    print("  Featurizing...")
    df['comp_obj'] = df['formula'].apply(Composition)
    ep = ElementProperty.from_preset('magpie')
    df = ep.featurize_dataframe(df, 'comp_obj', ignore_errors=True)
    cols = ep.feature_labels()
    df = df.dropna(subset=cols).reset_index(drop=True)

    X = df[cols].values
    y = df['G'].values

    # Same split style as diagnostics; the scaler defines the distance metric
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=0)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte = y[tr], y[te]

    rf = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                               random_state=0, n_jobs=-1).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    err = np.abs(pred - yte)

    # distance of each held-out point to nearest training point
    _, dmin = pairwise_distances_argmin_min(Xte, Xtr)

    print(f"\n  held-out points: {len(te)}")
    print(f"  distance range : {dmin.min():.2f} - {dmin.max():.2f} "
          f"(median {np.median(dmin):.2f})")
    print(f"  overall mean |error|: {err.mean():.1f} GPa")

    # ── sweep thresholds: error inside vs pool retained ───────────────────
    print("\n  " + "-" * 62)
    print(f"  {'threshold':>10}{'pctile':>8}{'mean|err| inside':>18}"
          f"{'% admitted':>13}")
    print("  " + "-" * 62)

    pctiles = [50, 60, 70, 75, 80, 85, 90, 95]
    flat_err = None
    rows = []
    for p in pctiles:
        thr = np.percentile(dmin, p)
        inside = dmin <= thr
        me = err[inside].mean()
        rows.append((p, thr, me, 100 * inside.mean()))
        if p == 50:
            flat_err = me   # reference: error in the closest half
        print(f"  {thr:>10.2f}{p:>7}%{me:>18.1f}{100*inside.mean():>12.0f}%")

    # recommend: largest threshold whose inside-error stays within
    # ~15% of the closest-half error (i.e. still "flat")
    print("\n  " + "-" * 62)
    tol = 1.15 * flat_err
    ok = [(p, thr, me, adm) for (p, thr, me, adm) in rows if me <= tol]
    if ok:
        p, thr, me, adm = ok[-1]   # largest admitting threshold still flat
        print(f"  RECOMMENDED THRESHOLD: distance <= {thr:.2f}")
        print(f"    (= {p}th percentile of candidate distances)")
        print(f"    mean |error| inside: {me:.1f} GPa  "
              f"(vs {flat_err:.1f} in closest half)")
        print(f"    pool admitted: {adm:.0f}%")
        print("\n    Rationale: largest distance bound at which oracle error")
        print("    stays within 15% of its closest-region value — the search")
        print("    keeps most of the pool while every candidate is scored in")
        print("    the oracle's reliable region. Pre-register this cutoff.")
    else:
        print("  No threshold keeps error flat; the oracle degrades quickly.")
        print("  Consider a tighter novelty gate or improving the oracle.")

    print("\n  NOTE: enforce this in admission using the SAME metric —")
    print("  Euclidean distance to nearest TRAIN point in the scaled Magpie")
    print("  feature space. Do not substitute the Mahalanobis dedup distance;")
    print("  they are different quantities and mixing them reintroduces the")
    print("  space-mismatch class of bug from the earlier work.")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mp_shear_metallic.csv')
