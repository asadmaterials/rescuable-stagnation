"""
Per-Split Oracle MAE Checker
============================
Recomputes each split's oracle MAE (mean absolute error on that split's
pool), which the main run did not save. Needed to set ORACLE_MAE correctly
in mp_analysis.py — the rescuable-event count (and thus the "underpowered
null" framing) depends on it.

Fast: loads the cached feature matrix and rebuilds the 5 oracles exactly as
the run did. No experiment re-run, no API calls.

Run from the project folder (same directory as mp_oracle.py, mp_runner.py,
and mp_shear_metallic.csv):

    python check_mae.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import mp_oracle as MO
from mp_runner import prepare_dataset, make_splits
from mp_oracle import MPOracle, FEATURE_COLS


def main():
    print("=" * 56)
    print("  PER-SPLIT ORACLE MAE  (for ORACLE_MAE in mp_analysis.py)")
    print("=" * 56)

    df, comps = prepare_dataset()          # uses cached features if present
    y_all = df['G'].values
    X_all = df[FEATURE_COLS].values
    n = len(df)
    print(f"  dataset: {n} compositions\n")

    print(f"  {'split':>6}{'MAE (GPa)':>12}{'R2':>8}{'pool n':>9}")
    maes, r2s = [], []
    for sp in range(5):                    # same 5 splits the run used
        tr_idx, pool_idx = make_splits(n, seed=sp)
        orc = MPOracle(df, FEATURE_COLS, tr_idx, pool_idx, seed=sp)
        pred = orc.query_batch(X_all[pool_idx])
        true = y_all[pool_idx]
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        maes.append(mae); r2s.append(r2)
        print(f"  {sp:>6}{mae:>12.2f}{r2:>8.3f}{len(pool_idx):>9}")

    print("\n" + "-" * 56)
    print(f"  mean per-split MAE = {np.mean(maes):.2f} GPa "
          f"(std {np.std(maes):.2f})")
    print(f"  mean per-split R2  = {np.mean(r2s):.3f}")
    print("-" * 56)
    print(f"\n  → Set ORACLE_MAE = {np.mean(maes):.1f} at the top of mp_analysis.py,")
    print("    then re-run:  python mp_analysis.py")
    print("    Section 4 (rescuable events) will update accordingly — that is")
    print("    the number the 'underpowered null' framing rests on.")


if __name__ == '__main__':
    main()
