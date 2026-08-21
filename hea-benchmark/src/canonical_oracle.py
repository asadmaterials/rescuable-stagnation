"""
Canonical Data Split & Oracle
==============================
Fixes the two foundational validity problems:

  1. ORACLE LEAKAGE — previously the RF oracle trained on the FULL dataset,
     so BO was "discovering" alloys the oracle had already memorized.
     Now the oracle trains ONLY on a train partition. The BO loop discovers
     over a candidate universe that includes held-out compositions the
     oracle never saw during training.

  2. SINGLE CANONICAL ORACLE — every experimental arm (random / mutation /
     LLM injection) is judged by the exact same oracle instance, trained on
     the exact same train split with the exact same hyperparameters. This
     makes cross-arm comparisons fair: the oracle may be imperfect, but it
     is imperfect *identically* for all arms.

Split design:
    TRAIN   (60%) — oracle fits here ONLY
    VAL     (20%) — hyperparameter / threshold tuning
    HIDDEN  (20%) — candidate universe for the BO loop; oracle never
                    trained on these, so querying them is genuine
                    out-of-training-sample evaluation

The oracle also exposes a confidence flag (distance to nearest train point)
so downstream analysis can mark predictions made far from training data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ── Canonical feature definition ──────────────────────────────────────────────
DESCRIPTOR_COLS = ['delta_S', 'delta_H', 'delta', 'VEC', 'delta_X', 'omega']

def get_feature_cols(df: pd.DataFrame) -> list:
    """Canonical feature columns: composition fractions + 6 descriptors."""
    return [c for c in df.columns
            if c not in ['HV', 'FORMULA', 'stability_score']]

def get_composition_cols(feature_cols: list) -> list:
    """Composition-only columns (exclude descriptors)."""
    return [c for c in feature_cols if c not in DESCRIPTOR_COLS]


# ── Canonical three-way split ─────────────────────────────────────────────────

def make_splits(
    df:          pd.DataFrame,
    train_frac:  float = 0.60,
    val_frac:    float = 0.20,
    random_seed: int   = 42,
) -> dict:
    """
    Deterministic train / validation / hidden split.

    The split is stratified by hardness quartile so all three partitions
    span the full HV range — otherwise the hidden set could accidentally
    contain all the hardest alloys (making discovery trivially impossible)
    or none of them (making it trivially easy).

    Parameters
    ----------
    df          : processed dataframe with 'HV' column
    train_frac  : fraction for oracle training
    val_frac    : fraction for tuning (hidden gets the remainder)
    random_seed : reproducibility

    Returns
    -------
    dict with keys 'train', 'val', 'hidden' → DataFrames
    """
    df = df.reset_index(drop=True).copy()

    # Stratify by HV quartile
    df['_hv_bin'] = pd.qcut(df['HV'], q=4, labels=False, duplicates='drop')

    # First split off train
    train_df, temp_df = train_test_split(
        df, train_size=train_frac,
        stratify=df['_hv_bin'], random_state=random_seed,
    )
    # Split remainder into val / hidden
    val_size_rel = val_frac / (1.0 - train_frac)
    val_df, hidden_df = train_test_split(
        temp_df, train_size=val_size_rel,
        stratify=temp_df['_hv_bin'], random_state=random_seed,
    )

    for d in (train_df, val_df, hidden_df):
        d.drop(columns='_hv_bin', inplace=True)

    print(f"  Split — train: {len(train_df)}  val: {len(val_df)}  "
          f"hidden: {len(hidden_df)}")
    print(f"  HV range — train: [{train_df['HV'].min():.0f}, {train_df['HV'].max():.0f}]  "
          f"hidden: [{hidden_df['HV'].min():.0f}, {hidden_df['HV'].max():.0f}]")

    return {
        'train':  train_df.reset_index(drop=True),
        'val':    val_df.reset_index(drop=True),
        'hidden': hidden_df.reset_index(drop=True),
    }


# ── Canonical oracle (trains on TRAIN only) ───────────────────────────────────

class CanonicalOracle:
    """
    The single ground-truth oracle used by ALL experimental arms.

    Trained ONCE on the train partition. Frozen thereafter. Every arm
    queries this same instance, so the comparison between arms is fair.

    Also exposes:
      - query()      : predict HV for a feature vector
      - confidence() : distance to nearest train point (low = reliable,
                       high = extrapolating beyond training data)

    The confidence flag lets the paper be honest about which predictions
    are trustworthy interpolations vs. shaky extrapolations, directly
    addressing the RF-cannot-extrapolate critique.
    """

    def __init__(
        self,
        train_df:     pd.DataFrame,
        feature_cols: list,
        n_estimators: int = 300,
        max_features: str = 'sqrt',
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.comp_cols    = get_composition_cols(feature_cols)
        self.query_count  = 0

        X = train_df[feature_cols].values
        y = train_df['HV'].values

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        self.model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_features = max_features,
            random_state = random_state,
            n_jobs       = -1,
        )
        self.model.fit(X_scaled, y)

        # Store scaled train points (composition only) for confidence flag
        comp_idx          = [feature_cols.index(c) for c in self.comp_cols]
        self._train_comp  = X_scaled[:, comp_idx]
        self._comp_idx    = comp_idx

        print(f"  CanonicalOracle trained on {len(train_df)} TRAIN samples "
              f"(n_trees={n_estimators}, max_features={max_features}).")

    def query(self, x: np.ndarray) -> float:
        """Predict HV for a single feature vector (n_features,)."""
        self.query_count += 1
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return float(self.model.predict(x_scaled)[0])

    def query_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict HV for a batch (n, n_features). Does not increment count."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def confidence(self, x: np.ndarray) -> float:
        """
        Distance from x to nearest TRAIN point in composition space.

        Low  → x is near training data, RF prediction is a reliable
               interpolation.
        High → x is far from any training point, RF prediction is an
               unreliable extrapolation (flag for the paper).
        """
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        x_comp   = x_scaled[:, self._comp_idx]
        dists    = np.linalg.norm(self._train_comp - x_comp, axis=1)
        return float(dists.min())


# ── Canonical dataset loader (A4-new: single clip, enforced) ──────────────────

WORKING_DATA = '../data/processed/hea_hardness_working.csv'
OMEGA_CLIP   = 50.0

def load_working_dataset(path: str = WORKING_DATA) -> pd.DataFrame:
    """
    Load the canonical working dataset with the omega clip applied and
    ASSERTED. This is the single point where omega clipping is enforced;
    all consumers must load through here rather than clipping ad hoc,
    so feature vectors for dataset points and admitted candidates share
    one scale (A4-new fix — prevents silent Mahalanobis/GP corruption).
    """
    df = pd.read_csv(path)
    # Enforce (not silently re-apply) the clip: the working CSV is written
    # pre-clipped; this guards against an unclipped file slipping in.
    if df['omega'].max() > OMEGA_CLIP + 1e-9:
        raise ValueError(
            f"omega exceeds clip {OMEGA_CLIP} in {path} "
            f"(max={df['omega'].max():.2f}). The working dataset must be "
            f"written pre-clipped; do not clip downstream.")
    return df


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_working_dataset('../data/processed/hea_hardness_working.csv')

    feature_cols = get_feature_cols(df)
    print(f"Feature cols ({len(feature_cols)}): {feature_cols}\n")

    splits = make_splits(df, random_seed=42)

    oracle = CanonicalOracle(splits['train'], feature_cols)

    # Sanity: query a hidden point (oracle never saw this during training)
    hidden = splits['hidden']
    x0     = hidden[feature_cols].values[0]
    true0  = hidden['HV'].values[0]
    pred0  = oracle.query(x0)
    conf0  = oracle.confidence(x0)

    print(f"\n  Hidden point 0:")
    print(f"    True HV      : {true0:.1f}")
    print(f"    Oracle pred  : {pred0:.1f}")
    print(f"    Confidence d : {conf0:.3f}  (lower = more reliable)")

    # Evaluate oracle quality on the WHOLE hidden set (genuine OOS)
    from sklearn.metrics import r2_score, mean_absolute_error
    X_hidden = hidden[feature_cols].values
    y_hidden = hidden['HV'].values
    y_pred   = oracle.query_batch(X_hidden)
    print(f"\n  Oracle quality on HIDDEN set (genuine out-of-sample):")
    print(f"    R²  : {r2_score(y_hidden, y_pred):.4f}")
    print(f"    MAE : {mean_absolute_error(y_hidden, y_pred):.2f} HV")
    print(f"    (This is the HONEST number — no leakage.)")
