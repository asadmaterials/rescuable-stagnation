"""
Phase 1 — Data Pipeline
=======================
Loads the Borg et al. (2020) MPEA dataset, filters for hardness entries,
parses alloy formulas into molar fractions, computes thermodynamic and
physical descriptors, applies a stability filter, and saves the cleaned
dataset ready for the BO loop.

Run this script once before anything else.
"""

import re
import os
import numpy as np
import pandas as pd

# ── Elemental property table ──────────────────────────────────────────────────
# Source: standard literature values used across HEA descriptor papers
# Keys: atomic radius (pm), electronegativity (Pauling), melting point (K),
#       valence electron count, mixing enthalpy parameters (Takeuchi & Inoue)
ELEMENT_DATA = {
    #         radius  X_paul  Tm(K)  VEC
    'Al': {'r': 143, 'X': 1.61, 'Tm': 933,  'VEC': 3},
    'Co': {'r': 125, 'X': 1.88, 'Tm': 1768, 'VEC': 9},
    'Cr': {'r': 128, 'X': 1.66, 'Tm': 2180, 'VEC': 6},
    'Cu': {'r': 128, 'X': 1.90, 'Tm': 1358, 'VEC': 11},
    'Fe': {'r': 126, 'X': 1.83, 'Tm': 1811, 'VEC': 8},
    'Hf': {'r': 159, 'X': 1.30, 'Tm': 2506, 'VEC': 4},
    'Mn': {'r': 127, 'X': 1.55, 'Tm': 1519, 'VEC': 7},
    'Mo': {'r': 139, 'X': 2.16, 'Tm': 2896, 'VEC': 6},
    'Nb': {'r': 146, 'X': 1.60, 'Tm': 2750, 'VEC': 5},
    'Ni': {'r': 124, 'X': 1.91, 'Tm': 1728, 'VEC': 10},
    'Si': {'r': 117, 'X': 1.90, 'Tm': 1687, 'VEC': 4},
    'Ta': {'r': 146, 'X': 1.50, 'Tm': 3290, 'VEC': 5},
    'Ti': {'r': 147, 'X': 1.54, 'Tm': 1941, 'VEC': 4},
    'V':  {'r': 134, 'X': 1.63, 'Tm': 2183, 'VEC': 5},
    'W':  {'r': 139, 'X': 2.36, 'Tm': 3695, 'VEC': 6},
    'Zr': {'r': 160, 'X': 1.33, 'Tm': 2128, 'VEC': 4},
    'B':  {'r': 87,  'X': 2.04, 'Tm': 2349, 'VEC': 3},
    'C':  {'r': 77,  'X': 2.55, 'Tm': 3823, 'VEC': 4},
    'Re': {'r': 137, 'X': 1.90, 'Tm': 3459, 'VEC': 7},
}

# Binary mixing enthalpy table (kJ/mol)
# Primary source  : Takeuchi & Inoue (2005) Metall. Mater. Trans. A
# Extended pairs  : de Boer et al. (1988), CALPHAD estimates
#
# IMPORTANT — missing pair policy:
#   Previously: missing pair → 0  (WRONG: silently corrupts ΔHmix)
#   Now: get_hmix() returns np.nan for missing pairs → rows are
#   flagged and dropped during descriptor computation.
#   All 24 previously missing pairs have now been added from
#   literature (B, C, Re, Hf containing pairs).
H_MIX = {
    # ── Al pairs ──────────────────────────────────────────────────────────
    ('Al', 'B'):  -46,  # de Boer et al.
    ('Al', 'Co'): -19, ('Al', 'Cr'): -10, ('Al', 'Cu'): -1,
    ('Al', 'Fe'): -11, ('Al', 'Hf'): -38, ('Al', 'Mn'): -19,
    ('Al', 'Mo'): -22, ('Al', 'Nb'): -18, ('Al', 'Ni'): -22,
    ('Al', 'Si'): -19, ('Al', 'Ta'): -19, ('Al', 'Ti'): -30,
    ('Al', 'V'):  -16, ('Al', 'W'):  -22, ('Al', 'Zr'): -44,
    # ── B pairs ───────────────────────────────────────────────────────────
    ('B',  'Co'): -24, ('B',  'Cr'): -3,  ('B',  'Cu'): -5,
    ('B',  'Fe'): -26, ('B',  'Mn'): -26, ('B',  'Ni'): -40,
    # ── C pairs ───────────────────────────────────────────────────────────
    ('C',  'Co'): -42, ('C',  'Cr'): -61, ('C',  'Fe'): -50,
    ('C',  'Mn'): -57, ('C',  'Mo'): -67, ('C',  'Nb'): -109,
    ('C',  'Ni'): -39, ('C',  'Re'): -80, ('C',  'Ta'): -144,
    ('C',  'Ti'): -154,('C',  'W'):  -72,
    # ── Co pairs ──────────────────────────────────────────────────────────
    ('Co', 'Cr'): -4,  ('Co', 'Cu'): 6,   ('Co', 'Fe'): -1,
    ('Co', 'Hf'): -23, ('Co', 'Mn'): -5,  ('Co', 'Mo'): -5,
    ('Co', 'Nb'): -25, ('Co', 'Ni'): 0,   ('Co', 'Si'): -38,
    ('Co', 'Ta'): -24, ('Co', 'Ti'): -28, ('Co', 'V'):  -14,
    ('Co', 'W'):  -1,  ('Co', 'Zr'): -41,
    # ── Cr pairs ──────────────────────────────────────────────────────────
    ('Cr', 'Cu'): 12,  ('Cr', 'Fe'): -1,  ('Cr', 'Hf'): -12,
    ('Cr', 'Mn'): 2,   ('Cr', 'Mo'): 0,   ('Cr', 'Nb'): -7,
    ('Cr', 'Ni'): -7,  ('Cr', 'Si'): -26, ('Cr', 'Ta'): -7,
    ('Cr', 'Ti'): -7,  ('Cr', 'V'):  -2,  ('Cr', 'W'):  0,
    ('Cr', 'Zr'): -12,
    # ── Cu pairs ──────────────────────────────────────────────────────────
    ('Cu', 'Fe'): 13,  ('Cu', 'Mn'): 4,   ('Cu', 'Mo'): 19,
    ('Cu', 'Nb'): 3,   ('Cu', 'Ni'): 4,   ('Cu', 'Si'): -3,
    ('Cu', 'Ta'): 2,   ('Cu', 'Ti'): -9,  ('Cu', 'V'):  5,
    ('Cu', 'W'):  22,  ('Cu', 'Zr'): -23,
    # ── Fe pairs ──────────────────────────────────────────────────────────
    ('Fe', 'Hf'): -23, ('Fe', 'Mn'): 0,   ('Fe', 'Mo'): -2,
    ('Fe', 'Nb'): -16, ('Fe', 'Ni'): -2,  ('Fe', 'Si'): -35,
    ('Fe', 'Ta'): -15, ('Fe', 'Ti'): -17, ('Fe', 'V'):  -7,
    ('Fe', 'W'):  0,   ('Fe', 'Zr'): -25,
    # ── Hf pairs ──────────────────────────────────────────────────────────
    ('Hf', 'Mo'): -4,  ('Hf', 'Nb'): 4,   ('Hf', 'Ni'): -42,
    ('Hf', 'Si'): -67, ('Hf', 'Ta'): 3,   ('Hf', 'Ti'): 0,
    ('Hf', 'V'):  -2,  ('Hf', 'W'):  -5,  ('Hf', 'Zr'): 0,
    # ── Mn pairs ──────────────────────────────────────────────────────────
    ('Mn', 'Mo'): -5,  ('Mn', 'Nb'): -8,  ('Mn', 'Ni'): -8,
    ('Mn', 'Si'): -45, ('Mn', 'Ta'): -8,  ('Mn', 'Ti'): -20,
    ('Mn', 'V'):  -1,  ('Mn', 'W'):  -5,  ('Mn', 'Zr'): -30,
    # ── Mo pairs ──────────────────────────────────────────────────────────
    ('Mo', 'Nb'): -6,  ('Mo', 'Ni'): -7,  ('Mo', 'Re'): -5,
    ('Mo', 'Si'): -34, ('Mo', 'Ta'): -5,  ('Mo', 'Ti'): -4,
    ('Mo', 'V'):  -1,  ('Mo', 'W'):  0,   ('Mo', 'Zr'): -6,
    # ── Nb pairs ──────────────────────────────────────────────────────────
    ('Nb', 'Ni'): -30, ('Nb', 'Re'): -10, ('Nb', 'Si'): -56,
    ('Nb', 'Ta'): 0,   ('Nb', 'Ti'): 2,   ('Nb', 'V'):  -1,
    ('Nb', 'W'):  -3,  ('Nb', 'Zr'): 4,
    # ── Ni pairs ──────────────────────────────────────────────────────────
    ('Ni', 'Si'): -40, ('Ni', 'Ta'): -29, ('Ni', 'Ti'): -35,
    ('Ni', 'V'):  -18, ('Ni', 'W'):  -3,  ('Ni', 'Zr'): -49,
    # ── Re pairs ──────────────────────────────────────────────────────────
    ('Re', 'Ta'): -8,  ('Re', 'W'):  -3,
    # ── Si pairs ──────────────────────────────────────────────────────────
    ('Si', 'Ta'): -54, ('Si', 'Ti'): -66, ('Si', 'V'):  -32,
    ('Si', 'W'):  -30, ('Si', 'Zr'): -84,
    # ── Ta pairs ──────────────────────────────────────────────────────────
    ('Ta', 'Ti'): 1,   ('Ta', 'V'):  -1,  ('Ta', 'W'):  -2,
    ('Ta', 'Zr'): 3,
    # ── Ti / V / W / Zr pairs ─────────────────────────────────────────────
    ('Ti', 'V'):  -2,  ('Ti', 'W'):  -7,  ('Ti', 'Zr'): 0,
    ('V',  'W'):  -1,  ('V',  'Zr'): -4,
    ('W',  'Zr'): -8,
}


# ── Formula parsing ───────────────────────────────────────────────────────────

def parse_formula(formula: str) -> dict:
    """
    Parse an alloy formula string into a dict of {element: molar_fraction}.

    Handles formats like:
      'Al0.25 Co1 Fe1 Ni1'   → molar ratios then normalized
      'Al20Co20Cr20Fe20Ni20' → already atomic percent
    """
    formula = str(formula).strip()

    # Match element + optional numeric coefficient
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)

    composition = {}
    for element, amount in matches:
        if element not in ELEMENT_DATA:
            continue  # skip elements we don't have data for
        amount = float(amount) if amount else 1.0
        composition[element] = composition.get(element, 0) + amount

    if not composition:
        return {}

    # Normalize to molar fractions (sum = 1)
    total = sum(composition.values())
    return {el: amt / total for el, amt in composition.items()}


# ── Descriptor computation ────────────────────────────────────────────────────

def get_hmix(pair: tuple) -> float:
    """
    Return mixing enthalpy for an element pair (kJ/mol).

    Checks both orderings of the pair. Returns np.nan if the pair
    is not in the table — the caller must handle this.

    IMPORTANT: Never default to 0. A missing pair with H=0 would
    incorrectly imply ideal mixing, corrupting ΔHmix and Ω.
    With the extended H_MIX table, all pairs in the Borg dataset
    are now covered. np.nan is a safety net for future datasets.
    """
    val = H_MIX.get(pair, H_MIX.get((pair[1], pair[0]), np.nan))
    return val


def compute_descriptors(composition: dict) -> dict:
    """
    Compute the 6 standard HEA thermodynamic / physical descriptors.

    Returns dict with keys:
      delta_S  — mixing entropy (J/mol·K)
      delta_H  — mixing enthalpy (kJ/mol) — nan if any pair missing
      delta    — atomic size mismatch (%)
      VEC      — valence electron concentration
      delta_X  — electronegativity difference
      omega    — stability parameter (Tm_avg * delta_S / |delta_H|)

    These six descriptors are the standard HEA feature set used across
    the literature (Zhang et al. 2008, Guo et al. 2011, Yeh et al. 2004).
    They are interpretable, physically motivated, and directly understood
    by the LLM reasoning component of the hybrid loop.

    nan propagation: if any binary pair is missing from H_MIX,
    delta_H and omega are set to nan. The pipeline drops these rows.
    This is safer than silently defaulting to 0 (previous behavior).

    Note: Tm_avg is computed internally for omega but not exposed as a
    standalone feature, keeping the descriptor set parsimonious.
    """
    elements = list(composition.keys())
    fracs    = np.array([composition[e] for e in elements])
    n        = len(elements)
    R        = 8.314

    # ── Mixing entropy ΔSmix = -R Σ xi·ln(xi) ────────────────────────────
    delta_S = -R * np.sum(fracs * np.log(fracs + 1e-12))

    # ── Mixing enthalpy ΔHmix — propagate nan on missing pairs ───────────
    delta_H = 0.0
    has_nan = False
    for i in range(n):
        for j in range(i + 1, n):
            h_ij = get_hmix((elements[i], elements[j]))
            if np.isnan(h_ij):
                has_nan = True
                break
            delta_H += 4 * h_ij * fracs[i] * fracs[j]
        if has_nan:
            break
    if has_nan:
        delta_H = np.nan

    # ── Atomic size mismatch δ ────────────────────────────────────────────
    radii  = np.array([ELEMENT_DATA[e]['r'] for e in elements])
    r_avg  = np.sum(fracs * radii)
    delta  = np.sqrt(np.sum(fracs * (1 - radii / r_avg) ** 2)) * 100

    # ── VEC ───────────────────────────────────────────────────────────────
    vecs   = np.array([ELEMENT_DATA[e]['VEC'] for e in elements])
    VEC    = np.sum(fracs * vecs)

    # ── Electronegativity difference Δχ ──────────────────────────────────
    X_vals  = np.array([ELEMENT_DATA[e]['X'] for e in elements])
    X_avg   = np.sum(fracs * X_vals)
    delta_X = np.sqrt(np.sum(fracs * (X_vals - X_avg) ** 2))

    # ── Average melting point (internal — used for omega only) ───────────
    Tm_vals = np.array([ELEMENT_DATA[e]['Tm'] for e in elements])
    Tm_avg  = np.sum(fracs * Tm_vals)

    # ── Ω stability parameter ─────────────────────────────────────────────
    if np.isnan(delta_H):
        omega = np.nan
    else:
        omega = (Tm_avg * delta_S) / (abs(delta_H) * 1000 + 1e-12)

    def safe_round(v, d):
        return round(float(v), d) if not np.isnan(v) else np.nan

    return {
        'delta_S': safe_round(delta_S, 4),
        'delta_H': safe_round(delta_H, 4),
        'delta'  : safe_round(delta,   4),
        'VEC'    : safe_round(VEC,     4),
        'delta_X': safe_round(delta_X, 4),
        'omega'  : safe_round(omega,   4),
    }


# ── Stability filter ──────────────────────────────────────────────────────────

def passes_stability(descriptors: dict) -> bool:
    """
    Apply empirical HEA stability criteria (Zhang et al. criteria).

    A composition is considered likely to form a stable solid solution if:
      - delta  < 6.6  (atomic size mismatch not too large)
      - omega  > 1.1  (entropy stabilization dominates enthalpy)
      - -22 <= delta_H <= 7  (moderate mixing enthalpy)
    """
    return (
        descriptors['delta']   <  6.6  and
        descriptors['omega']   >  1.1  and
        descriptors['delta_H'] >= -22  and
        descriptors['delta_H'] <= 7
    )


# ── Matminer / Magpie extended features ──────────────────────────────────────

def compute_magpie_features(formulas: pd.Series) -> pd.DataFrame:
    """
    Compute 132 Magpie features for each alloy formula using Matminer.

    Magpie (Materials Agnostic Platform for Informatics and Exploration)
    computes statistics (min, max, range, mean, avg_dev, mode) over
    22 elemental properties for each composition:
      - Atomic number, Mendeleev number, atomic weight
      - Melting point, boiling point, molar volume
      - Electronegativity, valence electrons, ionization energy
      - Atomic radius, bulk modulus, cohesive energy
      - And more...

    These capture physical intuition beyond the six classical descriptors
    and typically improve surrogate model R2 significantly.

    Parameters
    ----------
    formulas : pd.Series of formula strings (e.g. 'Al0.25 Co1 Fe1 Ni1')

    Returns
    -------
    DataFrame of shape (n_samples, 132) with Magpie feature columns.
    NaN rows indicate formulas that failed featurization.
    """
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except ImportError:
        raise ImportError(
            "matminer not installed. Run: "
            "pip install matminer --break-system-packages"
        )

    import warnings
    warnings.filterwarnings('ignore')

    ep = ElementProperty.from_preset('magpie')
    ep.set_n_jobs(1)

    feature_rows   = []
    failed_indices = []

    for idx, formula in formulas.items():
        try:
            # Normalize: 'Al0.25 Co1 Fe1 Ni1' -> 'Al0.25Co1Fe1Ni1'
            formula_clean = str(formula).replace(' ', '')
            comp          = Composition(formula_clean)
            features      = ep.featurize(comp)
            feature_rows.append(features)
        except Exception:
            # Append NaNs for failed rows — will be dropped downstream
            feature_rows.append([np.nan] * len(ep.feature_labels()))
            failed_indices.append(idx)

    if failed_indices:
        print(f"         Magpie: {len(failed_indices)} formulas failed "
              f"featurization -> will be dropped")

    df_magpie = pd.DataFrame(
        feature_rows,
        columns = ep.feature_labels(),
        index   = formulas.index,
    )

    return df_magpie


def compare_feature_sets(
    df_classical:           pd.DataFrame,
    df_magpie:              pd.DataFrame,
    feature_cols_classical: list,
    feature_cols_magpie:    list,
):
    """
    Compare surrogate model R2 between classical (6) and Magpie (132)
    feature sets using 5-fold cross-validated Random Forest.

    Prints a summary table and returns scores for both sets.
    """
    from sklearn.ensemble        import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing   import StandardScaler
    from sklearn.pipeline        import Pipeline

    print("\n" + "=" * 55)
    print("  Feature Set Comparison — 5-fold CV R2")
    print("=" * 55)

    results = {}

    for name, df, cols in [
        ('Classical (6 features)',   df_classical, feature_cols_classical),
        ('Magpie   (132 features)',  df_magpie,    feature_cols_magpie),
    ]:
        df_clean = df.dropna(subset=cols + ['HV'])
        X = df_clean[cols].values
        y = df_clean['HV'].values

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf',     RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ))
        ])

        scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
        results[name] = scores
        print(f"  {name:<30}  R2 = {scores.mean():.4f} +/- {scores.std():.4f}")

    improvement = (results['Magpie   (132 features)'].mean() -
                   results['Classical (6 features)'].mean())
    print(f"\n  Magpie improvement over classical: {improvement:+.4f} R2")
    print("=" * 55)

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    raw_path:        str  = '../data/raw/mpea_dataset.csv',
    processed_path:  str  = '../data/processed/hea_hardness.csv',
    filter_cast:     bool = True,
    apply_stability: bool = True,
    min_elements:    int  = 4,
    add_magpie:      bool = False,
    magpie_path:     str  = '../data/processed/hea_hardness_magpie.csv',
) -> pd.DataFrame:
    """
    Full data pipeline from raw Borg CSV to clean feature matrix.

    Parameters
    ----------
    raw_path        : path to raw Borg dataset CSV
    processed_path  : where to save the classical-descriptor dataset
    filter_cast     : if True, keep only as-cast samples for consistency
    apply_stability : if True, apply empirical stability filter
    min_elements    : minimum number of elements in alloy
    add_magpie      : if True, also generate 132 Magpie features and
                      save a second dataset at magpie_path
    magpie_path     : save path for extended Magpie feature dataset

    Returns
    -------
    df_clean : DataFrame with composition columns + descriptors + HV target
    """

    print("=" * 55)
    print("  HEA Data Pipeline — Borg et al. (2020)")
    print("=" * 55)

    # ── Step 1: Load raw data ─────────────────────────────────────────────
    df = pd.read_csv(raw_path)
    print(f"\nStep 1 — Loaded raw dataset:       {len(df):>5} rows")

    # ── Step 2: Filter for hardness entries ───────────────────────────────
    df = df[df['PROPERTY: HV'].notna()].copy()
    print(f"Step 2 — Rows with hardness (HV):  {len(df):>5} rows")

    # ── Step 3: Keep as-cast only (optional) ─────────────────────────────
    if filter_cast:
        df = df[df['PROPERTY: Processing method'] == 'CAST'].copy()
        print(f"Step 3 — As-cast only:             {len(df):>5} rows")
    else:
        print(f"Step 3 — Processing filter skipped:{len(df):>5} rows")

    # ── Step 4: Parse formulas ────────────────────────────────────────────
    print("\nStep 4 — Parsing formulas...")
    parsed = df['FORMULA'].apply(parse_formula)
    valid  = parsed.apply(lambda x: len(x) >= min_elements)
    df     = df[valid].copy()
    parsed = parsed[valid]
    print(f"         Formulas with >= {min_elements} known elements: {len(df):>4} rows")

    # ── Step 5: Build composition columns ────────────────────────────────
    print("Step 5 — Building composition matrix...")
    all_elements = sorted(set(
        el for comp in parsed for el in comp.keys()
    ))
    print(f"         Unique elements found: {all_elements}")

    comp_df = pd.DataFrame(
        [{el: comp.get(el, 0.0) for el in all_elements} for comp in parsed],
        index=df.index
    )

    # ── Step 6: Compute descriptors ───────────────────────────────────────
    print("Step 6 — Computing thermodynamic descriptors...")
    desc_list = [compute_descriptors(comp) for comp in parsed]
    desc_df   = pd.DataFrame(desc_list, index=df.index)

    # ── Step 7: Assemble full feature matrix ──────────────────────────────
    df_out = pd.concat([
        comp_df,
        desc_df,
        df[['PROPERTY: HV']].rename(columns={'PROPERTY: HV': 'HV'}),
        df[['FORMULA']].reset_index(drop=False).set_index('index')[['FORMULA']],
    ], axis=1)

    df_out = df_out.dropna(subset=['HV'])

    # ── Drop rows with nan descriptors (missing H_MIX pairs) ─────────────
    desc_cols   = ['delta_S', 'delta_H', 'delta', 'VEC', 'delta_X', 'omega']
    n_before    = len(df_out)
    df_out      = df_out.dropna(subset=[c for c in desc_cols if c in df_out.columns])
    n_dropped   = n_before - len(df_out)
    if n_dropped > 0:
        print(f"         Dropped {n_dropped} rows with nan descriptors "
              f"(missing H_MIX pairs — safer than defaulting to 0)")

    print(f"Step 7 — Feature matrix assembled: {len(df_out):>4} rows, "
          f"{len(df_out.columns)} columns")

    # ── Step 8: Stability filter ──────────────────────────────────────────
    if apply_stability:
        stable_mask = desc_df.apply(
            lambda row: passes_stability(row.to_dict()), axis=1
        )
        n_before = len(df_out)
        df_out   = df_out[stable_mask.values]
        print(f"Step 8 — After stability filter:   {len(df_out):>4} rows "
              f"({n_before - len(df_out)} removed)")
    else:
        print(f"Step 8 — Stability filter skipped: {len(df_out):>4} rows")

    # ── Step 9: Save classical dataset ───────────────────────────────────
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_out.to_csv(processed_path, index=False)
    print(f"\nStep 9 — Saved classical dataset to: {processed_path}")

    # ── Step 10: Magpie extended features (optional) ──────────────────────
    if add_magpie:
        print("\nStep 10 — Computing Magpie features (132 features)...")
        df_magpie_features = compute_magpie_features(df_out['FORMULA'])

        # Drop rows where Magpie featurization failed
        valid_magpie = df_magpie_features.notna().all(axis=1)
        n_dropped    = (~valid_magpie).sum()

        if n_dropped > 0:
            print(f"          Dropping {n_dropped} rows that failed Magpie featurization")

        # Combine: composition cols + classical descriptors + Magpie + HV + FORMULA
        df_magpie_out = pd.concat([
            df_out[valid_magpie].drop(columns=['HV', 'FORMULA']),
            df_magpie_features[valid_magpie],
            df_out.loc[valid_magpie, ['HV', 'FORMULA']],
        ], axis=1)

        # Remove any duplicate column names (safety check)
        df_magpie_out = df_magpie_out.loc[
            :, ~df_magpie_out.columns.duplicated()
        ]

        os.makedirs(os.path.dirname(magpie_path), exist_ok=True)
        df_magpie_out.to_csv(magpie_path, index=False)

        n_magpie_features = len(df_magpie_out.columns) - 2  # excl HV, FORMULA
        print(f"          Magpie dataset: {len(df_magpie_out)} rows, "
              f"{n_magpie_features} features")
        print(f"          Saved to: {magpie_path}")

        # Quick feature set comparison
        classical_cols = [c for c in df_out.columns
                          if c not in ['HV', 'FORMULA']]
        magpie_cols    = [c for c in df_magpie_out.columns
                          if c not in ['HV', 'FORMULA']]

        compare_feature_sets(
            df_classical           = df_out,
            df_magpie              = df_magpie_out,
            feature_cols_classical = classical_cols,
            feature_cols_magpie    = magpie_cols,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Final dataset summary")
    print("=" * 55)
    print(f"  Samples       : {len(df_out)}")
    print(f"  Features      : {len(df_out.columns) - 2}")  # excl HV + FORMULA
    print(f"  Elements      : {all_elements}")
    print(f"  HV range      : {df_out['HV'].min():.1f} — {df_out['HV'].max():.1f}")
    print(f"  HV mean ± std : {df_out['HV'].mean():.1f} ± {df_out['HV'].std():.1f}")
    print("=" * 55)

    return df_out


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Run 1: Classical descriptors (filtered) ───────────────────────────
    print("\n>>> Building classical descriptor dataset (filtered)...")
    df = run_pipeline(
        raw_path        = '../data/raw/mpea_dataset.csv',
        processed_path  = '../data/processed/hea_hardness.csv',
        filter_cast     = True,
        apply_stability = True,
        min_elements    = 4,
        add_magpie      = False,
    )

    # ── Run 2: Classical + Magpie (filtered) ─────────────────────────────
    print("\n\n>>> Building Magpie extended dataset (filtered)...")
    df_magpie = run_pipeline(
        raw_path        = '../data/raw/mpea_dataset.csv',
        processed_path  = '../data/processed/hea_hardness.csv',
        filter_cast     = True,
        apply_stability = True,
        min_elements    = 4,
        add_magpie      = True,
        magpie_path     = '../data/processed/hea_hardness_magpie.csv',
    )

    # ── Run 3: Classical + Magpie (unfiltered) ────────────────────────────
    print("\n\n>>> Building Magpie extended dataset (unfiltered)...")
    df_magpie_unfiltered = run_pipeline(
        raw_path        = '../data/raw/mpea_dataset.csv',
        processed_path  = '../data/processed/hea_hardness_unfiltered.csv',
        filter_cast     = True,
        apply_stability = False,
        min_elements    = 4,
        add_magpie      = True,
        magpie_path     = '../data/processed/hea_hardness_magpie_unfiltered.csv',
    )

    print("\n\n>>> All datasets built successfully.")
    print("Available datasets:")
    print("  data/processed/hea_hardness.csv                      <- classical, filtered")
    print("  data/processed/hea_hardness_magpie.csv               <- magpie,    filtered")
    print("  data/processed/hea_hardness_unfiltered.csv           <- classical, unfiltered")
    print("  data/processed/hea_hardness_magpie_unfiltered.csv    <- magpie,    unfiltered")
