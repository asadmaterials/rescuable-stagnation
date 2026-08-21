"""
MP Oracle & Admission  —  foundation module for the shear-modulus rescue benchmark
==================================================================================
Domain-specific pieces the ported five-arm harness bolts onto. Everything
here replaces the HEA-specific canonical_oracle / admission / ss_strengthening
trio; the arms, shared digest, statistics, and instrumentation are reused
unchanged.

CONTENTS
    load_mp_dataset()      cached MP shear-modulus data, physical filter,
                           metallic >=3-element, polymorph collapse
    featurize()            Magpie composition descriptors (matminer)
    DATASET_ELEMENTS       the in-dataset element vocabulary (LLM proposals
                           are constrained to these — decision A)
    MPOracle               per-split, leakage-free RF on shear modulus G,
                           with the distance-based extrapolation indicator
    channel_b_vrh()        Voigt-Reuss-Hill physics proxy (decision D)
    admit_candidate()      shared admission for ALL arms, enforcing the
                           pre-registered distance rule (decision from the
                           threshold analysis)

PRE-REGISTERED ADMISSION RULE
    A candidate is admitted iff, in STANDARDISED Magpie-feature space, its
    Euclidean distance to the nearest TRAIN point is within the 90th
    percentile of the pool's train-distances (recomputed per split), AND it
    is not a near-duplicate of an already-observed/injected point. The 90th
    percentile keeps ~90% of the pool while excluding the extreme-
    extrapolation tail where oracle error rises (11.3 vs 9.9 GPa inside).
    Metric is Euclidean-to-nearest-train in scaled space — NOT Mahalanobis
    — matching the error-vs-distance analysis exactly (avoids the metric-
    mismatch bug class from the HEA work).
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

CACHE = 'mp_shear_metallic.csv'
G_MIN, G_MAX = 0.0, 600.0            # physical validity bounds
ADMISSION_PERCENTILE = 90            # pre-registered distance cutoff
DEDUP_TExOL = 1e-6                   # (set at runtime from data; see below)

# ── Magpie featurizer, built once ─────────────────────────────────────────
_EP = ElementProperty.from_preset('magpie')
FEATURE_COLS = _EP.feature_labels()


# ══════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════

def load_mp_dataset(path=CACHE):
    """Load the cached, cleaned MP shear-modulus dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Produce it once with the screening script "
            f"(MP_benchmark_screening.py) or copy it from Drive.")
    df = pd.read_csv(path)
    # enforce the physical filter in case an unfiltered CSV slips in
    df = df[(df['G'] > G_MIN) & (df['G'] < G_MAX)].reset_index(drop=True)
    return df


def featurize(df):
    """Add Magpie features; drop rows with any missing descriptor."""
    df = df.copy()
    df['comp_obj'] = df['formula'].apply(Composition)
    df = _EP.featurize_dataframe(df, 'comp_obj', ignore_errors=True)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def dataset_elements(df):
    """The in-dataset element vocabulary — LLM proposals are constrained to
    these (decision A). Prevents proposing elements the featurizer or oracle
    has never seen."""
    els = set()
    for f in df['formula']:
        for e in Composition(f).elements:
            els.add(str(e))
    return sorted(els)


# ══════════════════════════════════════════════════════════════════════════
# Channel B — physics proxy (VRH shear modulus from elemental moduli)
# ══════════════════════════════════════════════════════════════════════════

ELEM_G = {
    'Li':4.2,'Be':132,'Na':3.3,'Mg':17,'Al':26,'K':1.3,'Ca':7.4,'Sc':29,
    'Ti':44,'V':47,'Cr':115,'Mn':76,'Fe':82,'Co':75,'Ni':76,'Cu':48,'Zn':43,
    'Ga':6.7,'Rb':0.6,'Sr':6.1,'Y':26,'Zr':33,'Nb':38,'Mo':120,'Tc':123,
    'Ru':173,'Rh':150,'Pd':44,'Ag':30,'Cd':19,'In':4.4,'Sn':18,'Cs':0.65,
    'Ba':4.9,'La':14,'Ce':13.5,'Pr':14.8,'Nd':16.3,'Sm':19.5,'Eu':7.9,
    'Gd':21.8,'Tb':22.1,'Dy':24.7,'Ho':26.3,'Er':28.3,'Tm':30.5,'Yb':9.9,
    'Lu':27.2,'Hf':30,'Ta':69,'W':161,'Re':178,'Os':222,'Ir':210,'Pt':61,
    'Au':27,'Tl':2.8,'Pb':5.6,'Bi':12,
}


def channel_b_vrh(comp):
    """Voigt-Reuss-Hill G estimate from elemental moduli. Independent of the
    RF (never fitted). comp may be a dict {el: frac} or a formula string.
    Returns None if any element lacks a tabulated modulus."""
    if isinstance(comp, str):
        comp = {str(e): a for e, a in Composition(comp).fractional_composition.items()}
    else:
        tot = sum(comp.values())
        comp = {k: v / tot for k, v in comp.items()}
    gv = gr = 0.0
    for el, x in comp.items():
        g = ELEM_G.get(str(el))
        if g is None or g <= 0:
            return None
        gv += x * g
        gr += x / g
    if gr <= 0:
        return None
    return 0.5 * (gv + 1.0 / gr)


# ══════════════════════════════════════════════════════════════════════════
# Oracle  (per split, leakage-free)
# ══════════════════════════════════════════════════════════════════════════

class MPOracle:
    """
    RF oracle on shear modulus G, trained on the TRAIN split only.
    Provides:
        query(vec)          -> oracle G for a feature vector
        query_batch(X)      -> vectorised
        distance(vec)       -> Euclidean distance to nearest TRAIN point in
                               scaled feature space (extrapolation indicator)
        admission_threshold -> the pre-registered distance cutoff for this
                               split (90th percentile of pool distances)
    """

    def __init__(self, df, feature_cols, train_idx, pool_idx, seed=0):
        self.fc = feature_cols
        X = df[feature_cols].values
        y = df['G'].values
        self.scaler = StandardScaler().fit(X[train_idx])
        self._Xtr_s = self.scaler.transform(X[train_idx])
        self.rf = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                        random_state=seed, n_jobs=-1)
        self.rf.fit(self._Xtr_s, y[train_idx])

        # pre-register the per-split distance cutoff from POOL distances
        Xp_s = self.scaler.transform(X[pool_idx])
        _, dpool = pairwise_distances_argmin_min(Xp_s, self._Xtr_s)
        self.admission_threshold = float(np.percentile(dpool, ADMISSION_PERCENTILE))

    def query(self, vec):
        return float(self.rf.predict(self.scaler.transform(vec.reshape(1, -1)))[0])

    def query_batch(self, X):
        return self.rf.predict(self.scaler.transform(X))

    def distance(self, vec):
        v = self.scaler.transform(vec.reshape(1, -1))
        _, d = pairwise_distances_argmin_min(v, self._Xtr_s)
        return float(d[0])


# ══════════════════════════════════════════════════════════════════════════
# Admission — shared by ALL arms
# ══════════════════════════════════════════════════════════════════════════

def composition_to_vector(comp, feature_cols):
    """Composition dict -> Magpie feature vector. None if unfeaturizable."""
    total = sum(comp.values())
    if total <= 0:
        return None
    frac = {k: v / total for k, v in comp.items()}
    try:
        row = _EP.featurize(Composition(frac))
    except Exception:
        return None
    v = np.asarray(row, dtype=float)
    if v.shape[0] != len(feature_cols) or not np.all(np.isfinite(v)):
        return None
    return v


class AdmissionResult:
    __slots__ = ('admitted', 'reason', 'vec', 'distance')
    def __init__(self, admitted, reason, vec=None, distance=None):
        self.admitted, self.reason, self.vec, self.distance = \
            admitted, reason, vec, distance


def admit_candidate(comp, feature_cols, oracle, reference_scaled,
                    dedup_tol):
    """
    Shared admission for every arm. Steps:
      1. featurize (unfeaturizable -> reject)
      2. DISTANCE GATE: scaled Euclidean distance to nearest TRAIN point
         must be <= oracle.admission_threshold (the pre-registered rule)
      3. DEDUP: not a near-duplicate (scaled Euclidean) of any already
         observed/injected point
    reference_scaled : scaled feature matrix of observed ∪ pool ∪ batch,
                       for dedup (same space as the distance gate).
    """
    vec = composition_to_vector(comp, feature_cols)
    if vec is None:
        return AdmissionResult(False, 'unfeaturizable')

    dist = oracle.distance(vec)
    if dist > oracle.admission_threshold:
        return AdmissionResult(False, 'beyond_reliable_region',
                               vec=vec, distance=dist)

    v_s = oracle.scaler.transform(vec.reshape(1, -1))
    if len(reference_scaled):
        _, dmin = pairwise_distances_argmin_min(v_s, reference_scaled)
        if float(dmin[0]) < dedup_tol:
            return AdmissionResult(False, 'near_duplicate',
                                   vec=vec, distance=dist)
    return AdmissionResult(True, 'admitted', vec=vec, distance=dist)


# ══════════════════════════════════════════════════════════════════════════
# self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("MP oracle/admission self-test")
    df = featurize(load_mp_dataset())
    print(f"  dataset: {len(df)} compositions, {len(FEATURE_COLS)} features")
    els = dataset_elements(df)
    print(f"  element vocabulary ({len(els)}): {els}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    tr, pool = idx[:int(0.6*len(df))], idx[int(0.6*len(df)):]
    orc = MPOracle(df, FEATURE_COLS, tr, pool, seed=0)
    print(f"  oracle trained; admission threshold (90th pct) = "
          f"{orc.admission_threshold:.2f}")

    # admission sanity: an in-dataset composition vs a wild one
    good = {els[0]: 0.34, els[1]: 0.33, els[2]: 0.33}
    Xtr_s = orc.scaler.transform(df[FEATURE_COLS].values[tr])
    r = admit_candidate(good, FEATURE_COLS, orc, Xtr_s, dedup_tol=0.05)
    print(f"  sample admission: admitted={r.admitted} reason={r.reason} "
          f"dist={r.distance:.2f} (thresh {orc.admission_threshold:.2f})")

    # channel B sanity
    b = channel_b_vrh(good)
    print(f"  channel B VRH for sample: {b:.1f} GPa" if b else "  channel B: None")
    print("  self-test OK")
