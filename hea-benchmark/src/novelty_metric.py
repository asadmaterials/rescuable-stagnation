"""
Novelty Metric — Mahalanobis distance in descriptor space
=========================================================
Fixes issue A2 from the pre-run review.

Previous metric: Euclidean distance on raw composition fractions.
Problems (flagged by both reviewers):
  - ignores simplex geometry (a 0.05 shift in Al ≠ 0.05 shift in W)
  - ignores chemistry: two compositionally distant alloys can be
    thermodynamically near-identical, and vice versa
  - "novel ≠ scientifically meaningful"

New metric: Mahalanobis distance in the 6-D classical descriptor space
(ΔSmix, ΔHmix, δ, VEC, Δχ, Ω). Two alloys are "far apart" when their
THERMODYNAMIC/ELECTRONIC character differs, not merely their fraction
vectors. The covariance matrix is estimated ONCE from the working
dataset, so the metric is fixed across all arms and runs (fairness).

The reference set for novelty is (observed ∪ current pool) — everything
the search could already reach — so a candidate identical to an
unqueried pool point correctly scores ~0.

The old composition-Euclidean metric is retained as
composition_euclidean() for supplementary reporting/continuity.
"""

import numpy as np
from canonical_oracle import DESCRIPTOR_COLS, get_composition_cols


def descriptor_covariance(dataset_X: np.ndarray, feature_cols: list):
    """
    Estimate the descriptor-space covariance from the working dataset
    and return its (regularized) inverse.

    Called ONCE per experiment; the same cov_inv is used by every arm
    and every seed so the novelty scale is fixed and fair.

    Parameters
    ----------
    dataset_X    : (n, n_features) full working-dataset feature matrix
    feature_cols : canonical feature columns

    Returns
    -------
    cov_inv : (6, 6) inverse covariance of the descriptor columns
    """
    desc_idx = [feature_cols.index(c) for c in DESCRIPTOR_COLS]
    D        = dataset_X[:, desc_idx]
    cov      = np.cov(D, rowvar=False)
    # Ridge regularization for numerical stability (descriptors correlate)
    cov     += 1e-6 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    return np.linalg.inv(cov)


def mahalanobis_novelty(
    x:            np.ndarray,
    reference_X:  np.ndarray,
    feature_cols: list,
    cov_inv:      np.ndarray,
) -> float:
    """
    Novelty of candidate x = minimum Mahalanobis distance (descriptor
    space) to any point in reference_X.

    Higher = more thermodynamically/electronically distinct from
    everything the search can already reach. ~0 = duplicate in
    descriptor terms.

    Parameters
    ----------
    x            : candidate feature vector (n_features,)
    reference_X  : (n, n_features) points to measure distance against
                   (observed points ∪ current pool)
    feature_cols : canonical feature columns
    cov_inv      : from descriptor_covariance()

    Returns
    -------
    float — min Mahalanobis distance in descriptor space
    """
    if len(reference_X) == 0:
        return np.inf

    desc_idx = [feature_cols.index(c) for c in DESCRIPTOR_COLS]
    d        = reference_X[:, desc_idx] - x[desc_idx]      # (n, 6)
    # Mahalanobis: sqrt(d @ cov_inv @ d.T) rowwise
    m2       = np.einsum('ij,jk,ik->i', d, cov_inv, d)
    m2       = np.maximum(m2, 0.0)                          # numerical guard
    return float(np.sqrt(m2.min()))


def composition_euclidean(
    x:            np.ndarray,
    reference_X:  np.ndarray,
    feature_cols: list,
) -> float:
    """
    LEGACY metric (supplementary reporting only): minimum Euclidean
    distance on raw composition fractions. Kept for continuity with
    earlier diagnostics; NOT used for admission or headline analysis.
    """
    if len(reference_X) == 0:
        return np.inf
    comp_cols = get_composition_cols(feature_cols)
    comp_idx  = [feature_cols.index(c) for c in comp_cols]
    d         = np.linalg.norm(reference_X[:, comp_idx] - x[comp_idx], axis=1)
    return float(d.min())


def default_min_novelty(dataset_X: np.ndarray, feature_cols: list,
                        cov_inv: np.ndarray, percentile: float = 5.0) -> float:
    """
    Principled dedup threshold: the given percentile of nearest-neighbor
    Mahalanobis descriptor distances WITHIN the working dataset,
    EXCLUDING zero distances.

    Zero-distance pairs are repeat measurements of the same alloy
    (the Borg dataset contains several) — they reflect measurement
    replication, not the dataset's compositional granularity, so they
    are excluded before taking the percentile. Without this exclusion
    the threshold collapses to 0 and dedup becomes vacuous.

    Interpretation: a candidate is rejected as a near-duplicate only if
    it is closer to an existing point than the closest ~5% of DISTINCT
    dataset alloys are to each other.
    """
    desc_idx = [feature_cols.index(c) for c in DESCRIPTOR_COLS]
    D        = dataset_X[:, desc_idx]
    n        = len(D)
    nn_dists = np.empty(n)
    for i in range(n):
        d       = D - D[i]
        m2      = np.einsum('ij,jk,ik->i', d, cov_inv, d)
        m2[i]   = np.inf                    # exclude self
        nn_dists[i] = np.sqrt(max(m2.min(), 0.0))
    # Exclude repeat-measurement zeros before taking the percentile
    distinct = nn_dists[nn_dists > 1e-9]
    if len(distinct) == 0:
        return 1e-6
    return float(np.percentile(distinct, percentile))
