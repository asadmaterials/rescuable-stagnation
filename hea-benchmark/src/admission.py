"""
Candidate Admission — the SINGLE path for all injection arms
=============================================================
Fixes issues A1, A5, D2 from the pre-run review:

  A1 — Previously the LLM arm's candidates passed through a hard
       passes_stability() gate that random/mutation candidates never
       faced. That asymmetry could bias the experiment AGAINST the
       hypothesis arm. Now ALL arms use this one admission function,
       and stability is recorded as an ANNOTATION (soft score), never
       used as a gate.

  A5 — Previously mutation could inject near-duplicates of already-
       queried points (tiny σ perturbations of the incumbent). Now a
       minimum-distance check (Mahalanobis in descriptor space, see
       novelty_metric.py) rejects near-duplicates for ALL arms
       identically.

  D2 — Previously simplex/hygiene checks were applied inconsistently
       (LLM: yes, random/mutation: partially). Now one function does
       descriptor computation, nan check, simplex check, dedup check,
       and stability annotation for every candidate from every source.

INVARIANT: every arm's candidates pass through admit_candidate() and
nothing else. Admission criteria are IDENTICAL across arms. The only
thing that differs between arms is how candidates are GENERATED.
"""

import numpy as np

from data_pipeline     import compute_descriptors
from canonical_oracle  import DESCRIPTOR_COLS, get_composition_cols
from constrained_bo    import compute_stability_score
from novelty_metric    import mahalanobis_novelty


class AdmissionResult:
    """Outcome of admitting one candidate, with full diagnostics."""
    __slots__ = ('vec', 'admitted', 'reason', 'stability', 'novelty')

    def __init__(self, vec=None, admitted=False, reason='',
                 stability=np.nan, novelty=np.nan):
        self.vec       = vec
        self.admitted  = admitted
        self.reason    = reason
        self.stability = stability
        self.novelty   = novelty


def admit_candidate(
    comp_dict:     dict,
    feature_cols:  list,
    reference_X:   np.ndarray,
    cov_inv:       np.ndarray,
    min_novelty:   float,
    simplex_tol:   float = 1e-2,
) -> AdmissionResult:
    """
    The single admission function for ALL injection arms.

    Steps (identical for every arm):
      1. Compute the 6 classical descriptors from the composition
      2. Reject if any descriptor is nan (missing H_MIX pair)
      3. Assemble the full feature vector
      4. Reject if composition violates the simplex (sum≠1, negative)
      5. Reject if too close (Mahalanobis descriptor distance) to any
         point in reference_X (= observed ∪ current pool) — dedup
      6. Annotate with soft stability score (NEVER used as a gate)

    Parameters
    ----------
    comp_dict    : {element: molar fraction}
    feature_cols : canonical feature column list
    reference_X  : points the candidate must be distinct from
                   (observed points ∪ current pool), shape (n, n_features)
    cov_inv      : inverse covariance of descriptor space (from
                   novelty_metric.descriptor_covariance)
    min_novelty  : minimum Mahalanobis descriptor distance to admit
    simplex_tol  : tolerance on sum-to-one

    Returns
    -------
    AdmissionResult with .admitted, .vec, .reason, .stability, .novelty
    """
    # ── 1-2: descriptors + nan check ──────────────────────────────────────
    descriptors = compute_descriptors(comp_dict)
    if any(isinstance(v, float) and np.isnan(v) for v in descriptors.values()):
        return AdmissionResult(reason='nan_descriptor')

    # ── 3: assemble feature vector ────────────────────────────────────────
    vec = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        if col in comp_dict:
            vec[i] = comp_dict[col]
        elif col in descriptors:
            v = descriptors[col]
            vec[i] = min(v, 50.0) if col == 'omega' else v

    # ── 4: simplex check ──────────────────────────────────────────────────
    comp_cols   = get_composition_cols(feature_cols)
    comp_idx    = [feature_cols.index(c) for c in comp_cols]
    fracs       = vec[comp_idx]
    if np.any(fracs < -simplex_tol) or np.any(fracs > 1 + simplex_tol):
        return AdmissionResult(reason='simplex_range')
    if abs(fracs.sum() - 1.0) > simplex_tol:
        return AdmissionResult(reason='simplex_sum')

    # ── 5: dedup via Mahalanobis descriptor distance ──────────────────────
    novelty = mahalanobis_novelty(vec, reference_X, feature_cols, cov_inv)
    if novelty < min_novelty:
        return AdmissionResult(reason='near_duplicate', novelty=novelty)

    # ── 6: soft stability ANNOTATION (recorded, never gating) ─────────────
    stability = float(compute_stability_score(
        delta   = np.array([descriptors['delta']]),
        omega   = np.array([min(descriptors['omega'], 50.0)]),
        delta_H = np.array([descriptors['delta_H']]),
    )[0])

    return AdmissionResult(
        vec=vec, admitted=True, reason='ok',
        stability=stability, novelty=novelty,
    )
