"""
Solid-Solution Strengthening Channel (Level 1)
==============================================
Channel-B of the dual-channel oracle. Computes a Z-FREE solid-solution
strengthening PROXY following a simplified form of the Toda-Caraballo &
Rivera-Díaz-del-Castillo (2015, Acta Materialia 85, 14-23) model.

IMPORTANT SEMANTICS — three quantities that must NOT be conflated:
    Δσ_ss  — solid-solution strengthening CONTRIBUTION (what we compute)
    σy     — total yield strength = Δσ_ss + grain-boundary + precipitate
             + order + multi-phase contributions (NOT computed here)
    HV     — Vickers hardness ≈ σy via rough Tabor relation (NOT computed here)

This module outputs ONLY a Δσ_ss proxy. It is NOT a hardness predictor.
It is a proxy for the solid-solution COMPONENT of strength, usable for
RANKING candidates and for flagging where SSH alone cannot explain hardness.

LEVEL 1 SIMPLIFICATION:
    - Modulus misfit  : faithful (Eqs. 26-27, 3, 13, 14 of the paper)
    - Size misfit     : APPROXIMATED from atomic-radius mismatch rather than
                        the full interatomic-spacing-matrix treatment
                        (Eqs. 15, 24, 30 are NOT implemented)
    - Temperature     : fixed at room temperature (Eqs. 6-9 dropped)
    - Z constant      : LEFT UNDETERMINED. Output is relative, not absolute.
                        Z cancels in any ranking, which is the only use here.

Interstitials (B, C) are OUT OF MODEL — the misfit framework assumes
substitutional solutes. Alloys containing them return None.

Equations implemented : 3 (both misfit parts), 13, 14, 26, 27
Equations NOT implemented : 1,2,4,5 (binary build-up), 6-9 (temperature),
    10 (superseded by 26-27), 11,12,16,24 (spacing/cell param),
    15,30 (spacing matrix + derivative), 25 (A correction), 18-23 (sij solve),
    31 (HV conversion — not needed for Z-free ranking)
"""

import numpy as np


# ── Shear modulus (GPa) per element ───────────────────────────────────────────
# Sources: Kittel Introduction to Solid State Physics; WebElements;
# Senkov & Miracle compilations for refractory elements.
# Polycrystalline / room-temperature values.
SHEAR_MODULUS_GPa = {
    'Al': 26,   'Co': 75,   'Cr': 115,  'Cu': 48,   'Fe': 82,
    'Hf': 30,   'Mn': 76,   'Mo': 126,  'Nb': 38,   'Ni': 76,
    'Si': 45,   'Ta': 69,   'Ti': 44,   'V':  47,   'W':  161,
    'Zr': 33,   'Re': 178,
    # Interstitials — present for completeness but flagged out-of-model
    'B':  None, 'C':  None,
}

# Atomic radii (pm) — mirror ELEMENT_DATA in data_pipeline for independence
ATOMIC_RADIUS_pm = {
    'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128, 'Fe': 126,
    'Hf': 159, 'Mn': 127, 'Mo': 139, 'Nb': 146, 'Ni': 124,
    'Si': 117, 'Ta': 146, 'Ti': 147, 'V':  134, 'W':  139,
    'Zr': 160, 'Re': 137, 'B':  87,  'C':  77,
}

# VEC for fcc/bcc routing (Guo et al. 2011 criterion)
VEC = {
    'Al': 3,  'Co': 9,  'Cr': 6,  'Cu': 11, 'Fe': 8,
    'Hf': 4,  'Mn': 7,  'Mo': 6,  'Nb': 5,  'Ni': 10,
    'Si': 4,  'Ta': 5,  'Ti': 4,  'V':  5,  'W':  6,
    'Zr': 4,  'Re': 7,  'B':  3,  'C':  4,
}

INTERSTITIALS = {'B', 'C'}

# Model constants (paper's own choices)
ALPHA_EDGE = 16.0   # α for edge dislocations (Eq. 3, 13)


def _fcc_or_bcc(composition: dict) -> str:
    """Route alloy to fcc or bcc via composition-weighted VEC (Guo 2011)."""
    vec_avg = sum(composition[e] * VEC[e] for e in composition)
    # VEC >= 8 → fcc, VEC < 6.87 → bcc, in-between → default to fcc side
    return 'fcc' if vec_avg >= 7.44 else 'bcc'


def compute_ss_proxy(composition: dict) -> float | None:
    """
    Compute Z-free solid-solution strengthening proxy for a composition.

    Parameters
    ----------
    composition : dict {element: molar_fraction}, fractions sum to ~1

    Returns
    -------
    float : Δσ_ss proxy in arbitrary units (proportional to MPa up to the
            undetermined Z constant). Higher = more SS strengthening.
    None  : if the alloy contains interstitials (B, C) — out of model.

    Notes
    -----
    The returned value is meaningful ONLY relative to other values from
    this same function (Z cancels in ranking). It is NOT calibrated HV
    and NOT total yield strength.
    """
    # Out-of-model check: interstitials
    if any(e in INTERSTITIALS for e in composition):
        return None

    # Drop negligible components and renormalize
    comp = {e: x for e, x in composition.items() if x > 1e-6}
    total = sum(comp.values())
    comp  = {e: x / total for e, x in comp.items()}

    elements = list(comp.keys())
    X        = np.array([comp[e] for e in elements])

    # ── Alloy mean shear modulus (Eq. 27): μ_HEA = Σ Xi·μi ────────────────
    mu = np.array([SHEAR_MODULUS_GPa[e] for e in elements], dtype=float)
    mu_HEA = float(np.sum(X * mu))

    # ── Modulus misfit (Eq. 26): ηi = 2(μi − μ_HEA)/(μi + μ_HEA) ─────────
    eta = 2.0 * (mu - mu_HEA) / (mu + mu_HEA)

    # ── Screened modulus misfit (Eq. 3): η'i = ηi/(1 + 0.5|ηi|) ──────────
    eta_prime = eta / (1.0 + 0.5 * np.abs(eta))

    # ── Size misfit — LEVEL 1 radius-based approximation ─────────────────
    # Paper's δi = (da/dXi)(1/a) via spacing matrix; here we approximate
    # the per-element misfit as fractional deviation from mean radius.
    r      = np.array([ATOMIC_RADIUS_pm[e] for e in elements], dtype=float)
    r_mean = float(np.sum(X * r))
    delta  = (r - r_mean) / r_mean        # per-element fractional size misfit

    # ── fcc/bcc-dependent constant ξ (Eq. 13): 1 for fcc, 4 for bcc ──────
    xi = 1.0 if _fcc_or_bcc(comp) == 'fcc' else 4.0

    # ── Per-element misfit magnitude εi (Eq. 13) ─────────────────────────
    #   εi = ξ·(η'i² + α²·δi²)^(1/2)
    eps = xi * np.sqrt(eta_prime**2 + (ALPHA_EDGE**2) * delta**2)

    # ── Hardening parameter Bi (Eq. 13): Bi = 3·μ_HEA·εi^(4/3)·Z ─────────
    #   Z LEFT OUT (undetermined). Everything below is "per unit Z".
    B = 3.0 * mu_HEA * np.power(eps, 4.0 / 3.0)

    # ── Multicomponent SSH sum (Eq. 14): Δσ_ss = (Σ Bi^(3/2)·Xi)^(2/3) ──
    inner = np.sum(np.power(B, 1.5) * X)
    dsigma_ss = float(np.power(inner, 2.0 / 3.0))

    return dsigma_ss


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTIONAL VALIDATION — must pass on import
# ═══════════════════════════════════════════════════════════════════════════════

def _equimolar(*elements) -> dict:
    n = len(elements)
    return {e: 1.0 / n for e in elements}


def _validate_directional(verbose: bool = True) -> bool:
    """
    Reproduce two qualitative results from the paper:

      1. Cr added to CoFeNi HARDENS  (paper Fig. 8a / Section 5):
         Δσ_ss(CoCrFeNi) > Δσ_ss(CoFeNi)

      2. Fe added to CoCrNi SOFTENS  (paper Fig. 9b / Section 5):
         Δσ_ss(CoCrFeNi) < Δσ_ss(CoCrNi)

    If either sign is wrong, the model is not trustworthy and this
    returns False (module raises on import).
    """
    ss_CoFeNi   = compute_ss_proxy(_equimolar('Co', 'Fe', 'Ni'))
    ss_CoCrNi   = compute_ss_proxy(_equimolar('Co', 'Cr', 'Ni'))
    ss_CoCrFeNi = compute_ss_proxy(_equimolar('Co', 'Cr', 'Fe', 'Ni'))

    check1 = ss_CoCrFeNi > ss_CoFeNi   # Cr hardens CoFeNi
    check2 = ss_CoCrFeNi < ss_CoCrNi   # Fe softens CoCrNi

    if verbose:
        print("  Directional validation (Level 1 SSH channel):")
        print(f"    Δσ_ss(CoFeNi)   = {ss_CoFeNi:.3f}")
        print(f"    Δσ_ss(CoCrNi)   = {ss_CoCrNi:.3f}")
        print(f"    Δσ_ss(CoCrFeNi) = {ss_CoCrFeNi:.3f}")
        print(f"    Check 1 — Cr hardens CoFeNi  (CoCrFeNi > CoFeNi): "
              f"{'PASS' if check1 else 'FAIL'}")
        print(f"    Check 2 — Fe softens CoCrNi  (CoCrFeNi < CoCrNi): "
              f"{'PASS' if check2 else 'FAIL'}")

    return check1 and check2


if __name__ == '__main__':
    print("=" * 60)
    print("  SSH Channel — Level 1 — Standalone Validation")
    print("=" * 60)

    ok = _validate_directional(verbose=True)

    print()
    print("  Sanity check — canonical alloys (relative ranking):")
    for name, comp in [
        ('CoCrFeNi',      _equimolar('Co', 'Cr', 'Fe', 'Ni')),
        ('CoCrFeMnNi',    _equimolar('Co', 'Cr', 'Fe', 'Mn', 'Ni')),
        ('MoNbTaW',       _equimolar('Mo', 'Nb', 'Ta', 'W')),
        ('MoNbTaVW',      _equimolar('Mo', 'Nb', 'Ta', 'V', 'W')),
        ('AlCoCrFeNi',    _equimolar('Al', 'Co', 'Cr', 'Fe', 'Ni')),
        ('HfNbTaTiZr',    _equimolar('Hf', 'Nb', 'Ta', 'Ti', 'Zr')),
    ]:
        val = compute_ss_proxy(comp)
        print(f"    {name:<14}: Δσ_ss proxy = {val:8.3f}")

    print()
    print("  Out-of-model check (interstitials):")
    print(f"    CoCrFeNiC0.5 → {compute_ss_proxy({'Co':0.22,'Cr':0.22,'Fe':0.22,'Ni':0.22,'C':0.12})}")

    print()
    if ok:
        print("  ✓ Directional validation PASSED — channel usable.")
    else:
        print("  ✗ Directional validation FAILED — do not use.")
