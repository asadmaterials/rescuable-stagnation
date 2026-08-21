"""
Dual-Channel Oracle Agreement
=============================
Combines the two independent oracle channels:

    Channel A (RF)  : data-driven hardness prediction, trained on
                      Borg (later + Gorsse + Couzinié). Interpolates well
                      near training data, cannot extrapolate.
    Channel B (SSH) : physics-based Z-free solid-solution strengthening
                      proxy. Mechanistic, extrapolates, but sees ONLY the
                      SS component of strength.

The two channels are INDEPENDENT by construction — B is never fitted to
the data A is trained on. Their agreement is measured by Spearman's ρ
(rank correlation), which is invariant to B's undetermined Z constant.

Interpretation of ρ (this is an EMPIRICAL FINDING, not a target to tune):
    high ρ        → physics corroborates the RF ranking
    low/negative ρ → systematic disagreement, which means EITHER
                     the RF is extrapolation-driven OR hardness in this
                     dataset is dominated by non-SSH mechanisms the
                     physics channel cannot see.

Per-candidate disagreement is disambiguated using the RF's own
confidence flag (distance to training data):
    disagree + RF far from train  → trust B's skepticism (RF extrapolating)
    disagree + RF near train      → trust A (non-SSH mechanism at play)
"""

import numpy as np
from scipy.stats import spearmanr

from ss_strengthening import compute_ss_proxy
from canonical_oracle import get_composition_cols, DESCRIPTOR_COLS


def _vec_to_composition(x: np.ndarray, feature_cols: list) -> dict:
    """Extract {element: fraction} dict from a feature vector."""
    comp_cols = get_composition_cols(feature_cols)
    comp = {}
    for c in comp_cols:
        frac = x[feature_cols.index(c)]
        if frac > 1e-6:
            comp[c] = float(frac)
    return comp


def spearman_agreement(
    candidate_X:  np.ndarray,
    oracle,
    feature_cols: list,
    verbose:      bool = True,
) -> dict:
    """
    Compute Spearman ρ between RF-predicted HV and SSH proxy over a set
    of candidates.

    Candidates containing interstitials (SSH returns None) are excluded
    from the correlation and reported separately.

    Parameters
    ----------
    candidate_X  : (n, n_features) matrix of candidate feature vectors
    oracle       : CanonicalOracle (channel A)
    feature_cols : feature column names
    verbose      : print summary

    Returns
    -------
    dict with:
      rho, pvalue        — Spearman correlation and significance
      n_compared         — candidates with valid SSH proxy
      n_excluded         — candidates excluded (interstitials)
      rf_hv              — RF predictions for compared candidates
      ss_proxy           — SSH proxies for compared candidates
      confidence         — RF distance-to-train for compared candidates
    """
    rf_hv, ss_proxy, conf, excluded = [], [], [], 0

    for x in candidate_X:
        comp = _vec_to_composition(x, feature_cols)
        ss   = compute_ss_proxy(comp)
        if ss is None:            # interstitial → out of model
            excluded += 1
            continue
        rf_hv.append(oracle.query(x))
        ss_proxy.append(ss)
        conf.append(oracle.confidence(x))

    rf_hv    = np.array(rf_hv)
    ss_proxy = np.array(ss_proxy)
    conf     = np.array(conf)

    if len(rf_hv) < 3:
        rho, pval = np.nan, np.nan
    else:
        rho, pval = spearmanr(rf_hv, ss_proxy)

    if verbose:
        print(f"  Spearman agreement (RF vs SSH proxy):")
        print(f"    Candidates compared : {len(rf_hv)}")
        print(f"    Excluded (interstit): {excluded}")
        print(f"    Spearman ρ          : {rho:.4f}")
        print(f"    p-value             : {pval:.4g}")

    return {
        'rho'        : float(rho),
        'pvalue'     : float(pval),
        'n_compared' : len(rf_hv),
        'n_excluded' : excluded,
        'rf_hv'      : rf_hv,
        'ss_proxy'   : ss_proxy,
        'confidence' : conf,
    }


def flag_disagreements(
    agreement:      dict,
    conf_threshold: float = None,
    top_frac:       float = 0.3,
) -> list:
    """
    Identify candidates where the two channels disagree, and disambiguate
    the cause using the RF confidence flag.

    A candidate is "disagreement" if it ranks in the top fraction by one
    channel but the bottom fraction by the other.

    Parameters
    ----------
    agreement      : dict returned by spearman_agreement()
    conf_threshold : distance beyond which RF is "extrapolating".
                     If None, uses the median confidence of the set.
    top_frac       : fraction defining "top" / "bottom" ranks

    Returns
    -------
    list of dicts, one per flagged candidate:
      index, rf_rank, ss_rank, confidence, diagnosis
    """
    rf_hv    = agreement['rf_hv']
    ss_proxy = agreement['ss_proxy']
    conf     = agreement['confidence']
    n        = len(rf_hv)
    if n < 3:
        return []

    if conf_threshold is None:
        conf_threshold = float(np.median(conf))

    # Rank (0 = lowest). Convert to percentile.
    rf_pct = rf_hv.argsort().argsort() / (n - 1)
    ss_pct = ss_proxy.argsort().argsort() / (n - 1)

    flags = []
    for i in range(n):
        high_rf_low_ss = rf_pct[i] >= (1 - top_frac) and ss_pct[i] <= top_frac
        low_rf_high_ss = rf_pct[i] <= top_frac and ss_pct[i] >= (1 - top_frac)

        if high_rf_low_ss or low_rf_high_ss:
            far = conf[i] > conf_threshold
            if high_rf_low_ss and far:
                diag = ("RF rates high but physics low AND RF is far from "
                        "training data → likely RF EXTRAPOLATION ERROR")
            elif high_rf_low_ss and not far:
                diag = ("RF rates high, physics low, RF near training data "
                        "→ likely NON-SSH mechanism (RF may be right)")
            elif low_rf_high_ss and far:
                diag = ("Physics rates high but RF low, RF far from training "
                        "→ candidate may be underrated by RF (extrapolation)")
            else:
                diag = ("Physics rates high, RF low, RF near training "
                        "→ SSH high but other factors lower real hardness")

            flags.append({
                'index'      : i,
                'rf_hv'      : float(rf_hv[i]),
                'ss_proxy'   : float(ss_proxy[i]),
                'rf_pct'     : float(rf_pct[i]),
                'ss_pct'     : float(ss_pct[i]),
                'confidence' : float(conf[i]),
                'diagnosis'  : diag,
            })

    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER RUN — Borg-only oracle (to be repointed at merged oracle later)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import pandas as pd
    from canonical_oracle import make_splits, CanonicalOracle, get_feature_cols

    print("=" * 60)
    print("  Dual-Channel Agreement — PLACEHOLDER (Borg-only oracle)")
    print("=" * 60)

    from canonical_oracle import load_working_dataset
    df = load_working_dataset()
    fc = get_feature_cols(df)

    splits = make_splits(df, random_seed=42)
    oracle = CanonicalOracle(splits['train'], fc)

    # Compare channels over the hidden set + the full pool
    print("\n  --- Agreement over HIDDEN set ---")
    hidden_X = splits['hidden'][fc].values
    agr_hidden = spearman_agreement(hidden_X, oracle, fc)

    print("\n  --- Agreement over FULL dataset ---")
    full_X = df[fc].values
    agr_full = spearman_agreement(full_X, oracle, fc)

    # Flag disagreements on the full set
    print("\n  --- Disagreement flags (full set) ---")
    flags = flag_disagreements(agr_full)
    print(f"    {len(flags)} candidates flagged as channel disagreements")
    for f in flags[:5]:
        print(f"\n    Candidate {f['index']}:")
        print(f"      RF HV={f['rf_hv']:.0f} (pct {f['rf_pct']:.2f})  "
              f"SSH={f['ss_proxy']:.1f} (pct {f['ss_pct']:.2f})  "
              f"conf={f['confidence']:.2f}")
        print(f"      → {f['diagnosis']}")

    print("\n" + "=" * 60)
    print(f"  Hidden-set ρ : {agr_hidden['rho']:.4f}")
    print(f"  Full-set   ρ : {agr_full['rho']:.4f}")
    print("  (ρ is an empirical finding about the data, not a tuning target.)")
    print("=" * 60)
