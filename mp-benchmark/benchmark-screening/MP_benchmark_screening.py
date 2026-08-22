"""
MP SHEAR-MODULUS BENCHMARK SCREENING  —  Google Colab script
=============================================================
GO/NO-GO gate for a new rescue benchmark, run BEFORE building anything.

"""

# The earlier Borg/HEA benchmark produced an uninterpretable null.
# Retrospective ceiling analysis showed that the runs converged near the
# base-pool ceiling, leaving little opportunity for rescue to alter outcomes.
# Because trigger-time headroom was not recorded in those historical runs,
# the exact historical rescuable-event rate cannot be reconstructed.

# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — install (Colab)
# ══════════════════════════════════════════════════════════════════════════
# !pip install -q mp-api matminer pymatgen scikit-learn botorch gpytorch

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CACHE = 'mp_shear_metallic.csv'

# A stagnation event counts as RESCUABLE only if at least this much
# improvement remains. Set to ~1 oracle MAE after the oracle is trained;
# below the oracle's own error, "improvement" is not meaningful.
# Reported at two thresholds so conclusions do not hinge on one choice.
RESCUABLE_FRACTIONS = (0.5, 1.0)     # x oracle MAE


# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — pull MP elasticity data
# ══════════════════════════════════════════════════════════════════════════

METALS = set("""
Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru
Rh Pd Ag Cd In Sn Cs Ba La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re
Os Ir Pt Au Tl Pb Bi
""".split())


def fetch_elasticity(api_key):
    """Pull elasticity docs from MP. Prints coverage so gaps are visible."""
    from mp_api.client import MPRester
    print("Querying Materials Project elasticity...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.elasticity.search(
            fields=["material_id", "formula_pretty", "composition",
                    "bulk_modulus", "shear_modulus"]
        )
    print(f"  returned {len(docs)} elasticity documents")

    rows = []
    for d in docs:
        try:
            g = d.shear_modulus
            k = d.bulk_modulus
            # MP returns dicts of {voigt, reuss, vrh}
            g_vrh = getattr(g, 'vrh', None) if g is not None else None
            k_vrh = getattr(k, 'vrh', None) if k is not None else None
            if g_vrh is None:
                continue
            rows.append({
                'material_id': str(d.material_id),
                'formula'    : d.formula_pretty,
                'G'          : float(g_vrh),
                'K'          : float(k_vrh) if k_vrh is not None else np.nan,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    # Physical sanity filter: real shear moduli are ~0-600 GPa. MP contains
    # some entries with unphysical fitted elastic tensors (values off by many
    # orders of magnitude) that must be dropped before anything downstream.
    before = len(df)
    df = df[(df['G'] > 0) & (df['G'] < 600)].reset_index(drop=True)
    print(f"  dropped {before-len(df)} entries with unphysical G (<=0 or >=600 GPa)")
    print(f"  with valid shear modulus G: {len(df)}")
    print(f"  with valid bulk modulus K : {df['K'].notna().sum()}")
    print("  -> if these differ a lot, note it; K and G come from the same")
    print("     elastic tensor, so large divergence signals validity filtering.")
    return df


def is_metallic(formula, min_elements=3):
    """Metallic composition with >= min_elements distinct metals, no non-metals."""
    from pymatgen.core import Composition
    try:
        comp = Composition(formula)
    except Exception:
        return False
    els = [str(e) for e in comp.elements]
    if len(els) < min_elements:
        return False
    return all(e in METALS for e in els)


def build_dataset(api_key, min_elements=3):
    if os.path.exists(CACHE):
        print(f"Loading cached {CACHE}")
        return pd.read_csv(CACHE)

    df = fetch_elasticity(api_key)
    print(f"\nFiltering to metallic compositions with >= {min_elements} elements...")
    mask = df['formula'].apply(lambda f: is_metallic(f, min_elements))
    df = df[mask].reset_index(drop=True)
    print(f"  metallic multi-component entries: {len(df)}")

    # Deduplicate by composition: MP can hold several polymorphs of one
    # composition. A composition-only benchmark must pick ONE value per
    # composition; we take the max G (the best achievable for that chemistry),
    # which is the quantity an optimizer targeting composition would seek.
    from pymatgen.core import Composition
    df['reduced'] = df['formula'].apply(
        lambda f: Composition(f).reduced_formula)
    before = len(df)
    df = (df.sort_values('G', ascending=False)
            .drop_duplicates('reduced', keep='first')
            .reset_index(drop=True))
    print(f"  after collapsing polymorphs: {len(df)}  (from {before})")
    df.to_csv(CACHE, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — features (Magpie) + Channel B physics
# ══════════════════════════════════════════════════════════════════════════

def featurize(df):
    """Magpie composition descriptors. Covers the whole periodic table, which
    also fixes the Re/B/C descriptor gaps that silently discarded 40% of LLM
    proposals in the Borg run."""
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    print("\nFeaturizing (Magpie)...")
    df = df.copy()
    df['comp_obj'] = df['formula'].apply(Composition)
    ep = ElementProperty.from_preset('magpie')
    feats = ep.featurize_dataframe(df, 'comp_obj', ignore_errors=True)
    feat_cols = ep.feature_labels()
    feats = feats.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"  featurized rows: {len(feats)}   features: {len(feat_cols)}")
    return feats, feat_cols


# Elemental shear moduli (GPa), literature values. Channel B is a RANKING
# proxy built from these; it is never fitted to MP data.
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


def channel_b_vrh(formula):
    """
    Channel B — Voigt-Reuss-Hill estimate of G from ELEMENTAL moduli.

    Voigt (iso-strain) : G_V = sum(x_i * G_i)
    Reuss (iso-stress) : G_R = 1 / sum(x_i / G_i)
    VRH                : (G_V + G_R) / 2

    Independent of the RF by construction: no fitting, no MP data. Serves the
    same role Toda-Caraballo did for the HEA work — a physics cross-check that
    extrapolates, used for RANKING agreement (Spearman), not calibrated
    prediction. Returns None if any element lacks a tabulated modulus.
    """
    from pymatgen.core import Composition
    try:
        comp = Composition(formula).fractional_composition
    except Exception:
        return None
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
# CELL 4 — oracle + dual-channel agreement
# ══════════════════════════════════════════════════════════════════════════

def build_oracle(feats, feat_cols, seed=0):
    """Leakage-free RF: fit on TRAIN only, report honest OOS error."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    X = feats[feat_cols].values
    y = feats['G'].values
    idx = np.arange(len(feats))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=seed)
    val, hid = train_test_split(rest, test_size=0.5, random_state=seed)

    sc = StandardScaler().fit(X[tr])
    rf = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                               random_state=seed, n_jobs=-1)
    rf.fit(sc.transform(X[tr]), y[tr])
    pred = rf.predict(sc.transform(X[hid]))
    r2, mae = r2_score(y[hid], pred), mean_absolute_error(y[hid], pred)
    print(f"\nORACLE (Channel A): OOS R2={r2:.3f}  MAE={mae:.1f} GPa "
          f"(train={len(tr)}, pool={len(val)+len(hid)})")
    return dict(rf=rf, scaler=sc, X=X, y=y, train=tr, val=val, hidden=hid,
                r2=r2, mae=mae, feat_cols=feat_cols)


def dual_channel_check(feats):
    """Spearman rank agreement between RF target and the physics proxy."""
    from scipy.stats import spearmanr
    ssb = feats['formula'].apply(channel_b_vrh)
    ok = ssb.notna()
    rho, p = spearmanr(feats.loc[ok, 'G'], ssb[ok])
    print(f"\nCHANNEL B (VRH physics): computable for {ok.sum()}/{len(feats)} "
          f"({100*ok.mean():.0f}%)")
    print(f"  Spearman(G_MP, G_VRH) = {rho:.3f}  (p={p:.2e})")
    print("  Ranking agreement only — VRH is not calibrated to MP values.")
    return ssb, float(rho)


# ══════════════════════════════════════════════════════════════════════════
# CELL 5 — THE GATE: rescuable-stagnation probe
# ══════════════════════════════════════════════════════════════════════════

def bo_none_arm(oracle, budget, n_initial, seed, stagnation_window=5,
                stagnation_thresh=0.02):
    """
    Minimal no-injection BO run (GP + EI over the pool), instrumented for
    stagnation. Mirrors the 'none' arm of the rescue harness: this is the
    baseline whose behaviour determines whether rescue is testable at all.
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from sklearn.preprocessing import StandardScaler
    torch.set_default_dtype(torch.float64)

    rng = np.random.default_rng(seed)
    pool_idx = np.concatenate([oracle['val'], oracle['hidden']])
    Xp = oracle['X'][pool_idx]
    yp = oracle['rf'].predict(oracle['scaler'].transform(Xp))  # oracle labels

    init = rng.choice(len(pool_idx), size=n_initial, replace=False)
    obs = list(init)
    best_hist = [float(yp[obs].max())]
    trace = []

    for it in range(1, budget + 1):
        unq = np.array([i for i in range(len(pool_idx)) if i not in obs])
        if len(unq) == 0:
            break
        sx, sy = StandardScaler(), StandardScaler()
        Xs = sx.fit_transform(Xp[obs])
        ys = sy.fit_transform(yp[obs].reshape(-1, 1)).flatten()
        gp = SingleTaskGP(torch.tensor(Xs),
                          torch.tensor(ys).unsqueeze(-1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gp.eval()
        ei = ExpectedImprovement(gp, best_f=float(ys.max()))
        with torch.no_grad():
            scores = ei(torch.tensor(sx.transform(Xp[unq])).unsqueeze(1))
        pick = int(unq[int(torch.argmax(scores))])
        obs.append(pick)
        best_hist.append(float(yp[obs].max()))

        # stagnation detector (same rule as the rescue harness)
        stagnating = False
        if len(best_hist) > stagnation_window:
            w = best_hist[-(stagnation_window + 1):]
            denom = abs(w[0]) if abs(w[0]) > 1e-9 else 1.0
            stagnating = ((w[-1] - w[0]) / denom) < stagnation_thresh
        trace.append({'iter': it, 'stagnating': bool(stagnating),
                      'incumbent': best_hist[-1]})

    return dict(best_history=best_hist, trace=trace,
                ceiling=float(yp.max()), final=best_hist[-1])


def rescuable_probe(oracle, budgets=(20, 40, 60, 80), n_initial=10,
                    n_seeds=3):
    """
    THE GATE. Reports, per budget, how many stagnation events occur while
    real improvement is still available. Passing 'has headroom' and 'stalls
    sometimes' SEPARATELY is not sufficient — the run can do both while
    never once being stuck in a state a rescue could improve.
    """
    mae = oracle['mae']
    print("\n" + "=" * 72)
    print("  RESCUABLE-STAGNATION PROBE  (the go/no-go gate)")
    print("=" * 72)
    print(f"  Oracle MAE = {mae:.1f} GPa; rescuable thresholds tested at "
          f"{', '.join(f'{f}x MAE' for f in RESCUABLE_FRACTIONS)}")
    print(f"\n  {'budget':>7}{'final':>9}{'ceiling':>9}{'gap':>8}{'stagn':>7}"
          f"{'resc@0.5':>10}{'resc@1.0':>10}{'verdict':>13}")

    results = {}
    for b in budgets:
        finals, gaps, stag, r05, r10 = [], [], [], [], []
        for s in range(n_seeds):
            r = bo_none_arm(oracle, budget=b, n_initial=n_initial, seed=s)
            ceil = r['ceiling']
            finals.append(r['final']); gaps.append(ceil - r['final'])
            n_s = n_5 = n_1 = 0
            for t in r['trace']:
                if not t['stagnating']:
                    continue
                n_s += 1
                gap_now = ceil - t['incumbent']
                if gap_now >= 0.5 * mae: n_5 += 1
                if gap_now >= 1.0 * mae: n_1 += 1
            stag.append(n_s); r05.append(n_5); r10.append(n_1)

        m_stag, m_r05, m_r10 = np.mean(stag), np.mean(r05), np.mean(r10)
        if m_r10 >= 3:
            v = "GOOD"
        elif m_r10 >= 1.5 or m_r05 >= 3:
            v = "marginal"
        elif m_stag < 2:
            v = "never stalls"
        else:
            v = "SATURATED"

        results[b] = dict(final=float(np.mean(finals)), gap=float(np.mean(gaps)),
                          stagnation=float(m_stag), rescuable_05=float(m_r05),
                          rescuable_10=float(m_r10), verdict=v)
        print(f"  {b:>7}{np.mean(finals):>9.1f}{np.mean([ceil]):>9.1f}"
              f"{np.mean(gaps):>8.1f}{m_stag:>7.1f}{m_r05:>10.1f}"
              f"{m_r10:>10.1f}{v:>13}")

    good = [b for b, r in results.items() if r['verdict'] == 'GOOD']
    print("\n  " + "-" * 68)
    if good:
        rec = max(good, key=lambda b: results[b]['rescuable_10'])
        print(f"  GO — recommended budget {rec}: "
              f"{results[rec]['rescuable_10']:.1f} rescuable events/run.")
        print("  The loop stalls while real improvement remains: the rescue")
        print("  hypothesis is testable on this benchmark. Proceed to build.")
    else:
        print("  NO-GO at these settings — no budget yields enough rescuable")
        print("  stagnation. Before abandoning, try: larger pool (lower train")
        print("  fraction), smaller n_initial, or a harder property.")
        print("  If no setting works, that is itself a finding: well-posed BO")
        print("  benchmarks may simply not exhibit rescuable stagnation.")
    return results


# ══════════════════════════════════════════════════════════════════════════
# CELL 6 — run it
# ══════════════════════════════════════════════════════════════════════════

def main():
    key = os.environ.get('MP_API_KEY', '').strip()  # strip stray \r\n from copy-paste
    if not key:
        raise SystemExit(
            "MP_API_KEY not set. In Colab: sidebar key icon -> add MP_API_KEY, "
            "then:\n"
            "  from google.colab import userdata; import os\n"
            "  os.environ['MP_API_KEY'] = userdata.get('MP_API_KEY')")

    df = build_dataset(key, min_elements=3)
    if len(df) < 500:
        print(f"\n!! Only {len(df)} compositions. This is the Borg failure mode")
        print("   (too small a pool). Consider min_elements=2 to widen scope.")

    print(f"\nG distribution: min={df['G'].min():.1f} max={df['G'].max():.1f} "
          f"mean={df['G'].mean():.1f} GPa")
    print(f"  top-1% threshold: {df['G'].quantile(0.99):.1f} GPa")
    print("  (a long high-G tail is what makes the optimum hard to find;")
    print("   a tight distribution risks the 'never stalls' regime)")

    feats, feat_cols = featurize(df)
    dual_channel_check(feats)
    oracle = build_oracle(feats, feat_cols)
    results = rescuable_probe(oracle)

    with open('mp_screening_results.json', 'w') as f:
        json.dump({'n_compositions': len(df), 'oracle_r2': oracle['r2'],
                   'oracle_mae': oracle['mae'], 'probe': results}, f, indent=2)
    print("\nSaved -> mp_screening_results.json")


if __name__ == '__main__':
    main()
