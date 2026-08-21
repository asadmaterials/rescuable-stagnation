"""
Constrained Bayesian Optimization with Continuous Stability Score
=================================================================
Replaces the binary stability filter with a continuous stability score
incorporated directly into the BO acquisition function as a constraint.

Key idea:
  - Binary filter: hard threshold, removes 48 alloys entirely
  - Constrained BO: soft threshold, allows exploration near stability
    boundaries while penalizing clearly unstable compositions

Three components:
  1. StabilityScorer  — computes continuous score in [0,1] from descriptors
  2. ConstrainedGP    — GP surrogate modeling both HV and stability jointly
  3. run_constrained_bo_loop — full loop using ConstrainedExpectedImprovement

Ablation study: run with thresholds [0.4, 0.6, 0.7, 0.8] to show
sensitivity and find the optimal constraint tightness.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestRegressor
from gpytorch.mlls           import ExactMarginalLogLikelihood
from botorch.models          import SingleTaskGP
from botorch.fit             import fit_gpytorch_mll
from botorch.acquisition     import ConstrainedExpectedImprovement
from botorch.models.multitask import MultiTaskGP

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STABILITY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x: np.ndarray, steepness: float = 2.0) -> np.ndarray:
    """Smooth step function: 0 when x<<0, 0.5 at x=0, 1 when x>>0."""
    return 1.0 / (1.0 + np.exp(-steepness * np.clip(x, -500, 500)))


def compute_stability_score(
    delta:   np.ndarray,
    omega:   np.ndarray,
    delta_H: np.ndarray,
) -> np.ndarray:
    """
    Continuous stability score in [0, 1] based on Zhang et al. criteria.

    Maps the three binary Zhang thresholds to smooth sigmoid components,
    then combines them as a weighted average.

    Components and weights:
      delta  < 6.6   (weight 0.35) — atomic size mismatch
      omega  > 1.1   (weight 0.35) — entropy vs enthalpy balance
      dH > -22       (weight 0.15) — lower bound on mixing enthalpy
      dH <  7        (weight 0.15) — upper bound on mixing enthalpy

    delta and omega get higher weight because they are the most
    physically predictive of solid solution formation in HEA literature
    (Zhang et al. 2008, Guo et al. 2011).

    Parameters
    ----------
    delta   : atomic size mismatch (%)
    omega   : stability parameter
    delta_H : mixing enthalpy (kJ/mol)

    Returns
    -------
    score : np.ndarray in [0, 1], higher = more stable
    """
    # Clip omega to prevent numerical explosion from near-zero delta_H
    omega = np.clip(omega, 0, 50)

    s_delta = sigmoid(6.6  - delta,     steepness=2.0)
    s_omega = sigmoid(omega - 1.1,      steepness=1.5)
    s_dH_lo = sigmoid(delta_H - (-22),  steepness=1.0)
    s_dH_hi = sigmoid(7.0   - delta_H,  steepness=1.0)

    score = (0.35 * s_delta +
             0.35 * s_omega +
             0.15 * s_dH_lo +
             0.15 * s_dH_hi)

    return score


def add_stability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'stability_score' column to a processed HEA dataframe.
    Operates on the existing delta, omega, delta_H columns.

    Parameters
    ----------
    df : DataFrame with columns delta, omega, delta_H

    Returns
    -------
    df with new column 'stability_score'
    """
    df = df.copy()
    df['stability_score'] = compute_stability_score(
        delta   = df['delta'].values,
        omega   = df['omega'].clip(upper=50).values,
        delta_H = df['delta_H'].values,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DUAL ORACLE
# Models both HV and stability score simultaneously
# ═══════════════════════════════════════════════════════════════════════════════

class DualOracle:
    """
    Oracle that returns both hardness (HV) and stability score for
    any queried composition.

    HV oracle      : Random Forest trained on full dataset
    Stability score: computed analytically from descriptors (no ML needed)

    Parameters
    ----------
    df           : full processed HEA dataframe (with stability_score column)
    feature_cols : input feature columns (composition + descriptors)
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        self.query_count  = 0

        # HV oracle — Random Forest
        X = df[feature_cols].values
        y = df['HV'].values
        self.scaler_hv = StandardScaler()
        X_scaled       = self.scaler_hv.fit_transform(X)
        self.hv_model  = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
        self.hv_model.fit(X_scaled, y)

        # Stability oracle — RF on stability score (optional, for uncertainty)
        y_stab            = df['stability_score'].values
        self.scaler_stab  = StandardScaler()
        self.stab_model   = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
        self.stab_model.fit(X_scaled, y_stab)

        print(f"  DualOracle trained on {len(df)} samples.")

    def query(self, x: np.ndarray) -> tuple[float, float]:
        """
        Return (hv, stability_score) for feature vector x.

        Parameters
        ----------
        x : feature vector (n_features,)

        Returns
        -------
        (hv, stability_score) tuple
        """
        self.query_count += 1
        x_scaled = self.scaler_hv.transform(x.reshape(1, -1))
        hv       = float(self.hv_model.predict(x_scaled)[0])
        stab     = float(self.stab_model.predict(x_scaled)[0])
        stab     = float(np.clip(stab, 0.0, 1.0))
        return hv, stab


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DUAL GP SURROGATE
# Models HV and stability score as two separate GPs
# ═══════════════════════════════════════════════════════════════════════════════

class DualGPSurrogate:
    """
    Two independent GP surrogates: one for HV, one for stability score.

    ConstrainedExpectedImprovement requires separate GP models for the
    objective (HV) and the constraint (stability score).

    The constraint GP learns the stability landscape from queried points,
    allowing the acquisition function to identify regions that are both
    high-HV and sufficiently stable.
    """

    def __init__(self):
        self.hv_model    = None
        self.stab_model  = None
        self.scaler_X    = StandardScaler()
        self.scaler_hv   = StandardScaler()
        self.scaler_stab = StandardScaler()
        self.is_fit      = False

    def fit(self, X: np.ndarray, y_hv: np.ndarray, y_stab: np.ndarray):
        """
        Fit both GP models on current observations.

        Parameters
        ----------
        X      : feature matrix (n_obs, n_features)
        y_hv   : observed HV values (n_obs,)
        y_stab : observed stability scores (n_obs,)
        """
        X_scaled    = self.scaler_X.fit_transform(X)
        hv_scaled   = self.scaler_hv.fit_transform(
            y_hv.reshape(-1, 1)
        ).flatten()
        stab_scaled = self.scaler_stab.fit_transform(
            y_stab.reshape(-1, 1)
        ).flatten()

        train_X      = torch.tensor(X_scaled,    dtype=torch.float64)
        train_hv     = torch.tensor(hv_scaled,   dtype=torch.float64).unsqueeze(-1)
        train_stab   = torch.tensor(stab_scaled, dtype=torch.float64).unsqueeze(-1)

        # GP for HV
        self.hv_model = SingleTaskGP(train_X, train_hv)
        mll_hv        = ExactMarginalLogLikelihood(
            self.hv_model.likelihood, self.hv_model
        )
        fit_gpytorch_mll(mll_hv)
        self.hv_model.eval()

        # GP for stability score
        self.stab_model = SingleTaskGP(train_X, train_stab)
        mll_stab        = ExactMarginalLogLikelihood(
            self.stab_model.likelihood, self.stab_model
        )
        fit_gpytorch_mll(mll_stab)
        self.stab_model.eval()

        self.is_fit = True

    def predict_hv(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_hv, std_hv) in original HV units."""
        assert self.is_fit
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
        with torch.no_grad():
            post     = self.hv_model.posterior(X_tensor)
            mean_sc  = post.mean.numpy().flatten()
            std_sc   = post.variance.sqrt().numpy().flatten()
        mean = self.scaler_hv.inverse_transform(
            mean_sc.reshape(-1, 1)
        ).flatten()
        std  = std_sc * self.scaler_hv.scale_[0]
        return mean, std

    def predict_stab(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_stab, std_stab) in original [0,1] units."""
        assert self.is_fit
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
        with torch.no_grad():
            post     = self.stab_model.posterior(X_tensor)
            mean_sc  = post.mean.numpy().flatten()
            std_sc   = post.variance.sqrt().numpy().flatten()
        mean = self.scaler_stab.inverse_transform(
            mean_sc.reshape(-1, 1)
        ).flatten()
        std  = std_sc * self.scaler_stab.scale_[0]
        mean = np.clip(mean, 0.0, 1.0)
        return mean, std


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRAINED BO LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_constrained_bo_loop(
    df:                  pd.DataFrame,
    feature_cols:        list,
    stability_threshold: float = 0.6,
    n_initial:           int   = 20,
    n_iterations:        int   = 50,
    random_seed:         int   = 42,
    verbose:             bool  = True,
) -> dict:
    """
    Constrained Bayesian Optimization loop.

    Uses ConstrainedExpectedImprovement (CEI) which simultaneously:
      - Maximizes expected improvement in HV
      - Enforces stability_score >= stability_threshold

    The constraint is probabilistic — CEI naturally down-weights
    candidates that are likely to violate the stability constraint,
    while still allowing exploration near the boundary.

    Parameters
    ----------
    df                  : processed HEA dataframe WITH stability_score column
    feature_cols        : input feature columns
    stability_threshold : minimum acceptable stability score (ablation param)
    n_initial           : random initialization samples
    n_iterations        : BO iterations
    random_seed         : reproducibility
    verbose             : print per-iteration output

    Returns
    -------
    results dict with keys:
      observed_X, observed_y_hv, observed_y_stab
      best_history, stability_history
      surrogate, oracle
    """

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("\n" + "=" * 60)
    print(f"  Constrained BO Loop  (threshold = {stability_threshold})")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────────────────────
    oracle    = DualOracle(df, feature_cols)
    surrogate = DualGPSurrogate()
    pool      = df[feature_cols].values.copy()
    n_pool    = len(pool)

    queried_indices = set()

    # ── Initial random sampling ───────────────────────────────────────────
    print(f"\nInitializing with {n_initial} random samples...")
    init_idx = np.random.choice(n_pool, size=n_initial, replace=False)

    observed_X      = []
    observed_y_hv   = []
    observed_y_stab = []

    for idx in init_idx:
        x        = pool[idx]
        hv, stab = oracle.query(x)
        observed_X.append(x)
        observed_y_hv.append(hv)
        observed_y_stab.append(stab)
        queried_indices.add(idx)

    observed_X      = np.array(observed_X)
    observed_y_hv   = np.array(observed_y_hv)
    observed_y_stab = np.array(observed_y_stab)

    # Best feasible HV = best HV among stable candidates
    feasible_mask = observed_y_stab >= stability_threshold
    if feasible_mask.any():
        best_so_far = float(observed_y_hv[feasible_mask].max())
    else:
        best_so_far = float(observed_y_hv.max())

    best_history      = [best_so_far]
    stability_history = [float(observed_y_stab.mean())]

    print(f"  Initial best feasible HV : {best_so_far:.1f}")
    print(f"  Initial mean stability   : {observed_y_stab.mean():.3f}")

    # ── Constrained BO iterations ─────────────────────────────────────────
    print(f"\nRunning {n_iterations} constrained BO iterations...")
    print(f"{'Iter':>5}  {'Best HV':>8}  {'New HV':>8}  "
          f"{'Stab':>6}  {'Feasible':>8}")
    print("-" * 50)

    for iteration in range(1, n_iterations + 1):

        # Step 1: Fit both GP surrogates
        surrogate.fit(observed_X, observed_y_hv, observed_y_stab)

        # Step 2: Build candidate tensor
        unqueried_mask    = np.array(
            [i not in queried_indices for i in range(n_pool)]
        )
        unqueried_indices = np.where(unqueried_mask)[0]

        if len(unqueried_indices) == 0:
            print("  Pool exhausted.")
            break

        X_cand        = pool[unqueried_indices]
        X_cand_scaled = surrogate.scaler_X.transform(X_cand)
        X_cand_tensor = torch.tensor(X_cand_scaled, dtype=torch.float64)

        # Step 3: Constrained Expected Improvement
        # Scale best_f and constraint threshold to GP's internal scale
        best_f_scaled = float(
            surrogate.scaler_hv.transform([[best_so_far]])[0][0]
        )

        # Constraint: stability_score >= threshold
        # In scaled space: (threshold - mean) / scale
        constraint_scaled = float(
            surrogate.scaler_stab.transform([[stability_threshold]])[0][0]
        )

        # BoTorch CEI: constraints = {output_index: (lower, upper)}
        # index 0 = HV (objective), index 1 = stability (constraint)
        # We need to build a model list for CEI
        # Use a ModelList combining both GPs
        from botorch.models import ModelListGP
        from botorch.acquisition import ConstrainedExpectedImprovement

        model_list = ModelListGP(surrogate.hv_model, surrogate.stab_model)

        CEI = ConstrainedExpectedImprovement(
            model            = model_list,
            best_f           = best_f_scaled,
            objective_index  = 0,
            constraints      = {1: (constraint_scaled, None)},
        )

        with torch.no_grad():
            cei_values = CEI(X_cand_tensor.unsqueeze(1))

        # Step 4: Pick best candidate
        best_local  = int(torch.argmax(cei_values).item())
        best_global = unqueried_indices[best_local]

        # Step 5: Query oracle
        x_new       = pool[best_global]
        hv_new, stab_new = oracle.query(x_new)
        queried_indices.add(best_global)

        # Step 6: Update
        observed_X      = np.vstack([observed_X, x_new])
        observed_y_hv   = np.append(observed_y_hv, hv_new)
        observed_y_stab = np.append(observed_y_stab, stab_new)

        # Update best feasible HV
        feasible_mask = observed_y_stab >= stability_threshold
        if feasible_mask.any():
            best_so_far = float(observed_y_hv[feasible_mask].max())

        best_history.append(best_so_far)
        stability_history.append(float(observed_y_stab.mean()))

        if verbose:
            feasible_str = "✓" if stab_new >= stability_threshold else "✗"
            print(f"{iteration:>5}  {best_so_far:>8.1f}  {hv_new:>8.1f}  "
                  f"{stab_new:>6.3f}  {feasible_str:>8}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Constrained BO Complete  (threshold={stability_threshold})")
    print("=" * 60)
    feasible_mask = observed_y_stab >= stability_threshold
    print(f"  Total queries        : {oracle.query_count}")
    print(f"  Best feasible HV     : {best_so_far:.1f}")
    print(f"  Feasible fraction    : "
          f"{feasible_mask.mean()*100:.1f}% of all queries")
    print(f"  Mean stability score : {observed_y_stab.mean():.3f}")
    print("=" * 60)

    return {
        'observed_X'       : observed_X,
        'observed_y_hv'    : observed_y_hv,
        'observed_y_stab'  : observed_y_stab,
        'best_history'     : best_history,
        'stability_history': stability_history,
        'surrogate'        : surrogate,
        'oracle'           : oracle,
        'feature_cols'     : feature_cols,
        'threshold'        : stability_threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ABLATION RUNNER
# Runs constrained BO across multiple stability thresholds
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability_ablation(
    df:            pd.DataFrame,
    feature_cols:  list,
    thresholds:    list  = [0.0, 0.4, 0.6, 0.7, 0.8],
    n_initial:     int   = 20,
    n_iterations:  int   = 50,
    n_seeds:       int   = 5,
    save_dir:      str   = '../results',
) -> pd.DataFrame:
    """
    Run constrained BO across multiple stability thresholds and seeds.

    threshold=0.0 is the unconstrained baseline (equivalent to no filter).
    Higher thresholds enforce stricter stability requirements.

    Parameters
    ----------
    df            : processed HEA dataframe WITH stability_score column
    feature_cols  : input feature columns
    thresholds    : list of stability thresholds to ablate
    n_initial     : random init samples
    n_iterations  : BO iterations per run
    n_seeds       : random seeds for statistical robustness
    save_dir      : directory for saving results and plots

    Returns
    -------
    summary DataFrame with mean/std best HV per threshold
    """

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/figures', exist_ok=True)

    all_results  = {}
    summary_rows = []

    print("\n" + "=" * 60)
    print("  Stability Threshold Ablation Study")
    print(f"  Thresholds : {thresholds}")
    print(f"  Seeds      : {n_seeds}")
    print(f"  Iterations : {n_iterations}")
    print("=" * 60)

    for threshold in thresholds:
        print(f"\n{'─'*60}")
        print(f"  Running threshold = {threshold} ...")
        print(f"{'─'*60}")

        seed_best_hvs      = []
        seed_histories     = []

        for seed in range(n_seeds):
            results = run_constrained_bo_loop(
                df                  = df,
                feature_cols        = feature_cols,
                stability_threshold = threshold,
                n_initial           = n_initial,
                n_iterations        = n_iterations,
                random_seed         = seed,
                verbose             = False,
            )
            best_hv = max(results['best_history'])
            seed_best_hvs.append(best_hv)
            seed_histories.append(results['best_history'])
            print(f"    Seed {seed}: best HV = {best_hv:.1f}")

        all_results[threshold] = {
            'best_hvs'  : seed_best_hvs,
            'histories' : seed_histories,
        }

        mean_hv = np.mean(seed_best_hvs)
        std_hv  = np.std(seed_best_hvs)
        summary_rows.append({
            'threshold'   : threshold,
            'mean_best_HV': round(mean_hv, 2),
            'std_best_HV' : round(std_hv,  2),
            'label'       : 'Unconstrained' if threshold == 0.0
                            else f'Constrained (≥{threshold})',
        })

        print(f"  → Mean best HV: {mean_hv:.1f} ± {std_hv:.1f}")

    # ── Summary table ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{save_dir}/stability_ablation_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_ablation(all_results, summary_df, save_dir)

    return summary_df


def _plot_ablation(all_results: dict, summary_df: pd.DataFrame, save_dir: str):
    """
    Two-panel ablation plot:
      Left  — convergence curves per threshold (mean ± std shaded)
      Right — bar chart of final best HV per threshold
    """

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: convergence curves ──────────────────────────────────────────
    ax = axes[0]
    for i, (threshold, data) in enumerate(all_results.items()):
        histories = np.array(data['histories'])
        mean_hist = histories.mean(axis=0)
        std_hist  = histories.std(axis=0)
        iters     = np.arange(len(mean_hist))
        label     = ('Unconstrained' if threshold == 0.0
                     else f'Constrained ≥{threshold}')
        color     = colors[i % len(colors)]

        ax.plot(iters, mean_hist, color=color, linewidth=2.5,
                label=label, zorder=2)
        ax.fill_between(iters,
                        mean_hist - std_hist,
                        mean_hist + std_hist,
                        alpha=0.15, color=color)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Feasible HV', fontsize=12)
    ax.set_title('Convergence by Stability Threshold', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: bar chart ──────────────────────────────────────────────────
    ax2   = axes[1]
    x_pos = np.arange(len(summary_df))
    bars  = ax2.bar(
        x_pos,
        summary_df['mean_best_HV'],
        yerr   = summary_df['std_best_HV'],
        color  = colors[:len(summary_df)],
        alpha  = 0.85,
        capsize= 6,
        width  = 0.6,
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary_df['label'], rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Mean Best HV (± std)', fontsize=12)
    ax2.set_title('Final Performance by Threshold', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, row in zip(bars, summary_df.itertuples()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row.std_best_HV + 5,
            f'{row.mean_best_HV:.0f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    plt.suptitle('Stability Constraint Ablation — HEA Hardness Optimization',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = f'{save_dir}/figures/stability_ablation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Ablation plot saved to: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load unfiltered dataset + add stability score ─────────────────────
    DATA_PATH = '../data/processed/hea_hardness_unfiltered.csv'
    df        = pd.read_csv(DATA_PATH)
    df['omega'] = df['omega'].clip(upper=50)
    df        = add_stability_score(df)

    print(f"Loaded: {df.shape}")
    print(f"Stability score — mean: {df['stability_score'].mean():.3f}, "
          f"std: {df['stability_score'].std():.3f}")

    FEATURE_COLS = [c for c in df.columns
                    if c not in ['HV', 'FORMULA', 'stability_score']]

    # ── Quick single run to verify pipeline works ─────────────────────────
    print("\n>>> Single run verification (threshold=0.6)...")
    results = run_constrained_bo_loop(
        df                  = df,
        feature_cols        = FEATURE_COLS,
        stability_threshold = 0.6,
        n_initial           = 20,
        n_iterations        = 30,
        random_seed         = 42,
        verbose             = True,
    )
    print(f"\nBest feasible HV found: {max(results['best_history']):.1f}")

    # ── Full ablation study ───────────────────────────────────────────────
    print("\n>>> Running full stability ablation study...")
    summary = run_stability_ablation(
        df            = df,
        feature_cols  = FEATURE_COLS,
        thresholds    = [0.0, 0.4, 0.6, 0.7, 0.8],
        n_initial     = 20,
        n_iterations  = 30,
        n_seeds       = 5,
        save_dir      = '../results',
    )
