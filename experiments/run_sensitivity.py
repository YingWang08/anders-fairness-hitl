"""
run_sensitivity.py
==================
Sensitivity Analysis for W2 (Reviewer #2):
Reruns the agricultural experiment across four bias injection rates
(10%, 20%, 30%, 40%) with a fixed random seed to test whether the
FE-HITL framework's fairness improvements are robust to the assumed
magnitude of historical bias, rather than an artifact of the specific
30% penalty used in the original data-generation process.

Usage:
    python run_sensitivity.py

Output:
    sensitivity_results.csv   — per-bias-level metrics
    sensitivity_summary.txt   — human-readable table (paste into paper / response letter)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── project imports (adjust paths if needed) ──────────────────────────────────
from src.models.baseline_lr import BaselineLR
from src.models.baseline_dl import BaselineDL
from src.models.debiased_hitl import DebiasedHITL
from src.models.fe_hitl import FEHITL
from src.fairness_metrics import (disparate_impact,
                                   equal_opportunity_difference,
                                   average_odds_difference)
from src.utils import compute_regression_metrics, set_seed


# ── inline data generator (parameterised bias rate) ───────────────────────────
def generate_agricultural_data_biased(n_samples=10000,
                                       random_seed=42,
                                       bias_rate=0.30):
    """
    Generate the simulated agricultural dataset with a configurable
    historical allocation penalty for Region A.

    Parameters
    ----------
    bias_rate : float
        Fraction by which Region A allocations are reduced.
        E.g. 0.30 means Region A receives (1 - 0.30) = 70 % of baseline.
    """
    np.random.seed(random_seed)
    from sklearn.model_selection import train_test_split

    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg         = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)

    region = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    base_allocation = (0.3 * arable_land +
                       0.2 * labor_force +
                       0.1 * farming_years +
                       0.2 * yield_3y_avg +
                       0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)

    # ← only this line changes across sensitivity levels
    allocation_bias = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = (base_allocation * allocation_bias
                             + np.random.normal(0, 2, n_samples))
    historical_allocation = np.maximum(historical_allocation, 0)

    target_allocation = base_allocation + np.random.normal(0, 1, n_samples)
    target_allocation = np.maximum(target_allocation, 0)

    df = pd.DataFrame({
        'arable_land':          arable_land,
        'labor_force':          labor_force,
        'farming_years':        farming_years,
        'yield_3y_avg':         yield_3y_avg,
        'irrigation_resources': irrigation_resources,
        'fertilizer_subsidy':   fertilizer_subsidy,
        'region':               region,
        'historical_allocation': historical_allocation,
        'target_allocation':    target_allocation,
    })

    train, temp = train_test_split(df, test_size=0.30, random_state=random_seed)
    val,   test = train_test_split(temp, test_size=0.50, random_state=random_seed)
    return train, val, test


def split_arrays(df):
    """Return (X, y, s) numpy arrays from a dataframe split.

    historical_allocation is included as a feature so that the injected
    regional bias (controlled by bias_rate) actually enters the model's
    input distribution — making the sensitivity sweep meaningful.
    Without it, bias_rate only affects a column that is never read.
    """
    feature_cols = ['arable_land', 'labor_force', 'farming_years',
                    'yield_3y_avg', 'irrigation_resources', 'fertilizer_subsidy',
                    'historical_allocation']   # <- bias_rate flows in via this column
    X = df[feature_cols].values
    y = df['target_allocation'].values
    s = (df['region'] == 'A').astype(int).values   # 1 = Region A (unprivileged)
    return X, y, s


# ── per-bias-level experiment ──────────────────────────────────────────────────
def run_one_bias_level(bias_rate: float, seed: int = 42) -> dict:
    """
    Train all four models on data generated with `bias_rate` and return
    a dict of metrics for each model.
    """
    set_seed(seed)

    train_df, val_df, test_df = generate_agricultural_data_biased(
        n_samples=10000, random_seed=seed, bias_rate=bias_rate
    )

    (X_train, y_train, s_train) = split_arrays(train_df)
    (X_val,   y_val,   s_val)   = split_arrays(val_df)
    (X_test,  y_test,  s_test)  = split_arrays(test_df)

    threshold = np.median(y_train)

    def binarize(y):
        return (y > threshold).astype(int)

    avg_alloc_B = np.mean(y_train[s_train == 0])
    context_list = [{'region': r, 'avg_allocation_B': avg_alloc_B}
                    for r in test_df['region'].values]

    results = {}

    # ── 1. Linear Regression baseline ─────────────────────────────────────────
    lr = BaselineLR(task='regression')
    lr.fit(X_train, y_train)
    p = lr.predict(X_test)
    results['LR'] = dict(
        R2  = compute_regression_metrics(y_test, p)[0],
        RMSE= compute_regression_metrics(y_test, p)[1],
        DI  = disparate_impact(binarize(p), s_test),
        EOD = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    # ── 2. Deep-Learning baseline ──────────────────────────────────────────────
    dl = BaselineDL(input_dim=X_train.shape[1], task='regression',
                    device='cpu', random_seed=seed)
    dl.fit(X_train, y_train, X_val, y_val)
    p = dl.predict(X_test)
    results['DL'] = dict(
        R2  = compute_regression_metrics(y_test, p)[0],
        RMSE= compute_regression_metrics(y_test, p)[1],
        DI  = disparate_impact(binarize(p), s_test),
        EOD = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    # ── 3. Debiased-HITL ──────────────────────────────────────────────────────
    dh = DebiasedHITL(input_dim=X_train.shape[1], task='regression',
                      device='cpu', random_seed=seed,
                      di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1)
    dh.set_binarize_threshold(threshold)
    dh.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    p = dh.predict(X_test, s_test, context_list=context_list, apply_intervention=True)
    results['Debiased-HITL'] = dict(
        R2  = compute_regression_metrics(y_test, p)[0],
        RMSE= compute_regression_metrics(y_test, p)[1],
        DI  = disparate_impact(binarize(p), s_test),
        EOD = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    # ── 4. FE-HITL ────────────────────────────────────────────────────────────
    fh = FEHITL(input_dim=X_train.shape[1], task='regression',
                device='cpu', random_seed=seed,
                epsilon=0.1, di_threshold=0.8, eod_threshold=0.1)
    fh.fit_base(X_train, y_train, X_val, y_val)
    p = fh.predict_with_intervention(X_test, s_test, context_list=context_list)
    results['FE-HITL'] = dict(
        R2  = compute_regression_metrics(y_test, p)[0],
        RMSE= compute_regression_metrics(y_test, p)[1],
        DI  = disparate_impact(binarize(p), s_test),
        EOD = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    return results


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    BIAS_LEVELS = [0.10, 0.20, 0.30, 0.40]   # 10 %, 20 %, 30 % (original), 40 %
    SEED        = 42

    all_rows = []

    print("\nSensitivity Analysis — varying historical bias injection rate")
    print("=" * 70)

    for bias in BIAS_LEVELS:
        pct = int(bias * 100)
        print(f"\n  Bias level = {pct}%  (Region A penalty = {pct}%)")
        model_metrics = run_one_bias_level(bias_rate=bias, seed=SEED)

        for model_name, m in model_metrics.items():
            row = {
                'bias_rate_pct': pct,
                'model':         model_name,
                'R2':            round(m['R2'],  3),
                'RMSE':          round(m['RMSE'], 3),
                'DI':            round(m['DI'],   3),
                'EOD':           round(abs(m['EOD']), 3),
                'AOD':           round(abs(m['AOD']), 3),
            }
            all_rows.append(row)
            print(f"    {model_name:<15}  R²={m['R2']:.3f}  DI={m['DI']:.3f}  "
                  f"EOD={abs(m['EOD']):.3f}  AOD={abs(m['AOD']):.3f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    csv_path = 'sensitivity_results.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Build a readable summary table ────────────────────────────────────────
    lines = []
    lines.append("Sensitivity Analysis: FE-HITL Performance across Bias Injection Rates")
    lines.append("=" * 72)
    lines.append(f"{'Bias':>6}  {'Model':<15}  {'R²':>6}  {'RMSE':>6}  "
                 f"{'DI':>6}  {'|EOD|':>6}  {'|AOD|':>6}")
    lines.append("-" * 72)
    for _, row in df_out.iterrows():
        lines.append(f"{str(row.bias_rate_pct)+'%':>6}  {row.model:<15}  "
                     f"{row.R2:6.3f}  {row.RMSE:6.3f}  "
                     f"{row.DI:6.3f}  {row.EOD:6.3f}  {row.AOD:6.3f}")
    lines.append("=" * 72)
    lines.append("Note: 30% is the bias level used in the main experiment.")
    summary_text = "\n".join(lines)

    txt_path = 'sensitivity_summary.txt'
    with open(txt_path, 'w' ,encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Summary table saved → {txt_path}")
    print("\n" + summary_text)


if __name__ == '__main__':
    main()