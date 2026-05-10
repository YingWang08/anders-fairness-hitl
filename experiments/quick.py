"""
quick.py  (Revised v4 — Bias embedded in target via historical dependence)

Quick validation test.
Runs only bias=30% with 3 seeds.

Key change: target_allocation is now a weighted sum of base_allocation AND
historical_allocation, so the bias in historical_allocation propagates to
the target itself.  Models cannot achieve high accuracy without reproducing
this bias, guaranteeing DI < 1 for baselines.
"""

import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

QUICK_TEST = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_lr import BaselineLR
from src.models.baseline_dl import BaselineDL
from src.models.debiased_hitl import DebiasedHITL
from src.models.fe_hitl import FEHITL
from src.fairness_metrics import equal_opportunity_difference, average_odds_difference
from src.utils import compute_regression_metrics, set_seed


def safe_disparate_impact(y_pred_bin, s_unprivileged):
    prob_unpriv = np.mean(y_pred_bin[s_unprivileged == 1])
    prob_priv   = np.mean(y_pred_bin[s_unprivileged == 0])
    if prob_priv == 0:
        return 1.0
    return prob_unpriv / prob_priv


def generate_agricultural_data_biased(n_samples=10000, random_seed=42, bias_rate=0.30):
    """
    Generate simulated agricultural dataset.

    CRITICAL DESIGN:
    - base_allocation uses unbiased features.
    - historical_allocation = base_allocation * bias + noise  (biased for Region A).
    - target_allocation = 0.5 * base_allocation + 0.5 * historical_allocation + noise.
      → target inherits the bias through its dependence on historical_allocation.
    - historical_allocation is ALSO kept as an input feature, so the model is
      forced to deal with biased information to predict the biased target.
    """
    np.random.seed(random_seed)
    from sklearn.model_selection import train_test_split

    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg_raw     = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)

    region = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    # Biased yield (feature only)
    yield_bias = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    yield_3y_avg = yield_3y_avg_raw * yield_bias

    # Unbiased base allocation
    base_allocation = (0.3 * arable_land +
                       0.2 * labor_force +
                       0.1 * farming_years +
                       0.2 * yield_3y_avg_raw +
                       0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)

    # Biased historical allocation (feature)
    allocation_bias = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = (base_allocation * allocation_bias
                             + np.random.normal(0, 2, n_samples))
    historical_allocation = np.maximum(historical_allocation, 0)

    # ---- KEY CHANGE ----------------------------------------------------
    # Target is a mixture of unbiased base and biased historical,
    # so the bias is structurally embedded in what the model must predict.
    target_allocation = (0.5 * base_allocation +
                         0.5 * historical_allocation +
                         np.random.normal(0, 2, n_samples))
    target_allocation = np.maximum(target_allocation, 0)
    # --------------------------------------------------------------------

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
    feature_cols = ['arable_land', 'labor_force', 'farming_years',
                    'yield_3y_avg', 'irrigation_resources', 'fertilizer_subsidy',
                    'historical_allocation']
    X = df[feature_cols].values
    y = df['target_allocation'].values
    s = (df['region'] == 'A').astype(int).values   # 1 = unprivileged
    return X, y, s


def run_one_bias_level(bias_rate: float, seed: int = 42) -> dict:
    set_seed(seed)

    train_df, val_df, test_df = generate_agricultural_data_biased(
        n_samples=10000, random_seed=seed, bias_rate=bias_rate
    )

    X_train, y_train, s_train = split_arrays(train_df)
    X_val,   y_val,   s_val   = split_arrays(val_df)
    X_test,  y_test,  s_test  = split_arrays(test_df)

    threshold = np.median(y_train)

    def binarize(y):
        return (y > threshold).astype(int)

    avg_alloc_B = np.mean(y_train[s_train == 0])
    context_list = [{'region': r, 'avg_allocation_B': avg_alloc_B}
                    for r in test_df['region'].values]

    results = {}

    # LR
    lr = BaselineLR(task='regression')
    lr.fit(X_train, y_train)
    p = lr.predict(X_test)
    results['LR'] = dict(
        R2   = compute_regression_metrics(y_test, p)[0],
        RMSE = compute_regression_metrics(y_test, p)[1],
        DI   = safe_disparate_impact(binarize(p), s_test),
        EOD  = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD  = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    # DL
    dl = BaselineDL(
        input_dim=X_train.shape[1],
        task='regression',
        hidden_dims=[128, 64, 32],
        device='cpu',
        random_seed=seed
    )
    dl.fit(X_train, y_train, X_val, y_val)
    p = dl.predict(X_test)
    results['DL'] = dict(
        R2   = compute_regression_metrics(y_test, p)[0],
        RMSE = compute_regression_metrics(y_test, p)[1],
        DI   = safe_disparate_impact(binarize(p), s_test),
        EOD  = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD  = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    # Debiased-HITL
    dh = DebiasedHITL(
        input_dim=X_train.shape[1],
        task='regression',
        device='cpu',
        random_seed=seed,
        di_threshold=0.8,
        eod_threshold=0.1,
        intervention_ratio=0.1
    )
    dh.set_binarize_threshold(threshold)
    dh.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    p = dh.predict(X_test, s_test, context_list=context_list, apply_intervention=True)
    results['Debiased-HITL'] = dict(
        R2   = compute_regression_metrics(y_test, p)[0],
        RMSE = compute_regression_metrics(y_test, p)[1],
        DI   = safe_disparate_impact(binarize(p), s_test),
        EOD  = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD  = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    # FE-HITL
    fh = FEHITL(
        input_dim=X_train.shape[1],
        task='regression',
        device='cpu',
        random_seed=seed,
        epsilon=0.3,
        di_threshold=0.8,
        eod_threshold=0.1
    )
    fh.fit_base(X_train, y_train, X_val, y_val)
    p = fh.predict_with_intervention(X_test, s_test, context_list=context_list)
    results['FE-HITL'] = dict(
        R2   = compute_regression_metrics(y_test, p)[0],
        RMSE = compute_regression_metrics(y_test, p)[1],
        DI   = safe_disparate_impact(binarize(p), s_test),
        EOD  = equal_opportunity_difference(binarize(y_test), binarize(p), s_test),
        AOD  = average_odds_difference(binarize(y_test), binarize(p), s_test),
    )

    return results


def main():
    BIAS_LEVELS = [0.30]
    SEEDS = list(range(42, 45))
    prefix = "quick_"
    mode_str = "QUICK TEST (bias=30%, 3 seeds) — BIAS IN TARGET"

    all_rows = []

    print("Sensitivity Analysis — varying historical bias injection rate")
    print("Mode:", mode_str)
    print("=" * 70)

    for bias in BIAS_LEVELS:
        pct = int(bias * 100)
        print(f"\n  Bias level = {pct}%  (Region A penalty = {pct}%)")

        for seed in SEEDS:
            model_metrics = run_one_bias_level(bias_rate=bias, seed=seed)

            for model_name, m in model_metrics.items():
                row = {
                    'bias_rate_pct': pct,
                    'seed':          seed,
                    'model':         model_name,
                    'R2':            round(m['R2'],   4),
                    'RMSE':          round(m['RMSE'], 4),
                    'DI':            round(m['DI'],   4),
                    'EOD':           round(abs(m['EOD']), 4),
                    'AOD':           round(abs(m['AOD']), 4),
                }
                all_rows.append(row)

    raw_file = f"{prefix}sensitivity_results.csv"
    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(raw_file, index=False)
    print(f"\nRaw results saved → {raw_file}  ({len(df_raw)} rows)")

    summary = df_raw.groupby(['bias_rate_pct', 'model']).agg(
        R2_mean   = ('R2', 'mean'),
        R2_std    = ('R2', 'std'),
        RMSE_mean = ('RMSE', 'mean'),
        RMSE_std  = ('RMSE', 'std'),
        DI_mean   = ('DI', 'mean'),
        DI_std    = ('DI', 'std'),
        EOD_mean  = ('EOD', 'mean'),
        EOD_std   = ('EOD', 'std'),
        AOD_mean  = ('AOD', 'mean'),
        AOD_std   = ('AOD', 'std'),
        n         = ('R2', 'count'),
    ).reset_index()

    stats_file = f"{prefix}sensitivity_summary_stats.csv"
    summary.to_csv(stats_file, index=False)
    print(f"Summary stats saved → {stats_file}")

    lines = []
    lines.append("Sensitivity Analysis: FE-HITL Performance (Bias in Target)")
    lines.append("(Mean ± SD over {} seeds)".format(len(SEEDS)))
    lines.append("=" * 85)
    header = (f"{'Bias':>6}  {'Model':<15}  {'R²':>12}  {'RMSE':>12}  "
              f"{'DI':>12}  {'|EOD|':>12}  {'|AOD|':>12}")
    lines.append(header)
    lines.append("-" * 85)

    for _, row in summary.iterrows():
        bias_str = f"{int(row.bias_rate_pct)}%"
        r2_str   = f"{row.R2_mean:.3f}±{row.R2_std:.3f}"
        rmse_str = f"{row.RMSE_mean:.2f}±{row.RMSE_std:.2f}"
        di_str   = f"{row.DI_mean:.3f}±{row.DI_std:.3f}"
        eod_str  = f"{row.EOD_mean:.3f}±{row.EOD_std:.3f}"
        aod_str  = f"{row.AOD_mean:.3f}±{row.AOD_std:.3f}"
        lines.append(f"{bias_str:>6}  {row.model:<15}  {r2_str:>12}  "
                     f"{rmse_str:>12}  {di_str:>12}  {eod_str:>12}  {aod_str:>12}")

    lines.append("=" * 85)
    lines.append("NOTE: target_allocation = 0.5*base + 0.5*historical + noise.")
    lines.append("Expected: DL DI ≈ 0.60–0.70 (model forced to learn bias).")
    summary_text = "\n".join(lines)

    txt_file = f"{prefix}sensitivity_summary.txt"
    with open(txt_file, 'w',encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Text summary saved → {txt_file}")
    print("\n" + summary_text)


if __name__ == '__main__':
    main()