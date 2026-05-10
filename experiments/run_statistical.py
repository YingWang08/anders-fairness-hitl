"""
run_statistical.py
==================
Statistical Robustness Analysis for W3 (Reviewer #2):

Problems addressed
------------------
• Only 5 runs → upgrade to N_RUNS = 30 independent runs (different random seeds)
• No confidence intervals → report 95% CI for every metric
• No effect sizes → report Cohen's d (pairwise vs LR baseline) for key metrics
• No multiple-comparison correction → apply Bonferroni correction on all p-values
• No seed-sensitivity report → show per-seed variance summary

Datasets covered
----------------
• Agricultural (regression): R², RMSE, DI, |EOD|
• German Credit (classification): Accuracy, DI, |EOD|

Usage
-----
    python run_statistical.py

Outputs
-------
    statistical_agricultural.csv     — 30-run raw results (agricultural)
    statistical_credit.csv           — 30-run raw results (German Credit)
    statistical_summary.txt          — full formatted tables for paper / response letter
    ci_table_agricultural.csv        — mean ± 95%CI per metric per model
    ci_table_credit.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_agricultural_data, load_german_credit_data
from src.models.baseline_lr    import BaselineLR
from src.models.baseline_dl    import BaselineDL
from src.models.debiased_hitl  import DebiasedHITL
from src.models.fe_hitl        import FEHITL
from src.fairness_metrics      import (disparate_impact,
                                        equal_opportunity_difference,
                                        average_odds_difference)
from src.utils import (compute_regression_metrics,
                        compute_classification_metrics,
                        set_seed)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_RUNS   = 30
# 30 distinct seeds (reproducible; feel free to extend or shuffle)
SEEDS    = list(range(0, N_RUNS))
MODELS   = ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']
ALPHA    = 0.05   # family-wise error rate (before Bonferroni)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: statistics
# ═══════════════════════════════════════════════════════════════════════════════

def mean_ci(values, confidence=0.95):
    """Return (mean, lower_CI, upper_CI) using t-distribution."""
    arr = np.array(values, dtype=float)
    n   = len(arr)
    m   = np.mean(arr)
    se  = stats.sem(arr)
    h   = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return m, m - h, m + h


def cohens_d(a, b):
    """Cohen's d: (mean_a - mean_b) / pooled_std."""
    a, b = np.array(a, float), np.array(b, float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def bonferroni_correction(p_values, alpha=ALPHA):
    """Return Bonferroni-corrected threshold and whether each p passes."""
    k         = len(p_values)
    threshold = alpha / k
    return threshold, [p < threshold for p in p_values]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-seed runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_agricultural_seed(seed):
    set_seed(seed)

    (X_train, y_train, s_train), \
    (X_val,   y_val,   s_val), \
    (X_test,  y_test,  s_test) = load_agricultural_data()

    threshold  = np.median(y_train)
    binarize   = lambda y: (y > threshold).astype(int)
    avg_B      = np.mean(y_train[s_train == 0])

    import pandas as _pd
    test_df    = _pd.read_csv('data/agricultural_test.csv')
    ctx        = [{'region': r, 'avg_allocation_B': avg_B}
                  for r in test_df['region'].values]

    out = {}

    # LR
    m = BaselineLR(task='regression')
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    r2, rmse = compute_regression_metrics(y_test, p)
    out['LR'] = dict(R2=r2, RMSE=rmse,
                     DI=disparate_impact(binarize(p), s_test),
                     EOD=abs(equal_opportunity_difference(binarize(y_test), binarize(p), s_test)),
                     AOD=abs(average_odds_difference(binarize(y_test), binarize(p), s_test)))

    # DL
    m = BaselineDL(input_dim=X_train.shape[1], task='regression',
                   device='cpu', random_seed=seed)
    m.fit(X_train, y_train, X_val, y_val)
    p = m.predict(X_test)
    r2, rmse = compute_regression_metrics(y_test, p)
    out['DL'] = dict(R2=r2, RMSE=rmse,
                     DI=disparate_impact(binarize(p), s_test),
                     EOD=abs(equal_opportunity_difference(binarize(y_test), binarize(p), s_test)),
                     AOD=abs(average_odds_difference(binarize(y_test), binarize(p), s_test)))

    # Debiased-HITL
    m = DebiasedHITL(input_dim=X_train.shape[1], task='regression',
                     device='cpu', random_seed=seed,
                     di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1)
    m.set_binarize_threshold(threshold)
    m.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    p = m.predict(X_test, s_test, context_list=ctx, apply_intervention=True)
    r2, rmse = compute_regression_metrics(y_test, p)
    out['Debiased-HITL'] = dict(R2=r2, RMSE=rmse,
                     DI=disparate_impact(binarize(p), s_test),
                     EOD=abs(equal_opportunity_difference(binarize(y_test), binarize(p), s_test)),
                     AOD=abs(average_odds_difference(binarize(y_test), binarize(p), s_test)))

    # FE-HITL
    m = FEHITL(input_dim=X_train.shape[1], task='regression',
               device='cpu', random_seed=seed,
               epsilon=0.1, di_threshold=0.8, eod_threshold=0.1)
    m.fit_base(X_train, y_train, X_val, y_val)
    p = m.predict_with_intervention(X_test, s_test, context_list=ctx)
    r2, rmse = compute_regression_metrics(y_test, p)
    out['FE-HITL'] = dict(R2=r2, RMSE=rmse,
                     DI=disparate_impact(binarize(p), s_test),
                     EOD=abs(equal_opportunity_difference(binarize(y_test), binarize(p), s_test)),
                     AOD=abs(average_odds_difference(binarize(y_test), binarize(p), s_test)))

    return out


def run_credit_seed(seed):
    set_seed(seed)

    (X_train, y_train, s_train), \
    (X_val,   y_val,   s_val), \
    (X_test,  y_test,  s_test) = load_german_credit_data()

    ctx = [{} for _ in range(len(X_test))]
    out = {}

    # LR
    m = BaselineLR(task='classification')
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    out['LR'] = dict(Acc=compute_classification_metrics(y_test, p),
                     DI=disparate_impact(p, s_test),
                     EOD=abs(equal_opportunity_difference(y_test, p, s_test)),
                     AOD=abs(average_odds_difference(y_test, p, s_test)))

    # DL
    m = BaselineDL(input_dim=X_train.shape[1], task='classification',
                   device='cpu', random_seed=seed)
    m.fit(X_train, y_train, X_val, y_val)
    p = m.predict(X_test)
    out['DL'] = dict(Acc=compute_classification_metrics(y_test, p),
                     DI=disparate_impact(p, s_test),
                     EOD=abs(equal_opportunity_difference(y_test, p, s_test)),
                     AOD=abs(average_odds_difference(y_test, p, s_test)))

    # Debiased-HITL
    m = DebiasedHITL(input_dim=X_train.shape[1], task='classification',
                     device='cpu', random_seed=seed,
                     di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1)
    m.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    p = m.predict(X_test, s_test, context_list=ctx, apply_intervention=True)
    out['Debiased-HITL'] = dict(Acc=compute_classification_metrics(y_test, p),
                     DI=disparate_impact(p, s_test),
                     EOD=abs(equal_opportunity_difference(y_test, p, s_test)),
                     AOD=abs(average_odds_difference(y_test, p, s_test)))

    # FE-HITL
    m = FEHITL(input_dim=X_train.shape[1], task='classification',
               device='cpu', random_seed=seed,
               epsilon=0.1, di_threshold=0.8, eod_threshold=0.1)
    m.fit_base(X_train, y_train, X_val, y_val)
    p = m.predict_with_intervention(X_test, s_test, context_list=ctx)
    out['FE-HITL'] = dict(Acc=compute_classification_metrics(y_test, p),
                     DI=disparate_impact(p, s_test),
                     EOD=abs(equal_opportunity_difference(y_test, p, s_test)),
                     AOD=abs(average_odds_difference(y_test, p, s_test)))

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collect_runs(runner_fn, seeds):
    """Run runner_fn for each seed; return {model: {metric: [values]}}."""
    accumulated = {m: {} for m in MODELS}
    for i, seed in enumerate(seeds):
        print(f"    run {i+1:2d}/{len(seeds)}  (seed={seed})", flush=True)
        res = runner_fn(seed)
        for model, metrics in res.items():
            for k, v in metrics.items():
                accumulated[model].setdefault(k, []).append(
                    float(v) if v is not None and not np.isnan(float(v) if v is not None else np.nan) else np.nan
                )
    return accumulated


def build_ci_table(accumulated):
    """Return a DataFrame with mean, 95%-CI lower/upper for each model × metric."""
    rows = []
    for model, metrics in accumulated.items():
        row = {'model': model}
        for metric, vals in metrics.items():
            clean = [v for v in vals if not np.isnan(v)]
            m, lo, hi = mean_ci(clean)
            row[f'{metric}_mean'] = round(m,  4)
            row[f'{metric}_lo']   = round(lo, 4)
            row[f'{metric}_hi']   = round(hi, 4)
            row[f'{metric}_std']  = round(np.std(clean, ddof=1), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def effect_size_vs_baseline(accumulated, baseline='LR', metrics=None):
    """
    For each non-baseline model, compute Cohen's d vs. baseline for each metric.
    Also run independent-samples t-tests and collect p-values for Bonferroni correction.
    """
    if metrics is None:
        # infer from first non-baseline model
        for m in MODELS:
            if m != baseline:
                metrics = list(accumulated[m].keys())
                break

    rows = []
    all_p = []

    for model in MODELS:
        if model == baseline:
            continue
        for metric in metrics:
            a  = [v for v in accumulated[model][metric]   if not np.isnan(v)]
            b  = [v for v in accumulated[baseline][metric] if not np.isnan(v)]
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a, b)
            d = cohens_d(a, b)
            rows.append({'model': model, 'metric': metric,
                         'cohen_d': round(d, 3), 'p_value': p_val,
                         't_stat': round(t_stat, 3)})
            all_p.append(p_val)

    # Bonferroni correction
    if all_p:
        corrected_thresh, sig_flags = bonferroni_correction(all_p, alpha=ALPHA)
        for i, row in enumerate(rows):
            row['bonferroni_thresh'] = round(corrected_thresh, 6)
            row['sig_corrected']     = sig_flags[i]
            row['p_value']           = round(row['p_value'], 6)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_ci(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def print_ci_table(ci_df, metrics, dataset_name):
    lines = []
    header = f"95% CI Table — {dataset_name}  (N = {N_RUNS} runs)"
    lines.append(header)
    lines.append("=" * len(header))
    col_w = 22
    hdr = f"{'Model':<15}" + "".join(f"{m:>{col_w}}" for m in metrics)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, row in ci_df.iterrows():
        line = f"{row['model']:<15}"
        for m in metrics:
            cell = fmt_ci(row[f'{m}_mean'], row[f'{m}_lo'], row[f'{m}_hi'])
            line += f"{cell:>{col_w}}"
        lines.append(line)
    lines.append("=" * len(hdr))
    lines.append("Format: mean [95% CI lower, upper]")
    return "\n".join(lines)


def print_effect_table(eff_df, dataset_name):
    lines = []
    header = f"Effect Sizes (Cohen's d) vs. LR Baseline — {dataset_name}"
    lines.append(header)
    lines.append("=" * 70)
    lines.append(f"{'Model':<15} {'Metric':<8} {'Cohen d':>9} "
                 f"{'t':>8} {'p':>10} {'Bonf. sig':>10}")
    lines.append("-" * 70)
    for _, row in eff_df.iterrows():
        sig = "YES" if row['sig_corrected'] else "NO"
        lines.append(f"{row['model']:<15} {row['metric']:<8} "
                     f"{row['cohen_d']:>9.3f} {row['t_stat']:>8.3f} "
                     f"{row['p_value']:>10.6f} {sig:>10}")
    lines.append("=" * 70)
    thresh = eff_df['bonferroni_thresh'].iloc[0] if len(eff_df) > 0 else ALPHA
    lines.append(f"Bonferroni-corrected threshold α' = {thresh:.6f}  "
                 f"(α={ALPHA}, k={len(eff_df)} comparisons)")
    lines.append("Cohen's d interpretation: small=0.2, medium=0.5, large=0.8")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    summary_blocks = []

    # ─── Agricultural ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Agricultural dataset  ({N_RUNS} runs)")
    print(f"{'='*60}")

    agri_acc = collect_runs(run_agricultural_seed, SEEDS)

    # Raw results (long format)
    raw_rows = []
    for model, metrics in agri_acc.items():
        n_runs = len(next(iter(metrics.values())))
        for i in range(n_runs):
            row = {'seed': SEEDS[i], 'model': model}
            for k, vals in metrics.items():
                row[k] = vals[i]
            raw_rows.append(row)
    df_agri_raw = pd.DataFrame(raw_rows)
    df_agri_raw.to_csv('statistical_agricultural.csv', index=False)
    print("\nRaw results → statistical_agricultural.csv")

    agri_metrics = ['R2', 'RMSE', 'DI', 'EOD', 'AOD']
    ci_agri = build_ci_table(agri_acc)
    ci_agri.to_csv('ci_table_agricultural.csv', index=False)

    eff_agri = effect_size_vs_baseline(agri_acc, baseline='LR',
                                        metrics=agri_metrics)

    ci_txt  = print_ci_table(ci_agri,  agri_metrics, 'Agricultural')
    eff_txt = print_effect_table(eff_agri, 'Agricultural')
    print("\n" + ci_txt)
    print("\n" + eff_txt)
    summary_blocks.extend([ci_txt, "", eff_txt, ""])

    # ─── German Credit ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"German Credit dataset  ({N_RUNS} runs)")
    print(f"{'='*60}")

    credit_acc = collect_runs(run_credit_seed, SEEDS)

    raw_rows = []
    for model, metrics in credit_acc.items():
        n_runs = len(next(iter(metrics.values())))
        for i in range(n_runs):
            row = {'seed': SEEDS[i], 'model': model}
            for k, vals in metrics.items():
                row[k] = vals[i]
            raw_rows.append(row)
    df_credit_raw = pd.DataFrame(raw_rows)
    df_credit_raw.to_csv('statistical_credit.csv', index=False)
    print("\nRaw results → statistical_credit.csv")

    credit_metrics = ['Acc', 'DI', 'EOD', 'AOD']
    ci_credit = build_ci_table(credit_acc)
    ci_credit.to_csv('ci_table_credit.csv', index=False)

    eff_credit = effect_size_vs_baseline(credit_acc, baseline='LR',
                                          metrics=credit_metrics)

    ci_txt2  = print_ci_table(ci_credit,  credit_metrics, 'German Credit')
    eff_txt2 = print_effect_table(eff_credit, 'German Credit')
    print("\n" + ci_txt2)
    print("\n" + eff_txt2)
    summary_blocks.extend([ci_txt2, "", eff_txt2])

    # ─── Seed variance summary ────────────────────────────────────────────────
    print("\n\nSeed-variance summary (FE-HITL, DI metric):")
    for dataset_name, acc in [('Agricultural', agri_acc),
                                ('German Credit', credit_acc)]:
        di_key = 'DI'
        vals   = acc['FE-HITL'].get(di_key, [])
        clean  = [v for v in vals if not np.isnan(v)]
        print(f"  {dataset_name}: mean={np.mean(clean):.3f}  "
              f"std={np.std(clean, ddof=1):.3f}  "
              f"min={np.min(clean):.3f}  max={np.max(clean):.3f}")

    # ─── Write full summary ───────────────────────────────────────────────────
    with open('statistical_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Statistical Robustness Analysis  (N = {N_RUNS} independent runs)\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("\n".join(summary_blocks))
    print("\nFull summary → statistical_summary.txt")


if __name__ == '__main__':
    main()