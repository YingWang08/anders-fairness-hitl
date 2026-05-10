"""
run_agricultural.py  (revised for W3)
======================================
Original single-seed experiment is preserved as `run_single(seed)`.
`main()` now iterates over N_RUNS seeds, prints every run, and
writes a summary CSV — satisfying Reviewer #2's W3 requirement.

For the full statistical analysis (CI, Cohen's d, Bonferroni), use
run_statistical.py instead. This file is kept lean so the original
table in the paper can be reproduced by calling run_single(42).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.data_loader       import load_agricultural_data
from src.models.baseline_lr    import BaselineLR
from src.models.baseline_dl    import BaselineDL
from src.models.debiased_hitl  import DebiasedHITL
from src.models.fe_hitl        import FEHITL
from src.fairness_metrics      import (disparate_impact,
                                        equal_opportunity_difference,
                                        average_odds_difference)
from src.utils import compute_regression_metrics, set_seed

N_RUNS = 30
SEEDS  = list(range(0, N_RUNS))


def run_single(seed=42):
    set_seed(seed)

    (X_train, y_train, s_train), \
    (X_val,   y_val,   s_val), \
    (X_test,  y_test,  s_test) = load_agricultural_data()

    threshold = np.median(y_train)
    binarize  = lambda y: (y > threshold).astype(int)

    test_df   = pd.read_csv('data/agricultural_test.csv')
    avg_B     = np.mean(y_train[s_train == 0])
    ctx       = [{'region': r, 'avg_allocation_B': avg_B}
                 for r in test_df['region'].values]

    rows = []

    def record(name, pred):
        r2, rmse = compute_regression_metrics(y_test, pred)
        rows.append(dict(
            seed=seed, model=name,
            R2=r2, RMSE=rmse,
            DI=disparate_impact(binarize(pred), s_test),
            EOD=abs(equal_opportunity_difference(binarize(y_test), binarize(pred), s_test)),
            AOD=abs(average_odds_difference(binarize(y_test), binarize(pred), s_test)),
        ))

    # LR
    m = BaselineLR(task='regression')
    m.fit(X_train, y_train)
    record('LR', m.predict(X_test))

    # DL
    m = BaselineDL(input_dim=X_train.shape[1], task='regression',
                   device='cpu', random_seed=seed)
    m.fit(X_train, y_train, X_val, y_val)
    record('DL', m.predict(X_test))

    # Debiased-HITL
    m = DebiasedHITL(input_dim=X_train.shape[1], task='regression',
                     device='cpu', random_seed=seed,
                     di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1)
    m.set_binarize_threshold(threshold)
    m.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    record('Debiased-HITL',
           m.predict(X_test, s_test, context_list=ctx, apply_intervention=True))

    # FE-HITL
    m = FEHITL(input_dim=X_train.shape[1], task='regression',
               device='cpu', random_seed=seed,
               epsilon=0.1, di_threshold=0.8, eod_threshold=0.1)
    m.fit_base(X_train, y_train, X_val, y_val)
    record('FE-HITL',
           m.predict_with_intervention(X_test, s_test, context_list=ctx))

    return rows


def main():
    all_rows = []

    print("\nAgricultural Dataset — Multi-seed experiment")
    print(f"N_RUNS = {N_RUNS}  |  seeds = {SEEDS[:5]} ... {SEEDS[-1]}")
    print("=" * 65)

    for i, seed in enumerate(SEEDS):
        rows = run_single(seed)
        all_rows.extend(rows)
        fe = next(r for r in rows if r['model'] == 'FE-HITL')
        print(f"  Run {i+1:2d} (seed={seed:2d})  "
              f"FE-HITL: R²={fe['R2']:.3f}  DI={fe['DI']:.3f}  EOD={fe['EOD']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv('agricultural_multi_seed.csv', index=False)
    print("\nAll runs saved → agricultural_multi_seed.csv")

    # Summary table: mean ± std across seeds
    print("\nSummary (mean ± std over 30 runs):")
    print(f"{'Model':<15} {'R²':>12} {'DI':>12} {'|EOD|':>12}")
    print("-" * 55)
    for model in ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']:
        sub = df[df['model'] == model]
        print(f"{model:<15} "
              f"{sub.R2.mean():.3f}±{sub.R2.std():.3f}  "
              f"{sub.DI.mean():.3f}±{sub.DI.std():.3f}  "
              f"{sub.EOD.mean():.3f}±{sub.EOD.std():.3f}")

    # ── Reproduce original single-seed table (seed=42) ────────────────────────
    print("\n\nOriginal single-run table (seed=42) for paper Table 1:")
    print("=" * 60)
    print(f"{'Model':<15} {'R²':<6} {'RMSE':<8} {'DI':<6} {'EOD':<6} {'AOD':<6}")
    print("-" * 60)
    orig = df[df['seed'] == 42]
    for model in ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']:
        r = orig[orig['model'] == model].iloc[0]
        print(f"{model:<15} {r.R2:.3f}  {r.RMSE:.3f}   "
              f"{r.DI:.3f}  {r.EOD:.3f}  {r.AOD:.3f}")


if __name__ == '__main__':
    main()