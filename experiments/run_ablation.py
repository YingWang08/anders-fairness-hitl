"""
run_ablation.py  (revised for W3)
===================================
Ablation study over N_RUNS=30 seeds.
Each variant is run with the same set of seeds as the main experiment.
Outputs ablation_multi_seed.csv and a mean±std summary table.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.data_loader   import load_agricultural_data
from src.models.fe_hitl import FEHITL
from src.fairness_metrics import (disparate_impact,
                                   equal_opportunity_difference,
                                   average_odds_difference)
from src.utils import compute_regression_metrics, set_seed

N_RUNS   = 30
SEEDS    = list(range(0, N_RUNS))
VARIANTS = ['Full', 'w/o HED', 'w/o MOG', 'w/o F&U']


def run_variant_seed(variant_name, seed=42):
    set_seed(seed)

    (X_train, y_train, s_train), \
    (X_val,   y_val,   s_val), \
    (X_test,  y_test,  s_test) = load_agricultural_data()

    threshold = np.median(y_train)
    binarize  = lambda y: (y > threshold).astype(int)

    test_df = pd.read_csv('data/agricultural_test.csv')
    avg_B   = np.mean(y_train[s_train == 0])
    ctx     = [{'region': r, 'avg_allocation_B': avg_B}
               for r in test_df['region'].values]

    fehitl = FEHITL(
        input_dim=X_train.shape[1], task='regression',
        device='cpu', random_seed=seed,
        epsilon=0.1, di_threshold=0.8, eod_threshold=0.1
    )

    # Disable specific modules per variant
    if variant_name == 'w/o HED':
        fehitl.epsilon = 0.0        # no forced human intervention
    elif variant_name == 'w/o MOG':
        fehitl.option_gen.num_options = 1   # single option = no real choice
    elif variant_name == 'w/o F&U':
        fehitl.retrain_with_feedback = lambda: None  # disable feedback retraining

    fehitl.fit_base(X_train, y_train, X_val, y_val)
    pred = fehitl.predict_with_intervention(X_test, s_test, context_list=ctx)

    r2, rmse = compute_regression_metrics(y_test, pred)
    di  = disparate_impact(binarize(pred), s_test)
    eod = abs(equal_opportunity_difference(binarize(y_test), binarize(pred), s_test))
    aod = abs(average_odds_difference(binarize(y_test), binarize(pred), s_test))

    return dict(seed=seed, variant=variant_name,
                R2=r2, RMSE=rmse, DI=di, EOD=eod, AOD=aod)


def main():
    all_rows = []

    print("\nAblation Study — Multi-seed experiment")
    print(f"N_RUNS = {N_RUNS}  |  Variants = {VARIANTS}")
    print("=" * 65)

    for i, seed in enumerate(SEEDS):
        print(f"  Run {i+1:2d}/{N_RUNS}  (seed={seed})", end='  ', flush=True)
        for variant in VARIANTS:
            row = run_variant_seed(variant, seed)
            all_rows.append(row)
        # Print FE-HITL (Full) result for this seed
        full_row = next(r for r in all_rows
                        if r['seed'] == seed and r['variant'] == 'Full')
        print(f"Full: DI={full_row['DI']:.3f}  EOD={full_row['EOD']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv('ablation_multi_seed.csv', index=False)
    print("\nAll runs saved → ablation_multi_seed.csv")

    # Summary table
    print("\nAblation Study Summary (mean ± std over 30 runs):")
    print("=" * 65)
    print(f"{'Variant':<12} {'R²':>13} {'DI':>13} {'|EOD|':>13}")
    print("-" * 55)
    for variant in VARIANTS:
        sub = df[df['variant'] == variant]
        print(f"{variant:<12} "
              f"{sub.R2.mean():.3f}±{sub.R2.std():.3f}  "
              f"{sub.DI.mean():.3f}±{sub.DI.std():.3f}  "
              f"{sub.EOD.mean():.3f}±{sub.EOD.std():.3f}")

    # Original single-seed table
    print("\n\nOriginal single-run table (seed=42) for paper ablation table:")
    print("=" * 60)
    print(f"{'Variant':<12} {'R²':<6} {'RMSE':<8} {'DI':<6} {'EOD':<6} {'AOD':<6}")
    print("-" * 60)
    orig = df[df['seed'] == 42]
    for variant in VARIANTS:
        r = orig[orig['variant'] == variant].iloc[0]
        print(f"{variant:<12} {r.R2:.3f}  {r.RMSE:.3f}   "
              f"{r.DI:.3f}  {r.EOD:.3f}  {r.AOD:.3f}")


if __name__ == '__main__':
    main()