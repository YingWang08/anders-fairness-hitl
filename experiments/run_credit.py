"""
run_credit.py  (revised for W3)
=================================
Same structure as the revised run_agricultural.py.
Original single-seed logic is preserved in run_single(seed).
main() runs N_RUNS independent seeds and writes a CSV.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.data_loader       import load_german_credit_data
from src.models.baseline_lr    import BaselineLR
from src.models.baseline_dl    import BaselineDL
from src.models.debiased_hitl  import DebiasedHITL
from src.models.fe_hitl        import FEHITL
from src.fairness_metrics      import (disparate_impact,
                                        equal_opportunity_difference,
                                        average_odds_difference)
from src.utils import compute_classification_metrics, set_seed

N_RUNS = 30
SEEDS  = list(range(0, N_RUNS))


def run_single(seed=42):
    set_seed(seed)

    (X_train, y_train, s_train), \
    (X_val,   y_val,   s_val), \
    (X_test,  y_test,  s_test) = load_german_credit_data()

    ctx = [{} for _ in range(len(X_test))]
    rows = []

    def record(name, pred):
        rows.append(dict(
            seed=seed, model=name,
            Acc=compute_classification_metrics(y_test, pred),
            DI=disparate_impact(pred, s_test),
            EOD=abs(equal_opportunity_difference(y_test, pred, s_test)),
            AOD=abs(average_odds_difference(y_test, pred, s_test)),
        ))

    # LR
    m = BaselineLR(task='classification')
    m.fit(X_train, y_train)
    record('LR', m.predict(X_test))

    # DL
    m = BaselineDL(input_dim=X_train.shape[1], task='classification',
                   device='cpu', random_seed=seed)
    m.fit(X_train, y_train, X_val, y_val)
    record('DL', m.predict(X_test))

    # Debiased-HITL
    m = DebiasedHITL(input_dim=X_train.shape[1], task='classification',
                     device='cpu', random_seed=seed,
                     di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1)
    m.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    record('Debiased-HITL',
           m.predict(X_test, s_test, context_list=ctx, apply_intervention=True))

    # FE-HITL
    m = FEHITL(input_dim=X_train.shape[1], task='classification',
               device='cpu', random_seed=seed,
               epsilon=0.1, di_threshold=0.8, eod_threshold=0.1)
    m.fit_base(X_train, y_train, X_val, y_val)
    record('FE-HITL',
           m.predict_with_intervention(X_test, s_test, context_list=ctx))

    return rows


def main():
    all_rows = []

    print("\nGerman Credit Dataset — Multi-seed experiment")
    print(f"N_RUNS = {N_RUNS}  |  seeds = {SEEDS[:5]} ... {SEEDS[-1]}")
    print("=" * 65)

    for i, seed in enumerate(SEEDS):
        rows = run_single(seed)
        all_rows.extend(rows)
        fe = next(r for r in rows if r['model'] == 'FE-HITL')
        print(f"  Run {i+1:2d} (seed={seed:2d})  "
              f"FE-HITL: Acc={fe['Acc']:.3f}  DI={fe['DI']:.3f}  EOD={fe['EOD']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv('credit_multi_seed.csv', index=False)
    print("\nAll runs saved → credit_multi_seed.csv")

    print("\nSummary (mean ± std over 30 runs):")
    print(f"{'Model':<15} {'Acc':>12} {'DI':>12} {'|EOD|':>12}")
    print("-" * 55)
    for model in ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']:
        sub = df[df['model'] == model]
        print(f"{model:<15} "
              f"{sub.Acc.mean():.3f}±{sub.Acc.std():.3f}  "
              f"{sub.DI.mean():.3f}±{sub.DI.std():.3f}  "
              f"{sub.EOD.mean():.3f}±{sub.EOD.std():.3f}")

    print("\n\nOriginal single-run table (seed=42) for paper Table 2:")
    print("=" * 55)
    print(f"{'Model':<15} {'Acc':<8} {'DI':<6} {'EOD':<6} {'AOD':<6}")
    print("-" * 55)
    orig = df[df['seed'] == 42]
    for model in ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']:
        r = orig[orig['model'] == model].iloc[0]
        print(f"{model:<15} {r.Acc:.3f}   {r.DI:.3f}  {r.EOD:.3f}  {r.AOD:.3f}")


if __name__ == '__main__':
    main()