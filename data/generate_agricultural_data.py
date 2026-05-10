"""
generate_agricultural_data.py  (revised for W2 sensitivity analysis)
=====================================================================
Now accepts `bias_rate` as a parameter (default 0.30 = original setting).
All other logic is unchanged. Called directly for the main experiment,
and imported by run_sensitivity.py for the sensitivity sweep.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def generate_agricultural_data(n_samples=10000,
                                random_seed=42,
                                save_dir='data',
                                bias_rate=0.30):
    """
    Generate the simulated agricultural resource-allocation dataset.

    Parameters
    ----------
    n_samples   : int   – number of farmers (default 10 000)
    random_seed : int   – NumPy random seed for reproducibility
    save_dir    : str   – directory to save CSV splits
    bias_rate   : float – historical allocation penalty for Region A.
                          0.30 means Region A receives 70 % of the
                          baseline allocation (original setting).
                          Used by run_sensitivity.py for the W2
                          sensitivity analysis (0.10, 0.20, 0.30, 0.40).
    """
    np.random.seed(random_seed)

    # ── Features ──────────────────────────────────────────────────────────────
    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg         = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)

    # Sensitive attribute: region (A = underdeveloped, B/C = developed)
    region = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    # ── Fair base allocation (objective factors only) ─────────────────────────
    base_allocation = (0.3 * arable_land +
                       0.2 * labor_force +
                       0.1 * farming_years +
                       0.2 * yield_3y_avg +
                       0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)

    # ── Inject historical bias ────────────────────────────────────────────────
    # Region A receives (1 - bias_rate) of the baseline allocation.
    # bias_rate=0.30 reproduces the original 30 % penalty.
    allocation_multiplier = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = (base_allocation * allocation_multiplier
                             + np.random.normal(0, 2, n_samples))
    historical_allocation = np.maximum(historical_allocation, 0)

    # ── Fair target allocation (model supervision signal) ────────────────────
    target_allocation = base_allocation + np.random.normal(0, 1, n_samples)
    target_allocation = np.maximum(target_allocation, 0)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        'arable_land':           arable_land,
        'labor_force':           labor_force,
        'farming_years':         farming_years,
        'yield_3y_avg':          yield_3y_avg,
        'irrigation_resources':  irrigation_resources,
        'fertilizer_subsidy':    fertilizer_subsidy,
        'region':                region,
        'historical_allocation': historical_allocation,
        'target_allocation':     target_allocation,
    })

    # ── Split 70 / 15 / 15 ───────────────────────────────────────────────────
    train, temp = train_test_split(df, test_size=0.30, random_state=random_seed)
    val,   test = train_test_split(temp, test_size=0.50, random_state=random_seed)

    # ── Save (only for the default run, not during sensitivity sweep) ─────────
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'agricultural_data_full.csv'), index=False)
    train.to_csv(os.path.join(save_dir, 'agricultural_train.csv'), index=False)
    val.to_csv(os.path.join(save_dir, 'agricultural_val.csv'),   index=False)
    test.to_csv(os.path.join(save_dir, 'agricultural_test.csv'), index=False)

    print(f"Agricultural dataset generated  "
          f"(bias_rate={bias_rate:.0%}, seed={random_seed})  → {save_dir}/")
    return df


if __name__ == '__main__':
    # Default call reproduces the original dataset exactly
    generate_agricultural_data()