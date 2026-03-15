import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def generate_agricultural_data(n_samples=10000, random_seed=42, save_dir='data'):
    np.random.seed(random_seed)

    # Features
    arable_land = np.random.uniform(0.5, 10.0, n_samples)          # hectares
    labor_force = np.random.randint(1, 6, n_samples)               # number of people
    farming_years = np.random.randint(1, 40, n_samples)            # years
    yield_3y_avg = np.random.uniform(2.0, 15.0, n_samples)         # tons
    irrigation_resources = np.random.uniform(0, 100, n_samples)    # index
    fertilizer_subsidy = np.random.uniform(0, 500, n_samples)      # amount

    # Sensitive attribute: region (A = underdeveloped, B/C = developed)
    region = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    # Base allocation (ideal fair allocation based on objective factors)
    base_allocation = (0.3 * arable_land +
                       0.2 * labor_force +
                       0.1 * farming_years +
                       0.2 * yield_3y_avg +
                       0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)

    # Inject historical bias: region A gets 30% less
    allocation_bias = np.where(region == 'A', 0.7, 1.0)
    historical_allocation = base_allocation * allocation_bias + np.random.normal(0, 2, n_samples)
    historical_allocation = np.maximum(historical_allocation, 0)

    # Target allocation (fair, but with noise)
    target_allocation = base_allocation + np.random.normal(0, 1, n_samples)
    target_allocation = np.maximum(target_allocation, 0)

    # Create DataFrame
    df = pd.DataFrame({
        'arable_land': arable_land,
        'labor_force': labor_force,
        'farming_years': farming_years,
        'yield_3y_avg': yield_3y_avg,
        'irrigation_resources': irrigation_resources,
        'fertilizer_subsidy': fertilizer_subsidy,
        'region': region,
        'historical_allocation': historical_allocation,
        'target_allocation': target_allocation
    })

    # Split into train/val/test (70/15/15)
    train, temp = train_test_split(df, test_size=0.3, random_state=random_seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=random_seed)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'agricultural_data_full.csv'), index=False)
    train.to_csv(os.path.join(save_dir, 'agricultural_train.csv'), index=False)
    val.to_csv(os.path.join(save_dir, 'agricultural_val.csv'), index=False)
    test.to_csv(os.path.join(save_dir, 'agricultural_test.csv'), index=False)

    print(f"Agricultural dataset generated and saved to {save_dir}")
    return df

if __name__ == '__main__':
    generate_agricultural_data()