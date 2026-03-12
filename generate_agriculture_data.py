import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_samples = 10000

# Generate features
farm_area = np.random.lognormal(mean=1.5, sigma=0.5, size=n_samples)  # hectares
labor = np.random.randint(1, 6, size=n_samples)  # number of laborers
farming_years = np.random.randint(1, 51, size=n_samples)  # years of farming
irrigation = np.random.binomial(1, 0.6, size=n_samples)  # irrigation access
fertilizer = np.random.gamma(shape=2, scale=2, size=n_samples)  # fertilizer subsidy (units)
past_yield = 3 * farm_area + 0.8 * labor + 0.2 * farming_years + 2 * irrigation + 0.5 * fertilizer + np.random.normal(0, 2, n_samples)

# Region (sensitive attribute): A (underdeveloped), B, C
region = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.3, 0.4, 0.3])
region_bias = np.where(region == 'A', 0.7, 1.0)  # A gets 30% less

# Target: resource allocation (continuous)
true_allocation = (
    2.5 * farm_area + 
    1.2 * labor + 
    0.3 * farming_years + 
    3.0 * irrigation + 
    0.8 * fertilizer + 
    0.5 * past_yield +
    np.random.normal(0, 3, n_samples)
)
# Inject historical bias: region A receives less
biased_allocation = true_allocation * region_bias

# Also adjust past yield for region A to simulate historical bias
past_yield = past_yield * region_bias

# Create DataFrame
df = pd.DataFrame({
    'farm_area': farm_area,
    'labor': labor,
    'farming_years': farming_years,
    'irrigation': irrigation,
    'fertilizer': fertilizer,
    'past_yield': past_yield,
    'region': region,
    'resource_allocation': biased_allocation
})

# Split into train/val/test
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv('agriculture_train.csv', index=False)
val.to_csv('agriculture_val.csv', index=False)
test.to_csv('agriculture_test.csv', index=False)

print("Generated agriculture dataset with 10,000 samples.")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")