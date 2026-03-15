import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_agricultural_data(data_dir='data'):
    """Load agricultural dataset splits."""
    train = pd.read_csv(f'{data_dir}/agricultural_train.csv')
    val = pd.read_csv(f'{data_dir}/agricultural_val.csv')
    test = pd.read_csv(f'{data_dir}/agricultural_test.csv')

    # Features (excluding target and sensitive attributes)
    feature_cols = ['arable_land', 'labor_force', 'farming_years', 'yield_3y_avg',
                    'irrigation_resources', 'fertilizer_subsidy']

    X_train = train[feature_cols].values
    y_train = train['target_allocation'].values
    s_train = (train['region'] == 'A').astype(int).values   # 1 if region A (unprivileged)

    X_val = val[feature_cols].values
    y_val = val['target_allocation'].values
    s_val = (val['region'] == 'A').astype(int).values

    X_test = test[feature_cols].values
    y_test = test['target_allocation'].values
    s_test = (test['region'] == 'A').astype(int).values

    return (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test)

def load_german_credit_data(data_dir='data'):
    """Load processed German Credit dataset."""
    df = pd.read_csv(f'{data_dir}/german_credit_processed.csv')

    # Separate features, target, sensitive
    # Target is 'class', sensitive is 'gender'
    X = df.drop(columns=['class', 'gender']).values
    y = df['class'].values
    s = df['gender'].values   # 1 = male (privileged), 0 = female (unprivileged)

    # Train/val/test split (70/15/15)
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test)