import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import torch
from src.models import DeepClassifier, SklearnLRClf, DebiasedHITL, FEHITL
from src.human_simulation import HumanSimulator
from src.train_eval import train_and_evaluate
from src.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load German Credit data
# Assuming file is at ../data/german_credit.csv
# If not, download from UCI
try:
    df = pd.read_csv('../data/german_credit.csv', header=None)
except:
    # Download from UCI
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    df = pd.read_csv(url, header=None, delim_whitespace=True)
    df.to_csv('../data/german_credit.csv', index=False)

# Column names (from UCI documentation)
columns = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 
           'savings', 'employment', 'installment_rate', 'personal_status_sex', 'other_debtors',
           'present_residence', 'property', 'age', 'other_installment_plans', 'housing',
           'existing_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'target']
df.columns = columns

# Target: 1=good, 2=bad -> convert to 1/0 (good=1, bad=0)
df['target'] = (df['target'] == 1).astype(int)

# Sensitive attribute: gender from personal_status_sex
# Codes: A91: male (divorced/separated), A92: female (divorced/separated/married), 
# A93: male (single), A94: male (married/widowed), A95: female (single)
# We'll map to binary: male=1, female=0
def gender_from_status(s):
    if s in ['A91', 'A93', 'A94']:
        return 1
    else:
        return 0
df['gender'] = df['personal_status_sex'].apply(gender_from_status)

# Select features: we'll use all numeric and encode categoricals simply
# For simplicity, we'll use a subset of features and one-hot encode some
feature_cols = ['duration', 'credit_amount', 'installment_rate', 'present_residence', 
                'age', 'existing_credits', 'num_liable']
# Convert categoricals to dummies for a few important ones
df = pd.get_dummies(df, columns=['checking_status', 'credit_history', 'purpose', 'savings', 'employment'], drop_first=True)
# Get all numeric columns after dummies
feature_cols = [c for c in df.columns if c not in ['target', 'personal_status_sex', 'gender']]

# Train/val/test split
from sklearn.model_selection import train_test_split
X = df[feature_cols].values.astype(np.float32)
y = df['target'].values.astype(np.float32)
s = df['gender'].values.astype(np.int32)

X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(X, y, s, test_size=0.3, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(X_temp, y_temp, s_temp, test_size=0.5, random_state=RANDOM_SEED)

# ========== Baseline LR ==========
lr_model = SklearnLRClf(max_iter=1000)
metrics_lr, _ = train_and_evaluate(lr_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                   task='classification', model_type='lr')
print("LR:", metrics_lr)

# ========== Baseline DL ==========
dl_model = DeepClassifier(input_dim=X_train.shape[1])
metrics_dl, _ = train_and_evaluate(dl_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                   task='classification', model_type='dl', epochs=30)
print("DL:", metrics_dl)

# ========== Debiased-HITL ==========
base_lr = SklearnLRClf(max_iter=1000)
debiased_model = DebiasedHITL(base_lr, fairness_weight=0.2)
metrics_debiased, _ = train_and_evaluate(debiased_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                         task='classification', model_type='debiased')
print("Debiased-HITL:", metrics_debiased)

# ========== Our FE-HITL ==========
base_dl = DeepClassifier(input_dim=X_train.shape[1])
human_sim = HumanSimulator(rule='fairness_first')
fehitl_model = FEHITL(base_dl, monitor=None, human_simulator=human_sim,
                      threshold_di=0.8, threshold_eod=0.1, intervention_ratio=0.1)
metrics_fehitl, y_pred = train_and_evaluate(fehitl_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                            task='classification', model_type='fehitl', epochs=30)
print("Our FE-HITL:", metrics_fehitl)

# Table 3
table3 = pd.DataFrame({
    'Model': ['LR', 'DL', 'Debiased-HITL', 'Our framework'],
    'Accuracy': [metrics_lr['Accuracy'], metrics_dl['Accuracy'], metrics_debiased['Accuracy'], metrics_fehitl['Accuracy']],
    'DI': [metrics_lr['DI'], metrics_dl['DI'], metrics_debiased['DI'], metrics_fehitl['DI']],
    'EOD': [metrics_lr['EOD'], metrics_dl['EOD'], metrics_debiased['EOD'], metrics_fehitl['EOD']],
    'AOD': [metrics_lr['AOD'], metrics_dl['AOD'], metrics_debiased['AOD'], metrics_fehitl['AOD']]
})
print("\nTable 3: German Credit Dataset Results")
print(table3.to_string(index=False))
table3.to_csv('../results/table3_german.csv', index=False)