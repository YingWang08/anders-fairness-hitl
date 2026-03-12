import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import torch
from src.models import DeepRegression, SklearnLRReg, DebiasedHITL, FEHITL
from src.human_simulation import HumanSimulator
from src.train_eval import train_and_evaluate
from src.fairness_metrics import compute_all_metrics
from src.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load data
train = pd.read_csv('../data/agriculture_train.csv')
val = pd.read_csv('../data/agriculture_val.csv')
test = pd.read_csv('../data/agriculture_test.csv')

# Features and target
feature_cols = ['farm_area', 'labor', 'farming_years', 'irrigation', 'fertilizer', 'past_yield']
target_col = 'resource_allocation'
sensitive_col = 'region'

# Encode region: A=0 (unprivileged), B/C=1 (privileged)
def encode_region(r):
    return 0 if r == 'A' else 1

train['sensitive'] = train[sensitive_col].apply(encode_region)
val['sensitive'] = val[sensitive_col].apply(encode_region)
test['sensitive'] = test[sensitive_col].apply(encode_region)

X_train = train[feature_cols].values.astype(np.float32)
y_train = train[target_col].values.astype(np.float32)
s_train = train['sensitive'].values.astype(np.int32)

X_val = val[feature_cols].values.astype(np.float32)
y_val = val[target_col].values.astype(np.float32)
s_val = val['sensitive'].values.astype(np.int32)

X_test = test[feature_cols].values.astype(np.float32)
y_test = test[target_col].values.astype(np.float32)
s_test = test['sensitive'].values.astype(np.int32)

# ========== Baseline LR ==========
lr_model = SklearnLRReg()
metrics_lr, _ = train_and_evaluate(lr_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                   task='regression', model_type='lr')
print("LR:", metrics_lr)

# ========== Baseline DL ==========
dl_model = DeepRegression(input_dim=len(feature_cols))
metrics_dl, _ = train_and_evaluate(dl_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                   task='regression', model_type='dl', epochs=50)
print("DL:", metrics_dl)

# ========== Debiased-HITL ==========
base_lr = SklearnLRReg()
debiased_model = DebiasedHITL(base_lr, fairness_weight=0.2)
metrics_debiased, _ = train_and_evaluate(debiased_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                         task='regression', model_type='debiased')
print("Debiased-HITL:", metrics_debiased)

# ========== Our FE-HITL ==========
base_dl = DeepRegression(input_dim=len(feature_cols))
human_sim = HumanSimulator(rule='fairness_first')
fehitl_model = FEHITL(base_dl, monitor=None, human_simulator=human_sim,
                      threshold_di=0.8, threshold_eod=0.1, intervention_ratio=0.1)
metrics_fehitl, y_pred = train_and_evaluate(fehitl_model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test,
                                            task='regression', model_type='fehitl', epochs=50)
print("Our FE-HITL:", metrics_fehitl)

# Collect results for Table 1
table1 = pd.DataFrame({
    'Model': ['LR', 'DL', 'Debiased-HITL', 'Our framework'],
    'R2': [metrics_lr['R2'], metrics_dl['R2'], metrics_debiased['R2'], metrics_fehitl['R2']],
    'RMSE': [metrics_lr['RMSE'], metrics_dl['RMSE'], metrics_debiased['RMSE'], metrics_fehitl['RMSE']],
    'DI': [metrics_lr['DI'], metrics_dl['DI'], metrics_debiased['DI'], metrics_fehitl['DI']],
    'EOD': [metrics_lr['EOD'], metrics_dl['EOD'], metrics_debiased['EOD'], metrics_fehitl['EOD']],
    'AOD': [metrics_lr['AOD'], metrics_dl['AOD'], metrics_debiased['AOD'], metrics_fehitl['AOD']]
})
print("\nTable 1: Agricultural Dataset Results")
print(table1.to_string(index=False))
table1.to_csv('../results/table1_agriculture.csv', index=False)