import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.data_loader import load_agricultural_data
from src.models.baseline_lr import BaselineLR
from src.models.baseline_dl import BaselineDL
from src.models.debiased_hitl import DebiasedHITL
from src.models.fe_hitl import FEHITL
from src.fairness_metrics import disparate_impact, equal_opportunity_difference, average_odds_difference
from src.utils import compute_regression_metrics, set_seed
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

def main():
    set_seed(42)

    # 加载数据
    (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test) = load_agricultural_data()

    # 对回归任务，需要二值化以计算公平性指标（以训练集目标值中位数为阈值）
    threshold = np.median(y_train)
    def binarize(y): return (y > threshold).astype(int)

    # 为模拟人类决策准备上下文（区域信息）
    test_df = pd.read_csv('data/agricultural_test.csv')
    region_test = test_df['region'].values
    avg_allocation_B_train = np.mean(y_train[s_train == 0])  # 非A区域的平均分配
    context_list = [{'region': r, 'avg_allocation_B': avg_allocation_B_train} for r in region_test]

    # -------------------- 1. 逻辑回归 (LR) --------------------
    lr = BaselineLR(task='regression')
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    r2_lr, rmse_lr = compute_regression_metrics(y_test, pred_lr)
    di_lr = disparate_impact(binarize(pred_lr), s_test)
    eod_lr = equal_opportunity_difference(binarize(y_test), binarize(pred_lr), s_test)
    aod_lr = average_odds_difference(binarize(y_test), binarize(pred_lr), s_test)

    # -------------------- 2. 深度学习 (DL) --------------------
    dl = BaselineDL(input_dim=X_train.shape[1], task='regression', device='cpu')
    dl.fit(X_train, y_train, X_val, y_val)
    pred_dl = dl.predict(X_test)
    r2_dl, rmse_dl = compute_regression_metrics(y_test, pred_dl)
    di_dl = disparate_impact(binarize(pred_dl), s_test)
    eod_dl = equal_opportunity_difference(binarize(y_test), binarize(pred_dl), s_test)
    aod_dl = average_odds_difference(binarize(y_test), binarize(pred_dl), s_test)

    # -------------------- 3. Debiased-HITL (复现) --------------------
    debiased = DebiasedHITL(
        input_dim=X_train.shape[1],
        task='regression',
        device='cpu',
        di_threshold=0.8,
        eod_threshold=0.1,
        intervention_ratio=0.1
    )
    debiased.set_binarize_threshold(threshold)  # 设置二值化阈值
    debiased.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    pred_debiased = debiased.predict(X_test, s_test, context_list=context_list, apply_intervention=True)
    r2_debiased, rmse_debiased = compute_regression_metrics(y_test, pred_debiased)
    di_debiased = disparate_impact(binarize(pred_debiased), s_test)
    eod_debiased = equal_opportunity_difference(binarize(y_test), binarize(pred_debiased), s_test)
    aod_debiased = average_odds_difference(binarize(y_test), binarize(pred_debiased), s_test)

    # -------------------- 4. FE-HITL (本文提出) --------------------
    fehitl = FEHITL(
        input_dim=X_train.shape[1],
        task='regression',
        device='cpu',
        epsilon=0.1,
        di_threshold=0.8,
        eod_threshold=0.1
    )
    fehitl.fit_base(X_train, y_train, X_val, y_val)
    pred_fehitl = fehitl.predict_with_intervention(X_test, s_test, context_list=context_list)
    r2_fehitl, rmse_fehitl = compute_regression_metrics(y_test, pred_fehitl)
    di_fehitl = disparate_impact(binarize(pred_fehitl), s_test)
    eod_fehitl = equal_opportunity_difference(binarize(y_test), binarize(pred_fehitl), s_test)
    aod_fehitl = average_odds_difference(binarize(y_test), binarize(pred_fehitl), s_test)

    # 打印结果表格（类似论文表1）
    print("\n" + "="*60)
    print("Agricultural Dataset Results (Regression)")
    print("="*60)
    print(f"{'Model':<15} {'R²':<6} {'RMSE':<8} {'DI':<6} {'EOD':<6} {'AOD':<6}")
    print("-"*60)
    print(f"{'LR':<15} {r2_lr:.2f}   {rmse_lr:.2f}    {di_lr:.2f}   {eod_lr:.2f}   {aod_lr:.2f}")
    print(f"{'DL':<15} {r2_dl:.2f}   {rmse_dl:.2f}    {di_dl:.2f}   {eod_dl:.2f}   {aod_dl:.2f}")
    print(f"{'Debiased-HITL':<15} {r2_debiased:.2f}   {rmse_debiased:.2f}    {di_debiased:.2f}   {eod_debiased:.2f}   {aod_debiased:.2f}")
    print(f"{'FE-HITL':<15} {r2_fehitl:.2f}   {rmse_fehitl:.2f}    {di_fehitl:.2f}   {eod_fehitl:.2f}   {aod_fehitl:.2f}")
    print("="*60)

if __name__ == '__main__':
    main()