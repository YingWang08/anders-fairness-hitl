import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.data_loader import load_german_credit_data
from src.models.baseline_lr import BaselineLR
from src.models.baseline_dl import BaselineDL
from src.models.debiased_hitl import DebiasedHITL
from src.models.fe_hitl import FEHITL
from src.fairness_metrics import disparate_impact, equal_opportunity_difference, average_odds_difference
from src.utils import compute_classification_metrics, set_seed
import warnings

warnings.filterwarnings('ignore')

def main():
    set_seed(42)

    # 加载数据
    (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test) = load_german_credit_data()

    # 分类任务无需二值化，直接使用预测标签
    # 上下文列表（本场景不需要额外信息，留空即可）
    context_list = [{} for _ in range(len(X_test))]

    # -------------------- 1. 逻辑回归 (LR) --------------------
    lr = BaselineLR(task='classification')
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    acc_lr = compute_classification_metrics(y_test, pred_lr)
    di_lr = disparate_impact(pred_lr, s_test)
    eod_lr = equal_opportunity_difference(y_test, pred_lr, s_test)
    aod_lr = average_odds_difference(y_test, pred_lr, s_test)

    # -------------------- 2. 深度学习 (DL) --------------------
    dl = BaselineDL(input_dim=X_train.shape[1], task='classification', device='cpu')
    dl.fit(X_train, y_train, X_val, y_val)
    pred_dl = dl.predict(X_test)
    acc_dl = compute_classification_metrics(y_test, pred_dl)
    di_dl = disparate_impact(pred_dl, s_test)
    eod_dl = equal_opportunity_difference(y_test, pred_dl, s_test)
    aod_dl = average_odds_difference(y_test, pred_dl, s_test)

    # -------------------- 3. Debiased-HITL --------------------
    debiased = DebiasedHITL(
        input_dim=X_train.shape[1],
        task='classification',
        device='cpu',
        di_threshold=0.8,
        eod_threshold=0.1,
        intervention_ratio=0.1
    )
    # 分类任务不需要二值化阈值
    debiased.fit(X_train, y_train, s_train, X_val, y_val, s_val)
    pred_debiased = debiased.predict(X_test, s_test, context_list=context_list, apply_intervention=True)
    acc_debiased = compute_classification_metrics(y_test, pred_debiased)
    di_debiased = disparate_impact(pred_debiased, s_test)
    eod_debiased = equal_opportunity_difference(y_test, pred_debiased, s_test)
    aod_debiased = average_odds_difference(y_test, pred_debiased, s_test)

    # -------------------- 4. FE-HITL --------------------
    fehitl = FEHITL(
        input_dim=X_train.shape[1],
        task='classification',
        device='cpu',
        epsilon=0.1,
        di_threshold=0.8,
        eod_threshold=0.1
    )
    fehitl.fit_base(X_train, y_train, X_val, y_val)
    pred_fehitl = fehitl.predict_with_intervention(X_test, s_test, context_list=context_list)
    acc_fehitl = compute_classification_metrics(y_test, pred_fehitl)
    di_fehitl = disparate_impact(pred_fehitl, s_test)
    eod_fehitl = equal_opportunity_difference(y_test, pred_fehitl, s_test)
    aod_fehitl = average_odds_difference(y_test, pred_fehitl, s_test)

    # 打印结果表格（类似论文表3）
    print("\n" + "="*60)
    print("German Credit Dataset Results (Classification)")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':<10} {'DI':<6} {'EOD':<6} {'AOD':<6}")
    print("-"*60)
    print(f"{'LR':<15} {acc_lr:.3f}     {di_lr:.2f}   {eod_lr:.2f}   {aod_lr:.2f}")
    print(f"{'DL':<15} {acc_dl:.3f}     {di_dl:.2f}   {eod_dl:.2f}   {aod_dl:.2f}")
    print(f"{'Debiased-HITL':<15} {acc_debiased:.3f}     {di_debiased:.2f}   {eod_debiased:.2f}   {aod_debiased:.2f}")
    print(f"{'FE-HITL':<15} {acc_fehitl:.3f}     {di_fehitl:.2f}   {eod_fehitl:.2f}   {aod_fehitl:.2f}")
    print("="*60)

if __name__ == '__main__':
    main()