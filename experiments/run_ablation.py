import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.data_loader import load_agricultural_data
from src.models.fe_hitl import FEHITL
from src.fairness_metrics import disparate_impact, equal_opportunity_difference, average_odds_difference
from src.utils import compute_regression_metrics, set_seed
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def run_ablation_variant(variant_name, config):
    set_seed(42)
    (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test) = load_agricultural_data()
    threshold = np.median(y_train)
    def binarize(y): return (y > threshold).astype(int)

    test_df = pd.read_csv('data/agricultural_test.csv')
    region_test = test_df['region'].values
    avg_allocation_B_train = np.mean(y_train[s_train == 0])
    context_list = [{'region': r, 'avg_allocation_B': avg_allocation_B_train} for r in region_test]

    # 创建 FE-HITL 实例（基本配置）
    fehitl = FEHITL(
        input_dim=X_train.shape[1],
        task='regression',
        device='cpu',
        epsilon=0.1,
        di_threshold=0.8,
        eod_threshold=0.1
    )

    # 根据变体禁用相应模块
    if variant_name == 'w/o HED':
        # 移除人类伦理决策层：将干预概率设为0
        fehitl.epsilon = 0.0
    elif variant_name == 'w/o MOG':
        # 移除多选项生成：干预时只提供一个选项
        fehitl.option_gen.num_options = 1
    elif variant_name == 'w/o F&U':
        # 移除反馈更新：禁用重训练
        fehitl.retrain_with_feedback = lambda: None

    fehitl.fit_base(X_train, y_train, X_val, y_val)
    pred = fehitl.predict_with_intervention(X_test, s_test, context_list=context_list)
    r2, rmse = compute_regression_metrics(y_test, pred)
    di = disparate_impact(binarize(pred), s_test)
    eod = equal_opportunity_difference(binarize(y_test), binarize(pred), s_test)
    aod = average_odds_difference(binarize(y_test), binarize(pred), s_test)
    return r2, rmse, di, eod, aod

def main():
    print("Ablation Study Results")
    variants = ['Full', 'w/o HED', 'w/o MOG', 'w/o F&U']
    results = []
    for var in variants:
        if var == 'Full':
            r2, rmse, di, eod, aod = run_ablation_variant('Full', {})
        else:
            r2, rmse, di, eod, aod = run_ablation_variant(var, {})
        results.append([var, r2, rmse, di, eod, aod])
        print(f"{var}\tR²={r2:.2f}\tRMSE={rmse:.2f}\tDI={di:.2f}\tEOD={eod:.2f}\tAOD={aod:.2f}")

    # 可选：保存结果到 CSV
    # df = pd.DataFrame(results, columns=['Variant', 'R2', 'RMSE', 'DI', 'EOD', 'AOD'])
    # df.to_csv('ablation_results.csv', index=False)

if __name__ == '__main__':
    main()