"""
FE-HITL Framework source package.
"""
# 导出主要类和函数，方便从包直接导入
from .fairness_metrics import disparate_impact, equal_opportunity_difference, average_odds_difference, compute_all_metrics
from .models import DeepRegression, DeepClassifier, SklearnLRReg, SklearnLRClf, DebiasedHITL, FEHITL
from .human_simulation import HumanSimulator
from .train_eval import train_and_evaluate
from .utils import set_seed, ensure_dir, save_results, load_agriculture_data, load_german_data

__all__ = [
    'disparate_impact', 'equal_opportunity_difference', 'average_odds_difference', 'compute_all_metrics',
    'DeepRegression', 'DeepClassifier', 'SklearnLRReg', 'SklearnLRClf', 'DebiasedHITL', 'FEHITL',
    'HumanSimulator', 'train_and_evaluate',
    'set_seed', 'ensure_dir', 'save_results', 'load_agriculture_data', 'load_german_data'
]