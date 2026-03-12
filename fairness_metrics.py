import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def disparate_impact(y_pred, sensitive_attr, privileged_group=1, unprivileged_group=0, threshold=None):
    """
    Disparate Impact for binary predictions. 
    If y_pred is continuous, binarize using threshold (e.g., median).
    """
    if threshold is not None:
        y_pred_bin = (y_pred > threshold).astype(int)
    else:
        y_pred_bin = y_pred.astype(int)
    
    mask_priv = (sensitive_attr == privileged_group)
    mask_unpriv = (sensitive_attr == unprivileged_group)
    
    if np.sum(mask_priv) == 0 or np.sum(mask_unpriv) == 0:
        return np.nan
    
    rate_priv = np.mean(y_pred_bin[mask_priv])
    rate_unpriv = np.mean(y_pred_bin[mask_unpriv])
    
    if rate_priv == 0:
        return np.inf
    return rate_unpriv / rate_priv

def equal_opportunity_difference(y_true, y_pred, sensitive_attr, privileged_group=1, unprivileged_group=0, threshold=None):
    """
    Equal Opportunity Difference: TPR_priv - TPR_unpriv (absolute)
    """
    if threshold is not None:
        y_pred_bin = (y_pred > threshold).astype(int)
    else:
        y_pred_bin = y_pred.astype(int)
    
    mask_priv = (sensitive_attr == privileged_group)
    mask_unpriv = (sensitive_attr == unprivileged_group)
    
    # True positive rate for each group
    def tpr(group_mask):
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred_bin[group_mask]
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0,1]).ravel()
        return tp / (tp + fn) if (tp+fn) > 0 else 0
    
    tpr_priv = tpr(mask_priv)
    tpr_unpriv = tpr(mask_unpriv)
    return abs(tpr_priv - tpr_unpriv)

def average_odds_difference(y_true, y_pred, sensitive_attr, privileged_group=1, unprivileged_group=0, threshold=None):
    """
    Average Odds Difference: average of |FPR_priv - FPR_unpriv| and |TPR_priv - TPR_unpriv|
    """
    if threshold is not None:
        y_pred_bin = (y_pred > threshold).astype(int)
    else:
        y_pred_bin = y_pred.astype(int)
    
    mask_priv = (sensitive_attr == privileged_group)
    mask_unpriv = (sensitive_attr == unprivileged_group)
    
    def rates(group_mask):
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred_bin[group_mask]
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        return tpr, fpr
    
    tpr_priv, fpr_priv = rates(mask_priv)
    tpr_unpriv, fpr_unpriv = rates(mask_unpriv)
    
    return (abs(tpr_priv - tpr_unpriv) + abs(fpr_priv - fpr_unpriv)) / 2

# For regression tasks, we need to binarize predictions based on a threshold.
# We'll use the median of y_pred as default threshold.
def compute_all_metrics(y_true, y_pred, sensitive_attr, privileged_value, unprivileged_value, task='regression'):
    if task == 'regression':
        threshold = np.median(y_pred)  # binarize at median
    else:
        threshold = None  # y_pred is already binary (probabilities >0.5)
    
    di = disparate_impact(y_pred, sensitive_attr, privileged_value, unprivileged_value, threshold)
    eod = equal_opportunity_difference(y_true, y_pred, sensitive_attr, privileged_value, unprivileged_value, threshold)
    aod = average_odds_difference(y_true, y_pred, sensitive_attr, privileged_value, unprivileged_value, threshold)
    
    return di, eod, aod