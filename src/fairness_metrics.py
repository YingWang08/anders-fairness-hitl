import numpy as np

def disparate_impact(y_pred, sensitive, positive_class=1):
    """
    Disparate Impact = P(y_pred=positive | s=unprivileged) / P(y_pred=positive | s=privileged)
    We assume sensitive=1 is privileged, 0 is unprivileged.
    """
    priv = sensitive == 1
    unpriv = sensitive == 0
    if np.sum(priv) == 0 or np.sum(unpriv) == 0:
        return np.nan
    priv_pos = np.mean(y_pred[priv] == positive_class)
    unpriv_pos = np.mean(y_pred[unpriv] == positive_class)
    if priv_pos == 0:
        return np.nan
    return unpriv_pos / priv_pos

def equal_opportunity_difference(y_true, y_pred, sensitive, positive_class=1):
    """
    Equal Opportunity Difference = TPR_unpriv - TPR_priv
    TPR = recall for positive class.
    """
    priv = sensitive == 1
    unpriv = sensitive == 0
    # True positive rate for privileged
    priv_tp = np.sum((y_pred[priv] == positive_class) & (y_true[priv] == positive_class))
    priv_pos_total = np.sum(y_true[priv] == positive_class)
    tpr_priv = priv_tp / priv_pos_total if priv_pos_total > 0 else 0
    # True positive rate for unprivileged
    unpriv_tp = np.sum((y_pred[unpriv] == positive_class) & (y_true[unpriv] == positive_class))
    unpriv_pos_total = np.sum(y_true[unpriv] == positive_class)
    tpr_unpriv = unpriv_tp / unpriv_pos_total if unpriv_pos_total > 0 else 0
    return tpr_unpriv - tpr_priv

def average_odds_difference(y_true, y_pred, sensitive, positive_class=1):
    """
    Average Odds Difference = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]
    """
    priv = sensitive == 1
    unpriv = sensitive == 0
    # TPR
    priv_tp = np.sum((y_pred[priv] == positive_class) & (y_true[priv] == positive_class))
    priv_pos_total = np.sum(y_true[priv] == positive_class)
    tpr_priv = priv_tp / priv_pos_total if priv_pos_total > 0 else 0
    unpriv_tp = np.sum((y_pred[unpriv] == positive_class) & (y_true[unpriv] == positive_class))
    unpriv_pos_total = np.sum(y_true[unpriv] == positive_class)
    tpr_unpriv = unpriv_tp / unpriv_pos_total if unpriv_pos_total > 0 else 0

    # FPR (assuming negative class is 0)
    priv_fp = np.sum((y_pred[priv] == positive_class) & (y_true[priv] != positive_class))
    priv_neg_total = np.sum(y_true[priv] != positive_class)
    fpr_priv = priv_fp / priv_neg_total if priv_neg_total > 0 else 0
    unpriv_fp = np.sum((y_pred[unpriv] == positive_class) & (y_true[unpriv] != positive_class))
    unpriv_neg_total = np.sum(y_true[unpriv] != positive_class)
    fpr_unpriv = unpriv_fp / unpriv_neg_total if unpriv_neg_total > 0 else 0

    return ((tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv)) / 2