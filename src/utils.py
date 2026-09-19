import numpy as np
import random

try:  # torch is NOT used by any executed experiment; it is only seeded if present
    import torch
except ImportError:
    torch = None

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def compute_regression_metrics(y_true, y_pred):
    """Compute R² and RMSE."""
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def compute_classification_metrics(y_true, y_pred):
    """Compute accuracy."""
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    return acc