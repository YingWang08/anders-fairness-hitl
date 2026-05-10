import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from .baseline_dl import BaselineDL

class DebiasedHITL:
    """
    Debiased HITL baseline using sklearn MLP.
    Implements simple post-hoc bias mitigation: if DI < threshold,
    predictions for unprivileged group are scaled up by a compensation factor.
    """
    def __init__(self, input_dim, task='regression', device='cpu', random_seed=42,
                 di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1):
        self.input_dim = input_dim
        self.task = task
        self.device = device
        self.random_seed = random_seed
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold
        self.intervention_ratio = intervention_ratio
        self.threshold = None
        self.base_model = None

    def set_binarize_threshold(self, threshold):
        self.threshold = threshold

    def fit(self, X_train, y_train, s_train, X_val, y_val, s_val):
        # Train a plain MLP (same as BaselineDL)
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        if self.task == 'classification':
            self.base_model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                solver='adam', max_iter=200, random_state=self.random_seed
            ).fit(X_train, y_train)
        else:
            self.base_model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                solver='adam', max_iter=200, random_state=self.random_seed
            ).fit(X_train, y_train)

    def predict(self, X, s, context_list=None, apply_intervention=True):
        X = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        if self.task == 'classification':
            preds = preds.astype(int)
            return preds

        if not apply_intervention:
            return preds

        # Simple fairness adjustment: if DI < threshold, boost unprivileged group
        s = np.asarray(s).ravel()
        unpriv_mask = (s == 1)
        priv_mask = (s == 0)

        # Convert to binary for DI calculation
        if self.threshold is None:
            self.threshold = np.median(preds)
        y_bin = (preds > self.threshold).astype(int)
        p_unpriv = np.mean(y_bin[unpriv_mask])
        p_priv = np.mean(y_bin[priv_mask])
        di = p_unpriv / p_priv if p_priv > 0 else 1.0

        if di < self.di_threshold:
            # Increase predictions for unprivileged group
            boost_factor = 1.0 + self.intervention_ratio * (self.di_threshold - di) / self.di_threshold
            preds[unpriv_mask] = preds[unpriv_mask] * boost_factor

        return preds