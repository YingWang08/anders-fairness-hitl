import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from .baseline_dl import BaselineDL

class FEHITL:
    """
    Fairness-Enhanced HITL (simulated) using sklearn MLP.
    Implements rule-based human arbitration: selects from candidate
    interventions based on predefined fairness/efficiency trade-offs.
    For simulation, we mimic multi-option generation and value arbitration.
    """
    def __init__(self, input_dim, task='regression', device='cpu', random_seed=42,
                 epsilon=0.1, di_threshold=0.8, eod_threshold=0.1):
        self.input_dim = input_dim
        self.task = task
        self.device = device
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold
        self.base_model = None
        self.threshold = None

    def fit_base(self, X_train, y_train, X_val, y_val):
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

    def predict_with_intervention(self, X, s, context_list=None):
        X = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        s = np.asarray(s).ravel()
        if self.task == 'classification':
            return preds.astype(int)

        if self.threshold is None:
            self.threshold = np.median(preds)

        # Simulate multi-option generation and value arbitration
        unpriv_mask = (s == 1)
        priv_mask = (s == 0)
        y_bin = (preds > self.threshold).astype(int)
        p_unpriv = np.mean(y_bin[unpriv_mask])
        p_priv = np.mean(y_bin[priv_mask])
        di = p_unpriv / p_priv if p_priv > 0 else 1.0

        # If DI below threshold, apply compensation
        if di < self.di_threshold:
            # Generate hypothetical boost (options)
            # For simplicity, we apply a moderate boost consistent with "fairness prioritization rule"
            boost = 1.0 + (self.di_threshold - di) * 0.7  # arbitrary scaling
            preds[unpriv_mask] = preds[unpriv_mask] * boost

        return preds