"""
baseline_dl.py — scikit-learn implementation of the deep learning baseline.
Compatible with all PyTorch versions; no torch import needed.
Retains DeepLearningModel as a stub for backwards compatibility.
"""

import torch.nn as nn
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from ..utils import set_seed


class DeepLearningModel(nn.Module):
    """
    Stub class for backward compatibility with debiased_hitl.py and fe_hitl.py.
    Actual training is performed by MLPRegressor/MLPClassifier in BaselineDL.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2, task='regression'):
        super().__init__()
        # Do nothing — PyTorch layers are never created
        self.task = task
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class BaselineDL:
    """
    Deep learning baseline using scikit-learn's multi-layer perceptron.
    API mirrors the original PyTorch version exactly.
    """
    def __init__(self, input_dim, task='regression', hidden_dims=None,
                 dropout=0.2, lr=0.001, batch_size=32, epochs=500, patience=10,
                 device='cpu', random_seed=42):
        set_seed(random_seed)
        self.task = task
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64, 32]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_seed = random_seed
        self.model = None        # will hold MLPRegressor / MLPClassifier

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        common_kwargs = dict(
            hidden_layer_sizes=tuple(self.hidden_dims),
            activation='relu',
            solver='adam',
            batch_size=self.batch_size,
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            n_iter_no_change=self.patience,
            random_state=self.random_seed,
            verbose=False,
        )

        if self.task == 'classification':
            self.model = MLPClassifier(**common_kwargs)
        else:
            self.model = MLPRegressor(**common_kwargs)

        # Use validation set for early stopping
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64).ravel()
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.hstack([y_train, y_val])
            self.model.validation_fraction = len(y_val) / len(y_combined)
            self.model.early_stopping = True
            self.model.fit(X_combined, y_combined)
        else:
            self.model.early_stopping = False
            self.model.fit(X_train, y_train)

        # Print convergence info (mirrors PyTorch early stopping message)
        if hasattr(self.model, 'n_iter_'):
            print(f"MLP converged at iteration {self.model.n_iter_}")
        elif hasattr(self.model, 'n_epochs_'):
            print(f"MLP converged after {self.model.n_epochs_} epochs")

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = self.model.predict(X)
        if self.task == 'classification':
            return preds.astype(int)
        return preds