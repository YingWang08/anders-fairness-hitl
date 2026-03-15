from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

class BaselineLR:
    """Wrapper for Logistic Regression (classification) or Linear Regression (regression)."""
    def __init__(self, task='regression'):
        self.task = task
        if task == 'regression':
            self.model = LinearRegression()
        elif task == 'classification':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError("task must be 'regression' or 'classification'")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.task == 'regression':
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if self.task == 'classification':
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError