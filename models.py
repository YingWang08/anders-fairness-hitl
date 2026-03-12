import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np

# ================== PyTorch Deep Learning Model ==================
class DeepRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze()

class DeepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # logit
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze()

# ================== Scikit-learn Wrappers ==================
class SklearnLRReg(LinearRegression):
    def fit(self, X, y):
        super().fit(X, y)
        return self
    def predict(self, X):
        return super().predict(X)

class SklearnLRClf(LogisticRegression):
    def fit(self, X, y):
        super().fit(X, y)
        return self
    def predict(self, X):
        return super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)

# ================== Debiased-HITL Baseline (Simplified) ==================
class DebiasedHITL:
    """
    Simplified version of Zhang et al. 2021: human feedback on uncertain samples
    combined with reweighting to improve fairness.
    """
    def __init__(self, base_model, fairness_weight=0.1):
        self.base_model = base_model
        self.fairness_weight = fairness_weight
    
    def fit(self, X, y, sensitive_attr):
        # Placeholder: in real implementation, would involve iterative training with human feedback.
        # For reproducibility, we just train a weighted model.
        from sklearn.utils import class_weight
        # compute weights to favor unprivileged group
        sample_weights = np.ones(len(y))
        sample_weights[sensitive_attr == 0] *= (1 + self.fairness_weight)  # unprivileged
        self.base_model.fit(X, y, sample_weight=sample_weights)
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)

# ================== Our FE-HITL Framework ==================
class FEHITL:
    """
    Fairness-Enhanced Human-in-the-Loop Framework
    """
    def __init__(self, base_model, monitor, human_simulator, threshold_di=0.8, threshold_eod=0.1, intervention_ratio=0.1):
        self.base_model = base_model
        self.monitor = monitor  # function to check fairness
        self.human_simulator = human_simulator  # function that simulates human decision
        self.threshold_di = threshold_di
        self.threshold_eod = threshold_eod
        self.intervention_ratio = intervention_ratio
        self.feedback_buffer = []  # experience replay
    
    def fit(self, X_train, y_train, s_train, X_val, y_val, s_val, epochs=50, batch_size=128, lr=0.001):
        # Convert to PyTorch tensors if using neural network
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Assume base_model is a PyTorch model
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.base_model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # for regression
        
        for epoch in range(epochs):
            self.base_model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.base_model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation and fairness monitoring
            self.base_model.eval()
            with torch.no_grad():
                val_pred = self.base_model(torch.FloatTensor(X_val)).numpy()
            # Compute fairness metrics
            from .fairness_metrics import compute_all_metrics
            di, eod, aod = compute_all_metrics(y_val, val_pred, s_val, 
                                                privileged_value=1, unprivileged_value=0, task='regression')
            
            # If fairness below threshold, trigger human intervention on a subset of training data
            if di < self.threshold_di or eod > self.threshold_eod:
                # Select samples for intervention (e.g., those where prediction is unfair)
                # Simplified: pick random 10% of training samples
                n_intervene = int(self.intervention_ratio * len(X_train))
                indices = np.random.choice(len(X_train), n_intervene, replace=False)
                
                # Get predictions for those samples
                with torch.no_grad():
                    pred_intervene = self.base_model(torch.FloatTensor(X_train[indices])).numpy()
                
                # Simulate human decision: get new target values
                new_targets = self.human_simulator(X_train[indices], pred_intervene, s_train[indices])
                
                # Update training labels (in-memory) and add to feedback buffer
                y_train[indices] = new_targets
                self.feedback_buffer.append((X_train[indices], new_targets))
                
                # Recreate loader with updated labels
                dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # After training, optionally fine-tune with feedback buffer (periodic update)
        # For simplicity, we skip here.
        return self
    
    def predict(self, X):
        self.base_model.eval()
        with torch.no_grad():
            return self.base_model(torch.FloatTensor(X)).numpy()