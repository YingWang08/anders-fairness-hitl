import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from .fairness_metrics import compute_all_metrics
from .models import DeepRegression, DeepClassifier, SklearnLRReg, SklearnLRClf, DebiasedHITL, FEHITL
from .human_simulation import HumanSimulator
import pandas as pd

def train_and_evaluate(model, X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test, 
                       task='regression', model_type='dl', epochs=50):
    """
    Train a model and return predictions and metrics.
    """
    if task == 'regression':
        if model_type == 'lr':
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
        elif model_type == 'dl':
            # Assume model is a PyTorch model
            # We'll use a simple training loop here
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(torch.FloatTensor(X_train))
                loss = criterion(pred, torch.FloatTensor(y_train))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                y_pred_test = model(torch.FloatTensor(X_test)).numpy()
        elif model_type == 'debiased':
            model.fit(X_train, y_train, s_train)  # debiased expects sensitive attr
            y_pred_test = model.predict(X_test)
        elif model_type == 'fehitl':
            # Our framework needs validation set for monitoring
            model.fit(X_train, y_train, s_train, X_val, y_val, s_val, epochs=epochs)
            y_pred_test = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Compute metrics
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        # Fairness metrics (binarized at median of y_pred_test)
        di, eod, aod = compute_all_metrics(y_test, y_pred_test, s_test, 
                                            privileged_value=1, unprivileged_value=0, task='regression')
        return {'R2': r2, 'RMSE': rmse, 'DI': di, 'EOD': eod, 'AOD': aod}, y_pred_test
    
    else:  # classification
        if model_type == 'lr':
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred_test = (y_pred_proba > 0.5).astype(int)
        elif model_type == 'dl':
            # Similar to regression but with BCE loss
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                logits = model(torch.FloatTensor(X_train))
                loss = criterion(logits, torch.FloatTensor(y_train))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_test))
                y_pred_proba = torch.sigmoid(logits).numpy()
                y_pred_test = (y_pred_proba > 0.5).astype(int)
        elif model_type == 'debiased':
            model.fit(X_train, y_train, s_train)
            y_pred_test = model.predict(X_test)
        elif model_type == 'fehitl':
            model.fit(X_train, y_train, s_train, X_val, y_val, s_val, epochs=epochs)
            y_pred_test = model.predict(X_test)
        else:
            raise ValueError
        
        acc = accuracy_score(y_test, y_pred_test)
        di, eod, aod = compute_all_metrics(y_test, y_pred_test, s_test,
                                            privileged_value=1, unprivileged_value=0, task='classification')
        return {'Accuracy': acc, 'DI': di, 'EOD': eod, 'AOD': aod}, y_pred_test