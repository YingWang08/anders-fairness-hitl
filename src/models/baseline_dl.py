import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ..utils import set_seed

class DeepLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2, task='regression'):
        super().__init__()
        self.task = task
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        if task == 'classification':
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()

class BaselineDL:
    def __init__(self, input_dim, task='regression', hidden_dims=[128, 64, 32],
                 dropout=0.2, lr=0.001, batch_size=32, epochs=500, patience=10,
                 device='cpu', random_seed=42):
        set_seed(random_seed)
        self.task = task
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = DeepLearningModel(self.input_dim, self.hidden_dims, self.dropout, self.task).to(self.device)
        criterion = nn.MSELoss() if self.task == 'regression' else nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation, just use train loss (not recommended)
                pass

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t).cpu().numpy()
        if self.task == 'classification':
            outputs = (outputs >= 0.5).astype(int)
        return outputs