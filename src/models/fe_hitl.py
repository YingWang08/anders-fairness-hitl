import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ..fairness_metrics import disparate_impact, equal_opportunity_difference
from ..human_simulator import HumanSimulator
from .baseline_dl import DeepLearningModel

class FairnessMonitor:
    def __init__(self, di_threshold=0.8, eod_threshold=0.1):
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold

    def check(self, y_pred, sensitive, y_true=None):
        """Return True if fairness constraints are violated."""
        # For regression, we need to binarize predictions for DI/EOD? We'll use thresholding.
        # Here we assume classification or binarized regression (e.g., above/below median).
        # In agricultural scenario, we binarize based on median allocation.
        # For simplicity, we'll use continuous EOD approximation? Not easy.
        # We'll implement a placeholder: check DI only after binarizing predictions.
        # This should be customized per scenario.
        # In our actual code, we'll compute fairness on the batch level.
        # For now, we'll return a flag based on DI.
        di = disparate_impact(y_pred, sensitive)
        if di is None or np.isnan(di):
            return False
        if di < self.di_threshold:
            return True
        # Optionally check EOD if y_true provided
        if y_true is not None:
            eod = equal_opportunity_difference(y_true, y_pred, sensitive)
            if abs(eod) > self.eod_threshold:
                return True
        return False

class MultiOptionGenerator:
    def __init__(self, base_model, num_options=3):
        self.base_model = base_model
        self.num_options = num_options

    def generate(self, X, sensitive, original_pred):
        """Generate multiple options with varying fairness-efficiency trade-offs."""
        options = []
        # For regression: adjust prediction by scaling factors
        # For classification: adjust threshold
        # Here we implement both with a unified approach: adjust prediction value.
        # We'll generate options that aim to improve DI.
        # Compute current DI on a reference set? Not per instance.
        # Instead, we'll use simple perturbation.
        factors = np.linspace(0.8, 1.2, self.num_options)
        for factor in factors:
            new_val = original_pred * factor
            # Estimate DI after this adjustment (impossible per instance, so we approximate)
            # For simulation, we'll assume it increases with factor for unprivileged group.
            # We'll compute a placeholder DI based on factor.
            # In real code, we would need a separate fairness estimator.
            di_est = min(1.0, factor * 0.9)  # crude
            options.append({
                'value': new_val,
                'di': di_est,
                'efficiency_loss': abs(1 - factor),
                'description': f'Scale by {factor:.2f}'
            })
        return options

class FEHITL:
    def __init__(self, input_dim, task='regression', hidden_dims=[128, 64, 32],
                 dropout=0.2, lr=0.001, batch_size=32, epochs=500, patience=10,
                 device='cpu', random_seed=42,
                 di_threshold=0.8, eod_threshold=0.1, epsilon=0.1):
        self.task = task
        self.epsilon = epsilon
        self.base_model = DeepLearningModel(input_dim, hidden_dims, dropout, task).to(device)
        self.monitor = FairnessMonitor(di_threshold, eod_threshold)
        self.human = HumanSimulator(fairness_threshold=di_threshold)
        self.option_gen = MultiOptionGenerator(self.base_model)
        self.device = device
        self.feedback_buffer = []  # store (X, s, corrected_y, context)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    def fit_base(self, X_train, y_train, X_val=None, y_val=None):
        """Train the base model without human intervention."""
        # Use BaselineDL training logic, but here we have self.base_model
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss() if self.task == 'regression' else nn.BCELoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.base_model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.base_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            if X_val is not None:
                self.base_model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_outputs = self.base_model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.base_model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_model_state is not None:
            self.base_model.load_state_dict(best_model_state)

    def predict(self, X, sensitive, y_true=None, context=None):
        """Make predictions with possible human intervention."""
        self.base_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.base_model(X_t).cpu().numpy()

        final_preds = []
        for i, (pred, si) in enumerate(zip(preds, sensitive)):
            # Determine if intervention needed (simplified: check fairness on this instance? Not possible)
            # We'll use a batch-level check: compute DI on current batch and see if below threshold.
            # For per-instance, we rely on epsilon probability when fairness is violated.
            # Here we compute DI on the batch before intervention (requires all preds)
            # We'll do it outside the loop, but for simplicity we'll just use epsilon.
            # In the paper, intervention is triggered when fairness metrics on a batch fall below threshold.
            # We'll implement that logic in the experiment script, not inside predict.
            # For simplicity in this class, we'll assume that the decision to intervene is given externally.
            # We'll add a flag `intervene` to the method.
            # Let's redesign: the experiment script will call a separate method for batch processing with intervention.
            # We'll keep predict simple: no intervention.
            final_preds.append(pred)
        return np.array(final_preds)

    def predict_with_intervention(self, X, sensitive, context_list=None):
        """Batch prediction with potential human intervention based on fairness monitoring."""
        self.base_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.base_model(X_t).cpu().numpy()

        # 检查批次公平性
        if self.monitor.check(preds, sensitive):
            # 干预触发
            final_preds = []
            for i, (pred, si) in enumerate(zip(preds, sensitive)):
                if np.random.rand() < self.epsilon:
                    # 生成选项
                    options = self.option_gen.generate(X[i], si, pred)
                    # 人类决策
                    ctx = context_list[i] if context_list else {}
                    selected_idx, basis = self.human.decide(options, ctx)
                    corrected = options[selected_idx]['value']
                    final_preds.append(corrected)
                    # 存储反馈
                    self.feedback_buffer.append((X[i], si, corrected, ctx))
                else:
                    final_preds.append(pred)
            # 可选：定期重新训练
            if len(self.feedback_buffer) >= 100:
                self.retrain_with_feedback()
            final_preds = np.array(final_preds)
        else:
            final_preds = preds

        # 分类任务需转换为整数标签
        if self.task == 'classification':
            final_preds = (final_preds >= 0.5).astype(int)
        return final_preds

    def retrain_with_feedback(self):
        """Fine-tune the model using feedback buffer."""
        if len(self.feedback_buffer) < 10:
            return
        # Prepare data from buffer
        X_fb = np.array([fb[0] for fb in self.feedback_buffer])
        y_fb = np.array([fb[2] for fb in self.feedback_buffer])
        # Convert to tensors
        X_t = torch.tensor(X_fb, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_fb, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss() if self.task == 'regression' else nn.BCELoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=self.lr * 0.1)  # smaller LR for fine-tuning

        self.base_model.train()
        for epoch in range(10):  # few epochs
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.base_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        self.base_model.eval()
        # Clear buffer? Keep for future?
        # We'll clear half to maintain diversity
        self.feedback_buffer = self.feedback_buffer[-50:]