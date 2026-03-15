# src/models/debiased_hitl.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ..fairness_metrics import disparate_impact, equal_opportunity_difference
from ..human_simulator import HumanSimulator
from .baseline_dl import DeepLearningModel
import copy

class DebiasedHITL:
    """
    Implementation of Debiased-HITL based on Zhang et al. (2021).
    Uses human feedback to correct model predictions and fine-tune the model.
    """
    def __init__(self, input_dim, task='regression', hidden_dims=[128, 64, 32],
                 dropout=0.2, lr=0.001, batch_size=32, epochs=500, patience=10,
                 device='cpu', random_seed=42,
                 di_threshold=0.8, eod_threshold=0.1,
                 intervention_ratio=0.1, feedback_buffer_size=1000,
                 fine_tune_epochs=10, fine_tune_lr_factor=0.1):
        """
        Args:
            input_dim: feature dimension
            task: 'regression' or 'classification'
            hidden_dims: list of hidden layer sizes
            dropout: dropout rate
            lr: learning rate for initial training
            batch_size: batch size
            epochs: max epochs for initial training
            patience: early stopping patience
            device: 'cpu' or 'cuda'
            random_seed: random seed
            di_threshold: threshold for disparate impact to trigger intervention
            eod_threshold: threshold for equal opportunity difference to trigger intervention
            intervention_ratio: probability of intervention when fairness threshold is violated
            feedback_buffer_size: max size of feedback buffer
            fine_tune_epochs: number of epochs for fine-tuning with feedback
            fine_tune_lr_factor: factor to reduce learning rate for fine-tuning
        """
        self.task = task
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold
        self.intervention_ratio = intervention_ratio
        self.feedback_buffer_size = feedback_buffer_size
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr_factor = fine_tune_lr_factor

        # Set seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(random_seed)

        # Initialize base model
        self.base_model = DeepLearningModel(input_dim, hidden_dims, dropout, task).to(self.device)
        # We'll keep a separate copy for fine-tuning? Not necessary, we'll fine-tune the same model.
        # But we need to store the best model after initial training.
        self.best_model_state = None

        # Initialize human simulator
        self.human = HumanSimulator(fairness_threshold=di_threshold)

        # Feedback buffer: list of tuples (X, sensitive, corrected_y, context)
        self.feedback_buffer = []

        # For tracking fairness on validation set during training
        self.val_fairness_history = []

    def _check_fairness(self, y_pred, y_true, sensitive, threshold_binarize=None):
        """Check if fairness constraints are violated."""
        if self.task == 'regression' and threshold_binarize is not None:
            y_pred_bin = (y_pred > threshold_binarize).astype(int)
            y_true_bin = (y_true > threshold_binarize).astype(int)
        else:
            y_pred_bin = y_pred
            y_true_bin = y_true

        di = disparate_impact(y_pred_bin, sensitive)
        if di is None or np.isnan(di):
            return False
        if di < self.di_threshold:
            return True
        if y_true is not None:
            eod = equal_opportunity_difference(y_true_bin, y_pred_bin, sensitive)
            if abs(eod) > self.eod_threshold:
                return True
        return False

    def _generate_options(self, X, sensitive, original_pred, context=None):
        """Generate multiple correction options (simplified)."""
        options = []
        # For regression: scale prediction by factors
        factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        for f in factors:
            new_val = original_pred * f
            # Crude DI estimate (just for option description)
            di_est = min(1.0, f * 0.9)  # placeholder
            options.append({
                'value': new_val,
                'di': di_est,
                'efficiency_loss': abs(1 - f),
                'description': f'Scale by {f:.2f}'
            })
        return options

    def _simulate_intervention(self, X, sensitive, original_pred, context=None):
        """Simulate human intervention on a single instance."""
        options = self._generate_options(X, sensitive, original_pred, context)
        selected_idx, basis = self.human.decide(options, context or {})
        corrected = options[selected_idx]['value']
        return corrected, basis

    def fit(self, X_train, y_train, s_train, X_val=None, y_val=None, s_val=None):
        """
        Train the base model, and periodically use human feedback on validation
        set to fine-tune the model.
        """
        # Convert training data to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss() if self.task == 'regression' else nn.BCELoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # For regression, we need a threshold to binarize for fairness checks
        if self.task == 'regression':
            threshold = np.median(y_train)
        else:
            threshold = None

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

            # Validation and fairness check
            if X_val is not None and y_val is not None:
                self.base_model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_outputs = self.base_model(X_val_t).cpu().numpy()
                    val_loss = criterion(self.base_model(X_val_t), y_val_t).item()

                # Check fairness on validation set
                fairness_violated = self._check_fairness(val_outputs, y_val, s_val, threshold)

                # If fairness violated, simulate interventions on validation set to collect feedback
                if fairness_violated and len(self.feedback_buffer) < self.feedback_buffer_size:
                    # For a subset of validation samples, simulate intervention
                    # We'll use the intervention_ratio to decide how many to intervene
                    n_val = len(X_val)
                    intervene_indices = np.where(np.random.rand(n_val) < self.intervention_ratio)[0]
                    for idx in intervene_indices:
                        X_i = X_val[idx]
                        s_i = s_val[idx]
                        pred_i = val_outputs[idx]
                        # Context: could include region if available (for agricultural)
                        # We'll pass empty context for simplicity
                        corrected, _ = self._simulate_intervention(X_i, s_i, pred_i, context={})
                        # Store in feedback buffer
                        self.feedback_buffer.append((X_i, s_i, corrected))
                        # Limit buffer size
                        if len(self.feedback_buffer) > self.feedback_buffer_size:
                            self.feedback_buffer.pop(0)

                # Early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.base_model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # After each epoch, if we have enough feedback, fine-tune the model
            if len(self.feedback_buffer) >= self.batch_size:
                self._fine_tune_with_feedback()

        # Load best model
        if best_model_state is not None:
            self.base_model.load_state_dict(best_model_state)
        self.best_model_state = best_model_state

    def _fine_tune_with_feedback(self):
        """Fine-tune the base model using feedback buffer."""
        if len(self.feedback_buffer) < self.batch_size:
            return
        # Prepare feedback data
        X_fb = np.array([fb[0] for fb in self.feedback_buffer])
        y_fb = np.array([fb[2] for fb in self.feedback_buffer])
        X_fb_t = torch.tensor(X_fb, dtype=torch.float32).to(self.device)
        y_fb_t = torch.tensor(y_fb, dtype=torch.float32).to(self.device)
        fb_dataset = TensorDataset(X_fb_t, y_fb_t)
        fb_loader = DataLoader(fb_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss() if self.task == 'regression' else nn.BCELoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=self.lr * self.fine_tune_lr_factor)

        self.base_model.train()
        for epoch in range(self.fine_tune_epochs):
            for X_batch, y_batch in fb_loader:
                optimizer.zero_grad()
                outputs = self.base_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        self.base_model.eval()
        # Optionally, clear some feedback to maintain diversity
        # Keep most recent half
        self.feedback_buffer = self.feedback_buffer[-self.feedback_buffer_size//2:]

    def predict(self, X, sensitive=None, context_list=None, apply_intervention=False):
        """
        Predict on new data. If apply_intervention is True, may trigger human intervention
        based on fairness check on the batch.
        """
        self.base_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.base_model(X_t).cpu().numpy()

        # 对于分类任务，如果预测的是概率，在后续使用前需要转换为类别标签
        # 我们将在最后统一处理，但干预过程中可能需要概率值
        # 因此我们保留原始预测，在返回前根据任务类型转换

        if not apply_intervention or sensitive is None:
            # 无干预，直接返回，但分类任务需转换为类别
            if self.task == 'classification':
                return (preds >= 0.5).astype(int)
            return preds

        # 检查批次公平性（需要二值化预测）
        if self.task == 'regression':
            if not hasattr(self, '_binarize_threshold'):
                # 如果没有设置阈值，跳过干预
                return preds
            preds_bin = (preds > self._binarize_threshold).astype(int)
        else:
            preds_bin = (preds >= 0.5).astype(int)

        di = disparate_impact(preds_bin, sensitive)
        if di is not None and not np.isnan(di) and di < self.di_threshold:
            # 公平性违反，对部分样本进行干预
            final_preds = []
            for i, (pred, si) in enumerate(zip(preds, sensitive)):
                if np.random.rand() < self.intervention_ratio:
                    ctx = context_list[i] if context_list else {}
                    corrected, _ = self._simulate_intervention(X[i], si, pred, ctx)
                    final_preds.append(corrected)
                    # 可选：添加到反馈缓冲区
                    if len(self.feedback_buffer) < self.feedback_buffer_size:
                        self.feedback_buffer.append((X[i], si, corrected))
                else:
                    final_preds.append(pred)
            final_preds = np.array(final_preds)
            # 如果是分类任务，将最终预测转换为类别标签
            if self.task == 'classification':
                final_preds = (final_preds >= 0.5).astype(int)
            return final_preds
        else:
            # 无干预，直接返回，分类任务需转换
            if self.task == 'classification':
                return (preds >= 0.5).astype(int)
            return preds

    def set_binarize_threshold(self, threshold):
        """Set threshold for binarizing regression predictions."""
        self._binarize_threshold = threshold