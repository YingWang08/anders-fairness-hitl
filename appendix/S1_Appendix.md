# S1 Appendix: Detailed Hyperparameter Settings and Implementation Specifics

## Model Architectures

### Deep Learning Baseline (DL)
- Three-layer fully connected network with hidden sizes: 128, 64, 32.
- Activation: ReLU after each hidden layer.
- Dropout rate: 0.2 after each hidden layer.
- Output layer: linear for regression (agricultural), sigmoid for classification (credit).

### FE-HITL Framework
- Base model same as DL.
- Fairness monitoring layer: computes DI and EOD on validation batch every 10 epochs during training; triggers intervention if DI < 0.8 or EOD > 0.1.
- Multi-option generation:
  - For regression: generates options by scaling predictions by factors [0.85, 0.90, 0.95, 1.05, 1.10, 1.15] (or a subset of 3–5) and selects those that change DI by at least 0.05.
  - For classification: adjusts decision thresholds using Platt scaling and presents alternatives.
- Human decision simulation: rule-based as described in `human_simulator.py` with rules:
  1. Reject options that discriminate against protected group (e.g., region A allocation <70% of region B average).
  2. Prefer options that restore DI ≥0.85 with efficiency loss ≤10%.
  3. Otherwise choose highest DI option.
- Feedback buffer: size 1000, stores (X, s, corrected prediction, context). Retraining triggered every 100 new feedback samples using fine-tuning (10 epochs, LR=0.0001).

## Training Hyperparameters
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 500 with early stopping (patience=10) based on validation loss.
- Loss function: MSE (agricultural), Binary Cross-Entropy (credit).
- Random seed: 42 for all Python, NumPy, PyTorch.

## Intervention Settings
- Intervention ratio ε = 0.1 (10% of instances where fairness threshold is violated).
- Fairness thresholds: DI < 0.8, EOD > 0.1 trigger intervention.

## Data Preprocessing
- Agricultural dataset: features normalized to zero mean and unit variance.
- German Credit: categorical variables converted to integer codes (for simplicity); numerical features standardized.

## Hardware and Software Environment
- Python 3.9
- PyTorch 1.10.0
- scikit-learn 1.0.0
- pandas 1.3.0
- numpy 1.21.0
- Experiments ran on a single NVIDIA Tesla T4 GPU (16GB) and Intel Xeon CPU @ 2.20GHz.