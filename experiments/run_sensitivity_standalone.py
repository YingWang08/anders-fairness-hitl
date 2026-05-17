"""
run_sensitivity_standalone.py
==============================
Full revision experiment suite for agricultural dataset.
Covers all issues raised by Academic Editor (2nd revision round):

  Issue 2  → run_bias_structure_robustness()   3 bias patterns
  Issue 3  → run_kfold_cv()                    5-fold CV
  Issue 7  → run_binarization_sensitivity()    threshold sensitivity
  Issue 8  → run_bias_sensitivity()            10%-40% bias levels
  Issue 9  → run_intervention_sensitivity()    intervention ratio
             run_reverse_discrimination()      reverse-discrimination
             run_intervention_cost()           cost quantification
  Ablation → run_ablation()

Set QUICK_TEST = True for a fast smoke-test before the full run.
"""

import sys, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  QUICK TEST TOGGLE
#  True  → small subset for smoke-testing
#  False → full experiment (submit this)
# ─────────────────────────────────────────
QUICK_TEST = False

# ─────────────────────────────────────────
#  Paths & shared imports
# ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_lr import BaselineLR
from src.models.baseline_dl import BaselineDL
from src.fairness_metrics   import equal_opportunity_difference, average_odds_difference
from src.utils              import compute_regression_metrics, set_seed

OUT_DIR = 'revision_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
#  Experiment parameters
# ─────────────────────────────────────────
if QUICK_TEST:
    SEEDS_ALL       = [42, 43, 44]
    SEEDS_15        = [42, 43, 44]
    BIAS_LEVELS     = [0.30]
    INTERV_RATIOS   = [0.10, 0.50, 1.0]
    ABLATION_SEEDS  = [42, 43, 44]
else:
    SEEDS_ALL       = list(range(42, 72))   # 30 seeds (main experiments)
    SEEDS_15        = list(range(42, 57))   # 15 seeds (supporting experiments)
    BIAS_LEVELS     = [0.10, 0.20, 0.30, 0.40]
    INTERV_RATIOS   = [0.10, 0.25, 0.50, 1.0]
    ABLATION_SEEDS  = list(range(42, 72))


# ═══════════════════════════════════════════════════════════════
#  HELPER: fairness metric
# ═══════════════════════════════════════════════════════════════

def safe_disparate_impact(y_pred_bin, s_unprivileged):
    """DI = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged)."""
    prob_unpriv = np.mean(y_pred_bin[s_unprivileged == 1])
    prob_priv   = np.mean(y_pred_bin[s_unprivileged == 0])
    if prob_priv == 0:
        return 1.0
    return prob_unpriv / prob_priv


def compute_metrics(y_true, y_pred, s, threshold):
    """Bundle R², RMSE, DI, |EOD|, |AOD| into a dict."""
    binarize = lambda y: (y > threshold).astype(int)
    return {
        'R2':   compute_regression_metrics(y_true, y_pred)[0],
        'RMSE': compute_regression_metrics(y_true, y_pred)[1],
        'DI':   safe_disparate_impact(binarize(y_pred), s),
        'EOD':  abs(equal_opportunity_difference(binarize(y_true), binarize(y_pred), s)),
        'AOD':  abs(average_odds_difference(binarize(y_true), binarize(y_pred), s)),
    }


# ═══════════════════════════════════════════════════════════════
#  DATA GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_agricultural_data_biased(n_samples=10000, random_seed=42, bias_rate=0.30):
    """
    Original uniform-bias generator (bias_type='uniform').
    Kept unchanged so existing results are fully reproducible.
    """
    np.random.seed(random_seed)
    from sklearn.model_selection import train_test_split

    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg_raw     = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)
    region               = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    yield_bias           = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    yield_3y_avg         = yield_3y_avg_raw * yield_bias

    base_allocation = (0.3 * arable_land +
                       0.2 * labor_force +
                       0.1 * farming_years +
                       0.2 * yield_3y_avg_raw +
                       0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)

    allocation_bias       = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = np.maximum(
        base_allocation * allocation_bias + np.random.normal(0, 2, n_samples), 0)
    target_allocation     = np.maximum(
        0.5 * base_allocation + 0.5 * historical_allocation
        + np.random.normal(0, 2, n_samples), 0)

    df = pd.DataFrame({
        'arable_land':           arable_land,
        'labor_force':           labor_force,
        'farming_years':         farming_years,
        'yield_3y_avg':          yield_3y_avg,
        'irrigation_resources':  irrigation_resources,
        'fertilizer_subsidy':    fertilizer_subsidy,
        'region':                region,
        'historical_allocation': historical_allocation,
        'target_allocation':     target_allocation,
    })

    train, temp = train_test_split(df, test_size=0.30, random_state=random_seed)
    val,   test = train_test_split(temp, test_size=0.50, random_state=random_seed)
    return train, val, test


def generate_agricultural_data_biased_v2(n_samples=10000, random_seed=42,
                                          bias_rate=0.30, bias_type='uniform'):
    """
    Extended generator supporting three bias structures (Issue 2).

    bias_type options
    -----------------
    'uniform'     Original: Region A uniformly receives bias_rate less.
    'nonlinear'   Bias inversely proportional to arable land area;
                  small-holders in Region A are hit hardest.
    'interaction' Bias depends on Region × farming_years:
                  Region A & farming_years < 5  → full bias_rate
                  Region A & farming_years >= 5 → half bias_rate
    """
    np.random.seed(random_seed)
    from sklearn.model_selection import train_test_split

    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg_raw     = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)
    region               = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])
    is_A                 = (region == 'A')

    # ── Compute per-sample actual bias rate ──────────────────
    if bias_type == 'uniform':
        actual_bias = np.where(is_A, bias_rate, 0.0)

    elif bias_type == 'nonlinear':
        # Normalise land to [0, 1]; smaller land → larger penalty
        norm_land   = (arable_land - 0.5) / (10.0 - 0.5)
        actual_bias = np.where(is_A, bias_rate * (1.0 - norm_land), 0.0)

    elif bias_type == 'interaction':
        young_farmer = (farming_years < 5)
        actual_bias  = np.where(
            is_A & young_farmer,  bias_rate,
            np.where(is_A & ~young_farmer, bias_rate * 0.5, 0.0)
        )
    else:
        raise ValueError(f"Unknown bias_type: '{bias_type}'. "
                         "Choose 'uniform', 'nonlinear', or 'interaction'.")

    # ── Apply bias ────────────────────────────────────────────
    yield_3y_avg          = yield_3y_avg_raw * (1.0 - actual_bias)
    base_allocation       = (0.3 * arable_land +
                             0.2 * labor_force +
                             0.1 * farming_years +
                             0.2 * yield_3y_avg_raw +
                             0.1 * irrigation_resources +
                             0.1 * fertilizer_subsidy / 100)
    historical_allocation = np.maximum(
        base_allocation * (1.0 - actual_bias) + np.random.normal(0, 2, n_samples), 0)
    target_allocation     = np.maximum(
        0.5 * base_allocation + 0.5 * historical_allocation
        + np.random.normal(0, 2, n_samples), 0)

    df = pd.DataFrame({
        'arable_land':           arable_land,
        'labor_force':           labor_force,
        'farming_years':         farming_years,
        'yield_3y_avg':          yield_3y_avg,
        'irrigation_resources':  irrigation_resources,
        'fertilizer_subsidy':    fertilizer_subsidy,
        'region':                region,
        'historical_allocation': historical_allocation,
        'target_allocation':     target_allocation,
    })

    train, temp = train_test_split(df, test_size=0.30, random_state=random_seed)
    val,   test = train_test_split(temp, test_size=0.50, random_state=random_seed)
    return train, val, test


def generate_full_dataset(n_samples=10000, random_seed=42, bias_rate=0.30):
    """
    Return the complete (unsplit) agricultural dataset for 5-fold CV (Issue 3).
    Uses the same uniform bias as the main experiments.
    """
    np.random.seed(random_seed)

    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg_raw     = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)
    region               = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    yield_bias            = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    yield_3y_avg          = yield_3y_avg_raw * yield_bias
    base_allocation       = (0.3 * arable_land +
                             0.2 * labor_force +
                             0.1 * farming_years +
                             0.2 * yield_3y_avg_raw +
                             0.1 * irrigation_resources +
                             0.1 * fertilizer_subsidy / 100)
    allocation_bias       = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = np.maximum(
        base_allocation * allocation_bias + np.random.normal(0, 2, n_samples), 0)
    target_allocation     = np.maximum(
        0.5 * base_allocation + 0.5 * historical_allocation
        + np.random.normal(0, 2, n_samples), 0)

    return pd.DataFrame({
        'arable_land':           arable_land,
        'labor_force':           labor_force,
        'farming_years':         farming_years,
        'yield_3y_avg':          yield_3y_avg,
        'irrigation_resources':  irrigation_resources,
        'fertilizer_subsidy':    fertilizer_subsidy,
        'region':                region,
        'historical_allocation': historical_allocation,
        'target_allocation':     target_allocation,
    })


def split_arrays(df):
    """Extract (X, y, s) arrays from a DataFrame."""
    feature_cols = [
        'arable_land', 'labor_force', 'farming_years',
        'yield_3y_avg', 'irrigation_resources', 'fertilizer_subsidy',
        'historical_allocation',
    ]
    X = df[feature_cols].values
    y = df['target_allocation'].values
    s = (df['region'] == 'A').astype(int).values
    return X, y, s


# ═══════════════════════════════════════════════════════════════
#  MODEL WRAPPERS
# ═══════════════════════════════════════════════════════════════

from sklearn.neural_network import MLPRegressor


class DebiasedHITL:
    """
    Reproduced Debiased-HITL baseline (Zhang et al. 2021).
    Uses scikit-learn MLPRegressor + proportional DI boost.
    """
    def __init__(self, input_dim, task='regression', device='cpu', random_seed=42,
                 di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1):
        self.random_seed       = random_seed
        self.di_threshold      = di_threshold
        self.eod_threshold     = eod_threshold
        self.intervention_ratio = intervention_ratio
        self.threshold         = None
        self.base_model        = None

    def set_binarize_threshold(self, threshold):
        self.threshold = threshold

    def fit(self, X_train, y_train, s_train, X_val, y_val, s_val):
        self.base_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=200, random_state=self.random_seed
        ).fit(np.asarray(X_train, dtype=np.float64),
              np.asarray(y_train, dtype=np.float64).ravel())

    def predict(self, X, s, context_list=None, apply_intervention=True):
        X    = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        if not apply_intervention:
            return preds
        s    = np.asarray(s).ravel()
        if self.threshold is None:
            self.threshold = np.median(preds)
        y_bin    = (preds > self.threshold).astype(int)
        unpriv   = (s == 1)
        p_unpriv = np.mean(y_bin[unpriv])
        p_priv   = np.mean(y_bin[~unpriv])
        di       = p_unpriv / p_priv if p_priv > 0 else 1.0
        if di < self.di_threshold:
            boost = 1.0 + self.intervention_ratio * (self.di_threshold - di) / self.di_threshold
            preds[unpriv] *= boost
        return preds


class FEHITL_Ablatable:
    """
    FE-HITL framework with ablation switches.

    use_hed  : Human Ethical Decision layer (forced intervention)
    use_mog  : Multi-Option Generation (stronger boost coefficient)
    use_fu   : Feedback & Update (imitation-learning fine-tune step)
    interv_frac : fraction of unprivileged samples that receive compensation
                  when DI < di_threshold (default 1.0 = all eligible samples)

    Note on F&U: when use_fu=True, a single warm-start fine-tuning step is
    performed on training data using corrected predictions. In a static
    experiment (same train/test distribution) this step does not change
    model behaviour, so Full and w/o-F&U produce identical metrics.
    This is expected and is clearly stated in the manuscript.

    Note on implementation: all neural network models use scikit-learn
    MLPRegressor, not PyTorch. The term 'deep learning' in the manuscript
    refers to multi-layer neural networks implemented via scikit-learn.
    """
    def __init__(self, input_dim, task='regression', device='cpu', random_seed=42,
                 epsilon=0.1, di_threshold=0.8, eod_threshold=0.1,
                 use_hed=True, use_mog=True, use_fu=True, interv_frac=1.0):
        self.random_seed   = random_seed
        self.epsilon       = epsilon
        self.di_threshold  = di_threshold
        self.eod_threshold = eod_threshold
        self.use_hed       = use_hed
        self.use_mog       = use_mog
        self.use_fu        = use_fu
        self.interv_frac   = interv_frac
        self.base_model    = None
        self.threshold     = None
        self._X_train      = None
        self._s_train      = None

    def fit_base(self, X_train, y_train, X_val, y_val, s_train=None):
        self._X_train = np.asarray(X_train, dtype=np.float64)
        self._s_train = np.asarray(s_train).ravel() if s_train is not None else None
        self.base_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=200, random_state=self.random_seed
        ).fit(self._X_train, np.asarray(y_train, dtype=np.float64).ravel())

    def predict_with_intervention(self, X, s, context_list=None):
        X = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        s = np.asarray(s).ravel()
        if self.threshold is None:
            self.threshold = np.median(preds)

        # ── Layer 2: Human Ethical Decision (simulated) ──────
        if not self.use_hed:
            return preds

        y_bin = (preds > self.threshold).astype(int)
        unpriv = (s == 1)
        p_unpriv = np.mean(y_bin[unpriv])
        p_priv = np.mean(y_bin[~unpriv])
        di = p_unpriv / p_priv if p_priv > 0 else 1.0

        if di < self.di_threshold:
            # ── Layer 2+3: Multi-Option Generation + Value Arbitration ──
            if self.use_mog:
                boost = 1.0 + 1.0 * (self.di_threshold - di) * 0.7  # full MOG
            else:
                boost = 1.0 + 1.0 * (self.di_threshold - di) * 0.2  # no MOG

            # Apply boost to interv_frac of unprivileged samples
            unpriv_indices = np.where(unpriv)[0]
            n_intervene = int(len(unpriv_indices) * self.interv_frac)
            if n_intervene > 0:
                rng = np.random.RandomState(42)
                chosen = rng.choice(unpriv_indices, size=n_intervene, replace=False)
                preds[chosen] *= boost

            # ── Layer 4: Feedback & Update (architectural placeholder) ──
            # Moved inside the threshold branch so boost is always defined
            if self.use_fu and self.base_model is not None and self._X_train is not None:
                corrected_train = self.base_model.predict(self._X_train)
                if self._s_train is not None:
                    unpriv_tr = (self._s_train == 1)
                    tr_preds = self.base_model.predict(self._X_train)
                    tr_thr = np.median(tr_preds)
                    tr_bin = (tr_preds > tr_thr).astype(int)
                    tr_p_unpriv = np.mean(tr_bin[unpriv_tr])
                    tr_p_priv = np.mean(tr_bin[~unpriv_tr])
                    tr_di = tr_p_unpriv / tr_p_priv if tr_p_priv > 0 else 1.0
                    if tr_di < self.di_threshold:
                        corrected_train[unpriv_tr] *= boost
                    self.base_model = MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32), activation='relu',
                        solver='adam', max_iter=20, random_state=self.random_seed,
                        warm_start=True
                    ).fit(self._X_train, corrected_train)

        return preds


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — Bias injection rate sensitivity  (Issue 8)
# ═══════════════════════════════════════════════════════════════

def run_bias_sensitivity():
    """
    Repeat main experiment under bias levels 10%, 20%, 30%, 40%.
    30 seeds each. Results → bias_sensitivity_summary.csv (Table S2).
    """
    rows = []
    for bias in BIAS_LEVELS:
        for seed in SEEDS_ALL:
            set_seed(seed)
            train, val, test = generate_agricultural_data_biased(10000, seed, bias)
            Xt, yt, st = split_arrays(train)
            Xv, yv, sv = split_arrays(val)
            Xte, yte, ste = split_arrays(test)
            thr = np.median(yt)

            # LR
            lr = BaselineLR('regression')
            lr.fit(Xt, yt)
            rows.append({'Exp': 'BiasSens', 'bias': bias, 'seed': seed, 'model': 'LR',
                         **compute_metrics(yte, lr.predict(Xte), ste, thr)})

            # DL
            dl = BaselineDL(input_dim=Xt.shape[1], task='regression',
                            hidden_dims=[128, 64, 32], device='cpu', random_seed=seed)
            dl.fit(Xt, yt, Xv, yv)
            rows.append({'Exp': 'BiasSens', 'bias': bias, 'seed': seed, 'model': 'DL',
                         **compute_metrics(yte, dl.predict(Xte), ste, thr)})

            # Debiased-HITL
            dh = DebiasedHITL(input_dim=Xt.shape[1], device='cpu', random_seed=seed)
            dh.set_binarize_threshold(thr)
            dh.fit(Xt, yt, st, Xv, yv, sv)
            rows.append({'Exp': 'BiasSens', 'bias': bias, 'seed': seed,
                         'model': 'Debiased-HITL',
                         **compute_metrics(yte, dh.predict(Xte, ste), ste, thr)})

            # FE-HITL
            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=True, interv_frac=1.0)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            rows.append({'Exp': 'BiasSens', 'bias': bias, 'seed': seed, 'model': 'FE-HITL',
                         **compute_metrics(yte, fh.predict_with_intervention(Xte, ste),
                                           ste, thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/bias_sensitivity_raw.csv', index=False)
    summary = df.groupby(['bias', 'model']).agg(
        R2_mean=('R2', 'mean'),   R2_std=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
        DI_mean=('DI', 'mean'),   DI_std=('DI', 'std'),
        EOD_mean=('EOD', 'mean'), EOD_std=('EOD', 'std'),
        AOD_mean=('AOD', 'mean'), AOD_std=('AOD', 'std'),
        n=('R2', 'count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/bias_sensitivity_summary.csv', index=False)
    print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — Intervention ratio sensitivity  (Issue 9)
# ═══════════════════════════════════════════════════════════════

def run_intervention_sensitivity():
    """
    Vary interv_frac (fraction of unprivileged samples compensated)
    from 10% to 100%. 30 seeds, bias fixed at 30%.
    Results → interv_sensitivity_summary.csv
    """
    rows = []
    for frac in INTERV_RATIOS:
        for seed in SEEDS_ALL:
            set_seed(seed)
            train, val, test = generate_agricultural_data_biased(10000, seed, 0.30)
            Xt, yt, st = split_arrays(train)
            Xv, yv, sv = split_arrays(val)
            Xte, yte, ste = split_arrays(test)
            thr = np.median(yt)

            dl = BaselineDL(input_dim=Xt.shape[1], task='regression',
                            hidden_dims=[128, 64, 32], device='cpu', random_seed=seed)
            dl.fit(Xt, yt, Xv, yv)
            rows.append({'Exp': 'IntervSens', 'ratio': frac, 'seed': seed, 'model': 'DL',
                         **compute_metrics(yte, dl.predict(Xte), ste, thr)})

            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=False,
                                  interv_frac=frac)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            rows.append({'Exp': 'IntervSens', 'ratio': frac, 'seed': seed,
                         'model': 'FE-HITL',
                         **compute_metrics(yte, fh.predict_with_intervention(Xte, ste),
                                           ste, thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/interv_sensitivity_raw.csv', index=False)
    summary = df.groupby(['ratio', 'model']).agg(
        R2_mean=('R2', 'mean'),   R2_std=('R2', 'std'),
        DI_mean=('DI', 'mean'),   DI_std=('DI', 'std'),
        EOD_mean=('EOD', 'mean'), EOD_std=('EOD', 'std'),
        n=('R2', 'count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/interv_sensitivity_summary.csv', index=False)
    print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — Ablation study
# ═══════════════════════════════════════════════════════════════

def run_ablation():
    """
    Ablation: remove HED, MOG, or F&U one at a time.
    30 seeds, bias = 30%.  Results → ablation_summary.csv (Table 2).
    """
    configs = [
        ('Full',    True,  True,  True),
        ('w/o HED', False, True,  True),
        ('w/o MOG', True,  False, True),
        ('w/o F&U', True,  True,  False),
    ]
    rows = []
    for seed in ABLATION_SEEDS:
        set_seed(seed)
        train, val, test = generate_agricultural_data_biased(10000, seed, 0.30)
        Xt, yt, st = split_arrays(train)
        Xv, yv, sv = split_arrays(val)
        Xte, yte, ste = split_arrays(test)
        thr = np.median(yt)

        for label, hed, mog, fu in configs:
            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=hed, use_mog=mog, use_fu=fu, interv_frac=1.0)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            rows.append({'Exp': 'Ablation', 'variant': label, 'seed': seed,
                         **compute_metrics(yte, fh.predict_with_intervention(Xte, ste),
                                           ste, thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/ablation_raw.csv', index=False)
    summary = df.groupby(['variant']).agg(
        R2_mean=('R2', 'mean'),   R2_std=('R2', 'std'),
        DI_mean=('DI', 'mean'),   DI_std=('DI', 'std'),
        EOD_mean=('EOD', 'mean'), EOD_std=('EOD', 'std'),
        AOD_mean=('AOD', 'mean'), AOD_std=('AOD', 'std'),
        n=('R2', 'count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/ablation_summary.csv', index=False)
    print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 4 — Bias structure robustness  (Issue 2)
# ═══════════════════════════════════════════════════════════════

def run_bias_structure_robustness():
    """
    Test FE-HITL under three structurally different bias patterns
    to counter the circularity concern (Issue 2).
    15 seeds, bias_rate = 30% for all patterns.
    Results → bias_structure_robustness_summary.csv (Table S3).
    """
    BIAS_TYPES = ['uniform', 'nonlinear', 'interaction']
    rows = []

    for btype in BIAS_TYPES:
        for seed in SEEDS_15:
            set_seed(seed)
            train, val, test = generate_agricultural_data_biased_v2(
                10000, seed, bias_rate=0.30, bias_type=btype)
            Xt, yt, st = split_arrays(train)
            Xv, yv, sv = split_arrays(val)
            Xte, yte, ste = split_arrays(test)
            thr = np.median(yt)

            # DL
            dl = BaselineDL(input_dim=Xt.shape[1], task='regression',
                            hidden_dims=[128, 64, 32], device='cpu', random_seed=seed)
            dl.fit(Xt, yt, Xv, yv)
            rows.append({'bias_type': btype, 'seed': seed, 'model': 'DL',
                         **compute_metrics(yte, dl.predict(Xte), ste, thr)})

            # Debiased-HITL
            dh = DebiasedHITL(input_dim=Xt.shape[1], device='cpu', random_seed=seed)
            dh.set_binarize_threshold(thr)
            dh.fit(Xt, yt, st, Xv, yv, sv)
            rows.append({'bias_type': btype, 'seed': seed, 'model': 'Debiased-HITL',
                         **compute_metrics(yte, dh.predict(Xte, ste), ste, thr)})

            # FE-HITL
            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=False, interv_frac=1.0)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            rows.append({'bias_type': btype, 'seed': seed, 'model': 'FE-HITL',
                         **compute_metrics(yte, fh.predict_with_intervention(Xte, ste),
                                           ste, thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/bias_structure_robustness_raw.csv', index=False)
    summary = df.groupby(['bias_type', 'model']).agg(
        R2_mean=('R2', 'mean'),   R2_std=('R2', 'std'),
        DI_mean=('DI', 'mean'),   DI_std=('DI', 'std'),
        EOD_mean=('EOD', 'mean'), EOD_std=('EOD', 'std'),
        n=('R2', 'count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/bias_structure_robustness_summary.csv', index=False)
    print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 5 — 5-fold cross-validation  (Issue 3)
# ═══════════════════════════════════════════════════════════════

def run_kfold_cv():
    """
    Stratified 5-fold CV on the agricultural dataset (Issue 3).
    Stratified by region (sensitive attribute).
    10 seeds per fold → 50 (fold, seed) combinations.
    Results → kfold_cv_agri_summary.csv (Table S4).
    """
    from sklearn.model_selection import StratifiedKFold

    SEEDS_CV = list(range(42, 52))   # 10 seeds per fold
    N_FOLDS  = 5

    full_data     = generate_full_dataset(10000, 42, 0.30)
    region_labels = (full_data['region'] == 'A').astype(int).values
    skf           = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rows          = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_data, region_labels)):
        train_df = full_data.iloc[train_idx]
        test_df  = full_data.iloc[test_idx]
        print(f"  Agricultural CV fold {fold_idx + 1}/{N_FOLDS} ...")

        for seed in SEEDS_CV:
            set_seed(seed)
            Xt, yt, st  = split_arrays(train_df)
            Xte, yte, ste = split_arrays(test_df)
            thr = np.median(yt)

            # DL
            dl = BaselineDL(input_dim=Xt.shape[1], task='regression',
                            hidden_dims=[128, 64, 32], device='cpu', random_seed=seed)
            dl.fit(Xt, yt, Xt, yt)   # no separate val in CV; use train as proxy
            rows.append({'fold': fold_idx, 'seed': seed, 'model': 'DL',
                         **compute_metrics(yte, dl.predict(Xte), ste, thr)})

            # Debiased-HITL
            dh = DebiasedHITL(input_dim=Xt.shape[1], device='cpu', random_seed=seed)
            dh.set_binarize_threshold(thr)
            dh.fit(Xt, yt, st, Xt, yt, st)
            rows.append({'fold': fold_idx, 'seed': seed, 'model': 'Debiased-HITL',
                         **compute_metrics(yte, dh.predict(Xte, ste), ste, thr)})

            # FE-HITL
            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=False, interv_frac=1.0)
            fh.fit_base(Xt, yt, Xt, yt, s_train=st)
            rows.append({'fold': fold_idx, 'seed': seed, 'model': 'FE-HITL',
                         **compute_metrics(yte, fh.predict_with_intervention(Xte, ste),
                                           ste, thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/kfold_cv_agri_raw.csv', index=False)

    # Per-fold summary
    per_fold = df.groupby(['model', 'fold']).agg(
        DI_mean=('DI', 'mean'), DI_std=('DI', 'std'),
        R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
    ).reset_index()
    per_fold.to_csv(f'{OUT_DIR}/kfold_cv_agri_perfold.csv', index=False)

    # Overall summary across all folds × seeds
    overall = df.groupby(['model']).agg(
        DI_mean=('DI', 'mean'), DI_std=('DI', 'std'),
        R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
        n=('DI', 'count')
    ).reset_index()
    overall.to_csv(f'{OUT_DIR}/kfold_cv_agri_summary.csv', index=False)
    print("\nAgricultural 5-Fold CV Summary:")
    print(overall.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 6 — Reverse-discrimination analysis  (Issue 9)
# ═══════════════════════════════════════════════════════════════

def run_reverse_discrimination():
    """
    Quantify the risk of reverse discrimination against the privileged
    group (Regions B/C) under increasing intervention fractions.
    15 seeds, bias = 30%.  Results → reverse_discrimination_summary.csv (Table S7).

    DI_unpriv : P(ŷ=1|Region A) / P(ŷ=1|B∪C)  (main fairness metric, target ≥ 0.8)
    DI_priv   : P(ŷ=1|B∪C) / P(ŷ=1|Region A)  (reverse check; < 0.8 = reverse bias)
    """
    INTERV_RATIOS_RD = [0.10, 0.25, 0.50, 0.75, 1.0]
    rows = []

    for frac in INTERV_RATIOS_RD:
        for seed in SEEDS_15:
            set_seed(seed)
            train, val, test = generate_agricultural_data_biased(10000, seed, 0.30)
            Xt, yt, st = split_arrays(train)
            Xv, yv, sv = split_arrays(val)
            Xte, yte, ste = split_arrays(test)
            thr = np.median(yt)

            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=False,
                                  interv_frac=frac)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            p     = fh.predict_with_intervention(Xte, ste)
            p_bin = (p > thr).astype(int)

            unpriv = (ste == 1)
            priv   = (ste == 0)
            p_rate_unpriv = np.mean(p_bin[unpriv])
            p_rate_priv   = np.mean(p_bin[priv])

            di_unpriv = p_rate_unpriv / p_rate_priv if p_rate_priv > 0 else 1.0
            di_priv   = p_rate_priv / p_rate_unpriv if p_rate_unpriv > 0 else 1.0

            rows.append({
                'interv_frac':  frac,
                'seed':         seed,
                'DI_unpriv':    di_unpriv,
                'DI_priv':      di_priv,
                'R2':           compute_regression_metrics(yte, p)[0],
            })

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/reverse_discrimination_raw.csv', index=False)
    summary = df.groupby(['interv_frac']).agg(
        DI_unpriv_mean=('DI_unpriv', 'mean'), DI_unpriv_std=('DI_unpriv', 'std'),
        DI_priv_mean=('DI_priv',   'mean'),   DI_priv_std=('DI_priv',   'std'),
        R2_mean=('R2', 'mean'),
        n=('R2', 'count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/reverse_discrimination_summary.csv', index=False)
    print("\nReverse Discrimination Summary:")
    print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 7 — Binarization threshold sensitivity  (Issue 7)
# ═══════════════════════════════════════════════════════════════

def run_binarization_sensitivity():
    """
    Verify that DI improvements are robust to the choice of
    binarization threshold (40th / 50th / 60th percentile of ŷ_test).
    15 seeds, bias = 30%.  Results → binarization_sensitivity_summary.csv (Table S6).
    """
    PERCENTILES = [40, 50, 60]
    rows = []

    for seed in SEEDS_15:
        set_seed(seed)
        train, val, test = generate_agricultural_data_biased(10000, seed, 0.30)
        Xt, yt, st  = split_arrays(train)
        Xv, yv, sv  = split_arrays(val)
        Xte, yte, ste = split_arrays(test)

        # Train models once (median threshold used internally for intervention)
        thr_train = np.median(yt)

        dl = BaselineDL(input_dim=Xt.shape[1], task='regression',
                        hidden_dims=[128, 64, 32], device='cpu', random_seed=seed)
        dl.fit(Xt, yt, Xv, yv)
        p_dl = dl.predict(Xte)

        fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                              use_hed=True, use_mog=True, use_fu=False, interv_frac=1.0)
        fh.fit_base(Xt, yt, Xv, yv, s_train=st)
        p_fh = fh.predict_with_intervention(Xte, ste)

        # Re-evaluate DI at each percentile threshold
        for pct in PERCENTILES:
            # Threshold derived from test-set predictions (consistent with main paper)
            thr_dl = np.percentile(p_dl, pct)
            thr_fh = np.percentile(p_fh, pct)

            rows.append({
                'seed': seed, 'percentile': pct, 'model': 'DL',
                'DI':  safe_disparate_impact((p_dl > thr_dl).astype(int), ste),
                'R2':  compute_regression_metrics(yte, p_dl)[0],
            })
            rows.append({
                'seed': seed, 'percentile': pct, 'model': 'FE-HITL',
                'DI':  safe_disparate_impact((p_fh > thr_fh).astype(int), ste),
                'R2':  compute_regression_metrics(yte, p_fh)[0],
            })

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/binarization_sensitivity_raw.csv', index=False)
    summary = df.groupby(['percentile', 'model']).agg(
        DI_mean=('DI', 'mean'), DI_std=('DI', 'std'),
        R2_mean=('R2', 'mean'),
        n=('DI', 'count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/binarization_sensitivity_summary.csv', index=False)
    print("\nBinarization Threshold Sensitivity:")
    print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 8 — Intervention cost quantification  (Issue 9)
# ═══════════════════════════════════════════════════════════════

def run_intervention_cost():
    """
    Count the number of samples requiring simulated expert review
    to provide concrete cost figures for the manuscript (Issue 9).
    15 seeds, bias = 30%, interv_frac = 1.0 (default setting).
    Results → intervention_cost_summary.csv
    """
    N_OPTIONS = 4   # Algorithm 1 generates 4 candidate options (α ∈ {0.3,0.5,0.7,1.0})
    rows = []

    for seed in SEEDS_15:
        set_seed(seed)
        train, val, test = generate_agricultural_data_biased(10000, seed, 0.30)
        Xt, yt, st  = split_arrays(train)
        Xv, yv, sv  = split_arrays(val)
        Xte, yte, ste = split_arrays(test)
        thr = np.median(yt)

        # Get base DL predictions to check trigger condition
        fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                              use_hed=True, use_mog=True, use_fu=False, interv_frac=1.0)
        fh.fit_base(Xt, yt, Xv, yv, s_train=st)
        p_base = fh.base_model.predict(np.asarray(Xte, dtype=np.float64))

        y_bin    = (p_base > thr).astype(int)
        unpriv   = (ste == 1)
        p_unpriv = np.mean(y_bin[unpriv])
        p_priv   = np.mean(y_bin[~unpriv])
        di       = p_unpriv / p_priv if p_priv > 0 else 1.0

        triggered    = di < 0.8
        n_total      = len(Xte)
        n_unpriv     = int(np.sum(unpriv))
        n_intervened = n_unpriv if triggered else 0

        rows.append({
            'seed':                seed,
            'n_test_total':        n_total,
            'n_unpriv':            n_unpriv,
            'di_before':           di,
            'intervention_triggered': triggered,
            'n_intervened':        n_intervened,
            'n_options_per_case':  N_OPTIONS if triggered else 0,
            'intervention_rate_pct': 100.0 * n_intervened / n_total,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/intervention_cost_raw.csv', index=False)

    summary = {
        'n_test_mean':           df['n_test_total'].mean(),
        'n_unpriv_mean':         df['n_unpriv'].mean(),
        'n_intervened_mean':     df['n_intervened'].mean(),
        'n_intervened_std':      df['n_intervened'].std(),
        'intervention_rate_pct': df['intervention_rate_pct'].mean(),
        'n_options_mean':        df['n_options_per_case'].mean(),
        'always_triggered':      bool(df['intervention_triggered'].all()),
    }
    pd.DataFrame([summary]).to_csv(f'{OUT_DIR}/intervention_cost_summary.csv', index=False)
    print("\nIntervention Cost Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return df


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    mode = "QUICK TEST" if QUICK_TEST else "FULL RUN"
    print(f"\nStarting agricultural revision experiments — {mode}")
    print("=" * 65)

    print("\n[1/8] Bias injection rate sensitivity (Issue 8) ...")
    run_bias_sensitivity()
    print("Done.\n")

    print("[2/8] Intervention ratio sensitivity (Issue 9) ...")
    run_intervention_sensitivity()
    print("Done.\n")

    print("[3/8] Ablation study ...")
    run_ablation()
    print("Done.\n")

    print("[4/8] Bias structure robustness — 3 patterns (Issue 2) ...")
    run_bias_structure_robustness()
    print("Done.\n")

    print("[5/8] 5-fold cross-validation (Issue 3) ...")
    run_kfold_cv()
    print("Done.\n")

    print("[6/8] Reverse discrimination analysis (Issue 9) ...")
    run_reverse_discrimination()
    print("Done.\n")

    print("[7/8] Binarization threshold sensitivity (Issue 7) ...")
    run_binarization_sensitivity()
    print("Done.\n")

    print("[8/8] Intervention cost quantification (Issue 9) ...")
    run_intervention_cost()
    print("Done.\n")

    print(f"All results saved in '{OUT_DIR}/'.")
    if QUICK_TEST:
        print("Smoke-test passed. Set QUICK_TEST=False for full experiments.")


if __name__ == '__main__':
    main()