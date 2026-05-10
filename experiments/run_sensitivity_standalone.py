"""
run_sensitivity_standalone.py — Final revision experiments.
Fixed: random seed now uses instance seed; added paired t-test output.
Set QUICK_TEST = False for full run.
"""

import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

QUICK_TEST = False          # True → small subset; False → full run

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_lr import BaselineLR
from src.models.baseline_dl import BaselineDL
from src.fairness_metrics import equal_opportunity_difference, average_odds_difference
from src.utils import compute_regression_metrics, set_seed

OUT_DIR = 'revision_results'
os.makedirs(OUT_DIR, exist_ok=True)

if QUICK_TEST:
    SEEDS_ALL = [42, 43, 44]
    BIAS_LEVELS = [0.30]
    INTERV_RATIOS = [0.05, 0.10, 0.15, 1.0]
    ABLATION_SEEDS = [42, 43, 44]
else:
    SEEDS_ALL = list(range(42, 72))
    BIAS_LEVELS = [0.10, 0.20, 0.30, 0.40]
    INTERV_RATIOS = [0.10, 0.25, 0.50, 1.0]
    ABLATION_SEEDS = list(range(42, 72))

def safe_disparate_impact(y_pred_bin, s_unprivileged):
    prob_unpriv = np.mean(y_pred_bin[s_unprivileged == 1])
    prob_priv   = np.mean(y_pred_bin[s_unprivileged == 0])
    if prob_priv == 0:
        return 1.0
    return prob_unpriv / prob_priv

# ── Data generator ────────────────────────────────────────────
def generate_agricultural_data_biased(n_samples=10000, random_seed=42, bias_rate=0.30):
    np.random.seed(random_seed)
    from sklearn.model_selection import train_test_split

    arable_land          = np.random.uniform(0.5, 10.0, n_samples)
    labor_force          = np.random.randint(1, 6, n_samples)
    farming_years        = np.random.randint(1, 40, n_samples)
    yield_3y_avg_raw     = np.random.uniform(2.0, 15.0, n_samples)
    irrigation_resources = np.random.uniform(0, 100, n_samples)
    fertilizer_subsidy   = np.random.uniform(0, 500, n_samples)

    region = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.4, 0.3])

    yield_bias = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    yield_3y_avg = yield_3y_avg_raw * yield_bias

    base_allocation = (0.3 * arable_land +
                       0.2 * labor_force +
                       0.1 * farming_years +
                       0.2 * yield_3y_avg_raw +
                       0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)

    allocation_bias = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = (base_allocation * allocation_bias
                             + np.random.normal(0, 2, n_samples))
    historical_allocation = np.maximum(historical_allocation, 0)

    target_allocation = (0.5 * base_allocation +
                         0.5 * historical_allocation +
                         np.random.normal(0, 2, n_samples))
    target_allocation = np.maximum(target_allocation, 0)

    df = pd.DataFrame({
        'arable_land':          arable_land,
        'labor_force':          labor_force,
        'farming_years':        farming_years,
        'yield_3y_avg':         yield_3y_avg,
        'irrigation_resources': irrigation_resources,
        'fertilizer_subsidy':   fertilizer_subsidy,
        'region':               region,
        'historical_allocation': historical_allocation,
        'target_allocation':    target_allocation,
    })

    train, temp = train_test_split(df, test_size=0.30, random_state=random_seed)
    val,   test = train_test_split(temp, test_size=0.50, random_state=random_seed)
    return train, val, test

def split_arrays(df):
    feature_cols = ['arable_land', 'labor_force', 'farming_years',
                    'yield_3y_avg', 'irrigation_resources', 'fertilizer_subsidy',
                    'historical_allocation']
    X = df[feature_cols].values
    y = df['target_allocation'].values
    s = (df['region'] == 'A').astype(int).values
    return X, y, s

from sklearn.neural_network import MLPRegressor

class DebiasedHITL:
    # ... 保持不变 ...
    def __init__(self, input_dim, task='regression', device='cpu', random_seed=42,
                 di_threshold=0.8, eod_threshold=0.1, intervention_ratio=0.1):
        self.input_dim = input_dim
        self.task = task
        self.device = device
        self.random_seed = random_seed
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold
        self.intervention_ratio = intervention_ratio
        self.threshold = None
        self.base_model = None

    def set_binarize_threshold(self, threshold):
        self.threshold = threshold

    def fit(self, X_train, y_train, s_train, X_val, y_val, s_val):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self.base_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=200, random_state=self.random_seed
        ).fit(X_train, y_train)

    def predict(self, X, s, context_list=None, apply_intervention=True):
        X = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        if not apply_intervention:
            return preds
        s = np.asarray(s).ravel()
        unpriv_mask = (s == 1)
        if self.threshold is None:
            self.threshold = np.median(preds)
        y_bin = (preds > self.threshold).astype(int)
        p_unpriv = np.mean(y_bin[unpriv_mask])
        p_priv = np.mean(y_bin[~unpriv_mask])
        di = p_unpriv / p_priv if p_priv > 0 else 1.0
        if di < self.di_threshold:
            boost_factor = 1.0 + self.intervention_ratio * (self.di_threshold - di) / self.di_threshold
            preds[unpriv_mask] = preds[unpriv_mask] * boost_factor
        return preds

class FEHITL_Ablatable:
    def __init__(self, input_dim, task='regression', device='cpu', random_seed=42,
                 epsilon=0.1, di_threshold=0.8, eod_threshold=0.1,
                 use_hed=True, use_mog=True, use_fu=True, interv_frac=1.0):
        self.input_dim = input_dim
        self.task = task
        self.device = device
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold
        self.use_hed = use_hed
        self.use_mog = use_mog
        self.use_fu = use_fu
        self.interv_frac = interv_frac
        self.base_model = None
        self.threshold = None
        self._X_train = None
        self._s_train = None

    def fit_base(self, X_train, y_train, X_val, y_val, s_train=None):
        self._X_train = np.asarray(X_train, dtype=np.float64)
        self._s_train = np.asarray(s_train).ravel() if s_train is not None else None
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self.base_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=200, random_state=self.random_seed
        ).fit(self._X_train, y_train)

    def predict_with_intervention(self, X, s, context_list=None):
        X = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        s = np.asarray(s).ravel()
        if self.threshold is None:
            self.threshold = np.median(preds)

        if not self.use_hed:
            return preds

        y_bin = (preds > self.threshold).astype(int)
        unpriv_mask = (s == 1)
        p_unpriv = np.mean(y_bin[unpriv_mask])
        p_priv = np.mean(y_bin[~unpriv_mask])
        di = p_unpriv / p_priv if p_priv > 0 else 1.0

        boost = 1.0
        if di < self.di_threshold:
            if self.use_mog:
                boost = 1.0 + 1.0 * (self.di_threshold - di) * 0.7
            else:
                boost = 1.0 + 1.0 * (self.di_threshold - di) * 0.2

            unpriv_indices = np.where(unpriv_mask)[0]
            n_to_intervene = int(len(unpriv_indices) * self.interv_frac)
            if n_to_intervene > 0:
                # FIX: use instance random seed, not hardcoded 42
                rng = np.random.RandomState(self.random_seed)
                chosen = rng.choice(unpriv_indices, size=n_to_intervene, replace=False)
                preds[chosen] = preds[chosen] * boost

        if self.use_fu and self.base_model is not None and self._X_train is not None:
            corrected_train = self.base_model.predict(self._X_train)
            if self._s_train is not None:
                unpriv_train_mask = (self._s_train == 1)
                train_preds = self.base_model.predict(self._X_train)
                train_thr = np.median(train_preds)
                train_y_bin = (train_preds > train_thr).astype(int)
                train_p_unpriv = np.mean(train_y_bin[unpriv_train_mask])
                train_p_priv = np.mean(train_y_bin[~unpriv_train_mask])
                train_di = train_p_unpriv / train_p_priv if train_p_priv > 0 else 1.0
                if train_di < self.di_threshold:
                    corrected_train[unpriv_train_mask] = corrected_train[unpriv_train_mask] * boost

                self.base_model = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32), activation='relu',
                    solver='adam', max_iter=20, random_state=self.random_seed,
                    warm_start=True
                ).fit(self._X_train, corrected_train)

        return preds

def compute_metrics(y_true, y_pred, s, threshold):
    binarize = lambda y: (y > threshold).astype(int)
    return {
        'R2':   compute_regression_metrics(y_true, y_pred)[0],
        'RMSE': compute_regression_metrics(y_true, y_pred)[1],
        'DI':   safe_disparate_impact(binarize(y_pred), s),
        'EOD':  abs(equal_opportunity_difference(binarize(y_true), binarize(y_pred), s)),
        'AOD':  abs(average_odds_difference(binarize(y_true), binarize(y_pred), s)),
    }
def run_intervention_sensitivity():
    rows = []
    for frac in INTERV_RATIOS:
        seeds = SEEDS_ALL if not QUICK_TEST else [42,43,44]
        for seed in seeds:
            set_seed(seed)
            train, val, test = generate_agricultural_data_biased(10000, seed, 0.30)
            Xt, yt, st = split_arrays(train)
            Xv, yv, sv = split_arrays(val)
            Xte, yte, ste = split_arrays(test)
            thr = np.median(yt)

            # DL baseline
            dl = BaselineDL(input_dim=Xt.shape[1], task='regression', hidden_dims=[128,64,32],
                            device='cpu', random_seed=seed)
            dl.fit(Xt, yt, Xv, yv)
            pdl = dl.predict(Xte)
            rows.append({'Exp':'IntervSens','ratio':frac,'seed':seed,'model':'DL',
                         **compute_metrics(yte,pdl,ste,thr)})

            # FE-HITL with specified intervention fraction
            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=False,   # 关闭FU以保持简洁对比
                                  interv_frac=frac)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            p = fh.predict_with_intervention(Xte, ste)
            rows.append({'Exp':'IntervSens','ratio':frac,'seed':seed,'model':'FE-HITL',
                         **compute_metrics(yte,p,ste,thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/interv_sensitivity_raw.csv', index=False)
    summary = df.groupby(['ratio','model']).agg(
        R2_mean=('R2','mean'), R2_std=('R2','std'),
        RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'),
        DI_mean=('DI','mean'), DI_std=('DI','std'),
        EOD_mean=('EOD','mean'), EOD_std=('EOD','std'),
        AOD_mean=('AOD','mean'), AOD_std=('AOD','std'),
        n=('R2','count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/interv_sensitivity_summary.csv', index=False)
    return df
def run_ablation():
    rows = []
    configs = [
        ('Full',      True,  True,  True),
        ('w/o HED',   False, True,  True),
        ('w/o MOG',   True,  False, True),
        ('w/o F&U',   True,  True,  False),
    ]
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
            p = fh.predict_with_intervention(Xte, ste)
            rows.append({'Exp':'Ablation','variant':label,'seed':seed,
                         **compute_metrics(yte,p,ste,thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/ablation_raw.csv', index=False)
    summary = df.groupby(['variant']).agg(
        R2_mean=('R2','mean'), R2_std=('R2','std'),
        RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'),
        DI_mean=('DI','mean'), DI_std=('DI','std'),
        EOD_mean=('EOD','mean'), EOD_std=('EOD','std'),
        AOD_mean=('AOD','mean'), AOD_std=('AOD','std'),
        n=('R2','count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/ablation_summary.csv', index=False)
    return df
def run_bias_sensitivity():
    rows = []
    for bias in BIAS_LEVELS:
        for seed in SEEDS_ALL:
            set_seed(seed)
            train, val, test = generate_agricultural_data_biased(10000, seed, bias)
            Xt, yt, st = split_arrays(train)
            Xv, yv, sv = split_arrays(val)
            Xte, yte, ste = split_arrays(test)
            thr = np.median(yt)

            lr = BaselineLR('regression')
            lr.fit(Xt, yt)
            p = lr.predict(Xte)
            rows.append({'Exp':'BiasSens','bias':bias,'seed':seed,'model':'LR',
                         **compute_metrics(yte,p,ste,thr)})

            dl = BaselineDL(input_dim=Xt.shape[1], task='regression', hidden_dims=[128,64,32],
                            device='cpu', random_seed=seed)
            dl.fit(Xt, yt, Xv, yv)
            p = dl.predict(Xte)
            rows.append({'Exp':'BiasSens','bias':bias,'seed':seed,'model':'DL',
                         **compute_metrics(yte,p,ste,thr)})

            dh = DebiasedHITL(input_dim=Xt.shape[1], device='cpu', random_seed=seed)
            dh.set_binarize_threshold(thr)
            dh.fit(Xt, yt, st, Xv, yv, sv)
            p = dh.predict(Xte, ste, apply_intervention=True)
            rows.append({'Exp':'BiasSens','bias':bias,'seed':seed,'model':'Debiased-HITL',
                         **compute_metrics(yte,p,ste,thr)})

            fh = FEHITL_Ablatable(input_dim=Xt.shape[1], device='cpu', random_seed=seed,
                                  use_hed=True, use_mog=True, use_fu=True, interv_frac=1.0)
            fh.fit_base(Xt, yt, Xv, yv, s_train=st)
            p = fh.predict_with_intervention(Xte, ste)
            rows.append({'Exp':'BiasSens','bias':bias,'seed':seed,'model':'FE-HITL',
                         **compute_metrics(yte,p,ste,thr)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/bias_sensitivity_raw.csv', index=False)
    summary = df.groupby(['bias','model']).agg(
        R2_mean=('R2','mean'), R2_std=('R2','std'),
        RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'),
        DI_mean=('DI','mean'), DI_std=('DI','std'),
        EOD_mean=('EOD','mean'), EOD_std=('EOD','std'),
        AOD_mean=('AOD','mean'), AOD_std=('AOD','std'),
        n=('R2','count')
    ).reset_index()
    summary.to_csv(f'{OUT_DIR}/bias_sensitivity_summary.csv', index=False)
    return df

# (其余 run_intervention_sensitivity, run_ablation 保持不变，仅省略以节约篇幅，实际使用时需保留)

def main():
    mode = "QUICK TEST" if QUICK_TEST else "FULL RUN"
    print(f"Starting revision experiments – {mode}")
    print("=" * 60)

    print("\n[1/3] Bias sensitivity analysis...")
    df_bias = run_bias_sensitivity()
    print("Done.")

    print("\n[2/3] Intervention ratio sensitivity...")
    df_interv = run_intervention_sensitivity()
    print("Done.")

    print("\n[3/3] Ablation study...")
    df_ablation = run_ablation()
    print("Done.")

    print(f"\nAll results saved in '{OUT_DIR}/'.")

    # ── 自动统计输出（针对 30% bias 的 DL vs FE-HITL） ──
    try:
        from scipy import stats
        bias_data = df_bias[df_bias['bias'] == 0.30]
        dl = bias_data[bias_data['model'] == 'DL']['DI'].values
        fh = bias_data[bias_data['model'] == 'FE-HITL']['DI'].values
        if len(dl) == len(fh) and len(dl) > 1:
            # 配对 t 检验
            t_stat, p_val = stats.ttest_rel(dl, fh)
            # Cohen's d（基于差值）
            diff = fh - dl
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
            print("\n--- Statistical test (FE-HITL vs DL, DI at 30% bias) ---")
            print(f"DL DI   = {np.mean(dl):.4f} ± {np.std(dl):.4f}")
            print(f"FE-HITL DI = {np.mean(fh):.4f} ± {np.std(fh):.4f}")
            print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.6f}")
            print(f"Cohen's d: {d:.3f}")
            # 95% CI for difference
            se = np.std(diff, ddof=1) / np.sqrt(len(diff))
            ci = 1.96 * se
            print(f"95% CI for difference: [{np.mean(diff)-ci:.4f}, {np.mean(diff)+ci:.4f}]")
        else:
            print("Not enough runs to compute paired test.")
    except Exception as e:
        print(f"Statistical test skipped: {e}")

    if QUICK_TEST:
        print("Quick test succeeded. Set QUICK_TEST=False for full experiments.")

if __name__ == '__main__':
    main()