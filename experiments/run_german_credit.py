"""
run_german_credit.py — German Credit classification experiment (30 seeds) + stats
Fixes: paired t-test, scaled random state.
"""

import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed

SEEDS = list(range(42, 72))
TEST_SIZE = 0.3
DI_THRESHOLD = 0.8
OUT_DIR = '.'
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# Fairness metrics (unchanged)
# ═══════════════════════════════════════════════════════
def safe_disparate_impact(y_pred, s_unpriv):
    s = np.asarray(s_unpriv).ravel()
    y = np.asarray(y_pred).ravel()
    p_unpriv = np.mean(y[s == 1])
    p_priv   = np.mean(y[s == 0])
    return p_unpriv / p_priv if p_priv > 0 else 1.0

def equal_opportunity_difference(y_true, y_pred, s):
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    s = np.asarray(s).ravel()
    pos = (y_t == 1)
    tpr_unpriv = np.mean(y_p[(pos) & (s == 1)]) if np.sum(pos & (s == 1)) > 0 else 0
    tpr_priv   = np.mean(y_p[(pos) & (s == 0)]) if np.sum(pos & (s == 0)) > 0 else 0
    return tpr_unpriv - tpr_priv

def average_odds_difference(y_true, y_pred, s):
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    s = np.asarray(s).ravel()
    pos = (y_t == 1)
    neg = (y_t == 0)
    tpr_unpriv = np.mean(y_p[(pos) & (s == 1)]) if np.sum(pos & (s == 1)) > 0 else 0
    tpr_priv   = np.mean(y_p[(pos) & (s == 0)]) if np.sum(pos & (s == 0)) > 0 else 0
    fpr_unpriv = np.mean(y_p[(neg) & (s == 1)]) if np.sum(neg & (s == 1)) > 0 else 0
    fpr_priv   = np.mean(y_p[(neg) & (s == 0)]) if np.sum(neg & (s == 0)) > 0 else 0
    return 0.5 * ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv))

# ═══════════════════════════════════════════════════════
# Bidirectional fairness correction (unchanged)
# ═══════════════════════════════════════════════════════
def apply_fairness_correction(probs, s, di_threshold=0.8, use_mog=True):
    s = np.asarray(s).ravel()
    unpriv_mask = (s == 1)
    priv_mask = ~unpriv_mask

    preds = (probs >= 0.5).astype(int)
    p_unpriv = np.mean(preds[unpriv_mask])
    p_priv = np.mean(preds[priv_mask])
    di = p_unpriv / p_priv if p_priv > 0 else 1.0

    corrected = probs.copy()

    if di < di_threshold:
        boost = 1.0 + (di_threshold - di) * 0.7 if use_mog else 1.0 + 0.05
        corrected[unpriv_mask] = np.clip(probs[unpriv_mask] * boost, 0, 1)
    elif di > 1.0 / di_threshold:
        excess = di - 1.0 / di_threshold
        shrink = 1.0 - 0.7 * excess / (1.0 / di_threshold)
        shrink = np.clip(shrink, 0.5, 1.0)
        corrected[unpriv_mask] = probs[unpriv_mask] * shrink

        # additional direct correction (kept as-is from previous version)
        new_preds = (corrected >= 0.5).astype(int)
        new_di = safe_disparate_impact(new_preds, s)
        if new_di > 1.0 / di_threshold:
            unpriv_pos_mask = unpriv_mask & (new_preds == 1)
            n_unpriv_pos = np.sum(unpriv_pos_mask)
            n_to_flip = max(1, int(n_unpriv_pos * 0.3))
            unpriv_pos_indices = np.where(unpriv_pos_mask)[0]
            sorted_idx = unpriv_pos_indices[np.argsort(-probs[unpriv_pos_indices])]
            flip_idx = sorted_idx[:n_to_flip]
            corrected[flip_idx] = 0.4

    return corrected

# ═══════════════════════════════════════════════════════
# Models (unchanged)
# ═══════════════════════════════════════════════════════
class BaselineDL_Classifier:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam', max_iter=300,
            random_state=self.random_seed
        ).fit(X_scaled, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class DebiasedHITL_Classifier:
    def __init__(self, random_seed=42, di_threshold=0.8, intervention_ratio=0.1):
        self.random_seed = random_seed
        self.di_threshold = di_threshold
        self.intervention_ratio = intervention_ratio
        self.base_model = None
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.base_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam', max_iter=300,
            random_state=self.random_seed
        ).fit(X_scaled, y_train)

    def predict(self, X, s):
        X_scaled = self.scaler.transform(X)
        probs = self.base_model.predict_proba(X_scaled)[:, 1]
        corrected = apply_fairness_correction(probs, s, self.di_threshold, use_mog=False)
        return (corrected >= 0.5).astype(int)

class FEHITL_Classifier:
    def __init__(self, random_seed=42, di_threshold=0.8,
                 use_hed=True, use_mog=True, use_fu=True, interv_frac=1.0):
        self.random_seed = random_seed
        self.di_threshold = di_threshold
        self.use_hed = use_hed
        self.use_mog = use_mog
        self.use_fu = use_fu
        self.interv_frac = interv_frac
        self.base_model = None
        self.scaler = StandardScaler()
        self._X_train_scaled = None
        self._s_train = None

    def fit_base(self, X_train, y_train):
        self._X_train_scaled = self.scaler.fit_transform(X_train)
        self.base_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam', max_iter=300,
            random_state=self.random_seed
        ).fit(self._X_train_scaled, y_train)

    def fit_full(self, X_train, y_train, s_train):
        self._s_train = s_train
        self.fit_base(X_train, y_train)

    def predict_with_intervention(self, X, s):
        X_scaled = self.scaler.transform(X)
        probs = self.base_model.predict_proba(X_scaled)[:, 1]
        s = np.asarray(s).ravel()

        if not self.use_hed:
            return (probs >= 0.5).astype(int)

        corrected = apply_fairness_correction(probs, s, self.di_threshold, self.use_mog)
        preds = (corrected >= 0.5).astype(int)

        if self.use_fu and self.base_model is not None and self._X_train_scaled is not None and self._s_train is not None:
            train_probs = self.base_model.predict_proba(self._X_train_scaled)[:, 1]
            train_corrected = apply_fairness_correction(train_probs, self._s_train, self.di_threshold, self.use_mog)
            self.base_model.partial_fit(self._X_train_scaled, (train_corrected >= 0.5).astype(int),
                                        classes=np.array([0, 1]))
        return preds

# ═══════════════════════════════════════════════════════
# Data loading (unchanged)
# ═══════════════════════════════════════════════════════
def load_german_credit():
    bunch = fetch_openml(data_id=31, as_frame=True, parser='auto')
    X = bunch.data
    y = bunch.target.map({'good': 0, 'bad': 1}).values

    if 'Sex' in X.columns:
        s = (X['Sex'] == 'female').astype(int).values
    else:
        sex_cols = [c for c in X.columns if 'sex' in c.lower() or 'personal_status' in c.lower()]
        s = X[sex_cols[0]].astype(str).str.contains('female', case=False).astype(int).values if sex_cols else np.zeros(len(y), dtype=int)

    X = pd.get_dummies(X, drop_first=True)
    return X.values, y, s

# ═══════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════
def main():
    print("Loading German Credit data...")
    X, y, s = load_german_credit()
    print(f"Data shape: {X.shape}, bad rate: {np.mean(y):.3f}, unpriv ratio: {np.mean(s):.3f}")

    rows = []
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        print(f"Seed {seed}: train={len(X_train)} test={len(X_test)}")

        for model_name, model in [
            ('LR', LogisticRegression(max_iter=1000, random_state=seed)),
            ('DL', BaselineDL_Classifier(random_seed=seed)),
            ('Debiased-HITL', DebiasedHITL_Classifier(random_seed=seed, di_threshold=DI_THRESHOLD)),
            ('FE-HITL', FEHITL_Classifier(random_seed=seed, di_threshold=DI_THRESHOLD,
                                          use_hed=True, use_mog=True, use_fu=True))
        ]:
            if model_name == 'LR':
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            elif model_name == 'DL':
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            elif model_name == 'Debiased-HITL':
                model.fit(X_train, y_train)
                pred = model.predict(X_test, s_test)
            else:
                model.fit_full(X_train, y_train, s_train)
                pred = model.predict_with_intervention(X_test, s_test)

            rows.append({
                'seed': seed, 'model': model_name,
                'Accuracy': accuracy_score(y_test, pred),
                'DI': safe_disparate_impact(pred, s_test),
                'EOD': abs(equal_opportunity_difference(y_test, pred, s_test)),
                'AOD': abs(average_odds_difference(y_test, pred, s_test))
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'german_credit_raw.csv'), index=False)
    print("\nRaw results saved.")

    summary = df.groupby('model').agg(
        Acc_mean=('Accuracy', 'mean'), Acc_std=('Accuracy', 'std'),
        DI_mean=('DI', 'mean'), DI_std=('DI', 'std'),
        EOD_mean=('EOD', 'mean'), EOD_std=('EOD', 'std'),
        AOD_mean=('AOD', 'mean'), AOD_std=('AOD', 'std'),
        n=('Accuracy', 'count')
    ).reset_index()

    for col in ['Acc', 'DI', 'EOD', 'AOD']:
        summary[f'{col}_CI95'] = 1.96 * summary[f'{col}_std'] / np.sqrt(summary['n'])

    summary.to_csv(os.path.join(OUT_DIR, 'german_credit_summary.csv'), index=False)
    print("Summary saved.\n")

    print("=" * 80)
    print("German Credit (30 seeds) – Key metrics")
    print("=" * 80)
    for _, row in summary.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Accuracy: {row['Acc_mean']:.3f} ± {row['Acc_std']:.3f} [95% CI: {row['Acc_mean']-row['Acc_CI95']:.3f}, {row['Acc_mean']+row['Acc_CI95']:.3f}]")
        print(f"  DI:        {row['DI_mean']:.3f} ± {row['DI_std']:.3f} [95% CI: {row['DI_mean']-row['DI_CI95']:.3f}, {row['DI_mean']+row['DI_CI95']:.3f}]")
        print(f"  |EOD|:     {row['EOD_mean']:.3f} ± {row['EOD_std']:.3f}")
        print(f"  |AOD|:     {row['AOD_mean']:.3f} ± {row['AOD_std']:.3f}")

    # ── Paired statistical test ──
    from scipy import stats
    fe_di = df[df['model'] == 'FE-HITL'].set_index('seed')['DI']
    dl_di = df[df['model'] == 'DL'].set_index('seed')['DI']
    common_seeds = fe_di.index.intersection(dl_di.index)
    if len(common_seeds) > 1:
        t_stat, p_val = stats.ttest_rel(fe_di.loc[common_seeds], dl_di.loc[common_seeds])
        d = (fe_di.mean() - dl_di.mean()) / np.sqrt((fe_di.var() + dl_di.var()) / 2)
        print(f"\nPaired t-test (FE-HITL vs DL, DI): t = {t_stat:.3f}, p = {p_val:.6f}")
        print(f"Cohen's d: {d:.3f}")
    else:
        print("Not enough paired seeds for test.")

if __name__ == '__main__':
    main()