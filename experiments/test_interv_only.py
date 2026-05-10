"""
test_interv_only.py — 独立测试干预比例敏感性
修复：interv ratio 控制受保护样本中被补偿的比例
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_dl import BaselineDL
from src.utils import set_seed
from sklearn.neural_network import MLPRegressor

# ── 数据生成（不变） ────────────────────────────────────────
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
    base_allocation = (0.3 * arable_land + 0.2 * labor_force + 0.1 * farming_years +
                       0.2 * yield_3y_avg_raw + 0.1 * irrigation_resources +
                       0.1 * fertilizer_subsidy / 100)
    allocation_bias = np.where(region == 'A', 1.0 - bias_rate, 1.0)
    historical_allocation = (base_allocation * allocation_bias + np.random.normal(0, 2, n_samples))
    historical_allocation = np.maximum(historical_allocation, 0)
    target_allocation = (0.5 * base_allocation + 0.5 * historical_allocation +
                         np.random.normal(0, 2, n_samples))
    target_allocation = np.maximum(target_allocation, 0)
    df = pd.DataFrame({'arable_land': arable_land, 'labor_force': labor_force,
                       'farming_years': farming_years, 'yield_3y_avg': yield_3y_avg,
                       'irrigation_resources': irrigation_resources,
                       'fertilizer_subsidy': fertilizer_subsidy, 'region': region,
                       'historical_allocation': historical_allocation,
                       'target_allocation': target_allocation})
    train, temp = train_test_split(df, test_size=0.30, random_state=random_seed)
    val, test = train_test_split(temp, test_size=0.50, random_state=random_seed)
    return train, val, test

def split_arrays(df):
    feature_cols = ['arable_land','labor_force','farming_years','yield_3y_avg',
                    'irrigation_resources','fertilizer_subsidy','historical_allocation']
    X = df[feature_cols].values
    y = df['target_allocation'].values
    s = (df['region'] == 'A').astype(int).values
    return X, y, s

def safe_di(y_pred_bin, s):
    p1 = np.mean(y_pred_bin[s==1])
    p0 = np.mean(y_pred_bin[s==0])
    return p1/p0 if p0>0 else 1.0

# ── FE‑HITL：补偿强度固定为 Full 级别，但仅对随机 sampRatio 的样本补偿 ──
class FEHITL_SampleBased:
    def __init__(self, input_dim, random_seed=42, interv_ratio=0.10):
        self.interv_ratio = interv_ratio   # 比例
        self.random_seed = random_seed
        self.base_model = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self.base_model = MLPRegressor(
            hidden_layer_sizes=(128,64,32), activation='relu',
            solver='adam', max_iter=200, random_state=self.random_seed
        ).fit(X_train, y_train)

    def predict(self, X, s):
        X = np.asarray(X, dtype=np.float64)
        preds = self.base_model.predict(X)
        s = np.asarray(s).ravel()
        thr = np.median(preds)
        y_bin = (preds > thr).astype(int)
        unpriv_mask = (s == 1)
        p1 = np.mean(y_bin[unpriv_mask])
        p0 = np.mean(y_bin[~unpriv_mask])
        di = p1/p0 if p0>0 else 1.0

        if di < 0.8:
            # 固定使用 scale=1.0 的 Full 强度 boost
            boost = 1.0 + 1.0 * (0.8 - di) * 0.7
            unpriv_indices = np.where(unpriv_mask)[0]
            n_interv = max(1, int(len(unpriv_indices) * self.interv_ratio))
            rng = np.random.RandomState(self.random_seed)
            chosen = rng.choice(unpriv_indices, size=n_interv, replace=False)
            preds[chosen] = preds[chosen] * boost
        return preds

# ── 主测试 ──────────────────────────────────────────────────
def main():
    ratios = [0.05, 0.10, 0.15, 1.0]      # 1.0 = Full intervention (对照)
    seeds = [42, 43, 44]
    print("Intervention Ratio Sensitivity (Sample‑based)")
    print("=" * 60)
    for ratio in ratios:
        di_list, dl_di_list = [], []
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
            dl_di = safe_di((pdl > thr).astype(int), ste)
            dl_di_list.append(dl_di)

            # FE‑HITL
            fh = FEHITL_SampleBased(input_dim=Xt.shape[1], random_seed=seed, interv_ratio=ratio)
            fh.fit(Xt, yt)
            p = fh.predict(Xte, ste)
            fh_di = safe_di((p > thr).astype(int), ste)
            di_list.append(fh_di)

        print(f"ratio={ratio:.2f} | DL DI={np.mean(dl_di_list):.3f} | "
              f"FE‑HITL DI mean={np.mean(di_list):.3f}")
    print("\n✅ 预期：DI 随 ratio 增加而递增，ratio=1.0 时接近 Full FE‑HITL (≈0.85)")

if __name__ == '__main__':
    main()