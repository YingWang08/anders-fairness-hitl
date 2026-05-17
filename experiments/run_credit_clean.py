"""
run_credit_revised.py
=====================
German Credit dataset experiments — Revised for PLOS ONE R2 submission.

Key changes from run_credit_clean.py
--------------------------------------
1. SHARED BASE MODEL: DebiasedHITL and FE-HITL receive predictions from the
   *same* trained MLP instance. All performance differences arise solely from
   post-processing correction strategies, not from different random initialisations.

2. DEBIASED-HITL (reproduced from Zhang et al. 2021):
   Proportional boost/shrink with a fixed intervention ratio = 0.10.
   Operates separately on DI < 0.8 (boost unprivileged) and DI > 1.25 (shrink
   unprivileged), mimicking the original paper's post-processing design.

3. FE-HITL (proposed framework — Algorithm 1):
   Multi-option generation: evaluates N_OPTIONS candidate scaling factors and
   selects the one whose corrected DI is closest to 1.0 (perfect parity),
   subject to a no-reverse-discrimination guard (corrected DI stays within
   [DI_GUARD_LOW, DI_GUARD_HIGH]). This directly operationalises the
   value-arbitration mechanism described in the manuscript.

4. STATISTICAL TESTS: paired t-tests + Benjamini-Hochberg correction +
   Cohen's dz are computed and saved for all key metric pairs.

5. OUTPUT FILES (saved to OUT_DIR = 'credit_revision_results/'):
   credit_main_raw.csv          — per-seed results, main experiment
   credit_main_summary.csv      — mean ± SD, 95% CI, by model
   credit_paired_tests.csv      — full paired-test table
   credit_cv_raw.csv            — per-fold-seed results, 5-fold CV
   credit_cv_summary.csv        — mean ± SD by model, 5-fold CV
   credit_cv_paired_tests.csv   — paired tests for CV results

Implementation notes
--------------------
- All neural-network models use scikit-learn MLPClassifier (hidden sizes
  [128, 64, 32], ReLU, Adam, early stopping). The manuscript term "deep
  learning (DL)" refers to this multi-layer architecture.
- Gender/protected-group column is detected automatically.
- Set QUICK_TEST = True for a 3-seed smoke-test before the full run.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────
QUICK_TEST   = False           # True = 3 seeds, for smoke-testing only
SEEDS_FULL   = list(range(42, 92))   # 30 seeds
SEEDS_QUICK  = [42, 43, 44]

# Fairness thresholds
DI_LOWER     = 0.80            # 4/5 rule lower bound
DI_UPPER     = 1.0 / DI_LOWER  # 1.25 — upper bound (symmetric)

# Debiased-HITL fixed intervention ratio (Zhang et al. 2021)
DEBIAS_RATIO = 0.10

# FE-HITL multi-option generation — candidate scaling factors (Algorithm 1)
# Six candidates for each direction, mapping to α ∈ {0.3, 0.5, 0.7, 0.9, 1.1, 1.3}
# scaled by (DI_UPPER - di) or (di - DI_LOWER) within the boost formula.
# We use explicit factor lists for full reproducibility.
MOG_FACTORS_BOOST  = [1.03, 1.06, 1.10, 1.15, 1.20, 1.25]  # for DI < DI_LOWER
MOG_FACTORS_SHRINK = [0.97, 0.94, 0.90, 0.85, 0.80, 0.75]  # for DI > DI_UPPER

# Value-arbitration guard: reject any candidate that causes reverse discrimination
DI_GUARD_LOW  = 0.75   # corrected DI must not fall below this
DI_GUARD_HIGH = 1.333  # corrected DI must not exceed this (= 1/0.75)

OUT_DIR = "credit_revision_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  FAIRNESS METRICS
# ──────────────────────────────────────────────────────────────

def safe_di(y_pred, s):
    """
    Disparate Impact = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged)
    Convention: s=1 → unprivileged (female); s=0 → privileged (male).
    DI > 1 means unprivileged group is MORE likely to receive positive outcome.
    DI < 1 means unprivileged group is LESS likely to receive positive outcome.
    """
    y = np.asarray(y_pred).ravel()
    s = np.asarray(s).ravel()
    mask_u = (s == 1)
    mask_p = (s == 0)
    if not np.any(mask_u) or not np.any(mask_p):
        return 1.0
    pu = np.mean(y[mask_u])
    pp = np.mean(y[mask_p])
    return pu / pp if pp > 1e-9 else 1.0


def equal_opportunity_difference(y_true, y_pred, s):
    """
    |EOD| = |TPR_unprivileged - TPR_privileged|  (absolute value)
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    s   = np.asarray(s).ravel()
    pos = (y_t == 1)
    mask_u = pos & (s == 1)
    mask_p = pos & (s == 0)
    tpr_u = np.mean(y_p[mask_u]) if np.any(mask_u) else 0.0
    tpr_p = np.mean(y_p[mask_p]) if np.any(mask_p) else 0.0
    return abs(tpr_u - tpr_p)


def average_odds_difference(y_true, y_pred, s):
    """
    |AOD| = |0.5 * [(FPR_u - FPR_p) + (TPR_u - TPR_p)]|
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    s   = np.asarray(s).ravel()
    pos = (y_t == 1)
    neg = (y_t == 0)
    tpr_u = np.mean(y_p[pos & (s == 1)]) if np.any(pos & (s == 1)) else 0.0
    tpr_p = np.mean(y_p[pos & (s == 0)]) if np.any(pos & (s == 0)) else 0.0
    fpr_u = np.mean(y_p[neg & (s == 1)]) if np.any(neg & (s == 1)) else 0.0
    fpr_p = np.mean(y_p[neg & (s == 0)]) if np.any(neg & (s == 0)) else 0.0
    return abs(0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p)))


def compute_all_metrics(y_true, y_pred, s):
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "DI":  safe_di(y_pred, s),
        "EOD": equal_opportunity_difference(y_true, y_pred, s),
        "AOD": average_odds_difference(y_true, y_pred, s),
    }


# ──────────────────────────────────────────────────────────────
#  POST-PROCESSING CORRECTION FUNCTIONS
#  Both receive probability scores from the SAME base model.
# ──────────────────────────────────────────────────────────────

def debiased_hitl_correction(probs, s):
    """
    Debiased-HITL post-processing (reproduced from Zhang et al. 2021).

    Strategy: proportional boost or shrink of the unprivileged group's
    predicted probabilities, using a fixed intervention ratio (DEBIAS_RATIO).

    - If DI < DI_LOWER (unprivileged under-predicted):
        boost = 1 + DEBIAS_RATIO * (DI_LOWER - di) / DI_LOWER
        probs[unprivileged] *= boost
    - If DI > DI_UPPER (unprivileged over-predicted):
        shrink = 1 - DEBIAS_RATIO * (di - DI_UPPER) / DI_UPPER
        probs[unprivileged] *= shrink  (floor at 0.5 to prevent extreme values)
    - Otherwise: no intervention.

    The fixed-ratio design means the correction strength depends purely on
    the magnitude of the disparity, with no optimisation over candidates.
    """
    probs = np.asarray(probs, dtype=float).copy()
    s     = np.asarray(s).ravel()
    mask_u = (s == 1)
    mask_p = (s == 0)

    if not np.any(mask_u) or not np.any(mask_p):
        return probs

    preds = (probs >= 0.5).astype(int)
    pu = np.mean(preds[mask_u])
    pp = np.mean(preds[mask_p])
    if pp < 1e-9:
        return probs
    di = pu / pp

    if di < DI_LOWER:
        # Unprivileged group under-predicted → boost
        boost = 1.0 + DEBIAS_RATIO * (DI_LOWER - di) / DI_LOWER
        probs[mask_u] = np.clip(probs[mask_u] * boost, 0.0, 1.0)

    elif di > DI_UPPER:
        # Unprivileged group over-predicted → shrink
        shrink = 1.0 - DEBIAS_RATIO * (di - DI_UPPER) / DI_UPPER
        shrink = max(shrink, 0.50)   # floor to avoid extreme suppression
        probs[mask_u] = np.clip(probs[mask_u] * shrink, 0.0, 1.0)

    return probs


def fehitl_correction(probs, s):
    """
    FE-HITL post-processing — Algorithm 1 (Multi-Option Generation + Value Arbitration).

    Strategy:
    1. Compute current DI from base probabilities.
    2. If DI is already within [DI_LOWER, DI_UPPER]: no intervention needed.
    3. Otherwise, generate MOG_FACTORS candidates (boost or shrink direction
       chosen by DI direction).
    4. For each candidate factor, compute the corrected DI.
    5. Value arbitration (simulated expert): select the factor whose corrected
       DI is CLOSEST to 1.0 (perfect parity), subject to the guard condition
       [DI_GUARD_LOW, DI_GUARD_HIGH] (no reverse discrimination).
    6. Apply the winning factor to the unprivileged group's probabilities.

    Key differences from Debiased-HITL:
    - Objective: minimise |DI_corrected - 1.0|, not push DI past a threshold.
    - Candidate pool: 6 factors (more granular search).
    - Guard condition: explicitly rejects candidates causing reverse discrimination.
    - No fixed intervention ratio: the best factor is selected adaptively.
    """
    probs = np.asarray(probs, dtype=float).copy()
    s     = np.asarray(s).ravel()
    mask_u = (s == 1)
    mask_p = (s == 0)

    if not np.any(mask_u) or not np.any(mask_p):
        return probs

    preds = (probs >= 0.5).astype(int)
    pu = np.mean(preds[mask_u])
    pp = np.mean(preds[mask_p])
    if pp < 1e-9:
        return probs
    di = pu / pp

    # No intervention if DI already acceptable
    if DI_LOWER <= di <= DI_UPPER:
        return probs

    # Choose candidate direction
    candidates = MOG_FACTORS_BOOST if di < DI_LOWER else MOG_FACTORS_SHRINK

    best_factor   = 1.0          # fallback: no change
    best_distance = abs(di - 1.0)  # current distance to perfect parity

    for factor in candidates:
        # Compute corrected probabilities and the resulting DI
        cand_probs  = np.clip(probs[mask_u] * factor, 0.0, 1.0)
        cand_preds  = (cand_probs >= 0.5).astype(int)
        cand_pu     = np.mean(cand_preds) if len(cand_preds) > 0 else 0.0
        cand_di     = cand_pu / pp if pp > 1e-9 else 1.0

        # Value-arbitration guard: reject reverse-discrimination candidates
        if not (DI_GUARD_LOW <= cand_di <= DI_GUARD_HIGH):
            continue

        distance = abs(cand_di - 1.0)
        if distance < best_distance:
            best_distance = distance
            best_factor   = factor

    probs[mask_u] = np.clip(probs[mask_u] * best_factor, 0.0, 1.0)
    return probs


# ──────────────────────────────────────────────────────────────
#  BASE MODEL
# ──────────────────────────────────────────────────────────────

class SharedMLP:
    """
    Scikit-learn MLPClassifier with StandardScaler.
    Implements DL / Debiased-HITL / FE-HITL from a single trained instance:
      - DL        → predict() directly
      - Debiased  → debiased_hitl_correction(predict_proba()) → threshold
      - FE-HITL   → fehitl_correction(predict_proba())        → threshold
    This ensures ALL three methods share the identical base model, so
    performance differences arise solely from post-processing.
    """
    def __init__(self, seed):
        self.scaler = StandardScaler()
        self.clf    = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        )

    def fit(self, X, y):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba_pos(self, X):
        """Return P(y=1) for each sample."""
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


# ──────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_german_credit():
    """
    Load UCI German Credit dataset (OpenML id=31).
    Protected attribute: gender (female → s=1, male → s=0).

    OpenML has two label versions for the personal_status column:
      - Raw codes  (older API):  'A91' 'A92' 'A93' 'A94' 'A95'
      - Text labels (newer API): 'male div/sep', 'female div/dep/mar',
                                  'male single', 'male mar/wid', 'female single'
    This function handles both transparently and raises a hard error if
    the protected-group count is 0 (prevents silent DI=1.0 failure).

    Returns: X (ndarray), y (ndarray, 0=good/1=bad), s (ndarray, 0/1)
      y=0 → good credit (positive outcome)
      y=1 → bad credit  (negative outcome)
      s=1 → female (unprivileged); s=0 → male (privileged)
    """
    bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df    = bunch.data.copy()

    # ── Target variable ──────────────────────────────────────────────
    # 'good' → 0 (approved), 'bad' → 1 (rejected)
    raw_target = bunch.target.astype(str).str.strip().str.lower()
    if raw_target.isin(["good", "bad"]).all():
        y = (raw_target == "bad").astype(int).values
    elif raw_target.isin(["1", "2"]).all():
        # Some OpenML versions encode good=1, bad=2
        y = (raw_target == "2").astype(int).values
    else:
        raise ValueError(
            f"Unrecognised target values: {raw_target.unique()}\n"
            "Expected 'good'/'bad' or '1'/'2'."
        )

    # ── Locate the personal-status / sex column ──────────────────────
    # Priority: exact 'personal_status' > contains 'personal' > contains 'sex'
    sex_col = None
    for col in df.columns:
        if col.lower() == "personal_status":
            sex_col = col; break
    if sex_col is None:
        for col in df.columns:
            if "personal" in col.lower():
                sex_col = col; break
    if sex_col is None:
        for col in df.columns:
            if "sex" in col.lower():
                sex_col = col; break
    if sex_col is None:
        raise ValueError(
            f"Cannot locate sex/gender column.\nAvailable columns: {list(df.columns)}"
        )

    # ── Parse the column — handle raw codes AND text labels ──────────
    raw_vals = df[sex_col].astype(str).str.strip()

    print(f"\n[Data] Detected gender column : '{sex_col}'")
    print(f"[Data] Unique values ({len(raw_vals.unique())}) : "
          f"{sorted(raw_vals.unique())}")

    # Female indicator — covers both label formats in one pass
    FEMALE_RAW   = {"A92", "A95"}                    # raw OpenML codes
    FEMALE_TEXT  = {"female div/dep/mar",             # newer OpenML text labels
                    "female single",
                    "female div/sep/mar",
                    "female mar",
                    "female"}

    def is_female(v: str) -> bool:
        v_stripped = v.strip()
        # Raw code match
        if v_stripped in FEMALE_RAW:
            return True
        # Text label match (case-insensitive, substring)
        vl = v_stripped.lower()
        if vl in FEMALE_TEXT:
            return True
        if vl.startswith("female"):
            return True
        return False

    s = raw_vals.apply(is_female).astype(int).values

    # ── Sanity checks ────────────────────────────────────────────────
    n_female = int(np.sum(s))
    n_male   = int(np.sum(s == 0))
    ratio    = n_female / len(s)

    print(f"[Data] Protected group (female, s=1) : {n_female} samples ({ratio:.3f})")
    print(f"[Data] Privileged group (male,   s=0) : {n_male} samples ({1-ratio:.3f})")
    print(f"[Data] n_samples total               : {len(y)}")
    print(f"[Data] n_bad (y=1)  / n_good (y=0)  : {int(np.sum(y))} / {int(np.sum(y==0))}")

    # Hard guard: if either group is empty the experiment is meaningless
    if n_female == 0:
        raise RuntimeError(
            "FATAL: Protected group (female) is EMPTY after parsing.\n"
            f"Column '{sex_col}' unique values: {sorted(raw_vals.unique())}\n"
            "Check is_female() logic above — add the observed value strings to "
            "FEMALE_RAW or FEMALE_TEXT."
        )
    if n_male == 0:
        raise RuntimeError(
            "FATAL: Privileged group (male) is EMPTY after parsing.\n"
            f"Column '{sex_col}' unique values: {sorted(raw_vals.unique())}"
        )
    if ratio < 0.05 or ratio > 0.95:
        import warnings as _w
        _w.warn(
            f"Unusual protected-group ratio: {ratio:.3f}. "
            "Verify that female/male assignment is correct.",
            UserWarning,
        )

    # ── Feature matrix (one-hot encode categoricals) ─────────────────
    X_df = pd.get_dummies(df, drop_first=True)
    print(f"[Data] n_features after OHE          : {X_df.shape[1]}")

    return X_df.values, y, s


# ──────────────────────────────────────────────────────────────
#  STATISTICAL TESTS
# ──────────────────────────────────────────────────────────────

def cohens_dz(a, b):
    diff = np.array(a) - np.array(b)
    sd   = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 1e-12 else 0.0


def paired_tests_bh(df_raw, metrics=("DI", "Acc", "EOD", "AOD")):
    """
    For each metric, compute three paired comparisons:
      FE-HITL vs DL, FE-HITL vs Debiased-HITL, Debiased-HITL vs DL.
    Apply Benjamini-Hochberg correction across all comparisons × metrics.
    Returns a DataFrame with one row per comparison.
    """
    comparisons = [
        ("FE-HITL",       "DL"),
        ("FE-HITL",       "Debiased-HITL"),
        ("Debiased-HITL", "DL"),
    ]
    rows = []
    for metric in metrics:
        for model_a, model_b in comparisons:
            a = df_raw[df_raw["model"] == model_a][metric].values
            b = df_raw[df_raw["model"] == model_b][metric].values
            if len(a) != len(b) or len(a) == 0:
                continue
            diff    = a - b
            t_stat, p_raw = stats.ttest_rel(a, b)
            n       = len(diff)
            mean_d  = np.mean(diff)
            ci_hw   = 1.96 * np.std(diff, ddof=1) / np.sqrt(n)
            rows.append({
                "metric"    : metric,
                "model_A"   : model_a,
                "model_B"   : model_b,
                "n_pairs"   : n,
                "mean_diff" : mean_d,
                "ci_low"    : mean_d - ci_hw,
                "ci_high"   : mean_d + ci_hw,
                "t_stat"    : t_stat,
                "df"        : n - 1,
                "p_raw"     : p_raw,
                "dz"        : cohens_dz(a, b),
            })

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        _, p_bh, _, _ = multipletests(result_df["p_raw"].values, method="fdr_bh")
        result_df["p_bh"] = p_bh
        result_df["sig_bh"] = result_df["p_bh"] < 0.05
    return result_df


def print_test_summary(test_df, title=""):
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    for _, row in test_df.iterrows():
        sig = "***" if row["p_bh"] < 0.001 else ("**" if row["p_bh"] < 0.01
              else ("*" if row["p_bh"] < 0.05 else "n.s."))
        print(f"  {row['metric']:4s}  {row['model_A']:15s} vs {row['model_B']:15s} : "
              f"Δ={row['mean_diff']:+.4f} [{row['ci_low']:+.4f},{row['ci_high']:+.4f}]  "
              f"t({int(row['df'])})={row['t_stat']:+.3f}  p(BH)={row['p_bh']:.4f}{sig}  "
              f"dz={row['dz']:+.3f}")


# ──────────────────────────────────────────────────────────────
#  MAIN EXPERIMENT (30-seed)
# ──────────────────────────────────────────────────────────────

def run_main_experiment(seeds, X_all, y_all, s_all):
    """
    For each seed:
      1. Stratified 70/30 train–test split.
      2. Train ONE shared MLP (SharedMLP) on the training set.
      3. Evaluate four methods from the same model:
         LR, DL (shared MLP), Debiased-HITL, FE-HITL.
    """
    print("\n" + "="*65)
    print(f"  MAIN EXPERIMENT — {len(seeds)} seeds")
    print("="*65)

    rows = []
    for seed in seeds:
        # ── Data split ──────────────────────────────────────────
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X_all, y_all, s_all,
            test_size=0.30, random_state=seed, stratify=y_all,
        )

        # ── Logistic Regression baseline ────────────────────────
        scaler_lr = StandardScaler()
        lr = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        lr.fit(scaler_lr.fit_transform(X_tr), y_tr)
        p_lr = lr.predict(scaler_lr.transform(X_te))
        rows.append({"model": "LR", "seed": seed,
                     **compute_all_metrics(y_te, p_lr, s_te)})

        # ── Shared MLP base model ────────────────────────────────
        mlp = SharedMLP(seed)
        mlp.fit(X_tr, y_tr)

        # DL: direct MLP prediction
        p_dl = mlp.predict(X_te)
        rows.append({"model": "DL", "seed": seed,
                     **compute_all_metrics(y_te, p_dl, s_te)})

        # Shared probability scores for post-processing
        probs = mlp.predict_proba_pos(X_te)

        # Debiased-HITL: fixed-ratio proportional correction
        probs_dh = debiased_hitl_correction(probs, s_te)
        p_dh     = (probs_dh >= 0.5).astype(int)
        rows.append({"model": "Debiased-HITL", "seed": seed,
                     **compute_all_metrics(y_te, p_dh, s_te)})

        # FE-HITL: multi-option generation + value arbitration
        probs_fh = fehitl_correction(probs, s_te)
        p_fh     = (probs_fh >= 0.5).astype(int)
        rows.append({"model": "FE-HITL", "seed": seed,
                     **compute_all_metrics(y_te, p_fh, s_te)})

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(f"{OUT_DIR}/credit_main_raw.csv", index=False)

    # ── Summary table ────────────────────────────────────────────
    def ci95(x):
        return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

    agg = df_raw.groupby("model").agg(
        Acc_mean=("Acc", "mean"), Acc_std=("Acc", "std"), Acc_ci=("Acc", ci95),
        DI_mean =("DI",  "mean"), DI_std =("DI",  "std"), DI_ci =("DI",  ci95),
        EOD_mean=("EOD", "mean"), EOD_std=("EOD", "std"), EOD_ci=("EOD", ci95),
        AOD_mean=("AOD", "mean"), AOD_std=("AOD", "std"), AOD_ci=("AOD", ci95),
        n=("Acc", "count"),
    ).reset_index()
    agg.to_csv(f"{OUT_DIR}/credit_main_summary.csv", index=False)

    print("\nMain experiment results (mean ± SD [95% CI]):")
    for _, row in agg.iterrows():
        print(f"  {row['model']:15s} | "
              f"Acc={row['Acc_mean']:.4f}±{row['Acc_std']:.4f} "
              f"[{row['Acc_mean']-row['Acc_ci']:.4f},{row['Acc_mean']+row['Acc_ci']:.4f}] | "
              f"DI={row['DI_mean']:.4f}±{row['DI_std']:.4f} "
              f"[{row['DI_mean']-row['DI_ci']:.4f},{row['DI_mean']+row['DI_ci']:.4f}] | "
              f"EOD={row['EOD_mean']:.4f}±{row['EOD_std']:.4f} | "
              f"AOD={row['AOD_mean']:.4f}±{row['AOD_std']:.4f}")

    # ── Paired tests ─────────────────────────────────────────────
    test_df = paired_tests_bh(df_raw)
    test_df.to_csv(f"{OUT_DIR}/credit_paired_tests.csv", index=False)
    print_test_summary(test_df, "Paired tests (BH-corrected) — main experiment")

    return df_raw, agg, test_df


# ──────────────────────────────────────────────────────────────
#  5-FOLD CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────

def run_kfold_cv(seeds, X_all, y_all, s_all, n_folds=5):
    """
    Stratified 5-fold CV.  For each (fold, seed) combination:
      - ONE SharedMLP is trained per (fold, seed).
      - DL, Debiased-HITL, and FE-HITL all use that same model instance.
    Total evaluations: 5 folds × 30 seeds = 150 per method.
    """
    print("\n" + "="*65)
    print(f"  5-FOLD CROSS-VALIDATION — {n_folds} folds × {len(seeds)} seeds")
    print("="*65)

    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"  Fold {fold_idx + 1}/{n_folds} ...", flush=True)
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        s_tr, s_te = s_all[tr_idx], s_all[te_idx]

        for seed in seeds:
            mlp = SharedMLP(seed)
            mlp.fit(X_tr, y_tr)
            probs = mlp.predict_proba_pos(X_te)

            # DL
            p_dl = mlp.predict(X_te)
            rows.append({"fold": fold_idx, "seed": seed, "model": "DL",
                         **compute_all_metrics(y_te, p_dl, s_te)})

            # Debiased-HITL
            probs_dh = debiased_hitl_correction(probs, s_te)
            p_dh     = (probs_dh >= 0.5).astype(int)
            rows.append({"fold": fold_idx, "seed": seed, "model": "Debiased-HITL",
                         **compute_all_metrics(y_te, p_dh, s_te)})

            # FE-HITL
            probs_fh = fehitl_correction(probs, s_te)
            p_fh     = (probs_fh >= 0.5).astype(int)
            rows.append({"fold": fold_idx, "seed": seed, "model": "FE-HITL",
                         **compute_all_metrics(y_te, p_fh, s_te)})

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(f"{OUT_DIR}/credit_cv_raw.csv", index=False)

    # ── Per-fold summary ─────────────────────────────────────────
    per_fold = df_cv.groupby(["model", "fold"]).agg(
        DI_mean=("DI", "mean"), DI_std=("DI", "std"),
        Acc_mean=("Acc", "mean"), EOD_mean=("EOD", "mean"),
        n=("DI", "count"),
    ).reset_index()
    per_fold.to_csv(f"{OUT_DIR}/credit_cv_perfold.csv", index=False)

    # ── Overall CV summary ───────────────────────────────────────
    def ci95(x): return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

    overall = df_cv.groupby("model").agg(
        Acc_mean=("Acc", "mean"), Acc_std=("Acc", "std"), Acc_ci=("Acc", ci95),
        DI_mean =("DI",  "mean"), DI_std =("DI",  "std"), DI_ci =("DI",  ci95),
        EOD_mean=("EOD", "mean"), EOD_std=("EOD", "std"), EOD_ci=("EOD", ci95),
        AOD_mean=("AOD", "mean"), AOD_std=("AOD", "std"), AOD_ci=("AOD", ci95),
        n=("Acc", "count"),
    ).reset_index()
    overall.to_csv(f"{OUT_DIR}/credit_cv_summary.csv", index=False)

    print("\n5-fold CV results (mean ± SD [95% CI]):")
    for _, row in overall.iterrows():
        print(f"  {row['model']:15s} | "
              f"Acc={row['Acc_mean']:.4f}±{row['Acc_std']:.4f} | "
              f"DI={row['DI_mean']:.4f}±{row['DI_std']:.4f} "
              f"[{row['DI_mean']-row['DI_ci']:.4f},{row['DI_mean']+row['DI_ci']:.4f}] | "
              f"EOD={row['EOD_mean']:.4f}±{row['EOD_std']:.4f}")

    # ── Paired tests for CV ──────────────────────────────────────
    cv_test_df = paired_tests_bh(df_cv)
    cv_test_df.to_csv(f"{OUT_DIR}/credit_cv_paired_tests.csv", index=False)
    print_test_summary(cv_test_df, "Paired tests (BH-corrected) — 5-fold CV")

    return df_cv, overall, cv_test_df


# ──────────────────────────────────────────────────────────────
#  CORRECTION MECHANISM DIAGNOSTIC (optional, informative)
# ──────────────────────────────────────────────────────────────

def run_correction_diagnostic(seeds, X_all, y_all, s_all, n_diag=5):
    """
    For a small subset of seeds, print per-seed correction details:
    base DI, Debiased DI, FE-HITL DI, chosen factor, to verify that
    the two methods are making different decisions.
    """
    print("\n" + "="*65)
    print(f"  CORRECTION DIAGNOSTIC — first {n_diag} seeds")
    print("="*65)
    print(f"  {'seed':>4}  {'base_DI':>8}  {'debias_DI':>10}  "
          f"{'fehitl_DI':>10}  {'debias_factor':>14}  {'fehitl_factor':>14}")

    for seed in seeds[:n_diag]:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X_all, y_all, s_all,
            test_size=0.30, random_state=seed, stratify=y_all,
        )
        mlp = SharedMLP(seed)
        mlp.fit(X_tr, y_tr)
        probs = mlp.predict_proba_pos(X_te)

        base_di = safe_di((probs >= 0.5).astype(int), s_te)

        probs_dh = debiased_hitl_correction(probs.copy(), s_te)
        di_dh    = safe_di((probs_dh >= 0.5).astype(int), s_te)
        # Reconstruct factor from ratio of mean probs for unprivileged group
        mask_u   = (s_te == 1)
        dh_factor = (np.mean(probs_dh[mask_u]) / np.mean(probs[mask_u])
                     if np.mean(probs[mask_u]) > 1e-9 else 1.0)

        probs_fh = fehitl_correction(probs.copy(), s_te)
        di_fh    = safe_di((probs_fh >= 0.5).astype(int), s_te)
        fh_factor = (np.mean(probs_fh[mask_u]) / np.mean(probs[mask_u])
                     if np.mean(probs[mask_u]) > 1e-9 else 1.0)

        print(f"  {seed:>4}  {base_di:>8.4f}  {di_dh:>10.4f}  "
              f"{di_fh:>10.4f}  {dh_factor:>14.4f}  {fh_factor:>14.4f}")


# ──────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    mode = "QUICK TEST" if QUICK_TEST else "FULL RUN (30 seeds)"
    print(f"\nGerman Credit Revision Experiments — {mode}")
    print(f"Output directory: {OUT_DIR}/")
    print("="*65)

    # Load data once; reuse across all experiments
    X_all, y_all, s_all = load_german_credit()

    seeds = SEEDS_QUICK if QUICK_TEST else SEEDS_FULL

    # Optional: verify that the two correction methods differ
    run_correction_diagnostic(seeds, X_all, y_all, s_all, n_diag=5)

    # Main 30-seed experiment
    df_main, summary_main, tests_main = run_main_experiment(
        seeds, X_all, y_all, s_all
    )

    # 5-fold cross-validation
    df_cv, summary_cv, tests_cv = run_kfold_cv(
        seeds, X_all, y_all, s_all, n_folds=5
    )

    print("\n" + "="*65)
    print(f"  All results saved to '{OUT_DIR}/'")
    if QUICK_TEST:
        print("  Smoke-test passed. Set QUICK_TEST = False for the full run.")
    print("="*65)


if __name__ == "__main__":
    main()