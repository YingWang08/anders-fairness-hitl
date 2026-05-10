"""
run_stats_supplement.py — 统计推断补充脚本
对应审稿人 W3 要求：
  "Increase to 30+ runs; report 95% CIs; report effect sizes (Cohen's d, η²);
   apply corrections for multiple comparisons (Bonferroni or Benjamini–Hochberg);
   report sensitivity to random seed selection."

用途：
  读取已有的 30 次实验原始 CSV（农业数据集 / German Credit / Adult Income），
  输出完整的统计推断报告，可直接粘贴进论文 Table 及 Statistical Analysis 小节。

输入文件（均来自 revision_results/ 或当前目录）：
  - bias_sensitivity_raw.csv     （农业数据集，偏置敏感性分析，来自 py2）
  - ablation_raw.csv             （农业数据集，消融实验，来自 py2）
  - german_credit_raw.csv        （German Credit，来自 py1）
  - adult_income_raw.csv         （Adult Income，来自 run_adult_income.py）

输出文件（均写入 revision_results/stats/）：
  - {dataset}_summary_full.csv   含均值/SD/CI/中位数/IQR 的完整描述统计
  - {dataset}_pairwise.csv       FE-HITL vs 其余的配对 t + BH 校正 + Cohen's d
  - {dataset}_anova.csv          ANOVA F 值 + η² 效应量
  - cross_dataset_comparison.csv 三数据集横向比较（DI/EOD 的 FE-HITL 效果）
  - seed_sensitivity.csv         种子敏感性分析（各 seed 下指标的 z-score 偏离）
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

IN_DIRS  = ['revision_results', '.']   # 按顺序搜索输入文件
OUT_DIR  = 'revision_results/stats'
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════
def find_file(filename):
    """在 IN_DIRS 中按顺序找到第一个存在的文件路径。"""
    for d in IN_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def ci95(values, use_t=True):
    """计算 95% CI，使用 t 分布（小样本更准确）。"""
    n   = len(values)
    se  = np.std(values, ddof=1) / np.sqrt(n)
    if use_t:
        t_crit = stats.t.ppf(0.975, df=n - 1)
    else:
        t_crit = 1.96
    margin = t_crit * se
    mean   = np.mean(values)
    return mean, np.std(values, ddof=1), mean - margin, mean + margin


def cohens_d_paired(a, b):
    """配对设计 Cohen's d = mean(diff) / std(diff)。"""
    diff = np.asarray(a) - np.asarray(b)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0


def eta_squared_oneway(groups):
    """
    单因素 ANOVA η²（总变异中组间变异的比例）。
    groups: list of 1-D arrays，每个 array 对应一个组。
    """
    grand_mean  = np.mean(np.concatenate(groups))
    ss_between  = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total    = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    return ss_between / ss_total if ss_total > 0 else 0.0


def seed_sensitivity_report(df, model_col='model', seed_col='seed',
                             metrics=None, target_model='FE-HITL'):
    """
    种子敏感性分析：
    对 target_model 的每个指标，按 seed 计算 z-score，
    标记离均值 >2σ 的 outlier seed，量化对结论稳健性的影响。
    """
    if metrics is None:
        metrics = ['DI', 'EOD', 'AOD']
    sub = df[df[model_col] == target_model].copy()
    rows = []
    for m in metrics:
        if m not in sub.columns:
            continue
        vals   = sub[m].values
        mean_v = np.mean(vals)
        std_v  = np.std(vals, ddof=1)
        z      = (vals - mean_v) / std_v if std_v > 0 else np.zeros_like(vals)
        for seed_val, v, z_val in zip(sub[seed_col].values, vals, z):
            rows.append({
                'model'   : target_model,
                'metric'  : m,
                'seed'    : seed_val,
                'value'   : v,
                'z_score' : z_val,
                'outlier' : abs(z_val) > 2,
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# 核心统计流程（对单个数据集 DataFrame 做完整分析）
# ════════════════════════════════════════════════════════════════════
def analyze_dataset(df, dataset_name, model_col='model',
                    target='FE-HITL', metrics=None, group_cols=None):
    """
    Parameters
    ----------
    df           : 原始实验 DataFrame（每行一个 seed × model 的结果）
    dataset_name : 用于命名输出文件的字符串
    model_col    : 模型名称列名
    target       : 与之比较的目标模型（FE-HITL 或 Full）
    metrics      : 需要分析的指标列名列表
    group_cols   : 除 model_col 外的分组列（如 bias 率、ablation variant）
    """
    if metrics is None:
        # 自动检测数字列
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        metrics  = [c for c in num_cols if c not in ['seed']]

    models = df[model_col].unique().tolist()

    # ── 1. 完整描述统计 ──────────────────────────────────────────
    desc_rows = []
    group_by  = [model_col] + (group_cols or [])
    for keys, grp in df.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_by, keys))
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna().values
            if len(vals) == 0:
                continue
            mean_v, std_v, ci_lo, ci_hi = ci95(vals)
            desc_rows.append({
                **base,
                'metric'  : m,
                'n'       : len(vals),
                'mean'    : round(mean_v, 4),
                'std'     : round(std_v, 4),
                'median'  : round(np.median(vals), 4),
                'IQR'     : round(np.percentile(vals, 75) - np.percentile(vals, 25), 4),
                'CI95_lo' : round(ci_lo, 4),
                'CI95_hi' : round(ci_hi, 4),
                'min'     : round(np.min(vals), 4),
                'max'     : round(np.max(vals), 4),
            })
    desc_df = pd.DataFrame(desc_rows)
    desc_path = os.path.join(OUT_DIR, f'{dataset_name}_summary_full.csv')
    desc_df.to_csv(desc_path, index=False)
    print(f"  [描述统计] → {desc_path}  ({len(desc_df)} rows)")

    # ── 2. 配对 t 检验 + Cohen's d + BH 校正 ────────────────────
    # 仅对无分组的完整数据集做（bias 分层的在各层分别做）
    fe_data       = df[df[model_col] == target].sort_values('seed')
    baselines     = [m for m in models if m != target]
    pairwise_rows = []
    raw_p         = []

    for bl in baselines:
        bl_data = df[df[model_col] == bl].sort_values('seed')
        for metric in metrics:
            if metric not in fe_data.columns or metric not in bl_data.columns:
                continue
            fe_vals = fe_data[metric].values
            bl_vals = bl_data[metric].values
            if len(fe_vals) != len(bl_vals) or len(fe_vals) < 2:
                continue
            t_stat, p_raw = stats.ttest_rel(fe_vals, bl_vals)
            d             = cohens_d_paired(fe_vals, bl_vals)
            diff          = fe_vals - bl_vals
            _, _, ci_lo, ci_hi = ci95(diff)
            pairwise_rows.append({
                'comparison'  : f'{target} vs {bl}',
                'metric'      : metric,
                f'{target}_M' : round(np.mean(fe_vals), 4),
                f'{bl}_M'     : round(np.mean(bl_vals), 4),
                'mean_diff'   : round(np.mean(diff), 4),
                'diff_CI_lo'  : round(ci_lo, 4),
                'diff_CI_hi'  : round(ci_hi, 4),
                't_stat'      : round(t_stat, 4),
                'df'          : len(fe_vals) - 1,
                'p_raw'       : round(p_raw, 6),
                "Cohen's_d"   : round(d, 4),
            })
            raw_p.append(p_raw)

    if pairwise_rows:
        _, p_bh, _, _ = multipletests(raw_p, method='fdr_bh')
        _, p_bonf, _, _ = multipletests(raw_p, method='bonferroni')
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df['p_BH']        = [round(x, 6) for x in p_bh]
        pairwise_df['p_Bonferroni']= [round(x, 6) for x in p_bonf]
        pairwise_df['sig_BH_0.05'] = p_bh < 0.05
        pairwise_df['sig_Bonf_0.05'] = p_bonf < 0.05
        pw_path = os.path.join(OUT_DIR, f'{dataset_name}_pairwise.csv')
        pairwise_df.to_csv(pw_path, index=False)
        print(f"  [配对检验] → {pw_path}  ({len(pairwise_df)} comparisons, "
              f"BH corrected)")
    else:
        pairwise_df = pd.DataFrame()
        print("  [配对检验] 跳过（数据不足）")

    # ── 3. 单因素 ANOVA + η² ─────────────────────────────────────
    anova_rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        groups = [df[df[model_col] == m][metric].dropna().values
                  for m in models]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_anova = f_oneway(*groups)
        eta2            = eta_squared_oneway(groups)
        anova_rows.append({
            'metric'   : metric,
            'F_stat'   : round(f_stat, 4),
            'df_between': len(groups) - 1,
            'df_within': sum(len(g) for g in groups) - len(groups),
            'p_anova'  : round(p_anova, 6),
            'eta2'     : round(eta2, 4),
            'sig_0.05' : p_anova < 0.05,
        })
    if anova_rows:
        anova_df   = pd.DataFrame(anova_rows)
        anova_path = os.path.join(OUT_DIR, f'{dataset_name}_anova.csv')
        anova_df.to_csv(anova_path, index=False)
        print(f"  [ANOVA+η²] → {anova_path}")
    else:
        anova_df = pd.DataFrame()

    # ── 4. 种子敏感性 ────────────────────────────────────────────
    if 'seed' in df.columns:
        seed_df = seed_sensitivity_report(df, model_col=model_col, target_model=target)
        seed_path = os.path.join(OUT_DIR, f'{dataset_name}_seed_sensitivity.csv')
        seed_df.to_csv(seed_path, index=False)
        n_out = seed_df['outlier'].sum()
        print(f"  [种子敏感性] → {seed_path}  (outlier seeds: {n_out})")
    else:
        seed_df = pd.DataFrame()

    return desc_df, pairwise_df, anova_df, seed_df


# ════════════════════════════════════════════════════════════════════
# 跨数据集横向比较（汇总三个数据集的 FE-HITL 效果）
# ════════════════════════════════════════════════════════════════════
def cross_dataset_comparison(datasets_info):
    """
    datasets_info: list of (name, df, metrics)
    汇总 FE-HITL 在各数据集上的 DI 和 EOD 均值/CI，
    生成审稿人 W6 所需的横向比较表。
    """
    rows = []
    for name, df, metrics in datasets_info:
        if df is None or df.empty:
            continue
        model_col = 'model' if 'model' in df.columns else 'variant'
        target_col = 'FE-HITL' if 'FE-HITL' in df[model_col].values else 'Full'
        sub = df[df[model_col] == target_col]
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = sub[m].dropna().values
            if len(vals) == 0:
                continue
            mean_v, std_v, ci_lo, ci_hi = ci95(vals)
            rows.append({
                'dataset'  : name,
                'model'    : target_col,
                'metric'   : m,
                'mean'     : round(mean_v, 4),
                'std'      : round(std_v, 4),
                'CI95_lo'  : round(ci_lo, 4),
                'CI95_hi'  : round(ci_hi, 4),
                'n_seeds'  : len(vals),
            })
    cross_df   = pd.DataFrame(rows)
    cross_path = os.path.join(OUT_DIR, 'cross_dataset_comparison.csv')
    cross_df.to_csv(cross_path, index=False)
    print(f"\n  [跨数据集对比] → {cross_path}  ({len(cross_df)} rows)")
    return cross_df


# ════════════════════════════════════════════════════════════════════
# 格式化终端报告（直接可用于论文 Results 小节）
# ════════════════════════════════════════════════════════════════════
def print_pairwise_report(dataset_name, pairwise_df):
    if pairwise_df.empty:
        return
    print(f"\n── {dataset_name}：FE-HITL 配对 t 检验结果（BH 校正）──")
    print(f"{'Comparison':<30} {'Metric':<10} {'Diff':>8} "
          f"{'95%CI':>18} {'t':>7} {'p_raw':>9} "
          f"{'p_BH':>9} {'d':>7} {'Sig?':>5}")
    print("-" * 110)
    for _, r in pairwise_df.iterrows():
        ci_str = f"[{r['diff_CI_lo']:+.4f},{r['diff_CI_hi']:+.4f}]"
        comp   = str(r['comparison'])[:29]
        metric = str(r['metric'])[:9]
        sig    = "✓" if r['sig_BH_0.05'] else "✗"
        d_val = r["Cohen's_d"]
        print(f"{comp:<30} {metric:<10} {r['mean_diff']:>+8.4f} "
              f"{ci_str:>18} {r['t_stat']:>7.3f} {r['p_raw']:>9.5f} "
              f"{r['p_BH']:>9.5f} {d_val:>7.4f} {sig:>5}")


def print_anova_report(dataset_name, anova_df):
    if anova_df.empty:
        return
    print(f"\n── {dataset_name}：单因素 ANOVA + η² ──")
    print(f"{'Metric':<12} {'F':>8} {'df_b':>5} {'df_w':>5} "
          f"{'p_anova':>10} {'η²':>8} {'Sig?':>5}")
    print("-" * 60)
    for _, r in anova_df.iterrows():
        sig = "✓" if r['sig_0.05'] else "✗"
        print(f"{str(r['metric']):<12} {r['F_stat']:>8.3f} "
              f"{r['df_between']:>5} {r['df_within']:>5} "
              f"{r['p_anova']:>10.5f} {r['eta2']:>8.4f} {sig:>5}")


# ════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("统计推断补充分析  (对应审稿人 W3 要求)")
    print("=" * 70)

    datasets_info_for_cross = []

    # ── 农业数据集：偏置敏感性 ──────────────────────────────────
    ag_bias_file = find_file('bias_sensitivity_raw.csv')
    if ag_bias_file:
        print(f"\n[1] 农业数据集 — 偏置敏感性分析  ({ag_bias_file})")
        df_ag_bias = pd.read_csv(ag_bias_file)
        # 按偏置率分层
        for bias_val, grp in df_ag_bias.groupby('bias'):
            tag = f'agricultural_bias{int(bias_val*100):02d}'
            analyze_dataset(
                grp.reset_index(drop=True), tag,
                model_col='model',
                target='FE-HITL',
                metrics=['R2', 'RMSE', 'DI', 'EOD', 'AOD'],
            )
        # 对 bias=0.30（原始设定）做跨数据集对比数据收集
        df_b30 = df_ag_bias[df_ag_bias['bias'] == 0.30]
        datasets_info_for_cross.append(
            ('Agricultural(bias=0.30)', df_b30, ['DI', 'EOD', 'AOD'])
        )
    else:
        print("\n[1] 未找到 bias_sensitivity_raw.csv，跳过农业偏置分析。")

    # ── 农业数据集：消融实验 ─────────────────────────────────────
    ablation_file = find_file('ablation_raw.csv')
    if ablation_file:
        print(f"\n[2] 农业数据集 — 消融实验  ({ablation_file})")
        df_abl = pd.read_csv(ablation_file)
        # 消融实验的 model/variant 列名可能是 'variant'
        mc = 'variant' if 'variant' in df_abl.columns else 'model'
        analyze_dataset(
            df_abl, 'agricultural_ablation',
            model_col=mc,
            target='Full',
            metrics=['R2', 'RMSE', 'DI', 'EOD', 'AOD'],
        )
    else:
        print("\n[2] 未找到 ablation_raw.csv，跳过消融实验分析。")

    # ── German Credit 数据集 ─────────────────────────────────────
    gc_file = find_file('german_credit_raw.csv')
    if gc_file:
        print(f"\n[3] German Credit 数据集  ({gc_file})")
        df_gc = pd.read_csv(gc_file)
        _, pw_gc, anova_gc, _ = analyze_dataset(
            df_gc, 'german_credit',
            model_col='model',
            target='FE-HITL',
            metrics=['Accuracy', 'DI', 'EOD', 'AOD'],
        )
        print_pairwise_report('German Credit', pw_gc)
        print_anova_report('German Credit', anova_gc)
        datasets_info_for_cross.append(('German Credit', df_gc, ['DI', 'EOD', 'AOD']))
    else:
        print("\n[3] 未找到 german_credit_raw.csv，跳过 German Credit 分析。")

    # ── Adult Income 数据集 ──────────────────────────────────────
    ai_file = find_file('adult_income_raw.csv')
    if ai_file:
        print(f"\n[4] Adult Income 数据集  ({ai_file})")
        df_ai = pd.read_csv(ai_file)
        _, pw_ai, anova_ai, _ = analyze_dataset(
            df_ai, 'adult_income',
            model_col='model',
            target='FE-HITL',
            metrics=['Accuracy', 'DI', 'EOD', 'AOD'],
        )
        print_pairwise_report('Adult Income', pw_ai)
        print_anova_report('Adult Income', anova_ai)
        datasets_info_for_cross.append(('Adult Income', df_ai, ['DI', 'EOD', 'AOD']))
    else:
        print("\n[4] 未找到 adult_income_raw.csv，跳过 Adult Income 分析。")

    # ── 跨数据集横向比较 ─────────────────────────────────────────
    if datasets_info_for_cross:
        print("\n[5] 生成跨数据集横向比较表...")
        cross_df = cross_dataset_comparison(datasets_info_for_cross)
        print("\n  FE-HITL 效果横向摘要：")
        print(cross_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"✓ 所有统计结果已写入 {OUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()