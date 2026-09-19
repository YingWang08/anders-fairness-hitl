"""
regen_stats.py
==============
Regenerates every DESCRIPTIVE table and every AGRICULTURAL paired test in the
manuscript from the deposited raw CSVs in one pass, then checks the output
against each number Reviewer #4 computed (reviewer_number_check.csv).
German Credit inference is in inference_fixed.py; workload in
fairness_cost_joint.py.

    python regen_stats.py            # run from the repository root
    python regen_stats.py --ci t     # t-based descriptive CIs instead of normal

Corrections relative to the earlier R3 draft of this script
    * BH family restored to the AUTHORS' ORIGINAL family (run_stats_supplement.py):
      FE-HITL minus {LR, DL, Debiased-HITL} x {R2, RMSE, DI, EOD, AOD} = 15.
      The draft had replaced FE-HITL vs LR by Debiased-HITL vs DL, which gives
      R^2 p_BH = 1.80e-8 instead of the reviewer's 1.20e-8.
    * "Table 4" is the GERMAN CREDIT table. The draft built it from agricultural
      data ([0.0270, 0.0451]); from credit_main_raw.csv the FE-HITL |AOD| CI is
      [0.0216, 0.0413], as the reviewer reports.
    * Ablation contrasts are seed-matched; the R2 "w/o MOG" row is relabelled as
      the correction-strength variant it is; the identical w/o-F&U row is
      reported as identical-by-construction, not tested.

Conventions (also written to MANIFEST.json)
    contrast direction : treatment minus comparator (FE-HITL - X; Full - variant)
    pairing            : on seed; unmatched seeds raise an error (no silent drops)
    descriptive CI     : mean +/- 1.96 SD/sqrt(n) (normal approximation, as stated
                         in the manuscript) unless --ci t
    contrast CI        : t_{n-1}, consistent with the paired t-test
    effect size        : Cohen's dz = mean(diff) / SD(diff)
    multiple testing   : Benjamini-Hochberg within each declared family
    precision          : full precision; rounding only in *_display columns
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from statsmodels.stats.multitest import multipletests  # direct dependency

AGRI_METRICS = ['R2', 'RMSE', 'DI', 'EOD', 'AOD']
MODELS = ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']
ABLATION_LABELS = {
    'Full': 'Full',
    'w/o HED': 'No correction, same base model (R2 label: w/o HED)',
    'w/o MOG': 'Reduced correction strength, coef 0.2 (R2 label: w/o MOG)',
    'w/o F&U': 'w/o F&U (identical to Full by order of evaluation)',
}


# ── Primitives ───────────────────────────────────────────────────────
def describe(x, ci):
    x = np.asarray(x, dtype=float)
    n, m, sd = len(x), float(np.mean(x)), float(np.std(x, ddof=1))
    q = stats.norm.ppf(0.975) if ci == 'normal' else stats.t.ppf(0.975, n - 1)
    hw = q * sd / np.sqrt(n)
    return {'mean': m, 'sd': sd, 'n': n, 'ci_low': m - hw, 'ci_high': m + hw}


def display(d, digits=3):
    return (f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f} "
            f"[{d['ci_low']:.{digits}f}, {d['ci_high']:.{digits}f}]")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n, md, sd = len(d), float(np.mean(d)), float(np.std(d, ddof=1))
    if sd == 0.0:
        return {'mean_diff': md, 'sd_diff': 0.0, 'ci_low': md, 'ci_high': md,
                't': np.nan, 'df': n - 1, 'p_raw': np.nan, 'dz': np.nan,
                'n_pairs': n, 'identical_by_construction': True}
    t, p = stats.ttest_rel(a, b)
    hw = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'mean_diff': md, 'sd_diff': sd, 'ci_low': md - hw, 'ci_high': md + hw,
            't': float(t), 'df': n - 1, 'p_raw': float(p), 'dz': md / sd,
            'n_pairs': n, 'identical_by_construction': False}


def aligned(df, col, a, b, metric):
    A = df[df[col] == a].set_index('seed')[metric]
    B = df[df[col] == b].set_index('seed')[metric]
    if A.index.has_duplicates or B.index.has_duplicates:
        raise ValueError(f'duplicate seeds for {a}/{b}')
    if set(A.index) != set(B.index):
        raise ValueError(f'unmatched seeds between {a} and {b} on {metric}')
    idx = A.index.sort_values()
    return A.loc[idx].values, B.loc[idx].values


def bh(rows, family):
    out = pd.DataFrame(rows)
    ok = out['p_raw'].notna()
    out['p_bh'] = np.nan
    if ok.any():
        out.loc[ok, 'p_bh'] = multipletests(out.loc[ok, 'p_raw'], method='fdr_bh')[1]
    out['bh_family'] = family
    out['bh_family_size'] = int(ok.sum())
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'UNKNOWN (not a git checkout)'


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agri-dir', default='experiments/revision_results')
    ap.add_argument('--credit-dir', default='experiments/credit_revision_results')
    ap.add_argument('--r3-dir', default='experiments/revision_results/r3')
    ap.add_argument('--out-dir', default='r3_tables')
    ap.add_argument('--ci', choices=['normal', 't'], default='normal')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    O = lambda name: os.path.join(args.out_dir, name)
    inputs = {}

    def load(folder, name, required=True):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            if required:
                raise FileNotFoundError(p)
            return None
        df = pd.read_csv(p)
        inputs[p] = {'sha256': sha256(p), 'rows': int(len(df))}
        return df

    bias = load(args.agri_dir, 'bias_sensitivity_raw.csv')
    abl = load(args.agri_dir, 'ablation_raw.csv')
    sweep = load(args.agri_dir, 'interv_sensitivity_raw.csv')
    struct = load(args.agri_dir, 'bias_structure_robustness_raw.csv')
    binar = load(args.agri_dir, 'binarization_sensitivity_raw.csv')
    rev = load(args.agri_dir, 'reverse_discrimination_raw.csv')
    cv = load(args.agri_dir, 'kfold_cv_agri_raw.csv')
    credit = load(args.credit_dir, 'credit_main_raw.csv')
    mog = load(args.r3_dir, 'r3_mog_ablation_raw.csv', required=False)
    credit['DI_gap'] = (credit['DI'] - 1.0).abs()
    main30 = bias[np.isclose(bias['bias'], 0.30)].copy()

    # Table 1 -------------------------------------------------------------
    t1 = []
    for m in MODELS:
        for k in AGRI_METRICS:
            d = describe(main30.loc[main30.model == m, k], args.ci)
            t1.append({'model': m, 'metric': k, **d, 'display': display(d)})
    t1 = pd.DataFrame(t1)
    t1.to_csv(O('table1_agri_main.csv'), index=False)

    # Agricultural paired tests: the authors' original 15-comparison family
    rows = []
    for comp in ['LR', 'DL', 'Debiased-HITL']:
        for k in AGRI_METRICS:
            a, b = aligned(main30, 'model', 'FE-HITL', comp, k)
            rows.append({'metric': k, 'contrast': f'FE-HITL - {comp}', **paired(a, b)})
    fam = ('agri main, bias 0.30, seeds 42-71: FE-HITL - {LR, DL, Debiased-HITL} '
           'x {R2, RMSE, DI, EOD, AOD} (authors\' original family)')
    tests = bh(rows, fam)
    tests.to_csv(O('tests_agri_main.csv'), index=False)

    # Same-base decomposition (Reviewer #4 point 5) ------------------------
    full = abl[abl.variant == 'Full'].set_index('seed').sort_index()
    fe = main30[main30.model == 'FE-HITL'].set_index('seed').sort_index()
    same_fe = float(np.max(np.abs(full[AGRI_METRICS].values - fe[AGRI_METRICS].values)))
    sb = abl[abl.variant.isin(['Full', 'w/o HED'])][['seed', 'variant'] + AGRI_METRICS]
    dl = main30[main30.model == 'DL'][['seed'] + AGRI_METRICS].assign(variant='DL')
    sbd = pd.concat([sb, dl], ignore_index=True)
    rows = []
    for k in AGRI_METRICS:
        a, b = aligned(sbd, 'variant', 'Full', 'w/o HED', k)
        rows.append({'metric': k, 'contrast': 'FE-HITL - same-base uncorrected',
                     'component': 'correction effect', **paired(a, b)})
        a, b = aligned(sbd, 'variant', 'w/o HED', 'DL', k)
        rows.append({'metric': k, 'contrast': 'same-base uncorrected - DL',
                     'component': 'base-model/pipeline difference', **paired(a, b)})
    bh(rows, 'same-base decomposition: 2 contrasts x 5 metrics').to_csv(
        O('table_same_base_agri.csv'), index=False)

    # Table 3: R2 ablation, relabelled, seed-matched -------------------------
    t3, rows = [], []
    for v, label in ABLATION_LABELS.items():
        for k in AGRI_METRICS:
            d = describe(abl.loc[abl.variant == v, k], args.ci)
            t3.append({'variant': label, 'metric': k, **d, 'display': display(d)})
            if v != 'Full':
                a, b = aligned(abl, 'variant', 'Full', v, k)
                rows.append({'metric': k, 'contrast': f'Full - {label}', **paired(a, b)})
    pd.DataFrame(t3).to_csv(O('table3_ablation_R2.csv'), index=False)
    bh(rows, 'R2 ablation: Full - {no correction, reduced strength} x 5 metrics; '
             'w/o F&U identical by construction (excluded)').to_csv(
        O('tests_ablation_R2.csv'), index=False)

    # Table 3b: genuine MOG ablation from run_r3_additions.py ----------------
    if mog is not None:
        ref = 'Full: Algorithm 1 (4 candidates + rule-based arbitration)'
        t3b, rows = [], []
        for v in mog['variant'].unique():
            for k in AGRI_METRICS:
                d = describe(mog.loc[mog.variant == v, k], args.ci)
                t3b.append({'variant': v, 'metric': k, **d, 'display': display(d)})
                if v != ref:
                    a, b = aligned(mog, 'variant', ref, v, k)
                    rows.append({'metric': k, 'contrast': f'Full(Alg.1) - {v}',
                                 **paired(a, b)})
        pd.DataFrame(t3b).to_csv(O('table3b_mog_ablation_R3.csv'), index=False)
        bh(rows, 'R3 MOG ablation: Full(Algorithm 1) - each variant x 5 metrics; '
                 'identical-by-construction contrasts excluded').to_csv(
            O('tests_mog_ablation_R3.csv'), index=False)
        sel = mog[mog.variant == ref]['selected_alpha'].value_counts(dropna=False)
        sel.rename('n_seeds').to_csv(O('mog_selected_alpha_counts.csv'))

    # Table 4: German Credit (descriptive only; inference in inference_fixed.py)
    t4 = []
    for m in MODELS:
        for k in ['Acc', 'DI', 'DI_gap', 'EOD', 'AOD']:
            d = describe(credit.loc[credit.model == m, k], args.ci)
            t4.append({'model': m, 'metric': k, **d, 'display': display(d)})
    pd.DataFrame(t4).to_csv(O('table4_credit_main.csv'), index=False)

    # Supporting tables with their seed sets ---------------------------------
    s2, rows = [], []
    for lvl in sorted(bias['bias'].unique()):
        sub = bias[np.isclose(bias['bias'], lvl)]
        for m in MODELS:
            for k in AGRI_METRICS:
                d = describe(sub.loc[sub.model == m, k], args.ci)
                s2.append({'bias': lvl, 'model': m, 'metric': k, **d,
                           'seeds': '42-71', 'display': display(d)})
        for k in ['DI', 'R2']:
            a, b = aligned(sub, 'model', 'FE-HITL', 'DL', k)
            rows.append({'bias': lvl, 'metric': k, 'contrast': 'FE-HITL - DL', **paired(a, b)})
    pd.DataFrame(s2).to_csv(O('tableS2_bias_sensitivity.csv'), index=False)
    bh(rows, 'bias levels: FE-HITL - DL x {DI, R2} x 4 levels').to_csv(
        O('tests_bias_levels.csv'), index=False)

    sup = []
    for key, df, grp, cols in [
            ('S3 bias structure', struct, ['bias_type', 'model'], ['DI', 'R2', 'EOD']),
            ('S6 binarization', binar, ['percentile', 'model'], ['DI', 'R2']),
            ('S7 reverse discrimination', rev, ['interv_frac'], ['DI_unpriv', 'DI_priv', 'R2']),
            ('Table 2 agri 5-fold CV', cv, ['model'], ['DI', 'R2'])]:
        for g, sub in df.groupby(grp):
            g = g if isinstance(g, tuple) else (g,)
            for k in cols:
                d = describe(sub[k], args.ci)
                sup.append({'table': key, 'group': ' | '.join(map(str, g)),
                            'metric': k, **d, 'display': display(d),
                            'seeds': ('dataset seed 42; 5 folds x training seeds 42-51 '
                                      '(fold x seed evaluations are NOT independent)'
                                      if key.startswith('Table 2') else '42-56 (15 seeds)')})
    pd.DataFrame(sup).to_csv(O('tables_supporting.csv'), index=False)

    # Reviewer #4 number check -------------------------------------------------
    def tv(metric, contrast, col):
        r = tests[(tests.metric == metric) & (tests.contrast == contrast)]
        return float(r[col].iloc[0])
    d1 = t1.set_index(['model', 'metric'])['mean']
    aod = describe(credit.loc[credit.model == 'FE-HITL', 'AOD'], 'normal')
    sw = sweep[sweep.model == 'FE-HITL'].groupby('ratio')[['DI', 'R2']].mean()
    rv = rev.groupby('interv_frac')['DI_unpriv'].mean()
    gap = credit.groupby('model')['DI_gap'].mean()
    checks = [
        ('1', 'DL DI, Table 1', 0.631978, d1[('DL', 'DI')], 5e-7),
        ('1', 'FE-HITL DI, Table 1', 0.863971, d1[('FE-HITL', 'DI')], 5e-7),
        ('1', 'DI improvement FE-HITL - DL (unrounded)', 0.231992, tv('DI', 'FE-HITL - DL', 'mean_diff'), 5e-7),
        ('1', 'paired t(29), DI', 18.1614, tv('DI', 'FE-HITL - DL', 't'), 5e-5),
        ('1', 'paired dz, DI', 3.3158, tv('DI', 'FE-HITL - DL', 'dz'), 5e-5),
        ('1', 'R2 difference FE-HITL - DL', -0.033567, tv('R2', 'FE-HITL - DL', 'mean_diff'), 5e-7),
        ('1', 'R2 p_BH, authors 15-comparison family', 1.20e-8, tv('R2', 'FE-HITL - DL', 'p_bh'), 5e-11),
        ('1', 'Table 4 FE-HITL |AOD| CI lower (normal)', 0.0216, aod['ci_low'], 5e-5),
        ('1', 'Table 4 FE-HITL |AOD| CI upper (normal)', 0.0413, aod['ci_high'], 5e-5),
        ('6', 'sweep DI at 10% (30 seeds)', 0.661395, sw.loc[0.10, 'DI'], 5e-7),
        ('6', 'sweep DI at 100% (30 seeds)', 0.863971, sw.loc[1.0, 'DI'], 5e-7),
        ('6', 'sweep R2 at 10% (30 seeds)', 0.721959, sw.loc[0.10, 'R2'], 5e-7),
        ('6', 'sweep R2 at 100% (30 seeds)', 0.702155, sw.loc[1.0, 'R2'], 5e-7),
        ('6', '15-seed reverse-discr. DI at 10%', 0.668, rv.loc[0.10], 5e-4),
        ('6', '15-seed reverse-discr. DI at 100%', 0.865, rv.loc[1.0], 5e-4),
        ('2', 'credit main mean |DI-1|, FE-HITL', 0.187794, gap['FE-HITL'], 5e-7),
        ('2', 'credit main mean |DI-1|, DL', 0.542591, gap['DL'], 5e-7),
    ]
    chk = pd.DataFrame([{'reviewer_point': p, 'quantity': q, 'reviewer_value': rv_,
                         'regenerated_value': float(v), 'abs_diff': abs(float(v) - rv_),
                         'tolerance': tol, 'match': abs(float(v) - rv_) <= tol}
                        for p, q, rv_, v, tol in checks])
    chk.to_csv(O('reviewer_number_check.csv'), index=False)

    # Manuscript values that must change -------------------------------------
    di = tests[(tests.metric == 'DI') & (tests.contrast == 'FE-HITL - DL')].iloc[0]
    r2r = tests[(tests.metric == 'R2') & (tests.contrast == 'FE-HITL - DL')].iloc[0]
    corr = [
        ('Results below Table 1; Fig 2 caption', 'DI paired difference', '+0.210', f"{di.mean_diff:+.3f}", 'tests_agri_main.csv: mean_diff'),
        ('Results below Table 1', 'DI difference 95% CI', '[+0.186, +0.233]', f"[{di.ci_low:+.3f}, {di.ci_high:+.3f}]", 'tests_agri_main.csv: ci_low/ci_high (t-based)'),
        ('Results below Table 1; Fig 2 caption', 'DI t statistic', 't(29) = -17.54', f"t(29) = {di.t:+.2f}", 'tests_agri_main.csv: t'),
        ('Results below Table 1; Fig 2 caption', 'DI dz', '3.20', f"{di.dz:.2f}", 'tests_agri_main.csv: dz'),
        ('Results below Table 1; Fig 2 caption', 'R2 BH-corrected p', '0.002', f"{r2r.p_bh:.1e} (report p < 0.001)", 'tests_agri_main.csv: p_bh'),
        ('Table 4', 'FE-HITL |AOD| 95% CI', '[0.020, 0.050]', f"[{aod['ci_low']:.3f}, {aod['ci_high']:.3f}]", 'table4_credit_main.csv'),
        ('Table 3', 'CI method', 't-based (e.g. Full DI [0.849, 0.879])', f'{args.ci} (same as Table 1)', 'table3_ablation_R2.csv'),
        ('Table 3 / Fig 3', '"w/o MOG" row', 'multi-option ablation', 'reduced correction strength (coef 0.2); genuine MOG ablation in Table 3b', 'table3b_mog_ablation_R3.csv'),
        ('Ablation text', 'one-way ANOVA F(3,116)=175.8 + "Tukey HSD with BH"', 'treats seeds as independent; Tukey not in code', 'seed-matched paired contrasts', 'tests_ablation_R2.csv'),
        ('Limitations', '10% -> 100% trade-off', '0.668/0.724 -> 0.865/0.703 (15 seeds, unlabelled)', f"30-seed sweep {sw.loc[0.10,'DI']:.3f}/{sw.loc[0.10,'R2']:.3f} -> {sw.loc[1.0,'DI']:.3f}/{sw.loc[1.0,'R2']:.3f}", 'interv_sensitivity_raw.csv'),
        ('Abstract; Results; Discussion; Conclusion', 'credit CV t(249), p < 0.001, dz = -0.60', 'pseudo-replicated', 'withdraw; see credit_inference_*.csv', 'inference_fixed.py'),
        ('Results; Fig 4 caption', 'credit main t(49) tests on signed DI', 'ignores split overlap; signed DI', 'see credit_inference_B2_generalization_main.csv', 'inference_fixed.py'),
        ('Intervention cost; Methods', '10% setting, ~9 min', 'different row from headline DI', 'report same-row fairness+workload', 'fairness_cost_joint.py'),
    ]
    pd.DataFrame(corr, columns=['location', 'quantity', 'R2_manuscript', 'regenerated',
                                'source']).to_csv(O('manuscript_value_corrections.csv'), index=False)

    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__},
        'conventions': {
            'contrast_direction': 'treatment - comparator (FE-HITL - X; Full - variant)',
            'pairing': 'seed; unmatched seeds raise an error',
            'descriptive_ci': ('mean +/- 1.96 SD/sqrt(n)' if args.ci == 'normal'
                               else 'mean +/- t_{n-1,0.975} SD/sqrt(n)'),
            'contrast_ci': 't_{n-1}, consistent with the paired t-test',
            'effect_size': "Cohen's dz", 'multiple_testing': 'BH within declared families'},
        'seed_sets': {'main_30': '42-71', 'supporting_15': '42-56',
                      'agri_cv': 'dataset seed 42; StratifiedKFold(5, random_state=42); training seeds 42-51',
                      'credit': '42-91 (50)'},
        'checks': {'FE-HITL(bias file) vs Full(ablation file) max abs diff': same_fe,
                   'reviewer numbers matched': f"{int(chk.match.sum())}/{len(chk)}"},
        'inputs': inputs,
    }
    with open(O('MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    pd.set_option('display.width', 200)
    print(tests[['metric', 'contrast', 'mean_diff', 't', 'dz', 'p_bh']].to_string(index=False))
    print('\nReviewer #4 number check:')
    print(chk[['reviewer_point', 'quantity', 'reviewer_value', 'regenerated_value',
               'match']].to_string(index=False))
    print(f"\n{int(chk.match.sum())}/{len(chk)} reviewer numbers reproduced. "
          f"Tables in {args.out_dir}/")


if __name__ == '__main__':
    main()
