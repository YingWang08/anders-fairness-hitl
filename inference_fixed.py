"""
inference_fixed.py
==================
German Credit inference for Reviewer #4 point 2, plus the agricultural 5-fold CV.

    python inference_fixed.py            # run from the repository root

Why the earlier R3 draft of this script was replaced
    * Its recommended test, a paired t(4) on fold means, removes the shared test
      observations but NOT the overlapping training data that Reviewer #4 named
      explicitly (5-fold training sets overlap by 75 %, not 80 %).
    * Its "mixed-effects cross-check" evaluated the same statistic against a
      normal distribution with only 5 clusters (credit |DI-1|: z = -3.36,
      p = 0.0008, whereas the identical statistic on t(4) gives p = 0.028).
    * It never read credit_main_raw.csv, the source of the reviewer's
      mean |DI-1| values, and never tested the 50-split main experiment.

What is reported, each labelled with the question it answers
    (A)  CONDITIONAL on this dataset and this 5-fold partition: fold = fixed
         effect; the 50 training seeds within a fold replicate. Answers "is the
         difference larger than training-seed noise on THIS partition?"
    (B1) GENERALIZATION, CV: seeds averaged within fold (k = 5 differences),
         corrected resampled t with variance factor (1/k + n_test/n_train).
    (B2) GENERALIZATION, main experiment: 50 random 70/30 splits of the same
         1,000 rows, corrected resampled t with factor (1/J + n_test/n_train).
         References: Nadeau & Bengio (2003) Mach Learn 52:239-281;
                     Bouckaert & Frank (2004) PAKDD, LNCS 3056:3-12.
    Naive statistics are printed beside the corrected ones for transparency.
    Primary fairness endpoint: |DI - 1| (the stated objective). Signed DI is
    descriptive only. (B1)/(B2) concern new samples from the population this
    dataset represents; other populations or domains need other datasets.
    Credit DI is computed on the model's positive class, which in the OpenML
    encoding is "bad credit" (y = 1); DI > 1 means women are predicted "bad"
    more often.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

CONTRASTS = [('FE-HITL', 'DL'), ('FE-HITL', 'Debiased-HITL'), ('Debiased-HITL', 'DL')]
ENDPOINTS = ['DI_gap', 'Acc', 'EOD', 'AOD']
TRIGGER = (0.80, 1.25)
GUARD = (0.75, 1.0 / 0.75)


def bh(rows, family):
    out = pd.DataFrame(rows)
    ok = out['p_raw'].notna()
    out['p_bh'] = np.nan
    if ok.any():
        out.loc[ok, 'p_bh'] = multipletests(out.loc[ok, 'p_raw'], method='fdr_bh')[1]
    out['bh_family'] = family
    out['bh_family_size'] = int(ok.sum())
    return out


def corrected_resampled_t(d, n_train, n_test):
    d = np.asarray(d, float)
    J, m, s2 = len(d), float(np.mean(d)), float(np.var(d, ddof=1))
    df = J - 1
    se_n, se_c = np.sqrt(s2 / J), np.sqrt((1.0 / J + n_test / n_train) * s2)
    q = stats.t.ppf(0.975, df)
    t_n = m / se_n if se_n > 0 else np.nan
    t_c = m / se_c if se_c > 0 else np.nan
    return {'n_units': J, 'mean_diff': m, 'se_corrected': se_c,
            'ci_low': m - q * se_c, 'ci_high': m + q * se_c,
            't_corrected': t_c, 'df': df,
            'p_raw': 2 * stats.t.sf(abs(t_c), df) if np.isfinite(t_c) else np.nan,
            't_naive': t_n,
            'p_naive': 2 * stats.t.sf(abs(t_n), df) if np.isfinite(t_n) else np.nan,
            'units_with_negative_diff': int(np.sum(d < 0)),
            'variance_factor': f'1/{J} + {n_test:g}/{n_train:g}'}


def conditional_fixed_fold(d, fold):
    d, fold = np.asarray(d, float), np.asarray(fold)
    dev = d - pd.Series(d).groupby(fold).transform('mean').values
    N, F = len(d), len(np.unique(fold))
    m, se = float(np.mean(d)), np.sqrt(np.sum(dev ** 2) / (N - F) / N)
    df, q = N - F, stats.t.ppf(0.975, N - F)
    t = m / se if se > 0 else np.nan
    return {'n_obs': N, 'n_folds': F, 'mean_diff': m, 'se': se,
            'ci_low': m - q * se, 'ci_high': m + q * se, 't': t, 'df': df,
            'p_raw': 2 * stats.t.sf(abs(t), df) if np.isfinite(t) else np.nan}


def wide(df, index, metric):
    return df.pivot_table(index=index, columns='model', values=metric,
                          aggfunc='first').reset_index()


def guard_accounting(df, keys):
    w = df.pivot_table(index=keys, columns='model',
                       values=['Acc', 'DI', 'EOD', 'AOD'], aggfunc='first')
    base, fe = w[('DI', 'DL')], w[('DI', 'FE-HITL')]
    trig = ~base.between(*TRIGGER)
    same = np.ones(len(w), bool)
    for m in ['Acc', 'DI', 'EOD', 'AOD']:
        same &= (w[(m, 'FE-HITL')] == w[(m, 'DL')]).values
    return pd.DataFrame({
        'base_DI_same_model': base.values, 'FE_HITL_final_DI': fe.values,
        'triggered': trig.values, 'fallback_unchanged': trig.values & same,
        'final_outside_guard_0.75_1.333': (~fe.between(*GUARD)).values,
        'final_outside_trigger_0.80_1.25': (~fe.between(*TRIGGER)).values,
        'base_DI_is_zero': (base == 0).values}, index=w.index).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--credit-dir', default='experiments/credit_revision_results')
    ap.add_argument('--agri-cv-file', default='experiments/revision_results/kfold_cv_agri_raw.csv')
    ap.add_argument('--out-dir', default='r3_tables')
    ap.add_argument('--credit-n', type=int, default=1000)
    ap.add_argument('--main-test-frac', type=float, default=0.30)
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    O = lambda n: os.path.join(args.out_dir, n)

    main_df = pd.read_csv(os.path.join(args.credit_dir, 'credit_main_raw.csv'))
    cv = pd.read_csv(os.path.join(args.credit_dir, 'credit_cv_raw.csv'))
    for df in (main_df, cv):
        df['DI_gap'] = (df['DI'] - 1.0).abs()

    desc = []
    for name, df in [('main: 50 random 70/30 splits', main_df),
                     ('CV: one 5-fold partition x 50 seeds', cv)]:
        for m in sorted(df.model.unique()):
            for e in ['DI', 'DI_gap', 'Acc', 'EOD', 'AOD']:
                x = df.loc[df.model == m, e].values
                desc.append({'experiment': name, 'model': m, 'endpoint': e,
                             'n_rows': len(x), 'mean': x.mean(),
                             'sd': x.std(ddof=1), 'median': np.median(x)})
    pd.DataFrame(desc).to_csv(O('credit_descriptive.csv'), index=False)

    rows = []
    for e in ENDPOINTS:
        w = wide(cv, ['fold', 'seed'], e)
        for a, b in CONTRASTS:
            rows.append({'endpoint': e, 'contrast': f'{a} - {b}',
                         **conditional_fixed_fold(w[a] - w[b], w['fold'])})
    A = bh(rows, '(A) credit CV, conditional on partition: 3 contrasts x 4 endpoints')
    A.to_csv(O('credit_inference_A_conditional_cv.csv'), index=False)

    n_te_cv = args.credit_n / args.k
    rows = []
    for e in ENDPOINTS:
        fm = cv.groupby(['fold', 'model'])[e].mean().unstack('model')
        for a, b in CONTRASTS:
            rows.append({'endpoint': e, 'contrast': f'{a} - {b}',
                         **corrected_resampled_t((fm[a] - fm[b]).values,
                                                 args.credit_n - n_te_cv, n_te_cv)})
    B1 = bh(rows, '(B1) credit CV generalization, corrected resampled t(4): 3 x 4')
    B1.to_csv(O('credit_inference_B1_generalization_cv.csv'), index=False)

    n_te = round(args.credit_n * args.main_test_frac)
    rows = []
    for e in ENDPOINTS:
        w = wide(main_df, ['seed'], e)
        for a, b in CONTRASTS:
            rows.append({'endpoint': e, 'contrast': f'{a} - {b}',
                         **corrected_resampled_t((w[a] - w[b]).values,
                                                 args.credit_n - n_te, n_te)})
    B2 = bh(rows, '(B2) credit main generalization, corrected resampled t(49): 3 x 4')
    B2.to_csv(O('credit_inference_B2_generalization_main.csv'), index=False)

    gm, gc = guard_accounting(main_df, ['seed']), guard_accounting(cv, ['fold', 'seed'])
    gm.to_csv(O('credit_guard_accounting_main_per_seed.csv'), index=False)
    summ = []
    for name, g in [('main (50 seeds)', gm), ('CV (250 fold x seed)', gc)]:
        summ.append({'experiment': name, 'n': len(g),
                     'not_triggered': int((~g.triggered).sum()),
                     'triggered': int(g.triggered.sum()),
                     'fallback_unchanged': int(g.fallback_unchanged.sum()),
                     'final_DI_outside_guard': int(g['final_outside_guard_0.75_1.333'].sum()),
                     'final_DI_outside_trigger_band': int(g['final_outside_trigger_0.80_1.25'].sum()),
                     'base_DI_zero': int(g.base_DI_is_zero.sum())})
    pd.DataFrame(summ).to_csv(O('credit_guard_accounting_summary.csv'), index=False)

    pd.DataFrame([
        {'task': 'credit', 'interval': 'trigger [0.80, 1.25]', 'applies_to': 'DI of the base predictions',
         'role': 'decides whether a correction is attempted', 'guarantee': 'none'},
        {'task': 'credit', 'interval': 'candidate guard [0.75, 1.333]', 'applies_to': 'DI a candidate would produce',
         'role': 'admissibility of candidates; wider than the trigger so a candidate landing just outside the trigger band is not discarded',
         'guarantee': 'none on outputs: if no candidate is admissible the unchanged predictions are returned'},
        {'task': 'agriculture', 'interval': 'trigger DI < 0.80', 'applies_to': 'DI at the monitoring threshold',
         'role': 'decides whether Algorithm 1 runs', 'guarantee': 'none'},
        {'task': 'agriculture', 'interval': 'target DI >= 0.85; DI <= 1.25; efficiency loss <= 10 %',
         'applies_to': 'each candidate', 'role': 'fairness-prioritisation, non-discrimination, efficiency rules',
         'guarantee': 'none on outputs: fallback takes the largest admissible DI, which may be < 0.85 or < 0.80'},
    ]).to_csv(O('interval_reconciliation.csv'), index=False)

    if os.path.exists(args.agri_cv_file):
        ag = pd.read_csv(args.agri_cv_file)
        rows = []
        for e in ['DI', 'R2']:
            w = wide(ag, ['fold', 'seed'], e)
            fm = ag.groupby(['fold', 'model'])[e].mean().unstack('model')
            for a, b in [('FE-HITL', 'DL'), ('FE-HITL', 'Debiased-HITL')]:
                Ar = conditional_fixed_fold(w[a] - w[b], w['fold'])
                Br = corrected_resampled_t((fm[a] - fm[b]).values, 8000, 2000)
                rows.append({'endpoint': e, 'contrast': f'{a} - {b}',
                             'mean_diff': Ar['mean_diff'],
                             'A_conditional_t': Ar['t'], 'A_df': Ar['df'], 'A_p': Ar['p_raw'],
                             'B1_corrected_t': Br['t_corrected'], 'B1_df': Br['df'],
                             'B1_p': Br['p_raw'], 'B1_ci_low': Br['ci_low'],
                             'B1_ci_high': Br['ci_high']})
        pd.DataFrame(rows).to_csv(O('agri_cv_inference.csv'), index=False)

    pd.set_option('display.width', 220)
    cols = ['endpoint', 'contrast', 'mean_diff', 'ci_low', 'ci_high', 't_corrected',
            'df', 'p_bh', 't_naive', 'p_naive']
    print('(B2) MAIN, generalization (corrected resampled t):')
    print(B2[cols].round(4).to_string(index=False))
    print('\n(B1) CV, generalization (corrected resampled t on fold means):')
    print(B1[cols + ['units_with_negative_diff']].round(4).to_string(index=False))
    print('\n(A) CV, conditional on this partition:')
    print(A[['endpoint', 'contrast', 'mean_diff', 't', 'df', 'p_bh']].to_string(index=False))
    print('\nGuard / fallback accounting:')
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == '__main__':
    main()
