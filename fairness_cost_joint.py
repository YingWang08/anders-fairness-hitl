"""
fairness_cost_joint.py
======================
Reviewer #4 point 6: fairness, predictive performance and estimated review
workload at the SAME intervention setting, on the SAME seeds.

    python fairness_cost_joint.py        # run from the repository root

Corrections relative to the earlier R3 draft
    * Default review time is the manuscript's own stated assumption, 3 s per
      option (not 30 s). The draft changed it tenfold without saying so.
    * The manuscript's "~9 minutes" is the 10 % setting (about 45 cases x
      4 options x 3 s); the draft's docstring attributed it to 100 %.
    * Denominators come from the SAME 30 seeds as the sweep
      (r3_case_counts.csv), not from the 15-seed cost run.
    * Gains are reported against DL AND against the same-base uncorrected model
      (R2 label "w/o HED"), which isolates the correction (point 5).

Denominator: interv_frac is a fraction of UNPRIVILEGED (region A) test cases;
about 30 % of the test set is region A, so interv_frac = 1.0 routes ~30 %.
Workload is ARITHMETIC FROM ASSUMPTIONS: nothing in the pipeline generates
options per case or times any review. Algorithm 1 generates its candidates
once per decision batch; "4 options reviewed per routed case" and "3 s per
option" are modelling assumptions and must be labelled as such.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n, m, sd = len(d), float(np.mean(d)), float(np.std(d, ddof=1))
    hw = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, m - hw, m + hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-file', default='experiments/revision_results/interv_sensitivity_raw.csv')
    ap.add_argument('--ablation-file', default='experiments/revision_results/ablation_raw.csv')
    ap.add_argument('--counts-file', default='experiments/revision_results/r3/r3_case_counts.csv')
    ap.add_argument('--reverse-file', default='experiments/revision_results/reverse_discrimination_raw.csv')
    ap.add_argument('--cost-file', default='experiments/revision_results/intervention_cost_raw.csv')
    ap.add_argument('--out-dir', default='r3_tables')
    ap.add_argument('--options-per-case', type=int, default=4)
    ap.add_argument('--seconds-per-option', type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    O = lambda n: os.path.join(args.out_dir, n)

    sweep, abl = pd.read_csv(args.sweep_file), pd.read_csv(args.ablation_file)
    base = abl[abl.variant == 'w/o HED'].set_index('seed').sort_index()
    if os.path.exists(args.counts_file):
        counts = pd.read_csv(args.counts_file).set_index('seed')
        count_src = 'r3_case_counts.csv (same seeds as the sweep)'
    else:
        counts = None
        cost = pd.read_csv(args.cost_file)
        count_src = 'WARNING: intervention_cost_raw.csv (15 seeds, 42-56), not the sweep seeds'

    rows = []
    for r in sorted(sweep['ratio'].unique()):
        fe = sweep[(sweep.model == 'FE-HITL') & np.isclose(sweep.ratio, r)].set_index('seed').sort_index()
        dl = sweep[(sweep.model == 'DL') & np.isclose(sweep.ratio, r)].set_index('seed').sort_index()
        if not (list(fe.index) == list(dl.index) == list(base.index)):
            raise ValueError('seed sets differ between sweep and ablation files')
        if counts is not None:
            c = counts.loc[fe.index]
            n_test, n_unpriv = c['n_test_total'].mean(), c['n_unpriv'].mean()
            n_routed = c[f'n_routed_frac_{r:.2f}'].mean()
        else:
            n_test, n_unpriv = cost['n_test_total'].mean(), cost['n_unpriv'].mean()
            n_routed = np.floor(n_unpriv * r)
        di, r2 = fe['DI'].values, fe['R2'].values
        hw = stats.norm.ppf(0.975) * di.std(ddof=1) / np.sqrt(len(di))
        g_dl, g_dl_lo, g_dl_hi = paired(di, dl['DI'].values)
        g_sb, g_sb_lo, g_sb_hi = paired(di, base['DI'].values)
        r_dl, _, _ = paired(r2, dl['R2'].values)
        r_sb, _, _ = paired(r2, base['R2'].values)
        minutes = n_routed * args.options_per_case * args.seconds_per_option / 60.0
        rows.append({
            'interv_frac_of_unprivileged': r, 'seeds': f'{fe.index.min()}-{fe.index.max()}',
            'n_seeds': len(di), 'n_test_total': n_test, 'n_unprivileged': n_unpriv,
            'n_routed_to_review': n_routed, 'share_of_all_test_cases': n_routed / n_test,
            'DI_mean': di.mean(), 'DI_ci_low': di.mean() - hw, 'DI_ci_high': di.mean() + hw,
            'share_of_seeds_DI_ge_0.8': float(np.mean(di >= 0.8)),
            'DI_gain_vs_DL': g_dl, 'DI_gain_vs_DL_ci': f'[{g_dl_lo:.3f}, {g_dl_hi:.3f}]',
            'DI_gain_vs_same_base': g_sb, 'DI_gain_vs_same_base_ci': f'[{g_sb_lo:.3f}, {g_sb_hi:.3f}]',
            'R2_mean': r2.mean(), 'R2_change_vs_DL': r_dl, 'R2_change_vs_same_base': r_sb,
            'ASSUMED_options_per_case': args.options_per_case,
            'ASSUMED_seconds_per_option': args.seconds_per_option,
            'review_minutes_per_batch': minutes, 'count_source': count_src})
    joint = pd.DataFrame(rows)
    joint.to_csv(O('table_fairness_cost_joint.csv'), index=False)

    sens = []
    for _, j in joint.iterrows():
        for opts in [1, 4]:
            for secs in [3, 10, 30, 60]:
                mins = j['n_routed_to_review'] * opts * secs / 60.0
                sens.append({'interv_frac_of_unprivileged': j['interv_frac_of_unprivileged'],
                             'DI_mean': j['DI_mean'], 'ASSUMED_options_per_case': opts,
                             'ASSUMED_seconds_per_option': secs,
                             'review_minutes_per_batch': mins, 'review_hours_per_batch': mins / 60})
    pd.DataFrame(sens).to_csv(O('workload_assumption_sensitivity.csv'), index=False)

    if os.path.exists(args.reverse_file):
        rev = pd.read_csv(args.reverse_file)
        t = rev.groupby('interv_frac')[['DI_unpriv', 'DI_priv', 'R2']].mean().reset_index()
        t['seeds'] = f"{rev.seed.min()}-{rev.seed.max()} ({rev.seed.nunique()} seeds)"
        if counts is not None:
            c15 = counts.loc[sorted(rev.seed.unique())]
            t['n_routed_to_review'] = [c15[f'n_routed_frac_{f:.2f}'].mean() for f in t.interv_frac]
            t['review_minutes_per_batch_ASSUMED'] = (t['n_routed_to_review'] * args.options_per_case
                                                     * args.seconds_per_option / 60.0)
        t.to_csv(O('table_reverse_discrimination_15seeds.csv'), index=False)

    pd.set_option('display.width', 220)
    print(joint[['interv_frac_of_unprivileged', 'n_routed_to_review', 'share_of_all_test_cases',
                 'DI_mean', 'share_of_seeds_DI_ge_0.8', 'DI_gain_vs_DL', 'DI_gain_vs_same_base',
                 'R2_mean', 'R2_change_vs_DL', 'review_minutes_per_batch']].round(3).to_string(index=False))
    lo, hi = joint.iloc[0], joint.iloc[-1]
    print(f"\nMethods setting ({lo.interv_frac_of_unprivileged:.0%}): DI {lo.DI_mean:.3f}, "
          f"~{lo.review_minutes_per_batch:.1f} min per batch.")
    print(f"Headline setting ({hi.interv_frac_of_unprivileged:.0%}): DI {hi.DI_mean:.3f}, "
          f"~{hi.review_minutes_per_batch:.1f} min per batch under the SAME assumptions "
          f"({args.options_per_case} options x {args.seconds_per_option:g} s).")
    if os.path.exists(args.reverse_file):
        print('\n15-seed reverse-discrimination suite (seeds labelled):')
        print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
