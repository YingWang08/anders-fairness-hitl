"""
run_r3_additions.py
===================
R3 analyses for Reviewer #4 points 3-6. The R2 experiment script
(run_sensitivity_standalone.py) is NOT modified, so every deposited R2 CSV,
which the reviewer reproduced to < 1e-14, remains the result of record.

    python experiments/run_r3_additions.py            # full run, ~5 min on a laptop
    python experiments/run_r3_additions.py --quick    # 3-seed smoke test

Outputs -> experiments/revision_results/r3/
  r3_mog_verification.csv    one row per (configuration, seed): the four-
                             candidate Algorithm 1 with the stated decision rules
                             vs the executed R2 routine (bitwise identity flag,
                             selected alpha, rule that fired, reproduction check
                             against the deposited CSVs where available)
  r3_mog_candidates.csv      every candidate evaluated, with its admissibility
  r3_mog_ablation_raw.csv    Full vs each single fixed candidate vs R2 reduced
                             strength vs no correction, one shared base per seed
  r3_fu_diagnostics.csv      Feedback & Update order-of-evaluation evidence
  r3_threshold_alignment.csv FE-HITL monitored at median(y_train) as well
  r3_case_counts.csv         per-seed denominators for the workload table
  r3_run_info.json           versions, seed sets, runtime
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.fe_hitl_r3 import ALPHAS, FEHITL_R3  # noqa: E402

FRACS = [0.10, 0.25, 0.50, 0.75, 1.0]


def load_r2_module():
    """Import run_sensitivity_standalone.py without executing main()."""
    spec = importlib.util.spec_from_file_location(
        'r2_agri', os.path.join(HERE, 'run_sensitivity_standalone.py'))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(HERE)   # its module-level makedirs('revision_results') stays in experiments/
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def context(r2, train, val, test, seed, cv=False):
    """Arrays for one (configuration, seed) plus ONE fitted FE-HITL base model."""
    Xt, yt, st = r2.split_arrays(train)
    Xe, ye, se = r2.split_arrays(test)
    if cv:   # R2 convention in run_kfold_cv(): training data doubles as validation
        Xv, yv, sv = Xt, yt, st
    else:
        Xv, yv, sv = r2.split_arrays(val)
    fh = FEHITL_R3(random_seed=seed).fit_base(Xt, yt, Xv, yv, s_train=st, s_val=sv)
    return dict(Xt=Xt, yt=yt, st=st, Xv=Xv, yv=yv, sv=sv, Xe=Xe, ye=ye, se=se,
                thr=np.median(yt), model=fh.base_model, seed=seed)


def variant(ctx, **kw):
    kw.setdefault('use_fu', False)
    v = FEHITL_R3(random_seed=ctx['seed'], **kw)
    return v.attach_base(ctx['model'], ctx['Xt'], ctx['yt'], ctx['st'],
                         ctx['Xv'], ctx['yv'], ctx['sv'])


def r2_routine(r2, ctx, frac):
    """The executed R2 routine on the SAME fitted base model."""
    o = r2.FEHITL_Ablatable(input_dim=7, device='cpu', random_seed=ctx['seed'],
                            use_hed=True, use_mog=True, use_fu=False,
                            interv_frac=frac)
    o.base_model = ctx['model']
    return o.predict_with_intervention(ctx['Xe'], ctx['se'])


def metrics(r2, ctx, p):
    return r2.compute_metrics(ctx['ye'], p, ctx['se'], ctx['thr'])


def verify(r2, ctx, config, used_by, frac, deposited_di, rows, cand_rows):
    p_r2 = r2_routine(r2, ctx, frac)
    v = variant(ctx, interv_frac=frac)                  # Algorithm 1, Full
    p_r3 = v.predict_with_intervention(ctx['Xe'], ctx['se'])
    d = v.last_decision
    m = metrics(r2, ctx, p_r3)
    m_r2 = metrics(r2, ctx, p_r2)
    rows.append({
        'config': config, 'used_by': used_by, 'seed': ctx['seed'],
        'interv_frac': frac,
        'bitwise_identical_to_R2': bool(np.array_equal(p_r2, p_r3)),
        'max_abs_diff_vs_R2': float(np.max(np.abs(p_r2 - p_r3))),
        'triggered': d['triggered'],
        'di_before_monitor': d.get('di_before_monitor'),
        'selected_alpha': d.get('selected_alpha'),
        'selection_rule': d.get('selection_rule'),
        'n_qualifying': d.get('n_qualifying'), 'n_admissible': d.get('n_admissible'),
        'n_efficiency_rejections': d.get('n_efficiency_rejections'),
        'n_reverse_rejections': d.get('n_reverse_rejections'),
        'di_after_monitor': d.get('di_after_monitor'),
        'n_routed': d.get('n_routed'), 'n_unpriv': d['n_unpriv'],
        **{k: m[k] for k in ('R2', 'RMSE', 'DI', 'EOD', 'AOD')},
        'deposited_DI': deposited_di,
        'abs_diff_R2routine_vs_deposited_DI': (
            None if deposited_di is None else abs(m_r2['DI'] - deposited_di)),
    })
    for c in (v.last_candidate_table or []):
        cand_rows.append({'config': config, 'seed': ctx['seed'],
                          'interv_frac': frac, **c})
    return d


def lookup(df, **kw):
    if df is None:
        return None
    sub = df
    for k, val in kw.items():
        sub = sub[np.isclose(sub[k], val)] if isinstance(val, float) else sub[sub[k] == val]
    return None if len(sub) != 1 else float(sub['DI'].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'revision_results', 'r3'))
    ap.add_argument('--deposited-dir', default=os.path.join(HERE, 'revision_results'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    r2 = load_r2_module()

    S30, S15, SCV = list(range(42, 72)), list(range(42, 57)), list(range(42, 52))
    BIAS = [0.10, 0.20, 0.30, 0.40]
    if args.quick:
        S30, S15, SCV, BIAS = [42, 43, 44], [42, 43], [42], [0.30]

    def read(name):
        p = os.path.join(args.deposited_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    dep_bias = read('bias_sensitivity_raw.csv')
    dep_sweep = read('interv_sensitivity_raw.csv')
    if dep_bias is not None:
        dep_bias = dep_bias[dep_bias['model'] == 'FE-HITL']
    if dep_sweep is not None:
        dep_sweep = dep_sweep[dep_sweep['model'] == 'FE-HITL']

    ver, cand, abl, fu, thr_rows, counts = [], [], [], [], [], []

    # ── Uniform generator: bias levels; at 0.30 also fractions, ablation, F&U
    for bias in BIAS:
        for seed in S30:
            r2.set_seed(seed)
            ctx = context(r2, *r2.generate_agricultural_data_biased(10000, seed, bias), seed)
            at30 = np.isclose(bias, 0.30)
            for frac in (FRACS if at30 else [1.0]):
                if frac == 0.75 and seed not in S15:
                    continue       # 0.75 exists only in the 15-seed reverse-discrimination suite
                if at30 and frac == 1.0:
                    used = 'bias sens.; interv. sweep; ablation; reverse discr.; binarization'
                elif at30 and frac == 0.75:
                    used = 'reverse discrimination (15 seeds)'
                elif at30:
                    used = 'interv. sweep (30 seeds); reverse discr. (seeds 42-56)'
                else:
                    used = 'bias sensitivity'
                dep = (lookup(dep_bias, bias=float(bias), seed=seed) if frac == 1.0
                       else lookup(dep_sweep, ratio=float(frac), seed=seed))
                if frac == 1.0 and at30 and dep is None:
                    dep = lookup(dep_sweep, ratio=1.0, seed=seed)
                d = verify(r2, ctx, f'uniform bias={bias:.2f} frac={frac:.2f}',
                           used, frac, dep, ver, cand)
                if at30 and frac == 1.0:
                    di_before = d.get('di_before_monitor')
            if not at30:
                continue

            # (a) genuine MOG ablation on ONE shared base model
            specs = [('Full: Algorithm 1 (4 candidates + rule-based arbitration)',
                      dict(mog_mode='algorithm1'))]
            specs += [(f'Single fixed candidate, alpha={a}',
                       dict(mog_mode='single', single_alpha=a)) for a in ALPHAS]
            specs += [('Reduced strength, coef 0.2 (R2 label "w/o MOG")',
                       dict(mog_mode='r2_reduced_strength')),
                      ('No correction, same base (R2 label "w/o HED")',
                       dict(use_hed=False))]
            for label, kw in specs:
                v = variant(ctx, **kw)
                p = v.predict_with_intervention(ctx['Xe'], ctx['se'])
                abl.append({'seed': seed, 'variant': label,
                            'selected_alpha': v.last_decision.get('selected_alpha'),
                            'selected_boost': v.last_decision.get('selected_boost'),
                            **metrics(r2, ctx, p)})

            # (b) Feedback & Update: order-of-evaluation evidence
            v_fu = variant(ctx, use_fu=True)
            p_ret = v_fu.predict_with_intervention(ctx['Xe'], ctx['se'])
            p_pre = variant(ctx).predict_with_intervention(ctx['Xe'], ctx['se'])
            fd = v_fu.fu_diagnostics
            if fd is None:          # trigger did not fire, so F&U never ran
                fu.append({'seed': seed, 'fu_ran': False})
            else:
                post = fd.pop('post_update_predictions')
                o = r2.FEHITL_Ablatable(input_dim=7, device='cpu', random_seed=seed,
                                        use_hed=True, use_mog=True, use_fu=True)
                o.base_model, o._X_train, o._s_train = ctx['model'], ctx['Xt'], ctx['st']
                p_r2_fu = o.predict_with_intervention(ctx['Xe'], ctx['se'])
                m_ret, m_post = metrics(r2, ctx, p_ret), metrics(r2, ctx, post)
                fu.append({
                    'seed': seed, 'fu_ran': True, **fd,
                    'returned_equals_pre_update_exact': bool(np.array_equal(p_ret, p_pre)),
                    'returned_equals_R2_class_with_FU': bool(np.array_equal(p_ret, p_r2_fu)),
                    'R2_class_model_replaced': o.base_model is not ctx['model'],
                    'n_test': int(len(p_ret)),
                    'n_cases_post_update_differs': int(np.sum(post != p_ret)),
                    'max_abs_diff_post_vs_returned': float(np.max(np.abs(post - p_ret))),
                    **{f'returned_{k}': m_ret[k] for k in ('R2', 'DI', 'EOD', 'AOD')},
                    **{f'HYPOTHETICAL_post_update_{k}': m_post[k]
                       for k in ('R2', 'DI', 'EOD', 'AOD')}})

            # (c) threshold alignment: monitor at median(y_train), as evaluated
            for label, kw in [('Algorithm 1', dict(mog_mode='algorithm1')),
                              ('Single alpha=1.0 (R2 closed form)',
                               dict(mog_mode='single', single_alpha=1.0))]:
                v = variant(ctx, **kw)
                v.set_binarize_threshold(ctx['thr'])
                p = v.predict_with_intervention(ctx['Xe'], ctx['se'])
                thr_rows.append({'seed': seed, 'variant': label,
                                 'monitor_threshold': 'median(y_train) [aligned]',
                                 'triggered': v.last_decision['triggered'],
                                 'di_before_monitor': v.last_decision.get('di_before_monitor'),
                                 'selected_alpha': v.last_decision.get('selected_alpha'),
                                 **metrics(r2, ctx, p)})

            # (d) denominators for the workload table
            n_unpriv = int(np.sum(ctx['se'] == 1))
            counts.append({'seed': seed, 'n_test_total': int(len(ctx['se'])),
                           'n_unpriv': n_unpriv,
                           **{f'n_routed_frac_{f:.2f}': int(n_unpriv * f) for f in FRACS},
                           'di_before_monitor': di_before,
                           'triggered': bool(di_before < 0.8)})
        print(f'  uniform bias={bias:.2f} done  ({time.time() - t0:.0f}s)', flush=True)

    # ── Bias-structure robustness (15 seeds)
    for btype in ['uniform', 'nonlinear', 'interaction']:
        for seed in S15:
            r2.set_seed(seed)
            ctx = context(r2, *r2.generate_agricultural_data_biased_v2(
                10000, seed, bias_rate=0.30, bias_type=btype), seed)
            verify(r2, ctx, f'structure={btype} bias=0.30 frac=1.00',
                   'bias-structure robustness', 1.0, None, ver, cand)
        print(f'  structure={btype} done  ({time.time() - t0:.0f}s)', flush=True)

    # ── Agricultural 5-fold CV (dataset seed 42, partition random_state=42)
    from sklearn.model_selection import StratifiedKFold
    full = r2.generate_full_dataset(10000, 42, 0.30)
    lab = (full['region'] == 'A').astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr_i, te_i) in enumerate(skf.split(full, lab)):
        for seed in SCV:
            r2.set_seed(seed)
            ctx = context(r2, full.iloc[tr_i], None, full.iloc[te_i], seed, cv=True)
            verify(r2, ctx, f'5-fold CV fold={k} frac=1.00', 'agricultural CV',
                   1.0, None, ver, cand)
    print(f'  CV done  ({time.time() - t0:.0f}s)', flush=True)

    out = {'r3_mog_verification.csv': ver, 'r3_mog_candidates.csv': cand,
           'r3_mog_ablation_raw.csv': abl, 'r3_fu_diagnostics.csv': fu,
           'r3_threshold_alignment.csv': thr_rows, 'r3_case_counts.csv': counts}
    for name, rows in out.items():
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, name), index=False)

    import scipy
    import sklearn
    info = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__, 'quick': args.quick,
            'seeds_30': S30, 'seeds_15': S15, 'cv_training_seeds': SCV,
            'runtime_s': round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, 'r3_run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    V = pd.DataFrame(ver)
    print('\n=== Algorithm 1 (as specified) vs executed R2 routine ===')
    print(f"runs: {len(V)} | bitwise identical: {int(V.bitwise_identical_to_R2.sum())} "
          f"| triggered: {int(V.triggered.sum())}")
    print('selected alpha counts:', V.selected_alpha.value_counts(dropna=False).to_dict())
    print('selection rules:', V.selection_rule.value_counts(dropna=False).to_dict())
    print('efficiency rejections:', int(V.n_efficiency_rejections.fillna(0).sum()),
          '| reverse rejections:', int(V.n_reverse_rejections.fillna(0).sum()))
    dd = V['abs_diff_R2routine_vs_deposited_DI'].dropna()
    print(f'reproduction vs deposited CSVs: {len(dd)} rows, max |diff| = {dd.max():.2e}')
    print(f"written to {args.out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
