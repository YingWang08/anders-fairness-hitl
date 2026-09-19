"""
fe_hitl_r3.py
=============
FE-HITL (agricultural regression pathway) with Algorithm 1 implemented exactly
as it is written in the manuscript, plus an auditable decision record.

Why this file replaces the earlier R3 draft
-------------------------------------------
The earlier draft ported the German-Credit candidate set (factors 1.03-1.25,
arbitration by min |DI-1|, guard [0.75, 1.333]) into the agricultural pathway.
Executed on the real pipeline it (i) was NOT equivalent to R2 (seeds 42-51:
mean DI 0.869 -> 1.002, R^2 0.710 -> 0.663), (ii) fell back to "no correction"
at interv_frac = 0.10 in 9/10 seeds because the lower guard rejects every
partial improvement, and (iii) implemented an algorithm the manuscript does not
describe for this task. This file implements what the manuscript claims.

Algorithm 1, agricultural task (manuscript, lines 5-10)
    for alpha in {0.3, 0.5, 0.7, 1.0}:
        boost = 1 + alpha * (DI_LOW - di_current) * 0.7
        candidate = base predictions, routed unprivileged cases * boost
        record (candidate, efficiency loss, predicted DI)

Arbitration = the manuscript's "simulated human decision rules"
    Fairness-prioritisation : an option restoring DI >= 0.85 whose efficiency
                              loss is <= 10 %.
    Non-discrimination      : reject options that disadvantage the other group,
                              operationalised as DI <= 1/0.8 (the privileged
                              group keeps DI_priv >= 0.8).
    Practicality            : among qualifying options, the smallest deviation
                              from the original allocation.
    Fallback (stated explicitly here; described by the manuscript's
    implementation note): if no option reaches 0.85, take the admissible option
    with the largest DI (ties -> smaller deviation); if none is admissible,
    return the unchanged base predictions.

What is and is not enforced
    trigger DI < 0.80            enforced (decides whether the routine runs)
    efficiency loss <= epsilon   enforced; relative R^2 loss estimated on the
                                 VALIDATION split (never test labels)
    DI <= 1.25 (non-discrim.)    enforced on candidates
    DI >= 0.85 target            enforced as the selection criterion
    EOD threshold                NOT enforced: EOD needs ground-truth labels,
                                 which do not exist when a decision is issued.
    guarantee on final outputs   NONE; see `selection_rule` in last_decision.

Exact-reproduction conventions (kept deliberately and documented)
    * Routed subset: np.random.RandomState(42) regardless of the experiment
      seed, exactly as in R2 (`subset_seed=42`).
    * Monitoring threshold theta: median of the base model's predictions on the
      decision batch, set on the first call, exactly as in R2. The REPORTED
      metrics and the Debiased-HITL comparator binarise at median(y_train);
      set_binarize_threshold() aligns them (sensitivity analysis).
    * Feedback & Update keeps R2's order of evaluation: returned predictions
      are computed BEFORE the refit. The refit constructs a NEW MLPRegressor,
      so warm_start=True has no effect: it is a 20-iteration fit from a fresh
      initialisation on pseudo-labels, not fine-tuning. Both facts are logged.

For alpha = 1.0 the boost expression is bitwise the R2 expression
1.0 + 1.0 * (0.8 - di) * 0.7, so whenever arbitration selects alpha = 1.0 the
output equals R2's FEHITL_Ablatable(use_mog=True) exactly.
experiments/run_r3_additions.py checks this on every configuration.

mog_mode
    'algorithm1'          four candidates + rule-based arbitration (Full)
    'single'              ONE fixed candidate (single_alpha), no arbitration:
                          the genuine "without multi-option generation" control
    'r2_reduced_strength' what R2 labelled "w/o MOG": 1 + (0.8 - di) * 0.2
                          (alpha ~ 0.286). A correction-STRENGTH variant, kept
                          only to reproduce the deposited R2 row.
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

DI_LOW = 0.80                  # trigger (four-fifths rule)
DI_HIGH = 1.0 / DI_LOW         # 1.25, reverse-discrimination bound
ALPHAS = (0.3, 0.5, 0.7, 1.0)  # Algorithm 1, line 5
GAIN = 0.7                     # Algorithm 1, line 7
DI_TARGET = 0.85               # fairness-prioritisation rule
EPSILON = 0.10                 # efficiency-loss bound
R2_REDUCED_GAIN = 0.2          # R2 "w/o MOG" coefficient


def disparate_impact(y_bin, s_unpriv):
    """DI = P(yhat=1 | unprivileged) / P(yhat=1 | privileged); R2 convention."""
    unpriv = (s_unpriv == 1)
    p_unpriv = np.mean(y_bin[unpriv])
    p_priv = np.mean(y_bin[~unpriv])
    return p_unpriv / p_priv if p_priv > 0 else 1.0


class FEHITL_R3:
    """Drop-in replacement for the R2 class FEHITL_Ablatable (same signature)."""

    def __init__(self, input_dim=None, task='regression', device='cpu',
                 random_seed=42, epsilon=EPSILON, di_threshold=DI_LOW,
                 eod_threshold=0.1, use_hed=True, use_mog=True, use_fu=True,
                 interv_frac=1.0, mog_mode=None, single_alpha=1.0,
                 di_target=DI_TARGET, reverse_bound=DI_HIGH, subset_seed=42,
                 fu_reevaluate=False):
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.di_threshold = di_threshold
        self.eod_threshold = eod_threshold          # reported, never enforced
        self.use_hed = use_hed
        self.use_fu = use_fu
        self.interv_frac = interv_frac
        # use_mog=False now means the genuine single-candidate control.
        self.mog_mode = mog_mode or ('algorithm1' if use_mog else 'single')
        if self.mog_mode not in ('algorithm1', 'single', 'r2_reduced_strength'):
            raise ValueError(f'unknown mog_mode {self.mog_mode!r}')
        self.use_mog = (self.mog_mode == 'algorithm1')
        self.single_alpha = single_alpha
        self.di_target = di_target
        self.reverse_bound = reverse_bound
        self.subset_seed = subset_seed
        self.fu_reevaluate = fu_reevaluate

        self.base_model = None
        self.threshold = None
        self._X_train = self._y_train = self._s_train = None
        self._X_eff = self._y_eff = self._s_eff = None
        self._eff_source = None
        self.last_candidate_table = None
        self.last_decision = None
        self.fu_diagnostics = None

    # ── Training (identical to R2) ──────────────────────────────────
    def fit_base(self, X_train, y_train, X_val=None, y_val=None,
                 s_train=None, s_val=None):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=200, random_state=self.random_seed
        ).fit(X, y)
        return self.attach_base(model, X, y, s_train, X_val, y_val, s_val)

    def attach_base(self, model, X_train=None, y_train=None, s_train=None,
                    X_val=None, y_val=None, s_val=None):
        """Use an already-fitted base model, so several variants share one fit."""
        self.base_model = model
        self._X_train = None if X_train is None else np.asarray(X_train, dtype=np.float64)
        self._y_train = None if y_train is None else np.asarray(y_train, dtype=np.float64).ravel()
        self._s_train = None if s_train is None else np.asarray(s_train).ravel()
        if X_val is not None and y_val is not None and s_val is not None:
            self._X_eff = np.asarray(X_val, dtype=np.float64)
            self._y_eff = np.asarray(y_val, dtype=np.float64).ravel()
            self._s_eff = np.asarray(s_val).ravel()
            self._eff_source = 'validation'
        elif self._X_train is not None and self._y_train is not None \
                and self._s_train is not None:
            self._X_eff, self._y_eff, self._s_eff = (
                self._X_train, self._y_train, self._s_train)
            self._eff_source = 'training (no validation split supplied)'
        else:
            self._X_eff = self._y_eff = self._s_eff = None
            self._eff_source = None
        return self

    def set_binarize_threshold(self, threshold):
        """Fix theta, e.g. median(y_train), to align with the evaluation."""
        self.threshold = float(threshold)

    # ── Candidates ──────────────────────────────────────────────────
    def _boost(self, alpha, di):
        # Algorithm 1, line 7, evaluated left to right as in R2.
        return 1.0 + alpha * (self.di_threshold - di) * GAIN

    def _efficiency_loss(self, boost, p_eff):
        """Relative R^2 loss on the efficiency data. The boost is applied to ALL
        unprivileged validation cases: an upper bound when interv_frac < 1."""
        if p_eff is None:
            return np.nan
        r2_base = r2_score(self._y_eff, p_eff)
        q = p_eff.copy()
        q[self._s_eff == 1] *= boost
        return (r2_base - r2_score(self._y_eff, q)) / abs(r2_base) \
            if r2_base != 0 else np.nan

    # ── Prediction with intervention ────────────────────────────────
    def predict_with_intervention(self, X, s, context_list=None):
        X = np.asarray(X, dtype=np.float64)
        s = np.asarray(s).ravel()
        base = self.base_model.predict(X)
        if self.threshold is None:
            self.threshold = np.median(base)          # R2 convention
        rec = {'n_test_total': int(len(s)), 'n_unpriv': int(np.sum(s == 1)),
               'monitor_threshold': float(self.threshold),
               'mog_mode': self.mog_mode, 'eod_threshold_enforced': False,
               'efficiency_source': self._eff_source}
        self.last_candidate_table, self.fu_diagnostics = None, None

        if not self.use_hed:                          # same-base control
            rec.update(triggered=False, reason='HED ablated', n_routed=0,
                       selected_alpha=None, selection_rule=None)
            self.last_decision = rec
            return base

        di = disparate_impact((base > self.threshold).astype(int), s)
        rec['di_before_monitor'] = float(di)
        if not di < self.di_threshold:
            rec.update(triggered=False, reason='DI >= DI_LOW', n_routed=0,
                       di_after_monitor=float(di), selected_alpha=None,
                       selection_rule=None)
            self.last_decision = rec
            return base

        unpriv_idx = np.where(s == 1)[0]
        n_route = int(len(unpriv_idx) * self.interv_frac)
        rec.update(triggered=True, n_routed=int(n_route),
                   routed_share_of_unpriv=n_route / max(len(unpriv_idx), 1),
                   routed_share_of_total=n_route / len(s))
        preds = base.copy()
        best = None

        if n_route > 0:
            rng = np.random.RandomState(self.subset_seed)
            chosen = rng.choice(unpriv_idx, size=n_route, replace=False)
            if self.mog_mode == 'algorithm1':
                specs = [(a, self._boost(a, di)) for a in ALPHAS]
            elif self.mog_mode == 'single':
                specs = [(self.single_alpha, self._boost(self.single_alpha, di))]
            else:   # bitwise the R2 "w/o MOG" expression
                specs = [(R2_REDUCED_GAIN / GAIN,
                          1.0 + 1.0 * (self.di_threshold - di) * R2_REDUCED_GAIN)]
            base_scale = np.mean(np.abs(base)) + 1e-12
            p_eff = (self.base_model.predict(self._X_eff)
                     if self._X_eff is not None else None)
            table = []
            for alpha, boost in specs:
                trial = base.copy()
                trial[chosen] *= boost
                di_c = disparate_impact((trial > self.threshold).astype(int), s)
                eff = self._efficiency_loss(boost, p_eff)
                row = {'alpha': float(alpha), 'boost': float(boost),
                       'di_after_monitor': float(di_c),
                       'di_priv_after_monitor': float(1.0 / di_c) if di_c > 0 else np.inf,
                       'efficiency_loss_rel': float(eff),
                       'deviation_rel': float(np.mean(np.abs(trial - base)) / base_scale),
                       'efficiency_ok': bool(np.isnan(eff) or eff <= self.epsilon),
                       'reverse_ok': bool(di_c <= self.reverse_bound),
                       'reaches_target': bool(di_c >= self.di_target)}
                row['admissible'] = row['efficiency_ok'] and row['reverse_ok']
                row['qualifying'] = row['admissible'] and row['reaches_target']
                table.append(row)

            if self.mog_mode == 'algorithm1':
                qual = [r for r in table if r['qualifying']]
                adm = [r for r in table if r['admissible']]
                if qual:
                    best = min(qual, key=lambda r: r['deviation_rel'])
                    rule = 'target met; minimum deviation (practicality)'
                elif adm:
                    best = max(adm, key=lambda r: (r['di_after_monitor'],
                                                   -r['deviation_rel']))
                    rule = 'target not reachable; largest admissible DI'
                else:
                    rule = 'no admissible candidate; unchanged predictions'
            else:
                best, rule = table[0], 'single candidate (no arbitration)'
            for r in table:
                r['selected'] = best is not None and r is best
            self.last_candidate_table = table
            if best is not None:
                preds[chosen] *= best['boost']
            rec.update(n_candidates=len(table),
                       n_admissible=sum(r['admissible'] for r in table),
                       n_qualifying=sum(r['qualifying'] for r in table),
                       n_efficiency_rejections=sum(not r['efficiency_ok'] for r in table),
                       n_reverse_rejections=sum(not r['reverse_ok'] for r in table))
        else:
            rule = 'no routed cases; unchanged predictions'

        rec.update(selected_alpha=None if best is None else best['alpha'],
                   selected_boost=None if best is None else best['boost'],
                   selection_rule=rule,
                   di_after_monitor=float(disparate_impact(
                       (preds > self.threshold).astype(int), s)))
        self.last_decision = rec

        # Layer 4: Feedback & Update, R2 order of evaluation --------------
        if self.use_fu and self._X_train is not None and self._s_train is not None:
            boost_fu = best['boost'] if best is not None else self._boost(1.0, di)
            post = self._feedback_and_update(X, boost_fu)
            self.fu_diagnostics['post_update_predictions'] = post
            if self.fu_reevaluate:        # hypothetical scenario only
                return post
        return preds

    # ── Feedback & Update (unvalidated architectural placeholder) ───
    def _feedback_and_update(self, X, boost):
        """Replicates R2 exactly; see the module docstring for what it is."""
        tr = self.base_model.predict(self._X_train)
        corrected = tr.copy()
        tr_di = disparate_impact((tr > np.median(tr)).astype(int), self._s_train)
        targets_corrected = tr_di < self.di_threshold
        if targets_corrected:
            corrected[self._s_train == 1] *= boost
        old = self.base_model
        self.base_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            max_iter=20, random_state=self.random_seed, warm_start=True
        ).fit(self._X_train, corrected)
        self.fu_diagnostics = {
            'fu_train_di': float(tr_di),
            'fu_targets_corrected': bool(targets_corrected),
            'model_replaced': self.base_model is not old,
            'refit_from_fresh_initialisation': True,
            'refit_iterations': int(self.base_model.n_iter_),
            'returned_predictions_computed_before_refit': not self.fu_reevaluate,
        }
        return self.base_model.predict(np.asarray(X, dtype=np.float64))
