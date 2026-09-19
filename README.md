# FE-HITL simulation code and data — R3 state (PLOS ONE PONE-D-26-12981)

> **Result of record.** Every number in the R3 manuscript comes from (i) the R2
> experiment scripts at commit `fa79a9f`, whose scientific implementation is
> unchanged and which Reviewer #4 independently reproduced to < 1e-14, and (ii) the
> read-only R3 additions below. `regen_stats.py` writes
> `r3_tables/reviewer_number_check.csv` (17/17 values quoted by Reviewer #4
> reproduced) and `r3_tables/MANIFEST.json` (input SHA-256 hashes, library versions,
> conventions, seed sets). If the manuscript and `r3_tables/` disagree, `r3_tables/`
> is correct.

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd experiments                            # outputs are written relative to the working directory
python run_sensitivity_standalone.py      # 8 agricultural suites  -> experiments/revision_results/
python run_credit_clean.py                # German Credit, 50 seeds -> experiments/credit_revision_results/ (needs openml.org)
python run_r3_additions.py                # R3 analyses            -> experiments/revision_results/r3/ (~4 min, 1 CPU)
cd ..
python regen_stats.py                     # descriptive tables, agricultural tests, reviewer-number check
python inference_fixed.py                 # German Credit inference: conditional vs generalization
python fairness_cost_joint.py             # fairness + performance + workload on the same row
python check_references.py --mailto you@example.org   # Crossref audit of the reference list (internet)
```

The R2 README said to run from the repository root, which writes to `./revision_results/`
rather than the deposited `experiments/revision_results/`. Run the experiment scripts from
`experiments/` as shown.

## 2. What changed in R3

| Reviewer #4 point | Change | Where |
|---|---|---|
| 1 statistics from one version | All tables regenerated from the deposited raw CSVs; BH family = the authors' original family; Table 4 built from German Credit data | `regen_stats.py` |
| 2 inference design; parity objective | Primary endpoint \|DI−1\|; conditional (partition-fixed) vs generalization (corrected resampled t) analyses; `t(249)` withdrawn; guard/fallback accounting | `inference_fixed.py` |
| 3 MOG ablation | Algorithm 1 implemented exactly as written and compared with the executed routine on 320 runs; genuine ablation against every single fixed candidate; R2 "w/o MOG" relabelled as reduced correction strength | `src/fe_hitl_r3.py`, `experiments/run_r3_additions.py` |
| 4 F&U explanation | Per-seed order-of-evaluation evidence | `r3_fu_diagnostics.csv` |
| 5 same-base control | FE-HITL vs the same uncorrected base model; decomposition of the R² cost vs DL | `table_same_base_agri.csv` |
| 6 fairness and workload | Same-row reporting on the same 30 seeds, with the manuscript's assumptions (4 options × 3 s) and a sensitivity grid | `fairness_cost_joint.py` |
| 7 citation, README, environment | Reference audit script; this README; pinned `requirements.txt`; torch made optional | `check_references.py`, `requirements.txt`, `src/utils.py` |

`patches/` holds documentation-only corrections for the two R2 scripts (docstrings and
comments; executable code verified unchanged by AST comparison).

## 3. Entry points and seeds

| # | Suite in `run_sensitivity_standalone.py` | Seeds | Output in `experiments/revision_results/` |
|---|---|---|---|
| 1 | `run_bias_sensitivity` (10–40 % bias) | 42–71 | `bias_sensitivity_*` (Table 1 = bias 0.30) |
| 2 | `run_intervention_sensitivity` (10–100 %) | 42–71 | `interv_sensitivity_*` |
| 3 | `run_ablation` | 42–71 | `ablation_*` |
| 4 | `run_bias_structure_robustness` | 42–56 | `bias_structure_robustness_*` |
| 5 | `run_kfold_cv` | dataset seed 42; `StratifiedKFold(5, random_state=42)`; training seeds 42–51 | `kfold_cv_agri_*` |
| 6 | `run_reverse_discrimination` | 42–56 | `reverse_discrimination_*` |
| 7 | `run_binarization_sensitivity` | 42–56 | `binarization_sensitivity_*` |
| 8 | `run_intervention_cost` | 42–56 | `intervention_cost_*` |

German Credit: `run_credit_clean.py`, seeds 42–91 (50), one 5-fold partition
(`random_state=42`) × 50 training seeds, output `experiments/credit_revision_results/`.
Agricultural seeds each draw a new synthetic dataset (independent replicates of the
simulated process); German Credit seeds all resample the same 1,000 rows.

## 4. Environment

`requirements.txt` pins the versions with which the agricultural pipeline was re-executed
and matched the deposited CSVs (Python 3.12.3). The original runs used Python 3.14.
`statsmodels` is a direct dependency. `torch` is not needed: `src/utils.py` and
`src/models/baseline_dl.py` now import it optionally (it was only seeded, never used).
Figures need `matplotlib`.
`requirements-lock.txt` is the exact environment (Python 3.14.3) in which all results were generated and re-verified.

## 5. Statistical conventions

- Contrast direction: treatment − comparator (FE-HITL − X; Full − variant); pairing on seed.
- Descriptive 95 % CI: mean ± 1.96·SD/√n (normal approximation; `--ci t` for t-based).
  Contrast CIs are t-based, consistent with the paired t-tests.
- Benjamini–Hochberg within declared families (named in each output). Agricultural
  headline family (the authors' original): FE-HITL − {LR, DL, Debiased-HITL} ×
  {R², RMSE, DI, EOD, AOD} = 15.
- German Credit (`inference_fixed.py`): (A) conditional on this dataset and partition;
  (B1)/(B2) Nadeau–Bengio corrected resampled t (variance factor 1/J + n_test/n_train),
  because resampled splits of one dataset overlap (5-fold training sets share 75 % of
  rows). None of these analyses supports claims about other populations or domains.

## 6. Algorithm 1: what is enforced and what the executed code does

The executed agricultural routine applies the α = 1.0 candidate of Algorithm 1
(boost = 1 + (0.8 − DI)·0.7) to the routed unprivileged cases. `src/fe_hitl_r3.py`
implements the full loop (α ∈ {0.3, 0.5, 0.7, 1.0}) with the manuscript's decision rules
(target DI ≥ 0.85; efficiency loss ≤ 10 % on validation data; DI ≤ 1.25; minimum
deviation; fallback: largest admissible DI). `run_r3_additions.py` compares the two:

- headline configuration (30 % bias, 100 %): identical in 30/30 seeds (α = 1.0 selected);
- all 320 runs: identical in 300. Of 273 triggered runs, 253 select α = 1.0; 14 are ties
  in monitored DI resolved toward a smaller α by the practicality rule; 1 smaller α already
  meets the target; in 5 (40 % bias) α = 1.0 violates the 10 % efficiency rule (10.2–17.1 %).

| Element | Status |
|---|---|
| Trigger DI < 0.80 | enforced |
| Credit candidate guard [0.75, 1.333] | restricts accepted candidates only; 5/50 main-experiment seeds fell back and ended outside it |
| Efficiency-loss bound | not evaluated by the executed agricultural routine; evaluated in `fe_hitl_r3.py` |
| EOD threshold | not enforced anywhere (requires labels at decision time) |
| Monitoring threshold | FE-HITL: median of its own batch predictions; metrics and Debiased-HITL: median(y_train). Aligning them changes FE-HITL DI by 0.002 |

## 7. Feedback & Update

Unvalidated architectural component. In R2 the refit was triggered and replaced the
model, but the returned predictions were computed before the refit, so Full and
w/o-F&U are identical by order of evaluation. The refit builds a new `MLPRegressor`, so
`warm_start=True` has no effect (20 iterations from a fresh initialisation). Returning
the updated model's predictions would change all 1,500 test predictions and lower mean
DI from 0.864 to 0.742 (`r3_fu_diagnostics.csv`); this is a hypothetical check, not a result.

## 8. Intervention denominator and workload

`interv_frac` is a fraction of unprivileged (region A) test cases (~30 % of the test set).
Workload figures are arithmetic from assumptions stated in the R2 manuscript (4 options
per routed case, 3 s per option); nothing is generated per case or timed.

## 9. Disclosures

- Agricultural data are synthetic. Thresholds (0.85 target, 10 % efficiency loss, 30 %
  bias) are heuristic modelling assumptions unless a source and passage is cited.
- German Credit DI uses the model's positive class, which in the OpenML encoding is
  "bad credit"; DI > 1 means women are predicted "bad" more often.
- "Debiased-HITL" is an author-designed fixed-ratio proportional baseline; the previously
  cited source could not be verified.

## 10. Superseded or unused files

Not used by any reported result: `experiments/run_agricultural.py`, `run_ablation.py`,
`run_credit.py`, `run_german_credit.py` (older 30-seed credit run), `run_sensitivity.py`,
`run_statistical.py`, `run_stats_supplement.py`; top-level CSV/TXT files in
`experiments/`; `experiments/results/`; `experiments/revision_results/stats/` (older
statistics from an earlier run, e.g. DL DI 0.6543); `src/models/fe_hitl.py`,
`src/models/debiased_hitl.py`, `src/human_simulator.py`, `src/data_loader.py`;
`data/*.csv` (experiments generate data in memory). `.gitignore.txt` is not a valid
ignore file name (rename to `.gitignore`); this is why `.idea/` and `__pycache__/` were committed.

## 11. License

MIT. If you use this code, please cite the manuscript.
