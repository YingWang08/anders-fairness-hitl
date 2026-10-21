# FE-HITL simulation code and data — PLOS ONE PONE-D-26-12981 (R3)

> **Results of record.** Every number in the R3 manuscript and S1 Appendix comes from
> (i) the R2 experiment scripts at commit `fa79a9f`, whose scientific implementation is
> unchanged and which Reviewer #4 independently reproduced to < 1e-14, and (ii) the
> read-only R3 additions below. All tables in the manuscript were filled from `r3_tables/`,
> which `regen_stats.py`, `inference_fixed.py` and `fairness_cost_joint.py` regenerate from
> the deposited raw CSVs. `r3_tables/reviewer_number_check.csv` shows the 17/17 values quoted
> by Reviewer #4; `r3_tables/MANIFEST.json` records input SHA-256 hashes, library versions,
> conventions and seed sets (its `git_commit` field, `acaba4d`, is the code state at
> generation time; later commits change only documentation, the manifest and figures).

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
python figures/check_figures.py                       # Figs 1-5: regenerate in a temporary folder, compare with figures/r3/
python figures/figures_r3.py r3_tables figures/regenerated   # Figs 1-5 (TIFF, LZW, 300 dpi) from r3_tables/
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
| Figures | All five manuscript figures drawn by one script from `r3_tables/` (PLOS format: TIFF/LZW, RGB, 300 dpi, Arial or its metric-compatible substitute Liberation Sans, no titles in images) | `figures/figures_r3.py`, `figures/r3/` |

Documentation-only corrections were made directly in the two R2 scripts (docstrings and
comments, plus the printed run label "50 seeds"); the executable code is otherwise unchanged
(verified by AST comparison against commit `fa79a9f`).

### Figure files (updated in R4)

`figures/r3/Fig1.tif`–`Fig5.tif` are the figure files submitted with the article
(`figures/r3/Fig*.png` are previews). All five were drawn by `figures/figures_r3.py` from
`r3_tables/`; the only statistical annotation (Fig 2: FE-HITL − DL on DI, Δ = +0.232,
adjusted p < 0.001) is the contrast reported in the Results. `python figures/check_figures.py`
regenerates the figures in a temporary folder and compares them pixel by pixel with the
deposited files. The deposited files were rendered with matplotlib 3.10.8, pillow 12.1.1
and Liberation Sans 2.1.5 (a metric-compatible substitute for Arial); with these versions
all five are reproduced identically. When Arial is installed, `figures_r3.py` uses it
instead, and the regenerated text then differs slightly in glyph shape.

Until R4, the commit cited in the manuscript (`1384e99`, release V3.0) still contained the
R2 plotting script `figures/figures_all.py`, which drew hand-entered values together with
statistics that were withdrawn in R3; `figures_r3.py` had not been committed. R4 adds the
R3 figure commit (`5ede506`), removes the last R2 figure image
(`figures/fig5_workflow.png`) and adds `figures/check_figures.py`. No analysis code, raw
result or table changed, and the figures are those submitted with R3.

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
simulated process); German Credit seeds all resample the same 1,000 rows (70/30 splits
stratified by outcome).

**Data-generating process.** The simulated data are generated in memory by
`generate_agricultural_data_biased()` (main experiments), `generate_agricultural_data_biased_v2()`
(bias structures) and `generate_full_dataset()` (cross-validation) in
`experiments/run_sensitivity_standalone.py`. Region is not a model input; the biased
historical allocation is. The target is `max(0.5*base + 0.5*historical + N(0, 2^2), 0)`, so it
retains about half of the injected penalty (S1 Appendix, Section 2).
`data/generate_agricultural_data.py` is an earlier generator with a different target and is
not used.

## 4. Environment

The results of record were generated with Python 3.14.3; `requirements-lock.txt` is the full
pip freeze of that environment. `requirements.txt` pins the direct dependencies; with these
pins, `experiments/run_r3_additions.py` and the three statistics scripts were re-executed on
Linux with Python 3.12.3 and reproduced every deposited value (maximum absolute difference
< 1e-12). `statsmodels` is a direct dependency. `torch` is not needed (imported optionally,
never used). `run_credit_clean.py` needs network access to openml.org; its R2 raw outputs are
the results of record and were reproduced independently by Reviewer #4 at commit `fa79a9f`.

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
- The same corrected test is applied to the five fold means of the agricultural
  cross-validation (`r3_tables/agri_cv_inference.csv`).
- DI is undefined when no privileged-group case receives the positive prediction; the code
  then records DI = 1. This happened only in 4 of the 250 German Credit CV evaluations
  (degenerate networks predicting almost no bad risks), identically for all three methods,
  so paired contrasts are unaffected.

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

## 10. Known implementation details (documented, not changed)

- Agricultural CV: `run_kfold_cv()` passes the training fold as the DL baseline's validation
  data (`dl.fit(Xt, yt, Xt, yt)`), so its internal early-stopping split contains duplicated
  training rows. This affects only the DL row of manuscript Table 2.
- Algorithm 1 as implemented in `src/fe_hitl_r3.py` computes the efficiency loss on the
  validation split; in the 50 CV runs (no validation split) it uses the training fold.
- The FRP shrink factor for German Credit is floored at 0.5; the floor is never reached.

## 11. Superseded or unused files

Not used by any reported result: `experiments/run_agricultural.py`, `run_ablation.py`,
`run_credit.py`, `run_german_credit.py` (older 30-seed credit run), `run_sensitivity.py`,
`run_statistical.py`, `run_stats_supplement.py`; top-level CSV/TXT files in
`experiments/`; `experiments/results/`; `experiments/revision_results/stats/` (older
statistics from an earlier run, e.g. DL DI 0.6543); `src/models/fe_hitl.py`,
`src/models/debiased_hitl.py`, `src/human_simulator.py`, `src/data_loader.py`, `merge_csv.py`;
`data/generate_agricultural_data.py` and `data/*.csv` (first-submission generator and outputs;
see `data/README.md`). Removed in the final R3 commit: the initial-submission
`appendix/S1_Appendix.md` (superseded by the S1 Appendix published with the article), the R2
figure images (the last one, `figures/fig5_workflow.png`, in R4) and `figures/figures_all.py`
(superseded by `figures/figures_r3.py`), and the
IDE and bytecode files; `.gitignore.txt` was renamed to `.gitignore`.

## 12. License

MIT. If you use this code, please cite the article.
