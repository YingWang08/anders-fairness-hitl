# data/

* `german.data` - UCI Statlog (German Credit) data (Hofmann 1994, doi:10.24432/C5NC77).
  The reported German Credit experiments load the same data from OpenML (data ID 31)
  in `experiments/run_credit_clean.py`; this copy is provided for convenience.
* `agricultural_*.csv`, `data/*.csv` and `generate_agricultural_data.py` are **superseded**
  outputs and code from the first submission. They use a different (unbiased) prediction
  target and are **not** used by any reported result. The simulated data in the manuscript
  are generated in memory by `experiments/run_sensitivity_standalone.py`
  (see S1 Appendix, Section 2, for the data-generating process).
