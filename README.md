# FE-HITL: Fairness-Enhanced Human-in-the-Loop Framework

This repository contains the complete code and data for the PLOS ONE paper:

**"From Günther Anders to Engineerable Fairness: A Fairness-Enhanced Human-in-the-Loop Framework for High-Risk AI Decision-Making"**  
by Wenjun Tian¹, Ying Wang²  
¹ School of Marxism, Northeastern University, Shenyang, China  
² School of Computer Science and Engineering, Beijing University of Agriculture, Beijing, China  
Corresponding author: Wenjun Tian (tianwenjunneu@outlook.com)

## Overview

We propose a Fairness-Enhanced Human-in-the-Loop (FE-HITL) framework that translates Günther Anders’ philosophical diagnoses (human obsolescence, Promethean shame, technological imperialism) into three concrete engineering compensation mechanisms: forced human intervention, explainable multi‑option generation, and value arbitration. The framework is validated on two high‑risk decision scenarios: an agricultural resource allocation simulation dataset (10,000 farmers) and the UCI German Credit dataset (1,000 samples). Experimental results show that FE‑HITL substantially improves algorithmic fairness (Disparate Impact, Equal Opportunity Difference) while maintaining or even enhancing predictive performance (R², accuracy). Ablation experiments confirm that all three compensation mechanisms are indispensable.

All code and data are publicly available under the MIT license to ensure full reproducibility.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── generate_agricultural_data.py      # Script to create the agricultural simulation dataset
│   ├── download_german_credit.py          # Script to download and preprocess the German Credit dataset
│   ├── agricultural_data_full.csv         # Generated after running the agricultural script
│   ├── agricultural_train.csv
│   ├── agricultural_val.csv
│   ├── agricultural_test.csv
│   └── german_credit_processed.csv        # Generated after running the credit download script
├── src/
│   ├── fairness_metrics.py                 # Functions to compute DI, EOD, AOD
│   ├── human_simulator.py                   # Rule‑based simulated human decision maker
│   ├── data_loader.py                       # Helper functions to load the datasets
│   ├── utils.py                              # Random seed setting, evaluation metrics
│   ├── models/
│   │   ├── baseline_lr.py                    # Logistic / Linear Regression baseline
│   │   ├── baseline_dl.py                    # Deep Learning baseline (PyTorch)
│   │   ├── debiased_hitl.py                  # Reimplementation of Debiased‑HITL (Zhang et al. 2021)
│   │   └── fe_hitl.py                         # Proposed FE-HITL framework
│   └── ablation.py                            # Helpers for ablation experiments
├── experiments/
│   ├── run_agricultural.py                    # Main experiment on agricultural dataset (Table 1)
│   ├── run_credit.py                           # Main experiment on German Credit dataset (Table 3)
│   └── run_ablation.py                         # Ablation study (Table 2)
├── figures/
│   ├── generate_fig1.py                        # Schematic of the FE-HITL framework (Figure 1)
│   ├── generate_fig2.py                        # Bar chart for agricultural dataset (Figure 2)
│   ├── generate_fig3.py                        # Bar chart for ablation study (Figure 3)
│   ├── generate_fig4.py                        # Bar chart for German Credit dataset (Figure 4)
│   └── generate_fig5.py                        # Example case workflow (Figure 5, schematic)
└── appendix/
    └── S1_Appendix.md                           # Detailed hyperparameters and implementation notes
```

## Requirements

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

The code has been tested with Python 3.9, PyTorch 1.10.0, scikit‑learn 1.0.0, and the versions specified in the file.

## Data

### Agricultural Simulation Dataset
Run the following command from the repository root to generate the agricultural dataset (10,000 samples) with injected historical bias:

```bash
python data/generate_agricultural_data.py
```

This creates four files in the `data/` folder:  
- `agricultural_data_full.csv` (complete dataset)  
- `agricultural_train.csv` (70% training)  
- `agricultural_val.csv` (15% validation)  
- `agricultural_test.csv` (15% testing)

### German Credit Dataset
The UCI German Credit dataset is downloaded and preprocessed automatically by:

```bash
python data/download_german_credit.py
```

The script saves the processed version as `data/german_credit_processed.csv`. It contains the target variable `class` (1 = good, 0 = bad) and the sensitive attribute `gender` (1 = male, 0 = female), as described in the paper.

## Reproducing Experiments

All experiments are designed to be run from the repository root. They print the main results (metrics) to the console; you may redirect the output or modify the scripts to save them as CSV files.

### 1. Agricultural Resource Allocation (Regression)

```bash
python experiments/run_agricultural.py
```

This script evaluates four models: Logistic Regression (LR), Deep Learning (DL), Debiased‑HITL (reproduced), and FE‑HITL. It outputs R², RMSE, Disparate Impact (DI), Equal Opportunity Difference (EOD), and Average Odds Difference (AOD), matching Table 1 in the paper.

### 2. German Credit Approval (Classification)

```bash
python experiments/run_credit.py
```

Outputs accuracy, DI, EOD, and AOD for the same four models, matching Table 3.

### 3. Ablation Study

```bash
python experiments/run_ablation.py
```

Runs the FE‑HITL framework with and without each of the three compensation mechanisms (forced human intervention, multi‑option generation, feedback & update). The results correspond to Table 2.

## Reproducing Figures

After running the experiments, you can generate the figures used in the paper. All figure scripts save the output as TIFF files (300 dpi) in the repository root (or current working directory).

```bash
python figures/generate_fig1.py      # Framework schematic (Figure 1)
python figures/generate_fig2.py      # Agricultural dataset comparison (Figure 2)
python figures/generate_fig3.py      # Ablation study (Figure 3)
python figures/generate_fig4.py      # German Credit comparison (Figure 4)
python figures/generate_fig5.py      # Example case workflow (Figure 5)
```

**Note:** Figures 1 and 5 are schematic diagrams; the provided scripts produce basic outlines. For publication‑quality images, we recommend refining them with vector graphics software (e.g., Adobe Illustrator, Inkscape). The data‑driven Figures 2–4 are generated directly from experimental results.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or data in your research, please cite our PLOS ONE paper:

```
Tian W, Wang Y. From Günther Anders to Engineerable Fairness: A Fairness‑Enhanced Human‑in‑the‑Loop Framework for High‑Risk AI Decision‑Making. PLoS ONE. 2026; … (in press).
```

BibTeX entry:

```bibtex
@article{tian2026anders,
  title   = {From G{"u}nther Anders to Engineerable Fairness: A Fairness‑Enhanced Human‑in‑the‑Loop Framework for High‑Risk AI Decision‑Making},
  author  = {Tian, Wenjun and Wang, Ying},
  journal = {PLoS ONE},
  year    = {2026},
  note    = {in press}
}
```

## Data Availability

All datasets generated or analyzed during this study are included in this published article and its supplementary information files. The simulated agricultural dataset is created by the script `data/generate_agricultural_data.py`. The German Credit dataset is publicly available from the UCI Machine Learning Repository and is downloaded automatically by `data/download_german_credit.py`. The complete source code is archived in the following public repositories:

- GitHub: [https://github.com/YingWang08/anders-fairness-hitl](https://github.com/YingWang08/anders-fairness-hitl)
- Figshare: [https://doi.org/10.6084/m9.figshare.31672465](https://doi.org/10.6084/m9.figshare.31672465)

## Contact

For questions or issues, please contact Wenjun Tian at tianwenjunneu@outlook.com.