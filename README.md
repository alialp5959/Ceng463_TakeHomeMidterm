# CENG 463 – Machine Learning: Take-Home Midterm (Fall 2026–27)

Full machine-learning workflow study: leakage-safe data preparation, classification, regression with regularization, tree ensembles, and robust model selection.

| Question | Topic | Dataset |
|---|---|---|
| Q1 | Data preparation, leakage, experimental design | Bank Marketing |
| Q2 | Classification algorithms and error analysis | Bank Marketing |
| Q3 | Regression, regularization, generalization | California Housing |
| Q4 | Decision trees, bagging, boosting | Bank Marketing |
| Q5 | Robustness, statistical comparison, model selection | Bank Marketing |

Dataset details, sources and checksums: [`data/README.md`](data/README.md).

## Repository layout

```
data/raw/          original datasets (never modified)
notebooks/         one notebook per question (Q1–Q5)
src/               shared code (config, split, preprocessing pipelines)
results/figures/   saved plots
results/tables/    saved tables (CSV / LaTeX)
report/            LaTeX source and compiled PDF
```

## Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                       # then run notebooks/Q1 ... Q5 in order
```

All randomness uses the fixed seed defined in `src/config.py`.

## Environment

Python 3.13, scikit-learn 1.9 (exact versions in `requirements.txt`).
