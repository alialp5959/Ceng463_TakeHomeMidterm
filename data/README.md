# Datasets

Both datasets are stored unmodified in `data/raw/`. Never edit these files; all cleaning happens in code.

## 1. Bank Marketing (classification — Questions 1, 2, 4, 5)

| | |
|---|---|
| File | `raw/bank-additional-full.csv` (separator `;`) |
| Original source | UCI Machine Learning Repository, Bank Marketing dataset — https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Copy obtained from | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip → `bank-additional.zip` → `bank-additional/bank-additional-full.csv` (unmodified) |
| License | CC BY 4.0 |
| Shape | 41,188 rows × 21 columns (20 features + target `y`) |
| Target | `y` — did the client subscribe to a term deposit? `no`: 36,548 (88.7%), `yes`: 4,640 (11.3%) |
| SHA-256 | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` |

Notes for the audit (Q1):
- Missing values are encoded as the string `"unknown"` (job, marital, education, default, housing, loan).
- 12 exact duplicate rows.
- `duration` (call duration) is only known *after* the call ends, and `duration = 0` implies `y = no`. The UCI documentation says to drop it for a realistic predictive model, which makes it a natural target-leakage example.
- `pdays = 999` means "not previously contacted" (a sentinel value, not a real number of days).

Citation: S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 62:22–31, 2014.

## 2. California Housing (regression — Question 3)

| | |
|---|---|
| File | `raw/california_housing.csv` |
| Original source | 1990 U.S. Census, via StatLib (Pace & Barry, 1997) |
| Copy obtained from | https://github.com/ageron/handson-ml2 (`datasets/housing/housing.tgz`) |
| Shape | 20,640 rows × 10 columns |
| Target | `median_house_value` (USD) |
| SHA-256 | `8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e` |

Notes for the audit (Q3):
- `total_bedrooms` has 207 missing values.
- `ocean_proximity` is categorical; the `ISLAND` category has only 5 rows (a rare/unseen-category risk).
- The target is capped at 500,001 (965 rows). This censoring affects residuals.
- `total_rooms`, `total_bedrooms`, `population` and `households` are strongly correlated.

Citation: R. K. Pace and R. Barry. *Sparse Spatial Autoregressions.* Statistics & Probability Letters, 33(3):291–297, 1997. Version with `ocean_proximity` from A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 2nd ed., O'Reilly, 2019.
