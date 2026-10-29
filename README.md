# CS-433 Machine Learning — Project 1

Team: Anthony Grand, Carolina Antunes Pereira

## Overview

This project predicts the risk of coronary heart disease (MICHD) from the
BRFSS health survey dataset, as part of the CS-433 Machine Learning course
(EPFL, Fall 2026), Project 1 (AIcrowd challenge:
`epfl-machine-learning-project-1`). Only Python and NumPy are used for the
models (matplotlib is only used for plots in `eda.py`).

## Repository structure

```
.
├── README.md              # this file
├── implementations.py     # the 6 required ML methods
├── preprocessing.py       # data cleaning and feature scaling
├── run.py                 # reproduces our best AIcrowd submission
├── cross_validation.py    # 4-fold CV used to choose the model and threshold
├── eda.py                 # exploratory data analysis
├── helpers.py             # provided helpers (load_csv_data, create_csv_submission)
└── data/                  # x_train.csv, y_train.csv, x_test.csv (not tracked in git)
```

## Setup

1. Clone this repository.
2. Install Python 3 with `numpy` (and `matplotlib` for `eda.py`). We ran
   our code with Python 3.12.
3. Download `x_train.csv`, `y_train.csv` and `x_test.csv` from AIcrowd and
   place them in a `data/` folder next to `run.py`.

## Reproducing our best submission

```
python run.py
```

This loads the data, applies our preprocessing, trains the final model and
writes `submission.csv` in the format required by AIcrowd. There is no
randomness involved (weights start at zero and full-batch gradient descent
is used), so the output is the same on every run. Reading the CSV files
takes a few minutes.

To reproduce the model comparison and the choice of decision threshold:

```
python cross_validation.py
```

This takes roughly 10-15 minutes. `python eda.py` prints the class balance
and missing-value statistics used to design the preprocessing.

## Method summary

**Preprocessing** (`preprocessing.py`, all statistics fitted on the training
set only and applied unchanged to the test set):

- BRFSS "don't know / refused" codes (7, 9, 77, 99, 777, 999, 7777, 9999)
  are replaced by NaN in the columns where they sit above the 95th
  percentile of the column's other values.
- Columns that are constant or more than 80% missing are dropped
  (321 -> 200 columns).
- Remaining NaN are imputed with the column mean, features are
  standardized and a bias column is added.

**Model** (`run.py`): L2-regularized logistic regression
(`reg_logistic_regression` from `implementations.py`) with
lambda = 0.001, learning rate 0.5, 200 gradient descent iterations and
initial weights 0. Labels are mapped from {-1, 1} to {0, 1} for training
and predictions are mapped back. The decision threshold on the predicted
probability is 0.2 instead of 0.5, because the classes are strongly
imbalanced (91% / 9%).

**Model selection** (`cross_validation.py`): 4-fold cross-validation
(random folds, seed 1), using F1-score on the positive class instead of
accuracy:

| Model (best lambda)             | Threshold | F1    |
|---------------------------------|-----------|-------|
| Ridge regression (0.001)        | 0.5       | 0.087 |
| Regularized logistic regression | 0.5       | 0.218 |
| Regularized logistic regression | 0.2       | 0.425 |

The lambda and the threshold were chosen on the same folds that report
these scores, so the last F1 is probably slightly optimistic. See the
report for details.

## Implementations

`implementations.py` contains the 6 required functions with the signatures
specified in the project description:

- `mean_squared_error_gd`
- `mean_squared_error_sgd`
- `least_squares`
- `ridge_regression`
- `logistic_regression`
- `reg_logistic_regression`
