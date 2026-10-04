# CS-433 Machine Learning — Project 1

Team members: <name 1>, <name 2>, <name 3>

## Overview

This project predicts the risk of coronary heart disease (MICHD) from the
BRFSS health survey dataset, as part of the CS-433 Machine Learning course
(EPFL, Fall 2026), Project 1 (AIcrowd challenge:
`INSERT HERE THE NAME`).

## Repository structure

```
.
├── README.md              # this file
├── implementations.py     # the 6 required ML methods
├── run.py                 # reproduces our best AIcrowd submission
├── helpers.py              # provided helper functions (load_csv_data, create_csv_submission)
└── dataset/                 # x_train.csv, y_train.csv, x_test.csv (not tracked in git)
```

## Setup

1. Clone this repository.
2. Create the environment (see `environment.yml` in the course repository,
   or install `numpy` and `matplotlib` manually).
3. Download `x_train.csv`, `y_train.csv`, `x_test.csv` from AIcrowd and place
   them in a `dataset/` folder next to `run.py`.

## Reproducing our best submission

```
python run.py
```

This script loads the training and test data, applies our preprocessing
pipeline, trains our final model, and writes `submission.csv` in the same
format required by AIcrowd (via `create_csv_submission` from `helpers.py`).

## Method summary

<TODO: briefly describe the final model, preprocessing steps, and
hyperparameters used, once decided. This should match what is described in
the report.>

## Implementations

`implementations.py` contains the 6 required functions with the signatures
specified in the project description:

- `mean_squared_error_gd`
- `mean_squared_error_sgd`
- `least_squares`
- `ridge_regression`
- `logistic_regression`
- `reg_logistic_regression`
