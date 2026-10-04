"""
Usage:
    python run.py

Expects x_train.csv, y_train.csv, x_test.csv in a `data/` folder next to
this script.

Model choice: regularized logistic regression, chosen over ridge
regression after 4-fold cross-validation (see cross_validation.py) showed
a much higher F1-score (0.22 vs 0.09 at the default 0.5 threshold), given
the strong class imbalance (91% / 9%). The decision threshold (0.2) was
also selected via cross-validation, since 0.5 was far from optimal for F1
on this imbalanced dataset.
"""
import os

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import reg_logistic_regression, sigmoid
from preprocessing import compute_preprocessing_stats, apply_preprocessing

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = "submission.csv"

# selected via 4-fold cross-validation (see cross_validation.py)
LAMBDA = 0.001
GAMMA = 0.5
MAX_ITERS = 200
THRESHOLD = 0.2


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)

    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}")
    print(f"y_train unique values: {np.unique(y_train)}")

    # fit preprocessing on train only, then apply the same transformation
    # to both train and test to avoid data leakage
    stats = compute_preprocessing_stats(x_train)
    x_train_processed = apply_preprocessing(x_train, stats)
    x_test_processed = apply_preprocessing(x_test, stats)
    print(f"Kept {stats['keep_mask'].sum()} / {len(stats['keep_mask'])} columns "
          f"after dropping constant / mostly-missing ones")

    # labels are expected in {0, 1} for our loss functions; AIcrowd uses
    # {-1, 1}, so convert if needed
    y_binary = np.where(y_train == -1, 0, y_train)

    initial_w = np.zeros(x_train_processed.shape[1])
    w, loss = reg_logistic_regression(
        y_binary, x_train_processed, LAMBDA, initial_w, MAX_ITERS, GAMMA
    )
    print(f"Training loss: {loss:.4f}")

    y_score = sigmoid(x_test_processed @ w)
    y_pred = np.where(y_score >= THRESHOLD, 1, -1)  # back to AIcrowd's {-1, 1} format

    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(f"Submission written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
