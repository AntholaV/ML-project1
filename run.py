"""
Expects x_train.csv, y_train.csv, x_test.csv in a `data/` folder next to
this script.
"""
import os

import numpy as np

from helpers import load_csv_data, create_csv_submission
from preprocessing import compute_preprocessing_stats, apply_preprocessing

# TODO: import and use the final chosen model from implementations.py
# (see cross_validation results for the comparison between methods)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = "submission.csv"


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

    # TODO: train the final model and generate predictions
    # y_binary = np.where(y_train == -1, 0, y_train)
    # w, loss = ...
    # y_pred = ...
    # create_csv_submission(test_ids, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
