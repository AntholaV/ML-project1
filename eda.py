"""
Prints class balance, missing-value counts, and per-column value ranges
for x_train, to help decide on preprocessing (NaN handling, special
BRFSS codes, scaling, etc.). Only uses numpy and matplotlib, as required.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from helpers import load_csv_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def class_balance(y):
    """Print the count and proportion of each class in y."""
    values, counts = np.unique(y, return_counts=True)
    print("\n--- Class balance (y_train) ---")
    for v, c in zip(values, counts):
        print(f"  {v:>5.0f}: {c:>7d}  ({100 * c / len(y):.2f}%)")


def missing_values(x, top_n=15):
    """Print the columns with the most NaN values."""
    nan_counts = np.isnan(x).sum(axis=0)
    nan_ratio = nan_counts / x.shape[0]
    order = np.argsort(nan_ratio)[::-1]

    print(f"\n--- Missing values: top {top_n} columns (of {x.shape[1]}) ---")
    for idx in order[:top_n]:
        print(f"  column {idx:>4d}: {nan_ratio[idx] * 100:6.2f}% missing")

    n_fully_missing = np.sum(nan_ratio == 1.0)
    n_never_missing = np.sum(nan_ratio == 0.0)
    print(f"\nColumns 100% missing: {n_fully_missing}")
    print(f"Columns with no missing values: {n_never_missing}")
    print(f"Overall missing rate: {100 * np.isnan(x).sum() / x.size:.2f}%")


def value_ranges(x, n_cols=10):
    """Print min/max/unique-count for a few columns, to spot special codes
    (e.g. 7, 9, 77, 99, 777, 999 used by BRFSS for 'don't know'/'refused')."""
    print(f"\n--- Value ranges: first {n_cols} columns ---")
    for col in range(min(n_cols, x.shape[1])):
        col_data = x[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            print(f"  column {col:>4d}: all NaN")
            continue
        n_unique = len(np.unique(valid))
        print(
            f"  column {col:>4d}: min={valid.min():>10.2f}  max={valid.max():>10.2f}  "
            f"unique values={n_unique}"
        )


def plot_missing_histogram(x, out_path="missing_values_histogram.png"):
    """Save a histogram of the per-column missing-value ratio."""
    nan_ratio = np.isnan(x).sum(axis=0) / x.shape[0]
    plt.figure(figsize=(8, 5))
    plt.hist(nan_ratio, bins=30)
    plt.xlabel("Fraction of missing values per column")
    plt.ylabel("Number of columns")
    plt.title("Distribution of missing-value rates across features")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved histogram to {out_path}")


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)
    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}")

    class_balance(y_train)
    missing_values(x_train)
    value_ranges(x_train)
    plot_missing_histogram(x_train)


if __name__ == "__main__":
    main()
