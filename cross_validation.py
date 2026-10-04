"""K-fold cross-validation for Project 1, with metrics suited to an
imbalanced dataset (F1-score, precision, recall) instead of accuracy alone.

Compares ridge_regression and reg_logistic_regression over a small grid of
lambda values, and reports the best decision threshold for the winning
model. Only uses numpy, as required (no scikit-learn).

Usage:
    python cross_validation.py
"""
import os
import time

import numpy as np

from helpers import load_csv_data
from implementations import ridge_regression, reg_logistic_regression, sigmoid
from preprocessing import compute_preprocessing_stats, apply_preprocessing

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 1
K_FOLD = 4

# small, deliberately modest grids to keep runtime reasonable; widen these
# once the pipeline is confirmed to work end to end
RIDGE_LAMBDAS = [0.001, 0.01, 0.1]
LOGISTIC_LAMBDAS = [0.001, 0.01]
LOGISTIC_GAMMA = 0.5
LOGISTIC_MAX_ITERS = 200

THRESHOLDS_TO_TRY = [0.1, 0.2, 0.3, 0.4, 0.5]


# --------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred):
    """y_true, y_pred are in {0, 1}. Returns accuracy, precision, recall, f1."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ------------------------------------------------------------ k-fold split
def build_k_indices(n, k_fold, seed=SEED):
    """Split n sample indices into k_fold random, roughly equal groups."""
    np.random.seed(seed)
    indices = np.random.permutation(n)
    fold_sizes = np.full(k_fold, n // k_fold)
    fold_sizes[: n % k_fold] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size
    return folds


# ------------------------------------------------------------- CV loops
def cross_validate_ridge(y, x, k_fold, lambdas):
    """For each lambda, train/evaluate ridge_regression over k folds.
    Returns {lambda: averaged metrics dict}."""
    folds = build_k_indices(len(y), k_fold)
    results = {}

    for lambda_ in lambdas:
        fold_metrics = []
        for k in range(k_fold):
            val_idx = folds[k]
            train_idx = np.hstack([folds[i] for i in range(k_fold) if i != k])

            # fit preprocessing on this fold's training split only
            stats = compute_preprocessing_stats(x[train_idx])
            x_tr = apply_preprocessing(x[train_idx], stats)
            x_val = apply_preprocessing(x[val_idx], stats)

            w, _ = ridge_regression(y[train_idx], x_tr, lambda_)
            y_score = x_val @ w
            y_pred = (y_score >= 0.5).astype(int)
            fold_metrics.append(compute_metrics(y[val_idx], y_pred))

        results[lambda_] = {
            key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]
        }
    return results


def cross_validate_logistic(y, x, k_fold, lambdas, gamma, max_iters):
    """Same as above but for reg_logistic_regression."""
    folds = build_k_indices(len(y), k_fold)
    results = {}

    for lambda_ in lambdas:
        fold_metrics = []
        for k in range(k_fold):
            val_idx = folds[k]
            train_idx = np.hstack([folds[i] for i in range(k_fold) if i != k])

            stats = compute_preprocessing_stats(x[train_idx])
            x_tr = apply_preprocessing(x[train_idx], stats)
            x_val = apply_preprocessing(x[val_idx], stats)

            initial_w = np.zeros(x_tr.shape[1])
            w, _ = reg_logistic_regression(
                y[train_idx], x_tr, lambda_, initial_w, max_iters, gamma
            )
            y_score = sigmoid(x_val @ w)
            y_pred = (y_score >= 0.5).astype(int)
            fold_metrics.append(compute_metrics(y[val_idx], y_pred))

        results[lambda_] = {
            key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]
        }
    return results


def print_results(name, results):
    print(f"\n--- {name} ---")
    print(f"{'lambda':>10} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for lambda_, metrics in results.items():
        print(
            f"{lambda_:>10} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}"
        )


def tune_threshold(y, x, k_fold, lambda_, gamma, max_iters, thresholds):
    """Sweep decision thresholds for the best logistic model found, since
    with 91%/9% class imbalance, 0.5 is unlikely to be optimal for F1."""
    folds = build_k_indices(len(y), k_fold)
    results = {t: [] for t in thresholds}

    for k in range(k_fold):
        val_idx = folds[k]
        train_idx = np.hstack([folds[i] for i in range(k_fold) if i != k])

        stats = compute_preprocessing_stats(x[train_idx])
        x_tr = apply_preprocessing(x[train_idx], stats)
        x_val = apply_preprocessing(x[val_idx], stats)

        initial_w = np.zeros(x_tr.shape[1])
        w, _ = reg_logistic_regression(y[train_idx], x_tr, lambda_, initial_w, max_iters, gamma)
        y_score = sigmoid(x_val @ w)

        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            results[t].append(compute_metrics(y[val_idx], y_pred)["f1"])

    print(f"\n--- Threshold sweep (lambda={lambda_}) ---")
    print(f"{'threshold':>10} {'avg f1':>10}")
    for t in thresholds:
        print(f"{t:>10} {np.mean(results[t]):>10.4f}")


def main():
    x_train, _, y_train, _, _ = load_csv_data(DATA_DIR)
    y_binary = np.where(y_train == -1, 0, y_train).astype(int)

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Running {K_FOLD}-fold cross-validation (this can take a few minutes)...")

    t0 = time.time()
    ridge_results = cross_validate_ridge(y_binary, x_train, K_FOLD, RIDGE_LAMBDAS)
    print_results("Ridge regression", ridge_results)
    print(f"(ridge CV took {time.time() - t0:.1f}s)")

    t0 = time.time()
    logistic_results = cross_validate_logistic(
        y_binary, x_train, K_FOLD, LOGISTIC_LAMBDAS, LOGISTIC_GAMMA, LOGISTIC_MAX_ITERS
    )
    print_results("Regularized logistic regression", logistic_results)
    print(f"(logistic CV took {time.time() - t0:.1f}s)")

    # pick the logistic lambda with the best F1 and sweep thresholds for it
    best_lambda = max(logistic_results, key=lambda l: logistic_results[l]["f1"])
    tune_threshold(
        y_binary, x_train, K_FOLD, best_lambda, LOGISTIC_GAMMA, LOGISTIC_MAX_ITERS, THRESHOLDS_TO_TRY
    )


if __name__ == "__main__":
    main()