import os
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Load x_train.csv, x_test.csv and y_train.csv from data_path.

    Args:
        data_path: folder containing the three CSV files.
        sub_sample: if True, keep only every 50th training sample (for fast tests).

    Returns:
        x_train, x_test, y_train, train_ids, test_ids
    """
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"), delimiter=",", skip_header=1, usecols=1
    )

    # First column is the sample Id; the rest are the features
    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # Optional sub-sampling to speed up development
    if sub_sample:
        x_train = x_train[::50]
        y_train = y_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """Write a submission file for AIcrowd with columns Id and Prediction.

    Args:
        ids: sample ids of the test set.
        y_pred: predicted labels, one per id.
        name: output file name (e.g. "submission.csv").
    """
    with open(name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=["Id", "Prediction"])
        writer.writeheader()
        for i, p in zip(ids, y_pred):
            writer.writerow({"Id": int(i), "Prediction": int(p)})
