"""
Handles:
- BRFSS "special" codes (e.g. 7, 9, 77, 99, 777, 999) that mean
  "don't know" / "refused" rather than a real measurement, replaced by NaN
- dropping columns that are constant or mostly missing (fit on train only)
- mean imputation and standardization (fit on train only)
- adding a bias column

Usage:
    stats = compute_preprocessing_stats(x_train)
    x_train_processed = apply_preprocessing(x_train, stats)
    x_test_processed = apply_preprocessing(x_test, stats)  # same stats, no leakage
"""
import numpy as np

# common BRFSS codes for "don't know" / "refused", depending on the
# number of digits used for that particular question
SPECIAL_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999)


def detect_special_codes(x, codes=SPECIAL_CODES, percentile=95, margin=1):
    """For each column, find which of `codes` behave like a "special" value:
    present in the column, and sitting well above the bulk of the other
    (non-code) values. Returns {column_index: set_of_codes}.

    This is a heuristic, not a guarantee: it can miss a real special code
    that happens to fall inside the normal range, and in principle could
    flag a genuine extreme value. Worth spot-checking a few columns by eye.
    """
    special = {}
    for col in range(x.shape[1]):
        col_data = x[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            continue
        found = set()
        for code in codes:
            is_code = valid == code
            if not is_code.any():
                continue
            others = valid[~is_code]
            if len(others) == 0:
                continue
            threshold = np.percentile(others, percentile) + margin
            if code > threshold:
                found.add(code)
        if found:
            special[col] = found
    return special


def replace_special_codes(x, special):
    """Replace detected special codes with NaN (does not mutate the input)."""
    x = x.copy()
    for col, codes in special.items():
        for code in codes:
            x[x[:, col] == code, col] = np.nan
    return x


def compute_preprocessing_stats(x_train, missing_threshold=0.8):
    """Fit all preprocessing decisions on the training set only.

    Args:
        x_train: raw training features.
        missing_threshold: drop columns with a NaN ratio above this,
            after special codes have been converted to NaN.

    Returns:
        dict with everything needed to preprocess train and test the
        same way (special codes, kept columns, mean, std).
    """
    special = detect_special_codes(x_train)
    x_clean = replace_special_codes(x_train, special)

    nan_ratio = np.isnan(x_clean).mean(axis=0)
    col_std_all = np.nanstd(x_clean, axis=0)
    keep_mask = (nan_ratio <= missing_threshold) & (col_std_all > 0)

    x_kept = x_clean[:, keep_mask]
    col_mean = np.nanmean(x_kept, axis=0)
    col_std = np.nanstd(x_kept, axis=0)
    col_std[col_std == 0] = 1  # safety net, should not trigger given keep_mask

    return {
        "special": special,
        "keep_mask": keep_mask,
        "col_mean": col_mean,
        "col_std": col_std,
    }


def apply_preprocessing(x, stats):
    """Apply a fitted preprocessing (from compute_preprocessing_stats) to
    x_train or x_test: special codes -> NaN, drop columns, impute, scale,
    add bias column."""
    x_clean = replace_special_codes(x, stats["special"])
    x_clean = x_clean[:, stats["keep_mask"]]

    nan_mask = np.isnan(x_clean)
    if nan_mask.any():
        col_mean_broadcast = np.broadcast_to(stats["col_mean"], x_clean.shape)
        x_clean[nan_mask] = col_mean_broadcast[nan_mask]

    x_clean = (x_clean - stats["col_mean"]) / stats["col_std"]
    x_clean = np.hstack([np.ones((x_clean.shape[0], 1)), x_clean])
    return x_clean