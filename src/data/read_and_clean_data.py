"""
Functions to load and clean CSV rating data.

This module provides utilities to:
- Load CSV files containing user-product ratings.
- Clean data by handling missing values, invalid ratings, and timestamps.
- Return data in a format suitable for recommendation algorithms.

Example:
    data = load_and_clean_data("data/ratings.csv")
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List


# Data loading
def load_all_data(file_path):
    """
    Load all user–item rating data from a CSV file.
    """
    rows = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            f, fieldnames=["user_id", "product_id", "rating", "timestamp"]
        )
        next(reader)  # skip header

        for row in reader:
            row["rating"] = float(row["rating"])
            row["timestamp"] = int(row["timestamp"])
            rows.append(row)
    return rows


def load_data(path: str) -> List[Dict[str, Any]]:
    """
    Load and validate user–item rating data from a CSV file.

    The CSV file must contain the following required columns:
    - user_id: Identifier of the user (string or integer)
    - product_id: Identifier of the item/product (string or integer)
    - rating: Numeric rating value
    - timestamp: Integer timestamp (e.g., Unix time or normalized time)

    Rows with missing required fields or invalid numeric values
    (rating or timestamp) are skipped.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the ratings data.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents
        a valid rating record with the following keys:
        - user_id
        - product_id
        - rating (float)
        - timestamp (int)

    Notes
    -----
    - The function assumes the first row of the CSV file is a header.
    - Rows with empty fields or conversion errors are silently ignored.

    Examples
    --------
    data = load_data("ratings.csv")
    data[0]
    {'user_id': '42', 'product_id': '128', 'rating': 4.5, 'timestamp': 1704067200}
    """
    rows = []
    required_fields = ("user_id", "product_id", "rating", "timestamp")
    float_cast = float
    int_cast = int

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Map columns
        idx_map = {k: header.index(k) for k in required_fields}

        for r in reader:
            if any(not r[idx_map[k]] for k in idx_map):
                continue
            try:
                rating = float_cast(r[idx_map["rating"]])
                timestamp = int_cast(r[idx_map["timestamp"]])
            except (ValueError, TypeError, OSError):
                continue
            rows.append(
                {
                    "user_id": r[idx_map["user_id"]],
                    "product_id": r[idx_map["product_id"]],
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
    return rows


# Data cleaning
def clean_data_slow(rows: List[Dict]) -> List[Dict]:
    """
    Impute invalid ratings (-1, 99) in a list of dicts
    using hybrid baseline:
        r_hat = global_mean + (user_mean - global_mean) + (item_mean - global_mean)
    Impute invalid timestamps (0) in a list of dicts using min positive timestamp from dataset.
    """
    # Compute min positive timestamp
    positive_timestamps = [r["timestamp"] for r in rows if r["timestamp"] > 0]
    min_positive_ts = (
        min(positive_timestamps) if positive_timestamps else 1
    )  # fallback to 1

    # Collect ratings for global/user/product, ignoring invalid ratings
    valid_ratings = [r["rating"] for r in rows if 0 <= r["rating"] <= 5]

    if not valid_ratings:
        global_mean = 2.5
    else:
        global_mean = sum(valid_ratings) / len(valid_ratings)

    # Compute user means
    user_ratings = {}
    for r in rows:
        if 0 <= r["rating"] <= 5:
            user_ratings.setdefault(r["user_id"], []).append(r["rating"])
    user_mean = {u: sum(vals) / len(vals) for u, vals in user_ratings.items()}

    # Compute item (product) means
    item_ratings = {}
    for r in rows:
        if 0 <= r["rating"] <= 5:
            item_ratings.setdefault(r["product_id"], []).append(r["rating"])
    item_mean = {i: sum(vals) / len(vals) for i, vals in item_ratings.items()}

    # Imputation
    for r in rows:
        # Replace timestamp 0 with min positive timestamp
        ts = r["timestamp"] if r["timestamp"] > 0 else min_positive_ts
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        r["timestamp"] = ts_dt

        rating = r["rating"]
        if not 0 <= rating <= 5:  # -1, 99, or other invalid
            u_mean = user_mean.get(r["user_id"], global_mean)
            i_mean = item_mean.get(r["product_id"], global_mean)
            r["rating"] = global_mean + (u_mean - global_mean) + (i_mean - global_mean)
            # Clip to 0-5
            r["rating"] = max(0, min(5, r["rating"]))
    return rows


def clean_data(rows: List[Dict]) -> List[Dict]:
    """
    Impute missing or invalid ratings (-1, 99, etc.) in a list of dicts
    using hybrid baseline:
        r_hat = global_mean + (user_mean - global_mean) + (item_mean - global_mean)
    Also converts timestamps >0 to datetime, replaces 0 timestamps with min positive.
    """
    if not rows:
        return rows

    # Initialize
    global_sum = 0.0
    global_count = 0
    user_sums = defaultdict(float)
    user_counts = defaultdict(int)
    item_sums = defaultdict(float)
    item_counts = defaultdict(int)
    min_positive_ts = float("inf")

    # First pass: aggregate valid ratings and find min positive timestamp
    for r in rows:
        ts = r.get("timestamp", 0)
        if ts > 0:
            min_positive_ts = min(min_positive_ts, ts)

        rating = r.get("rating", -1)
        if 0 <= rating <= 5:
            global_sum += rating
            global_count += 1
            user_sums[r["user_id"]] += rating
            user_counts[r["user_id"]] += 1
            item_sums[r["product_id"]] += rating
            item_counts[r["product_id"]] += 1

    min_positive_ts = min_positive_ts if min_positive_ts != float("inf") else 1
    global_mean = global_sum / global_count if global_count > 0 else 2.5

    # Compute user/item means
    user_mean = {u: user_sums[u] / user_counts[u] for u in user_sums}
    item_mean = {i: item_sums[i] / item_counts[i] for i in item_sums}

    # Second pass: impute ratings and convert timestamps
    for r in rows:
        # Replace timestamp 0 with min positive timestamp
        ts = r.get("timestamp", 0)
        ts = ts if ts > 0 else min_positive_ts
        r["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Impute invalid ratings
        rating = r.get("rating", -1)
        if not 0 <= rating <= 5:
            u_mean = user_mean.get(r["user_id"], global_mean)
            i_mean = item_mean.get(r["product_id"], global_mean)
            r["rating"] = max(
                0, min(5, global_mean + (u_mean - global_mean) + (i_mean - global_mean))
            )

    return rows


def load_and_clean_data(path: str) -> List[Dict[str, Any]]:
    """
    Function to load and clean CSV rating data.
    """
    rows = load_data(path)
    rows = clean_data(rows)
    return rows
