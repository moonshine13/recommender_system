"""
Data preprocessing utilities for model-based recommendations.

This module provides functions to prepare user-item rating data for 
model-based recommendation algorithms.
"""

import numpy as np


def leave_last_out_split(ratings):
    """
    Split ratings into train and test sets using leave-last-out strategy.

    For each user with number of rated products > 1,
    the last rating (by timestamp) is used as test data,
    and all previous ratings are used for training.

    Args:
        ratings (np.ndarray): Array of user-item ratings with columns
                              [user_id, item_id, rating, timestamp].

    Returns:
        tuple: (train_ratings, test_ratings) as numpy arrays.
    """
    ratings = ratings[np.argsort(ratings[:, 3])]  # sort by time
    train_list = []
    test_list = []
    for u in np.unique(ratings[:, 0]):
        user_ratings = ratings[ratings[:, 0] == u]
        if len(user_ratings) > 1:
            train_list.append(user_ratings[:-1])
            test_list.append(user_ratings[-1:])
        else:
            train_list.append(user_ratings)
    train_ratings = np.vstack(train_list)
    test_ratings = np.vstack(test_list)
    return train_ratings, test_ratings


def normalize_time(ts, t_min, t_max):
    """
    Normalize a timestamp to a value between 0 and 1.

    Args:
        ts (datetime): Timestamp to normalize.
        t_min (float): Minimum time value in the dataset.
        t_max (float): Maximum time value in the dataset.

    Returns:
        float: Normalized time value between 0 and 1.
    """
    t = ts.year + ts.month / 12.0
    return (t - t_min) / (t_max - t_min)


def preprocess_data(data):
    """
    Preprocess raw rating data for model-based recommendations.

    Maps user and item IDs to consecutive integers, normalizes timestamps,
    and converts ratings into a numpy array suitable for model input.

    Args:
        data (list of dict): Raw rating data with keys
                             ["user_id", "product_id", "rating", "timestamp"].

    Returns:
        tuple:
            - ratings_array (np.ndarray): Array of shape (n_ratings, 4) with columns
              [user_index, item_index, rating, normalized_time].
            - user_map (dict): Mapping from original user IDs to integer indices.
            - item_map (dict): Mapping from original item IDs to integer indices.
            - t_min (float): Minimum normalized timestamp.
            - t_max (float): Maximum normalized timestamp.
    """
    user_map = {}
    item_map = {}
    user_counter = 0
    item_counter = 0

    timestamps = np.array(
        [row["timestamp"].year + row["timestamp"].month / 12.0 for row in data]
    )
    t_min = timestamps.min()
    t_max = timestamps.max()
    processed_data = []
    for row in data:  ##[:10000]:
        u = row["user_id"]
        i = row["product_id"]
        r = float(row["rating"])
        t = normalize_time(row["timestamp"], t_min, t_max)
        if u not in user_map:
            user_map[u] = user_counter
            user_counter += 1

        if i not in item_map:
            item_map[i] = item_counter
            item_counter += 1

        processed_data.append((user_map[u], item_map[i], r, t))

    ratings_array = np.array(processed_data, dtype=float)

    # Ensure user/item indices are integers
    ratings_array[:, 0] = ratings_array[:, 0].astype(int)
    ratings_array[:, 1] = ratings_array[:, 1].astype(int)

    return ratings_array, user_map, item_map, t_min, t_max
