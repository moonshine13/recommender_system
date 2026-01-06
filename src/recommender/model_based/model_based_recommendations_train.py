"""
Training utilities for model-based recommendation algorithms.

This module provides functions to:
- Train collaborative filtering models (e.g., TimeSVD++) on user-item rating data.
- Handle time-aware and implicit feedback data.
- Save trained models for later prediction.
- Optionally evaluate performance on a test set.
"""

import argparse
import pickle

from src.data.read_and_clean_data import load_and_clean_data
from src.models.timesvdpp import TimeSVDppVectorized
from src.recommender.model_based.model_based_recommendations_data_preprocessing import (
    leave_last_out_split,
    preprocess_data,
)


def main(path: str, out_path: str):
    """
    Train a TimeSVD++ model on user-item rating data and save the trained model.

    This function:
    - Loads and cleans raw rating data from a CSV file.
    - Preprocesses data (maps user/item IDs, normalizes timestamps).
    - Splits data into train and test sets using leave-last-out.
    - Trains a TimeSVD++ model with specified hyperparameters.
    - Saves the trained model to a pickle file.

    Args:
        path (str): Path to the CSV file containing raw rating data.
        out_path (str): Path to save the trained model as a pickle file.

    Example:
        main("data/ratings.csv", "models/timesvdpp_model.pkl")
    """
    data = load_and_clean_data(path=path)

    ratings_array, user_map, item_map, t_min, t_max = preprocess_data(data)

    train_ratings, test_ratings = leave_last_out_split(ratings_array)
    print("Train Ratings:\n", len(train_ratings))
    print("Test Ratings:\n", len(test_ratings))

    model = None
    model = TimeSVDppVectorized(n_factors=10, n_epochs=50, lr=0.01, reg=0.05)

    model.fit(train_ratings, test_ratings)

    model.user_map = user_map
    model.item_map = item_map
    model.t_min = t_min
    model.t_max = t_max

    with open(out_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument("--path", type=str, required=True, help="Path to ratings")
    parser.add_argument("--out_path", type=str, required=True, help="Output model path")
    args = parser.parse_args()

    main(path=args.path, out_path=args.out_path)
