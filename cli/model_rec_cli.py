"""
Command-line interface for user-based recommendations.

Provides a CLI entrypoint to compute personalized recommendations
for a given user using the collaborative filtering model.

Usage:
    python -m cli.model_rec_cli --user_id 671 --exclude_rated True --n 5
"""

import argparse
import pickle
import sys

from src.recommender.model_based.model_based_recommendations_predict import (
    model_based_run,
)
from src.utils.logging import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get top products")
    parser.add_argument("--user_id", type=str, required=True, help="User ID")
    parser.add_argument(
        "--exclude_rated",
        type=bool,
        default=True,
        help="Exclude products rated by user",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of top products to return"
    )
    args = parser.parse_args()

    try:
        logger.info("Starting CLI model based recommendation calculation")
        with open("./src/models/timesvdpp_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.exception("No model found!")
        sys.exit(1)
    try:
        recs = model_based_run(
            model=model,
            user_id=args.user_id,
            exclude_rated=args.exclude_rated,
            n=args.n,
        )
        logger.info("Model based recommendations computed successfully")
        print(recs)
    except Exception as exc:
        logger.exception("Error computing model based recommendation")
        sys.exit(2)
