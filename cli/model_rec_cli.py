import argparse
import os
import pickle
from src.recommender.model_based.model_based_recommendations_predict import main as model_based_run
import sys
from src.utils.logging import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get top products")
    parser.add_argument("--user_id", type=str, required=True, help="User ID")
    parser.add_argument("--exclude_rated", type=bool, default=True, help="Exclude products rated by user")
    parser.add_argument("--n", type=int, default=5, help="Number of top products to return")
    args = parser.parse_args()
    

    try:
        logger.info("Starting CLI model based recommendation calculation")
        with open("./src/models/timesvdpp_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except:
        logger.error("No model found!")
        ## print("Warning no model found!")
        sys.exit(1)
    try:
        recs=model_based_run(
            model=model,
            user_id=args.user_id,
            exclude_rated=args.exclude_rated,
            n=args.n,
        )
        logger.info("Model based recommendations computed successfully")
        print(recs)
    except Exception as exc:
        ## print(f"Error: {exc}", file=sys.stderr)
        logger.exception("Error computing model based recommendation")
        sys.exit(2)