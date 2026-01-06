import argparse
import sys
from src.recommender.user_based.user_based_recommendations import main as user_based_run
from src.utils.logging import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get top products")
    parser.add_argument("--path", type=str, required=True, help="Path to ratings")
    parser.add_argument("--user_id", type=str, required=True, help="User ID")
    parser.add_argument("--k", type=int, default=5, help="Number of top similar users")
    parser.add_argument("--n", type=int, default=5, help="Number of top products to return")
    parser.add_argument("--type", type=str, default='user_based_with_time', choices=["user_based", "user_based_with_time"],help="Type of recommendation")
    args = parser.parse_args()

    try:
        logger.info("Starting CLI user based recommendation calculation")
        recs = user_based_run(
            path=args.path,
            user_id=args.user_id,
            k=args.k,
            n=args.n,
            type=args.type
        )
        logger.info("User based recommendations computed successfully")
        print(recs)
    except FileNotFoundError:
        ## print(f"Error: CSV file not found at {args.path}", file=sys.stderr)
        logger.error("CSV file not found at: %s", args.path)
        sys.exit(1)
    except Exception as exc:
        ## print(f"Error: {exc}", file=sys.stderr)
        logger.exception("Error computing user based recommendation")
        sys.exit(2)