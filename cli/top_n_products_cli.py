import argparse
import sys

from src.recommender.top_n_products.top_n_products import main as top_n_products_run
from src.utils.logging import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get top products")
    parser.add_argument("--path", type=str, required=True, help="Path to ratings")
    parser.add_argument("--days", type=int, default=365, help="Number of days to analyze")
    parser.add_argument("--min_ratings", type=int, default=10, help="Minimum number of ratings")
    parser.add_argument("--n", type=int, default=5, help="Number of top products to return")
    args = parser.parse_args()
    
    try:
        logger.info("Starting CLI top products calculation")
        top_products = top_n_products_run(
            path=args.path,
            days=args.days,
            min_ratings=args.min_ratings,
            n=args.n
        )
        logger.info("Top products computed successfully")
        print(top_products)
    except FileNotFoundError:
        ## print(f"Error: CSV file not found at {args.path}", file=sys.stderr)
        logger.error("CSV file not found at: %s", args.path)
        sys.exit(1)
    except Exception as exc:
        ## print(f"Error: {exc}", file=sys.stderr)
        logger.exception("Error computing top products")
        sys.exit(2)