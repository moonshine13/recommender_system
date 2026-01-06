"""
FastAPI endpoints for top-N products recommendations.

Provides routes to compute and return the top-N products.
"""

from fastapi import APIRouter, HTTPException, Query

from src.recommender.top_n_products.top_n_products import top_n_products_run
from src.utils.logging import logger

router = APIRouter()


# Endpoint: Top N products
@router.get(
    "/top_products/",
    summary="Get top-rated products",
    description="Returns the top-N products based on average rating within a time window.",
)
def get_top_products(
    path: str = Query("/data/ratings.csv", description="Path to the CSV file"),
    days: int = Query(365, description="Number of days to be analyzed"),
    min_ratings: int = Query(10, description="Minimum number of ratings"),
    n: int = Query(5, description="Number of top products"),
):
    """
    FastAPI endpoint to compute the top-N products.

    This endpoint analyzes user-product ratings over the specified number of days
    and returns the top-N products based on total ratings, optionally filtering
    by minimum number of ratings per product.

    Args:
        path (str): Path to the CSV file containing ratings. Defaults to "/data/ratings.csv".
        days (int): Number of recent days to include in the analysis. Defaults to 365.
        min_ratings (int): Minimum number of ratings a product must have to be considered. Defaults to 10.
        n (int): Number of top products to return. Defaults to 5.
    """
    try:
        logger.info("Starting API top products calculation")
        top = top_n_products_run(path=path, days=days, min_ratings=min_ratings, n=n)
        logger.info("Top products computed successfully")
    except Exception as e:
        logger.exception("Error computing top products")
        raise HTTPException(
            status_code=500, detail=f"Error computing top products: {str(e)}"
        ) from e
    return {"top_products": top}
