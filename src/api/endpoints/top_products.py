from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, Query
from src.recommender.top_n_products.top_n_products import main as top_n_products_run
from src.utils.logging import logger

router = APIRouter()

# -------------------------------
# Endpoint: Top N products
# -------------------------------
@router.get("/top_products/",
        summary="Get top-rated products",
        description="Returns the top-N products based on average rating within a time window."
)
def get_top_products(
    path: str = Query('/data/ratings.csv', description="Path to the CSV file"),
    days: int = Query(365, description="Number of days to be analyzed"),
    min_ratings: int = Query(10, description="Minimum number of ratings"),
    n: int = Query(5, description="Number of top products"),
):
    try:
        logger.info("Starting API top products calculation")
        top = top_n_products_run(
            path=path,
            days=days,
            min_ratings=min_ratings,
            n=n
        )
        logger.info("Top products computed successfully")
    except Exception as e:
        logger.exception("Error computing top products")
        raise HTTPException(status_code=500, detail=f"Error computing top products: {str(e)}") from e
    return {"top_products": top}
