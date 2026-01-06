from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, Query
from typing import Literal
from src.data.read_and_clean_data import load_and_clean_data
from src.recommender.user_based.user_based_recommendations import main as user_based_run
from src.utils.logging import logger

router = APIRouter()

# -------------------------------
# Endpoint: User-based recommendations
# -------------------------------
@router.get("/user_recommendations/",
        summary="Get user-based recommended products",
        description="Returns the top-N recommended products based on ratings for top-K similar users.",
)
def get_user_based_recommendations(
    path: str = Query('/data/ratings.csv', description="Path to the CSV file"),
    user_id: str = Query(..., description="User ID for recommendations"),
    k: int = Query(5, description="Number of top-k similar users"),
    n: int = Query(5, description="Number of top products"),
    type: Literal["user_based", "user_based_with_time"] = Query("user_based", description="Type of recommendation"), 
):
    try:
        logger.info("Starting API user based recommendation calculation")
        recs = user_based_run(path, user_id=user_id, n=n, k=k, type=type)
        logger.info("User based recommendations computed successfully")
    except Exception as e:
        logger.exception("Error computing user based recommendation")
        raise HTTPException(status_code=500, detail=f"Error computing recommendations: {str(e)}") from e
    return {"recommendations": recs}