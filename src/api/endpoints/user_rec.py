"""
FastAPI endpoints for user-based recommendations.
Provides routes to compute user-based and time-aware collaborative
filtering recommendations for a given user.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.recommender.user_based.user_based_recommendations import user_based_run
from src.utils.logging import logger

router = APIRouter()


# Endpoint: User-based recommendations
@router.get(
    "/user_recommendations/",
    summary="Get user-based recommended products",
    description="Returns the top-N recommended products based on ratings for top-K similar users.",
)
def get_user_based_recommendations(
    path: str = Query("/data/ratings.csv", description="Path to the CSV file"),
    user_id: str = Query(..., description="User ID for recommendations"),
    k: int = Query(5, description="Number of top-k similar users"),
    n: int = Query(5, description="Number of top products"),
    rec_type: Literal["user_based", "user_based_with_time"] = Query(
        "user_based", description="Type of recommendation"
    ),
):
    """
    FastAPI endpoint to compute user-based product recommendations for a given user.

    This endpoint uses collaborative filtering to recommend products to the specified
    user. It can optionally use time-aware recommendations and considers the top-K
    most similar users when generating recommendations.

    Args:
        path (str): Path to the CSV file containing ratings. Defaults to "/data/ratings.csv".
        user_id (str): ID of the user for whom recommendations are computed.
        k (int): Number of top-k similar users to consider. Defaults to 5.
        n (int): Number of top products to return. Defaults to 5.
        rec_type (Literal["user_based", "user_based_with_time"]): Type of recommendation
            to compute. Defaults to "user_based".
    """
    try:
        logger.info("Starting API user based recommendation calculation")
        recs = user_based_run(path, user_id=user_id, n=n, k=k, rec_type=rec_type)
        logger.info("User based recommendations computed successfully")
    except Exception as e:
        logger.exception("Error computing user based recommendation")
        raise HTTPException(
            status_code=500, detail=f"Error computing recommendations: {str(e)}"
        ) from e
    return {"recommendations": recs}
