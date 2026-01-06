"""
FastAPI endpoints for model-based recommendations.

Provides routes to compute and return model-based recommendations
for a given user, including error handling and logging.
"""

import pickle

from fastapi import APIRouter, HTTPException, Query

from src.recommender.model_based.model_based_recommendations_predict import (
    model_based_run,
)
from src.utils.logging import logger

router = APIRouter()

try:
    with open("/app/src/models/timesvdpp_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except:
    logger.error("No model found!")


# Endpoint: Model-based recommendations with time
@router.get(
    "/model_recommendations_with_time/",
    summary="Get model-based recommended products",
    description="Returns the top-N recommended products based on ratings in time.",
)
def get_model_recommendations_with_time(
    user_id: str = Query(..., description="User ID for recommendations"),
    exclude_rated: bool = Query(True, description="If exclude products rated by user"),
    n: int = Query(5, description="Number of top products"),
):
    """
    FastAPI endpoint to get model-based recommendations for a user at a specific timestamp.

    This endpoint uses the pre-trained model to compute the top-N recommended products
    for a given user. It optionally excludes products the user has already rated.

    Args:
        user_id (str): ID of the user for whom recommendations are computed.
        exclude_rated (bool, optional): If True, products already rated by the user
            will be excluded. Defaults to True.
        n (int, optional): Number of top products to return. Defaults to 5.
    """
    try:
        logger.info("Starting API model based recommendation calculation")
        recs = model_based_run(
            model,
            user_id=user_id,
            time=1476640644,
            exclude_rated=exclude_rated,
            n=n,
        )
        logger.info("Model based recommendations computed successfully")
    except Exception as e:
        logger.exception("Error computing model based recommendation")
        raise HTTPException(
            status_code=500, detail=f"Error computing recommendations: {str(e)}"
        ) from e
    return {"recommendations": recs}
