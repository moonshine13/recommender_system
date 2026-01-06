import os
import pickle
from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, Query

from src.recommender.model_based.model_based_recommendations_predict import main as model_based_run
from src.utils.logging import logger

router = APIRouter()

try:
    with open("/app/src/models/timesvdpp_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except:
    logger.error("No model found!")


# -------------------------------
# Endpoint: Model-based recommendations with time
# -------------------------------
@router.get("/model_recommendations_with_time/",
        summary="Get model-based recommended products",
        description="Returns the top-N recommended products based on ratings in time.",
)
def get_model_recommendations_with_time(
    user_id: str = Query(..., description="User ID for recommendations"),
    exclude_rated: bool =  Query(True, description="If exclude products rated by user"),
    n: int = Query(5, description="Number of top products"),
):
    try:
        logger.info("Starting API model based recommendation calculation")
        recs = model_based_run(model, user_id=user_id, time = 1476640644, exclude_rated = exclude_rated, n=n,)
        logger.info("Model based recommendations computed successfully")
    except Exception as e:
        logger.exception("Error computing model based recommendation")
        raise HTTPException(status_code=500, detail=f"Error computing recommendations: {str(e)}") from e 
    return {"recommendations": recs}