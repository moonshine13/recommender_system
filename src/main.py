"""
Main entrypoint for the FastAPI application.

This module initializes the FastAPI app and includes all endpoints
for user-based, model-based, and top-N product recommendations.

Usage:
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI

from src.api.endpoints import model_rec, top_products, user_rec

app = FastAPI(title="Recommendation API")

app.include_router(user_rec.router, prefix="/recommend/user", tags=["user-based"])
app.include_router(model_rec.router, prefix="/recommend/model", tags=["model-based"])
app.include_router(top_products.router, prefix="/products", tags=["products"])
