from fastapi import FastAPI
from src.api.endpoints import user_rec, model_rec, top_products
from fastapi import FastAPI

## TODO add readme
## TODO add docstrings

app = FastAPI(title="Recommendation API")

app.include_router(user_rec.router, prefix="/recommend/user", tags=["user-based"])
app.include_router(model_rec.router, prefix="/recommend/model", tags=["model-based"])
app.include_router(top_products.router, prefix="/products", tags=["products"])