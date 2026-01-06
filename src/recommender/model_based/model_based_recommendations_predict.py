"""
Functions to generate predictions using model-based recommendation algorithms.

This module provides utilities to:
- Compute predictions for a given user or set of users.
- Exclude already rated items if desired.
- Return top-N recommended products for each user.
"""

from datetime import datetime, timezone

from src.recommender.model_based.model_based_recommendations_data_preprocessing import (
    normalize_time,
)


def predict_all_items_for_user(
    model, user_idx: str, time=1476640644, exclude_rated=True
):
    """
    Predict ratings for all items for a given user.

    Args:
        model: Trained recommendation model with user/item maps and predict method.
        user_idx (str): User ID for whom to generate predictions.
        time (int, optional): Timestamp to use for time-aware predictions. Defaults to 1476640644.
        exclude_rated (bool, optional): Whether to skip items the user has already rated. Defaults to True.

    Returns:
        list of dict: Predicted ratings for all items, sorted descending by rating.
                      Each dict contains "product_id" and "predicted_rating".
    """
    ts = datetime.fromtimestamp(time, tz=timezone.utc)
    u = model.user_map.get(user_idx, -1)
    t = normalize_time(ts, model.t_min, model.t_max)

    predictions = []
    for p, i in model.item_map.items():
        # optionally skip items already rated
        if exclude_rated and i in model.user_rated_items[u]:
            continue
        pred = model.predict(u, i, t)
        predictions.append({"product_id": p, "predicted_rating": pred})

    # sort by predicted rating descending
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = [
        {
            **p,
            "predicted_rating": round(max(0, min(5, p["predicted_rating"].item())), 2),
        }
        for p in predictions
    ]
    return predictions


def model_based_recommendations_with_time(
    model,
    user_idx: str,
    time: int = 1476640644,
    exclude_rated: bool = True,
    n: int = 5,
):
    """
    Generate top-N model-based recommendations for a user at a given time.

    Args:
        model: Trained recommendation model with user/item maps and predict method.
        user_idx (str): User ID for whom recommendations are computed.
        time (int, optional): Timestamp for time-aware predictions. Defaults to 1476640644.
        exclude_rated (bool, optional): Whether to skip items the user has already rated. Defaults to True.
        n (int, optional): Number of top recommendations to return. Defaults to 5.

    Returns:
        list of dict: Top-N recommended items with "product_id" and "predicted_rating".
    """
    predictions = predict_all_items_for_user(
        model=model,
        user_idx=user_idx,
        time=time,
        exclude_rated=exclude_rated,
    )

    return predictions[:n]


def model_based_run(
    model,
    user_id: str,
    time: int = 1476640644,
    exclude_rated: bool = True,
    n: int = 5,
):
    """
    Generate top-N model-based recommendations for a specific user.

    Args:
        model: Trained recommendation model with user/item maps and predict method.
        user_id (str): User ID for whom recommendations are generated.
        time (int, optional): Timestamp for time-aware predictions. Defaults to 1476640644.
        exclude_rated (bool, optional): Whether to skip items the user has already rated. Defaults to True.
        n (int, optional): Number of top recommendations to return. Defaults to 5.

    Returns:
        list of dict: Top-N recommended items with "product_id" and "predicted_rating".
    """
    # Check if user exists
    if not user_id in model.user_map:
        raise ValueError(f"User ID {user_id} not found")
    recs = model_based_recommendations_with_time(
        model,
        user_idx=user_id,
        time=time,
        exclude_rated=exclude_rated,
        n=n,
    )
    return recs
