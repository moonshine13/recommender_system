
import pickle
import sys
from datetime import datetime, timezone
## from src.models.timesvdpp import TimeSVDppVectorized
from src.recommender.model_based.model_based_recommendations_data_preprocessing import (preprocess_data,
                                                                 leave_last_out_split,
                                                                 normalize_time,)

def predict_all_items_for_user(model, user_idx:str, time=1476640644, exclude_rated=True):
    ts = datetime.fromtimestamp(time, tz=timezone.utc)
    u = model.user_map.get(user_idx, -1)
    t = normalize_time(ts, model.t_min, model.t_max)

    predictions = []
    for p,i in model.item_map.items():
        # optionally skip items already rated
        if exclude_rated and i in model.user_rated_items[u]:
            continue
        pred = model.predict(u, i, t)
        predictions.append({"product_id": p, "predicted_rating":pred})

    # sort by predicted rating descending
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = [
        {**p, "predicted_rating": round(max(0, min(5, p["predicted_rating"].item())), 2)}
        for p in predictions
    ]
    return predictions


def model_based_recommendations_with_time(model,
    user_idx: str,
    time: int = 1476640644,
    exclude_rated: bool = True,
    n: int = 5,):
    predictions = predict_all_items_for_user(model=model,
                                             user_idx=user_idx,
                                             time=time,
                                             exclude_rated=exclude_rated,)

    return predictions[:n]


def main_old():
    from data.read_and_clean_data import load_and_clean_data
    data = load_and_clean_data(path = "./data/ratings.csv")
    ratings_array, _, _, _, _ = preprocess_data(data)

    train_ratings, test_ratings = leave_last_out_split(ratings_array)
    print("Train Ratings:\n", len(train_ratings))
    print("Test Ratings:\n", len(test_ratings))

    with open("./src/models/timesvdpp_model.pkl", "rb") as f:
        model = pickle.load(f)

    ## single prediction
    u = model.user_map.get('671', -1)
    i = model.item_map.get('126', -1)
    ts = datetime.fromtimestamp(1476640644, tz=timezone.utc)
    t = normalize_time(ts, model.t_min, model.t_max)
    print("Predicted rating:", model.predict(u, i, t))


    recs = model_based_recommendations_with_time(model,
                                                user_idx ='671',
                                                time = 1476640644,
                                                exclude_rated = True,
                                                n = 5,)
    print("Recommendations:", recs)

def main(
    model,
    user_id: str,
    time: int = 1476640644, 
    exclude_rated: bool = True,
    n: int = 5,
):
    # Check if user exists
    if not user_id in model.user_map:
        raise ValueError(f"User ID {user_id} not found")
    recs = model_based_recommendations_with_time(model, user_idx=user_id, time = time, exclude_rated = exclude_rated, n=n,)
    return recs
