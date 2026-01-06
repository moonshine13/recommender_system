import sys
from typing import List, Dict, Any
from collections import defaultdict
from src.utils.distance_metrics import cosine_similarity
from heapq import nlargest
from math import exp, log
import argparse
from src.utils.logging import logger
from src.data.read_and_clean_data import load_and_clean_data

# --------------------------------------------
# Strategy 2: User-Based Recommendations
# --------------------------------------------

def user_based_recommendations_slow(
    data: List[Dict[str, Any]],
    user_id: int,
    k: int = 5,
    n: int = 5,
    ):
    # Build user → product → rating matrix (sparse)
    ratings = defaultdict(dict)
    for row in data:
        ratings[row["user_id"]][row["product_id"]] = row["rating"]

    if user_id not in ratings:
        raise ValueError(f"User ID {user_id} not found")

    # Compute user mean ratings
    user_means = {u: sum(r.values()) / len(r) for u, r in ratings.items()}

    # Normalize ratings (rating - user mean)
    normalized = {
        u: {p: r - user_means[u] for p, r in items.items()}
        for u, items in ratings.items()
    }

    # Compute similarities with target user
    similarities = []
    target_vector = normalized[user_id]

    for other_user, vector in normalized.items():
        if other_user == user_id:
            continue
        sim = cosine_similarity(target_vector, vector)
        similarities.append((other_user, sim))

    # Select top-k similar users
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_users = similarities[:k]

    ## temporary
    ## print(f"{similar_users=}")

    if not similar_users:
        return []

    # Predict ratings
    scores = defaultdict(float)
    sim_sums = defaultdict(float)

    for other_user, sim in similar_users:
        for product, rating in normalized[other_user].items():
            if product not in ratings[user_id]:
                scores[product] += sim * rating
                sim_sums[product] += abs(sim)

    predictions = []
    for product in scores:
        if sim_sums[product] > 0:
            pred = scores[product] / sim_sums[product] + user_means[user_id]
            predictions.append({"product_id": product, "predicted_rating": round(pred,2)})

    # Sort and return top-N
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)

    return predictions[:n]

def user_based_recommendations(
    data: List[Dict[str, Any]],
    user_id: int,
    k: int = 5,
    n: int = 5,
) -> List[Dict[str, Any]]:
    # Build user → product → rating matrix
    ratings = defaultdict(dict)
    for row in data:
        ratings[row["user_id"]][row["product_id"]] = row["rating"]

    if user_id not in ratings:
        raise ValueError(f"User ID {user_id} not found")

    # Compute user mean ratings
    user_means = {u: sum(r.values()) / len(r) for u, r in ratings.items()}

    target_mean = user_means[user_id]
    target_ratings = ratings[user_id]

    # Compute similarities on the fly and keep top-k using heapq
    similarities = []
    for other_user, r_dict in ratings.items():
        if other_user == user_id:
            continue
        # Compute normalized ratings for similarity
        sim = cosine_similarity(
            {p: rating - user_means[other_user] for p, rating in r_dict.items()},
            {p: rating - target_mean for p, rating in target_ratings.items()},
        )
        if sim > 0:
            similarities.append((sim, other_user))

    top_sim_users = nlargest(k, similarities)  # max heap for top-k

    if not top_sim_users:
        logger.warning("User ID %s don't have similar similar users found in data", user_id)
        return []

    # Predict ratings
    scores = defaultdict(float)
    sim_sums = defaultdict(float)
    target_products = set(target_ratings.keys())

    for sim, other_user in top_sim_users:
        other_ratings = ratings[other_user]
        other_mean = user_means[other_user]
        for product, rating in other_ratings.items():
            if product in target_products:
                continue
            scores[product] += sim * (rating - other_mean)
            sim_sums[product] += abs(sim)

    predictions = [
        {"product_id": product, "predicted_rating": round(scores[product]/sim_sums[product] + target_mean, 2)}
        for product in scores if sim_sums[product] > 0
    ]

    # Return top-N predicted products
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = [
        {**p, "predicted_rating": round(max(0, min(5, p["predicted_rating"])), 2)}
        for p in predictions
    ]
    return predictions[:n]


def user_based_recommendations_with_time(
    data: List[Dict[str, Any]],
    user_id: int,
    k: int = 5,
    days_tau: float = 365,  # decay time in days (default 365 days)
    n: int = 5,
) -> List[Dict[str, Any]]:
    """
    User-based recommendations with timestamp weighting.
    Older ratings are downweighted by exp(-Δt / τ)
    """
    ratings = defaultdict(dict)
    timestamps = defaultdict(dict)

    decay_tau = days_tau*24*3600

    for row in data:
        uid = row["user_id"]
        pid = row["product_id"]
        ratings[uid][pid] = row["rating"]
        timestamps[uid][pid] = row["timestamp"]  # assumed in seconds

    if user_id not in ratings:
        raise ValueError(f"User ID {user_id} not found")

    user_means = {u: sum(r.values()) / len(r) for u, r in ratings.items()}
    target_mean = user_means[user_id]
    target_ratings = ratings[user_id]
    target_times = timestamps[user_id]
    max_ts = max(max(ts.values()) for ts in timestamps.values())  # max timestamp in dataset

    # Compute recency-weighted normalized ratings for target user
    target_norm = {}
    for p, r in target_ratings.items():
        delta = max_ts - target_times[p]
        seconds = delta.total_seconds()
        weight = exp(-seconds * log(2) / decay_tau)
        target_norm[p] = (r - target_mean) * weight

    # Compute similarities
    similarities = []
    for other_user, r_dict in ratings.items():
        if other_user == user_id:
            continue
        norm_vec = {}
        for p, r in r_dict.items():
            delta = max_ts - timestamps[other_user][p]
            seconds = delta.total_seconds()
            weight = exp(-seconds * log(2) / decay_tau)
            norm_vec[p] = (r - user_means[other_user]) * weight

        sim = cosine_similarity(target_norm, norm_vec)
        if sim > 0:
            similarities.append((sim, other_user))

    top_sim_users = nlargest(k, similarities)

    if not top_sim_users:
        logger.warning("User ID %s don't have similar similar users found in data", user_id)
        return []

    # Predict ratings with timestamp weighting
    scores = defaultdict(float)
    sim_sums = defaultdict(float)
    target_products = set(target_ratings.keys())

    for sim, other_user in top_sim_users:
        for p, r in ratings[other_user].items():
            if p in target_products:
                continue
            delta = max_ts - timestamps[other_user][p]
            seconds = delta.total_seconds()
            weight = exp(-seconds * log(2) / decay_tau)
            scores[p] += sim * (r - user_means[other_user]) * weight
            sim_sums[p] += abs(sim) * weight

    predictions = [
        {"product_id": p, "predicted_rating": round(scores[p]/sim_sums[p] + target_mean, 2)}
        for p in scores if sim_sums[p] > 0
    ]

    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = [
        {**p, "predicted_rating": round(max(0, min(5, p["predicted_rating"])), 2)}
        for p in predictions
    ]
    return predictions[:n]


def main(
    path: str,
    user_id: str,
    k: int = 5,
    n: int = 5,
    type: str = 'user_based_with_time'
):
    allowed_types = ['user_based', 'user_based_with_time']

    if type not in allowed_types:
        raise ValueError(f"Invalid type '{type}'. Choose from {allowed_types}.")
    
    data_lst = load_and_clean_data(path)

    if type == 'user_based':
        recs = user_based_recommendations(data=data_lst, 
                                            user_id=user_id, 
                                            n=n, 
                                            k=k)
    else:
        recs = user_based_recommendations_with_time(
            data=data_lst, 
            user_id=user_id, 
            n=n, 
            k=k)

    return recs
