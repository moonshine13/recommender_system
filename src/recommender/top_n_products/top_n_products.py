"""
Compute top-N products based on user ratings and activity.

This module provides functions to:
- Aggregate ratings over a specified period.
- Filter products by minimum number of ratings.
- Compute and return the top-N products based on average ratings.
"""

from collections import defaultdict
from datetime import timedelta
from heapq import nlargest
from typing import Any, Dict, List

from src.data.read_and_clean_data import load_and_clean_data


def top_n_products(
    data: List[Dict[str, Any]],
    days: int = 365,
    n: int = 5,
    min_ratings: int = 10,
):
    """
    Compute the top-N products based on recent ratings.

    Args:
        data (list of dict): List of ratings with keys "user_id", "product_id", "rating", "timestamp".
        days (int, optional): Number of past days to consider. Defaults to 365.
        n (int, optional): Number of top products to return. Defaults to 5.
        min_ratings (int, optional): Minimum number of ratings a product must have to be considered. Defaults to 10.

    Returns:
        list of dict: Top-N products with keys "product_id", "avg_rating", and "count",
                      sorted by descending average rating.
    """
    if not data:
        return []

    # Find latest timestamp
    max_ts = max(row["timestamp"] for row in data)
    cutoff = max_ts - timedelta(days=days)

    # Aggregate ratings per product in a single pass
    agg = defaultdict(lambda: [0.0, 0])  # [sum, count]
    for row in data:
        if row["timestamp"] >= cutoff:
            pid = row["product_id"]
            agg[pid][0] += row["rating"]
            agg[pid][1] += 1

    # Filter by min_ratings and compute averages
    filtered = (
        {"product_id": pid, "avg_rating": agg_val[0] / agg_val[1], "count": agg_val[1]}
        for pid, agg_val in agg.items()
        if agg_val[1] >= min_ratings
    )

    # Use heapq to get top N efficiently
    top_n = nlargest(n, filtered, key=lambda x: x["avg_rating"])

    # Round the avg_rating
    for item in top_n:
        item["avg_rating"] = round(item["avg_rating"], 2)

    return top_n


def top_n_products_run(
    path: str,
    days: int = 365,
    min_ratings: int = 10,
    n: int = 5,
):
    """
    Compute top-N products from a CSV file of ratings.

    Args:
        path (str): Path to the CSV file containing ratings.
        days (int, optional): Number of past days to consider. Defaults to 365.
        min_ratings (int, optional): Minimum number of ratings a product must have. Defaults to 10.
        n (int, optional): Number of top products to return. Defaults to 5.

    Returns:
        list of dict: Top-N products with keys "product_id", "avg_rating", and "count".
    """
    data_lst = load_and_clean_data(path)

    top_n = top_n_products(data_lst, days=days, n=n, min_ratings=min_ratings)
    return top_n
