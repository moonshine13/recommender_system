import argparse
import sys
from typing import List, Dict,Any
from datetime import timedelta
from heapq import nlargest
from collections import defaultdict

from src.data.read_and_clean_data import load_and_clean_data
# ----------------------------
# Strategy 1: Top-N Products
# ----------------------------

def top_n_products_slow(
    data: List[Dict[str, Any]],
    days: int = 365,
    n: int = 5,
    min_ratings: int = 10,
):
    # Find latest timestamp in database
    max_ts = max(row["timestamp"] for row in data)
    cutoff = max_ts - timedelta(days=days)

    # Filter recent records
    recent = (row for row in data if row["timestamp"] >= cutoff)

    # Aggregate ratings per product
    agg = {}
    for row in recent:
        pid = row["product_id"]
        rating = row["rating"]

        if pid not in agg:
            agg[pid] = {"sum": 0.0, "count": 0}

        agg[pid]["sum"] += rating
        agg[pid]["count"] += 1

    # Compute averages and apply min_ratings filter
    results = []
    for pid, stats in agg.items():
        if stats["count"] >= min_ratings:
            results.append(
                {
                    "product_id": pid,
                    "avg_rating": round(stats["sum"] / stats["count"],2),
                    "count": stats["count"],
                }
            )

    # Sort by average rating (descending)
    results.sort(key=lambda x: x["avg_rating"], reverse=True)

    # Return top N
    return results[:n]


def top_n_products(
    data: List[Dict[str, Any]],
    days: int = 365,
    n: int = 5,
    min_ratings: int = 10,
):
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


def main(
    path: str,
    days: int = 365,
    min_ratings: int = 10,
    n: int = 5,
):
    data_lst = load_and_clean_data(path)

    top_n = top_n_products(
        data_lst,
        days=days,
        n=n,
        min_ratings=min_ratings
    )
    return top_n
