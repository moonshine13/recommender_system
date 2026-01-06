"""
Descriptive statistics utilities for exploratory data analysis (EDA).

This module provides functions to:
- Compute summary statistics for numeric and categorical data.
- Analyze distributions, counts, and basic aggregates.
- Support data exploration before building recommendation models.
"""

import argparse
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

from src.data.read_and_clean_data import load_all_data


# Analysis helpers
def analyze_missing(rows):
    """
    Analyze missings
    """
    print("Missing values per column:")
    counts = Counter()
    for r in rows:
        for k, v in r.items():
            if v is None or v == "":
                counts[k] += 1
    for k in ["user_id", "product_id", "rating", "timestamp"]:
        print(f"{k}: {counts[k]}")
    print("")


def basic_stats(rows):
    """
    Calculate basic descriptive stats
    """
    ratings = [float(r["rating"]) for r in rows]
    timestamps = [int(r["timestamp"]) for r in rows]

    print("Statistical summary:")
    print(f"Rating min: {min(ratings)}")
    print(f"Rating max: {max(ratings)}")
    print(f"Rating mean: {sum(ratings) / len(ratings):.4f}")
    print(f"Total ratings: {len(ratings)}")
    print(f"Timestamp min: {min(timestamps)}")
    print(f"Timestamp max: {max(timestamps)}")
    print("")


# Plots
def plot_histogram_timestamps(rows):
    """
    Show timestamps histogram
    """
    timestamps = [r["timestamp"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.hist(timestamps, bins=20)
    plt.xlabel("Time")
    plt.ylabel("Number of Ratings")
    plt.title("Ratings Over Time")
    plt.show()


def plot_rating_counts(rows):
    """
    Show ratings counts
    """
    rating_counts = Counter(str(r["rating"]) for r in rows)
    ratings = sorted(rating_counts.keys())
    counts = [rating_counts[r] for r in ratings]
    plt.figure(figsize=(8, 5))
    plt.bar(ratings, counts)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Count of Each Rating")
    plt.show()


def plot_ratings_per_user(rows):
    """
    Show ratings per user counts
    """
    user_counts = Counter(str(r["user_id"]) for r in rows)

    plt.figure(figsize=(8, 5))
    plt.bar(user_counts.keys(), user_counts.values())
    plt.xlabel("User ID")
    plt.ylabel("Number of Ratings")
    plt.title("Ratings per User")
    plt.show()


def plot_ratings_per_product(rows):
    """
    Show ratings per product counts
    """
    product_counts = Counter(r["product_id"] for r in rows)
    top_100 = product_counts.most_common(100)

    products, counts = zip(*top_100)

    plt.figure(figsize=(10, 5))
    plt.bar(products, counts)
    plt.xlabel("Product ID")
    plt.ylabel("Number of Ratings")
    plt.title("Top Most Rated Products")
    plt.xticks(rotation=90)
    plt.show()


# Data quality checks
def check_zero_timestamps(rows):
    """
    Check rows with 0 timestamps
    """
    zeros = [r for r in rows if r["timestamp"] == 0]

    print("Rows with timestamp 0:")
    for r in zeros:
        print(r)

    non_zero = [r["timestamp"] for r in rows if r["timestamp"] != 0]
    if non_zero:
        print("Timestamp 2nd Min:", min(non_zero))
    print("")


def check_invalid_ratings(rows):
    """
    Check rows with invalid ratings
    """
    invalid = [r for r in rows if r["rating"] in (99, -1)]
    print("Invalid ratings:")
    for r in invalid:
        print(r)
    print("Number of invalid ratings:", len(invalid))
    print("")


def check_duplicates(rows):
    """
    Check duplicated ratings
    """
    seen = defaultdict(list)
    for r in rows:
        key = (r["user_id"], r["product_id"])
        seen[key].append(r)

    duplicates = [v for v in seen.values() if len(v) > 1]

    print("Duplicate user-product ratings:")
    for group in duplicates:
        for r in group:
            print(r)
    print("")


def main():
    """
    Run EDA
    """
    parser = argparse.ArgumentParser(description="CLI Exploratory Data Analysis")
    parser.add_argument("--file", type=str, default="./data/ratings.csv")
    args = parser.parse_args()

    rows = load_all_data(args.file)

    analyze_missing(rows)
    basic_stats(rows)
    plot_histogram_timestamps(rows)
    check_zero_timestamps(rows)
    plot_rating_counts(rows)
    plot_ratings_per_user(rows)
    plot_ratings_per_product(rows)
    check_invalid_ratings(rows)
    check_duplicates(rows)


if __name__ == "__main__":
    main()
