import csv
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
## from libs.read_and_clean_data import load_data

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_all_data(file_path):
    rows = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=["user_id", "product_id", "rating", "timestamp"]
        )
        next(reader)  # skip header

        for row in reader:
            row["rating"] = float(row["rating"])
            row["timestamp"] = int(row["timestamp"])
            rows.append(row)
            ## try:
            ##     row["rating"] = float(row["rating"])
            ##     row["timestamp"] = datetime.fromtimestamp(int(row["timestamp"]))
            ##     rows.append(row)
            ## except (ValueError, TypeError):
            ##     continue
    return rows


# ------------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------------

def analyze_missing(rows):
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

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_histogram_timestamps(rows):
    timestamps = [r["timestamp"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.hist(timestamps, bins=20)
    plt.xlabel("Time")
    plt.ylabel("Number of Ratings")
    plt.title("Ratings Over Time")
    plt.show()


def plot_rating_counts(rows):
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
    user_counts = Counter(str(r["user_id"]) for r in rows)

    plt.figure(figsize=(8, 5))
    plt.bar(user_counts.keys(), user_counts.values())
    plt.xlabel("User ID")
    plt.ylabel("Number of Ratings")
    plt.title("Ratings per User")
    plt.show()


def plot_ratings_per_product(rows):
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


# ------------------------------------------------------------------
# Data quality checks
# ------------------------------------------------------------------

def check_zero_timestamps(rows):
    zeros = [r for r in rows if r["timestamp"] == 0]

    print("Rows with timestamp 0:")
    for r in zeros:
        print(r)

    non_zero = [r["timestamp"] for r in rows if r["timestamp"] != 0]
    if non_zero:
        print("Timestamp 2nd Min:", min(non_zero))
    print("")

def check_invalid_ratings(rows):
    invalid = [r for r in rows if r["rating"] in (99, -1)]
    print("Invalid ratings:")
    for r in invalid:
        print(r)
    print("Number of invalid ratings:", len(invalid))
    print("")

def check_duplicates(rows):
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

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ## TODO: add number of valid ratings per user
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
