import numpy as np

# Leave-last-out split
def leave_last_out_split(ratings):
    ratings = ratings[np.argsort(ratings[:, 3])]  # sort by time
    train_list = []
    test_list = []
    for u in np.unique(ratings[:, 0]):
        user_ratings = ratings[ratings[:, 0] == u]
        if len(user_ratings) > 1:
            train_list.append(user_ratings[:-1])
            test_list.append(user_ratings[-1:])
        else:
            train_list.append(user_ratings)
    train_ratings = np.vstack(train_list)
    test_ratings = np.vstack(test_list)
    return train_ratings, test_ratings


def normalize_time(ts, t_min, t_max):
    t = ts.year + ts.month / 12.0
    return (t - t_min) / (t_max - t_min)


def preprocess_data(data):
    user_map = {}
    item_map = {}
    user_counter = 0
    item_counter = 0

    timestamps = np.array([row['timestamp'].year + row['timestamp'].month / 12.0 for row in data])
    t_min = timestamps.min()
    t_max = timestamps.max()
    processed_data = []
    for row in data:##[:10000]:
        u = row['user_id']
        i = row['product_id']
        r = float(row['rating'])
        t = normalize_time(row['timestamp'], t_min, t_max)
        if u not in user_map:
            user_map[u] = user_counter
            user_counter += 1

        if i not in item_map:
            item_map[i] = item_counter
            item_counter += 1

        processed_data.append(
            (user_map[u], item_map[i], r, t)
        )
    len(processed_data)
    ratings_array = np.array(processed_data, dtype=float)

    # Ensure user/item indices are integers
    ratings_array[:, 0] = ratings_array[:, 0].astype(int)
    ratings_array[:, 1] = ratings_array[:, 1].astype(int)
    return ratings_array, user_map, item_map, t_min, t_max

def main():
    from data.read_and_clean_data import load_and_clean_data
    data = load_and_clean_data(path = "./data/ratings.csv")

    ratings_array, user_map, item_map, t_min, t_max = preprocess_data(data)

    train_ratings, test_ratings = leave_last_out_split(ratings_array)
    print("Train Ratings:\n", len(train_ratings))
    print("Test Ratings:\n", len(test_ratings))
    print("Users Map:\n", len(user_map))
    print("Items Map:\n", len(item_map))
    print("Min Timestamp:\n", len(t_min))
    print("Max Timestamp:\n", len(t_max))
