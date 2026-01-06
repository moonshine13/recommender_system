import math
from typing import Dict, List, Any
from collections import defaultdict

def cosine_similarity(u: Dict[str, float], v: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors"""
    common_keys = u.keys() & v.keys()
    numerator = sum(u[k] * v[k] for k in common_keys)

    norm_u = math.sqrt(sum(x * x for x in u.values()))
    norm_v = math.sqrt(sum(x * x for x in v.values()))

    if norm_u == 0 or norm_v == 0:
        return 0.0

    return numerator / (norm_u * norm_v)


def compute_norms(vectors):
    return {
        u: math.sqrt(sum(v * v for v in vec.values()))
        for u, vec in vectors.items()
    }

def build_user_vectors(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Build sparse user-rating vectors from rating records.

    Returns
    -------
    dict[user_id, dict[item_id, rating]]
    """
    user_vectors = defaultdict(dict)

    for r in rows:
        u = r["user_id"]
        i = r["product_id"]
        rating = float(r["rating"])

        user_vectors[u][i] = rating

    return user_vectors

def cosine_similarity2(u, v, norm_u, norm_v):
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    if len(u) > len(v):
        u, v = v, u

    dot = 0.0
    for k, val in u.items():
        v_val = v.get(k)
        if v_val is not None:
            dot += val * v_val

    return dot / (norm_u * norm_v) if dot else 0.0


def main():
    ## TODO: cosine similarity_v2
    from data.read_and_clean_data import load_data
    rows = load_data(path="./data/ratings.csv")
    user_vectors = build_user_vectors(rows)
    user_norms = compute_norms(user_vectors)

    ## Hcosine_similarity(user_vectors["671"], user_vectors["670"])
    cosine_similarity2(
        user_vectors["671"],
        user_vectors["670"],
        user_norms["671"],
        user_norms["670"]
    )
