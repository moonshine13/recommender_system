"""
Distance and similarity metrics.

This module provides functions to compute similarity or distance between users or items,
such as cosine similarity.
"""

import math
from typing import Dict


def cosine_similarity(u: Dict[str, float], v: Dict[str, float]) -> float:
    """
    Cosine similarity between two sparse vectors
    """
    common_keys = u.keys() & v.keys()
    numerator = sum(u[k] * v[k] for k in common_keys)

    norm_u = math.sqrt(sum(x * x for x in u.values()))
    norm_v = math.sqrt(sum(x * x for x in v.values()))

    if norm_u == 0 or norm_v == 0:
        return 0.0

    return numerator / (norm_u * norm_v)

