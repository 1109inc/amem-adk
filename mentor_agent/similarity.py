from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        raise ValueError(f"Embedding size mismatch: {len(a)} != {len(b)}")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)