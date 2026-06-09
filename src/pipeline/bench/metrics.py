"""Embedding-quality metrics: cosine, same/cross-artist separation, precision@1."""

from __future__ import annotations

import math


def l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two already-L2-normalized vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def separation(vectors: list[list[float]], labels: list[str]) -> tuple[float, float, float]:
    """Return (mean intra-artist cosine, mean inter-artist cosine, intra - inter)."""
    norm = [l2_normalize(v) for v in vectors]
    intra: list[float] = []
    inter: list[float] = []
    n = len(norm)
    for i in range(n):
        for j in range(i + 1, n):
            c = cosine(norm[i], norm[j])
            (intra if labels[i] == labels[j] else inter).append(c)
    mi = sum(intra) / len(intra) if intra else 0.0
    me = sum(inter) / len(inter) if inter else 0.0
    return mi, me, mi - me


def precision_at_1(vectors: list[list[float]], labels: list[str]) -> float:
    """Fraction of clips whose nearest *other* clip shares its artist label."""
    norm = [l2_normalize(v) for v in vectors]
    n = len(norm)
    if n < 2:
        return 0.0
    hits = 0
    for i in range(n):
        best_c = -2.0
        best_j = -1
        for j in range(n):
            if i == j:
                continue
            c = cosine(norm[i], norm[j])
            if c > best_c:
                best_c = c
                best_j = j
        if best_j >= 0 and labels[best_j] == labels[i]:
            hits += 1
    return hits / n
