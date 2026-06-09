"""Quality metrics: cosine, separation, precision@1."""

from __future__ import annotations

from pipeline.bench.metrics import cosine, l2_normalize, precision_at_1, separation


def test_cosine_normalized():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_l2_normalize_unit_length():
    v = l2_normalize([3.0, 4.0])
    assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9


def test_separation_rewards_tight_clusters():
    vecs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    labels = ["a", "a", "b", "b"]
    intra, inter, sep = separation(vecs, labels)
    assert intra > inter
    assert sep > 0.0


def test_precision_at_1_perfect():
    vecs = [[1.0, 0.0], [0.99, 0.0], [0.0, 1.0], [0.0, 0.99]]
    labels = ["a", "a", "b", "b"]
    assert precision_at_1(vecs, labels) == 1.0


def test_precision_at_1_wrong_neighbour():
    # each artist has a single clip → nearest neighbour is always the other artist
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    labels = ["a", "b"]
    assert precision_at_1(vecs, labels) == 0.0
