from __future__ import annotations

from pipeline.block_tag import SEED, add, remove


class _FakeConn:
    """Captures (sql, params) of the last execute — enough to assert normalization."""

    def __init__(self):
        self.calls: list[tuple] = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        return self  # so .rowcount works on the returned value

    rowcount = 1


def test_add_normalizes_tag_to_lowercase_trimmed():
    c = _FakeConn()
    add(c, "  CDMX  ", "location")
    assert c.calls[-1][1] == ("cdmx", "location")


def test_remove_normalizes_tag():
    c = _FakeConn()
    remove(c, "Mexico City")
    assert c.calls[-1][1] == ("mexico city",)


def test_seed_is_lowercase_and_covers_reported_leaks():
    tags = {t for t, _r in SEED}
    assert tags == {t.lower() for t in tags}  # all normalized
    # the leaks that motivated this (Equinoxious cdmx/mexico, HoH san francisco)
    assert {"cdmx", "mexico", "san francisco", "oakland"} <= tags
