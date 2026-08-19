"""CanonicalMatchResolver 测试(唯一 Match Identity 域不变量)。"""

from __future__ import annotations

from datetime import date, datetime

from app.data.canonical.resolver import CanonicalMatchResolver


class _M:
    def __init__(self, hid, home, away, d):
        self.id = hid
        self.home_team = home
        self.away_team = away
        self.match_date = datetime.combine(d, datetime.min.time())


def _mk():
    return [
        _M(1, "Arsenal FC", "Chelsea FC", date(2026, 3, 10)),
        _M(2, "Liverpool FC", "AFC Bournemouth", date(2026, 3, 14)),
    ]


def test_same_orientation_resolves():
    r = CanonicalMatchResolver().index_matches(_mk())
    res = r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 10))
    assert res.match.id == 1 and res.orientation == "SAME"


def test_reversed_orientation_resolves():
    r = CanonicalMatchResolver().index_matches(_mk())
    # 来源以 B vs A 视角 → REVERSED 命中同一条 canonical
    res = r.resolve("Chelsea FC", "Arsenal FC", date(2026, 3, 11))
    assert res.match.id == 1 and res.orientation == "REVERSED"


def test_name_suffix_and_absent():
    r = CanonicalMatchResolver().index_matches(_mk())
    # 无后缀 + 无匹配 → None
    res = r.resolve("Manchester City", "Newcastle", date(2026, 3, 1))
    assert res.match is None
    # 后缀差异(AFC Bournemouth vs Bournemouth)同队
    res2 = r.resolve("Liverpool", "Bournemouth", date(2026, 3, 14))
    assert res2.match.id == 2


def test_day_tolerance():
    r = CanonicalMatchResolver(day_tolerance=1).index_matches(_mk())
    assert r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 11)).match.id == 1
    assert r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 20)).match is None
