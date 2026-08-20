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


def test_resolver_ambiguity_detection():
    """同一天多候选时标记 ambiguous。"""
    from app.data.canonical.resolver import CanonicalMatchResolver

    # 同一对球队,同一天(不同比赛)
    matches = [
        _M(1, "Arsenal FC", "Chelsea FC", date(2026, 3, 10)),
        _M(2, "Arsenal FC", "Chelsea FC", date(2026, 3, 10)),
    ]
    r = CanonicalMatchResolver(day_tolerance=1).index_matches(matches)
    # 解析 3/10 → 两个候选都 exact date,分数接近,应标记 ambiguous
    res = r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 10))
    assert res.match is not None
    assert res.ambiguous is True


def test_resolver_exact_date_preferred():
    """精确日期匹配优先于 ±1 天。"""
    from app.data.canonical.resolver import CanonicalMatchResolver

    matches = [
        _M(1, "Arsenal FC", "Chelsea FC", date(2026, 3, 10)),
        _M(2, "Arsenal FC", "Chelsea FC", date(2026, 3, 12)),
    ]
    r = CanonicalMatchResolver(day_tolerance=1).index_matches(matches)
    # 解析 3/12 → 应选 id=2(exact),不选 id=1(±1)
    res = r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 12))
    assert res.match.id == 2
    assert res.confidence == 0.99
    assert res.ambiguous is False


def test_resolver_confidence_levels():
    """Confidence 分级:exact date > ±1 day。"""
    from app.data.canonical.resolver import CanonicalMatchResolver

    matches = [_M(1, "Arsenal FC", "Chelsea FC", date(2026, 3, 10))]
    r = CanonicalMatchResolver(day_tolerance=1).index_matches(matches)
    # exact date
    res_exact = r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 10))
    assert res_exact.confidence == 0.99
    # ±1 day
    res_near = r.resolve("Arsenal FC", "Chelsea FC", date(2026, 3, 11))
    assert res_near.confidence == 0.90
