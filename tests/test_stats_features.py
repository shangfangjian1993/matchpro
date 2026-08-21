"""审查 ae724d5 P1-6:stats 特征防泄漏与按球队隔离测试。"""


class _FakeMatch:
    def __init__(self, id_, home, away):
        self.id = id_
        self.home_team = home
        self.away_team = away


class _FakeStats:
    def __init__(
        self,
        fouls=None,
        tackles=None,
        offsides=None,
        interceptions=None,
        blocked_shots=None,
        total_saves=None,
        shots_inside_box=None,
        shots_outside_box=None,
        big_chances=None,
    ):
        self.fouls = fouls
        self.tackles = tackles
        self.offsides = offsides
        self.interceptions = interceptions
        self.blocked_shots = blocked_shots
        self.total_saves = total_saves
        self.shots_inside_box = shots_inside_box
        self.shots_outside_box = shots_outside_box
        self.big_chances = big_chances


def _per_match(*items):
    """items: (match_id, side, stats) → per_match dict。"""
    pm = {}
    for mid, side, st in items:
        pm.setdefault(mid, {})[side] = st
    return pm


def test_match_does_not_use_own_stats():
    """本场 stats 不得进入本场特征(先取特征后更新)。"""
    from app.features.stats_features import _roll

    hs = _FakeStats(fouls=10)
    hist = [_FakeMatch(1, "A", "B"), _FakeMatch(2, "A", "B")]
    pm = _per_match(
        (1, "home", hs),
        (1, "away", _FakeStats(fouls=5)),
        (2, "home", _FakeStats(fouls=99)),
        (2, "away", _FakeStats(fouls=2)),
    )
    out = _roll(hist, pm)
    # 第 2 场 home_tms_fouls_avg_5 只能来自第 1 场 A 的 home=10(不能含本场 99)
    assert out[2]["home_tms_fouls_avg_5"] == 10.0
    # 第 1 场(无前史)→ home_tms None
    assert out[1]["home_tms_fouls_avg_5"] is None


def test_team_a_not_into_team_b():
    """Team A 的 stats 不得进入 Team B 的滚动特征。"""
    from app.features.stats_features import _roll

    hist = [
        _FakeMatch(1, "A", "B"),  # A 主场 fouls=10
        _FakeMatch(2, "C", "A"),  # A 客场 fouls=3
        _FakeMatch(3, "C", "D"),  # C 主场,D 客场(均无 A 历史)
    ]
    pm = _per_match(
        (1, "home", _FakeStats(fouls=10)),
        (1, "away", _FakeStats(fouls=1)),
        (2, "home", _FakeStats(fouls=7)),
        (2, "away", _FakeStats(fouls=3)),
        (3, "home", _FakeStats(fouls=6)),
        (3, "away", _FakeStats(fouls=2)),
    )
    out = _roll(hist, pm)
    # 第 3 场 home 是 C:home 侧前史 = 第 2 场 C 主场 fouls=7(而非 A 的 10)
    assert out[3]["home_tms_fouls_avg_5"] == 7.0
    # 第 3 场 away 是 D(无前史)→ None;不应出现 A 的 3
    assert out[3]["away_tms_fouls_avg_5"] is None


def test_home_away_side_separated():
    """同一队主/客场两侧独立(主场统计不进客场特征)。"""
    from app.features.stats_features import _roll

    hist = [_FakeMatch(1, "A", "B"), _FakeMatch(2, "B", "A")]
    pm = _per_match(
        (1, "home", _FakeStats(fouls=10)),
        (1, "away", _FakeStats(fouls=5)),
        (2, "home", _FakeStats(fouls=99)),
        (2, "away", _FakeStats(fouls=3)),
    )
    out = _roll(hist, pm)
    # 第 2 场 home=B:B 在历史只以"客场"出赛(第 1 场 away,fouls=5 记入 B 客场侧)
    # → B 的"主场侧"前史为空 → home_tms None(客场统计不进入主场特征)
    assert out[2]["home_tms_fouls_avg_5"] is None
    # 第 2 场 away=A:A 主场侧有 10,但客场侧前史为空 → away_tms None
    assert out[2]["away_tms_fouls_avg_5"] is None


def test_opponent_adjusted():
    """对手调整:归入(主队前史 − 客队客场前史)。"""
    from app.features.stats_features import _roll

    hist = [_FakeMatch(1, "A", "B"), _FakeMatch(2, "A", "B")]
    pm = _per_match(
        (1, "home", _FakeStats(fouls=10)),
        (1, "away", _FakeStats(fouls=3)),
        (2, "home", _FakeStats(fouls=99)),
        (2, "away", _FakeStats(fouls=1)),
    )
    out = _roll(hist, pm)
    assert out[2]["home_tms_fouls_rel"] == 7.0  # 10 - 3(客队B客场前史)


def test_group_feature():
    """分组聚合:defensive 组 = 各列前史均值的平均。"""
    from app.features.stats_features import _roll

    hist = [_FakeMatch(1, "A", "B"), _FakeMatch(2, "A", "B")]
    s = {
        "fouls": 10,
        "tackles": 5,
        "offsides": 1,
        "interceptions": 7,
        "blocked_shots": 2,
        "total_saves": 3,
        "shots_inside_box": 8,
        "shots_outside_box": 2,
        "big_chances": 1,
    }
    pm = {}
    pm[1] = {"home": _FakeStats(**s), "away": _FakeStats(**dict(s, fouls=3))}
    pm[2] = {
        "home": _FakeStats(**dict(s, fouls=99)),
        "away": _FakeStats(**dict(s, fouls=1)),
    }
    out = _roll(hist, pm)
    # defensive 组: interceptions7 blocked2 saves3 fouls(主队前史10) → (7+2+3+10)/4
    assert out[2]["home_tms_grp_defensive_avg"] == 5.5


def test_multi_window():
    """多窗口:同列输出 avg_3 / avg_5(3 场前史时两窗口均值相同)。"""
    from app.features.stats_features import _roll

    hist = [_FakeMatch(1, "A", "B"), _FakeMatch(2, "A", "B")]
    pm = _per_match(
        (1, "home", _FakeStats(fouls=10)),
        (1, "away", _FakeStats(fouls=2)),
        (2, "home", _FakeStats(fouls=99)),
        (2, "away", _FakeStats(fouls=1)),
    )
    out = _roll(hist, pm, windows=(1, 3))
    assert "home_tms_fouls_avg_1" in out[2]
    assert "home_tms_fouls_avg_3" in out[2]
    assert out[2]["home_tms_fouls_avg_1"] == 10.0
    assert out[2]["home_tms_fouls_avg_3"] == 10.0


# ── L2:公开 API(rolling_team_stats)特征必须存活(审查 P0-2)──
def test_stats_features_survive_public_api(db_ctx):
    """公开接口必须输出 _roll 的全部新特征(avg/ewm/rel/group)。"""
    from app.api.db import League, Match, init_db
    from app.features.stats_features import rolling_team_stats

    init_db()
    lg = League.query.filter_by(league_type="premier_league").first()
    if lg is None:
        return
    hist = (
        Match.query.filter_by(league_id=lg.id, match_status="finished")
        .order_by(Match.match_date.desc())
        .limit(40)
        .all()
    )[::-1]
    df = rolling_team_stats(hist, windows=(5,))
    assert not df.empty
    cols = set(df.columns)
    # 关键列族必须存在
    assert "home_tms_fouls_avg_5" in cols
    assert "home_tms_fouls_ewm" in cols
    assert "home_tms_fouls_rel" in cols
    assert "home_tms_grp_defensive_avg" in cols
    # 全部列必须以 home_tms_/away_tms_ 开头(契约)
    assert all(col.startswith(("home_tms_", "away_tms_")) for col in cols)


def test_stats_features_family_meta():
    """族元信息:REQUIRED_HISTORY 声明(hist_limit 契约用)。"""
    from app.features.stats_features import REQUIRED_HISTORY

    assert isinstance(REQUIRED_HISTORY, int) and REQUIRED_HISTORY > 0


def test_overall_rolling_windows():
    """审查 A70A601 P1-8:overall(全历史)滚动与 side 滚动并存。"""
    from app.features.stats_features import _roll

    fm1 = _FakeMatch(1, "AA", "BB")
    sA = _FakeStats(fouls=6, tackles=40, offsides=1)
    sB = _FakeStats(fouls=4, tackles=30, offsides=2)
    mapping = _roll([fm1], {1: {"home": sA, "away": sB}}, windows=(5, 3))
    out = mapping[1]
    assert "home_tms_fouls_overall_5" in out
    assert "away_tms_fouls_overall_5" in out
    assert "home_tms_fouls_overall_3" in out
    assert all(k.startswith(("home_tms_", "away_tms_")) for k in out)
