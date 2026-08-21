"""match_source_records 关系表测试。"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.api.db import League, Match, db, init_db, session_scope
from app.data.canonical.lineage import (
    consensus_ratio,
    get_conflicts,
    get_match_lineage,
    record_source,
)


@pytest.mark.db
def test_lineage_write_and_query(db_ctx):
    """写入来源谱系 → 查询 → 共识计算。"""
    init_db()
    with session_scope():
        league = League.query.filter_by(league_type="premier_league").first()
        if league is None:
            league = League(league_type="premier_league", name="Premier League")
            db.session.add(league)
            db.session.flush()

        m = Match(
            league_id=league.id,
            home_team="Arsenal FC",
            away_team="Chelsea FC",
            match_date=datetime(2026, 3, 10, tzinfo=timezone.utc),
            match_status="finished",
            home_goals=2,
            away_goals=1,
        )
        db.session.add(m)
        db.session.flush()

        record_source(match_id=m.id, source="bzzoiro", home_goals=2, away_goals=1)
        record_source(match_id=m.id, source="fdco", home_goals=2, away_goals=1)
        record_source(match_id=m.id, source="unknown", home_goals=1, away_goals=2, status="conflict")

        lineage = get_match_lineage(m.id)
        assert len(lineage) == 3

        consensus = consensus_ratio(m.id, canonical_home=2, canonical_away=1)
        assert consensus["agree"] == 2
        assert consensus["total"] == 3
        assert consensus["ratio"] == 2 / 3
        assert "bzzoiro" in consensus["agree_sources"]
        assert "unknown" in consensus["disagree_sources"]

        conflicts = get_conflicts(m.id)
        assert len(conflicts) == 1
        assert conflicts[0].source == "unknown"


@pytest.mark.db
def test_lineage_idempotent(db_ctx):
    """相同内容不重复写入(幂等)。"""
    init_db()
    with session_scope():
        league = League.query.filter_by(league_type="premier_league").first()
        if league is None:
            league = League(league_type="premier_league", name="Premier League")
            db.session.add(league)
            db.session.flush()

        m = Match(
            league_id=league.id,
            home_team="Liverpool FC",
            away_team="Man City",
            match_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
            match_status="finished",
            home_goals=1,
            away_goals=1,
        )
        db.session.add(m)
        db.session.flush()

        record_source(match_id=m.id, source="bzzoiro", home_goals=1, away_goals=1)
        record_source(match_id=m.id, source="bzzoiro", home_goals=1, away_goals=1)

        lineage = get_match_lineage(m.id)
        assert len(lineage) == 1
