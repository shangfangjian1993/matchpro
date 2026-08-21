"""谱系记录:写入/查询 match_source_records 表。

字段说明:
- hash: SHA256(content)[:16],用于幂等检测(检测重复记录)


并发注意:当前使用 query-then-insert 模式。
PostgreSQL 部署时应改为 INSERT ... ON CONFLICT DO NOTHING,
或捕获 IntegrityError 后查询已有记录。


统一接口,替代对 source_scores_json 的直接操作。
"""
from __future__ import annotations

import hashlib
from datetime import datetime

from app.api.db import MatchSourceRecord, db, utcnow


def _compute_hash(source: str, home_goals, away_goals, home_ht_goals, away_ht_goals) -> str:
    """内容哈希(检测重复记录)。
    
    返回: SHA256(content)[:16] (16 hex chars)。
    DB 字段: hash VARCHAR(64) — 足够容纳完整 SHA256。
    """
    content = f"{source}:{home_goals}:{away_goals}:{home_ht_goals}:{away_ht_goals}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def record_source(
    match_id: int,
    source: str,
    home_goals: int | None,
    away_goals: int | None,
    home_ht_goals: int | None = None,
    away_ht_goals: int | None = None,
    orientation: str = "SAME",
    external_id: str | None = None,
    available_at: datetime | None = None,
    status: str = "active",
) -> MatchSourceRecord:
    """记录一条来源快照(幂等:相同 hash 不重复写入)。

    返回 (record, created) — created=True 表示新写入。
    """
    content_hash = _compute_hash(source, home_goals, away_goals, home_ht_goals, away_ht_goals)
    
    # 查重
    existing = (
        db.session.query(MatchSourceRecord)
        .filter_by(match_id=match_id, source=source, hash=content_hash)
        .first()
    )
    if existing is not None:
        return existing
    
    rec = MatchSourceRecord(
        match_id=match_id,
        source=source,
        external_id=str(external_id) if external_id else None,
        home_goals=home_goals,
        away_goals=away_goals,
        home_ht_goals=home_ht_goals,
        away_ht_goals=away_ht_goals,
        orientation=orientation,
        available_at=available_at,
        ingested_at=utcnow(),
        status=status,
        hash=content_hash,
    )
    db.session.add(rec)
    return rec


def get_match_lineage(match_id: int) -> list[MatchSourceRecord]:
    """获取比赛的所有来源记录(按 ingested_at 排序)。"""
    return (
        db.session.query(MatchSourceRecord)
        .filter_by(match_id=match_id)
        .order_by(MatchSourceRecord.ingested_at.asc())
        .all()
    )


def get_conflicts(match_id: int) -> list[MatchSourceRecord]:
    """获取标记为 conflict 的来源记录。"""
    return (
        db.session.query(MatchSourceRecord)
        .filter_by(match_id=match_id, status="conflict")
        .all()
    )


def consensus_ratio(match_id: int, canonical_home: int | None, canonical_away: int | None) -> dict:
    """计算来源共识(基于关系表)。

    返回 {"agree_sources":[...],"disagree_sources":[...],"agree":n,"total":m,"ratio":r}。
    """
    records = get_match_lineage(match_id)
    if not records:
        return {"agree": 0, "total": 0, "ratio": 0.0, "agree_sources": [], "disagree_sources": []}
    
    agree, disagree = [], []
    for rec in records:
        if rec.home_goals == canonical_home and rec.away_goals == canonical_away:
            agree.append(rec.source)
        else:
            disagree.append(rec.source)
    
    return {
        "agree": len(agree),
        "total": len(records),
        "ratio": len(agree) / len(records) if records else 0.0,
        "agree_sources": agree,
        "disagree_sources": disagree,
    }
