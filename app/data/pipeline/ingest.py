"""入库核心:NormalizedMatch → matches 表(upsert + 幂等 + 进度持久化)。

幂等语义(一次成型,不再事后修补):
 - 键:(league_id, home_team, away_team, 日期)
 - 已存在:非空字段覆盖更新(scheduled 升级为 finished 时补比分);
   若新记录全部字段为 None(纯 xG 回填),只补空字段
 - 不存在:插入
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.data.pipeline.canonical.normalize import NormalizedMatch
from app.data.pipeline.canonical.reconcile import reconcile

logger = logging.getLogger(__name__)

# 进度持久化路径
PROGRESS_PATH = os.environ.get("PIPELINE_PROGRESS_PATH", "/opt/data/pipeline_progress.json")


def _load_progress() -> dict:
    """加载采集进度。"""
    try:
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_progress(progress: dict) -> None:
    """保存采集进度。"""
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_progress(data_type: str, league: str, season: int) -> dict:
    """获取采集进度。"""
    progress = _load_progress()
    key = f"{data_type}/{league}/{season}"
    return progress.get(key, {"status": "pending", "offset": 0, "updated": 0})


def set_progress(data_type: str, league: str, season: int, status: str, offset: int = 0, updated: int = 0) -> None:
    """更新采集进度。"""
    progress = _load_progress()
    key = f"{data_type}/{league}/{season}"
    progress[key] = {
        "status": status,
        "offset": offset,
        "updated": updated,
        "last_run": datetime.utcnow().isoformat(),
    }
    _save_progress(progress)


def upsert_matches(
    records: list[NormalizedMatch],
    source: str = "unknown",
) -> dict:
    """批量 upsert matches 记录。

    Args:
        records: 清洗后的 NormalizedMatch 列表
        source: 来源标识

    Returns:
        {"inserted": int, "updated": int, "skipped": int, "errors": list}
    """
    from app.api.db import Match, db, init_db, session_scope

    init_db()

    result = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}

    with session_scope():
        for nm in records:
            try:
                from app.api.db import League
                league = League.query.filter_by(league_type=nm.league_type).first()
                if league is None:
                    result["skipped"] += 1
                    continue

                # 查找现有记录
                existing = Match.query.filter_by(
                    league_id=league.id,
                    home_team=nm.home_team,
                    away_team=nm.away_team,
                    match_date=nm.date,
                ).first()

                if existing is None:
                    # 插入新记录
                    row_data = nm.to_row()
                    row_data["league_id"] = league.id
                    match = Match(**row_data)
                    db.session.add(match)
                    result["inserted"] += 1
                else:
                    # 更新现有记录(reconcile)
                    updates = reconcile(existing, nm.to_row(), source)
                    for k, v in updates.items():
                        setattr(existing, k, v)
                    if updates:
                        result["updated"] += 1
                    else:
                        result["skipped"] += 1
            except Exception as e:
                result["errors"].append(f"{nm.home_team} vs {nm.away_team}: {e}")
                logger.error("upsert error: %s", e)

    return result


def upsert_team_stats(
    match_id: int,
    team_id: int,
    side: str,
    data: dict,
) -> dict:
    """批量 upsert team_match_stats 记录。

    Returns:
        {"inserted": int, "updated": int, "skipped": int}
    """
    from app.api.db import TeamMatchStats, db, init_db, session_scope

    init_db()
    result = {"inserted": 0, "updated": 0, "skipped": 0}

    with session_scope():
        row = TeamMatchStats.query.filter_by(match_id=match_id, side=side).first()
        if row is None:
            row = TeamMatchStats(match_id=match_id, team_id=team_id, side=side, **data)
            db.session.add(row)
            result["inserted"] += 1
        else:
            for k, v in data.items():
                setattr(row, k, v)
            result["updated"] += 1

    return result
