"""fdo 源适配器:football-data.org 赛程/赛果 API。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.data.pipeline.http import http_get

logger = logging.getLogger(__name__)

FDO_BASE = "https://api.football-data.org/v4"

# fdo 联赛代码 → LeagueType
LEAGUE_MAP_FDO = {
    "PL": "premier_league",
    "PD": "la_liga",
    "BL1": "bundesliga",
    "SA": "serie_a",
    "FL1": "ligue_1",
    "CL": "champions_league",
    "EL": "europa_league",
    "WC": "world_cup",
    "EC": "euro",
}


def _key() -> str:
    """获取 FDO API Key。"""
    env = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
    if env:
        return env
    try:
        from app.core.paths import PROJECT_ROOT
        with open(os.path.join(PROJECT_ROOT, ".env"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("FOOTBALL_DATA_ORG_KEY"):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def available() -> bool:
    """FDO API Key 是否可用。"""
    return bool(_key())


def fetch_fdo(league_code: str, season: int) -> list[dict]:
    """单赛季全部比赛(含未来赛程)。"""
    if not available():
        logger.warning("FDO: 未设置 FOOTBALL_DATA_ORG_KEY")
        return []
    
    url = f"{FDO_BASE}/competitions/{league_code}/matches?season={season}"
    try:
        data = json.loads(http_get(url, headers={"X-Auth-Token": _key()}))
        return data.get("matches", [])
    except Exception as e:
        logger.error("FDO fetch failed %s/%s: %s", league_code, season, e)
        return []
