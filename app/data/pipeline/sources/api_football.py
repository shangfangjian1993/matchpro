"""api-football (api-sports.io) 数据源适配器。

提供 stats (比赛统计) 和 injuries (伤停) 数据。
free 套餐 100 次/天限速。
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

from app.data.pipeline.config import (
    API_FOOTBALL_BASE,
    API_FOOTBALL_INTERVAL,
    API_FOOTBALL_LEAGUE_IDS,
    API_FOOTBALL_STAT_MAP,
    API_FOOTBALL_TEAM_ALIAS,
)
from app.data.pipeline.sources.base import BaseSource, SourceResult

logger = logging.getLogger(__name__)


def _key() -> str:
    k = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not k:
        from app.core.paths import PROJECT_ROOT
        try:
            with open(os.path.join(str(PROJECT_ROOT), ".env"), encoding="utf-8") as f:
                for line in f:
                    if line.startswith("API_FOOTBALL_KEY"):
                        k = line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass
    return k


def _get(path: str, params: dict | None = None) -> dict:
    key = _key()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY 未设置")
    url = API_FOOTBALL_BASE + path
    if params:
        import urllib.parse
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    for attempt in range(3):
        req = urllib.request.Request(url, headers={
            "x-apisports-key": key,
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10)
                continue
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("api-football 请求失败")


def _num(v):
    if v is None:
        return None
    s = str(v).strip().replace("%", "")
    return float(s) if s else None


def _norm_team(name: str) -> str:
    if not name:
        return ""
    n = str(name).lower()
    for alias, canonical in API_FOOTBALL_TEAM_ALIAS.items():
        if alias in n:
            return canonical
    return n


class ApiFootballSource(BaseSource):
    SOURCE_NAME = "api_football"

    def fetch_raw(self, **kwargs) -> list[dict]:
        league_id = API_FOOTBALL_LEAGUE_IDS.get(self.league_type)
        if league_id is None:
            return []
        date = kwargs.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
        all_stats = []
        page = 1
        while True:
            d = _get("/fixtures", {
                "league": league_id,
                "date": date,
                "page": page,
                "season": kwargs.get("season", datetime.utcnow().year),
            })
            fixtures = d.get("response", [])
            if not fixtures:
                break
            for f in fixtures:
                fid = f.get("fixture", {}).get("id")
                if fid:
                    stat = _get(f"/fixtures/statistics", {"fixture": fid})
                    all_stats.append(stat)
                time.sleep(API_FOOTBALL_INTERVAL)
            page += 1
            if page > 10:
                break
        return all_stats

    def normalize_row(self, raw: dict):
        return None  # api-football stats 直接写入,不经过 matches 表


class ApiFootballInjuries(BaseSource):
    SOURCE_NAME = "api_football_injuries"

    def fetch_raw(self, **kwargs) -> list[dict]:
        league_id = API_FOOTBALL_LEAGUE_IDS.get(self.league_type)
        if league_id is None:
            return []
        date = kwargs.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
        d = _get("/injuries", {
            "league": league_id,
            "date": date,
        })
        return d.get("response", [])

    def normalize_row(self, raw: dict):
        return None
