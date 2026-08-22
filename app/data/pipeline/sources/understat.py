"""understat 源适配器:xG 数据。"""
from __future__ import annotations

import logging
from typing import Any

from app.data.pipeline.config import (
    UNDERSTAT_BASE,
    FDCO_TO_UNDERSTAT,
)
from app.data.pipeline.sources.base import BaseSource
from app.data.pipeline.canonical.normalize import NormalizedMatch

logger = logging.getLogger(__name__)


class UnderstatSource(BaseSource):
    """understat xG 源适配器。"""

    SOURCE_NAME = "understat"
    DATA_TYPE = "xg"

    def fetch_raw(self, **kwargs) -> list[dict]:
        """单赛季比赛数组(dates),含 xG。"""
        league_code = kwargs.get("league_code")
        season = kwargs.get("season")
        if not league_code or not season:
            raise ValueError("必须指定 league_code 和 season")

        url = UNDERSTAT_BASE.format(league=league_code, season=season)
        from app.data.pipeline.http import HttpClient

        http = self.http or HttpClient()
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/{league_code}/{season}",
        }
        data = http.http_get_json(url=url, extra_headers=headers)
        return data.get("dates", [])

    def normalize_row(self, raw: dict) -> NormalizedMatch | None:
        """understat dates 数组元素 → NormalizedMatch(仅回填 xG)。"""
        from app.data.pipeline.canonical.normalize import cleanse_understat_row

        return cleanse_understat_row(raw, self.league_type)

    def normalize(self, raw_records: list[dict]) -> list[NormalizedMatch]:
        """覆盖:understat 只保留 isResult 的场次。"""
        results: list[NormalizedMatch] = []
        for raw in raw_records:
            if not raw.get("isResult"):
                continue
            m = self.normalize_row(raw)
            if m is not None:
                results.append(m)
        return results


def fetch_understat(league_code: str, season: int) -> list[dict]:
    """兼容旧接口:单赛季比赛数组(dates),含 xG。"""
    from app.data.pipeline.http import HttpClient

    url = UNDERSTAT_BASE.format(league=league_code, season=season)
    http = HttpClient()
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{league_code}/{season}",
    }
    data = http.http_get_json(url=url, extra_headers=headers)
    return data.get("dates", [])

