"""fdco 源适配器:football-data.co.uk 历史 CSV。"""
from __future__ import annotations

import csv
import io
import logging

from app.data.pipeline.config import (
    FDCO_BASE,
    FDCO_COLUMN_MAP,
    fdco_season_code,
)
from app.data.pipeline.sources.base import BaseSource
from app.data.pipeline.canonical.normalize import NormalizedMatch

logger = logging.getLogger(__name__)


class FdcoSource(BaseSource):
    """football-data.co.uk CSV 源适配器。"""

    SOURCE_NAME = "fdco"
    DATA_TYPE = "results"

    def fetch_raw(self, **kwargs) -> list[dict]:
        """抓取单赛季单联赛 CSV → 原始行 dict 列表(不做清洗)。"""
        season_code = kwargs.get("season_code") or fdco_season_code(
            kwargs.get("season", 2025)
        )
        league_code = kwargs.get("league_code")
        if not league_code:
            raise ValueError("必须指定 league_code")

        url = FDCO_BASE.format(season=season_code, league=league_code)
        from app.data.pipeline.http import HttpClient

        http = self.http or HttpClient()
        text = http.http_get(url=url).decode("utf-8-sig", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))

    def normalize_row(self, raw: dict) -> NormalizedMatch | None:
        """CSV 行 → NormalizedMatch。"""
        from app.data.pipeline.canonical.normalize import cleanse_fdco_row

        return cleanse_fdco_row(raw, self.league_type)


def fetch_fdco(season_code: str, league_code: str) -> list[dict]:
    """兼容旧接口:单赛季单联赛 CSV → 原始行 dict 列表(不做清洗)。"""
    from app.data.pipeline.http import HttpClient

    url = FDCO_BASE.format(season=season_code, league=league_code)
    http = HttpClient()
    text = http.http_get(url=url).decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))

