"""StatsBomb 开放数据源适配器。

历史 xG 补充,数据只到 2020/21 赛季。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from app.data.pipeline.config import STATSBOMB_BASE
from app.data.pipeline.sources.base import BaseSource

logger = logging.getLogger(__name__)


class StatsBombSource(BaseSource):
    SOURCE_NAME = "statsbomb"

    def fetch_raw(self, **kwargs) -> list[dict]:
        """获取 statsbomb 数据。"""
        competition_id = kwargs.get("competition_id")
        season_id = kwargs.get("season_id")
        if not competition_id or not season_id:
            return []
        url = f"{STATSBOMB_BASE}/competitions/{competition_id}/seasons/{season_id}/events.json"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            logger.error("statsbomb fetch failed: %s", e)
            return []

    def normalize_row(self, raw: dict):
        return None
