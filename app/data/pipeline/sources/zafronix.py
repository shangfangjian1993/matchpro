"""Zafronix 数据源适配器。

欧战/国家队历史数据采集。
"""
from __future__ import annotations

import json
import logging
import urllib.request

from app.data.pipeline.config import ZAFRONIX_BASE
from app.data.pipeline.sources.base import BaseSource

logger = logging.getLogger(__name__)


class ZafronixSource(BaseSource):
    SOURCE_NAME = "zafronix"

    def fetch_raw(self, **kwargs) -> list[dict]:
        """获取 zafronix 欧战/国家队数据。"""
        endpoint = kwargs.get("endpoint", "/events")
        params = kwargs.get("params", {})
        import urllib.parse
        url = ZAFRONIX_BASE + endpoint
        if params:
            url += "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
                return data.get("results", data) if isinstance(data, dict) else data
        except Exception as e:
            logger.error("zafronix fetch failed: %s", e)
            return []

    def normalize_row(self, raw: dict):
        return None


class FdoSource(BaseSource):
    SOURCE_NAME = "fdo"

    def fetch_raw(self, **kwargs) -> list[dict]:
        """football-data.org (fdo) 赛程数据。"""
        import os
        key = os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
        if not key:
            return []
        competition = kwargs.get("competition")
        season = kwargs.get("season")
        import urllib.parse
        url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
        params = {}
        if season:
            params["season"] = season
        url += "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"X-Auth-Token": key})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
                return data.get("matches", [])
        except Exception as e:
            logger.error("fdo fetch failed: %s", e)
            return []

    def normalize_row(self, raw: dict):
        return None
