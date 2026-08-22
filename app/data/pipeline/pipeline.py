"""Pipeline 主类:编排数据采集全流程。

核心流程:
    1. 根据数据类型获取(primary, fallback)源
    2. 实例化对应的数据源适配器
    3. 执行 fetch → normalize → ingest
    4. 失败时自动降级到 fallback 源
    5. 记录进度(可断点续跑)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.data.pipeline.registry import PRIMARY_FALLBACK, FREQUENCY_TYPES
from app.data.pipeline.config import STATS_SEASONS

logger = logging.getLogger(__name__)


class Pipeline:
    """数据管线主类。"""

    # 五大联赛
    LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]

    def __init__(self) -> None:
        self._source_cache: dict[str, Any] = {}

    def _get_source(self, source_name: str, league: str, **kwargs: Any) -> Any:
        """实例化数据源适配器。"""
        from app.data.pipeline.sources.bzzoiro import BzzoiroSource
        from app.data.pipeline.sources.fdco import FdcoSource
        from app.data.pipeline.sources.understat import UnderstatSource
        from app.data.pipeline.sources.api_football import ApiFootballSource, ApiFootballInjuries
        from app.data.pipeline.sources.statsbomb import StatsBombSource
        from app.data.pipeline.sources.zafronix import ZafronixSource, FdoSource

        key = f"{source_name}:{league}"
        if key not in self._source_cache:
            if source_name == "bzzoiro_events":
                self._source_cache[key] = BzzoiroSource(league, data_type="results")
            elif source_name == "bzzoiro_stats":
                self._source_cache[key] = BzzoiroSource(league, data_type="stats")
            elif source_name == "bzzoiro_odds":
                self._source_cache[key] = BzzoiroSource(league, data_type="odds")
            elif source_name == "fdco":
                self._source_cache[key] = FdcoSource(league)
            elif source_name == "understat":
                self._source_cache[key] = UnderstatSource(league)
            elif source_name == "api_football":
                self._source_cache[key] = ApiFootballSource(league)
            elif source_name == "api_football_injuries":
                self._source_cache[key] = ApiFootballInjuries(league)
            elif source_name == "statsbomb":
                self._source_cache[key] = StatsBombSource(league)
            elif source_name == "zafronix":
                self._source_cache[key] = ZafronixSource(league)
            elif source_name == "fdo":
                self._source_cache[key] = FdoSource(league)
        return self._source_cache.get(key)

    def run(self, data_type: str, league: str | None = None, **kwargs: Any) -> dict:
        """执行单个数据类型的采集。

        Args:
            data_type: 数据类型(results/stats/odds/xg/tournaments/injuries)
            league: 联赛类型(None 则采集所有五大联赛)
            **kwargs: 额外参数(season, date, since_year 等)

        Returns:
            采集结果统计
        """
        primary, fallback = PRIMARY_FALLBACK.get(data_type, (None, None))
        if primary is None:
            return {"error": f"未知数据类型: {data_type}"}

        leagues = [league] if league else self.LEAGUES
        total_result = {
            "data_type": data_type,
            "leagues": leagues,
            "results": [],
            "success": False,
        }

        for lg in leagues:
            result = self._run_source(primary, lg, **kwargs)
            total_result["results"].append(result)

            # primary 失败且有 fallback
            if not result.get("success") and fallback:
                logger.warning("[%s] %s failed, fallback to %s", lg, primary, fallback)
                fb_result = self._run_source(fallback, lg, **kwargs)
                total_result["results"].append(fb_result)

        total_result["success"] = any(r.get("success") for r in total_result["results"])
        return total_result

    def _run_source(self, source_name: str, league: str, **kwargs: Any) -> dict:
        """执行单个数据源。"""
        source = self._get_source(source_name, league)
        if source is None:
            return {"source": source_name, "league": league, "success": False, "error": "source not found"}

        try:
            return source.run(**kwargs)
        except Exception as e:
            logger.error("[%s] %s error: %s", league, source_name, e)
            return {"source": source_name, "league": league, "success": False, "error": str(e)}

    def run_frequency(self, freq: str) -> dict:
        """按频次执行采集。

        Args:
            freq: 频次(realtime/daily/weekly/monthly)

        Returns:
            采集结果统计
        """
        data_types = FREQUENCY_TYPES.get(freq, [])
        result = {"freq": freq, "data_types": data_types, "results": []}
        for dt in data_types:
            r = self.run(dt)
            result["results"].append(r)
        return result

    def run_all(self, leagues: list[str] | None = None) -> dict:
        """全量采集所有数据类型。"""
        result = {}
        for data_type in PRIMARY_FALLBACK:
            result[data_type] = self.run(data_type, league=leagues[0] if leagues else None)
        return result
