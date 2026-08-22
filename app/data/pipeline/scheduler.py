"""频次调度器:将采集任务分派到对应的 cron job。"""

from __future__ import annotations

import logging
from typing import Any

from app.data.pipeline.pipeline import Pipeline
from app.data.pipeline.registry import FREQUENCY_TYPES

logger = logging.getLogger(__name__)


def run_daily(pipeline: Pipeline | None = None, **kwargs: Any) -> dict:
    """执行每日采集:赛果 + xG + 伤停。"""
    pipe = pipeline or Pipeline()
    return pipe.run_frequency("daily")


def run_weekly(pipeline: Pipeline | None = None, **kwargs: Any) -> dict:
    """执行每周采集:深度统计 + 收盘赔率。"""
    pipe = pipeline or Pipeline()
    return pipe.run_frequency("weekly")


def run_monthly(pipeline: Pipeline | None = None, **kwargs: Any) -> dict:
    """执行每月采集:欧战/国家队历史补全。"""
    pipe = pipeline or Pipeline()
    return pipe.run_frequency("monthly")


def run_all(pipeline: Pipeline | None = None, **kwargs: Any) -> dict:
    """执行全量采集(所有频次)。"""
    pipe = pipeline or Pipeline()
    result = {}
    for freq in FREQUENCY_TYPES:
        result[freq] = pipe.run_frequency(freq)
    return result


def run_pipeline(pipeline: Pipeline | None = None, **kwargs: Any) -> dict:
    """入口函数,供 auto_sync.py 调用。"""
    freq = kwargs.get("freq", "daily")
    if freq == "daily":
        return run_daily(pipeline, **kwargs)
    elif freq == "weekly":
        return run_weekly(pipeline, **kwargs)
    elif freq == "monthly":
        return run_monthly(pipeline, **kwargs)
    elif freq == "all":
        return run_all(pipeline, **kwargs)
    else:
        return {"ok": False, "error": f"Unknown freq: {freq}"}


if __name__ == "__main__":
    import sys
    freq = sys.argv[1] if len(sys.argv) > 1 else "daily"
    result = run_pipeline(freq=freq)
    print(result)
