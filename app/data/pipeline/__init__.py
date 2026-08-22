"""数据管线主入口。

使用方式:
    from app.data.pipeline import Pipeline

    pipe = Pipeline()
    # 采集赛果
    pipe.run("results", "premier_league")
    # 采集 xG
    pipe.run("xg", "premier_league")
    # 采集 stats
    pipe.run("stats", "premier_league", season=2025)
    # 全量采集
    pipe.run_all()
"""
from __future__ import annotations

from app.data.pipeline.pipeline import Pipeline

__all__ = ["Pipeline"]
