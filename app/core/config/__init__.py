"""app.core.config —— 配置包(统一 re-export,保持旧 import 兼容)。

审查 P2-2:core/config.py(God Config)拆分:
- league.py    LeagueType 联赛类型枚举
- model.py     ModelConfig / MultiLeagueConfig / config 全局实例
- loader.py    load_yaml() YAML 配置单一加载入口
- features.py  feature_flags() 特征开关单一入口

外部仍以 from app.core.config import LeagueType, ... 使用,无需改动调用方。
"""
from __future__ import annotations

from app.core.config.features import feature_flags
from app.core.config.league import LeagueType
from app.core.config.loader import load_yaml
from app.core.config.model import ModelConfig, MultiLeagueConfig, config
from app.core.constants import (
    MATCH_METRIC_COLUMNS,
    TOURNAMENT_LEAGUE_TYPES,
)

__all__ = [
    "LeagueType",
    "ModelConfig",
    "MultiLeagueConfig",
    "config",
    "load_yaml",
    "feature_flags",
    "MATCH_METRIC_COLUMNS",
    "TOURNAMENT_LEAGUE_TYPES",
]
