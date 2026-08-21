"""多联赛足球预测模型系统 — 联赛类型定义。


单一职责;模型超参(model.py)、YAML 加载(loader.py)、特征开关
(features.py)各自独立,__init__.py 统一 re-export 保持旧 import 兼容。
"""

from __future__ import annotations

from enum import Enum


class LeagueType(Enum):
 """联赛类型枚举"""

 PREMIER_LEAGUE = "premier_league"
 LA_LIGA = "la_liga"
 BUNDESLIGA = "bundesliga"
 LIGUE_1 = "ligue_1"
 SERIE_A = "serie_a"
 CHINESE_SUPER = "chinese_super"
 WORLD_CUP = "world_cup"
 EUROPEAN_CHAMPIONSHIP = "european_championship"
 CHAMPIONS_LEAGUE = "champions_league"
 EUROPA_LEAGUE = "europa_league"
