""": MAX_GOALS 可配置化。

支持按联赛配置不同的最大进球数,适应高进球联赛(如德甲)的尾部建模需求。
"""
from __future__ import annotations

# 默认最大进球数(全球统一)
DEFAULT_MAX_GOALS = 10

# 联赛特定配置(高进球联赛可配置更大的值)
LEAGUE_MAX_GOALS: dict[str, int] = {
 "bundesliga": 12, # 德甲进球较多,扩展尾部
 "premier_league": 10,
 "la_liga": 10,
 "serie_a": 10,
 "ligue_1": 10,
}


def get_max_goals(league_type: str = "") -> int:
 """获取联赛特定的最大进球数。"""
 return LEAGUE_MAX_GOALS.get(league_type.lower(), DEFAULT_MAX_GOALS)
