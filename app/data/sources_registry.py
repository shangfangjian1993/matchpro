"""数据源注册表(深度整理:首选 → 降级;实效性分级)。

需求来源:按数据完整性/质量/免费限制对每个数据类型排序联选,备用源仅
在主源失败/数据缺失时启用;并按数据实效性安排采集频次。

数据类型与源优先级:
  results  赛果/比分      primary=bzzoiro_events(全量+最新)  fallback=fdco(校验)
  stats    深度统计(近20季) primary=bzzoiro_stats            fallback=api_football
  odds     收盘赔率(近20季) primary=bzzoiro_odds             fallback=None
  xg       期望进球        primary=understat                 fallback=statsbomb(历史)
  tournaments 欧战/国家队  primary=zafronix                  fallback=fdo(赛程)
  injuries 伤停            primary=api_football              fallback=None

实效性频次:
  realtime  比赛进行中实时(live)—— 可选
  daily     每日:赛果、xG、伤停(比赛结束当日可入库)
  weekly    每周:深度统计/收盘赔率增量(近 20 季窗口)
  monthly   每月:欧战/国家队历史补全、新赛季全量 merge
"""
from __future__ import annotations


# 数据类型 → (首选, 降级)
PRIMARY_FALLBACK = {
    "results": ("bzzoiro_events", "fdco"),
    "stats": ("bzzoiro_stats", "api_football"),
    "odds": ("bzzoiro_odds", None),
    "xg": ("understat", "statsbomb"),
    "tournaments": ("zafronix", "fdo"),
    "injuries": ("api_football", None),
}

# 实效性:freq → [数据类型]
FREQUENCY_TYPES = {
    "realtime": ["live"],
    "daily": ["results", "xg", "injuries"],
    "weekly": ["stats", "odds"],
    "monthly": ["tournaments"],
}

# 采集窗口(赛季数)—— stats/odds 仅近 N 季(用户策略)
STATS_SEASONS = 20
# 训练窗口(训练只用最近 N 季;数据全量保留)
TRAIN_SEASONS = 10
