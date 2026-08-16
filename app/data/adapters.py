"""数据适配(审查 §14/§39 拆分:自 data_adapter 迁移)。

职责:matches_to_dataframe(ORM → 特征 DataFrame)+ 联赛枚举解析。
"""
from __future__ import annotations

import pandas as pd

from app.core.config import MATCH_METRIC_COLUMNS, LeagueType


def _resolve_league_type(league_type: str) -> LeagueType:
    """把前端传的字符串(如 PREMIER_LEAGUE 或 premier_league)解析为 LeagueType"""
    if not league_type:
        raise ValueError("缺少 league_type 参数")
    try:
        return LeagueType(league_type)      # 按枚举值(小写)匹配
    except ValueError:
        try:
            return LeagueType[league_type]  # 按枚举名(大写)匹配
        except KeyError:
            raise ValueError(f"不支持的联赛类型: {league_type}")


def matches_to_dataframe(matches, league_name: str | None = None,
                          league_season: str | None = None) -> pd.DataFrame:
    """把 matches 查询结果转成模型训练/预测所需的 DataFrame(含指标列)。

    league_name/league_season 可显式传入:避免每行访问 m.league 的 N+1 lazy load
    (27k 行 → 1 次)。
    """
    if league_name is None and matches:
        # 兼容旧调用:只读第一行的 league(其余行不再触发 lazy load)
        try:
            league_name = matches[0].league.name if matches[0].league else ""
            league_season = matches[0].league.season if matches[0].league else ""
        except Exception:
            league_name, league_season = "", ""
    rows = []
    for m in matches:
        row = {
            "date": m.match_date,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "goals": m.home_goals,
            "league": league_name or "",
            "season": league_season or "",
        }
        for col in MATCH_METRIC_COLUMNS:
            row[col] = getattr(m, col, None)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" in df.columns and not df["date"].isna().all():
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df

