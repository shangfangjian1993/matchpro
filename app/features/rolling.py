"""滚动特征原语(

提供:出场长表构建 + expanding/rolling 原语;家族模块(attack_defense/form/h2h)
在此之上实现各自 compute,由 factory 统一调度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_long_table(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
 """构建队伍出场长表(每行 = 一支队伍的一次出场)+ 原始 df(含 index 列)。

 long 列:index/team/gf/ga/win/gd/points/date(如有)/side/row_id(按时间排序)。
 """
 prepared = data.copy()
 df = prepared.reset_index(drop=True).reset_index()

 parts = []
 for side, team_col, gf_col, ga_col in (
 ("home", "home_team", "home_goals", "away_goals"),
 ("away", "away_team", "away_goals", "home_goals"),
 ):
 part = df[["index", team_col, gf_col, ga_col]].copy()
 part["side"] = side
 part = part.rename(columns={team_col: "team", gf_col: "gf", ga_col: "ga"})
 part["win"] = np.where(
 part["gf"].isna() | part["ga"].isna(),
 np.nan,
 (part["gf"] > part["ga"]).astype(float),
 )
 part["gd"] = part["gf"] - part["ga"]
 part["points"] = np.where(
 part["win"].isna(),
 np.nan,
 np.where(
 part["win"] == 1, 3.0, np.where(part["gf"] == part["ga"], 1.0, 0.0)
 ),
 )
 if "date" in df.columns:
 part["date"] = df["date"].values
 parts.append(part)

 long = pd.concat(parts, ignore_index=True)
 long["row_id"] = np.arange(len(long))
 if "date" in long.columns:
 long = long.sort_values(["date", "index"], kind="mergesort")
 else:
 long = long.sort_values("index", kind="mergesort")
 return long, df


def expanding_prior(long: pd.DataFrame, col: str, out: str) -> np.ndarray:
 """组内 expanding 均值,再后移一位排除本场;按 (team, row_id) 回填。"""
 s = long.groupby("team")[col].expanding().mean()
 s = s.groupby(level=0).shift(1)
 stat = s.reset_index()
 stat.columns = ["team", "row_idx", out]
 return long.merge(
 stat, left_on=["team", "row_id"], right_on=["team", "row_idx"], how="left"
 )[out].values


def valid_rolling(
 long: pd.DataFrame, col: str, window: int, agg: str, side: str | None = None
) -> np.ndarray:
 """仅对有效值行做 rolling(比分 NaN 预测行不占窗口),按 (team,row_id) 回填。"""
 sub = long if side is None else long[long["side"] == side]
 valid = sub.dropna(subset=[col])
 if valid.empty:
 return np.full(len(long), np.nan)
 roll = valid.groupby("team")[col].rolling(window, min_periods=1)
 s = roll.sum() if agg == "sum" else roll.mean()
 s = s.groupby(level=0).shift(1)
 stat = s.reset_index()
 stat.columns = ["team", "row_idx", "out"]
 return long.merge(
 stat, left_on=["team", "row_id"], right_on=["team", "row_idx"], how="left"
 )["out"].values
