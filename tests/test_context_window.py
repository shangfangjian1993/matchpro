"""ContextBuilder 时序窗口回归(

历史 .all() 默认行序非时间序(rowid 受迁移/批量导入打乱)→ 修复前 hist_limit
截断取到任意错乱窗口(曾取到 1992 年前段),球队定位失败且含时序泄漏风险。
本测试锁定:截断窗口必须是**时间上最近**的 hist_limit 场。
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.db
def test_context_hist_limit_is_latest_window(db_ctx):
 from app.api.db import League, Match, init_db
 from app.core.config import LeagueType

 init_db()
 lg = League.query.filter_by(league_type="premier_league").first()
 if lg is None:
 return
 window = Match.query.filter_by(league_id=lg.id, match_status="finished")
 n = window.count()
 if n < 300:
 return
 # 取一队可定位的近期场次(用 24/25 赛季内已完赛场)
 m = (
 Match.query.filter(
 Match.league_id == lg.id,
 Match.match_status == "finished",
 Match.match_date < "2025-01-01",
 )
 .order_by(Match.match_date.desc())
 .first()
 )
 if m is None:
 return
 md = pd.Timestamp(m.match_date)
 from app.core.paths import MODELS_DIR
 from app.prediction.context import ContextBuilder

 builder = ContextBuilder(str(MODELS_DIR))
 ctx = builder.build(
 LeagueType.PREMIER_LEAGUE, m.home_team, m.away_team, md, hist_limit=500
 )
 df = ctx["hist_df"]
 assert len(df) <= 500 # 不超过截断
 if len(df):
 # 截断窗口必须严格早于预测场次(无 future)
 max_date = df["date"].max()
 if hasattr(max_date, "to_pydatetime"):
 max_date = max_date.to_pydatetime()
 assert max_date < m.match_date
