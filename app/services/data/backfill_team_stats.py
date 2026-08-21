"""回填 team_match_stats:从 matches 现有指标列生成每队每场统计。

幂等:已存在的 (match_id, side) 更新,缺失插入。

用法:
 python scripts/backfill_team_stats.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cli import run, setup_logging

_FIELDS = [
 "xg",
 "shots",
 "shots_on_target",
 "corners",
 "possession",
 "yellow_cards",
 "red_cards",
 "ht_goals",
 "passing_accuracy",
 "xg_chain",
 "efficiency",
 "transition_speed",
 "defensive_actions",
 "counter_attacks",
 "tactical_rating",
 "experience",
]


def main() -> int:
 setup_logging("INFO")
 from app.api.db import Match, TeamMatchStats, db, init_db, session_scope
 from app.data.canonical.ingest import _write_team_stats

 init_db()
 with session_scope():
 # 需要 team_id 的比赛(有外键)
 matches = Match.query.filter(Match.home_team_id.isnot(None)).all()
 done = 0
 for m in matches:
 _write_team_stats(db, m)
 done += 1
 if done % 5000 == 0:
 db.session.commit()
 db.session.commit()
 n = db.session.query(TeamMatchStats).count()
 print(f"✅ 回填 {done} 场比赛指标 → team_match_stats 共 {n} 条")
 return 0


if __name__ == "__main__":
 raise SystemExit(run(main))
