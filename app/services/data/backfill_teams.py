"""数据回填:队名 → teams 实体,matches 外键,team_seasons 赛季归属。

从现有 matches 队名建立球队实体表(幂等,可重复执行):
1. 每队名建一条 teams 记录(含中文名)
2. matches.home_team_id/away_team_id 回填
3. 推导 team_seasons(球队 × 联赛 × 赛季)
4. leagues.comp_type 标注(世界杯/欧冠=cup,五大联赛=league)

用法:
    python scripts/backfill_teams.py
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cli import run, setup_logging


def _season_label(dt) -> str:
    """足球赛季标签:8 月起为新赛季(2026-08 → 2026-2027)"""
    year = dt.year
    return f"{year}-{year + 1}" if dt.month >= 8 else f"{year - 1}-{year}"


def main() -> int:
    setup_logging("INFO")
    from app.api.db import League, Match, Team, TeamSeason, db, init_db, session_scope
    from app.data.canonical.team_names_zh import to_zh

    init_db()
    with session_scope():
        # 1. 建立 teams(从 matches 队名;幂等)
        names = set()
        for (name,) in db.session.query(Match.home_team).distinct():
            names.add(name)
        for (name,) in db.session.query(Match.away_team).distinct():
            names.add(name)
        created = 0
        for name in sorted(names):
            if db.session.get(Team, name) is None:
                # Team 主键用 name?设计是 id 主键,按 name 查
                existing = Team.query.filter_by(name=name).first()
                if existing is None:
                    db.session.add(
                        Team(name=name, name_zh=to_zh(name), team_type="club")
                    )
                    created += 1
        db.session.commit()
        print(f"✅ teams: 共 {len(names)} 队名,新建 {created}")

        # 2. matches 回填 team_id
        team_by_name = {t.name: t.id for t in Team.query.all()}
        filled = 0
        for m in db.session.query(Match).filter(Match.home_team_id.is_(None)):
            m.home_team_id = team_by_name.get(m.home_team)
            m.away_team_id = team_by_name.get(m.away_team)
            filled += 1
        db.session.commit()
        print(f"✅ matches 回填 {filled} 条外键")

        # 3. team_seasons 推导(球队×联赛×赛季)
        combos = defaultdict(set)
        for team_id, league_id, date in (
            db.session.query(Match.home_team_id, Match.league_id, Match.match_date)
            .filter(Match.home_team_id.isnot(None))
            .all()
        ):
            if date:
                combos[(team_id, league_id)].add(_season_label(date))
        for (team_id, league_id), seasons in combos.items():
            for s in seasons:
                exists = TeamSeason.query.filter_by(
                    team_id=team_id, league_id=league_id, season=s
                ).first()
                if exists is None:
                    db.session.add(
                        TeamSeason(team_id=team_id, league_id=league_id, season=s)
                    )
        db.session.commit()
        print(
            f"✅ team_seasons: {len(combos)} 个球队-联赛组合,{sum(len(v) for v in combos.values())} 条赛季记录"
        )

        # 4. leagues.comp_type 标注(五大联赛=league,欧冠/世界杯/欧洲杯=cup)
        for lg in League.query.all():
            if lg.league_type in (
                "premier_league",
                "la_liga",
                "bundesliga",
                "serie_a",
                "ligue_1",
            ):
                lg.comp_type = "league"
            else:
                lg.comp_type = "cup"
        db.session.commit()
        print("✅ leagues.comp_type 标注完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
