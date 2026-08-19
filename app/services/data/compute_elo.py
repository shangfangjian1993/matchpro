"""计算并写回球队 ELO 评分(teams.elo_rating)。

- 俱乐部 ELO:五大联赛 + 欧冠统一演化(欧冠提供跨联赛校准)
- 国家队 ELO:世界杯 + 欧洲杯统一演化(中立场,无主场优势)
- 按时间顺序逐场更新(防泄漏:rating 始终反映赛前实力)
- 幂等:重跑结果一致(确定性计算)

用法:
    python scripts/compute_elo.py
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.cli import run, setup_logging

# 联赛归属:俱乐部赛事 / 国家队赛事
_CLUB_LEAGUES = {
    "premier_league",
    "la_liga",
    "bundesliga",
    "serie_a",
    "ligue_1",
    "champions_league",
}
_NATIONAL_LEAGUES = {"world_cup", "european_championship"}


def main() -> int:
    setup_logging("INFO")
    from app.api.db import League, Match, Team, db, init_db, session_scope
    from app.models.elo_goal.rating import EloSystem

    init_db()
    with session_scope():
        # 1. 读取比赛,按时间排序
        rows = (
            db.session.query(
                Match.league_id,
                Match.home_team,
                Match.away_team,
                Match.home_goals,
                Match.away_goals,
                Match.match_date,
            )
            .filter(
                Match.match_status == "finished",
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            )
            .all()
        )
        league_type = {lg.id: lg.league_type for lg in League.query.all()}

        club_games, national_games = [], []
        for lid, h, a, hg, ag, dt in rows:
            lt = league_type.get(lid, "")
            game = (dt, h, a, hg, ag)
            if lt in _CLUB_LEAGUES:
                club_games.append(game)
            elif lt in _NATIONAL_LEAGUES:
                national_games.append(game)
        club_games.sort(key=lambda g: g[0])
        national_games.sort(key=lambda g: g[0])
        print(
            f"俱乐部比赛: {len(club_games)} 场 | 国家队比赛: {len(national_games)} 场"
        )

        # 2. 演化(三维度:overall/attack/defense 独立实例)
        # attack/defense 维度用低 K(连续回归式,防膨胀)
        club_elo = EloSystem()
        club_att, club_def = EloSystem(k=8.0), EloSystem(k=8.0)
        for dt, h, a, hg, ag in club_games:
            ha = club_elo.home_advantage
            club_elo.update(h, a, hg, ag, home_adv=ha)
            club_att.update(h, a, hg, ag, home_adv=ha, mode="attack")
            club_def.update(h, a, hg, ag, home_adv=ha, mode="defense")
        nat_elo = EloSystem()
        nat_att, nat_def = EloSystem(k=8.0), EloSystem(k=8.0)
        for dt, h, a, hg, ag in national_games:
            nat_elo.update(h, a, hg, ag, home_adv=0.0, is_national=True)
            nat_att.update(h, a, hg, ag, home_adv=0.0, is_national=True, mode="attack")
            nat_def.update(h, a, hg, ag, home_adv=0.0, is_national=True, mode="defense")

        # 3. 写回 teams
        now = datetime.now(tz=timezone.utc)
        updated = 0
        for t in Team.query.all():
            if t.team_type == "national":
                elo, att, deff = nat_elo, nat_att, nat_def
            elif t.team_type == "club":
                elo, att, deff = club_elo, club_att, club_def
            else:
                continue
            rating = elo.rating(t.name)
            if rating != 1500.0:  # 未参赛球队保持空
                t.elo_rating = round(rating, 1)
                t.attack_elo = round(att.rating(t.name), 1)
                t.defense_elo = round(deff.rating(t.name), 1)
                t.elo_updated_at = now
                updated += 1
        db.session.commit()
        print(f"✅ 已写回 {updated} 支球队 ELO")

        # 4. 验证:Top 10
        top = (
            db.session.query(Team)
            .filter(Team.elo_rating.isnot(None))
            .order_by(Team.elo_rating.desc())
            .limit(10)
            .all()
        )
        print("\n=== Top 10 球队 ELO ===")
        for t in top:
            print(f"  {t.name_zh or t.name:<10} {t.elo_rating:.0f} ({t.team_type})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
