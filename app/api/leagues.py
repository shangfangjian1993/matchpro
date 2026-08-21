"""V2 联赛端点(新设计)"""

from fastapi import APIRouter, HTTPException

from app.api.db import League, Match, db
from app.data.canonical.team_names_zh import to_zh

router = APIRouter(prefix="/api/leagues", tags=["leagues"])


@router.get("")
def list_leagues():
 leagues = League.query.order_by(League.id).all()
 return {"items": [l.to_dict() for l in leagues], "total": len(leagues)}


@router.get("/{league_id}")
def get_league(league_id: int):
 league = db.session.get(League, league_id)
 if league is None:
 raise HTTPException(404, "联赛不存在")
 return league.to_dict()


@router.get("/{league_id}/teams")
def league_teams(league_id: int):
 if db.session.get(League, league_id) is None:
 raise HTTPException(404, "联赛不存在")
 home_teams = {
 r[0]
 for r in db.session.query(Match.home_team)
 .filter_by(league_id=league_id)
 .distinct()
 .all()
 }
 away_teams = {
 r[0]
 for r in db.session.query(Match.away_team)
 .filter_by(league_id=league_id)
 .distinct()
 .all()
 }
 teams = sorted(home_teams | away_teams)
 return {
 "items": [{"name": t, "name_zh": to_zh(t)} for t in teams],
 "total": len(teams),
 }


@router.get("/{league_id}/standings")
def league_standings(league_id: int):
 if db.session.get(League, league_id) is None:
 raise HTTPException(404, "联赛不存在")
 rows = (
 db.session.query(
 Match.home_team, Match.away_team, Match.home_goals, Match.away_goals
 )
 .filter_by(league_id=league_id, match_status="finished")
 .all()
 )
 table = {}
 for home_team, away_team, home_goals, away_goals in rows:
 for team, gf, ga in (
 (home_team, home_goals, away_goals),
 (away_team, away_goals, home_goals),
 ):
 row = table.setdefault(
 team,
 {
 "played": 0,
 "win": 0,
 "draw": 0,
 "loss": 0,
 "gf": 0,
 "ga": 0,
 "points": 0,
 },
 )
 row["played"] += 1
 row["gf"] += gf
 row["ga"] += ga
 if gf > ga:
 row["win"] += 1
 row["points"] += 3
 elif gf == ga:
 row["draw"] += 1
 row["points"] += 1
 else:
 row["loss"] += 1
 standings = [{"team": t, **v, "gd": v["gf"] - v["ga"]} for t, v in table.items()]
 standings.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
 for i, row in enumerate(standings, 1):
 row["position"] = i
 return standings
