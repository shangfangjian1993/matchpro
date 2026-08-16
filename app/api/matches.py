"""V2 比赛端点(新设计)"""
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.db import League, Match, db
from app.api.schemas import MatchBatchReq, MatchCreate
from app.api.security import get_current_user
from app.core.config import MATCH_METRIC_COLUMNS
from app.core.timeutil import utcnow
from app.data.adapters import _resolve_league_type
from app.data.canonical.cleanse import _parse_date

router = APIRouter(prefix="/api/matches", tags=["matches"])

FLOAT_FIELDS = {
    "home_xg", "away_xg", "home_possession", "home_passing_accuracy",
    "away_passing_accuracy", "home_xg_chain", "away_xg_chain",
    "home_efficiency", "away_efficiency", "home_transition_speed",
    "away_transition_speed", "home_defensive_actions", "away_defensive_actions",
    "home_counter_attacks", "away_counter_attacks", "home_tactical_rating",
    "away_tactical_rating", "home_experience", "away_experience",
}


def _resolve_league(data: dict):
    try:
        league_type = _resolve_league_type(data.get("league_type"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    league = League.query.filter_by(league_type=league_type.value).first()
    if league is None:
        raise HTTPException(404, f"数据库中还没有 {league_type.value} 的联赛,请先创建联赛")
    return league


def _validate_match(league, data: dict, index: int | None = None):
    prefix = f"第 {index} 条: " if index else ""
    home = (data.get("home_team") or "").strip()
    away = (data.get("away_team") or "").strip()
    if not home or not away:
        return None, f"{prefix}home_team/away_team 不能为空"
    if home == away:
        return None, f"{prefix}主客队不能相同"

    def _to_int(v, name, default=0):
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            raise HTTPException(400, f"{prefix}{name} 必须是整数")

    home_goals = _to_int(data.get("home_goals"), "home_goals", 0)
    away_goals = _to_int(data.get("away_goals"), "away_goals", 0)
    if home_goals < 0 or away_goals < 0:
        return None, f"{prefix}进球数不能为负"

    match_date = _parse_date(data.get("match_date")) if data.get("match_date") else utcnow()
    m = Match(
        league_id=league.id,
        home_team=home, away_team=away,
        home_goals=home_goals, away_goals=away_goals,
        match_date=match_date,
        match_status=(data.get("match_status") or "finished"),
        match_stage=data.get("match_stage") or "",
    )
    for col in MATCH_METRIC_COLUMNS:
        if col in data and data[col] is not None:
            try:
                setattr(m, col, float(data[col]) if col in FLOAT_FIELDS else int(data[col]))
            except (TypeError, ValueError):
                return None, f"{prefix}{col} 必须是数值"
    return m, None


@router.get("")
def list_matches(league_type: str | None = None, status: str | None = None,
                 limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    q = Match.query
    if league_type:
        try:
            lt = _resolve_league_type(league_type)
        except ValueError as e:
            raise HTTPException(400, str(e))
        league = League.query.filter_by(league_type=lt.value).first()
        if league is None:
            raise HTTPException(404, "该联赛不存在")
        q = q.filter_by(league_id=league.id)
    if status:
        q = q.filter_by(match_status=status)
    total = q.count()
    matches = q.order_by(Match.match_date.desc()).offset(offset).limit(limit).all()
    return {"items": [m.to_dict() for m in matches], "total": total}


@router.post("", status_code=201)
def create_match(body: MatchCreate, user=Depends(get_current_user)):
    data = body.model_dump(exclude_unset=True)
    league = _resolve_league(data)
    m, err = _validate_match(league, data)
    if err:
        raise HTTPException(400, err)
    dup = Match.query.filter_by(league_id=league.id, home_team=m.home_team,
                                away_team=m.away_team).filter(
        Match.match_date == m.match_date).first()
    if dup:
        return {"message": "该场比赛已存在,未重复录入",
                "match": dup.to_dict(), "duplicate": True}
    db.session.add(m)
    db.session.commit()
    return {"message": "比赛录入成功", "match": m.to_dict()}


@router.post("/batch", status_code=201)
def batch_create_matches(body: MatchBatchReq, user=Depends(get_current_user)):
    data = body.model_dump()
    league = _resolve_league(data)
    items = data["matches"]
    valid, errors = [], []
    for i, item in enumerate(items, 1):
        m, err = _validate_match(league, item, index=i)
        if err:
            errors.append({"index": i, "error": err})
        else:
            valid.append(m)
    if errors:
        raise HTTPException(400, {"message": f"{len(errors)} 条数据校验失败,未写入任何数据",
                                  "errors": errors[:20]})
    imported, skipped = [], 0
    existing = {
        (x.home_team, x.away_team, x.match_date.strftime("%Y-%m-%d %H:%M"))
        for x in Match.query.filter_by(league_id=league.id).all()
    }
    for m in valid:
        key = (m.home_team, m.away_team, m.match_date.strftime("%Y-%m-%d %H:%M"))
        if key in existing:
            skipped += 1
            continue
        existing.add(key)
        db.session.add(m)
        imported.append(m)
    db.session.commit()
    return {"message": f"批量录入完成:新增 {len(imported)} 场,跳过重复 {skipped} 场",
            "imported": len(imported), "skipped": skipped, "errors": []}
