"""V2 预测端点(新设计:预测 + 快照 + 复盘统一)"""
import os

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.db import Match, Prediction, PredictionSnapshot, db
from app.api.schemas import PredictMatchReq, PredictTournamentReq
from app.api.security import get_current_user
from app.data.adapters import _resolve_league_type
from app.prediction.service import predict_match
from app.prediction.tournament import predict_tournament

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

MODELS_DIR = os.environ.get("MODELS_DIR", "app/models")


@router.post("/match")
def predict_match_endpoint(body: PredictMatchReq, user=Depends(get_current_user)):
    try:
        league_type = _resolve_league_type(body.league_type)
        result = predict_match(
            league_type, body.home_team.strip(), body.away_team.strip(),
            body.date, models_dir=MODELS_DIR)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"预测失败: {e}")
    pred = Prediction(
        user_id=user.id,
        match_id=result.get("match_id"),
        league_type=result["league_type"],
        home_team=result["home_team"],
        away_team=result["away_team"],
        home_goals_prediction=result["predicted_home_goals"],
        away_goals_prediction=result["predicted_away_goals"],
        home_win_probability=result["home_win_probability"],
        draw_probability=result["draw_probability"],
        away_win_probability=result["away_win_probability"],
        confidence=result["confidence"],
    )
    db.session.add(pred)
    db.session.commit()
    return {"prediction": result, "prediction_id": pred.public_id}


@router.post("/tournament")
def predict_tournament_endpoint(body: PredictTournamentReq, user=Depends(get_current_user)):
    try:
        league_type = _resolve_league_type(body.league_type)
        teams = [t.name if isinstance(t, dict) else str(t) for t in body.teams]
        teams = [t for t in teams if t and t != "None"]
        result = predict_tournament(
            league_type, teams, body.num_simulations, models_dir=MODELS_DIR)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"赛事预测失败: {e}")
    return result


@router.get("")
def list_predictions(user=Depends(get_current_user)):
    predictions = (Prediction.query.filter_by(user_id=user.id)
                   .order_by(Prediction.created_at.desc()).limit(200).all())
    match_ids = {p.match_id for p in predictions if p.match_id}
    match_map = {m.id: m for m in Match.query.filter(Match.id.in_(match_ids)).all()} if match_ids else {}
    return {"items": [p.to_dict(match_map=match_map) for p in predictions], "total": len(predictions)}


# ---------------- 快照 / 复盘(V2 核心) ----------------

@router.get("/snapshots")
def snapshots(limit: int = Query(50, ge=1, le=500), league: str | None = None):
    q = PredictionSnapshot.query
    if league:
        # 快照无 league 列,经 league_id 关联(§2.1)
        from app.api.db import League
        _lg = League.query.filter_by(league_type=league).first()
        if _lg is None:
            return {"items": [], "total": 0}
        q = q.filter_by(league_id=_lg.id)
    rows = q.order_by(PredictionSnapshot.id.desc()).limit(limit).all()
    return {"items": [_snap_dict(s) for s in rows], "total": len(rows)}


def _snap_dict(s) -> dict:
    return {
        "id": s.id, "league": getattr(s, "league", None),
        "home_team": getattr(s, "home_team", None), "away_team": getattr(s, "away_team", None),
        "match_date": str(getattr(s, "match_date", "") or ""),
        "predicted_home_goals": getattr(s, "predicted_home_goals", None),
        "predicted_away_goals": getattr(s, "predicted_away_goals", None),
        "actual_home_goals": getattr(s, "actual_home_goals", None),
        "actual_away_goals": getattr(s, "actual_away_goals", None),
        "log_loss": getattr(s, "log_loss", None),
        "created_at": str(getattr(s, "created_at", "") or ""),
    }


@router.get("/replay/{snapshot_id}")
def replay(snapshot_id: int):
    s = db.session.get(PredictionSnapshot, snapshot_id)
    if s is None:
        raise HTTPException(404, f"快照 {snapshot_id} 不存在")
    d = _snap_dict(s)
    d["snapshot_json"] = getattr(s, "snapshot_json", None)
    d["probabilities_json"] = getattr(s, "probabilities_json", None)
    return d
