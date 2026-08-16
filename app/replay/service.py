"""Replay 服务(§1.1 app/replay):快照赛后回放 + §6 全量评估。"""
from __future__ import annotations

import json
import math


def backfill_snapshot(snapshot, match) -> bool:
    """回填单个快照(actual 比分 + §6 指标)。"""
    from app.replay.metrics import brier_score, rps, topk_coverage
    snapshot.actual_home_goals = match.home_goals
    snapshot.actual_away_goals = match.away_goals
    probs = json.loads(snapshot.probabilities_json)
    gh, ga = match.home_goals or 0, match.away_goals or 0
    actual = "home" if gh > ga else ("draw" if gh == ga else "away")
    pred = max(probs, key=lambda k: probs[k] if k in ("home_win", "draw", "away_win") else -1)
    snapshot.is_correct = (pred == "home" and gh > ga) or (pred == "draw" and gh == ga) \
                          or (pred == "away" and gh < ga)
    p_actual = probs.get({"home": "home_win", "draw": "draw", "away": "away_win"}[actual], 0.0)
    snapshot.log_loss = -math.log(max(p_actual, 1e-9))
    pvec = [probs.get("home_win", 0.0), probs.get("draw", 0.0), probs.get("away_win", 0.0)]
    ai = {"home": 0, "draw": 1, "away": 2}[actual]
    ev = json.loads(snapshot.snapshot_json or "{}")
    ev["evaluation"] = {
        "brier": round(brier_score(pvec, ai), 5),
        "rps": round(rps(pvec, ai), 5),
        "top3": topk_coverage(_score_matrix(ev), (gh, ga), 3),
        "top5": topk_coverage(_score_matrix(ev), (gh, ga), 5),
        "goal_mae": round((abs(ev.get("lambda", [0, 0])[0] - gh) +
                           abs(ev.get("lambda", [0, 0])[1] - ga)) / 2, 4),
    }
    snapshot.snapshot_json = json.dumps(ev, ensure_ascii=False)
    return True


def _score_matrix(ev: dict) -> list:
    """从快照重建比分概率矩阵(§REUSE:统一走 ensemble._pois_matrix)。"""
    lam = ev.get("lambda") or [1.5, 1.2]
    try:
        from app.models.ensemble import _pois_matrix
        return _pois_matrix(float(lam[0]), float(lam[1])).tolist()
    except Exception:
        return []


def replay_all(force: bool = False) -> dict:
    """回放全部快照并返回 §6 汇总。"""
    from app.api.db import Match, PredictionSnapshot, db
    snaps = (PredictionSnapshot.query.filter(PredictionSnapshot.actual_home_goals.is_(None)).all()
             if not force else PredictionSnapshot.query.all())
    filled = 0
    for s in snaps:
        m = Match.query.filter_by(league_id=s.league_id, home_team=s.home_team,
                                  away_team=s.away_team, match_status="finished").filter(
            db.func.date(Match.match_date) == db.func.date(s.kickoff)).first()
        if m is None:
            continue
        backfill_snapshot(s, m)
        filled += 1
    db.session.commit()
    summary = summarize()
    _record_to_experiment(summary)
    return summary


def _record_to_experiment(summary: dict) -> None:
    """§7.2 回放指标汇入 experiments(驱动自动学习/归因)。"""
    if not summary.get("count"):
        return
    try:
        import json as _json
        import uuid as _uuid

        from app.api.db import Experiment, db
        db.session.add(Experiment(
            public_id=_uuid.uuid4().hex,
            league_type="replay",
            dataset_version=f"snapshots_{summary['count']}",
            feature_version="replay",
            model_version="replay",
            metrics_json=_json.dumps(summary, ensure_ascii=False),
            notes="replay 评估汇总(§7.2 汇入 experiment)",
        ))
        db.session.commit()
    except Exception:
        pass


def summarize() -> dict:
    """§6 全量评估汇总。"""
    from app.api.db import PredictionSnapshot
    from app.replay.metrics import accuracy, brier_score, ece, rps
    done = PredictionSnapshot.query.filter(PredictionSnapshot.is_correct.isnot(None)).all()
    if not done:
        return {"count": 0}
    pvecs, acts = [], []
    for s in done:
        probs = json.loads(s.probabilities_json)
        pvecs.append([probs.get("home_win", 0.0), probs.get("draw", 0.0),
                      probs.get("away_win", 0.0)])
        gh, ga = s.actual_home_goals or 0, s.actual_away_goals or 0
        acts.append(0 if gh > ga else (1 if gh == ga else 2))
    n = len(done)
    return {
        "count": n,
        "accuracy": round(accuracy(pvecs, acts), 4),
        "log_loss": round(sum(s.log_loss or 0 for s in done) / n, 4),
        "brier": round(sum(brier_score(p, a) for p, a in zip(pvecs, acts)) / n, 4),
        "rps": round(sum(rps(p, a) for p, a in zip(pvecs, acts)) / n, 4),
        "ece": round(ece(pvecs, acts), 4),
    }
