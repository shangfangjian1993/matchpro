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
    """读取冻结的比分矩阵(审查 P1-7)。

    优先用快照保存时的 score_matrix —— 算法后续修改(Poisson/DC/NB/截断/
    归一化)不得改变历史快照结果;老快照无 score_matrix 时回退 λ 重建。
    """
    sm = ev.get("score_matrix")
    if sm:
        try:
            return [[float(x) for x in row] for row in sm]
        except Exception:
            pass
    lam = ev.get("lambda") or [1.5, 1.2]
    try:
        from app.models.ensemble import _pois_matrix
        return _pois_matrix(float(lam[0]), float(lam[1])).tolist()
    except Exception:
        return []


def replay_all(force: bool = False) -> dict:
    """回放全部快照并返回 §6 汇总。"""
    import pandas as pd
    from app.api.db import Match, PredictionSnapshot, db
    snaps = (PredictionSnapshot.query.filter(PredictionSnapshot.actual_home_goals.is_(None)).all()
             if not force else PredictionSnapshot.query.all())
    # 审查 P1-8:N+1 修复 —— 一次查询建内存索引,循环 O(1) 查找
    # (27,825 场规模下逐条 SQL 会非常慢)
    _all_matches = Match.query.filter_by(match_status="finished").all()
    _match_map = {}
    for _m in _all_matches:
        _key = (_m.league_id, _m.home_team, _m.away_team,
                str(pd.Timestamp(_m.match_date).date()))
        _match_map.setdefault(_key, []).append(_m)
    filled = 0
    for s in snaps:
        _key = (s.league_id, s.home_team, s.away_team,
                str(pd.Timestamp(s.kickoff).date()))
        m = _match_map.get(_key)
        if not m:
            continue
        backfill_snapshot(s, m[0])
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
    out = {
        "count": n,
        "accuracy": round(accuracy(pvecs, acts), 4),
        "log_loss": round(sum(s.log_loss or 0 for s in done) / n, 4),
        "brier": round(sum(brier_score(p, a) for p, a in zip(pvecs, acts)) / n, 4),
        "rps": round(sum(rps(p, a) for p, a in zip(pvecs, acts)) / n, 4),
        "ece": round(ece(pvecs, acts), 4),
    }
    # 审查九 二十二/三十四:扩展指标(校准斜率/锐度/Brier 分解/比分分布 LL)
    try:
        from app.replay.metrics import extended_metrics
        _smat, _scores = [], []
        for s in done:
            _ev = json.loads(s.snapshot_json or "{}")
            _m = _ev.get("score_matrix")
            if _m:
                _smat.append(_m)
                _scores.append((s.actual_home_goals or 0, s.actual_away_goals or 0))
        _ext = extended_metrics(pvecs, acts,
                                _smat if len(_smat) == len(acts) else None,
                                _scores if len(_scores) == len(acts) else None)
        out["calibration_slope"] = _ext["calibration_slope"]
        out["calibration_intercept"] = _ext["calibration_intercept"]
        out["sharpness"] = _ext["sharpness"]
        out["brier_decomposition"] = _ext["brier_decomp"]
        out["score_log_likelihood_mean"] = _ext.get("score_log_likelihood_mean")
        out["logloss_by_bucket"] = _ext["logloss_by_bucket"]
    except Exception:
        pass
    return out
