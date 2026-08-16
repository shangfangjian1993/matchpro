"""Prediction 服务(审查 §12/§13:核心预测链路;API 为最上层)。

predict_match 编排:上下文 → 伤停(Info Layer)→ Goal Engine → Outcome Engine
→ Calibration → Snapshot → 缓存。
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from app.api.db import League, Match
from app.core.cache import PredictionCache as _PredictionCache
from app.core.config import LeagueType
from app.data.adapters import matches_to_dataframe
from app.data.canonical.team_names_zh import to_en, to_zh
from app.models.loader import _load_model
from app.models.registry import _model_path
from app.prediction import calibration as calibration_service
from app.prediction import snapshot as snapshot_service
from app.prediction.goal_engine import compute_members
from app.prediction.outcome_engine import fuse, gbm_probs, load_gbm

logger = logging.getLogger(__name__)

_PREDICT_CACHE = _PredictionCache()
_predict_cache_key = _PREDICT_CACHE.key
_cache_get = _PREDICT_CACHE.get
_cache_put = _PREDICT_CACHE.put


def _poisson_proba(lambda_: float, max_goals: int = 10) -> np.ndarray:
    """Poisson 概率分布(0..max_goals)——统一走 distributions.pois_pmf_vec。"""
    from app.models.distributions import pois_pmf_vec
    return pois_pmf_vec(lambda_, max_goals)


def predict_match(league_type: LeagueType, home_team: str, away_team: str,
                  match_date=None, models_dir: str = "app/models") -> dict:
    """预测单场比赛:Goal Engine(λ/比分矩阵)→ Outcome Engine(1X2+GBM)
    → Calibration → Snapshot(冻结最终输出)。"""
    if not home_team or not away_team:
        raise ValueError("缺少主队或客队名称")
    if home_team == away_team:
        raise ValueError("主队和客队不能相同")
    home_team = to_en(home_team)
    away_team = to_en(away_team)

    league = League.query.filter_by(league_type=league_type.value).first()
    if league is None:
        raise ValueError(f"数据库中还没有 {league_type.value} 的联赛数据")

    model = _load_model(league_type, models_dir)

    matched = (
        Match.query.filter_by(league_id=league.id, home_team=home_team, away_team=away_team)
        .order_by(Match.match_date.desc()).first()
    )
    matched_match_id = matched.id if matched else None

    match_dt = pd.Timestamp(match_date) if match_date else pd.Timestamp.now()

    # 数据版本聚合(热缓存命中无需加载历史)
    from app.api.db import db as _db
    _hist_q = Match.query.filter(Match.league_id == league.id,
                                 Match.match_status == "finished")
    if match_dt is not None:
        _hist_q = _hist_q.filter(Match.match_date < match_dt.to_pydatetime())
    hist_max_id = _hist_q.with_entities(_db.func.max(Match.id)).scalar() or 0
    hist_max_updated = _hist_q.with_entities(_db.func.max(Match.updated_at)).scalar()
    cache_key = _predict_cache_key(league_type, home_team, away_team,
                                   match_dt.normalize() if match_date is None else match_dt,
                                   hist_max_id, models_dir, hist_max_updated)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    history = Match.query.filter_by(league_id=league.id, match_status="finished").all()
    if match_dt is not None:
        history = [m for m in history if pd.Timestamp(m.match_date) < match_dt]
    hist_df = matches_to_dataframe(history, league_name=league.name,
                                   league_season=league.season or "")
    if hist_df.empty:
        raise ValueError("数据库中没有历史比赛数据,无法构造赛前特征")
    # 审查 §18:ELO 注入统一在 Feature Factory(prepare_features 内)完成,此处不再外部注入

    # 无历史球队:全球分层兜底
    known_teams = set(hist_df["home_team"]) | set(hist_df["away_team"])
    unknown = [t for t in (home_team, away_team) if t not in known_teams]
    if unknown:
        # 审查 §8:全球分层模型为半成品分支,已删除;需要时再恢复。
        raise ValueError(
            f"球队「{'、'.join(unknown)}」在 {league_type.value} 的历史数据中不存在,"
            f"请先录入其比赛数据")

    home_row = {"date": match_dt, "home_team": home_team, "away_team": away_team,
                "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                "league": league.name, "season": league.season or ""}
    away_row = {"date": match_dt, "home_team": away_team, "away_team": home_team,
                "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                "league": league.name, "season": league.season or ""}

    # 两行合并一次特征构建(rolling 有效行修复后与分别预测严格一致)
    _m = model[1] if isinstance(model, tuple) else model
    _pred_df = pd.concat([hist_df, pd.DataFrame([home_row, away_row])], ignore_index=True)
    _raw = _m.predict(_pred_df)
    home_lambda, away_lambda = (float(_raw["predictions"][-2]),
                                float(_raw["predictions"][-1]))

    _degraded = False
    _att_diff = 0.0
    _feat = None  # prepare 结果(含 ELO;供 att_diff 与快照复用)
    try:
        _feat = _m.prepare_features(_pred_df)  # factory 内注入 ELO(§18)
        if "attack_elo_diff" in _feat.columns:
            _att_diff = float(_feat["attack_elo_diff"].iloc[-2])  # home_row 赛前 diff
    except Exception:
        _att_diff = 0.0

    # 伤停融合(Info Layer,λ 级修正)
    _h_mult, _a_mult = 1.0, 1.0
    try:
        import os as _os

        from data.injuries.collector import InjuriesCollector
        from data.injuries.signals import injuries_to_signals, signal_brief

        from app.prediction.info_fusion import signals_to_adjust
        _ic = InjuriesCollector()
        _day = match_dt.strftime("%Y-%m-%d")
        _cache = _os.path.join(_ic.cache_dir, f"date_{_day}.json")
        if _os.path.exists(_cache):
            _recs = _ic.fetch_by_date(_day, use_cache=True)
            if _recs:
                _sig = injuries_to_signals(
                    _ic.filter_by_team(_recs, home_team),
                    _ic.filter_by_team(_recs, away_team))
                if _sig["sources"]:
                    _h_mult, _a_mult = signals_to_adjust(_sig)
                    logger.info("伤停融合 %s: %s", match_dt.strftime("%Y-%m-%d"),
                                signal_brief(_sig).replace("\n", " | "))
    except Exception:
        pass

    try:
        from app.models.ensemble import elo_goal_lambda, fuse_probs, load_weights
        _w = load_weights(league_type.value)
        _tau, _phi = 0.0, 1e9
        try:
            import json as _json
            _pp = os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), "artifacts", "ensemble", "dc_nb_params.json")
            if os.path.exists(_pp):
                with open(_pp, encoding="utf-8") as _pf:
                    _prm = _json.load(_pf)
                _pl = _prm.get(league_type.value, {})
                _tau = float(_pl.get("tau", 0.0))
                _phi = float(_pl.get("phi", 1e9))
        except Exception:
            pass
        _lam_h = home_lambda * _h_mult
        _lam_a = away_lambda * _a_mult
        _lam_eh = elo_goal_lambda(_att_diff, True) * _h_mult
        _lam_ea = elo_goal_lambda(_att_diff, False) * _a_mult

        # Goal Engine:四成员概率 + 比分矩阵 + Score Outputs
        g = compute_members(_lam_h, _lam_a, _lam_eh, _lam_ea, _tau, _phi, _w)
        _members = g["members"]
        _score_out = g["score_out"]
        home_lambda, away_lambda = g["fused_lams"]
        _goal_probs = fuse_probs(_members, _w)

        # Outcome Engine:GBM(先算,后融合 —— 修复:原顺序导致 GBM 恒未参与)
        _gbm_probs = None
        _gbm = load_gbm(league_type, models_dir)
        if _gbm is not None:
            _gbm_probs = gbm_probs(_gbm, _pred_df, _m)
            if _gbm_probs is None:
                logger.warning("GBM 成员预测失败(降级,不影响主链路)")
                _degraded = True
        home_win, draw, away_win = fuse(_goal_probs, _gbm_probs, _w)
    except Exception:
        # 融合失败退回纯 HGBR 泊松卷积
        logger.warning("Ensemble 融合失败,退回纯 HGBR(降级)")
        _degraded = True
        p_home = _poisson_proba(home_lambda * _h_mult)
        p_away = _poisson_proba(away_lambda * _a_mult)
        grid = np.outer(p_home, p_away)
        home_win = float(grid[np.tril_indices_from(grid, -1)].sum())
        draw = float(np.trace(grid))
        away_win = float(grid[np.triu_indices_from(grid, 1)].sum())
        home_lambda = home_lambda * _h_mult
        away_lambda = away_lambda * _a_mult
        try:
            from app.models.ensemble import score_outputs
            _score_out = score_outputs(grid)
        except Exception:
            _score_out = {"top_scores": [], "over_2_5": None, "under_2_5": None,
                          "btts": None, "expected_xg": [round(home_lambda, 3), round(away_lambda, 3)]}
    confidence = max(home_win, draw, away_win)

    result = {
        "league_type": league_type.value,
        "home_team": home_team,
        "away_team": away_team,
        "home_team_zh": to_zh(home_team),
        "away_team_zh": to_zh(away_team),
        "predicted_home_goals": round(home_lambda, 2),
        "predicted_away_goals": round(away_lambda, 2),
        "home_win_probability": round(home_win, 4),
        "draw_probability": round(draw, 4),
        "away_win_probability": round(away_win, 4),
        "confidence": round(confidence, 4),
        "top_scores": _score_out["top_scores"],
        "over_2_5": _score_out["over_2_5"],
        "under_2_5": _score_out["under_2_5"],
        "btts": _score_out["btts"],
        "expected_xg": _score_out["expected_xg"],
        "match_id": matched_match_id,
        "match_date": match_dt.isoformat() if hasattr(match_dt, "isoformat") else str(match_dt),
        "prediction_status": "degraded" if _degraded else "ok",
    }

    # Calibration(快照前:快照冻结最终输出)
    result, _cal_info, _cal_degraded = calibration_service.apply(result, models_dir, league_type)
    _degraded = _degraded or _cal_degraded
    result["prediction_status"] = "degraded" if _degraded else "ok"

    # Snapshot(冻结最终输出 + 输入 + 全版本)
    snapshot_service.save(league, home_team, away_team, match_dt, match_date, result,
                          home_lambda, away_lambda, _att_diff, hist_df, model, _pred_df,
                          _m, hist_max_id, hist_max_updated, _model_path, models_dir, _cal_info,
                          _feat=_feat)

    _cache_put(cache_key, result)
    return result
