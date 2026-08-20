"""Prediction 服务(审查九 P1-8 拆分后):纯编排。

predict_match = 缓存 → ContextBuilder(上下文)→ PredictionEngine(核心链路)
→ Snapshot(冻结)→ 缓存写入。
数据准备在 context.py,推理在 engine.py,不确定性在 uncertainty.py,
Regime/矩阵调整在 regime.py + prior_blend.py —— 本文件不再承载业务细节。
"""

from __future__ import annotations

import logging

import pandas as pd

from app.api.db import Match
from app.core.cache import PredictionCache as _PredictionCache
from app.core.config import LeagueType
from app.prediction import snapshot as snapshot_service
from app.prediction.context import ContextBuilder
from app.prediction.engine import PredictionEngine

logger = logging.getLogger(__name__)

_PREDICT_CACHE = _PredictionCache()
_predict_cache_key = _PREDICT_CACHE.key
_cache_get = _PREDICT_CACHE.get
_cache_put = _PREDICT_CACHE.put


def predict_match(
    league_type: LeagueType,
    home_team: str,
    away_team: str,
    match_date=None,
    models_dir: str | None = None,
    evaluation_mode: str = "production",
) -> dict:
    """预测单场比赛(编排层)。

    evaluation_mode(审查六-7):production / historical_replay / walk_forward。
    """
    if models_dir is None:
        from app.core.paths import MODELS_DIR as _MD

        models_dir = str(_MD)
    if not home_team or not away_team:
        raise ValueError("缺少主队或客队名称")
    if home_team == away_team:
        raise ValueError("主队和客队不能相同")
    from app.data.canonical.team_names_zh import to_en

    home_team = to_en(home_team)
    away_team = to_en(away_team)

    league = (
        __import__("app.api.db", fromlist=["League"])
        .League.query.filter_by(league_type=league_type.value)
        .first()
    )
    if league is None:
        raise ValueError(f"数据库中还没有 {league_type.value} 的联赛数据")

    from app.core.timeutil import as_utc_naive

    match_dt = (
        as_utc_naive(match_date)
        if match_date
        else as_utc_naive(pd.Timestamp.now())
    )

    # 数据版本聚合(热缓存命中无需加载历史)
    from app.api.db import db as _db

    _hist_q = Match.query.filter(
        Match.league_id == league.id, Match.match_status == "finished"
    )
    if match_dt is not None:
        _hist_q = _hist_q.filter(Match.match_date < match_dt.to_pydatetime())
    hist_max_id = _hist_q.with_entities(_db.func.max(Match.id)).scalar() or 0
    hist_max_updated = _hist_q.with_entities(_db.func.max(Match.updated_at)).scalar()
    cache_key = _predict_cache_key(
        league_type,
        home_team,
        away_team,
        match_dt.normalize() if match_date is None else match_dt,
        hist_max_id,
        models_dir,
        hist_max_updated,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 上下文 + 引擎(统一预测链路 —— 生产/回测/OOF 同入口,审查 P0-1)
    ctx = ContextBuilder(models_dir).build(
        league_type, home_team, away_team, match_date
    )
    result = PredictionEngine(models_dir).predict(ctx)
    # 审查十 P1-4:概率不变量统一校验(违规 → 核心失败,不生成正式快照)
    try:
        from app.core.exceptions import CorePredictionError as _CPE
        from app.prediction.invariants import validate_prediction as _vp

        _vios = _vp(result)
        if _vios:
            raise _CPE("INVALID_PROBABILITY", "概率不变量违规: " + "; ".join(_vios[:5]))
    except _CPE:
        raise
    except Exception:
        pass  # 校验器自身异常不阻断预测
    _internal = result.pop("_internal", {})

    # Snapshot(冻结最终输出 + 输入 + 全版本)
    snapshot_service.save(
        league,
        home_team,
        away_team,
        match_dt,
        match_date,
        result,
        _internal.get("home_lambda"),
        _internal.get("away_lambda"),
        _internal.get("att_diff"),
        _internal.get("hist_df"),
        _internal.get("model"),
        _internal.get("_pred_df"),
        _internal.get("_m"),
        hist_max_id,
        hist_max_updated,
        __import__("app.models.registry", fromlist=["_model_path"])._model_path,
        models_dir,
        _internal.get("cal_info"),
        _feat=_internal.get("_feat"),
        _score_matrix=_internal.get("fused_matrix"),
        evaluation_mode=evaluation_mode,
    )

    _cache_put(cache_key, result)
    return result
