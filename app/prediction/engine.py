"""PredictionEngine(审查九 P0-1/P1-8 拆分):统一预测核心链路。

生产 / Walk-forward / Replay / OOF 全部调用同一引擎 —— 杜绝
"Research path ≠ Production path"(审查二十三)。

链路:Goal 成员(含 bayes)→ fuse → GBM → fuse_goal_outcome →
Calibration → Regime 矩阵级调整(IPF)→ Uncertainty → result。

输入:context dict(ContextBuilder 产出);输出:result dict
(不落快照/缓存 —— 由调用方编排)。
"""
from __future__ import annotations

import logging
import os

import numpy as np

from app.prediction import calibration as calibration_service
from app.prediction.goal_engine import compute_members
from app.prediction.outcome_engine import fuse, gbm_probs, load_gbm
from app.prediction.uncertainty import compute as uncertainty_compute
from app.prediction.uncertainty import recompute_after_adjust

logger = logging.getLogger(__name__)


def _poisson_proba(lambda_: float, max_goals: int = 10) -> np.ndarray:
    from app.models.distributions import pois_pmf_vec
    return pois_pmf_vec(lambda_, max_goals)


class PredictionEngine:
    """统一预测引擎(生产/回测/回放/OOF 共用)。"""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def predict(self, ctx: dict) -> dict:
        """ctx: ContextBuilder 产出(含模型/特征/伤停/λ 等);返回 result dict。"""
        league_type = ctx["league_type"]
        league = ctx["league"]
        home_team, away_team = ctx["home_team"], ctx["away_team"]
        match_dt = ctx["match_dt"]
        model = ctx["model"]
        _m = ctx["_m"]
        _pred_df = ctx["_pred_df"]
        home_lambda, away_lambda = ctx["home_lambda"], ctx["away_lambda"]
        _feat = ctx.get("_feat")
        _att_diff = ctx.get("_att_diff", 0.0)
        _h_mult, _a_mult = ctx.get("injury_mult", (1.0, 1.0))
        _degraded_components = list(ctx.get("degraded_components", []))
        _failure_codes = list(ctx.get("failure_codes", []))
        _degraded = False

        try:
            from app.models.ensemble import elo_goal_lambda, fuse_probs, load_weights
            _w = load_weights(league_type.value)
            _tau, _phi = 0.0, 1e9
            try:
                import json as _json
                _pp = os.path.join(str(__import__("app.core.paths", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR), "ensemble", "dc_nb_params.json")
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
            # 审查三十七 P2:上下文动态权重(强弱差距/分歧 → 微调,默认关)
            _ctx_info = None
            try:
                from app.prediction.context_weights import adjust as _ctx_adj
                _w, _ctx_info = _ctx_adj(_w, _att_diff,
                                         ctx.get("disagreement", 0.0))
            except Exception:
                _ctx_info = None
            _lam_eh = elo_goal_lambda(_att_diff, True) * _h_mult
            _lam_ea = elo_goal_lambda(_att_diff, False) * _a_mult
            _lam_bh = ctx.get("bayes_lam_h")
            _lam_ba = ctx.get("bayes_lam_a")
            g = compute_members(_lam_h, _lam_a, _lam_eh, _lam_ea, _tau, _phi, _w,
                                lam_bh=_lam_bh, lam_ba=_lam_ba)
            _members = g["members"]
            _score_out = g["score_out"]
            _fused_matrix = g.get("fused_matrix")
            if _fused_matrix is not None:
                _fused_matrix = _fused_matrix.tolist()
            home_lambda, away_lambda = g["fused_lams"]
            _goal_probs = fuse_probs(_members, _w)

            _gbm_probs = None
            _gbm = load_gbm(league_type, self.models_dir)
            if _gbm is None:
                _degraded_components.append("gbm")
                _failure_codes.append("GBM_UNAVAILABLE")
            else:
                _gbm_probs = gbm_probs(_gbm, _pred_df, _m)
                if _gbm_probs is None:
                    logger.warning("GBM 成员预测失败(降级,不影响主链路)")
                    _degraded = True
            home_win, draw, away_win = fuse(_goal_probs, _gbm_probs, _w)
        except Exception:
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
            _members, _gbm_probs = {}, None
            _fused_matrix = None

        # Uncertainty(基于成员概率与特征质量)
        _unc = uncertainty_compute((home_win, draw, away_win), _members, _gbm_probs,
                                   _feat, getattr(model, "feature_columns_", []))
        _agreement, _data_quality = _unc["agreement"], _unc["data_quality"]

        # 审查三十一:模型解释页数据 —— 成员一致性(1X2)与特征影响
        _members_1x2 = {}
        try:
            for _name, _p in (_members or {}).items():
                _members_1x2[_name] = [round(float(x), 4) for x in _p]
            if _gbm_probs is not None:
                _members_1x2["gbm"] = [round(float(x), 4) for x in _gbm_probs]
        except Exception:
            _members_1x2 = {}
        _feature_impact = []
        try:
            _fi = getattr(getattr(model, "model", None), "feature_importance_", None)
            if _fi is not None:
                _fcols = getattr(model, "feature_columns_", [])
                _pairs = sorted(zip(_fcols, [float(x) for x in _fi]),
                                key=lambda t: -abs(t[1]))[:8]
                _feature_impact = [{"feature": str(f), "importance": round(float(v), 4)}
                                   for f, v in _pairs]
        except Exception:
            _feature_impact = []

        result = {
            "league_type": league_type.value,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_zh": ctx.get("home_team_zh", home_team),
            "away_team_zh": ctx.get("away_team_zh", away_team),
            "predicted_home_goals": round(home_lambda, 2),
            "predicted_away_goals": round(away_lambda, 2),
            "home_win_probability": round(home_win, 4),
            "draw_probability": round(draw, 4),
            "away_win_probability": round(away_win, 4),
            "confidence": _unc["confidence"],
            "confidence_score": _unc["confidence_score"],
            "prediction_entropy": _unc["entropy"],
            "model_disagreement": _unc["disagreement"],
            "data_quality_score": _unc["data_quality"],
            "members_1x2": _members_1x2,
            "feature_impact": _feature_impact,
            "top_scores": _score_out["top_scores"],
            "over_2_5": _score_out["over_2_5"],
            "under_2_5": _score_out["under_2_5"],
            "btts": _score_out["btts"],
            "expected_xg": _score_out["expected_xg"],
            "context_weights": _ctx_info,
            "match_id": ctx.get("match_id"),
            "match_date": match_dt.isoformat() if hasattr(match_dt, "isoformat") else str(match_dt),
            "prediction_status": "degraded" if _degraded else "ok",
            "degraded_components": _degraded_components,
            "failure_codes": _failure_codes,
            "lineup_strength": ctx.get("lineup"),
        }

        # Calibration
        result, _cal_info, _cal_degraded = calibration_service.apply(
            result, self.models_dir, league_type)
        if _cal_degraded:
            _degraded_components.append("calibration")
            _failure_codes.append("CALIBRATION_FAILURE")
        _degraded = _degraded or _cal_degraded
        result["prediction_status"] = "degraded" if _degraded else "ok"

        # Regime 矩阵级调整(P0-3:1X2/比分全部同源)
        _m2 = None
        try:
            from app.prediction.prior_blend import blend_matrix as _prior_blend
            _m2, _blend_info = _prior_blend(
                league.id, match_dt,
                [result["home_win_probability"], result["draw_probability"],
                 result["away_win_probability"]],
                _fused_matrix)
            if _m2 is not None:
                from app.models.ensemble import score_outputs as _so
                _so2 = _so(np.asarray(_m2, dtype=float))
                _arr = np.asarray(_m2, dtype=float)
                _hw2 = float(_arr[np.tril_indices_from(_arr, -1)].sum())
                _dr2 = float(np.trace(_arr))
                _aw2 = float(_arr[np.triu_indices_from(_arr, 1)].sum())
                result["home_win_probability"] = round(_hw2, 4)
                result["draw_probability"] = round(_dr2, 4)
                result["away_win_probability"] = round(_aw2, 4)
                result["top_scores"] = _so2["top_scores"]
                result["over_2_5"] = _so2["over_2_5"]
                result["under_2_5"] = _so2["under_2_5"]
                result["btts"] = _so2["btts"]
                result["expected_xg"] = _so2["expected_xg"]
                result["prior_blend"] = _blend_info
                _unc2 = recompute_after_adjust((_hw2, _dr2, _aw2),
                                               _agreement, _data_quality)
                result["confidence"] = _unc2["confidence"]
                result["confidence_score"] = _unc2["confidence_score"]
                result["prediction_entropy"] = _unc2["entropy"]
        except Exception as _pe:
            _degraded_components.append("prior_blend")
            _failure_codes.append("PRIOR_BLEND_FAILURE")
            logger.warning("prior blend 失败(降级为未混合概率): %s", _pe)
            _degraded = True
            result["prediction_status"] = "degraded" if _degraded else "ok"

        result["_internal"] = {
            "home_lambda": home_lambda, "away_lambda": away_lambda,
            "att_diff": _att_diff, "hist_df": ctx["hist_df"],
            "model": model, "_m": _m, "_pred_df": _pred_df, "_feat": _feat,
            "fused_matrix": _m2 if _m2 is not None else _fused_matrix,
            "cal_info": _cal_info, "degraded": _degraded,
            "degraded_components": _degraded_components,
            "failure_codes": _failure_codes,
        }
        return result
