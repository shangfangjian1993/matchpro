"""PredictionEngine。

统一预测核心链路(生产 / Walk-forward / Replay / OOF 同一入口)。

P0-2/P0-3 概率流水线(严格顺序):
  Raw Models(HGBR/DC/NB/ELO/Bayes)→ Model Ensemble → Raw Matrix
      ↓
  Regime / Prior Blend(IPF,目标 1X2 = α·P + (1-α)·近期频率)
      ↓
  Final Score Matrix → 统一导出(1X2/O-U/BTTS/Top5/xG)
      ↓
  Calibration(校准**最终实际输出**概率 —— 校准对象与线上输出同一)
      ↓
  二次 IPF(校准后 1X2 回写矩阵,保证矩阵边缘 == 最终 1X2)
      ↓
  最终矩阵(快照冻结,唯一输出源)

P0-1 三级异常:
  P0-Core(INVALID_LAMBDA/INVALID_PROBABILITY/ENSEMBLE_FAILURE/
          SCORE_MATRIX_FAILURE/CORE_PREDICTION_FAILURE)→ 直接 raise,
          不生成正式 Snapshot,绝不静默退回纯 HGBR;
  P1-Degraded(GBM_UNAVAILABLE/CALIBRATION_UNAVAILABLE/OPTIONAL_PRIOR)→
          degraded snapshot;
  P2-Informational(FEATURE_IMPACT_UNAVAILABLE)→ 不影响。
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from app.core.exceptions import CorePredictionError
from app.prediction import calibration as calibration_service
from app.prediction.goal_engine import compute_members
from app.prediction.outcome_engine import fuse, gbm_probs, load_gbm
from app.prediction.uncertainty import compute as uncertainty_compute
from app.prediction.uncertainty import recompute_after_adjust

logger = logging.getLogger(__name__)


def _poisson_proba(lambda_: float, max_goals: int = 10) -> np.ndarray:
    from app.models.distributions import pois_pmf_vec

    return pois_pmf_vec(lambda_, max_goals)


def _check_lambda(v, code: str = "INVALID_LAMBDA", name: str = "λ") -> None:
    if v is None or not math.isfinite(float(v)) or float(v) <= 0:
        raise CorePredictionError(code, f"{name} 非法: {v}")


def _check_probs(*ps) -> None:
    for p in ps:
        if p is None or not math.isfinite(float(p)) or not (0.0 <= float(p) <= 1.0):
            raise CorePredictionError("INVALID_PROBABILITY", f"概率非法: {p}")


def _check_matrix(m) -> None:
    if m is None:
        raise CorePredictionError("SCORE_MATRIX_FAILURE", "score matrix 缺失")
    arr = np.asarray(m, dtype=float)
    if arr.size == 0:
        raise CorePredictionError("SCORE_MATRIX_FAILURE", "score matrix 为空")
    if not np.all(np.isfinite(arr)):
        raise CorePredictionError("SCORE_MATRIX_FAILURE", "score matrix 含 NaN/Inf")
    if np.any(arr < 0):
        raise CorePredictionError("SCORE_MATRIX_FAILURE", "score matrix 含负概率")
    if not np.isclose(arr.sum(), 1.0, atol=1e-4):
        raise CorePredictionError(
            "SCORE_MATRIX_FAILURE", f"score matrix 和={arr.sum():.6f} ≠ 1"
        )


class PredictionEngine:
    """统一预测引擎(生产/回测/回放/OOF 共用)。
    
    P0-2: 使用 ProductionArtifact 为唯一输入。
    P0-1: League Scoped 路径读取。
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def _league_dir(self, league_type_value: str) -> str:
        """P0-1: League Scoped Artifact 路径。"""
        from app.core.paths import ARTIFACTS_DIR
        return os.path.join(str(ARTIFACTS_DIR), "ensemble", league_type_value)

    def _load_production_artifact(self, league_type_value: str):
        """P0-2: 加载 ProductionArtifact (唯一输入)。"""
        import json
        
        league_dir = self._league_dir(league_type_value)
        artifact_path = os.path.join(league_dir, "production_artifact.json")
        
        if not os.path.exists(artifact_path):
            raise CorePredictionError(
                "ARTIFACT_MISSING",
                f"ProductionArtifact not found for {league_type_value}"
            )
        
        try:
            with open(artifact_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CorePredictionError(
                "ARTIFACT_CORRUPTED",
                f"ProductionArtifact JSON parse error for {league_type_value}: {e}"
            ) from e
        
        from app.services.training.ensemble.artifact import ProductionArtifact
        try:
            return ProductionArtifact.from_dict(data)
        except Exception as e:
            raise CorePredictionError(
                "ARTIFACT_INVALID",
                f"ProductionArtifact validation failed for {league_type_value}: {e}"
            ) from e

    def predict(self, ctx: dict) -> dict:
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
        _informational_codes: list = []
        # 不得因 _degraded 初始 False 而把"存在降级"错误报成 ok
        _degraded = bool(_degraded_components or _failure_codes)
        _ctx_info = None

        # ══ P0-Core:Goal/Ensemble 核心 —— 失败直接 raise,不静默降级 ══
        try:
            from app.models.ensemble import elo_goal_lambda

            # P0-1: 只从 ProductionArtifact 加载
            _artifact = self._load_production_artifact(league_type.value)
            if _artifact is None:
                raise CorePredictionError(
                    "ARTIFACT_MISSING",
                    f"ProductionArtifact not found for {league_type.value}"
                )
            _w = {
                "hgbr": _artifact.goal_lambda.get("hgbr", 0.0),
                "elo": _artifact.goal_lambda.get("elo", 0.0),
                "bayes": _artifact.goal_lambda.get("bayes", 0.0),
                "poisson": _artifact.score_distribution.get("poisson", 0.0),
                "dc": _artifact.score_distribution.get("dc", 0.0),
                "nb": _artifact.score_distribution.get("nb", 0.0),
                "shape": _artifact.outcome.get("shape", 1.0),
                "gbm": _artifact.outcome.get("gbm", 0.0),
            }
            
            # 权重结构校验(P0-1:权重损坏不得静默)
            _w_keys = set(_w) - {"log_loss", "n", "shrinkage"}
            # gbm is optional (old format may not have it)
            for _k in ("hgbr", "dc", "nb", "elo", "bayes"):
                if _k not in _w_keys:
                    raise CorePredictionError("ENSEMBLE_FAILURE", f"权重缺成员 {_k}")
            # Ensure gbm key exists (default to 0 if missing)
            if "gbm" not in _w:
                _w["gbm"] = 0.0
            for _k, _v in _w.items():
                if isinstance(_v, float) and not math.isfinite(_v):
                    raise CorePredictionError("ENSEMBLE_FAILURE", f"权重 {_k} 非有限值")
            # P1-7: Dynamic context weight parity - OOF and Production must use same weights
            # Context-adjusted weights are NOT applied in OOF, so we don't apply them in Production either
            # to maintain mathematical consistency between training and inference.
            _ctx_info = None
            # P0-1: τ/φ from Artifact
            _tau, _phi = _artifact.tau, _artifact.phi

            lambda_hgbr = home_lambda * _h_mult
            lambda_away = away_lambda * _a_mult
            _check_lambda(lambda_hgbr, name="λ_home")
            _check_lambda(lambda_away, name="λ_away")
            lambda_elo_h = elo_goal_lambda(_att_diff, True) * _h_mult
            lambda_elo_a = elo_goal_lambda(_att_diff, False) * _a_mult
            _check_lambda(lambda_elo_h, name="λ_elo_home")
            _check_lambda(lambda_elo_a, name="λ_elo_away")
            lambda_bayes_h = ctx.get("bayes_lam_h")
            lambda_bayes_a = ctx.get("bayes_lam_a")
            if lambda_bayes_h is not None:
                _check_lambda(lambda_bayes_h, name="λ_bayes_home")
            if lambda_bayes_a is not None:
                _check_lambda(lambda_bayes_a, name="λ_bayes_away")

            try:
                g = compute_members(
                    lambda_hgbr,
                    lambda_away,
                    lambda_elo_h,
                    lambda_elo_a,
                    _tau,
                    _phi,
                    _w,
                    lam_bh=lambda_bayes_h,
                    lam_ba=lambda_bayes_a,
                )
            except Exception as _ce:
                raise CorePredictionError(
                    "CORE_PREDICTION_FAILURE", f"Goal 成员计算失败: {_ce}"
                ) from _ce
            _members = g["members"]
            _score_out = g["score_out"]
            _fused_matrix = g.get("fused_matrix")
            _raw_matrix_for_ablation = _fused_matrix
            # 兼容:engine 可能早在下方重建;这里保留组件前原始矩阵供 A/B 审计
            if _fused_matrix is not None:
                _fused_matrix = _fused_matrix.tolist()
            _check_matrix(_fused_matrix)
            home_lambda, away_lambda = g["fused_lams"]
            _check_lambda(home_lambda, name="fused λ_home")
            _check_lambda(away_lambda, name="fused λ_away")
            # 三层计算:Goal 1X2 = Shape Ensemble(fused λ 的 Poi/DC/NB)
            _goal_probs = g["shape_1x2"]

            # P1-Degraded:GBM 可选成员
            _gbm_probs = None
            _gbm = load_gbm(league_type, self.models_dir)
            
            # P0-3: GBM hash 用 file bytes (非 str(model))
            if _gbm is not None and _artifact.gbm_model_hash:
                import hashlib
                import time

                from app.core.paths import ARTIFACTS_DIR as _GBM_ARTIFACTS_DIR
                gbm_path = os.path.join(
                    str(_GBM_ARTIFACTS_DIR),
                    league_type.value,
                    "gbm.pkl"
                )
                # P1: 区分 transient IO error (重试) 与 corruption (P0)
                max_retries = 3
                actual_hash = None
                for attempt in range(max_retries):
                    try:
                        with open(gbm_path, "rb") as _gf:
                            actual_hash = hashlib.sha256(_gf.read()).hexdigest()[:16]
                        break
                    except OSError as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (2 ** attempt))
                            continue
                        raise CorePredictionError(
                            "GBM_HASH_UNAVAILABLE",
                            f"Cannot read GBM model file for hash verification: {e}"
                        ) from e
                if actual_hash != _artifact.gbm_model_hash:
                    raise CorePredictionError(
                        "GBM_HASH_MISMATCH",
                        f"GBM hash mismatch: expected {_artifact.gbm_model_hash}, got {actual_hash}"
                    )
            
            if _gbm is None:
                # P0-5: Artifact 声明 gbm_weight > 0 但 GBM 缺失 → P0
                if _w.get("gbm", 0) > 0:
                    raise CorePredictionError(
                        "GBM_REQUIRED_MISSING",
                        f"Artifact requires GBM weight {_w['gbm']:.2f} but model unavailable"
                    )
                _degraded_components.append("gbm")
                _failure_codes.append("GBM_UNAVAILABLE")
            else:
                try:
                    _gbm_probs = gbm_probs(_gbm, _pred_df, _m)
                except Exception as _ge:
                    _degraded_components.append("gbm")
                    _failure_codes.append("GBM_INFERENCE_FAILURE")
                    logger.warning("GBM 推理失败(降级): %s", _ge)
                if _gbm_probs is None:
                    _degraded = True
            try:
                home_win, draw, away_win = fuse(_goal_probs, _gbm_probs, _w)
            except Exception as _fe:
                raise CorePredictionError(
                    "ENSEMBLE_FAILURE", f"1X2 融合失败: {_fe}"
                ) from _fe
            _check_probs(home_win, draw, away_win)
        except CorePredictionError:
            raise
        except Exception as _e:
            raise CorePredictionError(
                "CORE_PREDICTION_FAILURE", f"核心预测失败: {_e}"
            ) from _e

        # Uncertainty(基于成员概率与特征质量;P2:FEATURE_IMPACT 不可用不降级)
        _unc = uncertainty_compute(
            (home_win, draw, away_win),
            _members,
            _gbm_probs,
            _feat,
            getattr(model, "feature_columns_", []),
        )
        _agreement, _data_quality = _unc["agreement"], _unc["data_quality"]
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
            if _fi is not None and len(_fi):
                _fcols = getattr(model, "feature_columns_", [])
                _pairs = sorted(
                    zip(_fcols, [float(x) for x in _fi]), key=lambda t: -abs(t[1])
                )[:8]
                _feature_impact = [
                    {"feature": str(f), "importance": round(float(v), 4)}
                    for f, v in _pairs
                ]
            else:
                _informational_codes.append("FEATURE_IMPACT_UNAVAILABLE")
        except Exception:
            _feature_impact = []

        result = {
            "league_type": league_type.value,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_zh": ctx.get("home_team_zh", home_team),
            "away_team_zh": ctx.get("away_team_zh", away_team),
            # expected xG 严格分开 —— predicted_* 保留为 raw λ 的兼容别名。
            "raw_lambda_home": round(home_lambda, 4),
            "raw_lambda_away": round(away_lambda, 4),
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
            "context_weights": _ctx_info,
            "top_scores": _score_out["top_scores"],
            "over_2_5": _score_out["over_2_5"],
            "under_2_5": _score_out["under_2_5"],
            "btts": _score_out["btts"],
            "expected_xg": _score_out["expected_xg"],
            "match_id": ctx.get("match_id"),
            "match_date": match_dt.isoformat()
            if hasattr(match_dt, "isoformat")
            else str(match_dt),
            "degraded_components": _degraded_components,
            "failure_codes": _failure_codes,
            "informational_codes": _informational_codes,
            "lineup_strength": ctx.get("lineup"),
        }

        # ══ P0-3:Regime / IPF —— Final Matrix 唯一输出源()══
        _m2 = None
        _blend_info = None
        try:
            from app.prediction.prior_blend import blend_matrix as _prior_blend

            _m2, _blend_info = _prior_blend(
                league.id,
                match_dt,
                [
                    result["home_win_probability"],
                    result["draw_probability"],
                    result["away_win_probability"],
                ],
                _fused_matrix,
            )
            if _m2 is not None:
                _check_matrix(_m2)
                _arr = np.asarray(_m2, dtype=float)
                _hw2 = float(_arr[np.tril_indices_from(_arr, -1)].sum())
                _dr2 = float(np.trace(_arr))
                _aw2 = float(_arr[np.triu_indices_from(_arr, 1)].sum())
                _check_probs(_hw2, _dr2, _aw2)
                result["home_win_probability"] = round(_hw2, 4)
                result["draw_probability"] = round(_dr2, 4)
                result["away_win_probability"] = round(_aw2, 4)
                result["prior_blend"] = _blend_info
                # P0-2 配套:校准前最终概率(校准器拟合对象 = 最终输出)
                result["_pre_calibration_1x2"] = [
                    round(_hw2, 6),
                    round(_dr2, 6),
                    round(_aw2, 6),
                ]
                _unc2 = recompute_after_adjust(
                    (_hw2, _dr2, _aw2), _agreement, _data_quality
                )
                result["confidence"] = _unc2["confidence"]
                result["confidence_score"] = _unc2["confidence_score"]
                result["prediction_entropy"] = _unc2["entropy"]
        except CorePredictionError:
            raise
        except Exception as _pe:
            # P1-Degraded:可选先验不可用 → 降级(不 raise)
            _degraded_components.append("prior_blend")
            _failure_codes.append("OPTIONAL_PRIOR_UNAVAILABLE")
            logger.warning("prior blend 失败(降级为未混合概率): %s", _pe)
            _degraded = True

        # ══ P0-2:Calibration 校准**最终输出**概率(IPF 之后)══
        # 供 fit_calibration 训练 —— 训练对象必须与生产校准输入同分布,
        # 避免"用校准后输出再校准"的重复校准错配。
        result["_calibration_input_1x2"] = [
            result["home_win_probability"],
            result["draw_probability"],
            result["away_win_probability"],
        ]
        result, _cal_info, _cal_degraded = calibration_service.apply(
            result, self.models_dir, league_type
        )
        if _cal_degraded:
            _degraded_components.append("calibration")
            _failure_codes.append("CALIBRATION_UNAVAILABLE")
        _degraded = _degraded or _cal_degraded
        # ══ P0-3:二次 IPF —— 校准后 1X2 回写矩阵,保证矩阵边缘 == 最终 1X2 ══
        # 原始 fused_matrix),**都必须**执行最终 IPF。否则 blend 不可用场景
        # 下矩阵停留在旧矩阵而 1X2 已被 Calibration 修改 → 矩阵边缘 ≠ 最终 1X2,
        # 违反"Final Matrix 为唯一输出源"契约。
        _base_for_ipf = _m2 if _m2 is not None else _fused_matrix
        try:
            from app.prediction.regime import ipf_to_target

            _final_matrix = ipf_to_target(
                np.asarray(_base_for_ipf, dtype=float),
                (
                    result["home_win_probability"],
                    result["draw_probability"],
                    result["away_win_probability"],
                ),
            )
            _check_matrix(_final_matrix)
            _final_matrix = _final_matrix.tolist()
        except CorePredictionError:
            raise
        except Exception as _ie:
            raise CorePredictionError(
                "SCORE_MATRIX_FAILURE", f"最终 IPF 失败: {_ie}"
            ) from _ie
        # 从最终矩阵统一导出(P0-3:唯一输出源,与 blend 可用性无关)
        try:
            from app.models.ensemble import score_outputs as _so

            _so2 = _so(np.asarray(_final_matrix, dtype=float))
            result["top_scores"] = _so2["top_scores"]
            result["over_2_5"] = _so2["over_2_5"]
            result["under_2_5"] = _so2["under_2_5"]
            result["btts"] = _so2["btts"]
            result["expected_xg"] = _so2["expected_xg"]
            result["expected_home_goals"] = _so2["expected_xg"][0]
            result["expected_away_goals"] = _so2["expected_xg"][1]
        except Exception as _se:
            raise CorePredictionError(
                "SCORE_MATRIX_FAILURE", f"最终矩阵导出失败: {_se}"
            ) from _se

        # failure/informational 都纳入);FEATURE_IMPACT 等 P2 信息单列。
        result["prediction_status"] = (
            "degraded" if (_degraded_components or _failure_codes) else "ok"
        )
        result["informational_codes"] = _informational_codes
        result["_internal"] = {
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            "att_diff": _att_diff,
            "hist_df": ctx["hist_df"],
            "model": model,
            "_m": _m,
            "_pred_df": _pred_df,
            "_feat": _feat,
            "fused_matrix": _final_matrix,
            "raw_fused_matrix": _raw_matrix_for_ablation,
            "cal_info": _cal_info,
            "degraded": _degraded,
            "degraded_components": _degraded_components,
            "failure_codes": _failure_codes,
            # —— Ablation 诊断字段(零行为影响;可复现链路上携带成员分解)——
            "members": _members,  # {hgbr,dc,nb,elo,bayes} 1X2(成员原始)
            "gbm_probs": _gbm_probs,  # GBM 1X2(可 None)
            "member_weights": _w,
            "member_lambdas": {
                "hgbr": (lambda_hgbr, lambda_away),
                "dc": (lambda_hgbr, lambda_away),
                "nb": (lambda_hgbr, lambda_away),
                "elo": (lambda_elo_h, lambda_elo_a),
                "bayes": (lambda_bayes_h, lambda_bayes_a),
            },  # 成员权重(含 gbm 键)
            "tau": _tau,
            "phi": _phi,
        }
        return result
