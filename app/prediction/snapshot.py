"""Snapshot 落库(审查 §12/§14 拆分):冻结最终输出 + 输入 + 全版本。"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def save(league, home_team, away_team, match_dt, match_date, result,
         home_lambda, away_lambda, _att_diff, hist_df, model, _pred_df,
         _m, hist_max_id, hist_max_updated, _model_path, models_dir, _cal_info,
         _feat=None):
    """快照落库(幂等;冻结校准后最终输出)。失败仅 warning(可降级)。"""
    from app.api.db import PredictionSnapshot, Team, db

    try:
        # 版本与哈希(审查 P0-5:模型哈希/数据版本失败必须 raise,不可复现不得静默)
        from app.core.config import LeagueType as _LT
        _mpath = _model_path(_LT(league.league_type), models_dir) if hasattr(league, 'league_type') else None
        try:
            with open(_mpath, "rb") as _mf:
                _model_hash = hashlib.sha256(_mf.read()).hexdigest()[:64]
        except OSError as e:
            raise RuntimeError(f"模型哈希计算失败(快照不可复现): {e}") from e
        _data_hash = hashlib.sha256(
            f"{hist_max_id}|{hist_max_updated}|{home_team}|{away_team}".encode()).hexdigest()[:64]

        _team_ids = {t.name: t.id for t in Team.query.filter(
            Team.name.in_([home_team, away_team])).all()}
        try:
            from app.models.utils import prematch_elo
            _he_o, _ae_o, _ = prematch_elo(hist_df, home_team, away_team, "overall")
            _he_a, _ae_a, _att = prematch_elo(hist_df, home_team, away_team, "attack")
        except Exception:
            _he_o = _ae_o = _he_a = _ae_a = _att = None
        try:
            from app.models.ensemble import load_weights
            _ens_w = load_weights(league.league_type)
        except Exception:
            _ens_w = {}
        try:
            _dcp = json.load(open(os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), "artifacts", "ensemble", "dc_nb_params.json"),
                                  encoding="utf-8")).get(league.league_type, {})
        except Exception:
            _dcp = {}
        # §25:实际预测输入特征值(最终特征向量,100% 重放用)
        _features_snapshot = {}
        try:
            if _feat is None:
                _feat = _m.prepare_features(_pred_df)
            _fcols = [col for col in getattr(model, "feature_columns_", [])
                      if col in _feat.columns]
            _features_snapshot = {
                col: (None if pd.isna(_feat.iloc[-1][col])
                      else round(float(_feat.iloc[-1][col]), 6))
                for col in _fcols
            }
        except Exception as e:
            logger.warning("快照特征值提取失败(可降级): %s", e)
        try:
            from app.features.registry import logical_version as _fv
            _feature_ver = getattr(model, "feature_version_", None) or _fv()
        except Exception:
            _feature_ver = _data_hash[:12]
        _snapshot = {
            "data_version": _data_hash[:12],
            "feature_version": _feature_ver,
            "model_version": _model_hash[:12],
            "calibration_version": (f"{_cal_info['method']}:{_cal_info['n']}"
                                    if _cal_info else None),
            "features": _features_snapshot,
            "lambda": [home_lambda, away_lambda],
            "prematch_elo": {"home": _he_o, "away": _ae_o,
                             "attack_diff": _att if _att is not None else _att_diff},
            "feature_columns": list(getattr(model, "feature_columns_", None) or []),
            "model_params": getattr(model, "config", None).parameters if getattr(model, "config", None) else None,
            "ensemble_weights": {k: v for k, v in _ens_w.items()
                                 if k in ("hgbr", "dc", "nb", "elo", "gbm")},
            "dc_tau": _dcp.get("tau"), "nb_phi": _dcp.get("phi"),
            "calibration": _cal_info,
        }
        _probs = {
            "home_win": result["home_win_probability"],
            "draw": result["draw_probability"],
            "away_win": result["away_win_probability"],
            "top_scores": result.get("top_scores", []),
            "over_2_5": result.get("over_2_5"),
            "under_2_5": result.get("under_2_5"),
            "btts": result.get("btts"),
            "expected_xg": result.get("expected_xg"),
        }
        _sha = hashlib.sha256(
            (_data_hash + _model_hash + json.dumps(_snapshot, sort_keys=True, default=str)).encode()
        ).hexdigest()
        _kickoff = match_dt if match_date else match_dt.normalize()
        _existing = PredictionSnapshot.query.filter_by(
            league_id=league.id, home_team=home_team, away_team=away_team,
            kickoff=_kickoff.to_pydatetime() if hasattr(_kickoff, "to_pydatetime") else _kickoff,
        ).first()
        _snap_data = dict(
            league_id=league.id,
            home_team=home_team,
            away_team=away_team,
            home_team_id=_team_ids.get(home_team),
            away_team_id=_team_ids.get(away_team),
            kickoff=_kickoff,
            model_version=_model_hash[:12],
            feature_version=_feature_ver,
            data_hash=_data_hash, model_hash=_model_hash, sha256=_sha,
            snapshot_json=json.dumps(_snapshot, ensure_ascii=False, default=str),
            probabilities_json=json.dumps(_probs, ensure_ascii=False),
        )
        if _existing is None:
            db.session.add(PredictionSnapshot(**_snap_data))
        else:
            for _k, _v in _snap_data.items():
                setattr(_existing, _k, _v)
        db.session.commit()
    except Exception as e:
        # 审查 P0-5:快照写入失败属"可降级"(预测不受影响),warning 记录
        logger.warning("快照落库失败(预测不受影响): %s", e)
