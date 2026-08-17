"""Snapshot 落库(审查 §12/§14 拆分):冻结最终输出 + 输入 + 全版本。

审查 P0-5(2026-08 重写):
- 特征行修复:预测构造为 history + home_row + away_row,-2 为主队行、-1 为客队行。
  原实现只存 iloc[-1](客队行),与主队实际预测特征不对应;现分别保存
  home_features / away_features。
- 原子语义:核心字段失败(模型哈希、特征值、写库)→ raise,预测与快照
  视为一个原子事务,不允许"prediction_status=ok 但无快照";非核心诊断
  字段(prematch_elo/ensemble 权重/dc 参数)失败 → warning 降级。
"""
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
         _feat=None, _score_matrix=None, evaluation_mode: str = "production"):
    """快照落库(幂等;冻结校准后最终输出)。核心失败 raise —— 快照与预测原子。

    _score_matrix: 预测时冻结的比分矩阵(审查 P1-7,Replay 不得用 λ 重算)。
    evaluation_mode: production / historical_replay / walk_forward(审查六-7)。
    """
    """快照落库(幂等;冻结校准后最终输出)。核心失败 raise —— 快照与预测原子。"""
    from app.api.db import PredictionSnapshot, Team, db

    # ── 阶段 1(核心):模型哈希 —— 失败即不可复现,raise ──────────────────
    from app.core.config import LeagueType as _LT
    _mpath = _model_path(_LT(league.league_type), models_dir) if hasattr(league, 'league_type') else None
    if _mpath is None:
        raise RuntimeError("快照失败:无法定位模型文件路径(不可复现)")
    with open(_mpath, "rb") as _mf:
        _model_hash = hashlib.sha256(_mf.read()).hexdigest()[:64]
    _data_hash = hashlib.sha256(
        f"{hist_max_id}|{hist_max_updated}|{home_team}|{away_team}".encode()).hexdigest()[:64]

    # ── 阶段 2(非核心诊断):失败仅 warning ────────────────────────────────
    _team_ids = {}
    try:
        _team_ids = {t.name: t.id for t in Team.query.filter(
            Team.name.in_([home_team, away_team])).all()}
    except Exception as e:
        logger.warning("快照:球队 id 提取失败(降级): %s", e)
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
        _dcp = json.load(open(os.path.join(str(__import__("app.core.paths", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR), "ensemble", "dc_nb_params.json"),
                              encoding="utf-8")).get(league.league_type, {})
    except Exception:
        _dcp = {}

    # ── 阶段 3(核心):最终输入特征(100% 重放)── 失败 raise ───────────────
    if _feat is None:
        _feat = _m.prepare_features(_pred_df)
    _fcols = [col for col in getattr(model, "feature_columns_", [])
              if col in _feat.columns]
    if len(_feat) < 2:
        raise RuntimeError(f"快照失败:特征矩阵仅 {len(_feat)} 行,无主/客预测行")
    # 审查 P0-5:-2 行 = 主队预测行,-1 行 = 客队预测行(见 extract_feature_rows)
    _home_features, _away_features = extract_feature_rows(_feat, _fcols)
    try:
        from app.features.registry import logical_version as _fv
        _feature_ver = getattr(model, "feature_version_", None) or _fv()
    except Exception as e:
        raise RuntimeError(f"快照失败:特征版本不可计算(不可复现): {e}") from e

    # ── 阶段 3.5:完整模型集合哈希(审查 P0.5/十九)───────────────
    # Goal HGBR 之外,必须冻结 GBM / Ensemble / Calibration 各自版本,否则
    # 快照不能代表"预测那一刻的完整模型集合"。
    _gbm_hash = None
    try:
        _gbm_path = os.path.join(str(__import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR),
                                 league.league_type, "gbm.pkl")
        with open(_gbm_path, "rb") as _gf:
            _gbm_hash = hashlib.sha256(_gf.read()).hexdigest()[:12]
    except OSError:
        _gbm_hash = None
    _ens_hash = None
    try:
        _ens_dir = os.path.join(str(__import__("app.core.paths", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR),
                                "ensemble")
        _ens_raw = ""
        for _ef in ("ensemble_weights.json", "dc_nb_params.json"):
            with open(os.path.join(_ens_dir, _ef), "rb") as _f2:
                _ens_raw += _f2.read().hex()
        _ens_hash = hashlib.sha256(_ens_raw.encode()).hexdigest()[:12]
    except OSError:
        _ens_hash = None

    _snapshot = {
        "data_version": _data_hash[:12],
        "feature_version": _feature_ver,
        "model_version": _model_hash[:12],
        "evaluation_mode": evaluation_mode,
        "model_set": {
            "goal": {"version": os.path.basename(_mpath).replace(".pkl", ""),
                     "sha256": _model_hash},
            "gbm": {"sha256": _gbm_hash} if _gbm_hash else None,
            "ensemble": {"sha256": _ens_hash} if _ens_hash else None,
            "calibration": ({"method": _cal_info["method"],
                             "sha256": _cal_info.get("artifact_hash")}
                            if _cal_info else None),
        },
        "score_matrix": (_score_matrix if _score_matrix is not None else []),
        # 审查 P1-11:版本 = artifact 哈希(唯一标识);无哈希时回退 method:n
        "calibration_version": (
            f"{_cal_info['method']}:{_cal_info['artifact_hash']}"
            if _cal_info and _cal_info.get('artifact_hash')
            else (f"{_cal_info['method']}:{_cal_info['n']}" if _cal_info else None)
        ),
        "home_features": _home_features,      # P0-5:-2 行 = 主队预测特征
        "away_features": _away_features,      # P0-5:-1 行 = 客队预测特征
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
    # ── 阶段 4(核心):写库 —— 失败 raise,保证 预测+快照 原子 ───────────
    db.session.commit()

def extract_feature_rows(feat, feature_columns):
    """从经排序的预测特征矩阵提取主/客队预测行特征(审查 P0-5/三十)。

    预测输入 = history + home_row + away_row → 排序后:
      - 最后两行为预测行:-2 = 主队,-1 = 客队
      - 快照保存的 home_features/away_features 必须与模型实际预测使用的
        特征向量一致(否则"快照 = 最终输入冻结"不成立)。
    """
    if len(feat) < 2:
        raise RuntimeError(f"特征矩阵仅 {len(feat)} 行,无主/客预测行")
    home = {
        col: (None if pd.isna(feat.iloc[-2][col])
              else round(float(feat.iloc[-2][col]), 6))
        for col in feature_columns
    }
    away = {
        col: (None if pd.isna(feat.iloc[-1][col])
              else round(float(feat.iloc[-1][col]), 6))
        for col in feature_columns
    }
    return home, away
