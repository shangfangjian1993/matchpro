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


def save(
    league,
    home_team,
    away_team,
    match_dt,
    match_date,
    result,
    home_lambda,
    away_lambda,
    _att_diff,
    hist_df,
    model,
    _pred_df,
    _m,
    hist_max_id,
    hist_max_updated,
    _model_path,
    models_dir,
    _cal_info,
    _feat=None,
    _score_matrix=None,
    evaluation_mode: str = "production",
):
    """快照落库(幂等;冻结校准后最终输出)。核心失败 raise —— 快照与预测原子。

    _score_matrix: 预测时冻结的比分矩阵(审查 P1-7,Replay 不得用 λ 重算)。
    evaluation_mode: production / historical_replay / walk_forward(审查六-7)。
    """
    """快照落库(幂等;冻结校准后最终输出)。核心失败 raise —— 快照与预测原子。"""
    from app.api.db import PredictionSnapshot, Team, db

    # ── 阶段 1(核心):模型哈希 —— 失败即不可复现,raise ──────────────────
    from app.core.config import LeagueType as _LT

    _mpath = (
        _model_path(_LT(league.league_type), models_dir)
        if hasattr(league, "league_type")
        else None
    )
    if _mpath is None:
        raise RuntimeError("快照失败:无法定位模型文件路径(不可复现)")
    with open(_mpath, "rb") as _mf:
        _model_hash = hashlib.sha256(_mf.read()).hexdigest()[:64]
    # 审查 P0-4:data_hash = 参与预测历史记录的内容哈希(任何内容修改都可见);
    # hist_max_id/updated 保留为辅助诊断字段
    _data_hash = (
        data_content_hash(hist_df)
        or hashlib.sha256(
            f"{hist_max_id}|{hist_max_updated}|{home_team}|{away_team}".encode()
        ).hexdigest()[:64]
    )

    # ── 阶段 2(非核心诊断):失败仅 warning ────────────────────────────────
    _team_ids = {}
    try:
        _team_ids = {
            t.name: t.id
            for t in Team.query.filter(Team.name.in_([home_team, away_team])).all()
        }
    except Exception as e:
        logger.warning("快照:球队 id 提取失败(降级): %s", e)
    try:
        from app.models.utils import prematch_elo

        _he_o, _ae_o, _ = prematch_elo(hist_df, home_team, away_team, "overall")
        _he_a, _ae_a, _att = prematch_elo(hist_df, home_team, away_team, "attack")
    except Exception:
        _he_o = _ae_o = _he_a = _ae_a = _att = None
    # 审查 A70A601 P1-2:权重损坏(存在但解析/非法)由 load_weights raise、
    # 此处不吞 —— 快照不得在"伪装成默认权重"的状态下生成
    from app.models.ensemble import load_weights

    _ens_w = load_weights(league.league_type)
    try:
        with open(
            os.path.join(
                str(
                    __import__(
                        "app.core.paths", fromlist=["ARTIFACTS_DIR"]
                    ).ARTIFACTS_DIR
                ),
                "ensemble",
                "dc_nb_params.json",
            ),
            encoding="utf-8",
        ) as _dcp_f:
            _dcp = json.load(_dcp_f).get(league.league_type, {})
    except Exception:
        _dcp = {}
    # 审查 A70A601 §22:Feature Contribution(HGBR permutation importance)
    _feature_contrib = None
    try:
        _imp = getattr(model, "feature_importance_", None)
        if _imp is not None and len(_imp):
            _s = _imp.sort_values()
            _contrib = [
                {"feature": str(k), "value": round(float(v), 4)}
                for k, v in _s.items()
                if not (isinstance(v, float) and __import__("math").isnan(v))
            ]
            if _contrib:
                _feature_contrib = {
                    "method": getattr(model, "importance_method_", "permutation"),
                    "top_positive": list(reversed(_contrib[-5:])),
                    "top_negative": _contrib[:5],
                }
    except Exception:
        _feature_contrib = None
    # 审查 A70A601 P1-4:Bayes 成员版本冻结(公式哈希 + 超参常量)
    _bayes_ver = _bayes_kappa_hist = _bayes_kappa_recent = None
    try:
        from app.models import bayes_team as _bt
        from app.models.bayes_team import version as _bv

        _bayes_ver = _bv()
        _bayes_kappa_hist = _bt.LEAGUE_KAPPA
        _bayes_kappa_recent = _bt.RECENT_KAPPA
    except Exception:
        _bayes_ver = None

    # ── 阶段 3(核心):最终输入特征(100% 重放)── 失败 raise ───────────────
    if _feat is None:
        _feat = _m.prepare_features(_pred_df)
    _fcols = [
        col for col in getattr(model, "feature_columns_", []) if col in _feat.columns
    ]
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
        _gbm_path = os.path.join(
            str(__import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR),
            league.league_type,
            "gbm.pkl",
        )
        with open(_gbm_path, "rb") as _gf:
            _gbm_hash = hashlib.sha256(_gf.read()).hexdigest()[:12]
    except OSError:
        _gbm_hash = None
    _ens_hash = None
    try:
        _ens_dir = os.path.join(
            str(__import__("app.core.paths", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR),
            "ensemble",
        )
        _ens_raw = ""
        for _ef in ("ensemble_weights.json", "dc_nb_params.json"):
            with open(os.path.join(_ens_dir, _ef), "rb") as _f2:
                _ens_raw += _f2.read().hex()
        _ens_hash = hashlib.sha256(_ens_raw.encode()).hexdigest()[:12]
    except OSError:
        _ens_hash = None

    # ── 审查九 P1-6:四 Cutoff 时间体系 ────────────────────────────
    # prediction_cutoff:预测执行时刻;data/feature_cutoff:实际参与预测的
    # 历史数据截止(严格早于该场);model_cutoff:模型训练截止(artifact 元数据)
    try:
        _data_cutoff = (
            str(pd.to_datetime(hist_df["date"]).max().date()) if len(hist_df) else None
        )
    except Exception:
        _data_cutoff = None
    _model_cutoff = None
    try:
        _art_path = _mpath + ".json"
        if os.path.exists(_art_path):
            with open(_art_path, encoding="utf-8") as _af:
                _model_cutoff = json.load(_af).get("trained_at")
    except Exception:
        _model_cutoff = None
    try:
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        _pred_cutoff = _dt.now(tz=_tz.utc).isoformat(timespec="seconds")
    except Exception:
        _pred_cutoff = None

    # ── 审查九 P1-7:预测管线版本(逻辑代码哈希,非仅模型/特征哈希)──
    try:
        from app.prediction.versions import PIPELINE_VERSION as _pv
        from app.prediction.versions import pipeline_hash as _ph

        _pipeline_ver, _pipeline_hash = _pv, _ph()
    except Exception:
        _pipeline_ver, _pipeline_hash = None, None

    _snapshot = {
        "data_version": _data_hash[:12],
        "feature_version": _feature_ver,
        "model_version": _model_hash[:12],
        "evaluation_mode": evaluation_mode,
        "prediction_cutoff": _pred_cutoff,
        "data_cutoff": _data_cutoff,
        "feature_cutoff": _data_cutoff,
        "model_cutoff": _model_cutoff,
        "pipeline_version": _pipeline_ver,
        "pipeline_hash": _pipeline_hash,
        "model_set": {
            "goal": {
                "version": os.path.basename(_mpath).replace(".pkl", ""),
                "sha256": _model_hash,
            },
            "gbm": {"sha256": _gbm_hash} if _gbm_hash else None,
            "ensemble": {"sha256": _ens_hash} if _ens_hash else None,
            # 审查 A70A601 P1-4:Bayes 成员正式冻结(先验/超参已含于 data_hash
            # 与公式哈希;version 覆盖常数与实现变化)
            "bayes": (
                {
                    "version": _bayes_ver,
                    "kappa_hist": _bayes_kappa_hist,
                    "kappa_recent": _bayes_kappa_recent,
                }
                if _bayes_ver
                else None
            ),
            "calibration": (
                {
                    "method": _cal_info["method"],
                    "sha256": _cal_info.get("artifact_hash"),
                }
                if _cal_info
                else None
            ),
        },
        "score_matrix": (_score_matrix if _score_matrix is not None else []),
        # 审查 P1-11:版本 = artifact 哈希(唯一标识);无哈希时回退 method:n
        "calibration_version": (
            f"{_cal_info['method']}:{_cal_info['artifact_hash']}"
            if _cal_info and _cal_info.get("artifact_hash")
            else (f"{_cal_info['method']}:{_cal_info['n']}" if _cal_info else None)
        ),
        "home_features": _home_features,  # P0-5:-2 行 = 主队预测特征
        "away_features": _away_features,  # P0-5:-1 行 = 客队预测特征
        "lambda": [home_lambda, away_lambda],
        "prematch_elo": {
            "home": _he_o,
            "away": _ae_o,
            "attack_diff": _att if _att is not None else _att_diff,
        },
        "feature_columns": list(getattr(model, "feature_columns_", None) or []),
        "model_params": getattr(model, "config", None).parameters
        if getattr(model, "config", None)
        else None,
        "ensemble_weights": {
            k: v
            for k, v in _ens_w.items()
            if k in ("hgbr", "dc", "nb", "elo", "gbm", "bayes")
        },
        "dc_tau": _dcp.get("tau"),
        "feature_contribution": _feature_contrib,
        "nb_phi": _dcp.get("phi"),
        "calibration": _cal_info,
        "prior_blend": result.get("prior_blend"),
    }
    _probs = {
        # 审查 f01d7e4 P1-6:校准**输入**(校准前最终 1X2)—— fit_calibration
        # 用它训练,保证与生产校准输入同分布(旧快照无此键 → fallback 使用
        # home_win/draw/away_win)
        "pre_calibration": result.get("_calibration_input_1x2"),
        "home_win": result["home_win_probability"],
        "draw": result["draw_probability"],
        "away_win": result["away_win_probability"],
        "confidence_score": result.get("confidence_score"),
        "prediction_entropy": result.get("prediction_entropy"),
        "model_disagreement": result.get("model_disagreement"),
        "data_quality_score": result.get("data_quality_score"),
        "top_scores": result.get("top_scores", []),
        "over_2_5": result.get("over_2_5"),
        "under_2_5": result.get("under_2_5"),
        "btts": result.get("btts"),
        "expected_xg": result.get("expected_xg"),
    }
    _sha = hashlib.sha256(
        (
            _data_hash
            + _model_hash
            + json.dumps(_snapshot, sort_keys=True, default=str)
        ).encode()
    ).hexdigest()
    _kickoff = match_dt if match_date else match_dt.normalize()
    _existing = PredictionSnapshot.query.filter_by(
        league_id=league.id,
        home_team=home_team,
        away_team=away_team,
        kickoff=_kickoff.to_pydatetime()
        if hasattr(_kickoff, "to_pydatetime")
        else _kickoff,
    ).first()
    _snap_data = dict(  # noqa: C408 (关键字式动态键)
        league_id=league.id,
        home_team=home_team,
        away_team=away_team,
        home_team_id=_team_ids.get(home_team),
        away_team_id=_team_ids.get(away_team),
        kickoff=_kickoff,
        model_version=_model_hash[:12],
        feature_version=_feature_ver,
        data_hash=_data_hash,
        model_hash=_model_hash,
        sha256=_sha,
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


def data_content_hash(hist_df) -> str:
    """审查 P0-4:数据内容哈希 —— 对实际参与预测的历史记录内容做聚合哈希。

    原 data_hash 仅基于 hist_max_id|hist_max_updated|teams:历史中任何一条
    记录被修改(比分/日期/球队)而 id/updated 不变时,哈希不变 → 快照误判
    "数据未变"。内容哈希对每行 (date|home|away|hg|ga) 拼串后 sha256,
    任何参与预测的历史内容变化都会改变哈希。
    """
    cols = [
        col
        for col in ("date", "home_team", "away_team", "home_goals", "away_goals")
        if col in hist_df.columns
    ]
    if not cols:
        return ""
    s = hist_df[cols].astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:64]


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
        col: (
            None if pd.isna(feat.iloc[-2][col]) else round(float(feat.iloc[-2][col]), 6)
        )
        for col in feature_columns
    }
    away = {
        col: (
            None if pd.isna(feat.iloc[-1][col]) else round(float(feat.iloc[-1][col]), 6)
        )
        for col in feature_columns
    }
    return home, away
