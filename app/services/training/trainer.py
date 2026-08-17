"""训练 + 版本管理 + 实验/特征注册。"""
import json
import logging
import math
import os
from app.core.timeutil import utcnow

import numpy as np
import pandas as pd

from app.api.db import League, Match
from app.core.config import TOURNAMENT_LEAGUE_TYPES as TOURNAMENT_TYPES
from app.core.config import LeagueType
from app.data.adapters import matches_to_dataframe
from app.models.integrity import _write_checksum
from app.models.registry import (
    _MODELS_KEEP,
    _bump_version,
    _model_path,
    _prune_old_versions,
    _version_lock,
    _version_unlock,
)
from app.services.training.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

def train_model(league_type: LeagueType, target_column: str = "goals",
                cross_validation: bool = True, cv_folds: int = 5,
                models_dir: str | None = None) -> dict:
    """
    用数据库中的历史比赛训练模型并保存。
    返回训练指标(与前端 /models/performance 期望字段一致)。
    """

    if models_dir is None:
        from app.core.paths import MODELS_DIR as _MD
        models_dir = str(_MD)
    league = League.query.filter_by(league_type=league_type.value).first()
    if league is None:
        raise ValueError(f"数据库中还没有 {league_type.value} 的联赛数据,请先录入比赛数据")

    matches = Match.query.filter_by(league_id=league.id, match_status="finished").all()
    df = matches_to_dataframe(matches)
    # ELO 特征注入(每场取赛前 rating,防泄漏)
    # §18:ELO 注入统一在 Feature Factory(prepare 内)

    # 与模型类 min_training_rows 一致(联赛 100 / 赛事 50);此处尚无 model 实例,按类型判断
    min_rows = 50 if league_type in TOURNAMENT_TYPES else 100
    if len(df) < min_rows:
        raise ValueError(f"{league_type.value} 的完赛比赛数据不足(需要至少 {min_rows} 条,当前 {len(df)} 条),无法训练")

    # 用 ModelTrainer 训练:裸 model.train() 只返回训练集指标(缺 exact_accuracy 等),
    # ModelTrainer.train_model 在其上补全时间 holdout evaluation(后 20%)与交叉验证
    trainer = ModelTrainer()
    results = trainer.train_model(df, league_type, target_column,
                                  cross_validation=cross_validation, cv_folds=cv_folds,
)
    model = trainer.model

    os.makedirs(models_dir, exist_ok=True)
    # 版本化保存:递增 patch 版本号,同时维护 latest 指针文件
    # (临界区加文件锁,避免并发训练同联赛时版本号竞争)
    lock = _version_lock(models_dir)
    try:
        version = _bump_version(league_type, models_dir)
        path = _model_path(league_type, models_dir, version)
        model.save_model(path)
        _write_checksum(path)
        # §33:artifact.json(版本/sha256/特征版本/训练时间/行数/类型)
        try:
            import hashlib as _ah
            from datetime import datetime as _dt
            _art = {
                "version": version,
                "sha256": _ah.sha256(open(path, "rb").read()).hexdigest(),
                "feature_version": getattr(model, "feature_version_", None),
                "trained_at": _dt.now().isoformat(),
                "training_rows": len(df),
                "model_type": getattr(model, "config", None).model_type if getattr(model, "config", None) else "HGBR",
                "poisson_loss": float(results.get("evaluation", {}).get("poisson_loss", 0.0) or 0.0),
            }
            with open(path + ".json", "w", encoding="utf-8") as _af:
                json.dump(_art, _af, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 元数据写入失败不影响训练
        # 审查 P0-8:不再复制 latest 指针文件(引用关系由 active_models.json +
        # 版本文件承担;避免同一模型两份实体)
        # 保留策略:清理超出保留数的旧版本
        _prune_old_versions(league_type, models_dir, keep=_MODELS_KEEP)

        # ---- V2:Feature Store 注册(特征版本化)----
        try:
            from app.features.registry import register as _register_features
            _feats = list(getattr(model, "feature_columns_", None) or [])
            if _feats:
                model.feature_version_ = _register_features(_feats, league_type.value)
        except Exception:
            pass  # 特征注册失败不影响训练

        # ---- V2:Experiment Tracking(训练实验自动记录,与 return 同口径)----
        # §8 补三分类概率指标:对 holdout 用泊松卷积算 logloss/brier/rps
        try:
            _prob_metrics = _holdout_prob_metrics(model, df, target_column)
        except Exception:
            _prob_metrics = {}
        try:
            import hashlib as _hl
            import json as _json
            from datetime import datetime as _dt

            from app.api.db import Experiment, db
            _ev = results.get("evaluation") or results.get("training_metrics", {})
            _acc = (_ev.get("accuracy_metrics") or {}) if isinstance(_ev, dict) else {}
            db.session.add(Experiment(
                league_type=league_type.value,
                dataset_version=f"matches_{len(df)}",
                feature_version=str(getattr(model, "feature_version_", "unknown")),
                model_version=str(version),
                train_start=results.get("_t0"),
                train_end=_dt.now(),
                hyperparameters_json=_json.dumps(
                    getattr(model, "config", None).parameters if getattr(model, "config", None) else {},
                    default=str),
                metrics_json=_json.dumps({
                    # 审查 P0-3:各指标口径必须可区分,不得把不同 holdout /
                    # 不同样本量的指标混成一个"模型分数"
                    "poisson_loss": float(_ev.get("poisson_loss", 0.0) or 0.0),
                    "accuracy": float(_acc.get("exact_accuracy", _ev.get("exact_accuracy", 0.0)) or 0.0),
                    "log_loss": float(_prob_metrics.get("log_loss", 0.0) or 0.0),
                    "brier": float(_prob_metrics.get("brier", 0.0) or 0.0),
                    "rps": float(_prob_metrics.get("rps", 0.0) or 0.0),
                    "feature_count": int(results.get("feature_count", 0) or 0),
                    "data_rows": len(df),
                    # ---- 口径标注(P0-3)----
                    "metric_sources": "poisson_loss=regression_holdout;log_loss/brier/rps=probability_holdout;cv=time_cv",
                    "split": "date-grouped-80-20",
                    "prob_holdout_n": int(_prob_metrics.get("n", 0) or 0),
                    "time_cv": (
                        {"mean": float(results["cross_validation"]["cv_mean"]),
                         "std": float(results["cross_validation"]["cv_std"]),
                         "folds": int(results["cross_validation"]["cv_folds"])}
                        if isinstance(results.get("cross_validation"), dict)
                        and results["cross_validation"].get("cv_mean") is not None
                        else None
                    ),
                }, default=str),
                data_hash=_hl.sha256(str(len(df)).encode()).hexdigest()[:16],
            ))
            db.session.commit()
        except Exception:
            pass  # 实验记录失败不影响训练
    finally:
        _version_unlock(lock)

    # 指标取时间 holdout 评估集(evaluation):训练集指标会虚高且缺 exact_accuracy
    metrics = results.get("evaluation") or results.get("training_metrics", {})
    acc_metrics = metrics.get("accuracy_metrics") or {} if isinstance(metrics, dict) else {}
    return {
        "league_type": league_type.value,
        "model_type": results.get("model_type"),
        "model_version": version,
        "training_date": utcnow().isoformat(),
        # np.float64 直传 psycopg2 会生成 "np.float64(...)" 字面量
        # 导致 InvalidSchemaName(SQLite 测试环境不报错),必须转原生类型
        "poisson_loss": float(metrics.get("poisson_loss", 0.0) or 0.0),
        "mse": float(metrics.get("mse", 0.0) or 0.0),
        "mae": float(metrics.get("mae", 0.0) or 0.0),
        "rmse": float(metrics.get("rmse", 0.0) or 0.0),
        "exact_accuracy": float(
            acc_metrics.get("exact_accuracy", metrics.get("exact_accuracy", 0.0)) or 0.0
        ),
        "feature_count": int(results.get("feature_count", 0) or 0),
        "model_path": path,
        "cross_validation": results.get("cross_validation"),
        "training_data_shape": results.get("training_data_shape"),
    }

def _holdout_prob_metrics(model, df, target_column: str = "goals") -> dict:
    """§8:对时间 holdout 子集计算三分类概率指标(logloss/brier/rps)。

    与 predict_match 同口径:每场构造主客两行独立预测(防泄漏:该场之前的历史),
    泊松卷积 → 胜平负概率 vs 实际。holdout 场次采样上限 200(性能)。
    """

    from app.models.ensemble import match_probs
    from app.replay.metrics import brier_score, log_loss
    from app.replay.metrics import rps as _rps
    if len(df) < 200:
        return {}
    # 审查 P1-7:按日期分组切分(同一比赛日不跨 train/test)
    from app.models.utils import date_group_split
    hist, test = date_group_split(df, ratio=0.8)
    test = test.reset_index(drop=True)
    # 采样上限 60(性能;指标为参考性,与回测/复盘互补)
    if len(test) > 60:
        test = test.iloc[:: max(1, len(test) // 60)]
    pvecs, acts = [], []
    for _, row in test.iterrows():
        match_dt = pd.Timestamp(row["date"])
        past = hist[hist["date"] < match_dt]
        if len(past) < 50:
            continue
        # 最近 500 场截断(特征已收敛,提速);ELO 由 prepare(factory)注入
        past = past.tail(500)
        lam_h = []
        for _row in ({"date": match_dt, "home_team": row["home_team"], "away_team": row["away_team"],
                      "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                      "league": row.get("league", ""), "season": row.get("season", "")},
                     {"date": match_dt, "home_team": row["away_team"], "away_team": row["home_team"],
                      "home_goals": np.nan, "away_goals": np.nan, "goals": np.nan,
                      "league": row.get("league", ""), "season": row.get("season", "")}):
            _df = pd.concat([past, pd.DataFrame([_row])], ignore_index=True)
            lam_h.append(float(model.predict(_df)["predictions"][-1]))
        p = match_probs(lam_h[0], lam_h[1])
        gh, ga = row.get("home_goals"), row.get("away_goals")
        if gh is None or ga is None or (isinstance(gh, float) and math.isnan(gh)):
            continue
        act = 0 if gh > ga else (1 if gh == ga else 2)
        pvecs.append(p)
        acts.append(act)
    if len(pvecs) < 30:
        return {}
    return {
        "log_loss": round(sum(log_loss(p, a) for p, a in zip(pvecs, acts)) / len(pvecs), 5),
        "brier": round(sum(brier_score(p, a) for p, a in zip(pvecs, acts)) / len(pvecs), 5),
        "rps": round(sum(_rps(p, a) for p, a in zip(pvecs, acts)) / len(pvecs), 5),
        "n": len(pvecs),
    }
