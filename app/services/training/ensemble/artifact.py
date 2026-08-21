"""Production Artifact — 统一模型版本(绑定 Ensemble + τ/φ + Prior + Calibration + Lineage)。

ProductionArtifact 是 Production 的唯一输入,包含所有 learned parameters 和 metadata。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
import hashlib


@dataclass(frozen=True)
class CalibrationArtifact:
    """Calibration 子-artifact(包含真实 learned state)。"""
    method: str  # "beta", "platt", "isotonic"
    artifact_hash: str
    training_cutoff: str  # ISO date (真实数据截止时间)
    temporal_oof: bool
    val_ece: float
    test_ece: Optional[float] = None
    params: dict = field(default_factory=dict)  # learned parameters


@dataclass(frozen=True)
class PriorArtifact:
    """Prior 子-artifact。"""
    window: int
    alpha: float
    min_history: int


@dataclass(frozen=True)
class GbmArtifact:
    """GBM 子-artifact(包含模型引用)。"""
    model_path: str  # 相对路径
    model_hash: str  # SHA256 of model file
    feature_columns: list = field(default_factory=list)


@dataclass(frozen=True)
class LineageInfo:
    """Artifact 血统信息。"""
    artifact_version: str = "ensemble-v3"
    schema_version: int = 1
    model_version: str = ""
    feature_version: str = ""
    training_cutoff: str = ""
    oof_method: str = "expanding-window"
    oof_segments: int = 6
    oof_n: int = 0
    shrinkage: float = 0.15
    created_at: str = ""
    training_data_hash: str = ""
    calibration_hash: str = ""
    prior_hash: str = ""
    gbm_hash: str = ""


@dataclass(frozen=True)
class ProductionArtifact:
    """生产模型完整 artifact(唯一输入)。"""
    # P0-4: 增加 league 字段
    league: str  # "premier_league", "la_liga", etc.
    
    # Ensemble weights (typed - P0-2)
    goal_lambda: dict  # {hgbr, elo, bayes}
    score_distribution: dict  # {poisson, dc, nb}
    outcome: dict  # {shape, gbm}
    
    # Distribution parameters
    tau: float
    phi: float
    
    # P0-3: GBM learned state
    gbm_model_path: str = ""
    gbm_model_hash: str = ""
    
    # Sub-artifacts
    calibration: Optional[CalibrationArtifact] = None
    prior: Optional[PriorArtifact] = None
    lineage: LineageInfo = field(default_factory=LineageInfo)
    
    def to_dict(self) -> dict:
        """序列化为 dict(P1-6: 全精度,无 round/renormalize)。"""
        return {
            "league": self.league,
            "goal_lambda": {k: float(v) for k, v in self.goal_lambda.items()},
            "score_distribution": {k: float(v) for k, v in self.score_distribution.items()},
            "outcome": {k: float(v) for k, v in self.outcome.items()},
            "tau": float(self.tau),
            "phi": float(self.phi),
            "gbm_model_path": self.gbm_model_path,
            "gbm_model_hash": self.gbm_model_hash,
            "calibration": self.calibration.__dict__ if self.calibration else None,
            "prior": self.prior.__dict__ if self.prior else None,
            "lineage": self.lineage.__dict__,
        }
    
    def to_json(self) -> str:
        """序列化为 JSON。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> ProductionArtifact:
        """从 dict 反序列化。"""
        cal = None
        if data.get("calibration"):
            cal = CalibrationArtifact(**data["calibration"])
        prior = None
        if data.get("prior"):
            prior = PriorArtifact(**data["prior"])
        lineage = LineageInfo(**data.get("lineage", {}))
        return cls(
            league=data.get("league", ""),
            goal_lambda=data["goal_lambda"],
            score_distribution=data["score_distribution"],
            outcome=data["outcome"],
            tau=data["tau"],
            phi=data["phi"],
            gbm_model_path=data.get("gbm_model_path", ""),
            gbm_model_hash=data.get("gbm_model_hash", ""),
            calibration=cal,
            prior=prior,
            lineage=lineage,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> ProductionArtifact:
        """从 JSON 反序列化。"""
        return cls.from_dict(json.loads(json_str))
    
    def model_hash(self) -> str:
        """模型内容哈希(不含 created_at)。"""
        content = json.dumps({
            "league": self.league,
            "goal_lambda": self.goal_lambda,
            "score_distribution": self.score_distribution,
            "outcome": self.outcome,
            "tau": self.tau,
            "phi": self.phi,
            "gbm_model_hash": self.gbm_model_hash,
            "calibration": self.calibration.__dict__ if self.calibration else None,
            "prior": self.prior.__dict__ if self.prior else None,
            "training_cutoff": self.lineage.training_cutoff,
            "oof_segments": self.lineage.oof_segments,
            "shrinkage": self.lineage.shrinkage,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def artifact_hash(self) -> str:
        """完整 artifact 哈希(含 created_at)。"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_production_artifact(
    league: str,  # P0-4: 必须传入 league
    weights: dict,
    tau: float,
    phi: float,
    model_version: str = "",
    feature_version: str = "",
    oof_n: int = 0,
    shrinkage: float = 0.15,
    training_cutoff: str = "",
    gbm_model_path: str = "",
    gbm_model_hash: str = "",
    calibration: Optional[CalibrationArtifact] = None,
    prior: Optional[PriorArtifact] = None,
) -> ProductionArtifact:
    """从训练输出创建 ProductionArtifact。"""
    return ProductionArtifact(
        league=league,  # P0-4
        goal_lambda={
            "hgbr": float(weights.get("hgbr", 0.5)),
            "elo": float(weights.get("elo", 0.3)),
            "bayes": float(weights.get("bayes", 0.2)),
        },
        score_distribution={
            "poisson": float(weights.get("poisson", 0.5)),
            "dc": float(weights.get("dc", 0.3)),
            "nb": float(weights.get("nb", 0.2)),
        },
        outcome={
            "shape": float(weights.get("shape_weight", 1.0)),
            "gbm": float(weights.get("gbm_weight", 0.0)),
        },
        tau=float(tau),
        phi=float(phi),
        gbm_model_path=gbm_model_path,
        gbm_model_hash=gbm_model_hash,
        calibration=calibration,
        prior=prior,
        lineage=LineageInfo(
            model_version=model_version,
            feature_version=feature_version,
            training_cutoff=training_cutoff,
            oof_segments=6,
            oof_n=oof_n,
            shrinkage=shrinkage,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
