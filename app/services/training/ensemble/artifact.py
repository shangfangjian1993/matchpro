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
    """Calibration 子-artifact。"""
    method: str  # "beta", "platt", "isotonic"
    artifact_hash: str
    training_cutoff: str  # ISO date
    temporal_oof: bool
    val_ece: float
    test_ece: Optional[float] = None


@dataclass(frozen=True)
class PriorArtifact:
    """Prior 子-artifact。"""
    window: int
    alpha: float
    min_history: int


@dataclass(frozen=True)
class LineageInfo:
    """Artifact 血统信息。"""
    artifact_version: str = "ensemble-v3"
    model_version: str = ""
    feature_version: str = ""
    training_cutoff: str = ""  # ISO date
    oof_method: str = "expanding-window"
    oof_segments: int = 6
    oof_n: int = 0
    shrinkage: float = 0.15
    created_at: str = ""


@dataclass(frozen=True)
class ProductionArtifact:
    """生产模型完整 artifact(唯一输入)。"""
    # Ensemble weights
    goal_lambda: dict  # {hgbr, elo, bayes}
    score_distribution: dict  # {poisson, dc, nb}
    outcome: dict  # {shape, gbm}
    
    # Distribution parameters
    tau: float
    phi: float
    
    # Sub-artifacts
    calibration: Optional[CalibrationArtifact] = None
    prior: Optional[PriorArtifact] = None
    lineage: LineageInfo = field(default_factory=LineageInfo)
    
    def to_dict(self) -> dict:
        """序列化为 dict。"""
        return {
            "goal_lambda": self.goal_lambda,
            "score_distribution": self.score_distribution,
            "outcome": self.outcome,
            "tau": self.tau,
            "phi": self.phi,
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
            goal_lambda=data["goal_lambda"],
            score_distribution=data["score_distribution"],
            outcome=data["outcome"],
            tau=data["tau"],
            phi=data["phi"],
            calibration=cal,
            prior=prior,
            lineage=lineage,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> ProductionArtifact:
        """从 JSON 反序列化。"""
        return cls.from_dict(json.loads(json_str))
    
    def content_hash(self) -> str:
        """内容哈希(用于 snapshot 冻结)。"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_production_artifact(
    weights: dict,
    tau: float,
    phi: float,
    model_version: str = "",
    feature_version: str = "",
    oof_n: int = 0,
    shrinkage: float = 0.15,
) -> ProductionArtifact:
    """从训练输出创建 ProductionArtifact。"""
    return ProductionArtifact(
        goal_lambda={
            "hgbr": weights.get("hgbr", 0.5),
            "elo": weights.get("elo", 0.3),
            "bayes": weights.get("bayes", 0.2),
        },
        score_distribution={
            "poisson": weights.get("poisson", 0.5),
            "dc": weights.get("dc", 0.3),
            "nb": weights.get("nb", 0.2),
        },
        outcome={
            "shape": weights.get("shape_weight", 1.0),
            "gbm": weights.get("gbm_weight", 0.0),
        },
        tau=tau,
        phi=phi,
        lineage=LineageInfo(
            model_version=model_version,
            feature_version=feature_version,
            training_cutoff=datetime.now(timezone.utc).isoformat(),
            oof_segments=6,
            oof_n=oof_n,
            shrinkage=shrinkage,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
