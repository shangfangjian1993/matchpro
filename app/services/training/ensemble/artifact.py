"""Production Artifact — 统一模型版本(绑定 Ensemble + τ/φ + Prior + Calibration + Lineage)。

ProductionArtifact 是 Production 的唯一输入,包含所有 learned parameters 和 metadata。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class CalibrationArtifact:
    """Calibration 子-artifact(包含真实 learned state)。"""
    method: str  # "beta", "platt", "isotonic"
    artifact_hash: str
    training_cutoff: str  # ISO date (真实数据截止时间)
    temporal_oof: bool
    val_ece: float
    test_ece: float | None = None
    params: dict = field(default_factory=dict)  # learned parameters


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
    """生产模型完整 artifact(唯一输入)。
    
    P0-2: 使用 typed schema (GoalLambdaWeights/ScoreDistributionWeights/OutcomeWeights)。
    """
    league: str  # "premier_league", "la_liga", etc.
    goal_lambda: dict  # P0-2: typed via __post_init__ validation
    score_distribution: dict
    outcome: dict
    tau: float
    phi: float
    gbm_model_path: str = ""
    gbm_model_hash: str = ""
    calibration: CalibrationArtifact | None = None
    prior: PriorArtifact | None = None
    lineage: LineageInfo = field(default_factory=LineageInfo)
    
    def __post_init__(self):
        """验证权重合法性 (finite, 0<=w<=1, sum=1)。P1-2: 缺失 key → fail。"""
        import math
        
        # Layer-1: require mandatory keys, allow extras
        mandatory_gl = {"hgbr", "elo", "bayes"}
        missing_gl = mandatory_gl - set(self.goal_lambda.keys())
        if missing_gl:
            raise ValueError(f"Layer-1 missing mandatory keys: {missing_gl}")
        gl_sum = 0.0
        for k in mandatory_gl:
            v = float(self.goal_lambda[k])
            if not math.isfinite(v):
                raise ValueError(f"Layer-1 weight {k}={v} is not finite")
            if v < 0 or v > 1:
                raise ValueError(f"Layer-1 weight {k}={v} out of [0,1]")
            gl_sum += v
        if abs(gl_sum - 1.0) > 1e-9:
            raise ValueError(f"Layer-1 weights sum={gl_sum:.4f}, expected 1.0")
        
        # Layer-2: require mandatory keys, allow extras
        mandatory_sd = {"poisson", "dc", "nb"}
        missing_sd = mandatory_sd - set(self.score_distribution.keys())
        if missing_sd:
            raise ValueError(f"Layer-2 missing mandatory keys: {missing_sd}")
        sd_sum = 0.0
        for k in mandatory_sd:
            v = float(self.score_distribution[k])
            if not math.isfinite(v):
                raise ValueError(f"Layer-2 weight {k}={v} is not finite")
            if v < 0 or v > 1:
                raise ValueError(f"Layer-2 weight {k}={v} out of [0,1]")
            sd_sum += v
        if abs(sd_sum - 1.0) > 1e-9:
            raise ValueError(f"Layer-2 weights sum={sd_sum:.4f}, expected 1.0")
        
        # Layer-3: require mandatory keys
        mandatory_oc = {"shape", "gbm"}
        missing_oc = mandatory_oc - set(self.outcome.keys())
        if missing_oc:
            raise ValueError(f"Layer-3 missing mandatory keys: {missing_oc}")
        shape = float(self.outcome["shape"])
        gbm = float(self.outcome["gbm"])
        if not math.isfinite(shape) or not math.isfinite(gbm):
            raise ValueError(f"Layer-3 weights not finite: shape={shape}, gbm={gbm}")
        if shape < 0 or shape > 1 or gbm < 0 or gbm > 1:
            raise ValueError(f"Layer-3 weights out of [0,1]: shape={shape}, gbm={gbm}")
        if abs(shape + gbm - 1.0) > 1e-9:
            raise ValueError(f"Layer-3 weights sum={shape + gbm:.4f}, expected 1.0")
        
        # Validate tau (Dixon-Coles low-score correction)
        if not math.isfinite(self.tau):
            raise ValueError(f"tau={self.tau} is not finite")
        if abs(self.tau) > 0.5:
            raise ValueError(f"tau={self.tau} exceeds reasonable range [-0.5, 0.5]")
        # Validate phi (Negative Binomial overdispersion)
        if not math.isfinite(self.phi) or self.phi <= 0:
            raise ValueError(f"phi={self.phi} must be finite and > 0")
    
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
        """从 dict 反序列化。
        
        P0-2 FIX: 正确恢复 calibration 和 prior。
        """
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
            calibration=cal,  # P0-2 FIX: 传入 calibration
            prior=prior,       # P0-2 FIX: 传入 prior
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


class ArtifactValidationError(Exception):
    """ProductionArtifact validation failed."""


def create_production_artifact(
    league: str,
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
    calibration: CalibrationArtifact | None = None,
    prior: PriorArtifact | None = None,
    oof_segments: int = 6,
) -> ProductionArtifact:
    """从训练输出创建 ProductionArtifact。

    P1-5: training_cutoff 由调用方传入(真实数据截止时间)。
    P1-2: oof_segments 由调用方传入(真实 K_SEG)。
    """
    required_keys = {"hgbr", "elo", "bayes", "poisson", "dc", "nb", "shape", "gbm"}
    missing = required_keys - set(weights.keys())
    if missing:
        raise ArtifactValidationError(f"Missing required weights: {missing}")
    
    return ProductionArtifact(
        league=league,
        goal_lambda={
            "hgbr": float(weights["hgbr"]),
            "elo": float(weights["elo"]),
            "bayes": float(weights["bayes"]),
        },
        score_distribution={
            "poisson": float(weights["poisson"]),
            "dc": float(weights["dc"]),
            "nb": float(weights["nb"]),
        },
        outcome={
            "shape": float(weights["shape"]),
            "gbm": float(weights["gbm"]),
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
            training_cutoff=training_cutoff,  # P1-5: 真实 cutoff
            oof_segments=oof_segments,  # P1-2: 真实 K_SEG
            oof_n=oof_n,
            shrinkage=shrinkage,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
