"""训练配置(禁止magic numbers)。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleTrainingConfig:
    """Ensemble训练配置(不可变,保存到artifact metadata)。"""
    k_seg: int = 6
    sample_per_seg: int = 120
    min_oof_samples: int = 300
    shrinkage: float = 0.15
    max_weight: float = 0.7
    tau_min: float = -0.2
    tau_max: float = 0.2
    tau_step: float = 0.01
    phi_min: float = 1.0
    phi_max: float = 100.0
    outcome_method: str = "bounded"
    calibration_temporal_oof: bool = False
    tau_warning_threshold: float = 0.15
    phi_warning_threshold: float = 50.0
    max_goals: int = 10

    def to_dict(self) -> dict:
        return {
            "k_seg": self.k_seg,
            "sample_per_seg": self.sample_per_seg,
            "min_oof_samples": self.min_oof_samples,
            "shrinkage": self.shrinkage,
            "max_weight": self.max_weight,
            "tau_range": [self.tau_min, self.tau_max],
            "tau_step": self.tau_step,
            "phi_range": [self.phi_min, self.phi_max],
            "outcome_method": self.outcome_method,
            "calibration_temporal_oof": self.calibration_temporal_oof,
            "max_goals": self.max_goals,
        }


DEFAULT_CONFIG = EnsembleTrainingConfig()
