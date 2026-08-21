"""Weight Stability Reporting (opt-5)。

统计权重跨 season / OOF fold 的稳定性:
- mean / std / p10 / p50 / p90
- 按 Layer-1 / Layer-2 / Layer-3 分组
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeightStatistics:
    """权重统计。"""
    mean: float = 0.0
    std: float = 0.0
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    n: int = 0


@dataclass
class LayerStabilityReport:
    """单层稳定性报告。"""
    layer_name: str = ""
    members: dict = field(default_factory=dict)  # {member_name: WeightStatistics}


@dataclass
class StabilityReport:
    """完整稳定性报告。"""
    league: str = ""
    n_folds: int = 0
    n_seasons: int = 0
    layer1: Optional[LayerStabilityReport] = None
    layer2: Optional[LayerStabilityReport] = None
    layer3: Optional[LayerStabilityReport] = None
    
    def to_dict(self) -> dict:
        return {
            "league": self.league,
            "n_folds": self.n_folds,
            "n_seasons": self.n_seasons,
            "layer1": self.layer1.members if self.layer1 else {},
            "layer2": self.layer2.members if self.layer2 else {},
            "layer3": self.layer3.members if self.layer3 else {},
        }


def _compute_stats(values: list[float]) -> WeightStatistics:
    """计算统计量。"""
    if not values:
        return WeightStatistics()
    
    arr = np.array(values, dtype=float)
    return WeightStatistics(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        p10=float(np.percentile(arr, 10)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        n=len(values),
    )


def analyze_weight_stability(
    league: str,
    weight_history: list[dict],  # [{layer: {member: weight}}, ...]
    n_seasons: int = 0,
) -> StabilityReport:
    """分析权重稳定性。
    
    Args:
        league: 联赛名
        weight_history: 权重历史列表(每个元素是一个 dict,包含各层权重)
        n_seasons: 赛季数
    
    Returns:
        StabilityReport
    """
    report = StabilityReport(
        league=league,
        n_folds=len(weight_history),
        n_seasons=n_seasons,
    )
    
    # Layer-1
    l1_members = {"hgbr": [], "elo": [], "bayes": []}
    for wh in weight_history:
        gl = wh.get("goal_lambda", {})
        for m in l1_members:
            l1_members[m].append(gl.get(m, 0.0))
    
    report.layer1 = LayerStabilityReport(
        layer_name="goal_lambda",
        members={m: _compute_stats(v) for m, v in l1_members.items()},
    )
    
    # Layer-2
    l2_members = {"poisson": [], "dc": [], "nb": []}
    for wh in weight_history:
        sd = wh.get("score_distribution", {})
        for m in l2_members:
            l2_members[m].append(sd.get(m, 0.0))
    
    report.layer2 = LayerStabilityReport(
        layer_name="score_distribution",
        members={m: _compute_stats(v) for m, v in l2_members.items()},
    )
    
    # Layer-3
    l3_members = {"shape": [], "gbm": []}
    for wh in weight_history:
        oc = wh.get("outcome", {})
        for m in l3_members:
            l3_members[m].append(oc.get(m, 0.0))
    
    report.layer3 = LayerStabilityReport(
        layer_name="outcome",
        members={m: _compute_stats(v) for m, v in l3_members.items()},
    )
    
    return report
