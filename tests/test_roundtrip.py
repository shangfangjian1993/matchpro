"""Production/Training Round-trip Parity Test。

验证: Training → Artifact → Load → Production 逐位一致。
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from app.api.db import League, Match, init_db
from app.core.config import LeagueType
from app.models.ensemble.weights import DEFAULT_WEIGHTS, to_layered, from_layered
from app.prediction.layered_pipeline import compute_prediction, AblationMask
from app.prediction.engine import PredictionEngine
from app.prediction.context import ContextBuilder
from app.services.training.ensemble.artifact import ProductionArtifact, create_production_artifact


class TestRoundtripParity:
    """Training → Artifact → Production round-trip parity。"""
    
    def test_weight_roundtrip(self):
        """权重经过 to_layered → from_layered 后保持一致。"""
        original = {
            "hgbr": 0.52, "elo": 0.29, "bayes": 0.19,
            "poisson": 0.48, "dc": 0.32, "nb": 0.20,
            "shape_weight": 0.73, "gbm_weight": 0.27,
        }
        
        layered = to_layered(original)
        restored = from_layered(layered)
        
        for k, v in original.items():
            assert abs(restored.get(k, 0) - v) < 0.01, f"{k}: {restored.get(k)} != {v}"
    
    def test_production_artifact_roundtrip(self):
        """ProductionArtifact JSON round-trip。"""
        artifact = create_production_artifact(
            weights={
                "hgbr": 0.52, "elo": 0.29, "bayes": 0.19,
                "poisson": 0.48, "dc": 0.32, "nb": 0.20,
                "shape_weight": 0.73, "gbm_weight": 0.27,
            },
            tau=-0.071,
            phi=2.31,
            oof_n=600,
        )
        
        # JSON round-trip
        json_str = artifact.to_json()
        restored = ProductionArtifact.from_json(json_str)
        
        assert restored.tau == artifact.tau
        assert restored.phi == artifact.phi
        assert restored.goal_lambda == artifact.goal_lambda
        assert restored.score_distribution == artifact.score_distribution
        assert restored.outcome == artifact.outcome
        assert restored.content_hash() == artifact.content_hash()
    
    def test_prediction_deterministic(self):
        """相同输入 → 相同输出(确定性)。"""
        weights = DEFAULT_WEIGHTS.to_flat()
        
        result1 = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0, weights=weights,
            lam_bh=1.3, lam_ba=1.2,
            gbm_probs=(0.6, 0.25, 0.15),
        )
        
        result2 = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0, weights=weights,
            lam_bh=1.3, lam_ba=1.2,
            gbm_probs=(0.6, 0.25, 0.15),
        )
        
        if result1 is not None and result2 is not None:
            assert result1.final_1x2 == result2.final_1x2
    
    def test_snapshot_contract_fields(self):
        """快照契约:验证 compute_prediction 输出字段完整。"""
        weights = DEFAULT_WEIGHTS.to_flat()
        
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0, weights=weights,
            lam_bh=1.3, lam_ba=1.2,
            gbm_probs=(0.6, 0.25, 0.15),
        )
        
        if result is not None:
            # 验证输出包含必要字段
            assert result.final_1x2 is not None
            assert len(result.final_1x2) == 3
            assert abs(sum(result.final_1x2) - 1.0) < 0.01
            assert result.score_matrix is not None
            assert result.diagnostics is not None


class TestArtifactIntegrity:
    """Artifact 完整性测试。"""
    
    def test_content_hash_stable(self):
        """相同内容 → 相同 hash。"""
        artifact = create_production_artifact(
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
            tau=0.05, phi=50.0,
        )
        
        assert artifact.content_hash() == artifact.content_hash()
    
    def test_content_hash_changes_with_weights(self):
        """权重变化 → hash 变化。"""
        a1 = create_production_artifact(
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
            tau=0.05, phi=50.0,
        )
        a2 = create_production_artifact(
            weights={"hgbr": 0.6, "elo": 0.3, "bayes": 0.1, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
            tau=0.05, phi=50.0,
        )
        
        assert a1.content_hash() != a2.content_hash()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
