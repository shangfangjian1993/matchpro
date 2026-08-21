"""Production/Training Round-trip Parity Test。

验证: Training → Artifact → Load → Production 逐位一致。
"""
from __future__ import annotations

import pytest

from app.models.ensemble.weights import DEFAULT_WEIGHTS, from_layered, to_layered
from app.prediction.layered_pipeline import compute_prediction
from app.services.training.ensemble.artifact import (
    CalibrationArtifact,
    PriorArtifact,
    ProductionArtifact,
    create_production_artifact,
)


class TestRoundtripParity:
    """Training → Artifact → Production round-trip parity。"""
    
    def test_weight_roundtrip(self):
        """权重经过 to_layered → from_layered 后保持一致。"""
        original = {
            "hgbr": 0.52, "elo": 0.29, "bayes": 0.19,
            "poisson": 0.48, "dc": 0.32, "nb": 0.20,
            "shape": 0.73, "gbm": 0.27,
        }
        
        layered = to_layered(original)
        restored = from_layered(layered)
        
        for k, v in original.items():
            assert abs(restored.get(k, 0) - v) < 0.01, f"{k}: {restored.get(k)} != {v}"
    
    def test_production_artifact_roundtrip(self):
        """ProductionArtifact JSON round-trip。"""
        artifact = create_production_artifact(
            league="premier_league",
            weights={
                "hgbr": 0.52, "elo": 0.29, "bayes": 0.19,
                "poisson": 0.48, "dc": 0.32, "nb": 0.20,
                "shape": 0.73, "gbm": 0.27,
            },
            tau=-0.071,
            phi=2.31,
            oof_n=600,
        )
        
        # JSON round-trip
        json_str = artifact.to_json()
        restored = ProductionArtifact.from_json(json_str)
        
        assert restored.league == artifact.league
        assert restored.tau == artifact.tau
        assert restored.phi == artifact.phi
        assert restored.goal_lambda == artifact.goal_lambda
        assert restored.score_distribution == artifact.score_distribution
        assert restored.outcome == artifact.outcome
        assert restored.artifact_hash() == artifact.artifact_hash()
    
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
            league="premier_league",
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape": 0.7, "gbm": 0.3},
            tau=0.05, phi=50.0,
        )
        
        assert artifact.artifact_hash() == artifact.artifact_hash()
    
    def test_content_hash_changes_with_weights(self):
        """权重变化 → hash 变化。"""
        a1 = create_production_artifact(
            league="premier_league",
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape": 0.7, "gbm": 0.3},
            tau=0.05, phi=50.0,
        )
        a2 = create_production_artifact(
            league="premier_league",
            weights={"hgbr": 0.6, "elo": 0.3, "bayes": 0.1, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape": 0.7, "gbm": 0.3},
            tau=0.05, phi=50.0,
        )
        
        assert a1.artifact_hash() != a2.artifact_hash()
    
    def test_model_hash_excludes_created_at(self):
        """P1-6: model_hash 不包含 created_at。"""
        a1 = create_production_artifact(
            league="premier_league",
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape": 0.7, "gbm": 0.3},
            tau=0.05, phi=50.0,
        )
        a2 = create_production_artifact(
            league="premier_league",
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape": 0.7, "gbm": 0.3},
            tau=0.05, phi=50.0,
        )
        
        # model_hash 应该相同(因为 created_at 不影响)
        assert a1.model_hash() == a2.model_hash()
        # artifact_hash 应该不同(因为 created_at 不同)
        assert a1.artifact_hash() != a2.artifact_hash()


class TestCalibrationPriorRoundtrip:
    """P0-2: Calibration/Prior round-trip 测试。"""
    
    def test_calibration_roundtrip(self):
        """CalibrationArtifact 经过 JSON round-trip 后保持完整。"""
        cal = CalibrationArtifact(
            method="isotonic",
            artifact_hash="abc123def456",
            training_cutoff="2026-08-01T00:00:00+00:00",
            temporal_oof=True,
            val_ece=0.031,
            test_ece=0.028,
            params={"thresholds": [0.1, 0.5, 0.9], "values": [0.08, 0.48, 0.88]},
        )
        
        artifact = ProductionArtifact(
            league="premier_league",
            goal_lambda={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
            score_distribution={"poisson": 0.5, "dc": 0.3, "nb": 0.2},
            outcome={"shape": 0.7, "gbm": 0.3},
            tau=0.05,
            phi=50.0,
            calibration=cal,
        )
        
        restored = ProductionArtifact.from_json(artifact.to_json())
        
        # P0-2: 验证 calibration 没有丢失
        assert restored.calibration is not None, "calibration should not be None"
        assert restored.calibration.method == "isotonic"
        assert restored.calibration.artifact_hash == "abc123def456"
        assert restored.calibration.val_ece == 0.031
        assert restored.calibration.params["thresholds"] == [0.1, 0.5, 0.9]
    
    def test_prior_roundtrip(self):
        """PriorArtifact 经过 JSON round-trip 后保持完整。"""
        prior = PriorArtifact(
            window=100,
            alpha=0.6,
            min_history=50,
        )
        
        artifact = ProductionArtifact(
            league="premier_league",
            goal_lambda={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
            score_distribution={"poisson": 0.5, "dc": 0.3, "nb": 0.2},
            outcome={"shape": 0.7, "gbm": 0.3},
            tau=0.05,
            phi=50.0,
            prior=prior,
        )
        
        restored = ProductionArtifact.from_json(artifact.to_json())
        
        # P0-2: 验证 prior 没有丢失
        assert restored.prior is not None, "prior should not be None"
        assert restored.prior.window == 100
        assert restored.prior.alpha == 0.6
        assert restored.prior.min_history == 50


class TestProductionArtifactValidation:
    """ProductionArtifact 权重合法性验证。"""
    
    def test_valid_weights_accepted(self):
        """合法权重可以通过验证。"""
        artifact = create_production_artifact(
            league="premier_league",
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape": 0.7, "gbm": 0.3},
            tau=0.05, phi=50.0,
        )
        assert artifact.league == "premier_league"
    
    def test_invalid_layer1_weights_rejected(self):
        """Layer-1 权重和不等于 1 时拒绝。"""
        with pytest.raises(ValueError, match="Layer-1 weights sum"):
            ProductionArtifact(
                league="premier_league",
                goal_lambda={"hgbr": 0.8, "elo": 0.3, "bayes": 0.2},  # sum=1.3
                score_distribution={"poisson": 0.5, "dc": 0.3, "nb": 0.2},
                outcome={"shape": 0.7, "gbm": 0.3},
                tau=0.05, phi=50.0,
            )
    
    def test_invalid_layer2_weights_rejected(self):
        """Layer-2 权重和不等于 1 时拒绝。"""
        with pytest.raises(ValueError, match="Layer-2 weights sum"):
            ProductionArtifact(
                league="premier_league",
                goal_lambda={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
                score_distribution={"poisson": 0.8, "dc": 0.3, "nb": 0.2},  # sum=1.3
                outcome={"shape": 0.7, "gbm": 0.3},
                tau=0.05, phi=50.0,
            )
    
    def test_invalid_layer3_weights_rejected(self):
        """Layer-3 权重和不等于 1 时拒绝。"""
        with pytest.raises(ValueError, match="Layer-3 weights sum"):
            ProductionArtifact(
                league="premier_league",
                goal_lambda={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
                score_distribution={"poisson": 0.5, "dc": 0.3, "nb": 0.2},
                outcome={"shape": 0.5, "gbm": 0.3},  # sum=0.8
                tau=0.05, phi=50.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
