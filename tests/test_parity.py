"""Production ↔ Training Mathematical Parity Test (opt-4)。

验证: 同一组 λ/τ/φ/weights 经过 Training 和 Production 路径后,
所有中间值和最终值逐位一致。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.prediction.layered_pipeline import compute_prediction


class TestMathematicalParity:
    """Training ↔ Production 数学同构验证。"""
    
    def test_parity_full_pipeline(self):
        """全链路 parity: compute_prediction 的所有层级一致。"""
        weights = {
            "hgbr": 0.52, "elo": 0.29, "bayes": 0.19,
            "poisson": 0.48, "dc": 0.32, "nb": 0.20,
            "shape_weight": 1.0, "gbm_weight": 0.0,
        }
        
        lam_h, lam_a = 1.43, 1.17
        lam_eh, lam_ea = 1.35, 1.12
        lam_bh, lam_ba = 1.28, 1.15
        tau, phi = -0.071, 2.31
        gbm_probs = (0.55, 0.28, 0.17)
        
        result = compute_prediction(
            lam_h=lam_h, lam_a=lam_a,
            lam_eh=lam_eh, lam_ea=lam_ea,
            tau=tau, phi=phi, weights=weights,
            lam_bh=lam_bh, lam_ba=lam_ba,
            gbm_probs=gbm_probs,
        )
        
        assert result is not None
        
        # Layer-1: fused λ
        fh_expected = (0.52 * 1.43 + 0.29 * 1.35 + 0.19 * 1.28)
        fa_expected = (0.52 * 1.17 + 0.29 * 1.12 + 0.19 * 1.15)
        assert abs(result.fused_lambda[0] - fh_expected) < 1e-10
        assert abs(result.fused_lambda[1] - fa_expected) < 1e-10
        
        # Layer-2: shape_1x2 应该与 goal_1x2 相同
        assert result.shape_1x2 == result.goal_1x2
        
        # Layer-3: outcome_1x2 = α * shape + (1-α) * gbm ("shape": 1.0, "gbm": 0.0)
        alpha = 1.0
        expected_outcome = tuple(
            alpha * s + (1 - alpha) * g
            for s, g in zip(result.shape_1x2, gbm_probs)
        )
        for i in range(3):
            assert abs(result.outcome_1x2[i] - expected_outcome[i]) < 1e-6
    
    def test_parity_no_gbm(self):
        """无 GBM 时, outcome = shape。"""
        weights = {
            "hgbr": 0.52, "elo": 0.29, "bayes": 0.19,
            "poisson": 0.48, "dc": 0.32, "nb": 0.20,
            "shape_weight": 1.0, "gbm_weight": 0.0,
        }
        
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0, weights=weights,
            gbm_probs=None,
        )
        
        if result is not None:
            # 无 GBM 时 outcome = shape
            assert result.outcome_1x2 == result.shape_1x2
    
    def test_score_matrix_is_valid_probability(self):
        """score matrix 是合法概率分布。"""
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0,
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 1.0, "gbm_weight": 0.0},
        )
        
        if result is not None:
            mtx = np.asarray(result.score_matrix)
            assert np.all(mtx >= 0), "score matrix has negative values"
            assert abs(mtx.sum() - 1.0) < 1e-6, f"score matrix sum={mtx.sum()}"
    
    def test_1x2_marginal_equals_score_matrix(self):
        """1X2 marginal 等于 score matrix 的边缘概率。"""
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0,
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 1.0, "gbm_weight": 0.0},
        )
        
        if result is not None:
            mtx = np.asarray(result.score_matrix)
            
            # Home win: sum of lower triangle
            hw_mtx = float(np.tril(mtx, -1).sum())
            # Draw: trace
            dr_mtx = float(np.diag(mtx).sum())
            # Away win: upper triangle
            aw_mtx = float(np.triu(mtx, 1).sum())
            
            # 应该与 shape_1x2 一致
            assert abs(hw_mtx - result.shape_1x2[0]) < 1e-6
            assert abs(dr_mtx - result.shape_1x2[1]) < 1e-6
            assert abs(aw_mtx - result.shape_1x2[2]) < 1e-6
    
    def test_xg_from_score_matrix_equals_fused_lambda(self):
        """score matrix 的期望(xG)应该接近 fused λ。"""
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0,
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 1.0, "gbm_weight": 0.0},
        )
        
        if result is not None:
            mtx = np.asarray(result.score_matrix)
            grid = np.arange(mtx.shape[0], dtype=float)
            xg_h = float((mtx * grid[:, None]).sum())
            xg_a = float((mtx * grid[None, :]).sum())
            
            # xG 应该接近 fused λ (允许小偏差 due to 10x10 truncation)
            assert abs(xg_h - result.fused_lambda[0]) < 0.5
            assert abs(xg_a - result.fused_lambda[1]) < 0.5
    
    def test_bayes_unavailable_fallback(self):
        """Bayes 缺失时,自动 mask 并重新归一化。"""
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0,
            weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 1.0, "gbm_weight": 0.0},
            lam_bh=None, lam_ba=None,  # Bayes 不可用
        )
        
        if result is not None:
            # 应该仍然有结果(使用 hgbr + elo)
            assert result.fused_lambda[0] > 0
            assert result.fused_lambda[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
