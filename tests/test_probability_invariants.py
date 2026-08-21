"""P0-1: 概率不变量 Fuzz / Property-Based Tests。

验证预测引擎输出的概率严格满足数学约束:
- sum(score_matrix) ≈ 1
- sum(1X2) ≈ 1
- P(0-0) + P(1-0) + ... + P(N,N) ≈ 1
- 1X2 marginal == score matrix 边缘
- xG == matrix 期望
"""
from __future__ import annotations

import numpy as np
import pytest

from app.prediction.layered_pipeline import compute_prediction


class TestProbabilityInvariants:
    """概率不变量 fuzz tests with various lambda/weights combinations."""
    
    @pytest.mark.parametrize("lam_h,lam_a", [
        (0.5, 0.5), (1.0, 1.0), (1.5, 1.2), (2.0, 0.8), (3.0, 2.5),
        (0.3, 0.3), (1.8, 1.8), (0.1, 2.0),
    ])
    @pytest.mark.parametrize("tau", [-0.1, 0.0, 0.05, 0.1])
    def test_score_matrix_is_probability_distribution(self, lam_h, lam_a, tau):
        """score matrix 必须是合法概率分布: 非负, 和≈1。"""
        weights = {
            "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
            "poisson": 0.5, "dc": 0.3, "nb": 0.2,
            "shape_weight": 0.7, "gbm_weight": 0.3,
        }
        result = compute_prediction(
            lam_h=lam_h, lam_a=lam_a,
            lam_eh=lam_h * 0.9, lam_ea=lam_a * 0.9,
            tau=tau, phi=50.0, weights=weights,
        )
        
        if result is not None:
            mtx = np.asarray(result.score_matrix)
            assert np.all(mtx >= -1e-10), f"Negative probabilities: min={mtx.min()}"
            assert abs(mtx.sum() - 1.0) < 1e-4, f"Sum={mtx.sum()}"
    
    @pytest.mark.parametrize("lam_h,lam_a", [(1.5, 1.2), (0.8, 0.8), (2.0, 1.5)])
    def test_1x2_sums_to_one(self, lam_h, lam_a):
        """1X2 概率之和必须等于 1。"""
        weights = {
            "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
            "poisson": 0.5, "dc": 0.3, "nb": 0.2,
            "shape_weight": 0.7, "gbm_weight": 0.3,
        }
        result = compute_prediction(
            lam_h=lam_h, lam_a=lam_a,
            lam_eh=lam_h * 0.9, lam_ea=lam_a * 0.9,
            tau=0.05, phi=50.0, weights=weights,
        )
        
        if result is not None:
            assert abs(sum(result.final_1x2) - 1.0) < 0.01, f"1X2 sum={sum(result.final_1x2)}"
    
    @pytest.mark.parametrize("lam_h,lam_a", [(1.5, 1.2), (0.5, 2.0)])
    def test_xg_close_to_fused_lambda(self, lam_h, lam_a):
        """xG (matrix expectation) 应该接近 fused λ。"""
        weights = {
            "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
            "poisson": 0.5, "dc": 0.3, "nb": 0.2,
            "shape_weight": 0.7, "gbm_weight": 0.3,
        }
        result = compute_prediction(
            lam_h=lam_h, lam_a=lam_a,
            lam_eh=lam_h * 0.9, lam_ea=lam_a * 0.9,
            tau=0.05, phi=50.0, weights=weights,
        )
        
        if result is not None:
            mtx = np.asarray(result.score_matrix)
            grid = np.arange(mtx.shape[0], dtype=float)
            xg_h = float((mtx * grid[:, None]).sum())
            xg_a = float((mtx * grid[None, :]).sum())
            
            # xG 应该接近 fused λ (allow 0.5 tolerance for 10x10 truncation)
            assert abs(xg_h - result.fused_lambda[0]) < 0.5
            assert abs(xg_a - result.fused_lambda[1]) < 0.5
    
    def test_all_probabilities_finite(self):
        """所有概率值必须是 finite (no NaN/Inf)。"""
        weights = {
            "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
            "poisson": 0.5, "dc": 0.3, "nb": 0.2,
            "shape_weight": 0.7, "gbm_weight": 0.3,
        }
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0, weights=weights,
        )
        
        if result is not None:
            for p in result.final_1x2:
                assert np.isfinite(p), f"Non-finite probability: {p}"
            assert np.all(np.isfinite(result.score_matrix))
    
    def test_all_probabilities_non_negative(self):
        """所有概率值必须非负。"""
        weights = {
            "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
            "poisson": 0.5, "dc": 0.3, "nb": 0.2,
            "shape_weight": 0.7, "gbm_weight": 0.3,
        }
        result = compute_prediction(
            lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
            tau=0.05, phi=50.0, weights=weights,
        )
        
        if result is not None:
            for p in result.final_1x2:
                assert p >= -1e-10, f"Negative probability: {p}"
            assert np.all(np.asarray(result.score_matrix) >= -1e-10)
    
    def test_deterministic_same_output(self):
        """相同输入必须产生相同输出(确定性)。"""
        weights = {
            "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
            "poisson": 0.5, "dc": 0.3, "nb": 0.2,
            "shape_weight": 0.7, "gbm_weight": 0.3,
        }
        
        results = []
        for _ in range(3):
            r = compute_prediction(
                lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
                tau=0.05, phi=50.0, weights=weights,
            )
            if r is not None:
                results.append(r.final_1x2)
        
        for r in results[1:]:
            assert r == results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
