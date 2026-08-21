"""Score Distribution Metrics Test (opt-6 + opt-7)。

验证 xG vs λ invariant 和 score distribution 指标。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.models.distributions import MAX_GOALS
from app.models.ensemble.matrix import compute_tail_mass, extract_dc_low_score_probs, extract_nb_tail_probs
from app.prediction.layered_pipeline import compute_prediction


class TestScoreMetrics:
 """Score distribution 指标测试。"""
 
 def test_xg_vs_lambda_invariant(self):
 """xG (score matrix 期望) 应该接近 fused λ。"""
 result = compute_prediction(
 lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
 tau=0.05, phi=50.0,
 weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
 )
 
 if result is not None:
 mtx = np.asarray(result.score_matrix)
 grid = np.arange(mtx.shape[0], dtype=float)
 xg_h = float((mtx * grid[:, None]).sum())
 xg_a = float((mtx * grid[None, :]).sum())
 
 # xG 应该接近 fused λ
 assert abs(xg_h - result.fused_lambda[0]) < 0.5
 assert abs(xg_a - result.fused_lambda[1]) < 0.5
 
 def test_tail_mass_reasonable(self):
 """10x10 score matrix 的 tail mass 应该很小。"""
 result = compute_prediction(
 lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
 tau=0.05, phi=50.0,
 weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
 )
 
 if result is not None:
 tail = compute_tail_mass(result.score_matrix)
 # tail mass 应该 < 5%
 assert tail["tail_mass"] < 0.05
 
 def test_dc_calibration_output(self):
 """DC calibration 输出包含必要字段。"""
 result = compute_prediction(
 lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
 tau=0.05, phi=50.0,
 weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
 )
 
 if result is not None:
 dc_cal = extract_dc_low_score_probs(result.score_matrix)
 assert "p_00" in dc_cal
 assert "p_10" in dc_cal
 assert "p_01" in dc_cal
 assert "p_11" in dc_cal
 assert "p_low_score" in dc_cal
 
 # 概率应该在 [0, 1] 范围内
 for k in ["p_00", "p_10", "p_01", "p_11", "p_low_score"]:
 assert 0 <= dc_cal[k] <= 1
 
 def test_nb_tail_calibration_output(self):
 """NB tail calibration 输出包含必要字段。"""
 result = compute_prediction(
 lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
 tau=0.05, phi=50.0,
 weights={"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "poisson": 0.5, "dc": 0.3, "nb": 0.2, "shape_weight": 0.7, "gbm_weight": 0.3},
 )
 
 if result is not None:
 nb_cal = extract_nb_tail_probs(result.score_matrix)
 assert "p_total_ge4" in nb_cal
 assert "p_total_ge5" in nb_cal
 
 # P(>=4) >= P(>=5)
 assert nb_cal["p_total_ge4"] >= nb_cal["p_total_ge5"]
 # 概率应该在 [0, 1] 范围内
 assert 0 <= nb_cal["p_total_ge4"] <= 1
 assert 0 <= nb_cal["p_total_ge5"] <= 1


if __name__ == "__main__":
 pytest.main([__file__, "-v"])
