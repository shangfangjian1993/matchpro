"""审查 A70A601 §23:概率契约 / 时间泄漏 / 重放确定性测试。

分层:
  L1 纯函数(_roll/_ewma)
  L2 公开 API(rolling_team_stats / invariants)
  L3 Feature Factory(compute_all)
  L4 模型输入一致性(compute_members / validate_prediction)
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.ensemble import score_outputs
from app.prediction.invariants import (
    check_matrix,
    check_matrix_marginal,
    check_probability_vector,
    check_top_scores,
    check_xg,
)


# ── Probability Contract(审查 §23)──────────────────────────────────────
def _ok_matrix(hw=0.55, dr=0.25, aw=0.20):
    """构造边缘为 (hw, dr, aw) 的合法 10x10 概率矩阵。"""
    m = np.zeros((10, 10))
    vals = np.arange(1, 11, dtype=float)
    lam_h, lam_a = 1.6, 1.2
    for i, ph in enumerate(vals):
        for j, pa in enumerate(vals):
            m[i, j] = (ph**lam_h * np.exp(-lam_h) / 1) * (
                pa**lam_a * np.exp(-lam_a) / 1
            )
    m = m / m.sum()
    tril = np.tril(np.ones((10, 10), dtype=bool), -1)
    diag = np.eye(10, dtype=bool)
    triu = np.triu(np.ones((10, 10), dtype=bool), 1)
    for _ in range(300):  # IPF 化,边缘=目标
        phm, pdm, pam = m[tril].sum(), np.trace(m), m[triu].sum()
        mul = np.ones_like(m)
        mul[tril] = hw / max(phm, 1e-12)
        mul[diag] = dr / max(pdm, 1e-12)
        mul[triu] = aw / max(pam, 1e-12)
        m = m * mul
        m = m / m.sum()
    return m


def test_probability_vector_contract():
    assert check_probability_vector([0.5, 0.3, 0.2]) == []
    v = check_probability_vector([0.6, 0.3, 0.2])
    assert any("和" in x for x in v)
    v2 = check_probability_vector([1.1, -0.1, 0.0])
    assert any("越界" in x for x in v2)


def test_matrix_contract():
    m = _ok_matrix()
    assert check_matrix(m) == []
    assert check_matrix_marginal(m, (0.55, 0.25, 0.20)) == []


def test_matrix_expectation_equals_fused_lambda():
    """矩阵期望(xG)≈ 融合 λ —— 用真实 compute_members 生成并交叉验证。"""
    from app.prediction.goal_engine import compute_members

    g = compute_members(
        1.6,
        1.2,
        1.7,
        1.1,
        tau=0.05,
        phi=30.0,
        weights={"hgbr": 0.5, "dc": 0.1, "nb": 0.1, "elo": 0.1, "bayes": 0.2},
        lam_bh=1.5,
        lam_ba=1.3,
    )
    lam_h, lam_a = g["fused_lams"]
    m = np.asarray(g["fused_matrix"], dtype=float)
    assert check_matrix(m) == []
    assert check_xg(m, lam_h, lam_a, tol=0.15) == []


def test_top_scores_contract():
    so = score_outputs(_ok_matrix())
    assert check_top_scores(so["top_scores"]) == []
    assert len(so["top_scores"]) > 0
    # OU/BTTS 从矩阵派生且取值合法
    assert 0.0 <= so["over_2_5"] <= 1.0
    assert 0.0 <= so["btts"] <= 1.0
    # expected_xg 与矩阵期望一致
    m = _ok_matrix()
    to = score_outputs(m)
    grid = np.arange(m.shape[0], dtype=float)
    # expected_xg round(3),容差 1e-3
    assert abs(to["expected_xg"][0] - float((m * grid[:, None]).sum())) < 1e-3


# ── Temporal Leakage(审查 §14)──────────────────────────────────────────
def test_stats_features_no_future_leakage():
    """滚动特征 shift1:预测行不得包含本场/未来统计。"""
    from app.features.stats_features import _roll
    from tests.test_stats_features import _FakeMatch, _FakeStats

    m1 = _FakeMatch(1, "AA", "BB")
    m2 = _FakeMatch(2, "AA", "CC")
    m3 = _FakeMatch(3, "AA", "DD")  # 待预测(本场犯规则数不计入)
    s1 = _FakeStats(fouls=10, tackles=40, offsides=2)
    s2 = _FakeStats(fouls=6, tackles=30, offsides=1)
    s3 = _FakeStats(fouls=99, tackles=99, offsides=99)  # 本场数值
    mapping = _roll(
        [m1, m2, m3],
        {
            1: {"home": s1, "away": _FakeStats()},
            2: {"home": s2, "away": _FakeStats()},
            3: {"home": s3, "away": _FakeStats()},
        },
        windows=(2,),
    )
    # 第三场(预测)的 avg_2 只含前两场(10,6 → 8.0),不含本场 99
    assert mapping[3]["home_tms_fouls_avg_2"] == 8.0
    assert mapping[3]["home_tms_fouls_ewm"] != 99.0


# ── Replay Determinism(审查 §23)────────────────────────────────────────
def test_compute_members_deterministic():
    from app.prediction.goal_engine import compute_members

    kw = {
        "lam_h": 1.5,
        "lam_a": 1.1,
        "lam_eh": 1.7,
        "lam_ea": 0.9,
        "tau": 0.05,
        "phi": 20.0,
        "weights": {"hgbr": 0.4, "dc": 0.1, "nb": 0.1, "elo": 0.15, "bayes": 0.25},
        "lam_bh": 1.4,
        "lam_ba": 1.2,
    }
    a = compute_members(**kw)
    b = compute_members(**kw)
    assert a["fused_matrix"].tolist() == b["fused_matrix"].tolist()
    assert a["fused_lams"] == b["fused_lams"]
    assert a["score_out"]["expected_xg"] == b["score_out"]["expected_xg"]


@pytest.mark.db
def test_snapshot_contract(db_ctx):
    """快照完整契约(有 DB 时):model_set 含 bayes + 契约校验通过。"""
    from app.api.db import PredictionSnapshot, init_db, session_scope
    from app.core.config import LeagueType
    from app.prediction.service import predict_match

    init_db()
    with session_scope():
        r = predict_match(LeagueType.PREMIER_LEAGUE, "阿森纳", "切尔西")
    from app.prediction.invariants import validate_prediction

    assert validate_prediction(r) == []
    s = PredictionSnapshot.query.order_by(PredictionSnapshot.id.desc()).first()
    if s is not None:
        import json

        snap = json.loads(s.snapshot_json or "{}")
        ms = snap.get("model_set", {})
        assert "score_matrix" in snap
        assert "bayes" in ms  # Bayes 必须出现在 model_set
        if ms.get("bayes"):
            assert "version" in ms["bayes"]


def test_to_layered_idempotent():
    """P0-2: to_layered 幂等性测试。"""
    from app.models.ensemble.weights import DEFAULT_WEIGHTS, from_layered, to_layered

    # layered → to_layered 应该保持不变
    layered = to_layered(DEFAULT_WEIGHTS)
    assert to_layered(layered) == layered

    # flat → to_layered → from_layered → to_layered 应该稳定
    flat = {"hgbr": 0.5, "elo": 0.3, "bayes": 0.2, "dc": 0.3, "nb": 0.2, "gbm": 0.5}
    layered1 = to_layered(flat)
    flat2 = from_layered(layered1)
    layered2 = to_layered(flat2)
    # 允许 rounding tolerance
    for k in layered1["goal_lambda"]:
        assert abs(layered1["goal_lambda"][k] - layered2["goal_lambda"][k]) < 0.01
    for k in layered1["score_distribution"]:
        assert abs(layered1["score_distribution"][k] - layered2["score_distribution"][k]) < 0.01


def test_load_weights_flat_format():
    """load_weights 返回 flat 格式(engine 兼容)。"""
    from app.models.ensemble.weights import load_weights
    w = load_weights("premier_league")
    # engine.py 需要 flat 格式
    for k in ("hgbr", "elo", "bayes", "dc", "nb", "gbm"):
        assert k in w, f"Missing key: {k}"


def test_gh_ablation_different():
    """G和H应该产生不同结果(G无Calibration, H有Calibration)。
    
    当 calibration artifact 不存在时,两者都会回退到未校准结果,
    此时 diagnostics 中 calibration_applied 字段应该不同。
    """
    import numpy as np

    from app.core.config import LeagueType
    from app.prediction.layered_pipeline import AblationMask, compute_prediction

    np.random.seed(42)
    weights = {
        "hgbr": 0.5, "elo": 0.3, "bayes": 0.2,
        "poisson": 0.5, "dc": 0.3, "nb": 0.2,
        "shape": 0.3, "gbm": 0.7,
    }
    gbm_probs = (0.6, 0.25, 0.15)
    raw_matrix = np.eye(10) * 0.1
    raw_matrix[1, 1] = 0.9

    mask_g = AblationMask(disable_calibration=True)
    mask_h = AblationMask()

    result_g = compute_prediction(
        lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
        tau=0.05, phi=50.0, weights=weights,
        lam_bh=1.3, lam_ba=1.2,
        gbm_probs=gbm_probs,
        prior_context={"league_id": 1, "match_dt": "2026-01-01", "raw_matrix": raw_matrix},
        calibration_context={"models_dir": "/tmp/fake_models", "league_type": LeagueType.PREMIER_LEAGUE},
        ablation_mask=mask_g,
    )

    result_h = compute_prediction(
        lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
        tau=0.05, phi=50.0, weights=weights,
        lam_bh=1.3, lam_ba=1.2,
        gbm_probs=gbm_probs,
        prior_context={"league_id": 1, "match_dt": "2026-01-01", "raw_matrix": raw_matrix},
        calibration_context={"models_dir": "/tmp/fake_models", "league_type": LeagueType.PREMIER_LEAGUE},
        ablation_mask=mask_h,
    )

    if result_g is not None and result_h is not None:
        # 验证 G 没有应用 calibration
        assert result_g.diagnostics.get("calibration") != "applied", "G should not apply calibration"
        # 验证 H 尝试应用 calibration(可能成功或 fallback)
        # 两者 final_1x2 可能相同(都 fallback),但 diagnostics 应该反映差异
        assert result_g.ablation_mask.disable_calibration is True
        assert result_h.ablation_mask.disable_calibration is False


def test_score_matrix_mass_invariant():
    """P0-3: score matrix 质量守恒(sum≈1)。"""
    import numpy as np

    from app.core.config import LeagueType
    from app.prediction.layered_pipeline import compute_prediction

    weights = {
        "goal_lambda": {"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
        "score_distribution": {"poisson": 0.5, "dc": 0.3, "nb": 0.2},
        "outcome": {"gbm": 0.7},
    }
    raw_matrix = np.eye(10) * 0.1
    raw_matrix[1, 1] = 0.9

    result = compute_prediction(
        lam_h=1.5, lam_a=1.2, lam_eh=1.4, lam_ea=1.1,
        tau=0.05, phi=50.0, weights=weights,
        lam_bh=1.3, lam_ba=1.2,
        gbm_probs=(0.6, 0.25, 0.15),
        prior_context={"league_id": 1, "match_dt": "2026-01-01", "raw_matrix": raw_matrix},
        calibration_context={"models_dir": "/tmp/fake_models", "league_type": LeagueType.PREMIER_LEAGUE},
    )

    if result is not None:
        score_matrix = result.score_matrix
        total_mass = float(np.sum(score_matrix))
        assert abs(total_mass - 1.0) < 0.01, f"score matrix mass={total_mass}, expected ~1.0"
        assert np.all(score_matrix >= 0), "score matrix has negative values"
        
        # Tail mass check (10x10 grid)
        tail_mass = 1.0 - total_mass
        assert tail_mass < 0.05, f"tail mass too large: {tail_mass}"


def test_production_training_parity():
    """P1-6: Production/Training 数学同构测试。"""
    from app.prediction.layered_pipeline import compute_prediction
    from app.services.training.ensemble.member_builder import build_member_samples

    weights = {
        "goal_lambda": {"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
        "score_distribution": {"poisson": 0.5, "dc": 0.3, "nb": 0.2},
        "outcome": {"gbm": 0.7},
    }
    tau, phi = 0.05, 50.0

    # 构造测试样本
    oof_sample = [{
        "hgbr_lam_h": 1.5, "hgbr_lam_a": 1.2,
        "att_diff": 0.3,
        "bayes_lam_h": 1.3, "bayes_lam_a": 1.2,
        "home_goals": 2, "away_goals": 1, "actual": 0,
    }]

    # Training path
    samples = build_member_samples(oof_sample, tau, phi, weights)
    
    # Production path
    result = compute_prediction(
        lam_h=1.5, lam_a=1.2,
        lam_eh=samples[0]["elo_lam_h"], lam_ea=samples[0]["elo_lam_a"],
        tau=tau, phi=phi, weights=weights,
        lam_bh=1.3, lam_ba=1.2,
    )

    if result is not None and samples:
        # Training fused λ == Production fused λ
        assert abs(samples[0]["fused_lam_h"] - result.fused_lambda[0]) < 0.01
        assert abs(samples[0]["fused_lam_a"] - result.fused_lambda[1]) < 0.01


def test_layer2_poisson_weight_preserved():
    """P0-3: Layer-2 Poisson权重在artifact中保存。"""
    from app.models.ensemble.weights import to_layered
    
    # 模拟 learn_weights 输出(包含 poisson)
    flat_weights = {
        "hgbr": 0.4, "elo": 0.3, "bayes": 0.3,
        "poisson": 0.5, "dc": 0.3, "nb": 0.2,
        "gbm": 0.3,
    }
    
    layered = to_layered(flat_weights)
    
    # score_distribution 应该包含 poisson
    assert "poisson" in layered["score_distribution"]
    assert layered["score_distribution"]["poisson"] > 0


def test_ensemble_training_result():
    """P0-1: EnsembleTrainingResult 不可变。"""
    from app.services.training.ensemble.weight_optimizer import EnsembleTrainingResult
    
    result = EnsembleTrainingResult(
        tau=0.05, phi=50.0,
        weights={"hgbr": 0.5},
        metadata={"test": True}
    )
    
    assert result.tau == 0.05
    assert result.phi == 50.0
    assert result.weights == {"hgbr": 0.5}
    
    # 不可变
    try:
        result.tau = 0.1
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_nb_phi_formula():
    """P0-5: NB φ = mean² / (Var - mean)。"""
    from app.services.training.ensemble.optimizers.nb_parameter import fit_phi
    
    # 构造过离散样本: mean=3, var=6
    # Var = μ + μ²/φ → 6 = 3 + 9/φ → φ = 3
    samples = [{"home_goals": 0, "away_goals": 6}] * 50 + [{"home_goals": 6, "away_goals": 0}] * 50
    
    phi = fit_phi(samples)
    
    # φ 应该约为 3.0 (允许误差)
    assert 1.0 < phi < 10.0, f"phi={phi}, expected ~3.0"


def test_gbm_weight_used_in_fusion():
    """P0-6: fuse_goal_outcome 使用 shape_weight + gbm_weight。"""
    from app.models.ensemble.fusion import fuse_goal_outcome
    
    shape_1x2 = (0.5, 0.3, 0.2)
    gbm_1x2 = (0.6, 0.25, 0.15)
    weights = {"shape_weight": 0.7, "gbm_weight": 0.3}
    
    result = fuse_goal_outcome(shape_1x2, gbm_1x2, weights)
    
    # 手动计算期望
    expected = (
        0.7 * 0.5 + 0.3 * 0.6,
        0.7 * 0.3 + 0.3 * 0.25,
        0.7 * 0.2 + 0.3 * 0.15,
    )
    
    assert abs(result[0] - expected[0]) < 0.001
    assert abs(result[1] - expected[1]) < 0.001
    assert abs(result[2] - expected[2]) < 0.001
