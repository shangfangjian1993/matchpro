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
        if ms.get("bayes"):
            assert "version" in ms["bayes"]
