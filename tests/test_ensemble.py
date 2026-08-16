"""Ensemble 最小测试:成员一致性 + 派生输出。"""


def test_members_consistent():
    """τ=0 / φ=∞ 时 DC/NB 退化为泊松,与 HGBR 一致。"""
    from app.models.ensemble import dc_probs, match_probs, nb_probs
    pa = match_probs(1.5, 1.2)
    pb = dc_probs(1.5, 1.2, 0.0)
    pc = nb_probs(1.5, 1.2, 1e9)
    assert all(abs(x - y) < 1e-9 for x, y in zip(pa, pb))
    assert all(abs(x - y) < 1e-9 for x, y in zip(pa, pc))


def test_fuse_normalized():
    """概率融合归一。"""
    from app.models.ensemble import fuse_probs
    p = fuse_probs({"hgbr": (0.5, 0.3, 0.2), "elo": (0.4, 0.3, 0.3)},
                   {"hgbr": 0.7, "elo": 0.3})
    assert abs(sum(p) - 1.0) < 1e-9


def test_score_outputs():
    """比分矩阵派生:Top5/Over-Under/BTTS/xG。"""
    from app.models.ensemble import _pois_matrix, score_outputs
    out = score_outputs(_pois_matrix(1.5, 1.2))
    assert len(out["top_scores"]) == 5
    assert abs(out["over_2_5"] + out["under_2_5"] - 1.0) < 1e-9
    assert len(out["expected_xg"]) == 2
