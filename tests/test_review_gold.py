"""审查第 30-34 条黄金/关键测试(2026-08 第五轮审查)。

- 三十: 时间泄漏测试(同日/未来) + 快照特征一致性 + GBM 不改 λ + 权重归一化
- 三十一: 预测确定性(同输入 → 同输出)
- 三十二: 未来数据不得进入当前预测
- 三十三: Ensemble 一致性(matrix 1X2 == Goal 1X2; xG == fused λ)
- 三十四: GBM 隔离(GBM 只改 1X2,不改 λ/xG/矩阵)

全部为纯函数级测试(不依赖真实数据库)。
"""

import numpy as np
import pandas as pd


# ── 三十:时间泄漏 ──────────────────────────────────────────────────────────
def test_same_day_no_leakage():
    """同日比赛互不影响:前一场大胜不改变同日其他场次的 ELO(审查 P1-6)。"""
    from app.models.elo_goal.rating import with_elo_features

    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 5,
                "away_goals": 0,
            },
            {
                "date": "2026-01-01",
                "home_team": "C",
                "away_team": "D",
                "home_goals": 0,
                "away_goals": 0,
            },
        ]
    )
    out = with_elo_features(df)
    # C 队同日特征必须是赛前初始值(同日同时开赛,不受 A 队大胜影响)
    assert abs(out.iloc[1]["home_elo"] - 1500.0) < 1e-9
    assert abs(out.iloc[1]["away_elo"] - 1500.0) < 1e-9


def test_future_match_does_not_change_features():
    """添加未来比赛不得改变既有比赛的特征(审查三十二)。"""
    from app.models.elo_goal.rating import with_elo_features

    past = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "date": "2026-01-03",
                "home_team": "C",
                "away_team": "D",
                "home_goals": 0,
                "away_goals": 1,
            },
        ]
    )
    base = with_elo_features(past)
    future = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "date": "2026-01-03",
                "home_team": "C",
                "away_team": "D",
                "home_goals": 0,
                "away_goals": 1,
            },
            {
                "date": "2026-01-10",
                "home_team": "A",
                "away_team": "C",
                "home_goals": 4,
                "away_goals": 0,
            },  # 未来赛果
        ]
    )
    fut = with_elo_features(future)
    for col in ("home_elo", "away_elo", "elo_diff", "attack_elo_diff"):
        assert np.allclose(
            base[col].to_numpy(), fut[col].iloc[: len(base)].to_numpy()
        ), f"{col} 被未来数据污染"
    # 未来行的特征基于过去状态(非 1500 初始)
    assert fut.iloc[2]["home_elo"] > 1500.0


# ── 三十:快照特征 = 预测输入特征 ────────────────────────────────────────────
def test_snapshot_features_equal_prediction_features():
    """快照主/客特征必须与预测输入最后两行(-2 主队/-1 客队)完全一致。"""
    from app.prediction.snapshot import extract_feature_rows

    rng = np.random.default_rng(0)
    feat = pd.DataFrame(rng.normal(size=(10, 4)), columns=["a", "b", "c", "d"])
    feat.iloc[-1] = [9.0, 8.0, 7.0, 6.0]  # 客队预测行
    feat.iloc[-2] = [1.0, 2.0, 3.0, 4.0]  # 主队预测行
    home, away = extract_feature_rows(feat, ["a", "b", "c", "d"])
    assert home["a"] == 1.0 and away["a"] == 9.0
    assert home["d"] == 4.0 and away["d"] == 6.0
    # 与特征矩阵原值(round 6 位)逐一相等 —— 不是"另一个特征向量"
    for col in ("a", "b", "c", "d"):
        assert home[col] == round(float(feat.iloc[-2][col]), 6)
        assert away[col] == round(float(feat.iloc[-1][col]), 6)


# ── 三十/三十四:GBM 不改变 λ;Goal 权重归一化 ───────────────────────────────
def _base_weights(gbm_w=0.0):
    return {"hgbr": 0.60, "dc": 0.20, "nb": 0.05, "elo": 0.15, "gbm": gbm_w}


def test_gbm_does_not_change_lambda():
    """GBM 权重 0 vs 0.3 → fused λ 严格不变(GBM 只进 1X2)。"""
    from app.prediction.goal_engine import compute_members

    a = compute_members(1.5, 1.2, 1.8, 1.0, 0.05, 30.0, _base_weights(0.0))
    b = compute_members(1.5, 1.2, 1.8, 1.0, 0.05, 30.0, _base_weights(0.3))
    assert a["fused_lams"] == b["fused_lams"]
    assert a["members"]["hgbr"] == b["members"]["hgbr"]
    assert a["score_out"]["expected_xg"] == b["score_out"]["expected_xg"]


def test_goal_weights_normalize():
    """fused λ = Goal 成员权重的归一化加权平均(不被 GBM 稀释)。"""
    from app.prediction.goal_engine import compute_members

    out = compute_members(1.5, 1.2, 1.8, 1.0, 0.05, 30.0, _base_weights(0.2))
    wh = 0.60 + 0.20 + 0.05
    wg = wh + 0.15
    exp_h = wh / wg * 1.5 + 0.15 / wg * 1.8
    exp_a = wh / wg * 1.2 + 0.15 / wg * 1.0
    assert abs(out["fused_lams"][0] - exp_h) < 1e-12
    assert abs(out["fused_lams"][1] - exp_a) < 1e-12
    # 权重和为 1 时应严格等于加权平均(验证 λ 不缩放)
    w1 = {"hgbr": 0.6, "dc": 0.2, "nb": 0.05, "elo": 0.15, "gbm": 0.0}
    out1 = compute_members(1.5, 1.2, 1.8, 1.0, 0.05, 30.0, w1)
    assert (
        abs(out1["fused_lams"][0] - (0.6 * 1.5 + 0.2 * 1.5 + 0.05 * 1.5 + 0.15 * 1.8))
        < 1e-12
    )


# ── 三十三:Ensemble 一致性 ─────────────────────────────────────────────────
# 注意:HGBR/DC/NB 三个 Goal 成员当前共享同一个泊松 λ(hgbr λ),
# 归一化后 fused λ 落在 [min λ, max λ];matrix 边缘 1X2 与成员概率须一致。
def test_ensemble_consistency_hgbr_only():
    """仅 HGBR 时:矩阵边缘 1X2 == match_probs;xG == λ(hgbr 1.0 权重)。"""
    from app.models.ensemble import _pois_matrix, match_probs

    lam_h, lam_a = 1.5, 1.2
    m = _pois_matrix(lam_h, lam_a)
    # 边缘:主胜 = 下三角去对角(主队进球 > 客队),平局 = 对角,客胜 = 上三角
    # (与 distributions.matrix_to_probs 同一口径:tril(-1)=home, triu(1)=away)
    _n = m.shape[0]
    home_w = m[np.tril_indices(_n, -1)].sum()
    draw = np.trace(m)
    away_w = m[np.triu_indices(_n, 1)].sum()
    p = match_probs(lam_h, lam_a)
    assert abs(home_w - p[0]) < 1e-6
    assert abs(draw - p[1]) < 1e-6
    assert abs(away_w - p[2]) < 1e-6
    # xG(矩阵期望)= λ(10×10 截断 + 归一化有 ~1e-4 精度损失,容差 1e-3)
    grid = np.arange(_n)
    xg_h = (m * grid[:, None]).sum()
    xg_a = (m * grid[None, :]).sum()
    assert abs(xg_h - lam_h) < 1e-3
    assert abs(xg_a - lam_a) < 1e-3


def test_ensemble_consistency_fused_matrix_vs_lams():
    """融合矩阵期望 xG 与融合 λ 一致(归一化后 Goal 层一致)。"""
    from app.prediction.goal_engine import compute_members

    out = compute_members(1.5, 1.2, 1.8, 1.0, 0.05, 30.0, _base_weights(0.0))
    xg = out["score_out"]["expected_xg"]
    fl = out["fused_lams"]
    # ELO 成员矩阵期望 = ELO λ;融合 xG = 权重归一化混合 → 与 fused λ 应接近
    # (矩阵融合用原始权重归一化,λ 融合用归一化权重 —— 同一凸组合)
    assert abs(xg[0] - fl[0]) < 0.1
    assert abs(xg[1] - fl[1]) < 0.1


# ── 三十一:预测确定性 ───────────────────────────────────────────────────────
def test_prediction_determinism():
    """同输入两次计算 → 输出位级一致(100% 可复现的基石)。"""
    from app.prediction.goal_engine import compute_members

    kwargs = {
        "lam_h": 1.5,
        "lam_a": 1.2,
        "lam_eh": 1.8,
        "lam_ea": 1.0,
        "tau": 0.05,
        "phi": 30.0,
        "weights": _base_weights(0.1),
    }
    a = compute_members(**kwargs)
    b = compute_members(**kwargs)
    assert a["fused_lams"] == b["fused_lams"]
    assert list(a["members"].keys()) == list(b["members"].keys())
    for k in a["members"]:
        assert all(abs(x - y) < 1e-15 for x, y in zip(a["members"][k], b["members"][k]))
    assert a["score_out"]["top_scores"] == b["score_out"]["top_scores"]
    assert a["score_out"]["expected_xg"] == b["score_out"]["expected_xg"]


# ── 三十:权重学习动态成员(GBM 不可用 → 从优化中移除,非 [0,0,0])────────────
def test_learn_weights_removes_missing_member():
    """样本缺 gbm → 权重学习中 gbm=0 且其余成员归一化。"""
    from app.models.ensemble import learn_weights

    samples = [
        {
            "hgbr": [0.5, 0.3, 0.2],
            "dc": [0.4, 0.3, 0.3],
            "elo": [0.6, 0.2, 0.2],
            "actual": 0,
        },
        {
            "hgbr": [0.2, 0.3, 0.5],
            "dc": [0.3, 0.3, 0.4],
            "elo": [0.2, 0.3, 0.5],
            "actual": 2,
        },
        {
            "hgbr": [0.3, 0.5, 0.2],
            "dc": [0.3, 0.4, 0.3],
            "elo": [0.3, 0.4, 0.3],
            "actual": 1,
        },
    ]
    w = learn_weights(samples)
    assert w["gbm"] == 0.0
    assert abs(sum(w[k] for k in ("hgbr", "dc", "nb", "elo")) - 1.0) < 1e-6
    assert w["nb"] == 0.0


# ── 三十二:特征版本不变量 ────────────────────────────────────────────────
def test_logical_version_stable_and_includes_switch():
    """logical_version 两次调用一致,且包含特征开关状态(h2h)。

    审查七 V7-3:开关变化必须改变版本 —— 否则旧模型(带 h2h 列)会被
    当成与新特征兼容,预测时列失配。
    """
    from app.features.registry import logical_version

    v1 = logical_version()
    v2 = logical_version()
    assert v1 == v2
    # 手动复现(含 features 段)必须与函数一致 —— 证明开关参与版本计算
    import hashlib as _hl
    import json as _json

    from app.core.config import load_yaml
    from app.features import attack_defense, form, h2h, stats_features, strength

    parts = [m.version() for m in (strength, attack_defense, form, h2h, stats_features)]
    fc = load_yaml("models.yaml").get("features") or {}
    parts.append(
        "features="
        + _hl.sha256(_json.dumps(fc, sort_keys=True).encode()).hexdigest()[:8]
    )
    manual = _hl.sha256("|".join(parts).encode()).hexdigest()[:12]
    assert v1 == manual, "logical_version 未纳入特征开关状态(实现级版本失效)"


# ── 审查九:V7.5 不变量 ──────────────────────────────────────────────────
def test_learn_weights_max_weight_cap():
    """单成员权重上限:GBM 过度自信时不得全押(≤ max_weight)。"""
    import numpy as np

    from app.models.ensemble.weights import learn_weights

    rng = np.random.default_rng(1)
    samples = []
    for _ in range(150):
        base = np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.04, 3)
        base /= base.sum()
        a = int(rng.choice(3, p=base))
        g = np.array([0.98, 0.01, 0.01])
        g = np.roll(g, a)
        samples.append({"hgbr": list(base), "gbm": list(g), "actual": a})
    w = learn_weights(samples, shrinkage=0.15, max_weight=0.7)
    assert w["gbm"] <= 0.7 + 1e-6


def test_pipeline_hash_stable():
    """pipeline_hash 两次调用一致(源码哈希)。"""
    from app.prediction.versions import pipeline_hash

    assert pipeline_hash() == pipeline_hash()


def test_data_content_hash_changes_on_edit():
    """data_hash 内容化:修改任意历史记录内容 → 哈希变化。"""
    import pandas as pd

    from app.prediction.snapshot import data_content_hash

    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "date": "2026-01-02",
                "home_team": "C",
                "away_team": "D",
                "home_goals": 0,
                "away_goals": 0,
            },
        ]
    )
    h1 = data_content_hash(df)
    df2 = df.copy()
    df2.loc[1, "away_goals"] = 3
    assert data_content_hash(df2) != h1


def test_ipf_matrix_matches_target():
    """IPF:调整后矩阵边缘 == 目标 1X2。"""
    import numpy as np

    from app.models.ensemble import _pois_matrix
    from app.prediction.regime import ipf_to_target

    m = _pois_matrix(1.5, 1.2)
    t = (0.42, 0.32, 0.26)
    m2 = ipf_to_target(m, t)
    hw = m2[np.tril_indices_from(m2, -1)].sum()
    dr = np.trace(m2)
    aw = m2[np.triu_indices_from(m2, 1)].sum()
    assert abs(hw - 0.42) < 1e-6 and abs(dr - 0.32) < 1e-6 and abs(aw - 0.26) < 1e-6


def test_learn_weights_rejects_raw_oof_samples():
    """回归防护:原始 OOF 样本(lam 键)不得被当作成员概率样本。

    learn_weights 的 present 判定用 'name in s'(子串),若传入
    {"hgbr_lam_h": ...} 会命中 "hgbr" 但 s["hgbr"]=None → SLSQP 错乱
    (曾产出 gbm=1.0 假象)。成员概率样本必须经 member_builder 构建。
    """
    from app.services.training.ensemble.member_builder import build_member_samples

    raw = [
        {
            "hgbr_lam_h": 1.5,
            "hgbr_lam_a": 1.2,
            "att_diff": 10.0,
            "bayes_lam_h": 1.4,
            "bayes_lam_a": 1.3,
            "home_goals": 2,
            "away_goals": 1,
            "actual": 0,
            "gbm": [0.5, 0.3, 0.2],
        }
    ]
    samples = build_member_samples(raw, tau=0.0, phi=1e9)
    assert len(samples) == 1
    for k in ("hgbr", "dc", "nb", "elo", "bayes", "gbm"):
        assert k in samples[0], f"成员 {k} 缺失"
        assert len(samples[0][k]) == 3
