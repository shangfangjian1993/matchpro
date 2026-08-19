"""Goal Engine(审查 §12/§14 拆分 + A70A601 §4/§24⑤):成员层次结构明确。

成员三层结构(勿把"5 成员"当作 5 个独立 λ 模型):
  第一层 独立 λ:　HGBR(特征回归)、ELO(评级→λ)、Bayes(层次贝叶斯收缩)
  第二层 比分分布(共享第一层 λ,负责分布形态):
         Poisson(hgbr λ)、Dixon-Coles(hgbr λ + τ 低分修正)、
         NegativeBinomial(hgbr λ + φ 过离散)
  第三层 结果分类器:Outcome GBM(1X2,独立于 λ 融合)
DC/NB 不是独立预测模型,而是分布层变体;fused λ 只融合第一层
(审查 P0-2 已含 Bayes),分布层只贡献 score matrix 形态;权重学习
(Baseline-aware)由 OOF 对数释然数据驱动,避免伪多样性高估。

输出:成员概率、比分矩阵、Score Outputs(Top5/Over-Under/BTTS/xG)、
融合 λ(第一层 λ 加权,校验用)。
"""

from __future__ import annotations

from app.models.ensemble import (
    dc_probs,
    fuse_score_matrix,
    match_probs,
    nb_probs,
    score_outputs,
)


def compute_members(
    lam_h: float,
    lam_a: float,
    lam_eh: float,
    lam_ea: float,
    tau: float,
    phi: float,
    weights: dict,
    lam_bh: float | None = None,
    lam_ba: float | None = None,
) -> dict:
    """四成员概率 + 比分矩阵 + Score Outputs + 融合 λ。

    lam_eh/lam_ea: ELO-Goal 成员 λ(已含伤停乘数)。
    """
    from app.models.ensemble import _dc_matrix, _nb_matrix, _pois_matrix

    members = {
        "hgbr": match_probs(lam_h, lam_a),
        "dc": dc_probs(lam_h, lam_a, tau),
        "nb": nb_probs(lam_h, lam_a, phi),
        "elo": match_probs(lam_eh, lam_ea),
        # 审查九 P1-10:层次贝叶斯成员(联赛先验×攻防收缩;无样本=先验)
        "bayes": match_probs(lam_bh or lam_h, lam_ba or lam_a),
    }
    matrices = {
        "hgbr": _pois_matrix(lam_h, lam_a),
        "dc": _dc_matrix(lam_h, lam_a, tau),
        "nb": _nb_matrix(lam_h, lam_a, phi),
        "elo": _pois_matrix(lam_eh, lam_ea),
        "bayes": _pois_matrix(lam_bh or lam_h, lam_ba or lam_a),
    }
    fused_matrix = fuse_score_matrix(matrices, weights)
    score_out = score_outputs(fused_matrix)
    # 审查 P1-7:快照需冻结 score matrix(Replay 不得用 λ 重算,否则算法
    # 改动会改变历史快照结果);fused_matrix 直接给 Snapshot 落库
    wh = weights.get("hgbr", 1.0) + weights.get("dc", 0.0) + weights.get("nb", 0.0)
    we = weights.get("elo", 0.0)
    wb = weights.get("bayes", 0.0)
    wg = wh + we + wb
    # 审查 A70A601 P0-2:Bayes 参与 score matrix,则其 λ 必须进入 fused λ,
    # 否则"预测进球(fused λ)"与"expected_xg(矩阵期望)"分属不同概率体系。
    # bayes 兜底 lam_bh or lam_h 与 matrix 成员一致(_pois_matrix(lam_bh or lam_h)),
    # 保证同源 → 各成员 λ 全部计入后 fused λ ≈ 矩阵期望。
    if wg > 0:
        fused_lams = (
            (wh * lam_h + we * lam_eh + wb * (lam_bh or lam_h)) / wg,
            (wh * lam_a + we * lam_ea + wb * (lam_ba or lam_a)) / wg,
        )
    else:
        fused_lams = (lam_h, lam_a)
    return {
        "members": members,
        "score_out": score_out,
        "fused_lams": fused_lams,
        "fused_matrix": fused_matrix,
    }
