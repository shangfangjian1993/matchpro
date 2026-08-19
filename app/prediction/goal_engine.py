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
    fuse_probs,
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
    """三层真实计算(审查 e752f5f P1-17:存储分层 == 计算分层)。

    Layer-1 Goal λ Ensemble:  goal_lambda(hgbr/elo/bayes) → fused λ
    Layer-2 Score Distribution: Shape Ensemble —— Base Poisson(DC/NB 均
        基于 **fused λ**,非 hgbr λ):matrix = w_pois·Poi(λ̄) + w_dc·DC(λ̄) + w_nb·NB(λ̄)
        1X2 同理。Poisson 与 DC/NB 是层级关系,不是平行独立模型。
    Layer-3 Outcome GBM: 独立 1X2(engine 内 fuse_goal_outcome,不进矩阵)。

    members 仍保留各成员 1X2(组件审计/Ablation 用)。
    """
    from app.models.ensemble import _dc_matrix, _nb_matrix, _pois_matrix
    from app.models.ensemble.weights import to_layered

    lay = to_layered(weights)
    gl = lay["goal_lambda"]
    sd = lay["score_distribution"]

    # Layer-1:独立 λ 融合(hgbr/elo/bayes;bayes 兜底=hgbr 与旧矩阵口径一致)
    bh_ = lam_bh if lam_bh is not None else lam_h
    ba_ = lam_ba if lam_ba is not None else lam_a
    _gs = gl.get("hgbr", 0.0) + gl.get("elo", 0.0) + gl.get("bayes", 0.0)
    if _gs <= 0:
        fh, fa = lam_h, lam_a
    else:
        fh = (
            gl.get("hgbr", 0.0) * lam_h
            + gl.get("elo", 0.0) * lam_eh
            + gl.get("bayes", 0.0) * bh_
        ) / _gs
        fa = (
            gl.get("hgbr", 0.0) * lam_a
            + gl.get("elo", 0.0) * lam_ea
            + gl.get("bayes", 0.0) * ba_
        ) / _gs

    # Layer-2:Shape Ensemble(全部基于 fused λ)—— Poison 基 + DC + NB
    pois_p, dc_p, nb_p = (
        match_probs(fh, fa),
        dc_probs(fh, fa, tau),
        nb_probs(fh, fa, phi),
    )
    shape_1x2 = fuse_probs({"poisson": pois_p, "dc": dc_p, "nb": nb_p}, sd)
    pm = _pois_matrix(fh, fa)
    dm = _dc_matrix(fh, fa, tau)
    nm = _nb_matrix(fh, fa, phi)
    fused_matrix = fuse_score_matrix({"poisson": pm, "dc": dm, "nb": nm}, sd)
    score_out = score_outputs(fused_matrix)
    # 各成员 1X2(审计/组件化 Ablation 用;E 级 = shape_1x2)
    members = {
        "hgbr": match_probs(lam_h, lam_a),
        "dc": dc_probs(lam_h, lam_a, tau),
        "nb": nb_probs(lam_h, lam_a, phi),
        "elo": match_probs(lam_eh, lam_ea),
        "bayes": match_probs(bh_, ba_),
    }
    return {
        "members": members,
        "score_out": score_out,
        "fused_lams": (fh, fa),
        "fused_matrix": fused_matrix,
        "shape_1x2": shape_1x2,
        "layer_weights": lay,
    }
