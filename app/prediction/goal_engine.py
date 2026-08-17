"""Goal Engine(审查 §12/§14 拆分):HGBR/DC/NB/ELO → λ、成员概率、比分矩阵。

输出:成员概率(4 成员)、比分矩阵、Score Outputs(Top5/Over-Under/BTTS/xG)、
融合 λ(与概率一致的期望进球)。
"""
from __future__ import annotations

from app.models.ensemble import (
    dc_probs,
    fuse_score_matrix,
    match_probs,
    nb_probs,
    score_outputs,
)


def compute_members(lam_h: float, lam_a: float, lam_eh: float, lam_ea: float,
                    tau: float, phi: float, weights: dict) -> dict:
    """四成员概率 + 比分矩阵 + Score Outputs + 融合 λ。

    lam_eh/lam_ea: ELO-Goal 成员 λ(已含伤停乘数)。
    """
    from app.models.ensemble import _dc_matrix, _nb_matrix, _pois_matrix

    members = {
        "hgbr": match_probs(lam_h, lam_a),
        "dc": dc_probs(lam_h, lam_a, tau),
        "nb": nb_probs(lam_h, lam_a, phi),
        "elo": match_probs(lam_eh, lam_ea),
    }
    matrices = {
        "hgbr": _pois_matrix(lam_h, lam_a),
        "dc": _dc_matrix(lam_h, lam_a, tau),
        "nb": _nb_matrix(lam_h, lam_a, phi),
        "elo": _pois_matrix(lam_eh, lam_ea),
    }
    fused_matrix = fuse_score_matrix(matrices, weights)
    score_out = score_outputs(fused_matrix)
    # 审查 P1-7:快照需冻结 score matrix(Replay 不得用 λ 重算,否则算法
    # 改动会改变历史快照结果);fused_matrix 直接给 Snapshot 落库
    wh = weights.get("hgbr", 1.0) + weights.get("dc", 0.0) + weights.get("nb", 0.0)
    we = weights.get("elo", 0.0)
    wg = wh + we
    # 审查 P1-8/9:Goal 权重归一化 —— GBM 只进 Outcome(1X2),不得缩放 λ;
    # 归一化后 fused λ 与 score matrix(内部 out/sum)严格一致。
    if wg > 0:
        fused_lams = (wh / wg * lam_h + we / wg * lam_eh,
                      wh / wg * lam_a + we / wg * lam_ea)
    else:
        fused_lams = (lam_h, lam_a)
    return {"members": members, "score_out": score_out, "fused_lams": fused_lams,
            "fused_matrix": fused_matrix}
