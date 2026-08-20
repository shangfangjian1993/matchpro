"""成员概率构建:OOF 样本 → 各成员三分类概率(基于 fused λ,与生产一致)。"""
from __future__ import annotations


def build_member_samples(oof_samples, tau, phi, weights: dict | None = None):
    """从 OOF 样本构建成员概率样本(基于 fused λ,与生产 LayeredPipeline 一致)。
    
    Args:
        oof_samples: OOF 样本列表(含 hgbr_lam_h/a, att_diff, bayes_lam_h/a)
        tau: Dixon-Coles 参数
        phi: Negative Binomial 参数
        weights: 生产权重(dict),用于计算 fused lambda。为 None 时使用默认等权。
    """
    from app.models.ensemble import dc_probs, elo_goal_lambda, match_probs, nb_probs

    # 计算 fused lambda 的权重
    if weights:
        from app.models.ensemble.weights import to_layered
        lay = to_layered(weights)
        gl = lay["goal_lambda"]
    else:
        # 默认等权
        gl = {"hgbr": 1.0, "elo": 1.0, "bayes": 1.0}

    wsum = sum(gl.get(k, 0) for k in ["hgbr", "elo", "bayes"]) or 1.0
    w_hgbr = gl.get("hgbr", 0) / wsum
    w_elo = gl.get("elo", 0) / wsum
    w_bayes = gl.get("bayes", 0) / wsum

    samples = []
    for s in oof_samples:
        lam_eh = elo_goal_lambda(s["att_diff"], True)
        lam_ea = elo_goal_lambda(s["att_diff"], False)
        
        # 计算 fused lambda(与生产一致)
        bh = s.get("bayes_lam_h")
        ba = s.get("bayes_lam_a")
        
        if bh is not None and ba is not None:
            fh = w_hgbr * s["hgbr_lam_h"] + w_elo * lam_eh + w_bayes * bh
            fa = w_hgbr * s["hgbr_lam_a"] + w_elo * lam_ea + w_bayes * ba
        else:
            # Bayes 缺失 → 重新归一化
            w_sum_gl = w_hgbr + w_elo
            if w_sum_gl > 0:
                fh = (w_hgbr * s["hgbr_lam_h"] + w_elo * lam_eh) / w_sum_gl
                fa = (w_hgbr * s["hgbr_lam_a"] + w_elo * lam_ea) / w_sum_gl
            else:
                fh, fa = s["hgbr_lam_h"], s["hgbr_lam_a"]
        
        # DC/NB 基于 fused λ(与生产一致!)
        rec = {
            "hgbr": list(match_probs(s["hgbr_lam_h"], s["hgbr_lam_a"])),
            "poisson": list(match_probs(fh, fa)),
            "dc": list(dc_probs(fh, fa, tau)),
            "nb": list(nb_probs(fh, fa, phi)),
            "elo": list(match_probs(lam_eh, lam_ea)),
            "actual": s["actual"],
        }
        if bh is not None:
            rec["bayes"] = list(match_probs(bh, ba))
        if s.get("gbm") is not None:
            rec["gbm"] = s["gbm"]
        samples.append(rec)
    return samples
