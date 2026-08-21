"""成员概率构建:OOF 样本 → 各成员概率 + λ(用于分层学习)。

输出同时包含:
- λ 值(用于 Layer-1 Poisson NLL 优化)
- 1X2 概率(用于 Layer-2 LogLoss 优化)
"""
from __future__ import annotations


def build_member_samples(oof_samples, tau, phi, weights: dict | None = None):
    """从 OOF 样本构建成员概率样本。
    
    输出字段:
    - hgbr_lam_h/a, elo_lam_h/a, bayes_lam_h/a: λ 值(Layer-1)
    - hgbr, poisson, dc, nb, elo, bayes: 1X2 概率(Layer-2)
    """
    from app.models.ensemble import dc_probs, elo_goal_lambda, match_probs, nb_probs

    if weights:
        from app.models.ensemble.weights import to_layered
        lay = to_layered(weights)
        gl = lay["goal_lambda"]
    else:
        gl = {"hgbr": 1.0, "elo": 1.0, "bayes": 1.0}

    wsum = sum(gl.get(k, 0) for k in ["hgbr", "elo", "bayes"]) or 1.0
    w_hgbr = gl.get("hgbr", 0) / wsum
    w_elo = gl.get("elo", 0) / wsum
    w_bayes = gl.get("bayes", 0) / wsum

    samples = []
    for s in oof_samples:
        lam_eh = elo_goal_lambda(s["att_diff"], True)
        lam_ea = elo_goal_lambda(s["att_diff"], False)
        bh = s.get("bayes_lam_h")
        ba = s.get("bayes_lam_a")

        # Fused λ
        if bh is not None and ba is not None:
            fh = w_hgbr * s["hgbr_lam_h"] + w_elo * lam_eh + w_bayes * bh
            fa = w_hgbr * s["hgbr_lam_a"] + w_elo * lam_ea + w_bayes * ba
        else:
            w_sum_gl = w_hgbr + w_elo
            if w_sum_gl > 0:
                fh = (w_hgbr * s["hgbr_lam_h"] + w_elo * lam_eh) / w_sum_gl
                fa = (w_hgbr * s["hgbr_lam_a"] + w_elo * lam_ea) / w_sum_gl
            else:
                fh, fa = s["hgbr_lam_h"], s["hgbr_lam_a"]

        rec = {
            # λ 值(Layer-1)
            "hgbr_lam_h": s["hgbr_lam_h"],
            "hgbr_lam_a": s["hgbr_lam_a"],
            "elo_lam_h": lam_eh,
            "elo_lam_a": lam_ea,
            "bayes_lam_h": bh,
            "bayes_lam_a": ba,
            # 1X2 概率(Layer-2)
            "hgbr": list(match_probs(s["hgbr_lam_h"], s["hgbr_lam_a"])),
            "poisson": list(match_probs(fh, fa)),
            "dc": list(dc_probs(fh, fa, tau)),
            "nb": list(nb_probs(fh, fa, phi)),
            "elo": list(match_probs(lam_eh, lam_ea)),
            # 实际值
            "actual": s["actual"],
            "home_goals": s.get("home_goals", 0),
            "away_goals": s.get("away_goals", 0),
        }
        if bh is not None:
            rec["bayes"] = list(match_probs(bh, ba))
        if s.get("gbm") is not None:
            rec["gbm"] = s["gbm"]
        samples.append(rec)
    return samples
