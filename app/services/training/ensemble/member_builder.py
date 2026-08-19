"""成员概率构建(审查九 P1-9 拆分):OOF 样本 → 各成员三分类概率。"""

from __future__ import annotations


def build_member_samples(oof_samples, tau, phi):
    """从 OOF 样本构建成员概率样本(含 bayes 成员;gbm 不可用则不移入)。"""
    from app.models.ensemble import dc_probs, elo_goal_lambda, match_probs, nb_probs

    samples = []
    for s in oof_samples:
        lam_eh = elo_goal_lambda(s["att_diff"], True)
        lam_ea = elo_goal_lambda(s["att_diff"], False)
        rec = {
            "hgbr": list(match_probs(s["hgbr_lam_h"], s["hgbr_lam_a"])),
            "dc": list(dc_probs(s["hgbr_lam_h"], s["hgbr_lam_a"], tau)),
            "nb": list(nb_probs(s["hgbr_lam_h"], s["hgbr_lam_a"], phi)),
            "elo": list(match_probs(lam_eh, lam_ea)),
            "actual": s["actual"],
        }
        if s.get("bayes_lam_h") is not None:
            rec["bayes"] = list(match_probs(s["bayes_lam_h"], s["bayes_lam_a"]))
        if s.get("gbm") is not None:
            rec["gbm"] = s["gbm"]  # 不可用时不移入样本
        samples.append(rec)
    return samples
