"""成员概率构建:OOF 样本 → 各成员概率 + λ + shape_1x2(使用 learned weights)。

输出同时包含:
- λ 值(用于 Layer-1 Poisson NLL 优化)
- 1X2 概率(用于 Layer-2 LogLoss 优化)
- shape_1x2(使用 learned Layer-2 weights,而非等权)
"""
from __future__ import annotations


def build_member_samples(oof_samples, tau, phi, weights: dict | None = None):
 """从 OOF 样本构建成员概率样本。"""
 from app.models.ensemble import dc_probs, elo_goal_lambda, match_probs, nb_probs

 if weights:
 from app.models.ensemble.weights import to_layered
 lay = to_layered(weights)
 gl = lay["goal_lambda"]
 sd = lay.get("score_distribution", {})
 else:
 gl = {"hgbr": 1.0, "elo": 1.0, "bayes": 1.0}
 sd = {"poisson": 1.0, "dc": 1.0, "nb": 1.0}

 wsum = sum(gl.get(k, 0) for k in ["hgbr", "elo", "bayes"]) or 1.0
 w_hgbr = gl.get("hgbr", 0) / wsum
 w_elo = gl.get("elo", 0) / wsum
 w_bayes = gl.get("bayes", 0) / wsum

 # Layer-2 weights (用于 shape_1x2)
 wp = sd.get("poisson", 1.0)
 wd = sd.get("dc", 1.0)
 wn = sd.get("nb", 1.0)
 wsum_sd = wp + wd + wn or 1.0

 samples = []
 for s in oof_samples:
 lam_eh = elo_goal_lambda(s["att_diff"], True)
 lam_ea = elo_goal_lambda(s["att_diff"], False)
 bh = s.get("bayes_lam_h")
 ba = s.get("bayes_lam_a")

 # Fused λ (使用 learned Layer-1 weights)
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

 # Shape probabilities (基于 fused λ)
 pois_p = match_probs(fh, fa)
 dc_p = dc_probs(fh, fa, tau)
 nb_p = nb_probs(fh, fa, phi)
 
 
 shape_1x2 = [
 (wp * pois_p[0] + wd * dc_p[0] + wn * nb_p[0]) / wsum_sd,
 (wp * pois_p[1] + wd * dc_p[1] + wn * nb_p[1]) / wsum_sd,
 (wp * pois_p[2] + wd * dc_p[2] + wn * nb_p[2]) / wsum_sd,
 ]

 rec = {
 # λ 值(Layer-1)
 "hgbr_lam_h": s["hgbr_lam_h"],
 "hgbr_lam_a": s["hgbr_lam_a"],
 "elo_lam_h": lam_eh,
 "elo_lam_a": lam_ea,
 "bayes_lam_h": bh,
 "bayes_lam_a": ba,
 "fused_lam_h": fh,
 "fused_lam_a": fa,
 # 1X2 概率(Layer-2)
 "hgbr": list(match_probs(s["hgbr_lam_h"], s["hgbr_lam_a"])),
 "poisson": list(pois_p),
 "dc": list(dc_p),
 "nb": list(nb_p),
 "elo": list(match_probs(lam_eh, lam_ea)),
 # Shape ensemble (Layer-3, 使用 learned weights)
 "shape_1x2": shape_1x2,
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
