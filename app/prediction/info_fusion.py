from app.models.distributions import pois_pmf as poisson_pmf

"""资讯融合层:LLM 信号提取协议 + 规则映射 + 三层融合

链路: 父模型 λ → q/d 状态修正 → 资讯信号调整(LLM 提取) → 最终预测
信号协议(LLM 输出,只含原文可查证事实):
  {home_attack_impact, home_defense_impact, home_rotation_risk,
   away_attack_impact, away_defense_impact, away_rotation_risk,
   home_motivation, away_motivation, confidence, sources}
规则: 进攻/防守/动机/轮换 → λ 乘数,封顶 ±20% 防过度
"""

SIGNAL_KEYS = [
    "home_attack_impact",
    "home_defense_impact",
    "home_rotation_risk",
    "away_attack_impact",
    "away_defense_impact",
    "away_rotation_risk",
    "home_motivation",
    "away_motivation",
]
MAX_ADJ = 0.20  # 单信号封顶


def normalize_signals(raw: dict) -> dict:
    """校验/夹取信号字段;缺失取 0;confidence 缺失取 0.5"""
    sig = {k: float(raw.get(k, 0.0)) for k in SIGNAL_KEYS}
    for k, v in sig.items():
        sig[k] = max(-1.0, min(1.0, v))
    sig["confidence"] = float(raw.get("confidence", 0.5))
    sig["sources"] = raw.get("sources", [])
    return sig


def signals_to_adjust(sig: dict) -> tuple[float, float]:
    """信号 → (主队 λ 乘数, 客队 λ 乘数);只含原文可查证事实的保守映射"""
    ha = max(-MAX_ADJ, min(MAX_ADJ, sig["home_attack_impact"]))
    aa = max(-MAX_ADJ, min(MAX_ADJ, sig["away_attack_impact"]))
    hd = max(-MAX_ADJ, min(MAX_ADJ, sig["home_defense_impact"]))
    ad = max(-MAX_ADJ, min(MAX_ADJ, sig["away_defense_impact"]))
    hr = max(0.0, min(1.0, sig["home_rotation_risk"]))
    ar = max(0.0, min(1.0, sig["away_rotation_risk"]))
    hm = max(-MAX_ADJ, min(MAX_ADJ, sig["home_motivation"]))
    am = max(-MAX_ADJ, min(MAX_ADJ, sig["away_motivation"]))
    # 置信度缩放:低置信信号减半
    conf_scale = 0.5 + 0.5 * sig["confidence"]
    h_mult = 1.0 + conf_scale * (ha - ad + hm - 0.5 * hr)
    a_mult = 1.0 + conf_scale * (aa - hd + am - 0.5 * ar)
    # 封顶 ±25%
    h_mult = max(0.75, min(1.25, h_mult))
    a_mult = max(0.75, min(1.25, a_mult))
    return h_mult, a_mult


def score_probs(lam_h: float, lam_a: float, maxg: int = 11) -> list[list[float]]:
    g = [
        [poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a) for j in range(maxg + 1)]
        for i in range(maxg + 1)
    ]
    s = sum(sum(r) for r in g)
    return [[v / s for v in r] for r in g]


def summarize(P, maxg=11):
    hw = sum(P[i][j] for i in range(maxg + 1) for j in range(maxg + 1) if i > j)
    dr = sum(P[i][i] for i in range(maxg + 1))
    flat = sorted(
        [(P[i][j], i, j) for i in range(maxg + 1) for j in range(maxg + 1)],
        reverse=True,
    )
    top3 = ", ".join(f"{i}-{j}({100 * p:.1f}%)" for p, i, j in flat[:3])
    return {"hw": hw, "dr": dr, "aw": 1 - hw - dr, "top3": top3}


def fuse(numeric: dict, signals: dict | None, league_name: str = "") -> dict:
    """三层融合:numeric 含 lam_h/lam_a(已含 q/d 修正),signals 为 LLM 资讯信号"""
    if not signals:
        lam_h, lam_a = numeric["lam_h"], numeric["lam_a"]
        info = {"adjusted": False, "reason": "无资讯信号,使用数值层预测"}
    else:
        sig = normalize_signals(signals)
        h_mult, a_mult = signals_to_adjust(sig)
        lam_h, lam_a = numeric["lam_h"] * h_mult, numeric["lam_a"] * a_mult
        info = {
            "adjusted": True,
            "h_mult": h_mult,
            "a_mult": a_mult,
            "confidence": sig["confidence"],
            "sources": sig["sources"],
        }
    P = score_probs(lam_h, lam_a)
    s = summarize(P)
    return {
        "league": league_name,
        "lam_h": round(lam_h, 3),
        "lam_a": round(lam_a, 3),
        "home_win": round(s["hw"], 4),
        "draw": round(s["dr"], 4),
        "away_win": round(s["aw"], 4),
        "top_scores": [
            {"score": f"{i}-{j}", "prob": round(p, 4)}
            for p, i, j in sorted(
                [(P[i][j], i, j) for i in range(11) for j in range(11)], reverse=True
            )[:3]
        ],
        "info": info,
    }
