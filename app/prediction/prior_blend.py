"""近期环境先验混合(审查七 V7-1:2026 高平局赛季修复)。

诊断:英超 2026 实际平局率 32%(历史 ~21%),模型按历史分布预测 →
平局系统性低估 ~10pp、强队主胜过度自信(0.7+ 桶实际 55.6%)。

修复:P_final = α·P_model + (1-α)·近期实际频率(截止该场最近 window 场)。
离线验证(W=100, α=0.6):2026 ll 1.1254→1.0952、ECE 0.127→0.044;
2025 不退化(ll 0.9949→0.9916, ECE 0.042)。2026-08 上线。

配置:configs/models.yaml → prediction.prior_blend(enabled/window/alpha/min_history)。
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cfg() -> dict:
    try:
        from app.core.config import load_yaml
        return (load_yaml("models.yaml").get("prediction") or {}).get("prior_blend") or {}
    except Exception:
        return {}


def recent_freqs(league_id, cutoff_dt, window: int = 100, min_history: int = 50):
    """截止 cutoff_dt(严格更早)最近 window 场完赛的实际 1X2 频率。"""
    from app.api.db import Match
    rows = (Match.query.filter(
                Match.league_id == league_id,
                Match.match_status == "finished",
                Match.match_date < cutoff_dt)
            .order_by(Match.match_date.desc()).limit(window).all())
    if len(rows) < min_history:
        return None
    n = len(rows)
    h = sum(1 for m in rows if (m.home_goals or 0) > (m.away_goals or 0))
    d = sum(1 for m in rows if (m.home_goals or 0) == (m.away_goals or 0))
    a = n - h - d
    if n == 0:
        return None
    return np.array([h / n, d / n, a / n], dtype=float)


def blend(league_id, cutoff_dt, probs) -> tuple[np.ndarray | None, dict | None]:
    """P_final = α·P_model + (1-α)·近期频率;返回 (混合概率, 审计信息)。"""
    cfg = _cfg()
    if not cfg.get("enabled", True):
        return None, None
    window = int(cfg.get("window", 100))
    alpha = float(cfg.get("alpha", 0.6))
    min_history = int(cfg.get("min_history", 50))
    freqs = recent_freqs(league_id, cutoff_dt, window, min_history)
    if freqs is None:
        return None, None
    p = np.asarray(probs, dtype=float)
    out = alpha * p + (1.0 - alpha) * freqs
    s = out.sum()
    if s <= 0:
        return None, None
    out = out / s
    return out, {"method": "recent_freq_prior", "alpha": alpha, "window": window,
                 "freqs": [round(float(x), 4) for x in freqs]}


def blend_matrix(league_id, cutoff_dt, probs, matrix):
    """审查九 P0-3/P1-12:矩阵级 Regime 调整。

    1) regime 检测 → 动态 α(稳定 0.85 / 剧烈 shift 0.55);
    2) 目标 1X2 = α·P_model + (1-α)·近期频率;
    3) IPF 调整 score matrix 使边缘 = 目标 1X2;
    4) 所有输出(1X2/O-U/BTTS/Top5/xG)必须从返回矩阵重新导出。

    返回 (调整后矩阵, 审计信息);无历史/禁用时返回 (None, None)。
    """
    cfg = _cfg()
    if not cfg.get("enabled", True):
        return None, None
    window = int(cfg.get("window", 100))
    min_history = int(cfg.get("min_history", 50))
    freqs = recent_freqs(league_id, cutoff_dt, window, min_history)
    if freqs is None:
        return None, None
    # 审查 P1-12:regime 检测 + 动态 α
    try:
        from app.prediction.regime import detect as _detect
        from app.prediction.regime import dynamic_alpha as _dyn_alpha
        _reg = _detect(league_id, cutoff_dt, window=window)
        alpha = _dyn_alpha(_reg.get("shift_score", 0.0),
                           regime=_reg.get("regime", "NORMAL"))
    except Exception:
        _reg, alpha = {}, float(cfg.get("alpha", 0.6))
    p = np.asarray(probs, dtype=float)
    target = alpha * p + (1.0 - alpha) * freqs
    s = target.sum()
    if s <= 0:
        return None, None
    target = target / s
    try:
        from app.prediction.regime import ipf_to_target
        m2 = ipf_to_target(np.asarray(matrix, dtype=float), tuple(target))
    except Exception:
        return None, None
    info = {"method": "ipf_regime_adjust", "alpha": round(alpha, 4),
            "window": window, "freqs": [round(float(x), 4) for x in freqs],
            "regime": _reg.get("regime"), "shift_score": _reg.get("shift_score"),
            "target_1x2": [round(float(x), 4) for x in target]}
    return m2, info
