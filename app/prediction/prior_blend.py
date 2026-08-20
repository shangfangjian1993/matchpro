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

        return (load_yaml("models.yaml").get("prediction") or {}).get(
            "prior_blend"
        ) or {}
    except Exception:
        return {}


# 100 场 ≈2.6 赛季,已跨越教练/阵容/升降级 —— 不能承载全部近期信息;
# 短窗(最近)体现即时漂移,长窗提供稳定基准。
PRIOR_WINDOWS = (20, 50, 100, 250)


def recent_freqs(
    league_id,
    cutoff_dt,
    window: int = 100,
    min_history: int = 50,
):
    """截止 cutoff_dt(严格更早)最近 window 场完赛的实际 1X2 频率。"""
    freqs = recent_freqs_multi(
        league_id, cutoff_dt, windows=(window,), min_samples=min_history
    )
    if not freqs:
        return None
    return freqs[window]


def recent_freqs_multi(
    league_id,
    cutoff_dt,
    windows: tuple = PRIOR_WINDOWS,
    min_samples: int = 20,
) -> dict:
    """单次查询,返回 {window: freq_vector|None}。

    一次取 max(windows) 场,按各窗口截取计算频率 —— 避免对每窗口单独
    查询(多尺度场景 4 次查询 → 1 次)。样本 < min_samples 的窗口置 None。
    """
    from app.api.db import Match

    max_w = max(windows)
    rows = (
        Match.query.filter(
            Match.league_id == league_id,
            Match.match_status == "finished",
            Match.match_date < cutoff_dt,
        )
        .order_by(Match.match_date.desc())
        .limit(max_w)
        .all()
    )
    if len(rows) < min_samples:
        return {}
    out: dict = {}
    for w in windows:
        chunk = rows[:w]
        if len(chunk) < min_samples:
            continue
        n = len(chunk)
        h = sum(1 for m in chunk if (m.home_goals or 0) > (m.away_goals or 0))
        dd = sum(1 for m in chunk if (m.home_goals or 0) == (m.away_goals or 0))
        out[w] = np.array([h / n, dd / n, (n - h - dd) / n], dtype=float)
    return out


def _multi_scale_weights(shift: float) -> np.ndarray:
    """多尺度近窗权重:nw_i 随 drift 向短窗倾斜(审查 A70A601 P1-6)。

    稳定(shift≈0):近窗权重几乎全部落在长窗 250(稳定基准);
    剧烈漂移(shift→1):权重显著转向 20/50(即时环境)。
    返回归一化后的 [w20, w50, w100, w250]。
    """
    s = float(np.clip(shift, 0.0, 1.0))
    w20 = 0.02 + 0.18 * s
    w50 = 0.03 + 0.12 * s
    w100 = 0.04 + 0.06 * s
    w250 = max(0.0, 1.0 - (w20 + w50 + w100))
    nw = np.array([w20, w50, w100, w250])
    tot = nw.sum()
    if tot <= 0:
        nw = np.array([0.0, 0.0, 0.25, 0.75])
        tot = nw.sum()
    return nw / tot


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
    return out, {
        "method": "recent_freq_prior",
        "alpha": alpha,
        "window": window,
        "freqs": [round(float(x), 4) for x in freqs],
    }


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
    windows = tuple(int(x) for x in (cfg.get("windows") or PRIOR_WINDOWS))
    min_history = int(cfg.get("min_history", 20))
    freqs_multi = recent_freqs_multi(
        league_id, cutoff_dt, windows=windows, min_samples=min_history
    )
    if not freqs_multi:
        return None, None
    try:
        from app.prediction.regime import detect as _detect
        from app.prediction.regime import dynamic_alpha as _dyn_alpha

        _reg = _detect(league_id, cutoff_dt, window=max(windows))
        alpha = _dyn_alpha(
            _reg.get("shift_score", 0.0), regime=_reg.get("regime", "NORMAL")
        )
    except Exception:
        _reg, alpha = {}, float(cfg.get("alpha", 0.6))
    p = np.asarray(probs, dtype=float)
    # 近窗资源 = 1-α(模型权重);多尺度权重 nw 决定资源在 20/50/100/250 间分配
    nw = _multi_scale_weights(_reg.get("shift_score", 0.0))
    _used_windows = [w for w in windows if w in freqs_multi]
    if not _used_windows:
        return None, None
    _nw = np.array(
        [nw[list(windows).index(w)] if w in windows else 0.0 for w in _used_windows]
    )
    _nw = _nw / _nw.sum()
    _freq_part = np.zeros(3)
    for _i, w_ in enumerate(_used_windows):
        _freq_part += _nw[_i] * freqs_multi[w_]
    target = alpha * p + (1.0 - alpha) * _freq_part
    s = target.sum()
    if s <= 0:
        return None, None
    target = target / s
    try:
        from app.prediction.regime import ipf_to_target

        m2 = ipf_to_target(np.asarray(matrix, dtype=float), tuple(target))
    except Exception:
        return None, None
    info = {
        "method": "ipf_regime_adjust_multi",
        "alpha": round(alpha, 4),
        "windows": list(_used_windows),
        "window_weights": [round(float(x), 4) for x in _nw],
        "freqs": {
            w: [round(float(x), 4) for x in freqs_multi[w]] for w in _used_windows
        },
        "regime": _reg.get("regime"),
        "shift_score": _reg.get("shift_score"),
        "target_1x2": [round(float(x), 4) for x in target],
    }
    return m2, info
