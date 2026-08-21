"""权重配置与学习(Layered格式)。

Layer-1: Goal λ Ensemble (HGBR/ELO/Bayes)
Layer-2: Score Distribution (Poisson/DC/NB,基于 fused λ)
Layer-3: Outcome GBM
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

# 默认分层权重(生产使用)
DEFAULT_WEIGHTS = {
    "goal_lambda": {"hgbr": 0.5, "elo": 0.3, "bayes": 0.2},
    "score_distribution": {"poisson": 0.5, "dc": 0.3, "nb": 0.2},
    "outcome": {"gbm": 1.0},
}

LAYER_MAP = {
    "goal_lambda": ("hgbr", "elo", "bayes"),
    "score_distribution": ("poisson", "dc", "nb"),
    "outcome": ("gbm",),
}


def to_layered(data: dict) -> dict:
    """转换为分层格式(幂等)。

    支持:
    - flat 格式(旧): {"hgbr": 0.5, "elo": 0.3, ...}
    - layered 格式(新): {"goal_lambda": {...}, ...}
    """
    if _is_layered(data):
        # 已经是 layered → 归一化后返回(幂等)
        return _normalize_layered(data)
    # flat → layered
    goal = {k: float(data.get(k, 0.0)) for k in LAYER_MAP["goal_lambda"]}
    gsum = sum(goal.values()) or 1.0
    sd = {
        "poisson": gsum,
        "dc": float(data.get("dc", 0.0)),
        "nb": float(data.get("nb", 0.0)),
    }
    ssum = sum(sd.values()) or 1.0
    return {
        "goal_lambda": {k: round(v / gsum, 4) for k, v in goal.items()},
        "score_distribution": {k: round(v / ssum, 4) for k, v in sd.items()},
        "outcome": {"gbm": round(float(data.get("gbm", 0.0)), 4)},
    }


def _normalize_layered(layered: dict) -> dict:
    """归一化 layered 格式(确保权重和为 1)。"""
    goal = layered.get("goal_lambda", {})
    gsum = sum(float(goal.get(k, 0.0)) for k in LAYER_MAP["goal_lambda"]) or 1.0
    sd = layered.get("score_distribution", {})
    ssum = sum(float(sd.get(k, 0.0)) for k in LAYER_MAP["score_distribution"]) or 1.0
    outcome = layered.get("outcome", {})
    gb = float(outcome.get("gbm", 0.0))
    return {
        "goal_lambda": {k: round(float(goal.get(k, 0.0)) / gsum, 4) for k in LAYER_MAP["goal_lambda"]},
        "score_distribution": {k: round(float(sd.get(k, 0.0)) / ssum, 4) for k in LAYER_MAP["score_distribution"]},
        "outcome": {"gbm": round(gb, 4)},
    }


def from_layered(layered: dict) -> dict:
    """分层 → flat(兼容旧接口)。"""
    goal = layered.get("goal_lambda", {})
    sd = layered.get("score_distribution", {})
    gb = layered.get("outcome", {}).get("gbm", 0.0)
    poisson = float(sd.get("poisson", 0.0))
    gsum = sum(float(goal.get(k, 0.0)) for k in LAYER_MAP["goal_lambda"]) or 1.0
    w = {
        "hgbr": poisson * (float(goal.get("hgbr", 0.0)) / gsum),
        "elo": poisson * (float(goal.get("elo", 0.0)) / gsum),
        "bayes": poisson * (float(goal.get("bayes", 0.0)) / gsum),
        "dc": float(sd.get("dc", 0.0)),
        "nb": float(sd.get("nb", 0.0)),
        "gbm": gb,
    }
    return w


def _is_layered(data) -> bool:
    return isinstance(data, dict) and "goal_lambda" in data


_WEIGHTS_PATH = None


def set_weights_path(path: str | None):
    global _WEIGHTS_PATH
    _WEIGHTS_PATH = path


class OptimizationError(Exception):
    """SLSQP 优化失败。"""


def learn_weights(
    samples: list[dict],
    tau: float = 0.0,
    phi: float = 1e9,
    shrinkage: float = 0.15,
    prior: dict | None = None,
    max_weight: float = 0.7,
) -> dict:
    """分层权重学习:L-1(λ fusion) + L-2(shape) 分别优化。

    Layer-1 使用 Poisson Goal NLL(直接优化 λ 融合)
    Layer-2 使用 1X2 LogLoss(优化 score distribution)
    Layer-3 (GBM) 使用 outcome fusion(单独学习)
    """
    _candidates_gl = ["hgbr", "elo", "bayes"]
    _candidates_sd = ["poisson", "dc", "nb"]

    _present_gl = [n for n in _candidates_gl if any(n in s for s in samples)]
    _present_sd = [n for n in _candidates_sd if any(n in s for s in samples)]

    n = max(1, len(samples))

    # Layer-1: Goal λ 权重(使用 Poisson NLL)
    w_gl = _optimize_layer_weights_poisson(samples, _present_gl, shrinkage, max_weight, n, prior)

    # Layer-2: Score Distribution 权重(使用 1X2 LogLoss)
    w_sd = _optimize_layer_weights_1x2(samples, _present_sd, shrinkage, max_weight, n, prior)

    out = {
        "hgbr": w_gl.get("hgbr", 0.0),
        "elo": w_gl.get("elo", 0.0),
        "bayes": w_gl.get("bayes", 0.0),
        "dc": w_sd.get("dc", 0.0),
        "nb": w_sd.get("nb", 0.0),
        "gbm": 0.0,
        "log_loss": w_sd.get("log_loss", 0.0),
        "n": n,
        "shrinkage": shrinkage,
    }
    return out


def _optimize_layer_weights_poisson(samples, candidates, shrinkage, max_weight, n, prior):
    """Layer-1: 使用 Poisson Goal NLL 优化 λ 权重。"""
    all_names = ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]
    if not candidates:
        return {name: 1.0 if name == "hgbr" else 0.0 for name in all_names}

    if prior is None:
        _prior = _compute_baseline_prior_poisson(samples, candidates, n)
    else:
        _prior = np.array([prior.get(name, 0.0) for name in candidates], dtype=float)
        _prior = _prior / _prior.sum() if _prior.sum() > 0 else np.ones(len(candidates)) / len(candidates)

    if len(candidates) < 2:
        return {name: (1.0 if name == candidates[0] else 0.0) for name in all_names}

    def _nll(w):
        ll = 0.0
        for s in samples:
            # 计算 fused λ
            lam_h = sum(w[i] * s[f"{name}_lam_h"] for i, name in enumerate(candidates) if f"{name}_lam_h" in s)
            lam_a = sum(w[i] * s[f"{name}_lam_a"] for i, name in enumerate(candidates) if f"{name}_lam_a" in s)
            # Poisson NLL
            from math import lgamma, log
            gh = int(s.get("home_goals", 0))
            ga = int(s.get("away_goals", 0))
            ll -= (gh * log(max(lam_h, 1e-12)) - lam_h - lgamma(gh + 1))
            ll -= (ga * log(max(lam_a, 1e-12)) - lam_a - lgamma(ga + 1))
        ll = ll / n
        if shrinkage > 0 and len(candidates) > 1:
            ll += shrinkage * float(np.mean((w - _prior) ** 2))
        return ll

    return _run_slsqp(_nll, _prior, candidates, max_weight, all_names, n, "Layer-1")


def _optimize_layer_weights_1x2(samples, candidates, shrinkage, max_weight, n, prior):
    """Layer-2: 使用 1X2 LogLoss 优化 shape 权重。"""
    all_names = ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]
    if not candidates:
        return {name: 1.0 if name == "poisson" else 0.0 for name in all_names}

    if prior is None:
        _prior = _compute_baseline_prior_1x2(samples, candidates, n)
    else:
        _prior = np.array([prior.get(name, 0.0) for name in candidates], dtype=float)
        _prior = _prior / _prior.sum() if _prior.sum() > 0 else np.ones(len(candidates)) / len(candidates)

    if len(candidates) < 2:
        return {name: (1.0 if name == candidates[0] else 0.0) for name in all_names}

    def _nll(w):
        ll = 0.0
        for s in samples:
            p = np.zeros(3)
            for i, name in enumerate(candidates):
                if name not in s:
                    continue
                p += w[i] * np.asarray(s[name])
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            ll -= math.log(p[s["actual"]])
        ll = ll / n
        if shrinkage > 0 and len(candidates) > 1:
            ll += shrinkage * float(np.mean((w - _prior) ** 2))
        return ll

    return _run_slsqp(_nll, _prior, candidates, max_weight, all_names, n, "Layer-2")


def _run_slsqp(objective, prior, candidates, max_weight, all_names, n, layer_name):
    """运行 SLSQP 优化,检查 success。"""
    from scipy.optimize import minimize
    w0 = np.asarray(prior, dtype=float).copy()
    w0 = np.clip(w0, 0.0, max_weight)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(len(candidates), 1.0 / len(candidates))

    res = minimize(
        objective, w0, method="SLSQP",
        bounds=[(0.0, max_weight)] * len(candidates),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 200, "ftol": 1e-9},
    )

    if not res.success:
        import logging
        logging.getLogger(__name__).warning(f"{layer_name} SLSQP failed: {res.message}")

    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()

    out = {name: 0.0 for name in all_names}
    for i, name in enumerate(candidates):
        out[name] = float(w[i])
    out["log_loss"] = float(objective(w))
    out["slsqp_success"] = res.success
    out["slsqp_message"] = res.message
    return out


def _compute_baseline_prior_poisson(samples, candidates, n):
    """基于 Poisson NLL 计算 Layer-1 先验。"""
    if len(samples) < 30:
        return np.array(
            [0.6 if name == "hgbr" else 0.4 / max(1, len(candidates) - 1)
             for name in candidates],
            dtype=float,
        )
    # 使用 goal NLL
    _nll = {}
    for name in candidates:
        _v = []
        for s in samples:
            try:
                lam_h = s.get(f"{name}_lam_h", 1.5)
                lam_a = s.get(f"{name}_lam_a", 1.2)
                gh = int(s.get("home_goals", 0))
                ga = int(s.get("away_goals", 0))
                from math import lgamma, log
                ll = -(gh * log(max(lam_h, 1e-12)) - lam_h - lgamma(gh + 1))
                ll -= (ga * log(max(lam_a, 1e-12)) - lam_a - lgamma(ga + 1))
                _v.append(-ll)
            except Exception:
                pass
        _nll[name] = float(np.mean(_v)) if _v else math.inf

    return _prior_from_nll(_nll, candidates)


def _compute_baseline_prior_1x2(samples, candidates, n):
    """基于 1X2 NLL 计算 Layer-2 先验。"""
    if len(samples) < 30:
        return np.array(
            [0.6 if name == "poisson" else 0.4 / max(1, len(candidates) - 1)
             for name in candidates],
            dtype=float,
        )

    _nll = {}
    for name in candidates:
        _v = []
        for s in samples:
            try:
                _p = np.clip(np.asarray(s[name], dtype=float), 1e-12, None)
                _p = _p / _p.sum()
                _v.append(-math.log(_p[int(s["actual"])]))
            except Exception:
                pass
        _nll[name] = float(np.mean(_v)) if _v else math.inf

    return _prior_from_nll(_nll, candidates)


def _prior_from_nll(_nll, candidates):
    """从 NLL 计算 prior(softmax)。"""
    _T = 0.15
    _logits = np.array([-_nll.get(n, math.inf) / _T for n in candidates])
    _inf_mask = np.array([not math.isfinite(_nll.get(n, math.inf)) for n in candidates])

    if np.all(_inf_mask):
        return np.ones(len(candidates)) / len(candidates)

    _m = _logits[~_inf_mask].max()
    _e = np.zeros(len(candidates))
    _e[~_inf_mask] = np.exp(_logits[~_inf_mask] - _m)
    return _e / _e.sum()


def load_weights(league_key: str, default: dict | None = None) -> dict:
    """加载该联赛学习到的权重(审查 A70A601 P1-2:异常分级)。"""
    w = dict(default or DEFAULT_WEIGHTS)
    path = _WEIGHTS_PATH
    if path is None:
        try:
            from app.core.paths import ARTIFACTS_DIR as _AD
            path = os.path.join(str(_AD), "ensemble", "ensemble_weights.json")
        except Exception:
            path = None
    if not path or not os.path.exists(path):
        return w
    try:
        with open(path, encoding="utf-8") as _wf:
            data = json.load(_wf)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as _e:
        raise ValueError(f"ensemble_weights.json 损坏: {_e}") from _e
    if not isinstance(data, dict):
        raise TypeError(
            f"ensemble_weights.json 顶层应为 dict,实际 {type(data).__name__}"
        )
    if league_key in data:
        _entry = data[league_key]
        if not isinstance(_entry, dict):
            raise TypeError(
                f"ensemble_weights[{league_key}] 应为 dict,实际 {type(_entry).__name__}"
            )
        # engine.py 需要 flat 格式(hgbr/elo/bayes/dc/nb/gbm)
        if _is_layered(_entry):
            return from_layered(_entry)  # layered → flat
        # 已经是 flat
        return _entry
    # 默认: DEFAULT_WEIGHTS 是 layered,需要转 flat
    return from_layered(w)
