"""权重配置与学习(审查 §36:ensemble 拆分)。"""

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

# 兼容旧扁平格式(仅用于加载历史权重文件)
# 新代码应直接使用分层格式
LAYER_MAP = {
    "goal_lambda": ("hgbr", "elo", "bayes"),
    "score_distribution": ("poisson", "dc", "nb"),
    "outcome": ("gbm",),
}


def to_layered(flat: dict) -> dict:
    """扁平权重(旧格式) → 三层视图。仅用于兼容历史权重文件加载。"""
    goal = {k: float(flat.get(k, 0.0)) for k in LAYER_MAP["goal_lambda"]}
    gsum = sum(goal.values()) or 1.0
    sd = {
        "poisson": gsum,
        "dc": float(flat.get("dc", 0.0)),
        "nb": float(flat.get("nb", 0.0)),
    }
    ssum = sum(sd.values()) or 1.0
    return {
        "goal_lambda": {k: round(v / gsum, 4) for k, v in goal.items()},
        "score_distribution": {k: round(v / ssum, 4) for k, v in sd.items()},
        "outcome": {"gbm": round(float(flat.get("gbm", 0.0)), 4)},
    }


def from_layered(layered: dict) -> dict:
    """三层视图 → 扁平(旧格式)。仅用于兼容历史权重文件加载。"""
    goal = layered.get("goal_lambda", {})
    sd = layered.get("score_distribution", {})
    gb = layered.get("outcome", {}).get("gbm", 0.0)
    poisson = float(sd.get("poisson", 0.0))
    # poisson 基权重按 goal 成员比例回分(hgbr/elo/bayes)
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


def learn_weights(
    samples: list[dict],
    tau: float = 0.0,
    phi: float = 1e9,
    shrinkage: float = 0.15,
    prior: dict | None = None,
    max_weight: float = 0.7,
) -> dict:
    """分层权重学习:Layer-1(goal lambda) + Layer-2(score distribution) 分别优化。

    与生产 LayeredPipeline 一致:
      Layer-1: lambda = w_h*lambda_hgbr + w_e*lambda_elo + w_b*lambda_bayes
      Layer-2: P = a_p*Pois(fused) + a_dc*DC(fused) + a_nb*NB(fused)

    返回: 兼容旧格式的 flat dict(hgbr/elo/bayes/dc/nb/gbm),但数学上对应分层权重。
    """
    _candidates_gl = ["hgbr", "elo", "bayes"]
    _candidates_sd = ["poisson", "dc", "nb"]
    
    _present_gl = [n for n in _candidates_gl if any(n in s for s in samples)]
    _present_sd = [n for n in _candidates_sd if any(n in s for s in samples)]
    
    n = max(1, len(samples))
    
    # Layer-1: Goal λ 权重
    w_gl = _optimize_layer_weights(samples, _present_gl, shrinkage, max_weight, n, prior)
    
    # Layer-2: Score Distribution 权重
    w_sd = _optimize_layer_weights(samples, _present_sd, shrinkage, max_weight, n, prior)
    
    # 组合为 flat 格式(兼容旧接口)
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


def _optimize_layer_weights(samples, candidates, shrinkage, max_weight, n, prior):
    """优化单层权重(goal_lambda 或 score_distribution)。"""
    all_names = ["hgbr", "elo", "bayes", "poisson", "dc", "nb"]
    if not candidates:
        return {name: 1.0 if name == "poisson" else 0.0 for name in all_names}
    
    if prior is None:
        _prior = _compute_baseline_prior(samples, candidates, n)
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
    
    from scipy.optimize import minimize
    w0 = np.asarray(_prior, dtype=float).copy()
    w0 = np.clip(w0, 0.0, max_weight)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(len(candidates), 1.0 / len(candidates))
    
    res = minimize(
        _nll, w0, method="SLSQP",
        bounds=[(0.0, max_weight)] * len(candidates),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 200, "ftol": 1e-9},
    )
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    
    out = {name: 0.0 for name in all_names}
    for i, name in enumerate(candidates):
        out[name] = float(w[i])
    out["log_loss"] = float(_nll(w))
    return out


def _compute_baseline_prior(samples, candidates, n):
    """基于 OOF 平均 NLL 计算先验(Baseline-aware)。"""
    if len(samples) < 30:
        return np.array(
            [0.6 if name in ["hgbr", "poisson"] else 0.4 / max(1, len(candidates) - 1)
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
    """加载该联赛学习到的权重(审查 A70A601 P1-2:异常分级)。

    - 路径不可解析/文件不存在 → 回退默认(degraded,合法)。
    - 文件**存在但损坏**(JSON 解析失败)或值非法/越界 → raise ValueError:
      不得伪装成默认配置 —— 上层(engine)需据此 fail-fast。
    """
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
        if _is_layered(_entry):
            _entry = from_layered(_entry)
        for k in ("hgbr", "dc", "nb", "elo", "gbm", "bayes"):
            if k not in _entry:
                continue
            try:
                _val = float(_entry[k])
            except (TypeError, ValueError) as _e:
                raise ValueError(f"权重 {k} 非法值: {_entry[k]!r}") from _e
            if not math.isfinite(_val) or not (0.0 <= _val <= 1.0):
                raise ValueError(f"权重 {k} 越界/非有限: {_val!r}")
            w[k] = _val
    return w
