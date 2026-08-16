"""权重配置与学习(审查 §36:ensemble 拆分)。"""
from __future__ import annotations

import json
import math
import os

import numpy as np

DEFAULT_WEIGHTS = {"hgbr": 1.0, "dc": 0.0, "nb": 0.0, "elo": 0.0, "gbm": 0.0}
_WEIGHTS_PATH = None


def set_weights_path(path: str | None):
    global _WEIGHTS_PATH
    _WEIGHTS_PATH = path


def learn_weights(samples: list[dict], tau: float = 0.0, phi: float = 1e9) -> dict:
    """滚动学习:SLSQP 优化 w≥0、Σw=1 最小化 log-loss(§4;成员含 gbm)。"""
    names = ["hgbr", "dc", "nb", "elo", "gbm"]
    n = max(1, len(samples))

    def _nll(w):
        ll = 0.0
        for s in samples:
            p = np.zeros(3)
            for i, name in enumerate(names):
                p += w[i] * np.asarray(s.get(name, s["hgbr"]))
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            ll -= math.log(p[s["actual"]])
        return ll / n

    from scipy.optimize import minimize
    w0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    res = minimize(_nll, w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * 5,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                   options={"maxiter": 200, "ftol": 1e-9})
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    return {"hgbr": float(w[0]), "dc": float(w[1]), "nb": float(w[2]),
            "elo": float(w[3]), "gbm": float(w[4]),
            "log_loss": float(_nll(w)), "n": n}


def load_weights(league_key: str, default: dict | None = None) -> dict:
    """加载该联赛学习到的权重;无文件/无该联赛时回退默认。"""
    w = dict(default or DEFAULT_WEIGHTS)
    path = _WEIGHTS_PATH
    if path is None:
        try:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
                "artifacts", "ensemble", "ensemble_weights.json")
        except Exception:
            path = None
    try:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as _wf:
                data = json.load(_wf)
            if league_key in data:
                for k in ("hgbr", "dc", "nb", "elo", "gbm"):
                    if k in data[league_key]:
                        w[k] = float(data[league_key][k])
    except Exception:
        pass
    return w
