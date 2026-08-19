"""权重配置与学习(审查 §36:ensemble 拆分)。"""

from __future__ import annotations

import json
import math
import os

import numpy as np

DEFAULT_WEIGHTS = {
    "hgbr": 1.0,
    "dc": 0.0,
    "nb": 0.0,
    "elo": 0.0,
    "gbm": 0.0,
    "bayes": 0.0,
}
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
    """shrinkage:向先验(默认 hgbr 主导)的 L2 收缩 —— 防止小样本 OOF 下
    学到极端权重(hgbr=0 / gbm=1 之类),把"概率收缩"式退化拉回平衡。

    max_weight:单成员权重上限 —— GBM 等分类器概率过度自信时 ll 最优但
    校准差,全押单一成员丧失 Ensemble 意义(审查九 P1-5 配套)。"""
    """滚动学习:SLSQP 优化 w≥0、Σw=1 最小化 log-loss(§4)。

    审查 P1-16:成员动态化 —— 样本中实际出现的成员才参与优化;
    GBM 不可用(加载失败/预测失败)→ 完全从优化中移除,而非以 [0,0,0] 假装存在。
    """
    names_all = ["hgbr", "dc", "nb", "elo", "gbm", "bayes"]
    present = [n for n in names_all if any(n in s for s in samples)]
    n = max(1, len(samples))

    _prior = np.array(
        [(prior or {}).get(name, 1.0 if name == "hgbr" else 0.0) for name in present],
        dtype=float,
    )
    _prior = _prior / _prior.sum()

    def _nll_pure(w):
        ll = 0.0
        for s in samples:
            p = np.zeros(3)
            for i, name in enumerate(present):
                p += w[i] * np.asarray(s[name])
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            ll -= math.log(p[s["actual"]])
        return ll / n

    def _nll(w):
        ll = 0.0
        for s in samples:
            p = np.zeros(3)
            for i, name in enumerate(present):
                p += w[i] * np.asarray(s[name])
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            ll -= math.log(p[s["actual"]])
        ll = ll / n
        # 审查十二:小样本权重过拟合 → L2 收缩向先验(默认 hgbr 主导)
        if shrinkage > 0 and len(present) > 1:
            ll += shrinkage * float(np.mean((w - _prior) ** 2))
        return ll

    # 只有 1 个成员时无需优化
    if len(present) < 2:
        w = {name: (1.0 if name == present[0] else 0.0) for name in names_all}
        w["log_loss"] = float(_nll([1.0]))
        w["n"] = n
        return w

    from scipy.optimize import minimize

    w0 = np.array([1.0 if name == "hgbr" else 0.0 for name in present])
    res = minimize(
        _nll,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(present),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 200, "ftol": 1e-9},
    )
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    out = {name: 0.0 for name in names_all}
    for i, name in enumerate(present):
        out[name] = float(w[i])
    # 报告的 log_loss 用纯 NLL(不含收缩惩罚项)
    out["log_loss"] = float(_nll_pure(w))
    out["n"] = n
    out["shrinkage"] = shrinkage
    return out


def load_weights(league_key: str, default: dict | None = None) -> dict:
    """加载该联赛学习到的权重;无文件/无该联赛时回退默认。"""
    w = dict(default or DEFAULT_WEIGHTS)
    path = _WEIGHTS_PATH
    if path is None:
        try:
            from app.core.paths import ARTIFACTS_DIR as _AD

            path = os.path.join(str(_AD), "ensemble", "ensemble_weights.json")
        except Exception:
            path = None
    try:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as _wf:
                data = json.load(_wf)
            if league_key in data:
                for k in ("hgbr", "dc", "nb", "elo", "gbm", "bayes"):
                    if k in data[league_key]:
                        w[k] = float(data[league_key][k])
    except Exception:
        pass
    return w
