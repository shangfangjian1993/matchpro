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
#   goal_lambda        λ 融合: HGBR/ELO/Bayes(独立 λ 成员)
#   score_distribution 分布形态: Poisson(hgbr+elo+bayes 共享的 Poise 基)/DC/NB
#   outcome           1×2 分类器: GBM(独立于 λ/矩阵)
# 加载时兼容旧扁平文件(自动 from_layered 平移);学习仍沿用 OOF 优化(扁平),
# 落盘与审计按三层视图。poisson 基权重 = 全部 Poise 型成员质量之和。
LAYER_MAP = {
    "goal_lambda": ("hgbr", "elo", "bayes"),
    "score_distribution": ("poisson", "dc", "nb"),
    "outcome": ("gbm",),
}


def to_layered(flat: dict) -> dict:
    """扁平权重 → 三层视图(goal 成员归一;poisson 基=hgbr+elo+bayes;sd 归一)。"""
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
    """三层视图 → 扁平(供既有融合/矩阵逻辑消费;goal 归一权重 = 扁平归一一致)。"""
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
    """shrinkage:向先验(默认 hgbr 主导)的 L2 收缩 —— 防止小样本 OOF 下
    学到极端权重(hgbr=0 / gbm=1 之类),把"概率收缩"式退化拉回平衡。

    max_weight:单成员权重上限 —— GBM 等分类器概率过度自信时 ll 最优但
    校准差,全押单一成员丧失 Ensemble 意义(审查九 P1-5 配套)。"""
    """滚动学习:SLSQP 优化 w≥0、Σw=1 最小化 log-loss(§4)。

    审查 P1-16:成员动态化 —— 样本中实际出现的成员才参与优化;
    GBM 不可用(加载失败/预测失败)→ 完全从优化中移除,而非以 [0,0,0] 假装存在。
    """
    names_all = ["hgbr", "dc", "nb", "elo", "gbm", "bayes"]
    # 部分缺失成员在融合循环会 KeyError(materialize),且优化用全部样本分母
    _effective = {n: sum(1 for s in samples if n in s) for n in names_all}
    present = [n for n in names_all if _effective[n] >= max(1, 0.9 * len(samples))]
    n = max(1, len(samples))

    # 旧实现:小样本/新成员 OOF 证据不足时,L2 永远拉向 HGBR → 新成员极难
    # 挑战 HGBR,权重学习有效自由度低(DC/NB 与 HGBR 高度相关时尤其明显)。
    # 新实现:prior 由 **out-of-time OOF 平均负对数似然** 数据驱动
    #   prior_i ∝ exp(-mean_nll_i / T),T=0.15
    # "谁在 OOT 数据上真正贡献增量信息,谁拿更高收缩先验" —— baseline-aware。
    if prior is None:
        if len(samples) >= 30:
            _nll = {}
            for _name in present:
                _v = []
                for s in samples:
                    try:
                        _p = np.clip(np.asarray(s[_name], dtype=float), 1e-12, None)
                        _p = _p / _p.sum()
                        _v.append(-math.log(_p[int(s["actual"])]))
                    except Exception:
                        pass
                _nll[_name] = float(np.mean(_v)) if _v else math.inf
            _T = 0.15
            _inf_mask = np.array(
                [not math.isfinite(_nll.get(n_, math.inf)) for n_ in present]
            )
            _logits = np.array([-_nll[n_] / _T for n_ in present])
            if np.all(_inf_mask):
                _prior = np.ones(len(present)) / len(present)
            else:
                _m = _logits[~_inf_mask].max()
                _e = np.zeros(len(present))
                _e[~_inf_mask] = np.exp(_logits[~_inf_mask] - _m)
                _prior = _e / _e.sum()
        else:
            # 样本极少(过拟合风险):温和默认 —— hgbr 0.6,其余均分
            _prior = np.array(
                [
                    0.6
                    if name == "hgbr"
                    else (0.4 / (len(present) - 1) if len(present) > 1 else 1.0)
                    for name in present
                ],
                dtype=float,
            )
            _prior = _prior / _prior.sum()
    else:
        _prior = np.array([prior.get(name, 0.0) for name in present], dtype=float)
        _prior = (
            _prior / _prior.sum()
            if _prior.sum() > 0
            else (np.ones(len(present)) / len(present))
        )

    def _nll_pure(w):
        ll = 0.0
        for s in samples:
            p = np.zeros(3)
            for i, name in enumerate(present):
                if name not in s:
                    continue  # 该样本无此成员:mask(不贡献、不报错)
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
                if name not in s:
                    continue  # 缺失成员 mask
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

    # 初始点 = 基线先验(可行:每维 ≤ max_weight、和为 1)—— 不从 infeasible 点开始
    w0 = np.asarray(_prior, dtype=float).copy()
    if w0.sum() > 0:
        w0 = w0 / w0.sum() * min(1.0, max_weight * len(w0))
    w0 = np.clip(w0, 0.0, max_weight)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(len(present), 1.0 / len(present))
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
