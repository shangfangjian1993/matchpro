"""P2-D:自动模型选择。

从 experiments 表聚合各联赛全部历史实验,按 holdout poisson_loss 最低者
写入 models/active_models.json —— predict 加载时优先该版本(active 指针)。
"""
import json
import os
import sys

_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.api.db import Experiment, init_db, session_scope
from app.models.registry import _model_path as _mp


def main():
    init_db()
    # 审查 P0-4:统一经 paths.MODELS_DIR(artifacts/models)
    from app.core.paths import MODELS_DIR as _MD
    models_dir = os.environ.get("MODELS_DIR", str(_MD))
    with session_scope():
        rows = Experiment.query.all()
        # 只考虑最新特征版本(与当前 prepare_features 代码兼容,防止旧特征缺列)
        fv_by_lt = {}
        for e in rows:
            lt = getattr(e, "league_type", None) or ""
            fv = getattr(e, "feature_version", None) or ""
            if lt and fv:
                fv_by_lt.setdefault(lt, []).append((e.id, fv))
        latest_fv = {lt: max(items, key=lambda x: x[0])[1] for lt, items in fv_by_lt.items()}
        best = {}
        for e in rows:
            lt = getattr(e, "league_type", None) or ""
            mv = getattr(e, "model_version", None) or ""
            if not lt or not mv or lt in ("global", "replay"):
                continue
            # 特征版本必须是最新的(否则 predict 时特征生成与模型白名单不匹配)
            if getattr(e, "feature_version", None) != latest_fv.get(lt):
                continue
            # 审查 P0-7:只考虑文件系统真实存在的版本(MODELS_KEEP prune 后
            # experiments 里的旧版本可能已删除 → 前置过滤,避免悬空指针/空选择)
            
            from app.core.config import LeagueType as _LT
            if not os.path.exists(_mp(_LT[lt.upper()], models_dir, mv)):
                continue
            try:
                m = json.loads(getattr(e, "metrics_json", None) or "{}")
            except Exception:
                continue
            # 审查 P1-12:多指标门禁 —— 单一 poisson_loss 最优不代表 1X2/校准最优。
            # 综合评分 = 0.40·poisson + 0.25·logloss + 0.20·brier + 0.15·rps
            # (四项均为越小越好;任一缺失或为 0 占位即拒绝候选 —— 旧版本实验
            #  概率指标未计算时存 0.0,0 值 logloss/brier/rps 会使综合分虚低)
            required = ("poisson_loss", "log_loss", "brier", "rps")
            if any(m.get(k) is None or m.get(k) <= 0 for k in required):
                continue
            score = (0.40 * m["poisson_loss"] + 0.25 * m["log_loss"]
                     + 0.20 * m["brier"] + 0.15 * m["rps"])
            cur = best.get(lt)
            # 并列分时选更新实验(id 大者 = 最新训练),避免"同指标旧版本"被选中
            if (cur is None or score < cur["score"] - 1e-12
                    or (abs(score - cur["score"]) <= 1e-12 and e.id > cur["id"])):
                best[lt] = {"version": mv, "score": score, "id": e.id,
                            "poisson_loss": m["poisson_loss"],
                            "log_loss": m["log_loss"], "brier": m["brier"],
                            "rps": m["rps"],
                            "feature_version": getattr(e, "feature_version", None)}
        if not best:
            print("无历史实验,跳过")
            return
        active = {lt: v["version"] for lt, v in best.items()}
        out = os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), "runtime", "active_models.json")
        # 审查 P1-12:审计元数据(选中的版本 + 各指标 + 综合分)
        meta_out = os.path.join(os.path.dirname(out), "active_meta.json")
        os.makedirs(models_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as _of:
            json.dump(active, _of, ensure_ascii=False, indent=2)
        with open(meta_out, "w", encoding="utf-8") as _mf:
            json.dump({lt: {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in info.items()}
                       for lt, info in best.items()}, _mf, ensure_ascii=False, indent=2)
        print(f"✅ active_models.json 已写入 {out}")
        for lt, v in sorted(best.items()):
            print(f"   {lt}: v{v['version']} (score {v['score']:.4f} | "
                  f"poisson {v['poisson_loss']:.4f} | ll {v['log_loss']:.4f} | "
                  f"brier {v['brier']:.4f} | rps {v['rps']:.4f})")


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
