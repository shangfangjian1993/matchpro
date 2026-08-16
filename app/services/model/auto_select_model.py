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
    models_dir = os.environ.get("MODELS_DIR", os.path.join(_ROOT, "app", "models"))
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
            pl = m.get("poisson_loss")
            if pl is None:
                continue
            cur = best.get(lt)
            if cur is None or pl < cur["poisson_loss"]:
                best[lt] = {"version": mv, "poisson_loss": pl,
                            "feature_version": getattr(e, "feature_version", None)}
        if not best:
            print("无历史实验,跳过")
            return
        active = {lt: v["version"] for lt, v in best.items()}
        out = os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), "runtime", "active_models.json")
        os.makedirs(models_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as _of:
            json.dump(active, _of, ensure_ascii=False, indent=2)
        print(f"✅ active_models.json 已写入 {out}")
        for lt, v in sorted(best.items()):
            print(f"   {lt}: v{v['version']} (poisson {v['poisson_loss']:.4f})")


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
