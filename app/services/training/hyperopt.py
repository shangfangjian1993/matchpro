"""HGBR 超参搜索(评审 P0-2.1):在英超 holdout 上按 Log-Loss/RPS 选优。

候选:lr ∈ {0.03, 0.06, 0.1} × max_depth ∈ {4, 6}(max_iter=300 固定)。
评估:三分类概率指标(log_loss/brier/rps,复用 _holdout_prob_metrics,防泄漏)。
用法: python -m app.services.hyperopt
"""
import json
import os
import sys

_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CANDIDATES = [
    {"learning_rate": lr, "max_depth": md, "max_iter": 300,
     "min_samples_leaf": 10, "early_stopping_rounds": 25,
     "validation_fraction": 0.15, "random_state": 42, "loss": "poisson"}
    for lr in (0.03, 0.06, 0.1) for md in (4, 6)
]


def main():
    from app.api.db import League, Match, init_db, session_scope
    from app.core.config import LeagueType, ModelConfig
    from app.data.adapters import matches_to_dataframe
    from app.models.poisson.league_factory import LeagueModelFactory
    from app.services.training.trainer import _holdout_prob_metrics

    init_db()
    with session_scope():
        league = League.query.filter_by(league_type="premier_league").first()
        matches = Match.query.filter_by(league_id=league.id, match_status="finished").all()
        df = matches_to_dataframe(matches, league_name=league.name,
                                  league_season=league.season or "")
        print(f"数据: {len(df)} 场,超参搜索 {len(CANDIDATES)} 组合", flush=True)

        results = []
        for i, params in enumerate(CANDIDATES, 1):
            cfg = ModelConfig(model_type="HGBR", league_type=LeagueType.PREMIER_LEAGUE,
                              version="search", parameters=params)
            model = LeagueModelFactory.create_league_model(LeagueType.PREMIER_LEAGUE, cfg)
            try:
                model.train(df, "goals")
                pm = _holdout_prob_metrics(model, df, "goals")
                results.append({"params": params, **pm})
                print(f"  [{i}/{len(CANDIDATES)}] lr={params['learning_rate']} "
                      f"depth={params['max_depth']} → ll={pm.get('log_loss', 'N/A')} "
                      f"brier={pm.get('brier', 'N/A')} rps={pm.get('rps', 'N/A')}", flush=True)
            except Exception as e:
                print(f"  [{i}] 失败: {str(e)[:100]}", flush=True)

        valid = [r for r in results if r.get("log_loss")]
        if not valid:
            print("无有效结果"); return
        best = min(valid, key=lambda r: (r.get("log_loss", 9e9), r.get("rps", 9e9)))
        print("\n=== 最优 ===")
        print(json.dumps(best, ensure_ascii=False, indent=2))
        # 对比基线(当前配置 lr=0.06 depth=6)
        base = [r for r in valid if r["params"]["learning_rate"] == 0.06
                and r["params"]["max_depth"] == 6]
        if base:
            print("基线(lr=0.06 d=6):", {k: base[0][k] for k in ("log_loss", "brier", "rps")})
        # 建议写入 configs/leagues.yaml
        out = {
            "best_params": best["params"],
            "metrics": {k: best[k] for k in ("log_loss", "brier", "rps")},
        }
        _exp_dir = os.path.join(_ROOT, "artifacts", "experiments", "hyperopt")
        os.makedirs(_exp_dir, exist_ok=True)
        with open(os.path.join(_exp_dir, "best.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n✅ 已保存 artifacts/experiments/hyperopt/best.json")


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
