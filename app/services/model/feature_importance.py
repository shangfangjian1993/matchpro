"""特征重要性报告(评审 2.1):permutation importance(poisson 损失下降)。

HGBR 无内置 feature_importances_(sklearn 限制),用 permutation importance
在时间 holdout 上评估 78 特征对预测质量的影响,识别低贡献特征(消融候选)。
"""
import sys

_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _scorer(model, X, y):
    """负 poisson 损失(越大越好,permutation_importance 需要最大化)。"""
    pred = model.predict(X)["predictions"]
    import numpy as np
    return -np.mean(np.maximum(pred, 1e-10) - y * np.log(np.maximum(pred, 1e-10)))


def main():
    import numpy as np

    from app.api.db import League, Match, init_db, session_scope
    from app.core.config import LeagueType
    from app.data.adapters import matches_to_dataframe
    from app.features.registry import family_of
    from app.models.loader import _load_model

    init_db()
    with session_scope():
        league = League.query.filter_by(league_type="premier_league").first()
        model = _load_model(LeagueType.PREMIER_LEAGUE, str(__import__("app.core.paths", fromlist=["MODELS_DIR"]).MODELS_DIR))
        matches = Match.query.filter_by(league_id=league.id, match_status="finished").all()
        df = matches_to_dataframe(matches, league_name=league.name,
                                  league_season=league.season or "")
        # 滚动特征由 prepare_features 生成(与训练同口径)
        prepared = model.prepare_features(df)
        cols = [c for c in model.feature_columns_ if c in prepared.columns]
        n = len(prepared); split = int(n * 0.8)
        X_test = prepared[cols].iloc[split:]
        y_test = df["home_goals"].astype(float).iloc[split:]

        print(f"holdout: {len(X_test)} 场 | 特征 {len(cols)} | permutation importance...", flush=True)
        # 手动 permutation importance(模型无 fit 方法,sklearn 校验拒绝)
        rng = np.random.default_rng(42)
        base_score = _scorer(model, X_test, y_test)
        imps = {}
        for i, col in enumerate(cols, 1):
            drops = []
            for _ in range(2):
                X_p = X_test.copy()
                X_p[col] = rng.permutation(X_p[col].to_numpy())
                drops.append(base_score - _scorer(model, X_p, y_test))
            imps[col] = float(np.mean(drops))
            if i % 10 == 0:
                print(f"  [{i}/{len(cols)}]", flush=True)
        items = sorted(imps.items(), key=lambda kv: -kv[1])
        print("\n=== Top 15(poisson 损失上升幅度)===")
        for name, v in items[:15]:
            print(f"  {name}: {v:.4f} [{family_of(name)}]")
        print("\n=== Bottom 10(消融候选)===")
        for name, v in items[-10:]:
            print(f"  {name}: {v:.4f} [{family_of(name)}]")
        fam = {}
        for name, v in items:
            fam[family_of(name)] = fam.get(family_of(name), 0.0) + v
        total = sum(fam.values())
        print("\n=== 家族重要性占比 ===")
        for f, v in sorted(fam.items(), key=lambda kv: -kv[1]):
            print(f"  {f}: {100*v/total:.1f}%")
        low = [(n, v) for n, v in items if v < 0.0005]
        print(f"\n低贡献特征({len(low)}):", [n for n, _ in low][:12])


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
