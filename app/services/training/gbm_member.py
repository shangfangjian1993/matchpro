"""GBM 成员训练评估(评审 P1):5 联赛独立训练 + holdout 指标。

对比 HGBR-Poisson 成员(experiments 记录),数据驱动决定接入 Ensemble 权重。
用法: python -m app.services.gbm_member
"""
import sys

_ROOT = str(__import__('app.core.paths', fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    import numpy as np
    import pandas as pd

    from app.api.db import League, Match, init_db, session_scope
    from app.core.config import LeagueType
    from app.data.adapters import matches_to_dataframe
    from app.models.ensemble.gbm import GbmClassifier
    from app.models.loader import _load_model

    init_db()
    with session_scope():
        for lt in (LeagueType.PREMIER_LEAGUE, LeagueType.LA_LIGA,
                   LeagueType.BUNDESLIGA, LeagueType.SERIE_A, LeagueType.LIGUE_1):
            league = League.query.filter_by(league_type=lt.value).first()
            if league is None:
                continue
            model = _load_model(lt, "app/models")
            matches = Match.query.filter_by(league_id=league.id, match_status="finished").all()
            df = matches_to_dataframe(matches, league_name=league.name,
                                      league_season=league.season or "")
            prepared = model.prepare_features(df)
            cols = [c for c in model.feature_columns_ if c in prepared.columns]
            gh, ga = df["home_goals"].to_numpy(), df["away_goals"].to_numpy()
            y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
            gbm = GbmClassifier()
            m = gbm.train(prepared[cols], pd.Series(y, index=prepared.index))
            out = f"app/models/artifacts/{lt.value}/gbm.pkl"
            gbm.save(out)
            print(f"{lt.value}: GBM ll={m['log_loss']} brier={m['brier']} "
                  f"rps={m['rps']} acc={m['accuracy']} → {out}", flush=True)


if __name__ == "__main__":
    from app.services.cli import run
    raise SystemExit(run(main))
