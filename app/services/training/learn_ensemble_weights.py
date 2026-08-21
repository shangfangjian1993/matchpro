"""P2-A:Ensemble 权重滚动学习 —— 时间分段 OOF(审查 P0-3/P0-4,2026-08 重写)。

本文件仅为 CLI 薄壳;核心逻辑拆分为 app/services/training/ensemble/
(oof_generator / temporary_trainer / member_builder / weight_optimizer /
artifact_writer)—— 审查九 P1-9 拆分,支持后续 Dynamic/Rolling Ensemble。

审查 P1-5:OOF 样本量 120 → 600(6 段 × 100/段,下限 300)。
"""

import os
import sys

_ROOT = str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT)
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    from app.api.db import League, Match, init_db, session_scope
    from app.core.config import LeagueType
    from app.services.cli import add_log_level_arg, make_parser, setup_logging

    ap = make_parser("Ensemble 权重学习(时间分段 OOF:段前训练 → 段内预测 → SLSQP)")
    add_log_level_arg(ap)
    args = ap.parse_args()
    setup_logging(args.log_level)

    init_db()
    with session_scope():
        from app.services.training.ensemble import run_all

        leagues = []
        for lt in [
            LeagueType.PREMIER_LEAGUE,
            LeagueType.LA_LIGA,
            LeagueType.BUNDESLIGA,
            LeagueType.SERIE_A,
            LeagueType.LIGUE_1,
        ]:
            league = League.query.filter_by(league_type=lt.value).first()
            if league is None:
                continue
            matches = Match.query.filter_by(
                league_id=league.id, match_status="finished"
            ).all()
            matches.sort(key=lambda m: str(m.match_date or ""))
            leagues.append((lt, league, matches))
        run_all(leagues)
    print("\n✅ Ensemble OOF 权重学习完成")


if __name__ == "__main__":
    from app.services.cli import run

    raise SystemExit(run(main))
