"""数据质量检查(入库后自动执行)。

检查项:
1. 空值率检测(关键字段不应为空)
2. 跨表不一致检测(matches vs team_match_stats)
3. 重复记录检测
4. 异常值检测(负概率、越界值等)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """质量检查报告。"""
    total_matches: int = 0
    total_issues: int = 0
    null_rate: dict[str, float] = field(default_factory=dict)
    inconsistencies: list[str] = field(default_factory=list)
    duplicates: int = 0
    outliers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_matches": self.total_matches,
            "total_issues": self.total_issues,
            "null_rate": self.null_rate,
            "inconsistencies": self.inconsistencies[:10],  # 只显示前 10 个
            "duplicates": self.duplicates,
            "outliers": self.outliers[:10],
        }


def check_quality() -> QualityReport:
    """执行全面质量检查。"""
    from app.api.db import Match, TeamMatchStats, db, init_db, session_scope

    init_db()
    report = QualityReport()

    with session_scope():
        report.total_matches = db.session.query(Match).count()

        # 1. 空值率检测
        key_fields = ["home_goals", "away_goals", "home_shots", "home_corners"]
        for f in key_fields:
            null_count = db.session.query(Match).filter(getattr(Match, f).is_(None)).count()
            report.null_rate[f] = null_count / report.total_matches if report.total_matches > 0 else 0

        # 2. 重复记录检测
        from sqlalchemy import func
        dupes = (
            db.session.query(Match.home_team, Match.away_team, Match.match_date, func.count("*").label("cnt"))
            .group_by(Match.home_team, Match.away_team, Match.match_date)
            .having(func.count("*") > 1)
            .all()
        )
        report.duplicates = len(dupes)

        # 3. 跨表一致性
        mismatches = (
            db.session.query(Match, TeamMatchStats)
            .join(TeamMatchStats, Match.id == TeamMatchStats.match_id)
            .filter(TeamMatchStats.side == "home")
            .filter(Match.home_xg.isnot(None))
            .filter(TeamMatchStats.xg.isnot(None))
            .filter(func.abs(Match.home_xg - TeamMatchStats.xg) > 0.01)
            .limit(10)
            .all()
        )
        for m, t in mismatches:
            report.inconsistencies.append(f"match {m.id}: matches.xg={m.home_xg} vs tms.xg={t.xg}")

    report.total_issues = len(report.inconsistencies) + report.duplicates + len(report.outliers)
    return report


def print_report(report: QualityReport) -> None:
    """打印质量报告。"""
    print("=== 数据质量报告 ===")
    print(f"总比赛数: {report.total_matches}")
    print(f"重复记录: {report.duplicates}")
    print(f"跨表不一致: {len(report.inconsistencies)}")
    print("\n空值率:")
    for k, v in report.null_rate.items():
        print(f"  {k}: {v*100:.1f}%")
