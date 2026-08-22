#!/usr/bin/env python3
"""数据管线 CLI 入口。

用法:
    python scripts/pipeline.py --job daily
    python scripts/pipeline.py --job weekly
    python scripts/pipeline.py --job monthly
    python scripts/pipeline.py --job all
    python scripts/pipeline.py --type results --league premier_league
    python scripts/pipeline.py --type xg --league all
    python scripts/pipeline.py --type stats --league la_liga --season 2025
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="MatchPro 数据管线")
    parser.add_argument("--job", choices=["daily", "weekly", "monthly", "all"], default=None)
    parser.add_argument("--type", choices=["results", "stats", "odds", "xg", "tournaments", "injuries"], default=None)
    parser.add_argument("--league", default=None, help="联赛类型(premier_league/la_liga/...),不指定则采集全部")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--date", default=None, help="采集日期(YYYY-MM-DD)")
    parser.add_argument("--since-year", type=int, default=None)
    args = parser.parse_args()

    from app.data.pipeline import Pipeline

    pipe = Pipeline()

    if args.job:
        if args.job == "all":
            result = pipe.run_all()
        else:
            result = pipe.run_frequency(args.job)
    elif args.type:
        kwargs = {}
        if args.season:
            kwargs["season"] = args.season
        if args.date:
            kwargs["date"] = args.date
        if args.since_year:
            kwargs["since_year"] = args.since_year
        result = pipe.run(args.type, args.league, **kwargs)
    else:
        parser.print_help()
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
