"""统一采集编排(重构:按实效性频次执行,主源失败自动降级备用源)。

入口:
    python -m app.services.data.collector --freq daily      # 每日:赛果/xG/伤停
    python -m app.services.data.collector --freq weekly     # 每周:统计/赔率增量
    python -m app.services.data.collector --freq monthly    # 每月:欧战历史/新季 merge
    python -m app.services.data.collector --type results    # 单类型

返回结构化报告(供调度器/通知): {type: {source, ok, detail}}。
"""

from __future__ import annotations

import logging
import sys

_ROOT = str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)

_LEAGUES_EURO = ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1")


def _run_bzzoiro(fn_name: str, league: str) -> dict:
    from app.data.sources import bzzoiro

    fn = getattr(bzzoiro, fn_name)
    if not bzzoiro.available():
        return {"ok": False, "detail": "缺 BZZOIRO_KEY"}
    r = fn(league)
    return {
        "ok": True,
        "detail": {k: r[k] for k in ("fetched", "inserted", "updated") if k in r} or r,
    }


def _run_understat(season) -> dict:
    from app.data.pipeline import run_xg

    r = run_xg([season], ["E0", "SP1", "D1", "I1", "F1"])
    return {
        "ok": not r.get("errors"),
        "detail": {"updated": r.get("updated"), "errors": len(r.get("errors", []))},
    }


def _run_fdco(season) -> dict:
    from app.data.pipeline import run_history

    r = run_history([season], ["E0", "SP1", "D1", "I1", "F1"])
    return {
        "ok": not r.get("errors"),
        "detail": {
            "inserted": r.get("inserted"),
            "updated": r.get("updated"),
            "errors": len(r.get("errors", [])),
        },
    }


def _run_zafronix(season) -> dict:
    from app.data.sources import zafronix

    if not zafronix.available():
        return {"ok": False, "detail": "缺 ZAFRONIX_KEY"}
    out = {}
    for lt, y in (("world_cup", season), ("european_championship", season)):
        try:
            r = zafronix.ingest_year(lt, y, verbose=False)
            out[lt] = r.get("inserted", 0)
        except Exception as e:
            out[lt] = type(e).__name__
    return {"ok": bool(out), "detail": out}


_TYPE_RUNNERS = {
    "results": {
        "bzzoiro_events": lambda: _run_bzzoiro("import_recent", "premier_league"),
        "fdco": lambda: _run_fdco(_current_season()),
    },
    "xg": {"understat": lambda: _run_understat(_current_season())},
    "tournaments": {"zafronix": lambda: _run_zafronix(_current_season())},
    "stats": {"bzzoiro_stats": lambda: _run_bzzoiro_incr("ingest_stats")},
    "odds": {"bzzoiro_odds": lambda: _run_bzzoiro_incr("ingest_odds")},
    "injuries": {"api_football": lambda: {"ok": False, "detail": "伤停采集待接入"}},
}


def _current_season() -> int:
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _run_bzzoiro_incr(fn_name: str) -> dict:
    """近 20 季窗口的 stats/odds 增量(截断早期页,续跑进度)。"""
    import datetime as dt

    from app.data.sources import bzzoiro
    from app.data.sources.bzzoiro import LEAGUE_IDS, _get
    from app.data.sources_registry import STATS_SEASONS

    if not bzzoiro.available():
        return {"ok": False, "detail": "缺 BZZOIRO_KEY"}
    total_done = 0
    for league in _LEAGUES_EURO:
        lid = LEAGUE_IDS.get(league)
        if lid is None:
            continue
        # 窗口截止
        d0 = _get("/events/", {"league_id": lid, "status": "finished", "limit": 1})
        latest = dt.datetime.fromisoformat(
            d0["results"][0]["event_date"].replace("Z", "+00:00")
        )
        last_season = latest.year if latest.month >= 8 else latest.year - 1
        cutoff = dt.datetime(
            last_season - STATS_SEASONS + 1, 8, 1, tzinfo=dt.timezone.utc
        )
        total = d0.get("count", 0)
        ofs, done = 0, 0
        while ofs < total:
            batch = bzzoiro.fetch_events(
                league_id=lid, status="finished", limit=200, offset=ofs
            )
            if not batch:
                break
            batch = [
                e
                for e in batch
                if dt.datetime.fromisoformat(e["event_date"].replace("Z", "+00:00"))
                >= cutoff
            ]
            if batch:
                fn = getattr(bzzoiro, fn_name)
                r = fn(league, limit_events=len(batch), offset=ofs, verbose=False)
                done += r.get("updated", r.get("written", 0))
            ofs += 200
            if not batch:
                continue
            total_done += done
        logger.info("  %s %s: 累计 %d", league, fn_name, done)
    return {"ok": True, "detail": {"updated_total": total_done}}


def run_type(dtype: str) -> dict:
    """执行单类型:primary 失败 → fallback。返回 {source, ok, detail}。"""
    from app.data.sources_registry import PRIMARY_FALLBACK

    primary, fallback = PRIMARY_FALLBACK.get(dtype, (None, None))
    runners = _TYPE_RUNNERS.get(dtype, {})
    for source in (primary, fallback):
        if source is None or source not in runners:
            continue
        try:
            res = runners[source]()
        except Exception as e:
            logger.warning("源 %s 异常: %s", source, e)
            res = {"ok": False, "detail": type(e).__name__}
        if res.get("ok"):
            return {"source": source, "ok": True, "detail": res.get("detail")}
        logger.warning(
            "源 %s 未达标(fallback→%s): %s", source, fallback, res.get("detail")
        )
    return {"source": primary, "ok": False, "detail": "primary 与 fallback 均失败"}


def run_frequency(freq: str) -> dict:
    from app.data.sources_registry import FREQUENCY_TYPES

    report = {}
    for dtype in FREQUENCY_TYPES.get(freq, []):
        report[dtype] = run_type(dtype)
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["realtime", "daily", "weekly", "monthly"])
    ap.add_argument(
        "--type",
        dest="dtype",
        choices=list(
            __import__(
                "app.data.sources_registry", fromlist=["PRIMARY_FALLBACK"]
            ).PRIMARY_FALLBACK
        ),
    )
    args = ap.parse_args()
    import json

    from app.api.db import init_db

    init_db()
    report = (
        run_frequency(args.freq) if args.freq else {args.dtype: run_type(args.dtype)}
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(v.get("ok", False) for v in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
