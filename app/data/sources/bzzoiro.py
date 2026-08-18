"""Bzzoiro Sports API v2 接入(新数据源,Token 鉴权)。

文档: https://sports.bzzoiro.com/docs/football/
Base:  https://sports.bzzoiro.com/api/v2/
Auth:  Authorization: Token <key>

资源(实测可用):
- /api/v2/events/          39.5 万场历史+未来(教练/裁判/半场/天气/德比/中立场/旅途...)
- /api/v2/events/live/     实时
- /api/v2/odds/            近期 1x2 盘口(consensus 66 家聚合 + implied probability + movement)
- /api/v2/leagues/         80 联赛
- /api/v2/predictions/     第三方预测(仅参考)

价值评估:
- odds:真正弥补我们唯一缺口(赛前 consensus 赔率;现盘近 3 天)
- events:覆盖全球多联赛 + 增强字段(我们 5 大联赛历史已有,增量在字段/更多联赛)

用法:
    python pipelines/ingest.py --bzzoiro-odds [--league premier_league]
"""
from __future__ import annotations

import logging
import os
from app.data.canonical.cleanse import NormalizedMatch

logger = logging.getLogger(__name__)

BASE = "https://sports.bzzoiro.com/api/v2/_"


def _key() -> str:
    k = os.environ.get("BZZOIRO_KEY") or ""
    if not k:
        for line in open(os.path.join(str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT), ".env"), encoding="utf-8"):
            if line.startswith("BZZOIRO_KEY"):
                k = line.split("=", 1)[1].strip()
                break
    return k


def available() -> bool:
    return bool(_key())


def _get(path: str, params: dict | None = None) -> dict:
    import json
    import time
    url = f"{BASE.replace('_','')}{path.lstrip('/')}"
    if params:
        import urllib.parse
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={"Authorization": f"Token {_key()}",
                                               "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("bzzoiro 请求重试耗尽")


def fetch_events(league_id: int | None = None, status: str | None = None,
                 limit: int = 50, offset: int = 0) -> list[dict]:
    """历史/未来事件(分页)。"""
    d = _get("/events/", {"league_id": league_id, "status": status,
                          "limit": limit, "offset": offset})
    return d.get("results") or []


def fetch_odds(event_id: int | None = None, market: str = "1x2",
               limit: int = 50, offset: int = 0) -> list[dict]:
    """盘口(默认 consensus 1x2)。"""
    d = _get("/odds/", {"event_id": event_id, "market": market,
                        "limit": limit, "offset": offset})
    return d.get("results") or []


def find_league(name: str) -> int | None:
    """按名称(子串,不敏感)找 league_id。"""
    for off in (0, 50):
        for lg in _leagues_page(off):
            if name.lower() in (lg.get("name") or "").lower():
                return lg.get("id")
    return None


def _leagues_page(offset: int = 0) -> list[dict]:
    d = _get("/leagues/", {"limit": 50, "offset": offset})
    return d.get("results") or []


# LeagueType.value → bzzoiro league_id
LEAGUE_IDS = {
    "premier_league": 1,
    "la_liga": 3,
    "bundesliga": 5,
    "serie_a": 4,
    "ligue_1": 6,
    "champions_league": 7,
    "europa_league": 8,
}

import time as _time
import datetime as _dt


def to_normalized(raw: dict, league_type_value: str) -> NormalizedMatch:
    """bzzoiro event → NormalizedMatch(upsert 幂等)。"""
    date = _dt.datetime.fromisoformat(raw["event_date"].replace("Z", "+00:00"))
    date = date.replace(tzinfo=None)
    return NormalizedMatch(
        league_type=league_type_value,
        date=date,
        home_team=str(raw["home_team"]).strip(),
        away_team=str(raw["away_team"]).strip(),
        home_goals=int(raw["home_score"]) if raw.get("home_score") is not None else None,
        away_goals=int(raw["away_score"]) if raw.get("away_score") is not None else None,
        home_ht_goals=int(raw["home_score_ht"]) if raw.get("home_score_ht") is not None else None,
        away_ht_goals=int(raw["away_score_ht"]) if raw.get("away_score_ht") is not None else None,
        match_status="finished",
        season_label=_season_label(date),
        match_stage=str(raw.get("round_name") or raw.get("stage") or ""),
    )


def _season_label(date: _dt.datetime) -> str:
    y = date.year if date.month >= 8 else date.year - 1
    return f"{y}-{y + 1}"


def import_league(league_type_value: str, since_year: int | None = None,
                  verbose: bool = True) -> dict:
    """全量导入某联赛(翻页,upsert 幂等);since_year 可选只导入某年及以后。

    返回 {"fetched": n, "inserted": n, "updated": n, "skipped": n, "errors": n}。
    """
    from app.data.canonical.ingest import upsert_matches
    league_id = LEAGUE_IDS.get(league_type_value)
    if league_id is None:
        raise ValueError(f"bzzoiro 未映射: {league_type_value}")
    rows, offset, limit = [], 0, 100
    while True:
        d = _get("/events/", {"league_id": league_id, "status": "finished",
                              "limit": limit, "offset": offset})
        batch = d.get("results") or []
        if not batch:
            break
        if since_year:
            batch = [b for b in batch if _dt.datetime.fromisoformat(
                b["event_date"].replace("Z", "+00:00")).date().year >= since_year]
        rows.extend(batch)
        total = d.get("count", 0)
        offset += limit
        if offset >= total or len(batch) < limit:
            break
        _time.sleep(0.35)  # 免费套餐稳健
    normalized = [to_normalized(r, league_type_value) for r in rows]
    res = upsert_matches(normalized)
    res["fetched"] = len(rows)
    if verbose:
        print(f"  {league_type_value}: 拉取 {len(rows)} 场 → "
              f"新增 {res['inserted']} 更新 {res['updated']} 跳过 {res['skipped']} "
              f"错误 {len(res['errors'])}", flush=True)
    return res


def _norm(name: str) -> str:
    """队名归一(与 api-football 一致风格,用于 bzzoiro↔fdco 对齐)。"""
    import re
    n = str(name).lower()
    n = re.sub(r"\b(fc|cf|sc|afc|acf|wanderers|club)\b", " ", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def merge_league(league_type_value: str, since_year: int | None = None,
                 verbose: bool = True) -> dict:
    """按 bzzoiro 主导覆盖合并(bzzoiro 是权威覆盖源)。

    对 bzzoiro 每场:归一队名在 DB 同联赛(±3 天)找旧行 —— 找到则更新
    (保留 id/球队实体,比分/时间以 bzzoiro 为准);否则新增。
    返回 {"fetched":n,"updated":n,"inserted":n,"errors":n}。
    """
    from app.api.db import db, session_scope
    from app.api.db import League, Match
    from app.data.canonical.ingest import upsert_matches

    league_id = LEAGUE_IDS.get(league_type_value)
    if league_id is None:
        raise ValueError(f"bzzoiro 未映射: {league_type_value}")

    rows, offset, limit = [], 0, 100
    while True:
        d = _get("/events/", {"league_id": league_id, "status": "finished",
                              "limit": limit, "offset": offset})
        batch = d.get("results") or []
        if not batch:
            break
        if since_year:
            batch = [b for b in batch if _dt.datetime.fromisoformat(
                b["event_date"].replace("Z", "+00:00")).date().year >= since_year]
        rows.extend(batch)
        total = d.get("count", 0)
        offset += limit
        if offset >= total or len(batch) < limit:
            break
        _time.sleep(0.35)

    insert_new, update_existing, errors = [], 0, 0
    with session_scope():
        league = League.query.filter_by(league_type=league_type_value).first()
        if league is None:
            league = League(league_type=league_type_value, name=league_type_value)
            db.session.add(league)
            db.session.flush()
        # 索引:old 行 (home_team, away_team) → 含 ±1 天容差日期
        existing = {}
        for m in Match.query.filter_by(league_id=league.id).all():
            existing.setdefault((m.home_team, m.away_team), []).append(m)

        def _find(raw):
            hn, an = _norm(raw["home_team"]), _norm(raw["away_team"])
            d0 = _dt.datetime.fromisoformat(raw["event_date"].replace("Z", "+00:00")).date()
            for (oh, oa), ms in existing.items():
                if (_norm(oh) == hn and _norm(oa) == an) or (_norm(oh) == an and _norm(oa) == hn):
                    for m in ms:
                        if abs((m.match_date.date() - d0).days) <= 1:
                            return m
            return None

        for r in rows:
            try:
                nm = to_normalized(r, league_type_value)
                old = _find(r)
                if old is not None:
                    old.home_goals = nm.home_goals
                    old.away_goals = nm.away_goals
                    old.home_ht_goals = nm.home_ht_goals
                    old.away_ht_goals = nm.away_ht_goals
                    update_existing += 1
                else:
                    insert_new.append(nm)
            except Exception:
                errors += 1
        db.session.flush()
    res = upsert_matches(insert_new) if insert_new else {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
    if verbose:
        print(f"  {league_type_value}: 拉取 {len(rows)} | 更新 {update_existing} | "
              f"新增 {res['inserted']} | 错误 {errors}", flush=True)
    return {"fetched": len(rows), "updated": update_existing,
            "inserted": res["inserted"], "errors": errors + len(res.get("errors", []))}
