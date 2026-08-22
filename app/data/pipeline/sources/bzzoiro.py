"""Bzzoiro Sports API v2 接入(新数据源,Token 鉴权)。

双 key 自动轮换(遇到 429 自动切换下一个)。

文档: https://sports.bzzoiro.com/docs/football/
Base:  https://sports.bzzoiro.com/api/v2/
Auth:  Authorization: Token ***
"""
from __future__ import annotations

import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from app.data.pipeline.config import BZZOIRO_BASE, BZZOIRO_LEAGUE_IDS, BZZOIRO_STATS_MAP
from app.data.pipeline.sources.base import BaseSource, SourceResult
from app.data.pipeline.canonical.normalize import NormalizedMatch

logger = logging.getLogger(__name__)


# 多 key 自动轮换(遇到 429 自动切换下一个)
_KEYS: list[str] | None = None
_KEY_INDEX: int = 0


def _load_keys() -> list[str]:
    """加载所有 bzzoiro API key(环境变量/.env/外部文件)。"""
    global _KEYS
    if _KEYS is not None:
        return _KEYS
    _KEYS = []
    # 1. 环境变量
    _env_key = os.environ.get("BZZOIRO_KEY", "").strip()
    if _env_key:
        _KEYS.append(_env_key)
    # 2. .env 文件
    try:
        from app.core.paths import PROJECT_ROOT
        env_path = os.path.join(str(PROJECT_ROOT), ".env")
        with open(env_path, encoding="utf-8") as _env_f:
            for line in _env_f:
                if line.startswith("BZZOIRO_KEY"):
                    k = line.split("=", 1)[1].strip()
                    if k and k not in _KEYS:
                        _KEYS.append(k)
    except Exception:
        pass
    # 3. /opt/data/bzzoiro_keys.txt
    try:
        with open("/opt/data/bzzoiro_keys.txt", encoding="utf-8") as _kf:
            for line in _kf:
                k = line.strip()
                if k and k not in _KEYS:
                    _KEYS.append(k)
    except Exception:
        pass
    return _KEYS


def _key() -> str:
    """获取当前 key。"""
    keys = _load_keys()
    if not keys:
        return ""
    return keys[_KEY_INDEX % len(keys)]


def _rotate_key() -> None:
    """遇到 429 时切换到下一个 key。"""
    global _KEY_INDEX
    _KEY_INDEX = (_KEY_INDEX + 1) % max(1, len(_load_keys()))


def available() -> bool:
    return bool(_load_keys())


def _get(path: str, params: dict | None = None) -> dict:
    """GET 请求(自动 key 轮换+重试)。"""
    import json

    keys = _load_keys()
    if not keys:
        raise RuntimeError("无可用 BZZOIRO_KEY")

    base = BZZOIRO_BASE.replace("_", "")
    url = base + path.lstrip("/")
    if params:
        url += "?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}
        )

    # 尝试所有 key,遇到 429 自动轮换
    for _key_attempt in range(len(keys)):
        current_key = _key()
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Token {current_key}", "Accept": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=25) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    _rotate_key()
                    break  # 换下一个 key
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
            except Exception:
                time.sleep(1.0)
    raise RuntimeError("bzzoiro 所有 key 均耗尽或请求失败")


def fetch_events(
    league_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """历史/未来事件(分页)。"""
    d = _get(
        "/events/",
        {"league_id": league_id, "status": status, "limit": limit, "offset": offset},
    )
    return d.get("results") or []


def fetch_odds(
    event_id: int | None = None,
    market: str = "1x2",
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """盘口(默认 consensus 1x2)。"""
    d = _get(
        "/odds/",
        {"event_id": event_id, "market": market, "limit": limit, "offset": offset},
    )
    return d.get("results") or []


def find_league(name: str) -> int | None:
    """按名称(子串,不敏感)找 league_id(真正分页到 total)。"""
    off, limit = 0, 50
    while True:
        d = _get("/leagues/", {"limit": limit, "offset": off})
        batch = d.get("results") or []
        for lg in batch:
            if name.lower() in (lg.get("name") or "").lower():
                return lg.get("id")
        total = d.get("count", 0)
        off += limit
        if off >= total or len(batch) < limit:
            break
    return None


def _season_label(date: datetime) -> str:
    y = date.year if date.month >= 8 else date.year - 1
    return f"{y}-{y + 1}"


def _norm(name: str) -> str:
    """队名归一(与 api-football 一致风格,用于 bzzoiro↔fdco 对齐)。"""
    n = str(name).lower()
    n = re.sub(r"\b(fc|cf|sc|afc|acf|wanderers|club)\b", " ", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def to_normalized(raw: dict, league_type_value: str) -> NormalizedMatch:
    """bzzoiro event → NormalizedMatch(upsert 幂等)。"""
    date = datetime.fromisoformat(raw["event_date"].replace("Z", "+00:00"))
    date = date.replace(tzinfo=None)
    return NormalizedMatch(
        league_type=league_type_value,
        date=date,
        home_team=str(raw["home_team"]).strip(),
        away_team=str(raw["away_team"]).strip(),
        home_goals=int(raw["home_score"])
        if raw.get("home_score") is not None
        else None,
        away_goals=int(raw["away_score"])
        if raw.get("away_score") is not None
        else None,
        home_ht_goals=int(raw["home_score_ht"])
        if raw.get("home_score_ht") is not None
        else None,
        away_ht_goals=int(raw["away_score_ht"])
        if raw.get("away_score_ht") is not None
        else None,
        match_status="finished",
        season_label=_season_label(date),
        match_stage=str(raw.get("round_name") or raw.get("stage") or ""),
    )


class BzzoiroSource(BaseSource):
    """Bzzoiro 数据源适配器。

    支持数据类型: results(赛果), stats(深度统计), odds(收盘赔率)。
    """

    SOURCE_NAME = "bzzoiro_events"

    def __init__(
        self,
        league_type: str,
        data_type: str = "results",
        http_client=None,
        cache=None,
    ) -> None:
        super().__init__(league_type, http_client, cache)
        self.data_type = data_type
        if data_type == "stats":
            self.SOURCE_NAME = "bzzoiro_stats"
        elif data_type == "odds":
            self.SOURCE_NAME = "bzzoiro_odds"
        self._league_id: int | None = None

    @property
    def league_id(self) -> int | None:
        if self._league_id is None:
            self._league_id = BZZOIRO_LEAGUE_IDS.get(self.league_type)
        return self._league_id

    def fetch_raw(self, **kwargs) -> list[dict]:
        """抓取 bzzoiro 原始事件数据。"""
        if self.league_id is None:
            raise ValueError(f"bzzoiro 未映射: {self.league_type}")

        status = kwargs.get("status", "finished")
        since_year = kwargs.get("since_year")
        limit = kwargs.get("limit", 100)

        rows: list[dict] = []
        offset = 0
        while True:
            d = _get(
                "/events/",
                {
                    "league_id": self.league_id,
                    "status": status,
                    "limit": limit,
                    "offset": offset,
                },
            )
            batch = d.get("results") or []
            if not batch:
                break
            if since_year:
                batch = [
                    b
                    for b in batch
                    if datetime.fromisoformat(b["event_date"].replace("Z", "+00:00"))
                    .date()
                    .year
                    >= since_year
                ]
            rows.extend(batch)
            total = d.get("count", 0)
            offset += limit
            if offset >= total or len(batch) < limit:
                break
            time.sleep(0.35)  # 免费套餐稳健
        return rows

    def normalize_row(self, raw: dict) -> NormalizedMatch | None:
        """bzzoiro 原始事件 → NormalizedMatch。"""
        return to_normalized(raw, self.league_type)

    def fetch(self, **kwargs) -> SourceResult:
        """bzzoiro 专用 fetch(含 events/stats/odds 三种模式)。"""
        if self.data_type == "stats":
            return self._fetch_stats(**kwargs)
        if self.data_type == "odds":
            return self._fetch_odds(**kwargs)
        return super().fetch(**kwargs)

    def _fetch_stats(self, **kwargs) -> SourceResult:
        """抓取 stats(逐场)。"""
        from app.api.db import TeamMatchStats, db, session_scope
        from app.data.canonical.resolver import CanonicalMatchResolver

        result = SourceResult(source="bzzoiro_stats", data_type="stats")

        # 构建 event match index
        league_id = self.league_id
        if league_id is None:
            result.add_error(f"未映射联赛: {self.league_type}")
            return result

        events = self.fetch_raw(status="finished", limit=kwargs.get("limit_events"))
        result.raw_count = len(events)

        with session_scope():
            from app.api.db import League, Match

            league = League.query.filter_by(league_type=self.league_type).first()
            if league is None:
                result.add_error(f"数据库无联赛: {self.league_type}")
                return result

            _resolver = CanonicalMatchResolver().index_matches(
                Match.query.filter_by(league_id=league.id).all()
            )

            # 构建索引: (date, home_norm, away_norm) → Match
            idx: dict[tuple, list] = {}
            for m in Match.query.filter_by(league_id=league.id).all():
                hn, an = _norm(m.home_team), _norm(m.away_team)
                key = (str(m.match_date.date()), hn, an)
                idx.setdefault(key, []).append(m)
                idx.setdefault((str(m.match_date.date()), an, hn), []).append(m)

            matched = updated = errors = 0
            for e in events:
                try:
                    d0 = datetime.fromisoformat(
                        e["event_date"].replace("Z", "+00:00")
                    ).date()
                    m = None
                    for cand in idx.get(
                        (str(d0), _norm(e["home_team"]), _norm(e["away_team"])), []
                    ):
                        m = cand
                        break
                    if m is None:
                        continue
                    matched += 1
                    st = _get(f"/events/{e['id']}/stats/").get("stats") or {}
                    for side, mid, tid in (
                        ("home", m.id, m.home_team_id),
                        ("away", m.id, m.away_team_id),
                    ):
                        src = st.get(side) or {}
                        row = TeamMatchStats.query.filter_by(
                            match_id=mid, side=side
                        ).first()
                        data = {"match_id": mid, "team_id": tid, "side": side}
                        for k_src, k_dst in BZZOIRO_STATS_MAP.items():
                            v = src.get(k_src)
                            if v is not None:
                                data[k_dst] = (
                                    int(v)
                                    if isinstance(v, (int, float))
                                    and not isinstance(v, bool)
                                    and float(v).is_integer()
                                    else (
                                        float(v)
                                        if isinstance(v, (int, float))
                                        else None
                                    )
                                )
                        if row is None:
                            db.session.add(TeamMatchStats(**data))
                        else:
                            for k, v in data.items():
                                setattr(row, k, v)
                    updated += 1
                except Exception:
                    errors += 1
                time.sleep(0.25)
            db.session.commit()

        result.records = []  # stats 不入 matches 表,直接写 team_match_stats
        return result

    def _fetch_odds(self, **kwargs) -> SourceResult:
        """抓取 odds(逐场)。"""
        from app.api.db import MatchOdds, db, session_scope

        result = SourceResult(source="bzzoiro_odds", data_type="odds")

        events = self.fetch_raw(status="finished", limit=kwargs.get("limit_events"))
        result.raw_count = len(events)

        with session_scope():
            from app.api.db import League, Match

            league = League.query.filter_by(league_type=self.league_type).first()
            if league is None:
                result.add_error(f"数据库无联赛: {self.league_type}")
                return result

            # 构建索引
            idx: dict[tuple, list] = {}
            for m in Match.query.filter_by(league_id=league.id).all():
                hn, an = _norm(m.home_team), _norm(m.away_team)
                key = (str(m.match_date.date()), hn, an)
                idx.setdefault(key, []).append(m)
                idx.setdefault((str(m.match_date.date()), an, hn), []).append(m)

            matched = written = errors = 0
            for e in events:
                try:
                    d0 = datetime.fromisoformat(
                        e["event_date"].replace("Z", "+00:00")
                    ).date()
                    m = None
                    for cand in idx.get(
                        (str(d0), _norm(e["home_team"]), _norm(e["away_team"])), []
                    ):
                        m = cand
                        break
                    if m is None:
                        continue
                    matched += 1
                    od = _get(f"/events/{e['id']}/odds/").get("odds") or {}
                    row = MatchOdds.query.filter_by(match_id=m.id).first()
                    data = {
                        "match_id": m.id,
                        "league_id": m.league_id,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "event_date": m.match_date,
                        "home_win": od.get("home_win"),
                        "draw": od.get("draw"),
                        "away_win": od.get("away_win"),
                        "over_15": od.get("over_15_goals"),
                        "under_15": od.get("under_15_goals"),
                        "over_25": od.get("over_25_goals"),
                        "under_25": od.get("under_25_goals"),
                        "over_35": od.get("over_35_goals"),
                        "under_35": od.get("under_35_goals"),
                        "btts_yes": od.get("btts_yes"),
                        "btts_no": od.get("btts_no"),
                        "updated_at": datetime.now(tz=timezone.utc),
                    }
                    if row is None:
                        db.session.add(MatchOdds(**data))
                    else:
                        for k, v in data.items():
                            setattr(row, k, v)
                    written += 1
                except Exception:
                    errors += 1
                time.sleep(0.25)
            db.session.commit()

        result.records = []
        return result


# ---- 兼容旧接口 ----

def import_league(
    league_type_value: str,
    since_year: int | None = None,
    verbose: bool = True,
) -> dict:
    """全量导入某联赛(翻页,upsert 幂等)。"""
    source = BzzoiroSource(league_type_value, data_type="results")
    result = source.fetch(since_year=since_year, status="finished")
    if result.records:
        ingest_result = source.ingest(result.records)
    else:
        ingest_result = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
    ingest_result["fetched"] = result.raw_count
    if verbose:
        print(
            f"  {league_type_value}: 拉取 {result.raw_count} 场 → "
            f"新增 {ingest_result['inserted']} 更新 {ingest_result['updated']} "
            f"跳过 {ingest_result['skipped']} 错误 {len(ingest_result['errors'])}",
            flush=True,
        )
    return ingest_result


def import_recent(
    league_type_value: str,
    seasons: int = 1,
    verbose: bool = True,
) -> dict:
    """近 N 季增量 merge(轻量,供 daily 采集)。"""
    from app.api.db import League, Match, db, session_scope
    from app.data.canonical.reconcile import maybe_update
    from app.data.canonical.lineage import record_source
    from app.data.canonical.resolver import CanonicalMatchResolver

    league_id = BZZOIRO_LEAGUE_IDS.get(league_type_value)
    if league_id is None:
        raise ValueError(f"bzzoiro 未映射: {league_type_value}")

    d0 = _get("/events/", {"league_id": league_id, "status": "finished", "limit": 1})
    latest = datetime.fromisoformat(
        d0["results"][0]["event_date"].replace("Z", "+00:00")
    )
    last_season = latest.year if latest.month >= 8 else latest.year - 1
    cutoff = datetime(last_season - seasons + 1, 8, 1)

    rows: list[dict] = []
    ofs = 0
    while True:
        batch = fetch_events(
            league_id=league_id, status="finished", limit=200, offset=ofs
        )
        if not batch:
            break
        kept = [
            e
            for e in batch
            if datetime.fromisoformat(
                e["event_date"].replace("Z", "+00:00")
            ).replace(tzinfo=None)
            >= cutoff
        ]
        if kept:
            rows.extend(kept)
        ofs += 200
        time.sleep(0.25)
        if not kept:
            continue

    with session_scope():
        league = League.query.filter_by(league_type=league_type_value).first()
        _resolver = (
            CanonicalMatchResolver().index_matches(
                Match.query.filter_by(league_id=league.id).all()
            )
            if league is not None
            else None
        )

        def _find(nm):
            if _resolver is None:
                return None, "SAME"
            _r = _resolver.resolve(nm.home_team, nm.away_team, nm.date)
            return _r.match, _r.orientation

        inserted_list: list[NormalizedMatch] = []
        updated, errors = 0, 0
        for e in rows:
            try:
                nm = to_normalized(e, league_type_value)
                old, orientation = _find(nm)
                if old is not None:
                    maybe_update(old, nm, "bzzoiro_events", orientation=orientation)
                    record_source(
                        match_id=old.id,
                        source="bzzoiro_events",
                        home_goals=nm.home_goals,
                        away_goals=nm.away_goals,
                        home_ht_goals=nm.home_ht_goals,
                        away_ht_goals=nm.away_ht_goals,
                        orientation=orientation,
                    )
                    updated += 1
                else:
                    inserted_list.append(nm)
            except Exception:
                errors += 1
        db.session.flush()

    from app.data.pipeline.ingest import upsert_matches

    r = (
        upsert_matches(inserted_list, source="bzzoiro_events")
        if inserted_list
        else {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
    )
    if verbose:
        print(
            f"  {league_type_value} 近{seasons}季: 拉取{len(rows)} 更新{updated} 新增{r['inserted']} 错误{errors}",
            flush=True,
        )
    return {
        "fetched": len(rows),
        "updated": updated,
        "inserted": r["inserted"],
        "errors": errors,
    }


def ingest_stats(
    league_type_value: str,
    limit_events: int | None = None,
    offset: int = 0,
    verbose: bool = True,
) -> dict:
    """逐场拉 stats → 写 team_match_stats。"""
    source = BzzoiroSource(league_type_value, data_type="stats")
    result = source._fetch_stats(limit_events=limit_events)
    return {
        "events": result.raw_count,
        "matched": result.raw_count,
        "updated": result.raw_count,
        "errors": len(result.errors),
    }


def ingest_odds(
    league_type_value: str,
    limit_events: int | None = None,
    offset: int = 0,
    verbose: bool = True,
) -> dict:
    """逐场拉收盘 odds → match_odds。"""
    source = BzzoiroSource(league_type_value, data_type="odds")
    result = source._fetch_odds(limit_events=limit_events)
    return {
        "events": result.raw_count,
        "matched": result.raw_count,
        "written": result.raw_count,
        "errors": len(result.errors),
    }

