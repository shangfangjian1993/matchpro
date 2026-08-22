#!/usr/bin/env python3
"""历史数据回填:bzzoiro stats → team_match_stats (近 20 季五大联赛)。"""
from __future__ import annotations

import json, logging, os, sys, time, sqlite3, re
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill")

PROGRESS = "/opt/data/backfill_progress.json"

def load_prog():
    try:
        with open(PROGRESS) as f: return json.load(f)
    except: return {}

def save_prog(p):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "w") as f: json.dump(p, f, indent=2)

def get_keys():
    ks = []
    e = os.environ.get("BZZOIRO_KEY", "")
    if e: ks.append(e)
    try:
        with open(".env") as f:
            for l in f:
                if l.startswith("BZZOIRO_KEY"):
                    k = l.split("=",1)[1].strip()
                    if k and k not in ks: ks.append(k)
    except: pass
    try:
        with open("/opt/data/bzzoiro_keys.txt") as f:
            for l in f:
                k = l.strip()
                if k and k not in ks: ks.append(k)
    except: pass
    return ks

def bz_get(path, params, keys, kidx):
    import urllib.request, urllib.parse, urllib.error
    url = "https://sports.bzzoiro.com/api/v2/" + path.lstrip("/")
    if params:
        url += "?" + urllib.parse.urlencode({k:v for k,v in params.items() if v is not None})
    for i in range(len(keys)):
        idx = (kidx + i) % len(keys)
        req = urllib.request.Request(url, headers={"Authorization": f"Token {keys[idx]}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r), (idx + 1) % len(keys)
        except urllib.error.HTTPError as e:
            if e.code == 429: continue
            raise
        except Exception:
            time.sleep(1)
    return {}, kidx

def norm(n):
    n = re.sub(r'\b(fc|cf|sc|afc|acf|wanderers|club)\b', ' ', str(n).lower())
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', n)).strip()

def run():
    keys = get_keys()
    if not keys:
        log.error("No BZZOIRO keys!")
        return
    
    from app.api.db import init_db, Match, TeamMatchStats, League, db, session_scope
    init_db()
    
    leagues = {"premier_league": 1, "la_liga": 3, "bundesliga": 5, "serie_a": 4, "ligue_1": 6}
    now = datetime.now(timezone.utc)
    cur_season = now.year if now.month >= 8 else now.year - 1
    seasons = list(range(cur_season, cur_season - 20, -1))
    
    prog = load_prog()
    kidx = prog.get("key_idx", 0)
    total_matched = prog.get("total_matched", 0)
    total_skipped = prog.get("total_skipped", 0)
    total_errors = prog.get("total_errors", 0)
    
    STATS_MAP = {
        "expected_goals": "xg", "total_shots": "shots", "shots_on_target": "shots_on_target",
        "ball_possession": "possession", "corner_kicks": "corners", "yellow_cards": "yellow_cards",
        "red_cards": "red_cards", "fouls": "fouls", "offsides": "offsides", "tackles": "tackles",
        "interceptions": "interceptions", "clearances": "clearances", "blocked_shots": "blocked_shots",
        "big_chances": "big_chances", "total_saves": "total_saves",
        "shots_inside_box": "shots_inside_box", "shots_outside_box": "shots_outside_box",
    }
    
    for season in seasons:
        for lid, lname in leagues.items():
            pkey = f"{lname}_{season}"
            if prog.get(pkey, {}).get("done"):
                log.info(f"SKIP {pname} {season}")
                continue
            
            log.info(f"Fetching {lname} {season}...")
            
            # fetch events
            events = []
            offset = 0
            while True:
                try:
                    data, kidx = bz_get("/events/", {"league_id": lid, "status": "finished", "limit": 100, "offset": offset}, keys, kidx)
                    batch = data.get("results", [])
                    if not batch: break
                    for e in batch:
                        dt = datetime.fromisoformat(e["event_date"].replace("Z", "+00:00"))
                        y = dt.year if dt.month >= 8 else dt.year - 1
                        if y == season:
                            events.append(e)
                    offset += 100
                    time.sleep(0.35)
                    if offset >= data.get("count", 0) or len(batch) < 100:
                        break
                except Exception as ex:
                    log.error(f"fetch error: {ex}")
                    time.sleep(5)
            
            log.info(f"  Got {len(events)} events")
            
            matched = skipped = errors = 0
            
            with session_scope():
                league = League.query.filter_by(league_type=lname).first()
                if not league:
                    continue
                
                for event in events:
                    try:
                        dt = datetime.fromisoformat(event["event_date"].replace("Z", "+00:00"))
                        dt_naive = dt.replace(tzinfo=None)
                        home_n = norm(event.get("home_team", ""))
                        away_n = norm(event.get("away_team", ""))
                        
                        match = Match.query.filter_by(league_id=league.id, match_date=dt_naive).filter(
                            Match.home_team.like(f"%{event['home_team'][:15]}%")
                        ).first()
                        
                        if not match:
                            for m in Match.query.filter_by(league_id=league.id, match_date=dt_naive).all():
                                if norm(m.home_team) == home_n and norm(m.away_team) == away_n:
                                    match = m
                                    break
                        
                        if not match:
                            skipped += 1
                            continue
                        
                        matched += 1
                        
                        stats_data, kidx = bz_get(f"/events/{event['id']}/stats/", keys, kidx)
                        st = stats_data.get("stats", {})
                        
                        for side, tid in [("home", match.home_team_id), ("away", match.away_team_id)]:
                            src = st.get(side, {})
                            row = TeamMatchStats.query.filter_by(match_id=match.id, side=side).first()
                            d = {"match_id": match.id, "team_id": tid, "side": side}
                            for ksrc, kdst in STATS_MAP.items():
                                v = src.get(ksrc)
                                if v is not None:
                                    try: d[kdst] = float(v) if isinstance(v, (int, float)) else None
                                    except: pass
                            
                            if row is None:
                                db.session.add(TeamMatchStats(**d))
                            else:
                                for k, v in d.items():
                                    setattr(row, k, v)
                        
                        time.sleep(0.25)
                    except Exception as ex:
                        errors += 1
                        log.error(f"  Error: {ex}")
                
                db.session.commit()
            
            total_matched += matched
            total_skipped += skipped
            total_errors += errors
            
            prog[pkey] = {"done": True, "matched": matched}
            prog["key_idx"] = kidx
            prog["total_matched"] = total_matched
            prog["total_skipped"] = total_skipped
            prog["total_errors"] = total_errors
            save_prog(prog)
            
            log.info(f"  {lname} {season}: matched={matched}, skipped={skipped}, errors={errors}")
    
    log.info(f"=== DONE ===\nTotal matched={total_matched}, skipped={total_skipped}, errors={total_errors}")

if __name__ == "__main__":
    run()
