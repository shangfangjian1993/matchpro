#!/usr/bin/env python3
"""回填 api-football passing_accuracy。"""
from __future__ import annotations

import json, logging, os, sys, time, urllib.request, urllib.error, sqlite3, re
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api_fb")

def norm(n):
    n = re.sub(r'\b(fc|cf|sc|afc|acf|wanderers|club|cfc|united|city|hotspur|albion|rovers|villa|borough|county|athletic|forest|palace|ham|wolves|stoke|wednesday|brighton|bournemouth|brentford|fulham|newcastle|west|nottingham|anderlecht|gent|brugge|standard|antwerp|charleroi|genk|cercle)\b', '', str(n).lower())
    return re.sub(r'[^a-z]', '', n)

def _key():
    k = os.environ.get("API_FOOTBALL_KEY", "")
    if k: return k
    try:
        with open(".env") as f:
            for l in f:
                if l.startswith("API_FOOTBALL_KEY"):
                    return l.split("=",1)[1].strip()
    except: pass
    return ""

def api_get(path, params=None):
    k = _key()
    if not k: raise RuntimeError("No API_FOOTBALL_KEY")
    url = f"https://v3.football.api-sports.io{path}"
    if params:
        import urllib.parse
        url += "?" + urllib.parse.urlencode({k:v for k,v in params.items() if v is not None})
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"x-apisports-key": k})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10)
                continue
            raise
        except Exception:
            time.sleep(2)
    return None

def run():
    if not _key():
        log.error("API_FOOTBALL_KEY not set")
        return
    
    conn = sqlite3.connect('data/football.db')
    c = conn.cursor()
    
    leagues = {"premier_league": 39, "la_liga": 140, "bundesliga": 78, "serie_a": 135, "ligue_1": 61}
    seasons = list(range(2014, 2026))
    
    total_updated = 0
    total_errors = 0
    
    for season in seasons:
        for league_name, league_id in leagues.items():
            log.info(f"Processing {league_name}/{season}...")
            
            # 获取赛季所有比赛
            try:
                data = api_get("/fixtures", {"league": league_id, "season": season})
                if not data:
                    continue
                fixtures = data.get("response", [])
            except Exception as e:
                log.error(f"  Fetch error: {e}")
                total_errors += 1
                continue
            
            log.info(f"  Got {len(fixtures)} fixtures")
            
            for fixture in fixtures:
                try:
                    fid = fixture.get("fixture", {}).get("id")
                    if not fid:
                        continue
                    
                    # 获取 stats
                    stat_data = api_get(f"/fixtures/statistics", {"fixture": fid})
                    if not stat_data:
                        continue
                    
                    teams_stats = stat_data.get("response", [])
                    if len(teams_stats) < 2:
                        continue
                    
                    home_stats = teams_stats[0].get("statistics", [])
                    away_stats = teams_stats[1].get("statistics", [])
                    
                    passing_home = None
                    passing_away = None
                    
                    for s in home_stats:
                        if s.get("type") == "Passes %":
                            passing_home = s.get("value")
                            break
                    
                    for s in away_stats:
                        if s.get("type") == "Passes %":
                            passing_away = s.get("value")
                            break
                    
                    if passing_home is None and passing_away is None:
                        continue
                    
                    # 匹配到 matches 表
                    fixture_date = fixture.get("fixture", {}).get("date", "")[:10]
                    home_team = fixture.get("teams", {}).get("home", {}).get("name", "")
                    away_team = fixture.get("teams", {}).get("away", {}).get("name", "")
                    
                    c.execute("SELECT id FROM matches WHERE match_date LIKE ? AND season_id = ?", (f"{fixture_date}%", season))
                    candidates = c.fetchall()
                    
                    for (cid,) in candidates:
                        c2 = conn.cursor()
                        c2.execute("SELECT home_team, away_team FROM matches WHERE id = ?", (cid,))
                        row = c2.fetchone()
                        if row:
                            home_n = norm(row[0])
                            away_n = norm(row[1])
                            home_fb_n = norm(home_team)
                            away_fb_n = norm(away_team)
                            if (home_fb_n[:4] in home_n or home_n[:4] in home_fb_n) and len(home_fb_n) >= 3:
                                c.execute("UPDATE matches SET home_passing_accuracy = ?, away_passing_accuracy = ? WHERE id = ?",
                                    (passing_home, passing_away, cid))
                                total_updated += c.rowcount
                                break
                    
                    time.sleep(0.5)  # api-football free 100/day
                except Exception as e:
                    total_errors += 1
                    log.error(f"  Error: {e}")
            
            conn.commit()
    
    conn.close()
    log.info(f"=== DONE ===\nUpdated: {total_updated}, Errors: {total_errors}")

if __name__ == "__main__":
    run()
