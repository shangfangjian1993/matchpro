"""统一数据清洗入库:时间标准化 + 队名归一化 + 指标映射。

设计原则:
1. 时间:全部转为 UTC naive datetime(无时区),统一格式
2. 队名:全部转为规范名(小写、去后缀、去重音)
3. 指标:统一字段名(通过 SOURCE_FIELD_MAPS 映射)
4. 幂等:重复运行安全(已存在的不覆盖)
5. 可追踪:记录每个字段的来源(source)
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.data.canonical.config import SOURCE_FIELD_MAPS, LEAGUES

# ============================================================
# 时间标准化
# ============================================================

def normalize_time(value: Any) -> Optional[datetime]:
    """任意时间格式 → UTC naive datetime。
    
    支持: ISO 字符串、DD/MM/YYYY、DD/MM/YY、Unix 时间戳、datetime 对象。
    返回: UTC naive datetime(无 tzinfo)。
    """
    if value is None or value == "":
        return None
    
    if isinstance(datetime, type(value)):
        # datetime 对象
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    
    if isinstance(value, (int, float)):
        # Unix 时间戳
        return datetime.utcfromtimestamp(value)
    
    s = str(value).strip()
    if not s:
        return None
    
    # ISO 格式(含 Z)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    
    # 尝试 ISO 格式
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass
    
    # 常见格式
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d.%m.%Y",
        "%d-%m-%Y",
    ]:
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    
    return None


# ============================================================
# 队名归一化
# ============================================================

# 队名后缀/前缀(移除后为核心名)
NAME_SUFFIXES = [
    r"\bFC\b", r"\bCF\b", r"\bCFA\b", r"\bAFC\b", r"\bSC\b",
    r"\bUnited\b", r"\bCity\b", r"\bTown\b", r"\bRovers\b",
    r"\bWanderers\b", r"\bAlbion\b", r"\bVilla\b", r"\bForest\b",
    r"\bCounty\b", r"\bBorough\b", r"\bAthletic\b", r"\bHotspur\b",
    r"\bPalace\b", r"\bHam\b", r"\bSpurs\b",
]

# 核心队名映射(小写无后缀 → 规范名)
CANONICAL_NAMES: Dict[str, str] = {
    # 英超
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "brighton and hove albion": "Brighton and Hove Albion",
    "brighton": "Brighton and Hove Albion",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "leeds united": "Leeds United",
    "leeds": "Leeds United",
    "leicester city": "Leicester City",
    "leicester": "Leicester City",
    "liverpool": "Liverpool",
    "manchester city": "Manchester City",
    "mancity": "Manchester City",
    "manchester united": "Manchester United",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "newcastle united": "Newcastle United",
    "newcastle": "Newcastle United",
    "nottingham forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "southampton": "Southampton",
    "tottenham hotspur": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "west ham united": "West Ham United",
    "west ham": "West Ham United",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "afc bournemouth": "AFC Bournemouth",
    "bournemouth": "AFC Bournemouth",
    "brentford": "Brentford",
    "ipswich town": "Ipswich Town",
    "ipswich": "Ipswich Town",
    "luton town": "Luton Town",
    "luton": "Luton Town",
    "norwich city": "Norwich City",
    "norwich": "Norwich City",
    "huddersfield town": "Huddersfield Town",
    "huddersfield": "Huddersfield Town",
    "swansea city": "Swansea City",
    "swansea": "Swansea City",
    "cardiff city": "Cardiff City",
    "cardiff": "Cardiff City",
    "stoke city": "Stoke City",
    "stoke": "Stoke City",
    "hull city": "Hull City",
    "hull": "Hull City",
    "west bromwich albion": "West Bromwich Albion",
    "west brom": "West Bromwich Albion",
    "sheffield united": "Sheffield United",
    "sheffield": "Sheffield United",
    "middlesbrough": "Middlesbrough",
    "sunderland": "Sunderland",
    
    # 西甲
    "real madrid": "Real Madrid",
    "barcelona": "Barcelona",
    "atlético madrid": "Atlético Madrid",
    "atletico madrid": "Atlético Madrid",
    "athletic club": "Athletic Club",
    "ath bilbao": "Athletic Club",
    "athletic bilbao": "Athletic Club",
    "real sociedad": "Real Sociedad",
    "sociedad": "Real Sociedad",
    "real betis": "Real Betis",
    "betis": "Real Betis",
    "villarreal": "Villarreal",
    "sevilla": "Sevilla",
    "valencia": "Valencia",
    "getafe": "Getafe",
    "osasuna": "Osasuna",
    "celta vigo": "Celta Vigo",
    "celta": "Celta Vigo",
    "girona": "Girona",
    "mallorca": "Mallorca",
    "las palmas": "Las Palmas",
    "rayo vallecano": "Rayo Vallecano",
    "deportivo alavés": "Deportivo Alavés",
    "alavés": "Deportivo Alavés",
    "alaves": "Deportivo Alavés",
    "leganés": "Leganés",
    "leganes": "Leganés",
    "espanyol": "Espanyol",
    "real oviedo": "Real Oviedo",
    "oviedo": "Real Oviedo",
    "sd huesca": "SD Huesca",
    "huesca": "SD Huesca",
    "deportivo la coruña": "Deportivo La Coruña",
    "la coruña": "Deportivo La Coruña",
    "la coruna": "Deportivo La Coruña",
    
    # 德甲
    "bayern münchen": "Bayern München",
    "bayern munich": "Bayern München",
    "bayern": "Bayern München",
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "rb leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    "borussia mönchengladbach": "Borussia Mönchengladbach",
    "mönchengladbach": "Borussia Mönchengladbach",
    "gladbach": "Borussia Mönchengladbach",
    "borussia m'gladbach": "Borussia Mönchengladbach",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    "vfb stuttgart": "VfB Stuttgart",
    "stuttgart": "VfB stuttgart",
    "vfl wolfsburg": "VfL Wolfsburg",
    "wolfsburg": "VfL Wolfsburg",
    "werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",
    "tsg hoffenheim": "TSG Hoffenheim",
    "hoffenheim": "TSG Hoffenheim",
    "tsg 1899 hoffenheim": "TSG Hoffenheim",
    "sc freiburg": "SC Freiburg",
    "freiburg": "SC Freiburg",
    "fc köln": "1. FC Köln",
    "fc koln": "1. FC Köln",
    "köln": "1. FC Köln",
    "1. fc köln": "1. FC Köln",
    "1 fc köln": "1. FC Köln",
    "fsv mainz 05": "1. FSV Mainz 05",
    "mainz 05": "1. FSV Mainz 05",
    "mainz": "1. FSV Mainz 05",
    "1. fsv mainz 05": "1. FSV Mainz 05",
    "fc augsburg": "FC Augsburg",
    "augsburg": "FC Augsburg",
    "vfl bochum": "VfL Bochum",
    "bochum": "VfL Bochum",
    "fc st. pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "holstein kiel": "Holstein Kiel",
    "kiel": "Holstein Kiel",
    "1. fc union berlin": "1. FC Union Berlin",
    "union berlin": "1. FC Union Berlin",
    "1. fc heidenheim": "1. FC Heidenheim 1846",
    "fc heidenheim": "1. FC Heidenheim 1846",
    "heidenheim": "1. FC Heidenheim 1846",
    "fortuna düsseldorf": "Fortuna Düsseldorf",
    "düsseldorf": "Fortuna Düsseldorf",
    "greuther fürth": "SpVgg Greuther Fürth",
    "fürth": "SpVgg Greuther Fürth",
    "hannover 96": "Hannover 96",
    "hertha berlin": "Hertha Berlin",
    "hertha bsc": "Hertha Berlin",
    "fc ingolstadt 04": "FC Ingolstadt 04",
    "ingolstadt": "FC Ingolstadt 04",
    "sc paderborn 07": "SC Paderborn 07",
    "paderborn": "SC Paderborn 07",
    "spvgg greuther fürth": "SpVgg Greuther Fürth",
    "sc paderborn": "SC Paderborn 07",
    
    # 意甲
    "ac milan": "AC Milan",
    "milan": "AC milan",
    "inter": "Inter",
    "internazionale": "Inter",
    "fc internazionale milano": "Inter",
    "juventus": "Juventus",
    "juve": "Juventus",
    "napoli": "Napoli",
    "ssc napoli": "Napoli",
    "lazio": "Lazio",
    "ss lazio": "Lazio",
    "as roma": "AS Roma",
    "roma": "AS Roma",
    "atalanta": "Atalanta",
    "acf fiorentina": "ACF Fiorentina",
    "fiorentina": "ACF Fiorentina",
    "bologna": "Bologna",
    "torino": "Torino",
    "genoa": "Genoa",
    "genoa cfc": "Genoa",
    "cagliari": "Cagliari",
    "udinese": "Udinese",
    "sassuolo": "Sassuolo",
    "verona": "Verona",
    "hellas verona": "Verona",
    "parma": "Parma",
    "parma calcio 1913": "Parma",
    "lecce": "Lecce",
    "us lecce": "Lecce",
    "como": "Como",
    "empoli": "Empoli",
    "monza": "Monza",
    "ac monza": "Monza",
    "frosinone": "Frosinone",
    "frosinone calcio": "Frosinone",
    "salernitana": "Salernitana",
    "us salernitana 1919": "Salernitana",
    "spezia": "Spezia",
    "crotone": "Crotone",
    "benevento": "Benevento",
    "reggina": "Reggina",
    "pescara": "Pescara",
    "padova": "Padova",
    "palermo": "Palermo",
    "catania": "Catania",
    "bari": "Bari",
    "brescia": "Brescia",
    "vicenza": "Vicenza",
    "piacenza": "Piacenza",
    "treviso": "Treviso",
    "novara": "Novara",
    "reggiana": "Reggiana",
    "cremonese": "Cremonese",
    "sudtirol": "Südtirol",
    "pisa": "Pisa",
    "pisa sc": "Pisa",
    "modena": "Modena",
    "como 1907": "Como",
    
    # 法甲
    "paris saint-germain": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",
    "marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "lyon": "Lyon",
    "olympique lyonnais": "Lyon",
    "monaco": "Monaco",
    "as monaco fc": "Monaco",
    "lille": "Lille",
    "lille osc": "Lille",
    "nice": "Nice",
    "ogc nice": "Nice",
    "rennes": "Rennes",
    "stade rennais fc": "Rennes",
    "stade rennais": "Rennes",
    "strasbourg": "Strasbourg",
    "rc strasbourg alsace": "Strasbourg",
    "lens": "Lens",
    "racing club de lens": "Lens",
    "nantes": "Nantes",
    "fc nantes": "Nantes",
    "toulouse": "Toulouse",
    "toulouse fc": "Toulouse",
    "brest": "Brest",
    "stade brestois 29": "Brest",
    "reims": "Reims",
    "stade de reims": "Reims",
    "montpellier": "Montpellier",
    "montpellier hsc": "Montpellier",
    "auxerre": "Auxerre",
    "AJ auxerre": "Auxerre",
    "angers": "Angers",
    "angers sco": "Angers",
    "le havre": "Le Havre",
    "le havre ac": "Le Havre",
    "saint-étienne": "AS Saint-Étienne",
    "as saint-étienne": "AS Saint-Étienne",
    "saint etienne": "AS Saint-Étienne",
    "nîmes": "Nîmes Olympique",
    "nimes": "Nîmes Olympique",
    "nîmes olympique": "Nîmes Olympique",
    "guingamp": "EA Guingamp",
    "ea guingamp": "EA Guingamp",
    "lorient": "FC Lorient",
    "fc lorient": "FC Lorient",
    "stade de reims": "Stade de Reims",
    "stade malherbe caen": "Stade Malherbe Caen",
    "caen": "Stade Malherbe Caen",
    "dijon": "Dijon FCO",
    "dijon fco": "Dijon FCO",
    "bordeaux": "FC Girondins de Bordeaux",
    "fc girondins de bordeaux": "FC Girondins de Bordeaux",
    "troyes": "ES Troyes AC",
    "es troyes ac": "ES Troyes AC",
    "clermont": "Clermont Foot",
    "clermont foot": "Clermont Foot",
    "metz": "FC Metz",
    "fc metz": "FC Metz",
    "amiens": "Amiens SC",
    "amiens sc": "Amiens SC",
    "bastia": "SC Bastia",
    "sc bastia": "SC Bastia",
    
    # 欧战
    "fc bayern münchen": "Bayern München",
    "borussia mönchengladbach": "Borussia Mönchengladbach",
    "borussia dortmund": "Borussia Dortmund",
    "vfl wolfsburg": "VfL Wolfsburg",
    "bayer 04 leverkusen": "Bayer Leverkusen",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "1. fsv mainz 05": "1. FSV Mainz 05",
    "tsg 1899 hoffenheim": "TSG Hoffenheim",
    "sc freiburg": "SC Freiburg",
    "vfb stuttgart": "VfB Stuttgart",
    "rb leipzig": "RB Leipzig",
    "1. fc union berlin": "1. FC Union Berlin",
    "fc augsburg": "FC Augsburg",
    "1. fc köln": "1. FC Köln",
    "vfl bochum": "VfL Bochum",
    "fc st. pauli": "FC St. Pauli",
    "holstein kiel": "Holstein Kiel",
    "1. fc heidenheim": "1. FC Heidenheim 1846",
    "fortuna düsseldorf": "Fortuna Düsseldorf",
    "hannover 96": "Hannover 96",
    "hertha bsc": "Hertha Berlin",
    "fc ingolstadt 04": "FC Ingolstadt 04",
    "sc paderborn 07": "SC Paderborn 07",
    "spvgg greuther fürth": "SpVgg Greuther Fürth",
    "karlsruher sc": "Karlsruher SC",
    "sc paderborn": "SC Paderborn 07",
    "1. fc kaiserslautern": "1. FC Kaiserslautern",
    "1. fc nürnberg": "1. FC Nürnberg",
    "fc schalke 04": "FC Schalke 04",
    "hamburger sv": "Hamburger SV",
    "1. fc magdeburg": "1. FC Magdeburg",
    "sv darmstadt 98": "SV Darmstadt 98",
    "sv sandhausen": "SV Sandhausen",
    "vfl osnabrück": "VfL Osnabrück",
    "evz": "EV Zug",  # placeholder
    "cska moscow": "CSKA Moscow",
    "dinamo moscow": "Dinamo Moscow",
    "spartak moscow": "Spartak Moscow",
    "lokomotiv moscow": "Lokomotiv Moscow",
    "zenit saint petersburg": "Zenit Saint Petersburg",
    "fc krasnodar": "FC Krasnodar",
    "fc rostov": "FC Rostov",
    "fc sochi": "FC Sochi",
    "fc akhmat grozny": "FC Akhmat Grozny",
    "fc ural": "FC Ural",
    "fc tambov": "FC Tambov",
    "fc orenburg": "FC Orenburg",
    "fc khimki": "FC Khimki",
    "fc nizhny novgorod": "FC Nizhny Novgorod",
    "fc ufa": "FC Ufa",
    "fc arsenal tula": "FC Arsenal Tula",
    "fc krylya sovetov samara": "FC Krylya Sovetov Samara",
    "fc volga nizhny novgorod": "FC Volga Nizhny Novgorod",
    "fc amkar perm": "FC Amkar Perm",
    "fc mordovia saransk": "FC Mordovia Saransk",
    "fc tom tomsk": "FC Tom Tomsk",
    "fc anzhi makhachkala": "FC Anzhi Makhachkala",
    "fc baltika kaliningrad": "FC Baltika Kaliningrad",
    "fc fakel voronezh": "FC Fakel Voronezh",
    "fc torpedo moscow": "FC Torpedo Moscow",
    "fc shinnik yaroslavl": "FC Shinnik Yaroslavl",
    "fc sibir novosibirsk": "FC Sibir Novosibirsk",
    "fc ska-khabarovsk": "FC SKA-Khabarovsk",
    "fc yenisey krasnoyarsk": "FC Yenisey Krasnoyarsk",
    "fc tyumen": "FC Tyumen",
    "fc okean nakhodka": "FC Okean Nakhodka",
    "fc luch vladivostok": "FC Luch Vladivostok",
    "fc chernomorets novorossiysk": "FC Chernomorets Novorossiysk",
    "fc metallurg lipetsk": "FC Metallurg Lipetsk",
    "fc saturn ramenskoye": "FC Saturn Ramenskoye",
    "fc uralan elista": "FC Uralan Elista",
    "fc chkalovets novosibirsk": "FC Chkalovets Novosibirsk",
    "fc lokomotiv nizhny novgorod": "FC Lokomotiv Nizhny Novgorod",
    "fc zhemchuzhina sochi": "FC Zhemchuzhina Sochi",
    "fc kamaz naberezhnye chelny": "FC Kamaz Naberezhnye Chelny",
    "fc neftekhimik nizhnekamsk": "FC Neftekhimik Nizhnekamsk",
    "fc gazovik orenburg": "FC Gazovik Orenburg",
    "fc metallurg-kuzbass kemerovo": "FC Metallurg-Kuzbass Kemerovo",
    "fc tolzatti": "FC Tolzatti",
    "fc volgar astrakhan": "FC Volgar Astrakhan",
    "fc dynamo bryansk": "FC Dynamo Bryansk",
    "fc avangard kursk": "FC Avangard Kursk",
    "fc sokol saratov": "FC Sokol Saratov",
    "fc tekstilshchik ivanovo": "FC Tekstilshchik Ivanovo",
    "fc spartak vladikavkaz": "FC Spartak Vladikavkaz",
    "fc lokomotiv chita": "FC Lokomotiv Chita",
    "fc dynamo stavropol": "FC Dynamo Stavropol",
    "fc fakel moscow": "FC Fakel Moscow",
    "fc zelenograd": "FC Zelenograd",
    "fc podolsk": "FC Podolsk",
    "fc izhevsk": "FC Izhevsk",
    "fc angarsk": "FC Angarsk",
    "fc budyonnovsk": "FC Budyonnovsk",
    "fc krasnoyarsk": "FC Krasnoyarsk",
    "fc pyatigorsk": "FC Pyatigorsk",
    "fc vladikavkaz": "FC Vladikavkaz",
    "fc tver": "FC Tver",
    "fc kolomna": "FC Kolomna",
    "fc nalchik": "FC Nalchik",
    "fc saransk": "FC Saransk",
    "fc ulyanovsk": "FC Ulyanovsk",
    "fc chelyabinsk": "FC Chelyabinsk",
    "fc voronezh": "FC Voronezh",
    "fc vladimir": "FC Vladimir",
    "fc novgorod": "FC Novgorod",
    "fc togliatti": "FC Togliatti",
    "fc volzhsky": "FC Volzhsky",
    "fc kostroma": "FC Kostroma",
    "fc ryazan": "FC Ryazan",
    "fc smolensk": "FC Smolensk",
    "fc ivanovo": "FC Ivanovo",
    "fc kalpakovo": "FC Kalpakovo",
    "fc serpukhov": "FC Serpukhov",
    "fc orel": "FC Orel",
    "fc tambov": "FC Tambov",
    "fc velikiye luki": "FC Velikiye Luki",
    "fc novotroitsk": "FC Novotroitsk",
    "fc syktyvkar": "FC Syktyvkar",
    "fc yoshkar-ola": "FC Yoshkar-Ola",
    "fc saratov": "FC Saratov",
    "fc penza": "FC Penza",
    "fc kuzbass": "FC Kuzbass",
    "fc zvezda": "FC Zvezda",
    "fc avangard": "FC Avangard",
    "fc ural": "FC Ural",
    "fc chernomorets": "FC Chernomorets",
    "fc vityaz": "FC Vityaz",
    "fc metallurg": "FC Metallurg",
    "fc torpedo": "FC Torpedo",
    "fc dynamo": "FC Dynamo",
    "fc zenit": "FC Zenit",
    "fc lokomotiv": "FC Lokomotiv",
    "fc rostov": "FC Rostov",
    "fc sochi": "FC Sochi",
    "fc akhmat": "FC Akhmat",
    "fc ufa": "FC Ufa",
    "fc khimki": "FC Khimki",
    "fc nizhny": "FC Nizhny",
    "fc arsenal": "FC Arsenal",
    "fc krylya": "FC Krylya",
    "fc volga": "FC Volga",
    "fc amkar": "FC Amkar",
    "fc mordovia": "FC Mordovia",
    "fc tom": "FC Tom",
    "fc anzhi": "FC Anzhi",
    "fc baltika": "FC Baltika",
    "fc fakel": "FC Fakel",
    "fc shinnik": "FC Shinnik",
    "fc sibir": "FC Sibir",
    "fc ska": "FC SKA",
    "fc yenisey": "FC Yenisey",
    "fc tyumen": "FC Tyumen",
    "fc okean": "FC Okean",
    "fc luch": "FC Luch",
    "fc chernomorets": "FC Chernomorets",
    "fc metallurg": "FC Metallurg",
    "fc saturn": "FC Saturn",
    "fc uralan": "FC Uralan",
    "fc chkalovets": "FC Chkalovets",
    "fc lokomotiv": "FC Lokomotiv",
    "fc zhemchuzhina": "FC Zhemchuzhina",
    "fc kamaz": "FC Kamaz",
    "fc neftekhimik": "FC Neftekhimik",
    "fc gazovik": "FC Gazovik",
    "fc metallurg-kuzbass": "FC Metallurg-Kuzbass",
    "fc volgar": "FC Volgar",
    "fc dynamo": "FC Dynamo",
    "fc avangard": "FC Avangard",
    "fc sokol": "FC Sokol",
    "fc tekstilshchik": "FC Tekstilshchik",
    "fc spartak": "FC Spartak",
    "fc lokomotiv": "FC Lokomotiv",
    "fc dynamo": "FC Dynamo",
    "fc fakel": "FC Fakel",
    "fc zelenograd": "FC Zelenograd",
    "fc podolsk": "FC Podolsk",
    "fc izhevsk": "FC Izhevsk",
    "fc angarsk": "FC Angarsk",
    "fc budyonnovsk": "FC Budyonnovsk",
    "fc pyatigorsk": "FC Pyatigorsk",
    "fc vladikavkaz": "FC Vladikavkaz",
    "fc tver": "FC Tver",
    "fc kolomna": "FC Kolomna",
    "fc nalchik": "FC Nalchik",
    "fc saransk": "FC Saransk",
    "fc ulyanovsk": "FC Ulyanovsk",
    "fc chelyabinsk": "FC Chelyabinsk",
    "fc voronezh": "FC Voronezh",
    "fc vladimir": "FC Vladimir",
    "fc novgorod": "FC Novgorod",
    "fc togliatti": "FC Togliatti",
    "fc volzhsky": "FC Volzhsky",
    "fc kostroma": "FC Kostroma",
    "fc ryazan": "FC Ryazan",
    "fc smolensk": "FC Smolensk",
    "fc ivanovo": "FC Ivanovo",
    "fc kalpakovo": "FC Kalpakovo",
    "fc serpukhov": "FC Serpukhov",
    "fc orel": "FC Orel",
    "fc velikiye luki": "FC Velikiye Luki",
    "fc novotroitsk": "FC Novotroitsk",
    "fc syktyvkar": "FC Syktyvkar",
    "fc yoshkar-ola": "FC Yoshkar-Ola",
    "fc saratov": "FC Saratov",
    "fc penza": "FC Penza",
    "fc kuzbass": "FC Kuzbass",
    "fc zvezda": "FC Zvezda",
}

def normalize_team_name(name: str) -> str:
    """队名归一化:移除后缀、去重音、小写、查找规范名。"""
    if not name:
        return ""
    
    # 去重音
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ASCII", "ignore").decode("ASCII")
    
    # 移除后缀
    cleaned = ascii_name.strip()
    for suffix in NAME_SUFFIXES:
        cleaned = re.sub(suffix, "", cleaned, flags=re.IGNORECASE).strip()
    
    # 小写
    lower = cleaned.lower()
    
    # 查找规范名
    if lower in CANONICAL_NAMES:
        return CANONICAL_NAMES[lower]
    
    # 尝试模糊匹配
    for key, canonical in CANONICAL_NAMES.items():
        if key in lower or lower in key:
            return canonical
    
    # 返回原值(首字母大写)
    return ascii_name.strip().title()


# ============================================================
# 指标映射
# ============================================================

def map_fields(source: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """将源字段映射到统一字段。"""
    mapping = SOURCE_FIELD_MAPS.get(source, {})
    result = {}
    for src_field, unified_field in mapping.items():
        if src_field in raw and raw[src_field] is not None:
            result[unified_field] = raw[src_field]
    return result


# ============================================================
# 赛季推导
# ============================================================

def derive_season(dt: datetime) -> int:
    """从 datetime 推导赛季起始年(8月为界)。"""
    if dt is None:
        return None
    return dt.year if dt.month >= 8 else dt.year - 1


def derive_season_label(dt: datetime) -> str:
    """从 datetime 推导赛季标签(如 '2024-2025')。"""
    season = derive_season(dt)
    return f"{season}-{season + 1}" if season else ""


# ============================================================
# 类型转换
# ============================================================

def to_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, str) and v.strip() in ("", "-")):
        return None
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def to_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, str) and v.strip() in ("", "-")):
        return None
    try:
        return float(str(v).strip().replace("%", ""))
    except (TypeError, ValueError):
        return None
