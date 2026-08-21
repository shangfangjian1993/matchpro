"""队名中英文映射服务:team_names 表(DB) + 内存缓存 + 内置种子兜底。

数据流:
 team_names 表(可运行时管理)→ 首次访问加载到内存缓存 → to_zh/to_en 查询
 DB 不可用/未 seed 时回退内置种子(_BUILTIN,210 条,离线可用)

用法:
 from app.data.canonical.team_names_zh import to_zh, to_en, invalidate, seed_from_builtin
 to_zh("Arsenal FC") # → 阿森纳
 to_en("阿森纳") # → Arsenal FC
"""

from __future__ import annotations

import threading

# ---- 内置种子(启动兜底 + seed 数据源;与 team_names 表保持一致)----
_BUILTIN: dict[str, str] = {
 "AFC Bournemouth": "伯恩茅斯",
 "Arsenal FC": "阿森纳",
 "Aston Villa FC": "阿斯顿维拉",
 "Brentford FC": "布伦特福德",
 "Brighton & Hove Albion FC": "布莱顿",
 "Brighton and Hove Albion": "布莱顿",
 "Burnley FC": "伯恩利",
 "Cardiff City FC": "加的夫城",
 "Chelsea FC": "切尔西",
 "Coventry City FC": "考文垂",
 "Crystal Palace FC": "水晶宫",
 "Everton FC": "埃弗顿",
 "Fulham FC": "富勒姆",
 "Huddersfield Town FC": "哈德斯菲尔德",
 "Hull City AFC": "赫尔城",
 "Hull City": "赫尔城",
 "Ipswich Town FC": "伊普斯维奇",
 "Leeds United FC": "利兹联",
 "Leicester City FC": "莱斯特城",
 "Liverpool FC": "利物浦",
 "Luton Town FC": "卢顿",
 "Manchester City FC": "曼城",
 "Manchester United FC": "曼联",
 "Middlesbrough FC": "米德尔斯堡",
 "Newcastle United FC": "纽卡斯尔联",
 "Norwich City FC": "诺维奇",
 "Nottingham Forest FC": "诺丁汉森林",
 "Queens Park Rangers FC": "女王公园巡游者",
 "Reading FC": "雷丁",
 "Sheffield United FC": "谢菲尔德联",
 "Southampton FC": "南安普顿",
 "Stoke City FC": "斯托克城",
 "Sunderland AFC": "桑德兰",
 "Swansea City FC": "斯旺西",
 "Tottenham Hotspur FC": "托特纳姆热刺",
 "Watford FC": "沃特福德",
 "West Bromwich Albion FC": "西布罗姆维奇",
 "West Ham United FC": "西汉姆联",
 "Wigan Athletic FC": "维冈竞技",
 "Wolverhampton Wanderers FC": "狼队",
 "Athletic Club": "毕尔巴鄂竞技",
 "CA Osasuna": "奥萨苏纳",
 "CD Leganés": "莱加内斯",
 "Club Atlético de Madrid": "马德里竞技",
 "Cádiz CF": "加的斯",
 "Deportivo Alavés": "阿拉维斯",
 "Deportivo Alavés FC": "阿拉维斯",
 "Deportivo La Coruña": "拉科鲁尼亚",
 "RC Deportivo La Coruña": "拉科鲁尼亚",
 "Elche CF": "埃尔切",
 "FC Barcelona": "巴塞罗那",
 "Getafe CF": "赫塔菲",
 "Girona FC": "赫罗纳",
 "Granada CF": "格拉纳达",
 "Levante UD": "莱万特",
 "Málaga CF": "马拉加",
 "RC Celta de Vigo": "塞尔塔",
 "RCD Espanyol de Barcelona": "西班牙人",
 "RCD Mallorca": "马略卡",
 "Rayo Vallecano de Madrid": "巴列卡诺",
 "Real Betis Balompié": "皇家贝蒂斯",
 "Real Madrid CF": "皇家马德里",
 "Real Oviedo": "皇家奥维耶多",
 "Real Racing Club de Santander": "桑坦德竞技",
 "Real Sociedad de Fútbol": "皇家社会",
 "Real Valladolid CF": "巴拉多利德",
 "Real Zaragoza": "萨拉戈萨",
 "SD Eibar": "埃瓦尔",
 "SD Huesca": "韦斯卡",
 "Sevilla FC": "塞维利亚",
 "Sporting Gijón": "希洪竞技",
 "UD Almería": "阿尔梅里亚",
 "UD Las Palmas": "拉斯帕尔马斯",
 "Valencia CF": "瓦伦西亚",
 "Villarreal CF": "比利亚雷亚尔",
 "1. FC Heidenheim 1846": "海登海姆",
 "1. FC Köln": "科隆",
 "1. FC Nürnberg": "纽伦堡",
 "1. FC Union Berlin": "柏林联合",
 "1. FSV Mainz 05": "美因茨",
 "Arminia Bielefeld": "比勒费尔德",
 "Bayer 04 Leverkusen": "勒沃库森",
 "Bayer Leverkusen": "勒沃库森",
 "Bayern München": "拜仁慕尼黑",
 "FC Bayern München": "拜仁慕尼黑",
 "Bayern Munich": "拜仁慕尼黑",
 "Bochum": "波鸿",
 "VfL Bochum 1848": "波鸿",
 "Borussia Dortmund": "多特蒙德",
 "Borussia M.Gladbach": "门兴格拉德巴赫",
 "Darmstadt": "达姆施塔特",
 "SV Darmstadt 98": "达姆施塔特",
 "Eintracht Braunschweig": "布伦瑞克",
 "Eintracht Frankfurt": "法兰克福",
 "FC Augsburg": "奥格斯堡",
 "FC Heidenheim": "海登海姆",
 "FC Ingolstadt 04": "因戈尔施塔特",
 "FC Köln": "科隆",
 "FC Schalke 04": "沙尔克04",
 "FC St. Pauli 1910": "圣保利",
 "Fortuna Düsseldorf": "杜塞尔多夫",
 "Freiburg": "弗赖堡",
 "SC Freiburg": "弗赖堡",
 "Greuther Fürth": "菲尔特",
 "SpVgg Greuther Fürth": "菲尔特",
 "Hamburger SV": "汉堡",
 "Hannover 96": "汉诺威96",
 "Hertha Berlin": "柏林赫塔",
 "Hoffenheim": "霍芬海姆",
 "TSG 1899 Hoffenheim": "霍芬海姆",
 "Holstein Kiel": "基尔",
 "Ingolstadt": "因戈尔施塔特",
 "Mainz 05": "美因茨",
 "Paderborn": "帕德博恩",
 "SC Paderborn 07": "帕德博恩",
 "RB Leipzig": "莱比锡红牛",
 "Schalke 04": "沙尔克04",
 "Stuttgart": "斯图加特",
 "VfB Stuttgart": "斯图加特",
 "SV 07 Elversberg": "埃尔沃斯堡",
 "Union Berlin": "柏林联合",
 "Werder Bremen": "云达不莱梅",
 "SV Werder Bremen": "云达不莱梅",
 "Wolfsburg": "沃尔夫斯堡",
 "VfL Wolfsburg": "沃尔夫斯堡",
 "AC Monza": "蒙扎",
 "AC Milan": "AC米兰",
 "Atalanta BC": "亚特兰大",
 "Benevento Calcio": "贝内文托",
 "Bologna FC 1909": "博洛尼亚",
 "Brescia Calcio": "布雷西亚",
 "Cagliari Calcio": "卡利亚里",
 "Carpi FC 1909": "卡尔皮",
 "Catania Calcio": "卡塔尼亚",
 "Cesena FC": "切塞纳",
 "Chievo Verona": "切沃",
 "Como 1907": "科莫",
 "Cremonese": "克雷莫纳",
 "US Cremonese": "克雷莫纳",
 "Crotone": "克罗托内",
 "FC Crotone": "克罗托内",
 "Empoli FC": "恩波利",
 "FC Internazionale Milano": "国际米兰",
 "Fiorentina": "佛罗伦萨",
 "ACF Fiorentina": "佛罗伦萨",
 "Frosinone Calcio": "弗罗西诺内",
 "Genoa CFC": "热那亚",
 "Hellas Verona FC": "维罗纳",
 "Inter": "国际米兰",
 "Juventus FC": "尤文图斯",
 "Lazio": "拉齐奥",
 "SS Lazio": "拉齐奥",
 "Lecce": "莱切",
 "US Lecce": "莱切",
 "Livorno Calcio": "利沃诺",
 "Monza": "蒙扎",
 "Napoli": "那不勒斯",
 "SSC Napoli": "那不勒斯",
 "Palermo FC": "巴勒莫",
 "Parma Calcio 1913": "帕尔马",
 "Pescara Calcio": "佩斯卡拉",
 "Pisa": "比萨",
 "AC Pisa 1909": "比萨",
 "Roma": "罗马",
 "AS Roma": "罗马",
 "Salernitana": "萨勒尼塔纳",
 "US Salernitana 1919": "萨勒尼塔纳",
 "Sampdoria": "桑普多利亚",
 "UC Sampdoria": "桑普多利亚",
 "Sassuolo": "萨索洛",
 "US Sassuolo Calcio": "萨索洛",
 "Siena Calcio": "锡耶纳",
 "SPAL": "斯帕尔",
 "Spezia Calcio": "斯佩齐亚",
 "Torino FC": "都灵",
 "Udinese Calcio": "乌迪内斯",
 "Venezia FC": "威尼斯",
 "Verona": "维罗纳",
 "AC Ajaccio": "阿雅克肖",
 "AJ Auxerre": "欧塞尔",
 "Amiens SC": "亚眠",
 "Angers SCO": "昂热",
 "AS Monaco FC": "摩纳哥",
 "AS Nancy Lorraine": "南锡",
 "AS Saint-Étienne": "圣埃蒂安",
 "Clermont Foot 63": "克莱蒙",
 "Dijon FCO": "第戎",
 "EA Guingamp": "甘冈",
 "ES Troyes AC": "特鲁瓦",
 "Evian Thonon Gaillard FC": "伊维恩",
 "FC Girondins de Bordeaux": "波尔多",
 "FC Lorient": "洛里昂",
 "FC Metz": "梅斯",
 "FC Nantes": "南特",
 "FC Sochaux-Montbéliard": "索肖",
 "GFC Ajaccio": "阿雅克肖GFCO",
 "Le Havre AC": "勒阿弗尔",
 "Le Mans FC": "勒芒",
 "Lille OSC": "里尔",
 "Montpellier HSC": "蒙彼利埃",
 "Nîmes Olympique": "尼姆",
 "OGC Nice": "尼斯",
 "Olympique Lyonnais": "里昂",
 "Olympique de Marseille": "马赛",
 "Paris Saint-Germain FC": "巴黎圣日耳曼",
 "RC Strasbourg Alsace": "斯特拉斯堡",
 "Racing Club de Lens": "朗斯",
 "Reims": "兰斯",
 "Stade de Reims": "兰斯",
 "Rennes": "雷恩",
 "Stade Rennais FC 1901": "雷恩",
 "SC Bastia": "巴斯蒂亚",
 "SM Caen": "卡昂",
 "Stade Brestois 29": "布雷斯特",
 "Toulouse FC": "图卢兹",
 "Valenciennes FC": "瓦朗谢讷",
 "AC Sparta Praha": "布拉格斯巴达",
 "AFC Ajax": "阿贾克斯",
 "BSC Young Boys": "伯尔尼年轻人",
 "Celtic FC": "凯尔特人",
 "Club Brugge KV": "布鲁日",
 "FC Porto": "波尔图",
 "FC Salzburg": "萨尔茨堡红牛",
 "Fenerbahce": "费内巴切",
 "Galatasaray": "加拉塔萨雷",
 "Olympiacos Piraeus": "奥林匹亚科斯",
 "PSV Eindhoven": "埃因霍温",
 "Red Bull Salzburg": "萨尔茨堡红牛",
 "Shakhtar Donetsk": "顿涅茨克矿工",
 "Slavia Praha": "布拉格斯拉维亚",
 "Sporting CP": "葡萄牙体育",
 "Sporting Lisbon": "葡萄牙体育",
 "Benfica": "本菲卡",
 "SL Benfica": "本菲卡",
 "FC Copenhagen": "哥本哈根",
 "Dinamo Zagreb": "萨格勒布迪纳摩",
 "FC Basel": "巴塞尔",
 "BSC Young Boys Bern": "伯尔尼年轻人",
 "Rangers FC": "格拉斯哥流浪者",
 "Steaua Bucuresti": "布加勒斯特星",
 "Feyenoord": "费耶诺德",
 "AZ Alkmaar": "阿尔克马尔",
 "Ludogorets Razgrad": "卢多戈雷茨",
 "Legia Warsaw": "华沙莱吉亚",
 # ---- 国家队(世界杯/欧洲杯)----
 "Albania": "阿尔巴尼亚",
 "Algeria": "阿尔及利亚",
 "Argentina": "阿根廷",
 "Australia": "澳大利亚",
 "Austria": "奥地利",
 "Belgium": "比利时",
 "Bosnia and Herzegovina": "波黑",
 "Brazil": "巴西",
 "Bulgaria": "保加利亚",
 "Cameroon": "喀麦隆",
 "Canada": "加拿大",
 "Chile": "智利",
 "Colombia": "哥伦比亚",
 "Costa Rica": "哥斯达黎加",
 "Croatia": "克罗地亚",
 "Czechia": "捷克",
 "Denmark": "丹麦",
 "Ecuador": "厄瓜多尔",
 "Egypt": "埃及",
 "England": "英格兰",
 "France": "法国",
 "Georgia": "格鲁吉亚",
 "Germany": "德国",
 "Ghana": "加纳",
 "Greece": "希腊",
 "Hungary": "匈牙利",
 "Iceland": "冰岛",
 "Iran": "伊朗",
 "Italy": "意大利",
 "Japan": "日本",
 "Mexico": "墨西哥",
 "Morocco": "摩洛哥",
 "Netherlands": "荷兰",
 "Nigeria": "尼日利亚",
 "Norway": "挪威",
 "Poland": "波兰",
 "Portugal": "葡萄牙",
 "Romania": "罗马尼亚",
 "Saudi Arabia": "沙特阿拉伯",
 "Scotland": "苏格兰",
 "Senegal": "塞内加尔",
 "Serbia": "塞尔维亚",
 "Slovakia": "斯洛伐克",
 "Slovenia": "斯洛文尼亚",
 "South Africa": "南非",
 "South Korea": "韩国",
 "Spain": "西班牙",
 "Sweden": "瑞典",
 "Switzerland": "瑞士",
 "Tunisia": "突尼斯",
 "Turkey": "土耳其",
 "Ukraine": "乌克兰",
 "United States": "美国",
 "Uruguay": "乌拉圭",
 "Wales": "威尔士",
 "Borussia Mönchengladbach": "门兴格拉德巴赫",
 "Córdoba CF": "科尔多瓦",
 "FC København": "哥本哈根",
 "FC Red Bull Salzburg": "萨尔茨堡红牛",
 "FK Bodø/Glimt": "博德闪耀",
 "FK Crvena Zvezda": "贝尔格莱德红星",
 "FK Kairat": "卡伊拉特",
 "FK Shakhtar Donetsk": "顿涅茨克矿工",
 "Feyenoord Rotterdam": "费耶诺德",
 "GNK Dinamo Zagreb": "萨格勒布迪纳摩",
 "Galatasaray SK": "加拉塔萨雷",
 "PAE Olympiakos SFP": "奥林匹亚科斯",
 "PSV": "埃因霍温",
 "Paphos FC": "帕福斯",
 "Paris FC": "巴黎FC",
 "Qarabağ Ağdam FK": "卡拉巴赫",
 "Queens Park Rangers": "女王公园巡游者",
 "Royal Antwerp FC": "安特卫普",
 "Royale Union Saint-Gilloise": "圣吉罗斯联合",
 "SK Slavia Praha": "布拉格斯拉维亚",
 "SK Sturm Graz": "格拉茨风暴",
 "Sport Lisboa e Benfica": "本菲卡",
 "Sporting Clube de Braga": "布拉加",
 "Sporting Clube de Portugal": "葡萄牙体育",
 "ŠK Slovan Bratislava": "布拉迪斯拉发",
}

_BUILTIN_REV: dict[str, str] = {}
for _en, _zh in _BUILTIN.items():
 _BUILTIN_REV.setdefault(_zh, _en)

# ---- DB 缓存(首次访问加载;invalidate() 失效)----
_db_fwd: dict[str, str] | None = None # en → zh
_db_rev: dict[str, str] | None = None # zh → en
_cache_lock = threading.Lock()


def _load_from_db() -> tuple[dict[str, str], dict[str, str]] | None:
 """从 team_names 表加载全量映射;无表/无 app context/异常返回 None(回退内置)。"""
 try:
 from app.api.db import TeamName, db

 rows = db.session.query(TeamName).all()
 if not rows:
 return None
 fwd = {r.en_name: r.zh_name for r in rows}
 rev: dict[str, str] = {}
 for en, zh in fwd.items():
 rev.setdefault(zh, en)
 return fwd, rev
 except Exception:
 return None


def _cache() -> tuple[dict[str, str], dict[str, str]]:
 """获取 (正向, 反向) 缓存;未加载时尝试从 DB 加载(失败回退空,查询层再回退内置)。"""
 global _db_fwd, _db_rev
 with _cache_lock:
 if _db_fwd is None:
 loaded = _load_from_db()
 if loaded:
 _db_fwd, _db_rev = loaded
 else:
 _db_fwd, _db_rev = {}, {}
 return _db_fwd, _db_rev


def invalidate() -> None:
 """清空 DB 缓存(team_names 表更新后调用,下次查询重新加载)。"""
 global _db_fwd, _db_rev
 with _cache_lock:
 _db_fwd = _db_rev = None


def to_zh(name: str) -> str:
 """英文规范名 → 中文显示名(DB 优先,内置兜底,无映射回退原名)。"""
 if not name:
 return name
 fwd, _ = _cache()
 if name in fwd:
 return fwd[name]
 return _BUILTIN.get(name, name)


def to_en(name: str) -> str:
 """中文(或英文) → 英文规范名;已是英文原样返回。"""
 if not name:
 return name
 if name in _BUILTIN:
 return name
 _, rev = _cache()
 if name in rev:
 return rev[name]
 return _BUILTIN_REV.get(name.strip(), name)


def seed_from_builtin() -> int:
 """把内置种子写入 team_names 表(幂等 upsert);返回写入条数。"""
 from app.api.db import TeamName, db

 n = 0
 for en, zh in _BUILTIN.items():
 row = db.session.get(TeamName, en)
 if row is None:
 db.session.add(TeamName(en_name=en, zh_name=zh, source="builtin"))
 n += 1
 elif row.zh_name != zh:
 row.zh_name = zh
 n += 1
 db.session.commit()
 invalidate()
 return n


if __name__ == "__main__":
 # 自测(无 DB 时验证内置映射)
 tests = [
 ("Arsenal FC", "阿森纳"),
 ("FC Internazionale Milano", "国际米兰"),
 ("Sporting Clube de Portugal", "葡萄牙体育"),
 ]
 for en, zh in tests:
 assert to_zh(en) == zh, f"{en} -> {to_zh(en)}"
 assert to_en("阿森纳") == "Arsenal FC"
 print(f"✅ 内置映射自测通过({len(_BUILTIN)} 条)")
