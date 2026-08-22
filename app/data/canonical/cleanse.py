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
from app.data.canonical.team_names import CANONICAL_NAMES

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


# ============================================================
# 向后兼容导出(供 pipeline.py / sources/bzzoiro.py / ingest.py 使用)
# ============================================================

from app.data.pipeline.canonical.normalize import NormalizedMatch  # noqa: F401

def cleanse_fdco_row(row: dict, league_type: str, unmatched: list | None = None) -> "NormalizedMatch | None":
    """向后兼容:fdco CSV 行 → NormalizedMatch。"""
    from app.data.pipeline.canonical.normalize import _normalize_fdco
    return _normalize_fdco(row, league_type)

def cleanse_understat_row(raw: dict, league_type: str) -> "NormalizedMatch | None":
    """向后兼容:understat → NormalizedMatch。"""
    from app.data.pipeline.canonical.normalize import _normalize_understat
    return _normalize_understat(raw, league_type)

def _parse_date(value) -> "datetime | None":
    """向后兼容:任意格式 → datetime。"""
    return normalize_time(value)

def validate(nm: "NormalizedMatch") -> bool:
    """向后兼容:基本校验。"""
    if not nm or not nm.home_team or not nm.away_team:
        return False
    if nm.home_team == nm.away_team:
        return False
    return True
