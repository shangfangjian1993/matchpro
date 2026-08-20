"""时间工具:统一 UTC 时间获取。

datetime.utcnow() 自 Python 3.12 起弃用(且不明确时区语义),
统一使用 utcnow():返回 naive UTC(与现有 naive UTC 数据库列保持一致,
避免改列类型引入兼容问题)。
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """当前 UTC 时间(naive,与 datetime.utcnow() 语义一致,无弃用警告)。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def as_utc_naive(value):
    """统一时间基准(审查 e752f5f P1-13/14)。

    所有 cutoff/compare 都以 **UTC naive** 为锚:world aware → 转 UTC 并去
    tzinfo;naive 视为 UTC 原样。避免 tz-naive/tz-aware 混比 TypeError 或
    cutoff 偏移(数据库 match_date 多为 naive,bzzoiro 导入时已去 tz)。
    """
    if value is None:
        return None
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
    except Exception:
        return value
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts
