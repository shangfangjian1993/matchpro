"""时间工具:统一 UTC 时间获取。

datetime.utcnow() 自 Python 3.12 起弃用(且不明确时区语义),
统一使用 utcnow():返回 naive UTC(与现有 naive UTC 数据库列保持一致,
避免改列类型引入兼容问题)。
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """当前 UTC 时间(naive,与 datetime.utcnow() 语义一致,无弃用警告)。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)
