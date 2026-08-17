"""SQLite → PostgreSQL 数据迁移工具(PostgreSQL+Redis 改造)。

用法:
    DATABASE_URL_SRC="sqlite:////data/matchpro/data/football.db" \
    DATABASE_URL_PG="postgresql+psycopg2://football:pass@localhost:5432/football" \
    .venv/bin/python -m app.services.data.migrate_to_pg

流程:PG create_all(BASE.metadata)→ 逐表 SELECT 源 → INSERT 目标(事务)。
SQLite 特有类型(JSON/Boolean/DateTime)由 SQLAlchemy 统一适配;
自增主键由 PG SERIAL/IDENTITY 接管。
"""
from __future__ import annotations

import os
import sys

_ROOT = str(__import__("app.core.paths", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api.db import Base


TABLES = [
    "users", "leagues", "teams", "matches", "team_match_stats",
    "predictions", "prediction_snapshots", "experiments",
    "feature_store", "injuries", "players", "settings",
]


def _url(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> int:
    src_url = _url("DATABASE_URL_SRC", "sqlite:////data/matchpro/data/football.db")
    pg_url = _url("DATABASE_URL_PG", "")
    if not pg_url:
        print("请设置 DATABASE_URL_PG(目标 PostgreSQL)")
        return 1
    src = create_engine(src_url)
    dst = create_engine(pg_url)
    print(f"源: {src_url}\n目标: {pg_url}")
    # 建表(PG)
    Base.metadata.create_all(dst)
    total = 0
    with Session(src) as s, Session(dst) as d:
        for table_name in TABLES:
            try:
                table = Base.metadata.tables.get(table_name)
            except Exception:
                table = None
            if table is None:
                print(f"  - {table_name}: 不在 metadata(跳过)")
                continue
            rows = s.execute(select(table)).all()
            if not rows:
                print(f"  - {table_name}: 0 行")
                continue
            # 清空目标表(幂等)
            d.execute(table.delete())
            for row in rows:
                d.execute(table.insert(), dict(row._mapping))
            d.commit()
            total += len(rows)
            print(f"  ✅ {table_name}: {len(rows)} 行")
    print(f"\n迁移完成: {total} 行写入 PostgreSQL")
    # 校验计数
    with Session(dst) as d:
        for table_name in ("leagues", "matches", "teams", "prediction_snapshots"):
            try:
                table = Base.metadata.tables[table_name]
                cnt = d.execute(select(table)).all()
                print(f"  目标 {table_name}: {len(cnt)} 行")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
