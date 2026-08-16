"""Alembic 环境:读取项目 SQLAlchemy 模型元数据,URL 从 DATABASE_URL 取"""
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# 项目根(含包 multi_league_model_system)与其父目录加入导入路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PROJECT_DIR
for p in (PROJECT_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.api.db import Base as target_metadata_base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 生产/容器环境用 DATABASE_URL 覆盖 ini 默认值
database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

target_metadata = target_metadata_base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
