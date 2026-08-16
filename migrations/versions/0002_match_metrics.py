"""add match metric columns

Revision ID: 0002_match_metrics
Revises: 389df40ba0fc
Create Date: 2026-08-13 23:45:00.000000

指标扩展列规划(2026-08):主客分开、一场一行。
- 全 NULL 时模型特征选择自动跳过(不影响现有训练)
- init.sql 已包含这些列的新部署,此处仅对旧库执行 ADD COLUMN
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_match_metrics"
down_revision: str | None = "389df40ba0fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (列名, 类型) —— 与 api/db.py Match 模型、docker/init.sql 对齐
METRIC_COLUMNS = [
    ("home_xg", sa.Float()),
    ("away_xg", sa.Float()),
    ("home_shots", sa.Integer()),
    ("away_shots", sa.Integer()),
    ("home_shots_on_target", sa.Integer()),
    ("away_shots_on_target", sa.Integer()),
    ("home_corners", sa.Integer()),
    ("away_corners", sa.Integer()),
    ("home_possession", sa.Float()),
    ("home_yellow_cards", sa.Integer()),
    ("away_yellow_cards", sa.Integer()),
    ("home_red_cards", sa.Integer()),
    ("away_red_cards", sa.Integer()),
    ("home_ht_goals", sa.Integer()),
    ("away_ht_goals", sa.Integer()),
    ("home_pass_accuracy", sa.Float()),
    ("away_pass_accuracy", sa.Float()),
    ("home_xg_chain", sa.Float()),
    ("away_xg_chain", sa.Float()),
    ("home_efficiency", sa.Float()),
    ("away_efficiency", sa.Float()),
    ("home_transition_speed", sa.Float()),
    ("away_transition_speed", sa.Float()),
    ("home_defensive_actions", sa.Float()),
    ("away_defensive_actions", sa.Float()),
    ("home_counter_attacks", sa.Float()),
    ("away_counter_attacks", sa.Float()),
    ("home_tactical_rating", sa.Float()),
    ("away_tactical_rating", sa.Float()),
    ("home_experience", sa.Float()),
    ("away_experience", sa.Float()),
    ("match_stage", sa.String(length=20)),
]


def _add_column_if_not_exists(table: str, column: str, col_type) -> None:
    """方言兼容加列:PostgreSQL 原生 IF NOT EXISTS;其他方言(如 SQLite)捕获重复列错误"""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} "
            f"{col_type.compile(dialect=bind.dialect)}"
        )
    else:
        try:
            op.add_column(table, sa.Column(column, col_type))
        except Exception:
            pass  # 列已存在(如 init.sql 已建)


def upgrade() -> None:
    for column, col_type in METRIC_COLUMNS:
        _add_column_if_not_exists("matches", column, col_type)


def downgrade() -> None:
    bind = op.get_bind()
    for column, _ in METRIC_COLUMNS:
        try:
            if bind.dialect.name == "postgresql":
                op.execute(f"ALTER TABLE matches DROP COLUMN IF EXISTS {column}")
            else:
                op.drop_column("matches", column)
        except Exception:
            pass
