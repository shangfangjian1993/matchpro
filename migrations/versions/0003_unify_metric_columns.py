"""unify metric column names

Revision ID: 0003_unify_metric_columns
Revises: 0002_match_metrics
Create Date: 2026-08-14 00:05:00.000000

统一指标列命名:pass_accuracy -> passing_accuracy(所有联赛同一 schema,
各联赛模型通过 metric_columns 配置按需引用;未引用联赛列保持 NULL)。
已应用 0002 的旧库执行 rename;init.sql 新部署直接建 passing_accuracy,
rename 前检查列存在性,自动跳过。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_unify_metric_columns"
down_revision: str | None = "0002_match_metrics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RENAME_PAIRS = [
    ("home_pass_accuracy", "home_passing_accuracy"),
    ("away_pass_accuracy", "away_passing_accuracy"),
]


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        row = bind.execute(
            sa.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": column},
        ).first()
        return bool(row)
    rows = bind.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(r[1] == column for r in rows)


def upgrade() -> None:
    for old_name, new_name in RENAME_PAIRS:
        if _column_exists("matches", old_name):
            op.execute(
                f"ALTER TABLE matches RENAME COLUMN {old_name} TO {new_name}"
            )


def downgrade() -> None:
    for old_name, new_name in RENAME_PAIRS:
        if _column_exists("matches", new_name):
            op.execute(
                f"ALTER TABLE matches RENAME COLUMN {new_name} TO {old_name}"
            )
