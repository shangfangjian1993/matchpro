"""add public_id to externally exposed resources

Revision ID: 0004_public_id
Revises: 0003_unify_metric_columns
Create Date: 2026-08-14
"""
import uuid

import sqlalchemy as sa
from alembic import op

revision = "0004_public_id"
down_revision = "0003_unify_metric_columns"
branch_labels = None
depends_on = None

TABLES = ("predictions", "models", "notifications", "training_tasks")


def upgrade():
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"
    for t in TABLES:
        op.add_column(t, sa.Column("public_id", sa.String(36), nullable=True))
    # 回填 UUID(SQLite/Postgres 通用:逐表逐行更新)
    for t in TABLES:
        conn = op.get_bind()
        ids = conn.execute(sa.text(f"SELECT id FROM {t}")).fetchall()
        for (rid,) in ids:
            conn.execute(
                sa.text(f"UPDATE {t} SET public_id = :pid WHERE id = :id"),
                {"pid": str(uuid.uuid4()), "id": rid},
            )
        if is_sqlite:
            # SQLite 不支持 ALTER COLUMN SET NOT NULL / 加约束;非空由应用层 default 保证
            continue
        op.alter_column(t, "public_id", nullable=False)
        op.create_unique_constraint(f"uq_{t}_public_id", t, ["public_id"])


def downgrade():
    for t in TABLES:
        op.drop_constraint(f"uq_{t}_public_id", t, type_="unique")
        op.drop_column(t, "public_id")
