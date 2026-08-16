"""add feature_store table (V2: 特征注册表/版本化)

Revision ID: 0009_feature_store
Revises: 0008_experiments
Create Date: 2026-08-15

特征版本化:每次训练记录实际特征集(6 大族分类 + 集合哈希),
支持"V2.3 为什么比 V2.2 好"的特征维度归因与回滚。
"""
import sqlalchemy as sa
from alembic import op

revision = "0009_feature_store"
down_revision = "0008_experiments"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "feature_store",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("league_type", sa.String(50), nullable=False),
        sa.Column("feature_name", sa.String(80), nullable=False),
        sa.Column("family", sa.String(30), nullable=False),
        sa.Column("version", sa.String(16), nullable=False),
        sa.Column("formula_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(10), nullable=False, server_default="active"),
        sa.UniqueConstraint("league_type", "feature_name", "version",
                            name="uq_feature_version"),
    )
    op.create_index("ix_feature_league", "feature_store", ["league_type"])


def downgrade():
    op.drop_index("ix_feature_league", table_name="feature_store")
    op.drop_table("feature_store")
