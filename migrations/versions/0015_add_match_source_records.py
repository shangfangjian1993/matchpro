"""match_source_records 表

关系型 lineage 表,替代 source_scores_json JSON blob。
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0015_add_match_source_records"
down_revision = "0014_add_match_lineage"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "match_source_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False, index=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("external_id", sa.String(100), nullable=True),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column("home_ht_goals", sa.Integer(), nullable=True),
        sa.Column("away_ht_goals", sa.Integer(), nullable=True),
        sa.Column("orientation", sa.String(10), server_default="SAME"),
        sa.Column("available_at", sa.DateTime(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("hash", sa.String(64), nullable=True),
    )
    op.create_index("ix_match_source", "match_source_records", ["match_id", "source"])
    op.create_unique_constraint(
        "uq_match_source_hash",
        "match_source_records",
        ["match_id", "source", "hash"],
    )


def downgrade():
    op.drop_table("match_source_records")
