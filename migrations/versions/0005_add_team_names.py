"""add team_names table (EN <-> ZH name mapping)

Revision ID: 0005_team_names
Revises: 0004_public_id
Create Date: 2026-08-15
"""
import sqlalchemy as sa
from alembic import op

revision = "0005_team_names"
down_revision = "0004_public_id"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "team_names",
        sa.Column("en_name", sa.String(120), primary_key=True),
        sa.Column("zh_name", sa.String(60), nullable=False),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_team_names_zh", "team_names", ["zh_name"])


def downgrade():
    op.drop_index("ix_team_names_zh", table_name="team_names")
    op.drop_table("team_names")
