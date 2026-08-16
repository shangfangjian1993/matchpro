"""add team_match_stats table (V2: matches 拆分,每队每场指标)

Revision ID: 0010_team_match_stats
Revises: 0009_feature_store
Create Date: 2026-08-15

matches 胖表拆分:比赛级指标移到 team_match_stats(每队每场一行),
未来新增指标(deep_progression/xT/packing)无需改主表。
"""
import sqlalchemy as sa
from alembic import op

revision = "0010_team_match_stats"
down_revision = "0009_feature_store"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "team_match_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("side", sa.String(10), nullable=False),  # home/away
        sa.Column("xg", sa.Float(), nullable=True),
        sa.Column("shots", sa.Integer(), nullable=True),
        sa.Column("shots_on_target", sa.Integer(), nullable=True),
        sa.Column("corners", sa.Integer(), nullable=True),
        sa.Column("possession", sa.Float(), nullable=True),
        sa.Column("yellow_cards", sa.Integer(), nullable=True),
        sa.Column("red_cards", sa.Integer(), nullable=True),
        sa.Column("ht_goals", sa.Integer(), nullable=True),
        sa.Column("passing_accuracy", sa.Float(), nullable=True),
        sa.Column("xg_chain", sa.Float(), nullable=True),
        sa.Column("efficiency", sa.Float(), nullable=True),
        sa.Column("transition_speed", sa.Float(), nullable=True),
        sa.Column("defensive_actions", sa.Float(), nullable=True),
        sa.Column("counter_attacks", sa.Float(), nullable=True),
        sa.Column("tactical_rating", sa.Float(), nullable=True),
        sa.Column("experience", sa.Float(), nullable=True),
        sa.UniqueConstraint("match_id", "side", name="uq_team_match_side"),
    )
    op.create_index("ix_tms_team", "team_match_stats", ["team_id"])


def downgrade():
    op.drop_index("ix_tms_team", table_name="team_match_stats")
    op.drop_table("team_match_stats")
