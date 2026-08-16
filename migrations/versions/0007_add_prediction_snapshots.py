"""add prediction_snapshots table (V2: 预测快照/回放)

Revision ID: 0007_prediction_snapshots
Revises: 0006_teams_relations
Create Date: 2026-08-15

V2 核心:任何预测可 100% 重放——
- snapshot_json:全部输入(特征/ELO/伤停/参数)
- probabilities_json:胜平负/比分矩阵/Top5
- data_hash/model_hash/sha256:三哈希可复现
- 赛后回填 actual_* / is_correct → Replay 评估
"""
import sqlalchemy as sa
from alembic import op

revision = "0007_prediction_snapshots"
down_revision = "0006_teams_relations"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "prediction_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("league_id", sa.Integer(), sa.ForeignKey("leagues.id"), nullable=False),
        sa.Column("home_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("away_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=True),
        sa.Column("home_team", sa.String(120), nullable=False),
        sa.Column("away_team", sa.String(120), nullable=False),
        sa.Column("kickoff", sa.DateTime(), nullable=True),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("feature_version", sa.String(40), nullable=False),
        sa.Column("data_hash", sa.String(64), nullable=False),
        sa.Column("model_hash", sa.String(64), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("snapshot_json", sa.Text(), nullable=False),
        sa.Column("probabilities_json", sa.Text(), nullable=False),
        sa.Column("actual_home_goals", sa.Integer(), nullable=True),
        sa.Column("actual_away_goals", sa.Integer(), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("log_loss", sa.Float(), nullable=True),
        sa.UniqueConstraint("league_id", "home_team", "away_team", "kickoff",
                            name="uq_snapshot_match"),
    )
    op.create_index("ix_snapshot_kickoff", "prediction_snapshots", ["kickoff"])


def downgrade():
    op.drop_index("ix_snapshot_kickoff", table_name="prediction_snapshots")
    op.drop_table("prediction_snapshots")
