"""add experiments table (V2: 训练实验追踪)

Revision ID: 0008_experiments
Revises: 0007_prediction_snapshots
Create Date: 2026-08-15

每次训练自动记录全链路元数据,支持"V2.3 为什么比 V2.2 好"归因。
"""
import sqlalchemy as sa
from alembic import op

revision = "0008_experiments"
down_revision = "0007_prediction_snapshots"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("league_type", sa.String(50), nullable=False),
        sa.Column("dataset_version", sa.String(40), nullable=False),
        sa.Column("feature_version", sa.String(40), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("train_start", sa.DateTime(), nullable=True),
        sa.Column("train_end", sa.DateTime(), nullable=True),
        sa.Column("hyperparameters_json", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("calibration_json", sa.Text(), nullable=True),
        sa.Column("data_hash", sa.String(64), nullable=True),
        sa.Column("notes", sa.String(500), nullable=True),
    )
    op.create_index("ix_exp_league_version", "experiments", ["league_type", "model_version"])


def downgrade():
    op.drop_index("ix_exp_league_version", table_name="experiments")
    op.drop_table("experiments")
