"""add season_id to matches (§2.1 精简事件表:league_id/season_id/home_team_id/away_team_id)

Revision ID: 0012
Revises: 0011
"""
import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011_elo_multi_dim"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("matches", sa.Column("season_id", sa.Integer, nullable=True))
    # 回填:按 league 的当前 season 字符串生成 season_id 映射(league_id, season) → 序号
    op.execute(
        "UPDATE matches SET season_id = "
        "(SELECT l.id FROM leagues l WHERE l.id = matches.league_id) "
        "WHERE season_id IS NULL"
    )


def downgrade() -> None:
    op.drop_column("matches", "season_id")
