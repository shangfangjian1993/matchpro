"""add teams attack_elo / defense_elo (V2: ELO 多维化)

Revision ID: 0011_elo_multi_dim
Revises: 0010_team_match_stats
Create Date: 2026-08-15

多维 ELO:Overall(结果)+ Attack(进球对决)+ Defense(失球对决),
区分"进攻强但防守一般"与"防守稳但进攻弱"的球队。
"""
import sqlalchemy as sa
from alembic import op

revision = "0011_elo_multi_dim"
down_revision = "0010_team_match_stats"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        op.execute("ALTER TABLE teams ADD COLUMN attack_elo DOUBLE PRECISION")
        op.execute("ALTER TABLE teams ADD COLUMN defense_elo DOUBLE PRECISION")
    else:
        op.add_column("teams", sa.Column("attack_elo", sa.Float(), nullable=True))
        op.add_column("teams", sa.Column("defense_elo", sa.Float(), nullable=True))


def downgrade():
    op.drop_column("teams", "defense_elo")
    op.drop_column("teams", "attack_elo")
