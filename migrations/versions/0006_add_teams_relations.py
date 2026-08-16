"""add teams / team_seasons tables; leagues.comp_type; matches team_id FKs

Revision ID: 0006_teams_relations
Revises: 0005_team_names
Create Date: 2026-08-15

实体-事件模型:
- teams:球队实体表(俱乐部/国家队统一,含 ELO 评分落点 elo_rating)
- team_seasons:球队 × 赛季(联赛归属/升降级)
- leagues 增强 comp_type(league/cup/national,统一赛事表)
- matches 增 home_team_id/away_team_id(外键,队名冗余保留兼容)
"""
import sqlalchemy as sa
from alembic import op

revision = "0006_teams_relations"
down_revision = "0005_team_names"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("name_zh", sa.String(60), nullable=True),
        sa.Column("team_type", sa.String(20), nullable=False, server_default="club"),
        sa.Column("country", sa.String(50), nullable=True),
        sa.Column("founded_year", sa.Integer(), nullable=True),
        sa.Column("stadium", sa.String(100), nullable=True),
        sa.Column("city", sa.String(50), nullable=True),
        sa.Column("elo_rating", sa.Float(), nullable=True),
        sa.Column("elo_updated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_teams_name", "teams", ["name"], unique=True)

    op.create_table(
        "team_seasons",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("league_id", sa.Integer(), sa.ForeignKey("leagues.id"), nullable=False),
        sa.Column("season", sa.String(20), nullable=False),
        sa.Column("tier", sa.Integer(), nullable=True),
        sa.Column("promoted", sa.Boolean(), nullable=True),
        sa.Column("relegated", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("team_id", "league_id", "season", name="uq_team_season"),
    )
    op.create_index("ix_team_seasons_team", "team_seasons", ["team_id"])

    # leagues 增强:comp_type(统一赛事表类型标注)
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite 不支持 ADD COLUMN 带默认?支持(带固定默认)
        op.execute("ALTER TABLE leagues ADD COLUMN comp_type VARCHAR(20) DEFAULT 'league'")
    else:
        op.add_column("leagues", sa.Column("comp_type", sa.String(20),
                                           nullable=False, server_default="league"))

    # matches 加 team_id 外键(可空,回填后启用)
    if bind.dialect.name == "sqlite":
        op.execute("ALTER TABLE matches ADD COLUMN home_team_id INTEGER")
        op.execute("ALTER TABLE matches ADD COLUMN away_team_id INTEGER")
    else:
        op.add_column("matches", sa.Column("home_team_id", sa.Integer(), nullable=True))
        op.add_column("matches", sa.Column("away_team_id", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("matches", "away_team_id")
    op.drop_column("matches", "home_team_id")
    op.drop_column("leagues", "comp_type")
    op.drop_index("ix_team_seasons_team", table_name="team_seasons")
    op.drop_table("team_seasons")
    op.drop_index("ix_teams_name", table_name="teams")
    op.drop_table("teams")
