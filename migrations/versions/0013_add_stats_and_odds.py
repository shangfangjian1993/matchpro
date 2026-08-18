"""0013_add_stats_columns_and_match_odds

- team_match_stats 新增 10 个 bzzoiro 深度统计列
- 新表 match_odds(收盘赔率:1x2 + O/U 1.5/2.5/3.5 + BTTS)
"""
import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

_NEW_COLS = [
    ("fouls", sa.Integer), ("offsides", sa.Integer), ("tackles", sa.Integer),
    ("interceptions", sa.Integer), ("clearances", sa.Integer),
    ("blocked_shots", sa.Integer), ("big_chances", sa.Integer),
    ("total_saves", sa.Integer), ("shots_inside_box", sa.Integer),
    ("shots_outside_box", sa.Integer),
]


def upgrade() -> None:
    for col, typ in _NEW_COLS:
        op.add_column("team_match_stats", sa.Column(col, typ(), nullable=True))

    op.create_table(
        "match_odds",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=True),
        sa.Column("league_id", sa.Integer, nullable=True),
        sa.Column("home_team", sa.String(100), nullable=True),
        sa.Column("away_team", sa.String(100), nullable=True),
        sa.Column("event_date", sa.DateTime, nullable=True),
        sa.Column("home_win", sa.Float, nullable=True),
        sa.Column("draw", sa.Float, nullable=True),
        sa.Column("away_win", sa.Float, nullable=True),
        sa.Column("over_15", sa.Float, nullable=True),
        sa.Column("under_15", sa.Float, nullable=True),
        sa.Column("over_25", sa.Float, nullable=True),
        sa.Column("under_25", sa.Float, nullable=True),
        sa.Column("over_35", sa.Float, nullable=True),
        sa.Column("under_35", sa.Float, nullable=True),
        sa.Column("btts_yes", sa.Float, nullable=True),
        sa.Column("btts_no", sa.Float, nullable=True),
        sa.Column("updated_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("match_odds")
    for col, _ in _NEW_COLS:
        op.drop_column("team_match_stats", col)
