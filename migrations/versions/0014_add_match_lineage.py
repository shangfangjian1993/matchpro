"""0014_match_lineage

- matches 新增 Source → Canonical 谱系列(审查 A70A601 §15-17):
  source/common/json 快照 + 对账状态,消除"新来源静默覆盖历史"。

只加列、不改主键;历史行后续回填 source="legacy" 由脚本完成(本迁移不改数据)。
"""
import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("matches", sa.Column("source", sa.String(32), nullable=True))
    op.add_column("matches", sa.Column("sources_json", sa.Text(), nullable=True))
    op.add_column("matches", sa.Column("source_scores_json", sa.Text(), nullable=True))
    op.add_column("matches", sa.Column("last_reconciled_at", sa.DateTime(), nullable=True))
    op.add_column("matches", sa.Column("reconciliation", sa.String(24), nullable=True))


def downgrade() -> None:
    for col in ("reconciliation", "last_reconciled_at", "source_scores_json",
                "sources_json", "source"):
        op.drop_column("matches", col)
