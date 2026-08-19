"""数据契约:主客场反转不得污染 canonical 比分(审查 f01d7e4 P0-1)。

canonical: A 2-1 B;source 以 B 1-2 A 视角返回(REVERSED)
→ 更新后 canonical 仍为 A 2-1 B(半场也按方向对齐)。
"""

from __future__ import annotations

import json

import pytest


class _Old:
    def __init__(self, hg=2, ag=1, hh=1, ah=0):
        self.home_goals = hg
        self.away_goals = ag
        self.home_ht_goals = hh
        self.away_ht_goals = ah
        self.source = None
        self.sources_json = None
        self.source_scores_json = None
        self.reconciliation = None
        self.last_reconciled_at = None


class _NM:
    def __init__(self, hg, ag, hh=None, ah=None):
        self.home_goals = hg
        self.away_goals = ag
        self.home_ht_goals = hh
        self.away_ht_goals = ah


@pytest.fixture(autouse=True)
def _lineage_on(monkeypatch):
    from app.data.canonical import reconcile

    monkeypatch.setattr(reconcile, "lineage_available", lambda *a, **k: True)


def test_reversed_orientation_does_not_flip_canonical():
    from app.data.canonical.reconcile import maybe_update

    old = _Old(hg=2, ag=1, hh=1, ah=0)  # canonical: A 2-1 B(半场 1-0)
    nm = _NM(hg=1, ag=2, hh=0, ah=1)  # 来源视角: B 1-2 A(same canonical,但方向反)
    # SAME 方向(来源 B 1-2 A 若 B 是 home):则 canonical A home 应为 nm.away=2, nm.home=1
    rec = maybe_update(old, nm, "bzzoiro", orientation="REVERSED")
    assert rec in ("consensus", "override")
    # REVERSED:canonical 保持 A 2-1 B —— 用 nm(away)=2 → home, nm(home)=1 → away
    assert old.home_goals == 2 and old.away_goals == 1
    assert old.home_ht_goals == 1 and old.away_ht_goals == 0


def test_reversed_orientation_conflict_preserves_old():
    from app.data.canonical.reconcile import maybe_update

    old = _Old(hg=2, ag=1)
    old.source = "fdco"
    old.sources_json = json.dumps(["fdco"])
    nm = _NM(hg=3, ag=0)  # 来源(不管方向)真实比分不同 → 冲突保留旧
    rec = maybe_update(old, nm, "bzzoiro", orientation="REVERSED")
    assert rec == "conflict"
    assert old.home_goals == 2 and old.away_goals == 1  # 保留旧值


def test_multi_source_irreversible_pollution(monkeypatch):
    """审查 §25:多次来源顺序反转/冲突后,canonical 不被污染且标记 conflict。

    源1: A 2-1 B(SAME);源2: B 9-0 A(REVERSED 但真实比分异常,须冲突)
    源3: A 2-1 B(SAME)→ canonical 恒为 A 2-1 B。
    """
    from app.data.canonical.reconcile import maybe_update

    old = _Old(hg=2, ag=1, hh=1, ah=0)
    # 首源进入(自动 override)
    assert maybe_update(old, _NM(2, 1, 1, 0), "fdco", orientation="SAME") == "consensus"
    old.source = "fdco"
    # 冲突源(不同比分,REVERSED)→ 保留旧
    rec = maybe_update(old, _NM(0, 9, 0, 5), "bzzoiro", orientation="REVERSED")
    assert rec == "conflict"
    # 再次同值源 → consensus 且 canonical 不变
    assert maybe_update(old, _NM(2, 1, 1, 0), "api_football", orientation="SAME") in (
        "consensus",
        "override",
    )
    assert old.home_goals == 2 and old.away_goals == 1
