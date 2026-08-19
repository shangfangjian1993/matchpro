"""Source → Canonical reconcile 测试(审查 A70A601 §15-17)。

覆盖:
- 未迁移(无 lineage 列)→ legacy 直写(兼容采集)
- 已迁移:同值→consensus;异值跨源→conflict 保留旧;force_override→覆盖+快照
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _lineage_on(monkeypatch):
    """默认模拟已迁移(有 lineage 列);legacy 测试单独关闭。"""
    from app.data.canonical import reconcile

    monkeypatch.setattr(reconcile, "lineage_available", lambda *a, **k: True)


class _Old:
    """模拟 ORM Match 行属性。"""

    def __init__(self):
        self.home_goals = 2
        self.away_goals = 1
        self.home_ht_goals = 1
        self.away_ht_goals = 0
        self.source = None
        self.sources_json = None
        self.source_scores_json = None
        self.reconciliation = None
        self.last_reconciled_at = None
        self.source_consensus = None


class _NM:
    def __init__(self, hg, ag, hh=None, ah=None):
        self.home_goals = hg
        self.away_goals = ag
        self.home_ht_goals = hh
        self.away_ht_goals = ah


def test_legacy_when_no_lineage(monkeypatch):
    from app.data.canonical import reconcile

    monkeypatch.setattr(reconcile, "lineage_available", lambda *a, **k: False)
    old = _Old()
    nm = _NM(3, 0, 1, 0)
    assert reconcile.maybe_update(old, nm, "bzzoiro") == "legacy_override"
    assert old.home_goals == 3  # 直写覆盖(旧行为)


def test_consensus_when_same_score():
    from app.data.canonical import reconcile

    old = _Old()
    nm = _NM(2, 1, 1, 0)
    assert reconcile.maybe_update(old, nm, "bzzoiro") == "consensus"
    assert json.loads(old.sources_json) == ["bzzoiro"]
    assert old.reconciliation == "consensus"
    # 审查 §六建议:last_verified_at 与 source_consensus 已记录
    assert old.last_reconciled_at is not None
    assert old.source_consensus is not None
    _cs = json.loads(old.source_consensus)
    assert "/" in _cs["consensus"]


def test_conflict_preserves_old_value():
    from app.data.canonical import reconcile

    old = _Old()
    old.source = "fdco"  # 已有来源
    old.sources_json = json.dumps(["fdco"])
    nm = _NM(3, 1, 2, 0)  # 冲突(home 2→3)
    assert reconcile.maybe_update(old, nm, "bzzoiro") == "conflict"
    assert old.home_goals == 2  # 保留旧值(不静默覆盖)
    snap = json.loads(old.source_scores_json)
    assert snap["home_goals"]["fdco"] == 2  # 旧值已入快照
    assert "bzzoiro" in json.loads(old.sources_json)


def test_override_when_first_source():
    from app.data.canonical import reconcile

    old = _Old()
    old.source = None  # 无既有来源(全新/legacy)
    old.src_json = None
    old.sources_json = None
    nm = _NM(3, 0, 2, 0)
    assert reconcile.maybe_update(old, nm, "bzzoiro") == "override"
    assert old.home_goals == 3
    assert old.source == "bzzoiro"
    assert old.reconciliation == "override"


def test_force_override_records_snapshot():
    from app.data.canonical import reconcile

    old = _Old()
    old.reconciliation = None
    nm = _NM(1, 1, 0, 1)
    assert reconcile.maybe_update(old, nm, "ingest", force_override=True) == "override"
    assert old.home_goals == 1  # 占位升级覆盖
    snap = json.loads(old.source_scores_json)
    assert snap["home_goals"]["legacy"] == 2  # 旧值已快照
