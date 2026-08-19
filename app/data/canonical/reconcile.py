"""Source → Canonical 对账(审查 A70A601 §15-17:消除静默覆盖)。

maybe_update(old, nm, source):
- DB 有 lineage 列(已迁移 0014):
    同值 → 仅 sources_json 追加(consensus)
    异值 → 记录旧值快照到 source_scores_json;跨源冲突默认"保留旧+标记
           conflict"(不再以"最后来的"静默覆盖历史);同源时间戳新=覆盖。
- DB 未迁移(列不存在)→ 回退旧直写行为(不依赖迁移,采集不受阻塞)。

兼容性:迁移前与迁移后行为均安全;迁移后自动启用谱系。
"""

from __future__ import annotations

import json

# 计数字段(与 NormalizedMatch 对齐)
_SCORE_FIELDS = ("home_goals", "away_goals", "home_ht_goals", "away_ht_goals")


def lineage_available(columns: tuple[str, ...] | None = None) -> bool:
    """Match 表是否已有 lineage 列(迁移 0014)。columns 可注入(测试用)。"""
    if columns is not None:
        return "source" in columns and "sources_json" in columns
    from app.api.db import Match

    cols = {c.name for c in Match.__table__.columns}
    return "source" in cols and "sources_json" in cols


def _load_json(text: str | None, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _same_score(old, nm) -> bool:
    for f in _SCORE_FIELDS:
        nv = getattr(nm, f, None)
        ov = getattr(old, f, None)
        if nv is not None and ov is not None and nv != ov:
            return False
    return True


def maybe_update(old, nm, source: str, force_override: bool = False) -> str:
    """决定并执行对旧行的更新;返回 reconciliation 结果标记。

    force_override=True:占位升级(如 scheduled 0:0 → finished 真实比分),
    总是覆盖比分并记录来源/快照(非跨源冲突)。
    返回值: "consensus" / "override" / "conflict" / "legacy_override"
    """
    if not lineage_available():
        # 未迁移:保持旧直写行为(不引入依赖)
        for f in _SCORE_FIELDS:
            v = getattr(nm, f, None)
            if v is not None:
                setattr(old, f, v)
        return "legacy_override"

    sources = _load_json(getattr(old, "sources_json", None), [])
    if source not in sources:
        sources.append(source)
        old.sources_json = json.dumps(sources, ensure_ascii=False)

    same = _same_score(old, nm)
    if force_override:
        # 占位升级:记录旧值快照后覆盖(不算冲突)
        snap = _load_json(getattr(old, "source_scores_json", None), {})
        snap_entry = old.source or "legacy"
        for f in _SCORE_FIELDS:
            v = getattr(old, f, None)
            if v is not None:
                snap.setdefault(f, {})[snap_entry] = v
        old.source_scores_json = json.dumps(snap, ensure_ascii=False)
        for f in _SCORE_FIELDS:
            v = getattr(nm, f, None)
            if v is not None:
                setattr(old, f, v)
        old.source = source
        old.reconciliation = "override"
        return "override"
    if same:
        old.source = old.source or source
        opm = old.reconciliation
        if opm is None:
            old.reconciliation = "consensus"
        return "consensus"

    # 值冲突:先记旧值快照(可溯),再决策
    snap = _load_json(getattr(old, "source_scores_json", None), {})
    snap_entry = {
        "source": old.source or "legacy",
        "updated_at": str(getattr(old, "last_reconciled_at", "")),
    }
    for f in _SCORE_FIELDS:
        v = getattr(old, f, None)
        nap = snap.get(f, {})
        snap_key = snap_entry["source"]
        if nap.get(snap_key) is None or nap[snap_key] != v:
            snap.setdefault(f, {})[snap_key] = v
    old.source_scores_json = json.dumps(snap, ensure_ascii=False)

    # 决策:无旧来源(legacy/单源)且冲突 → 标记 conflict,保留旧值
    if old.source is None or old.source == "legacy":
        old.source = source
        for f in _SCORE_FIELDS:
            v = getattr(nm, f, None)
            if v is not None:
                setattr(old, f, v)
        old.reconciliation = "override"
        return "override"
    # 跨源冲突:保留旧值并标记(不静默覆盖历史)
    old.reconciliation = "conflict"
    return "conflict"
