"""Source → Canonical 对账(单一 reconciliation 语义)。

不变量(Business invariant):
- canonical score 必须保持 canonical orientation(来源主客场反转不污染)。
- 不因"最后来的源"静默覆盖历史;跨源冲突 → 保留旧值 + 标记 conflict。
- source_consensus 为**来源级共识**(多少数据源与 canonical 一致),非字段级。

结构:lineage 需要 migrate 0014 列才启用;未迁移时回退旧直写(兼容)。
"""

from __future__ import annotations

import json

from app.core.timeutil import utcnow

# 比分/半场字段(home/away 成对,支持方向对齐)
_SCORE_PAIRS = (
    ("home_goals", "away_goals"),
    ("home_ht_goals", "away_ht_goals"),
)
_SCORE_FIELDS = tuple(x for pr in _SCORE_PAIRS for x in pr)


def lineage_available(columns: tuple[str, ...] | None = None) -> bool:
    """Match 表是否已有 lineage 列(migrate 0014)。columns 可注入(测试用)。"""
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


def _aligned_scores(old, nm, orientation: str) -> dict:
    """按 orientation 把来源比分对齐到 canonical 方向(仅比分/半场字段)。"""
    aligned = {}
    for l, r in _SCORE_PAIRS:
        nv_l, nv_r = getattr(nm, l, None), getattr(nm, r, None)
        if orientation == "REVERSED":
            aligned[l], aligned[r] = nv_r, nv_l
        else:
            aligned[l], aligned[r] = nv_l, nv_r
    return aligned


def _same_score(old, aligned_scores: dict) -> bool:
    """aligned_scores(已对齐到 canonical 方向)与旧行当前值是否一致。
    全字段(已完成的)均一致才算同分。"""
    for l, r in _SCORE_PAIRS:
        ov = getattr(old, l, None)
        if aligned_scores[l] is not None and ov is not None and aligned_scores[l] != ov:
            return False
        ov = getattr(old, r, None)
        if aligned_scores[r] is not None and ov is not None and aligned_scores[r] != ov:
            return False
    return True


def _write_scores(old, aligned_scores: dict):
    for l, r in _SCORE_PAIRS:
        if aligned_scores[l] is not None:
            setattr(old, l, aligned_scores[l])
        if aligned_scores[r] is not None:
            setattr(old, r, aligned_scores[r])


def _source_profile(old) -> dict:
    """source_scores_json → {source: {field: value}}（兼容旧 field→source 结构）。"""
    raw = _load_json(getattr(old, "source_scores_json", None), {})
    if not isinstance(raw, dict):
        return {}
    if "sources" in raw:
        return raw["sources"]
    # 旧结构 {field: {source: value}} → 转
    out: dict = {}
    for f, srcmap in raw.items():
        if isinstance(srcmap, dict):
            for src, v in srcmap.items():
                out.setdefault(src, {})[f] = v
    return out


def _record_source(old, source: str, aligned_scores: dict):
    """把该来源(对齐后)的比分快照写入 source_scores_json。"""
    prof = _source_profile(old)
    prof[source] = {f: aligned_scores.get(f) for f in _SCORE_FIELDS}
    old.source_scores_json = json.dumps({"sources": prof}, ensure_ascii=False)


def _source_consensus(old, aligned_scores: dict) -> dict:
    """来源级共识:与 canonical 当前分值**整场一致**的来源计数。

    返回 {"agree_sources":[...], "disagree_sources":[...], "agree":n,
    "total":m, "ratio":r}。canonical 当前值 = old 行当前(home/away 方向)。
    """
    prof = _source_profile(old)
    agree, disagree = [], []
    for src, sc in prof.items():
        if not isinstance(sc, dict):
            continue
        # 只要求 home/away 整场(半场缺失不扣分)判该源一致
        h, a = sc.get("home_goals"), sc.get("away_goals")
        if h is None or a is None:
            continue
        if h == getattr(old, "home_goals", None) and a == getattr(
            old, "away_goals", None
        ):
            agree.append(src)
        else:
            disagree.append(src)
    total = len(agree) + len(disagree)
    return {
        "agree_sources": agree,
        "disagree_sources": disagree,
        "agree": len(agree),
        "total": total,
        "ratio": round(len(agree) / total, 4) if total else None,
    }


def _touch(old, aligned_scores: dict | None = None):
    """每次对账触达:记录 verified 时间与来源级共识。"""
    try:
        old.last_reconciled_at = utcnow()
    except Exception:
        pass
    if aligned_scores is None:
        return
    try:
        old.source_consensus = json.dumps(
            _source_consensus(old, aligned_scores), ensure_ascii=False
        )
    except Exception:
        pass


def maybe_update(
    old,
    nm,
    source: str,
    orientation: str = "SAME",
    force_override: bool = False,
) -> str:
    """决定并执行对旧行的更新;返回 reconciliation 结果标记。

    orientation: 来源相对 canonical 方向(SAME/REVERSED)—— 反转时比分/半场
    按对齐写入,绝不用来源方向污染 canonical。
    force_override: 占位升级(如 scheduled 0:0 → finished 真实比分)总是覆盖,
    并记录来源/快照(非跨源冲突)。
    返回: "consensus" / "override" / "conflict" / "legacy_override"
    """
    aligned_scores = _aligned_scores(old, nm, orientation)
    if not lineage_available():
        # 未迁移:回退旧直写(仍用对齐值防反转污染,不引入依赖)
        _write_scores(old, aligned_scores)
        return "legacy_override"

    sources = _load_json(getattr(old, "sources_json", None), [])
    if source not in sources:
        sources.append(source)
        old.sources_json = json.dumps(sources, ensure_ascii=False)
    _record_source(old, source, aligned_scores)

    same = _same_score(old, aligned_scores)
    if force_override:
        _write_scores(old, aligned_scores)
        old.source = source
        old.reconciliation = "override"
        _touch(old, aligned_scores)
        return "override"
    if same:
        old.source = old.source or source
        if old.reconciliation is None:
            old.reconciliation = "consensus"
        _touch(old, aligned_scores)
        return "consensus"

    # 值冲突:首源(无既有来源)→ 覆盖并标记 override;跨源 → 保留旧+conflict
    if old.source is None or old.source == "legacy":
        old.source = source
        _write_scores(old, aligned_scores)
        old.reconciliation = "override"
        _touch(old, aligned_scores)
        return "override"
    old.reconciliation = "conflict"
    _touch(old, aligned_scores)
    return "conflict"
