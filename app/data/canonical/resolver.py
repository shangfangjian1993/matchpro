"""唯一 Match Identity 解析(域的单一入口)。

Business invariants:
- 一场比赛只有一条 canonical 行 —— 任何来源都能解析到它,且保持 canonical
  主客场方向(A vs B 与 B vs A 视为同一场比赛,方向不同)。
- 归一化(队名后缀剥离/别名)统一在 resolver 内,各来源不再各自实现
  _find/_norm/date tolerance。

resolve(...) → MatchResolution(match, orientation=SAME|REVERSED, ...)
所有来源(api-football/fdco/bzzoiro/understat/zafronix)只能经由此层。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from app.data.canonical.team_names import canonical_en


@dataclass
class MatchResolution:
    match: object | None
    orientation: str = "SAME"  # "SAME" | "REVERSED"
    confidence: float = 0.0
    reason: str = ""


def normalize_side(team: str) -> str:
    """统一的归一健名(队名后缀剥离+别名→规范 key;详见 canonical_en)。"""
    return canonical_en(team)


class CanonicalMatchResolver:
    """单场归属的一次性解析器(索引构建 + 解析)。

    用法:
        r = CanonicalMatchResolver()
        r.index_matches([...])     # 一次索引联赛内所有行
        res = r.resolve(home, away, match_date)
    """

    def __init__(self, day_tolerance: int = 1):
        self._day_tolerance = day_tolerance
        # (hn, an) → list[(Match, orientation)]:先同向后反向
        self._idx: dict[tuple, list] = {}

    def index_matches(self, matches: list) -> CanonicalMatchResolver:
        for m in matches:
            hn, an = normalize_side(m.home_team), normalize_side(m.away_team)
            self._idx.setdefault((hn, an), []).append((m, "SAME"))
            self._idx.setdefault((an, hn), []).append((m, "REVERSED"))
        return self

    def resolve(
        self,
        home: str,
        away: str,
        match_date: datetime | date | None,
        prefer: str = "SAME",
    ) -> MatchResolution:
        """解析 (home, away, date) 到 canonical 行 + 方向。

        prefer 为 "SAME" 时优先同向、再反向;为 "REVERSED" 相反。
        """
        hn, an = normalize_side(home), normalize_side(away)
        d = match_date
        if d is not None and not isinstance(d, (date, datetime)):
            try:
                from pandas import Timestamp

                d = (
                    Timestamp(d).date()
                    if not isinstance(d, datetime)
                    else (d.date() if isinstance(d, datetime) else d)
                )
            except Exception:
                d = None
        d0 = d.date() if isinstance(d, datetime) else d
        tol = self._day_tolerance

        def _near(m):
            md = getattr(m, "match_date", None)
            if md is None or d0 is None:
                return True
            mday = md.date() if isinstance(md, datetime) else md
            try:
                return abs((mday - d0).days) <= tol
            except TypeError:
                return True

        # 索引已双向建立:key=(来源归一主客) → [(canonical行, 相对方向)]
        # 命中条目的 mob 即"来源相对 canonical 的方向"(同向=SAME / 对调=REVERSED)
        keys = (
            [("SAME", (hn, an))] if prefer != "REVERSED" else [("REVERSED", (an, hn))]
        )
        keys.append(
            ("REVERSED", (an, hn)) if prefer != "REVERSED" else ("SAME", (hn, an))
        )
        for _p, key in keys:
            for m, mob in self._idx.get(key, []):
                if _near(m):
                    return MatchResolution(
                        match=m,
                        orientation=mob,
                        confidence=1.0,
                        reason=f"index-{mob.lower()}",
                    )
        return MatchResolution(
            match=None, orientation="SAME", confidence=0.0, reason="no-match"
        )
