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
 orientation: str = "SAME" # "SAME" | "REVERSED"
 confidence: float = 0.0
 reason: str = ""
 ambiguous: bool = False # 多候选歧义标记


def normalize_side(team: str) -> str:
 """统一的归一健名(队名后缀剥离+别名→规范 key;详见 canonical_en)。"""
 return canonical_en(team)


class CanonicalMatchResolver:
 """单场归属的一次性解析器(索引构建 + 解析)。

 用法:
 r = CanonicalMatchResolver()
 r.index_matches([...]) # 一次索引联赛内所有行
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
 if isinstance(d, str):
 d = datetime.fromisoformat(d.replace("Z", "+00:00"))
 else:
 d = datetime.combine(d, datetime.min.time())
 except Exception:
 d = None
 d0 = d.date() if isinstance(d, datetime) else d
 tol = self._day_tolerance

 def _score_candidate(m, mob):
 """评分:exact date > ±1 day,同向 > 反向。返回 (score, confidence, reason)。"""
 md = getattr(m, "match_date", None)
 if md is None or d0 is None:
 return (0, 0.5, "no-date")
 mday = md.date() if isinstance(md, datetime) else md
 try:
 diff = abs((mday - d0).days)
 except TypeError:
 return (0, 0.5, "date-error")
 if diff > tol:
 return (-1, 0.0, "out-of-range") # 不符合
 # 评分
 score = 100 - diff * 30 # exact=100, ±1=70
 if mob == "SAME":
 score += 10
 confidence = 0.99 if diff == 0 else 0.90
 return (score, confidence, "exact-date" if diff == 0 else f"±{diff}-day")

 # 查找:优先用户给出的方向,再尝试反向
 # 索引:(source_home, source_away) → [(canonical, orientation)]
 candidates = []
 
 # 优先方向:用户给的 (home, away)
 for m, mob in self._idx.get((hn, an), []):
 score, conf, reason = _score_candidate(m, mob)
 if score >= 0:
 candidates.append((score, m, mob, conf, reason))
 
 # 反向:用户给的 (away, home) — 对应 canonical 的 (home, away) 方向
 for m, mob in self._idx.get((an, hn), []):
 # 用户输入 (home=X, away=Y) 但索引中 (Y, X) 存的是 REVERSED
 # 这里 mob 是索引中存储的方向,对于反向查找,mob="REVERSED" 表示
 # canonical 中 Y 是 home,X 是 away,与用户输入一致
 score, conf, reason = _score_candidate(m, mob)
 if score >= 0:
 # 反向查找时,不给予 SAME 的 +10 奖励(因为方向已经不对)
 if mob == "SAME":
 score -= 10 # 取消奖励
 candidates.append((score, m, mob, conf, reason))

 if not candidates:
 return MatchResolution(
 match=None, orientation="SAME", confidence=0.0, reason="no-match"
 )

 # 按分数排序
 candidates.sort(key=lambda x: -x[0])
 best_score, best_m, best_mob, best_conf, best_reason = candidates[0]

 # 歧义检测:仅当 top 2 指向不同比赛且分数接近时标记 ambiguous
 ambiguous = False
 if len(candidates) >= 2:
 second_score, second_m = candidates[1][0], candidates[1][1]
 if second_m.id != best_m.id and best_score - second_score <= 20:
 ambiguous = True
 best_reason = f"ambiguous-{best_reason}"

 return MatchResolution(
 match=best_m,
 orientation=best_mob,
 confidence=best_conf,
 reason=best_reason,
 ambiguous=ambiguous,
 )
