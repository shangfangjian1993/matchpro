from __future__ import annotations

"""统一入库层:清洗后的 NormalizedMatch → matches 表(upsert)。

幂等语义(一次成型,不再事后修补):
  - 键:(league_id, home_team, away_team, 日期)
  - 已存在:非空字段覆盖更新(scheduled 升级为 finished 时补比分);
            若新记录全部字段为 None(纯 xG 回填),只补空字段
  - 不存在:插入
"""

import logging

from app.data.canonical.cleanse import NormalizedMatch, validate
from app.data.canonical.store import _app_ctx, _get_or_create_league

logger = logging.getLogger(__name__)

# 可更新的指标字段(与 NormalizedMatch 对应)
_UPDATE_FIELDS = (
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_possession",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "home_ht_goals",
    "away_ht_goals",
    "home_passing_accuracy",
    "away_passing_accuracy",
    "match_stage",
)


def _get_or_create_team(db, name: str, team_type: str = "club") -> int:
    """按规范队名查/建球队实体;返回 team.id(新队自动建档,含中文名)。"""
    from app.api.db import Team
    from app.data.canonical.team_names_zh import to_zh

    team = db.session.query(Team).filter_by(name=name).first()
    if team is None:
        team = Team(name=name, name_zh=to_zh(name), team_type=team_type)
        db.session.add(team)
        db.session.flush()
    return team.id


def _season_label(dt) -> str:
    """足球赛季标签:8 月起为新赛季(2026-08 → 2026-2027)"""
    year = dt.year
    return f"{year}-{year + 1}" if dt.month >= 8 else f"{year - 1}-{year}"


def _upsert_team_season(db, team_id: int, league_id: int, season: str) -> None:
    """维护球队×联赛×赛季归属(幂等)。"""
    from app.api.db import TeamSeason

    exists = (
        db.session.query(TeamSeason)
        .filter_by(team_id=team_id, league_id=league_id, season=season)
        .first()
    )
    if exists is None:
        db.session.add(TeamSeason(team_id=team_id, league_id=league_id, season=season))


# matches 胖表拆分:比赛指标 → team_match_stats(每队每场一行)
_TMS_FIELDS = {
    "xg": "xg",
    "shots": "shots",
    "shots_on_target": "shots_on_target",
    "corners": "corners",
    "possession": "possession",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "ht_goals": "ht_goals",
    "passing_accuracy": "passing_accuracy",
    "xg_chain": "xg_chain",
    "efficiency": "efficiency",
    "transition_speed": "transition_speed",
    "defensive_actions": "defensive_actions",
    "counter_attacks": "counter_attacks",
    "tactical_rating": "tactical_rating",
    "experience": "experience",
}


def _write_team_stats(db, match) -> None:
    """把 match 的指标双写进 team_match_stats(幂等:存在则更新)。"""
    from app.api.db import TeamMatchStats

    for side, prefix, team_id in (
        ("home", "home_", match.home_team_id),
        ("away", "away_", match.away_team_id),
    ):
        row = (
            db.session.query(TeamMatchStats)
            .filter_by(match_id=match.id, side=side)
            .first()
        )
        data = {"match_id": match.id, "team_id": team_id, "side": side}
        for src, dst in _TMS_FIELDS.items():
            data[dst] = getattr(match, prefix + src, None)
        if row is None:
            db.session.add(TeamMatchStats(**data))
        else:
            for k, v in data.items():
                setattr(row, k, v)


def upsert_matches(matches: list[NormalizedMatch], source: str = "canonical") -> dict:
    """批量 upsert;返回 {"inserted": n, "updated": n, "skipped": n, "errors": [...]}

    source:调用方标识(fdco/zafronix/bzzoiro/…),用于 Source→Canonical 谱系
    (迁移 0014 后生效;未迁移时自动降级,不影响采集)。
    """
    from app.api.db import League, Match

    if not matches:
        return {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}

    # 按联赛分组(避免重复查库)
    by_league: dict[str, list[NormalizedMatch]] = {}
    for m in matches:
        by_league.setdefault(m.league_type, []).append(m)

    _, db = _app_ctx()
    result = {"inserted": 0, "updated": 0, "skipped": 0, "errors": []}
    from app.api.db import session_scope

    with session_scope():
        for league_type, group in by_league.items():
            league = _get_or_create_league(
                db, League, league_type, season_label=group[0].season_label
            )
            # 唯一 Match Identity:统一 CanonicalMatchResolver(双向解析 +
            # 方向 + ±1 天容差) —— A/B 与 B/A 永不产生第二行 canonical
            from app.data.canonical.resolver import CanonicalMatchResolver

            _resolver = CanonicalMatchResolver().index_matches(
                Match.query.filter_by(league_id=league.id).all()
            )

            def _find_old(nm, _res=_resolver):
                _r = _res.resolve(nm.home_team, nm.away_team, nm.date)
                return _r.match, _r.orientation

            for nm in group:
                errs = validate(nm)
                if errs:
                    result["errors"].append(
                        {
                            "team": f"{nm.home_team} vs {nm.away_team}",
                            "date": str(nm.date),
                            "error": errs[0],
                        }
                    )
                    continue
                # 球队实体自动建档(新队名 → teams 表)+ 外键解析;
                # 国家队赛事(世界杯/欧洲杯)球队标注 team_type=national
                is_national = league_type in ("WORLD_CUP", "EUROPEAN_CHAMPIONSHIP")
                team_type = "national" if is_national else "club"
                nm.home_team_id = _get_or_create_team(db, nm.home_team, team_type)
                nm.away_team_id = _get_or_create_team(db, nm.away_team, team_type)
                season = _season_label(nm.date)
                old, orientation = _find_old(nm)
                if old is None:
                    m = Match(
                        league_id=league.id,
                        home_team=nm.home_team,
                        away_team=nm.away_team,
                        match_date=nm.date,
                        match_status=nm.match_status,
                        home_team_id=nm.home_team_id,
                        away_team_id=nm.away_team_id,
                    )
                    _apply_fields(m, nm)
                    db.session.add(m)
                    result["inserted"] += 1
                    _resolver.index_matches([m])  # 新行并入 resolver,同批后续可定位
                    _upsert_team_season(db, nm.home_team_id, league.id, season)
                    _upsert_team_season(db, nm.away_team_id, league.id, season)
                    db.session.flush()  # 确保 m.id
                    _write_team_stats(db, m)
                else:
                    changed = False
                    # 旧行缺失外键时补齐
                    if old.home_team_id is None:
                        old.home_team_id = nm.home_team_id
                        changed = True
                    if old.away_team_id is None:
                        old.away_team_id = nm.away_team_id
                        changed = True
                    _upsert_team_season(db, nm.home_team_id, league.id, season)
                    _upsert_team_season(db, nm.away_team_id, league.id, season)
                    # 状态变化:升级到 finished 时强制覆盖比分(旧 scheduled 的 0 是占位值,
                    # merge_only 无法区分占位 0 与真实 0:0)
                    if old.match_status != nm.match_status and (
                        nm.match_status == "finished" or old.match_status != "finished"
                    ):
                        # 状态只允许单向升级(scheduled/postponed → finished);
                        # 防止已完赛行被 scheduled/postponed 覆盖成脏行(状态与比分矛盾)
                        old.match_status = nm.match_status
                        changed = True
                        if nm.match_status == "finished" and nm.home_goals is not None:
                            # 占位升级(scheduled 0:0 → finished 真实比分):
                            # force_override 覆盖并记录来源/旧值快照(0014 后)
                            from app.data.canonical.reconcile import maybe_update

                            maybe_update(
                                old,
                                nm,
                                source,
                                orientation=orientation,
                                force_override=True,
                            )
                    # 已完赛既有行收到与 canonical 不同比分 → 走 reconcile:
                    # 同值=consensus,异值=conflict 保留旧(不静默覆盖)——与
                    # bzzoiro.merge_league 统一语义(唯一 ingestion 路径)
                    elif (
                        nm.match_status == "finished"
                        and old.match_status == "finished"
                    ):
                        _rec2 = maybe_update(
                            old, nm, source, orientation=orientation
                        )
                    # 指标字段只补空(不覆盖已存在的真实值)
                    changed = (
                        _apply_fields(old, nm, merge_only=True, orientation=orientation)
                        or changed
                    )
                    _write_team_stats(db, old)
                    if changed:
                        result["updated"] += 1
                    else:
                        result["skipped"] += 1
        db.session.commit()
    return result


def _apply_fields(
    match, nm: NormalizedMatch, merge_only: bool = False, orientation: str = "SAME"
) -> bool:
    """把清洗结果写入 ORM 对象;merge_only=True 只补空字段。返回是否有变更。"""
    changed = False
    _pair_swap = {
        "home_goals": "away_goals",
        "away_goals": "home_goals",
        "home_ht_goals": "away_ht_goals",
        "away_ht_goals": "home_ht_goals",
    }
    for field in _UPDATE_FIELDS:
        v = getattr(nm, field)
        if v is None:
            continue
        # 来源方向 REVERSED 时,home/away 成对字段取对侧(防反向补空)
        if orientation == "REVERSED" and field in _pair_swap:
            v = getattr(nm, _pair_swap[field])
            if v is None:
                continue
        old_v = getattr(match, field)
        if merge_only and old_v is not None:
            continue
        if old_v != v:
            setattr(match, field, v)
            changed = True
    return changed
