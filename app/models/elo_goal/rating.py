"""足球 ELO 评分引擎(eloratings.net 风格)。

- 初始 1500;期望胜率 E = 1/(1+10^((R_away + H - R_home)/400))
- 更新 R' = R + K·G·(S - E);G = 净胜球权重(1球=1, 2球=1.5, ≥3球=(11+净胜)/8)
- 主场优势 H:俱乐部 70(联赛/欧冠),国家队中立场 0
- 平局 S=0.5;加时按结果计,点球大战简化为平局(数据无点球标记)

用法:
    elo = EloSystem()
    for home, away, gh, ga, is_national in matches_sorted_by_time:
        elo.update(home, away, gh, ga, home_adv=0 if is_national else 70)
    elo.rating("Manchester City FC")  # → float
"""

from __future__ import annotations

from collections import defaultdict


class EloSystem:
    """足球 ELO 系统:按时间顺序逐场更新,天然防泄漏(赛前 rating = 当前状态)。"""

    def __init__(
        self,
        initial: float = 1500.0,
        k: float = 20.0,
        home_advantage: float = 70.0,
        national_k: float = 30.0,
        dyn_k: bool = True,
        k_max_ratio: float = 2.0,
        k_tau: float = 50.0,
    ):
        self.initial = initial
        self.k = k  # 俱乐部 K 因子
        self.national_k = national_k  # 国家队 K 因子(比赛少,收敛快)
        self.home_advantage = home_advantage
        # 审查七 V7-2:Dynamic K —— 新球队(出场少)K 高快速收敛,
        # 稳定强队 K 平滑。k_eff = k × (1 + (k_max_ratio-1)·exp(-出场数/k_tau))
        self.dyn_k = dyn_k
        self.k_max_ratio = k_max_ratio
        self.k_tau = k_tau
        self._ratings: dict[str, float] = defaultdict(lambda: initial)
        self._appearances: dict[str, int] = defaultdict(int)

    def _k_factor(self, team: str) -> float:
        """动态 K 因子:出场 0 → k_max_ratio;出场 → ∞ → 1.0。"""
        if not self.dyn_k:
            return 1.0
        a = self._appearances.get(team, 0)
        return 1.0 + (self.k_max_ratio - 1.0) * __import__("math").exp(-a / self.k_tau)

    def rating(self, team: str) -> float:
        return self._ratings[team]

    def expected(self, home: str, away: str, home_adv: float = 70.0) -> float:
        """主队期望得分(0~1)。"""
        diff = (self._ratings[away] + home_adv) - self._ratings[home]
        return 1.0 / (1.0 + 10 ** (diff / 400.0))

    @staticmethod
    def goal_weight(gd: int) -> float:
        """净胜球权重:0/1 球=1,2 球=1.5,≥3 球=(11+净胜)/8。"""
        a = abs(gd)
        if a <= 1:
            return 1.0
        if a == 2:
            return 1.5
        return (11.0 + a) / 8.0

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        home_adv: float = 70.0,
        is_national: bool = False,
        mode: str = "overall",
    ) -> tuple[float, float]:
        """按赛果更新两队 rating;返回 (主队新 rating, 客队新 rating)。

        模式(V2 多维 ELO,每个维度独立实例互不污染):
        - overall:结果胜负更新(胜/平/负 + 净胜球权重)
        - attack:进球对决(进球多的一方攻击 ELO 上升)
        - defense:失球对决(失球少的一方防守 ELO 上升)
        """
        k = self.national_k if is_national else self.k
        # 审查七 V7-2:Dynamic K —— 主客队按各自出场数取因子
        k_home = k * self._k_factor(home)
        k_away = k * self._k_factor(away)
        if mode == "overall":
            e_home = self.expected(home, away, home_adv)
            e_away = 1.0 - e_home
            s_home = (
                1.0
                if home_goals > away_goals
                else (0.5 if home_goals == away_goals else 0.0)
            )
            g = self.goal_weight(home_goals - away_goals)
            self._ratings[home] += k_home * g * (s_home - e_home)
            self._ratings[away] += k_away * g * ((1.0 - s_home) - e_away)
        elif mode == "attack":
            # 攻击 ELO:进球数 vs 期望进球(rating 差推导,连续回归式)
            # 基线按实际场均进球校准(约 1.6/队),spread 控制差分配对期望
            base, spread = 1.6, 1.0
            diff = (self._ratings[home] - self._ratings[away]) / 400.0
            e_home = base + diff * spread
            e_away = base - diff * spread
            # 更新幅度 clip(防膨胀):单场 ±25
            self._ratings[home] += max(-25.0, min(25.0, k_home * (home_goals - e_home)))
            self._ratings[away] += max(-25.0, min(25.0, k_away * (away_goals - e_away)))
        elif mode == "defense":
            # 防守 ELO:失球数 vs 期望失球(防守好 → 期望失球少)
            base, spread = 1.6, 1.0
            diff = (self._ratings[home] - self._ratings[away]) / 400.0
            e_home = base - diff * spread
            e_away = base + diff * spread
            self._ratings[home] += max(-25.0, min(25.0, k_home * (e_home - away_goals)))
            self._ratings[away] += max(-25.0, min(25.0, k_away * (e_away - home_goals)))
        else:
            raise ValueError(f"未知 ELO 模式: {mode}")
        self._appearances[home] += 1
        self._appearances[away] += 1
        return self._ratings[home], self._ratings[away]


if __name__ == "__main__":
    # 自测:强队应上升,弱队应下降
    elo = EloSystem()
    elo.update("Strong FC", "Weak FC", 3, 0)
    assert elo.rating("Strong FC") > 1500
    assert elo.rating("Weak FC") < 1500
    # 平局小幅变化
    elo.update("Strong FC", "Weak FC", 1, 1)
    print("✅ EloSystem 自测通过")
    print(
        f"  Strong: {elo.rating('Strong FC'):.1f} | Weak: {elo.rating('Weak FC'):.1f}"
    )


def with_elo_features(df, is_national: bool = False, dyn_k: bool | None = None):
    """dyn_k:None 时读 configs/models.yaml elo_goal.dyn_k(默认 True)。"""
    """为比赛 DataFrame 附加赛前多维 ELO 特征(防泄漏:每场取赛前值)。

    输出列:
      home_elo / away_elo / elo_diff            (overall)
      home_attack_elo / away_attack_elo / attack_elo_diff   (攻击)
      home_defense_elo / away_defense_elo / defense_elo_diff (防守)
    输入需含:date / home_team / away_team / home_goals / away_goals(比分可 NaN)。
    """
    import numpy as np

    if dyn_k is None:
        try:
            from app.core.config import load_yaml

            dyn_k = bool(
                (load_yaml("models.yaml").get("elo_goal") or {}).get("dyn_k", True)
            )
        except Exception:
            dyn_k = True
    overall = EloSystem(dyn_k=dyn_k)
    attack = EloSystem(k=8.0, dyn_k=dyn_k)
    defense = EloSystem(k=8.0, dyn_k=dyn_k)
    df = df.sort_values("date").reset_index(drop=True)
    # 列数组迭代(替代 iterrows:避免逐行 pandas 行对象开销,~3x 提速;
    # ELO 更新是序列依赖的,无法完全向量化)
    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    ghs = df["home_goals"].to_numpy() if "home_goals" in df.columns else None
    gas = df["away_goals"].to_numpy() if "away_goals" in df.columns else None
    dates = df["date"].to_numpy()
    n = len(df)
    home_elo = np.empty(n)
    away_elo = np.empty(n)
    home_att = np.empty(n)
    away_att = np.empty(n)
    home_def = np.empty(n)
    away_def = np.empty(n)
    is_nan = (ghs is None) or (gas is None)
    # 修复:按"日初快照"计算同一天全部比赛的特征(同场同时开赛,互不影响),
    # 当天结束后再统一更新 ELO(次日起基于前一日最终状态)。
    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        o_snap = dict(overall._ratings)
        a_snap = dict(attack._ratings)
        d_snap = dict(defense._ratings)
        init_o, init_a, init_d = overall.initial, attack.initial, defense.initial
        for k in range(i, j):
            h, a = homes[k], aways[k]
            home_elo[k] = o_snap.get(h, init_o)
            away_elo[k] = o_snap.get(a, init_o)
            home_att[k] = a_snap.get(h, init_a)
            away_att[k] = a_snap.get(a, init_a)
            home_def[k] = d_snap.get(h, init_d)
            away_def[k] = d_snap.get(a, init_d)
        for k in range(i, j):
            gh, ga = (ghs[k], gas[k]) if not is_nan else (np.nan, np.nan)
            if (
                gh is not None
                and ga is not None
                and np.isfinite(gh)
                and np.isfinite(ga)
            ):  # 非 None 非 NaN
                h, a = homes[k], aways[k]
                ha = 0.0 if is_national else overall.home_advantage
                overall.update(
                    h, a, int(gh), int(ga), home_adv=ha, is_national=is_national
                )
                attack.update(
                    h,
                    a,
                    int(gh),
                    int(ga),
                    home_adv=ha,
                    is_national=is_national,
                    mode="attack",
                )
                defense.update(
                    h,
                    a,
                    int(gh),
                    int(ga),
                    home_adv=ha,
                    is_national=is_national,
                    mode="defense",
                )
        i = j
    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["home_attack_elo"] = home_att
    df["away_attack_elo"] = away_att
    df["home_defense_elo"] = home_def
    df["away_defense_elo"] = away_def
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["attack_elo_diff"] = df["home_attack_elo"] - df["away_attack_elo"]
    df["defense_elo_diff"] = df["home_defense_elo"] - df["away_defense_elo"]
    return df
