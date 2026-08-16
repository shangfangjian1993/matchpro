"""V2 API 请求/响应模型(新设计:类型安全,Pydantic 校验)。"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthLogin(BaseModel):
    username: str = Field(min_length=1, description="用户名")
    password: str = Field(min_length=1, description="密码")


class AuthRegister(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    email: str
    password: str = Field(min_length=8)


class PredictMatchReq(BaseModel):
    league_type: str = Field(description="联赛枚举值,如 premier_league")
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    date: str | None = None


class TournamentTeam(BaseModel):
    name: str


class PredictTournamentReq(BaseModel):
    league_type: str
    teams: list[TournamentTeam | str] = Field(min_length=2)
    num_simulations: int = Field(default=1000, ge=10, le=100000)


class MatchCreate(BaseModel):
    """单场录入;指标列(白名单)经 extra 透传。"""
    model_config = ConfigDict(extra="allow")

    league_type: str
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    home_goals: int = 0
    away_goals: int = 0
    match_date: str | None = None
    match_status: str = "finished"
    match_stage: str = ""


class MatchBatchReq(BaseModel):
    league_type: str
    matches: list[dict[str, Any]] = Field(min_length=1, max_length=2000)


class TrainingSubmitReq(BaseModel):
    league_type: str
    target_column: str = "goals"
    cross_validation: bool = True
    cv_folds: int = Field(default=5, ge=2, le=20)


class SettingsUpdate(BaseModel):
    notifications: bool | None = None
    darkMode: bool | None = None
    timezone: str | None = None
    language: str | None = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class ApiSettingsUpdate(BaseModel):
    enabled: bool | None = None
    rateLimit: int | None = Field(default=None, ge=1, le=100000)
    apiKey: str | None = None

