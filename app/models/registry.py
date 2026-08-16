"""模型注册表(审查 §14/§39 拆分:自 data_adapter 迁移)。

职责:版本管理(路径解析/版本扫描/递增/prune/lock/active 指针)。
"""
from __future__ import annotations

import logging
import os

from app.core.config import LeagueType
from app.models.integrity import _sha256_of

logger = logging.getLogger(__name__)

_MODELS_KEEP = int(os.environ.get("MODELS_KEEP", "3"))
def _resolve_league_type(league_type: str) -> LeagueType:
    """把前端传的字符串(如 PREMIER_LEAGUE 或 premier_league)解析为 LeagueType"""
    if not league_type:
        raise ValueError("缺少 league_type 参数")
    try:
        return LeagueType(league_type)      # 按枚举值(小写)匹配
    except ValueError:
        try:
            return LeagueType[league_type]  # 按枚举名(大写)匹配
        except KeyError:
            raise ValueError(f"不支持的联赛类型: {league_type}")

def _model_path(league_type: LeagueType, models_dir: str, version: str | None = None) -> str:
    """模型文件路径:version 为 None 时返回 latest 指针文件;否则返回版本化文件"""
    if version:
        return os.path.join(models_dir, "artifacts", league_type.value, f"{version}.pkl")
    _active = _active_version(league_type, models_dir)
    if _active:
        _p = os.path.join(models_dir, "artifacts", league_type.value, f"{_active}.pkl")
        if os.path.exists(_p):
            return _p
    _vs = _existing_versions(league_type, models_dir)
    if _vs:
        return os.path.join(models_dir, "artifacts", league_type.value, f"{_vs[-1]}.pkl")
    # 审查 §44:无 latest 语义(active_models.json 为唯一指针);版本缺失 → 调用方处理
    return os.path.join(models_dir, "artifacts", league_type.value, "_missing.pkl")


def _is_pointer_file(league_type: LeagueType, models_dir: str, path: str) -> bool:
    """判断是否为旧 latest 指针复制文件(与某版本文件哈希相同)。"""
    try:
        _h = _sha256_of(path)
    except Exception:
        return False
    for _v in _existing_versions(league_type, models_dir):
        _p = os.path.join(models_dir, "artifacts", league_type.value, f"{_v}.pkl")
        try:
            if os.path.exists(_p) and _sha256_of(_p) == _h:
                return True
        except Exception:
            continue
    return False

def _version_key(v: str):
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0, 0, 0)

def _existing_versions(league_type: LeagueType, models_dir: str) -> list:
    """扫描 models_dir 中该联赛的已保存版本(升序)"""
    d = os.path.join(models_dir, "artifacts", league_type.value)
    versions = []
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.endswith(".pkl") and fn != "gbm.pkl":
                versions.append(fn[:-4])
    return sorted(set(versions), key=_version_key)

def _bump_version(league_type: LeagueType, models_dir: str) -> str:
    """版本号递增:1.0.0 -> 1.0.1 -> ...(无历史版本时从 1.0.0 开始)"""
    versions = _existing_versions(league_type, models_dir)
    if not versions:
        return "1.0.0"
    latest = versions[-1]
    try:
        major, minor, patch = (int(x) for x in latest.split("."))
        return f"{major}.{minor}.{patch + 1}"
    except ValueError:
        return "1.0.0"

def _prune_old_versions(league_type: LeagueType, models_dir: str, keep: int = _MODELS_KEEP) -> None:
    """保留策略:每联赛仅保留最近 keep 个版本(含 .sha256),删除更旧的。

    需在版本锁内调用,避免与并发训练竞争。latest 指针文件(非 v* 命名)不受影响。
    """
    keep = max(1, int(keep))
    versions = _existing_versions(league_type, models_dir)
    # 审查 P0-8:active_models.json 指向的版本受保护,不参与 prune
    # (曾发生:最优版本被 prune 删除 → active 悬空 → 加载退化)
    _protected = _active_version(league_type, models_dir)
    for v in versions[:-keep]:
        if v == _protected:
            continue
        for suffix in ("", ".sha256"):
            fp = _model_path(league_type, models_dir, v) + suffix
            try:
                os.remove(fp)
                logger.info("清理旧模型版本: %s", fp)
            except OSError:
                pass

def _version_lock(models_dir: str):
    """跨进程文件锁(flock):保护 '版本递增+写文件' 临界区,防止并发训练同联赛竞争"""
    try:
        import fcntl
        os.makedirs(models_dir, exist_ok=True)
        f = open(os.path.join(models_dir, ".train.lock"), "w")  # 锁需持句柄,不能 with
        fcntl.flock(f, fcntl.LOCK_EX)
        return f
    except (ImportError, OSError):
        return None  # 无 fcntl 平台退化为无锁

def _version_unlock(f) -> None:
    if f is None:
        return
    try:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        pass
    f.close()


def _active_version(league_type: LeagueType, models_dir: str) -> str | None:
    """P2-D 自动模型选择:active_models.json 中该联赛的最优版本(实验跟踪驱动)。"""
    try:
        from app.core.paths import PROJECT_ROOT
        p = os.path.join(str(PROJECT_ROOT), "runtime", "active_models.json")
        if os.path.exists(p):
            import json as _json
            with open(p, encoding="utf-8") as f:
                return _json.load(f).get(league_type.value)
    except Exception:
        pass
    return None

