"""bzzoiro 模块契约(

曾发生:31a60cd 清理把 available/_get/fetch_*/LEAGUE_IDS/to_normalized/
import_* 等全部缩进进 _key() 内(语法仍合法,py_compile/ruff/import 均过,
但 from ... import LEAGUE_IDS 全线 ImportError → 采集崩溃循环)。
本测试从 **源码文本层级** 直接断言(不依赖 import 结果),任何缩进回归立即失败。
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "app" / "data" / "sources" / "bzzoiro.py"


def _src_text() -> str:
 return _SRC.read_text(encoding="utf-8")


MODULE_LEVEL_SYMBOLS = [
 "def _key",
 "def available",
 "def _get",
 "def fetch_events",
 "def fetch_odds",
 "def find_league",
 "def _leagues_page",
 "LEAGUE_IDS = {",
 "def to_normalized",
 "def _season_label",
 "def import_league",
 "def _norm",
 "def merge_league",
 "def _event_match_index",
 "def ingest_stats",
 "def ingest_odds",
 "def import_recent",
]


def test_bzzoiro_module_level_structure():
 """源码层级:上述符号必须以模块级(行首无缩进)出现。"""
 text = _src_text()
 for sym in MODULE_LEVEL_SYMBOLS:
 assert re.search(rf"^{re.escape(sym)}", text, re.MULTILINE), (
 f"{sym} 不在模块级(可能又被嵌套缩进!)"
 )
 # 反向:不得出现缩进 4 的 available/_get(嵌套在别的函数里的特征)
 for nested in ("^ def available", "^ def _get", "^ LEAGUE_IDS"):
 assert not re.search(nested, text, re.MULTILINE), f"检测到嵌套定义: {nested}"


def test_bzzoiro_module_contract_import():
 """Import Contract:全部公共 API 存在且 callable/类型正确。"""
 from app.data.sources import bzzoiro

 for name in (
 "_key",
 "available",
 "_get",
 "fetch_events",
 "fetch_odds",
 "find_league",
 "to_normalized",
 "import_league",
 "merge_league",
 "ingest_stats",
 "ingest_odds",
 "import_recent",
 ):
 assert callable(getattr(bzzoiro, name)), f"bzzoiro.{name} 应可调用"
 assert isinstance(bzzoiro.LEAGUE_IDS, dict)
 assert bzzoiro.LEAGUE_IDS.get("premier_league") == 1


def test_bzzoiro_key_env_priority(monkeypatch):
 """环境变量路径(
 from app.data.sources import bzzoiro

 monkeypatch.setenv("BZZOIRO_KEY", "test-key-abc")
 assert bzzoiro._key() == "test-key-abc"
 assert bzzoiro.available() is True


def test_bzzoiro_key_available_consistency(monkeypatch):
 """无 env 时 _key() 返回 str 且 available ≡ bool(key)(.env 兜底或空)。"""
 from app.data.sources import bzzoiro

 monkeypatch.delenv("BZZOIRO_KEY", raising=False)
 k = bzzoiro._key()
 assert isinstance(k, str)
 assert bzzoiro.available() is bool(k)
 assert bzzoiro._key() == k # 幂等
