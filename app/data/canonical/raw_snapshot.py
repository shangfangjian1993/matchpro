"""Raw Snapshot 数据湖:不可变原始数据落盘。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_ROOT = os.environ.get("RAW_SNAPSHOT_ROOT", "/opt/data/raw_snapshots")


def _snapshot_path(source: str, league: str, season: int, data_type: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    p = Path(RAW_ROOT) / source / league / str(season) / data_type / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_snapshot(source: str, league: str, season: int, data_type: str, 
                  data: Any, meta: dict | None = None) -> dict:
    path = _snapshot_path(source, league, season, data_type)
    if isinstance(data, (list, dict)):
        raw_bytes = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        data_file = path / "data.json"
    else:
        raw_bytes = data if isinstance(data, bytes) else str(data).encode("utf-8")
        data_file = path / "data.bin"
    data_file.write_bytes(raw_bytes)
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    metadata = {
        "source": source, "league": league, "season": season, "data_type": data_type,
        "download_time": datetime.now(timezone.utc).isoformat(),
        "checksum": checksum, "size_bytes": len(raw_bytes),
        **(meta or {}),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return {"path": str(path), "hash": checksum, "bytes": len(raw_bytes)}


def load_latest_snapshot(source: str, league: str, season: int, data_type: str) -> dict | None:
    base = Path(RAW_ROOT) / source / league / str(season) / data_type
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), key=lambda d: d.name, reverse=True)
    for d in dirs:
        data_file = d / "data.json"
        if not data_file.exists():
            data_file = d / "data.bin"
        meta_file = d / "metadata.json"
        if data_file.exists():
            result = {"data": None, "metadata": {}}
            if data_file.suffix == ".json":
                result["data"] = json.loads(data_file.read_text())
            else:
                result["data"] = data_file.read_bytes()
            if meta_file.exists():
                result["metadata"] = json.loads(meta_file.read_text())
            return result
    return None


__all__ = ["save_snapshot", "load_latest_snapshot", "RAW_ROOT"]
