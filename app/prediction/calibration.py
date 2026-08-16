"""Calibration 应用层(审查 §12/§14 拆分):raw → calibrated(有 .cal 时)。"""
from __future__ import annotations

import os


def apply(result: dict, models_dir: str, league_type) -> tuple[dict, dict | None, bool]:
    """应用校准器到 result(1X2 概率);返回 (result, cal_info, degraded)。"""
    cal_info = None
    degraded = False
    try:
        from app.calibration.calibrator import Calibrator
        _cal = Calibrator.load(os.path.join(models_dir, f"{league_type.value}_model.cal"))
        if _cal is not None:
            cp = _cal.apply({
                "home_win": result["home_win_probability"],
                "draw": result["draw_probability"],
                "away_win": result["away_win_probability"],
            })
            result["home_win_probability"] = cp["home_win"]
            result["draw_probability"] = cp["draw"]
            result["away_win_probability"] = cp["away_win"]
            result["calibration"] = cp.get("calibration")
            cal_info = {"n": cp.get("calibration"), "method": cp.get("cal_method", "beta")}
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("校准应用失败(降级为未校准概率): %s", e)
        degraded = True
    return result, cal_info, degraded
