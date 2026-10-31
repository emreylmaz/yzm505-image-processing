"""Karşılaştırma metrikleri ve metrics.json yazımı."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(_abs_diff(a, b).max())


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(_abs_diff(a, b).mean())


def _psnr_from_diff(diff: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean(diff**2))
    return math.inf if mse == 0 else 10.0 * math.log10(peak**2 / mse)


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """PSNR (dB). Görüntüler özdeşse `inf` döner (JSON'a "inf" string'i olarak yazılır)."""
    return _psnr_from_diff(_abs_diff(a, b), peak)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Standart karşılaştırma paketi: max/mean abs diff, PSNR, farklı piksel oranı."""
    diff = _abs_diff(a, b)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "psnr_db": _psnr_from_diff(diff),
        "pct_pixels_differ": float(np.mean(diff > 0) * 100.0),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return "inf" if obj > 0 else ("-inf" if obj < 0 else "nan")
    return obj


def write_metrics(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
