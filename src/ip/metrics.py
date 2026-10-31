"""Karşılaştırma metrikleri ve metrics.json yazımı."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """PSNR (dB). Görüntüler özdeşse `inf` döner (JSON'a "inf" string'i olarak yazılır)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(peak**2 / mse)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Standart karşılaştırma paketi: max/mean abs diff, PSNR, farklı piksel oranı."""
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "psnr_db": psnr(a, b),
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
