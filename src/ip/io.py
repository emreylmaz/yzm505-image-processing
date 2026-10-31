"""Görüntü okuma/yazma yardımcıları.

Tüm modüller içeride float64 çalışır; uint8'e dönüşüm yalnızca en sonda `to_uint8` ile yapılır.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def require_data(name: str) -> Path:
    """data/<name> yolunu döndürür; dosya yoksa açıklayıcı hata verir."""
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"data/{name} bulunamadı — lütfen sınavın giriş görüntüsünü data/ klasörüne ekleyin."
        )
    return path


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float -> [0,255] clip -> en yakına yuvarla -> uint8. Pipeline'ın tek çıkış noktası."""
    return np.clip(np.rint(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Herhangi bir dtype'ı uint8 griye çevirir (ASSUMPTION 7).

    - uint8: olduğu gibi
    - bool: {0,255}
    - diğer tamsayı: dtype aralığından [0,255]'e lineer
    - float: [0,1] aralığındaysa x255, değilse min-max ile [0,255]
    """
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.copy()
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        f = (arr.astype(np.float64) - info.min) / (info.max - info.min) * 255.0
        return to_uint8(f)
    f = arr.astype(np.float64)
    lo, hi = float(f.min()), float(f.max())
    if lo >= 0.0 and hi <= 1.0:
        return to_uint8(f * 255.0)
    return to_uint8((f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f))


def load_gray(path: str | Path) -> np.ndarray:
    """.tif okur ve 2B uint8 gri görüntü döndürür (RGB ise ITU-R BT.601 luma)."""
    arr = tifffile.imread(str(path))
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        rgb = normalize_to_uint8(arr[..., :3]).astype(np.float64)
        return to_uint8(rgb @ np.array([0.299, 0.587, 0.114]))
    if arr.ndim != 2:
        raise ValueError(f"{path}: beklenmeyen şekil {arr.shape}")
    return normalize_to_uint8(arr)


def save_png(img: np.ndarray, path: str | Path) -> Path:
    """uint8 (veya float → to_uint8) görüntüyü PNG olarak kaydeder."""
    import cv2  # yalnızca dosya yazımı için; algoritmalarda kullanılmaz

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = img if img.dtype == np.uint8 else to_uint8(img)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"PNG yazılamadı: {path}")
    return path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
