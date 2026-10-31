"""Q1 — Kontrast iyileştirme: manuel Contrast Stretching ve Histogram Equalization.

Manual fonksiyonlar yalnızca NumPy dizi işlemleri kullanır (cv2/skimage yok) ve float64 döndürür.
`*_builtin` fonksiyonlar sadece karşılaştırma içindir.
"""

from __future__ import annotations

import numpy as np

from ip.io import to_uint8


def histogram(img_u8: np.ndarray) -> np.ndarray:
    """256 kutulu histogram: h[k] = #{piksel = k}."""
    return np.bincount(np.asarray(img_u8, dtype=np.uint8).ravel(), minlength=256)


def contrast_stretch(
    img: np.ndarray, low_pct: float | None = None, high_pct: float | None = None
) -> np.ndarray:
    """Lineer Contrast Stretching (min-max normalizasyonu).

    s = (r - r_min) / (r_max - r_min) · 255
    low_pct/high_pct verilirse r_min/r_max yüzdeliklerden alınır ve dışı clip'lenir
    (uç değerlere dayanıklı varyant). Varsayılan: gerçek min/max → çıktı tam [0,255].
    """
    f = np.asarray(img, dtype=np.float64)
    lo = f.min() if low_pct is None else np.percentile(f, low_pct)
    hi = f.max() if high_pct is None else np.percentile(f, high_pct)
    if hi <= lo:  # sabit görüntü: germe tanımsız
        return np.zeros_like(f)
    return np.clip((f - lo) / (hi - lo) * 255.0, 0.0, 255.0)


def equalization_lut(img_u8: np.ndarray) -> np.ndarray:
    """Histogram Equalization dönüşüm tablosu (LUT), float64, 256 eleman.

    CDF(k) = Σ_{j≤k} h[j];  T(k) = round( (CDF(k) - CDF_min) / (N - CDF_min) · 255 )
    CDF_min: ilk sıfır olmayan kutunun CDF'i → en koyu mevcut seviye 0'a, en parlak 255'e gider.
    """
    h = histogram(img_u8)
    cdf = np.cumsum(h).astype(np.float64)
    n = cdf[-1]
    cdf_min = cdf[np.flatnonzero(h)[0]]
    if n == cdf_min:  # tek seviyeli görüntü
        return np.arange(256, dtype=np.float64)
    lut = np.floor((cdf - cdf_min) / (n - cdf_min) * 255.0 + 0.5)
    return np.clip(lut, 0.0, 255.0)


def hist_equalize(img_u8: np.ndarray) -> np.ndarray:
    """Manuel Histogram Equalization: s = T(r) (LUT ile dizi indeksleme). float64 döner."""
    img_u8 = np.asarray(img_u8, dtype=np.uint8)
    return equalization_lut(img_u8)[img_u8]


def cdf_linearity_max_dev(img_u8: np.ndarray) -> float:
    """Normalize CDF'in ideal lineer CDF'ten (k/255) maksimum sapması (0 = tam düz histogram)."""
    h = histogram(img_u8)
    cdf = np.cumsum(h) / h.sum()
    levels = np.flatnonzero(h)
    ideal = np.arange(256) / 255.0
    return float(np.max(np.abs(cdf[levels] - ideal[levels])))


# ----------------------------------------------------------------------------- built-in
def equalize_builtin_cv2(img_u8: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.equalizeHist(img_u8)


def equalize_builtin_skimage(img_u8: np.ndarray) -> np.ndarray:
    """skimage.exposure.equalize_hist [0,1] float döner (CDF_min çıkarılmaz) → ×255."""
    from skimage import exposure

    return to_uint8(exposure.equalize_hist(img_u8, nbins=256) * 255)
