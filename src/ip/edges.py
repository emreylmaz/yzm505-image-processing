"""Kenar operatörleri (analiz aracı; cv2 kullanılabilir) ve keskinlik/gürültü metrikleri.

Q1 için SEÇTİĞİM keskinlik metriği (raporda açıkça etiketlenir):
  edge_sharpness = kenar maskesi içindeki ortalama Sobel gradyan büyüklüğü
  maske = NESNE kenarları: referans görüntü önce Gaussian σ=2 ile yumuşatılır (gürültü kenar sayılmasın),
          sonra Canny + 3×3 dilate. Tüm filtreler için AYNI maske kullanılır.
İkincil: variance of Laplacian. Gürültü göstergesi: düz bölgede Laplacian yanıtının std'si.
"""

from __future__ import annotations

import numpy as np

CANNY_LO, CANNY_HI = 50, 150


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gx, gy (ksize=3, float64, BORDER_REFLECT_101)."""
    import cv2

    f = np.asarray(img, dtype=np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return gx, gy


def sobel_mag(img: np.ndarray) -> np.ndarray:
    gx, gy = sobel(img)
    return np.hypot(gx, gy)


def sobel_display(mag: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Sobel büyüklüğünü görüntüleme için [0,255]'e ölçekler (karşılaştırmalarda ortak vmax kullan)."""
    vmax = vmax or float(mag.max()) or 1.0
    return np.clip(np.rint(mag / vmax * 255), 0, 255).astype(np.uint8)


def canny(img_u8: np.ndarray, lo: int = CANNY_LO, hi: int = CANNY_HI) -> np.ndarray:
    """cv2.Canny (L1 gradyan, aperture 3). Çıktı bool."""
    import cv2

    return cv2.Canny(img_u8, lo, hi) > 0


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """(2r+1)×(2r+1) kare yapı elemanıyla ikili genişletme (NumPy)."""
    p = np.pad(mask.astype(bool), r)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy : dy + h, dx : dx + w]
    return out


MASK_SIGMA = 2.0


def _smoothed(ref_u8: np.ndarray, sigma: float) -> np.ndarray:
    from ip.filters import gaussian_filter
    from ip.io import to_uint8

    return to_uint8(gaussian_filter(ref_u8, sigma))


def edge_mask(ref_u8: np.ndarray, r: int = 1, sigma: float = MASK_SIGMA) -> np.ndarray:
    """Nesne kenar maskesi: Canny(Gaussian_σ(ref)) + (2r+1)² dilate. sigma=0 → yumuşatmasız."""
    base = _smoothed(ref_u8, sigma) if sigma > 0 else ref_u8
    return dilate(canny(base), r)


def flat_mask(
    ref_u8: np.ndarray, pct: float = 40.0, r: int = 3, sigma: float = MASK_SIGMA
) -> np.ndarray:
    """Düz bölge: yumuşatılmış referansta Sobel alt %pct'de VE nesne kenarlarından r px uzakta."""
    base = _smoothed(ref_u8, sigma) if sigma > 0 else ref_u8
    mag = sobel_mag(base)
    return (mag <= np.percentile(mag, pct)) & ~dilate(canny(base), r)


def _laplacian(img: np.ndarray) -> np.ndarray:
    from ip.filters import laplacian_vec

    return laplacian_vec(img)


def edge_sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    """SEÇİLMİŞ METRİK: maske içindeki ortalama Sobel gradyan büyüklüğü."""
    return float(sobel_mag(img)[mask].mean())


def laplacian_variance(img: np.ndarray) -> float:
    return float(_laplacian(img).var())


def flat_noise(img: np.ndarray, mask: np.ndarray) -> float:
    """Düz bölgede Laplacian yanıtının std'si (lineer gölgelemeyi yok sayar, yüksek frekans gürültüyü ölçer)."""
    return float(_laplacian(img)[mask].std())


def best_edge_roi(mask: np.ndarray, size: int = 40) -> tuple[int, int, int, int]:
    """Kenar piksel yoğunluğu en yüksek size×size pencere (deterministik; integral image ile)."""
    ii = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    return int(y), int(x), size, size


def orientation_stats(
    img: np.ndarray, expected_deg: tuple[float, ...], mag_pct: float = 90.0
) -> dict:
    """Güçlü gradyanlı piksellerde (mag ≥ pct yüzdelik) gradyan yönünün beklenen yönlerden sapması.

    Yön 180° periyotlu ele alınır. Dönen değerler: ortalama mutlak sapma (°), ±10°'nin dışındaki
    güçlü piksellerin yüzdesi (yanlış yönlü / "basamak" kenar bileşenleri).
    """
    gx, gy = sobel(img)
    mag = np.hypot(gx, gy)
    strong = mag >= np.percentile(mag[mag > 0], mag_pct) if np.any(mag > 0) else mag > 0
    ang = np.degrees(np.arctan2(gy[strong], gx[strong])) % 180.0
    exp = np.array(expected_deg, dtype=np.float64) % 180.0
    d = np.abs(ang[:, None] - exp[None, :])
    d = np.minimum(d, 180.0 - d).min(axis=1)
    return {
        "n_strong": int(strong.sum()),
        "mean_abs_dev_deg": float(d.mean()),
        "pct_off_axis_gt10deg": float(np.mean(d > 10.0) * 100.0),
    }
