"""Uzamsal filtreler: manual (NumPy) sürümler + `*_builtin` (OpenCV) karşılaştırma sarmalayıcıları.

Border (padding) modları — her fonksiyonda açık parametre:
  "reflect101" : np.pad(mode="reflect")   == cv2.BORDER_REFLECT_101 (cv2 varsayılanı)  ...|c b|a b c|b a|...
  "replicate"  : np.pad(mode="edge")      == cv2.BORDER_REPLICATE                       ...|a a|a b c|c c|...
  "zero"       : np.pad(mode="constant")  == cv2.BORDER_CONSTANT (0)
Manual fonksiyonlar float64 döndürür; uint8'e dönüşüm çağıranın işidir (ip.io.to_uint8).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_NP_MODES = {"reflect101": "reflect", "replicate": "edge", "zero": "constant"}

LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def pad(img: np.ndarray, r: int, mode: str = "reflect101") -> np.ndarray:
    """Her kenara r piksel padding."""
    if mode not in _NP_MODES:
        raise ValueError(f"border modu {list(_NP_MODES)} olmalı, '{mode}' verildi")
    return np.pad(np.asarray(img, dtype=np.float64), r, mode=_NP_MODES[mode])


def _windows(img: np.ndarray, k: int, mode: str) -> np.ndarray:
    if k % 2 == 0 or k < 1:
        raise ValueError("kernel boyutu tek ve pozitif olmalı")
    return sliding_window_view(pad(img, k // 2, mode), (k, k))


def correlate2d(img: np.ndarray, kernel: np.ndarray, border: str = "reflect101") -> np.ndarray:
    """Genel 2B korelasyon (cv2.filter2D gibi, kernel çevrilmez)."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kare kernel bekleniyor")
    win = _windows(img, kernel.shape[0], border)
    return np.einsum("ijkl,kl->ij", win, kernel)


# ----------------------------------------------------------------------------- manual
def mean_filter(img: np.ndarray, k: int = 3, border: str = "reflect101") -> np.ndarray:
    """k×k kutu (box/mean) filtresi."""
    return _windows(img, k, border).mean(axis=(-2, -1))


def median_filter(img: np.ndarray, k: int = 3, border: str = "replicate") -> np.ndarray:
    """k×k median filtresi. Varsayılan border 'replicate' (cv2.medianBlur ile aynı)."""
    return np.median(_windows(img, k, border), axis=(-2, -1))


def gaussian_ksize(sigma: float) -> int:
    """ksize = 2·ceil(3σ) + 1 (±3σ kapsar)."""
    return 2 * math.ceil(3 * sigma) + 1


def gaussian_kernel(sigma: float, ksize: int | None = None) -> np.ndarray:
    """Normalize edilmiş 2B Gaussian kernel: G(x,y) ∝ exp(-(x²+y²)/(2σ²))."""
    if sigma <= 0:
        raise ValueError("sigma > 0 olmalı")
    ksize = ksize or gaussian_ksize(sigma)
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def gaussian_filter(
    img: np.ndarray, sigma: float, ksize: int | None = None, border: str = "reflect101"
) -> np.ndarray:
    return correlate2d(img, gaussian_kernel(sigma, ksize), border)


# ----------------------------------------------------------------------------- Laplacian (Q3)
def laplacian_loops(
    img: np.ndarray, kernel: np.ndarray = LAPLACIAN_4, border: str = "reflect101"
) -> np.ndarray:
    """Laplacian (high-pass) — RAPOR SÜRÜMÜ, iç içe for döngüleri. float64 döndürür.

    L(y,x) = Σ_{i=-r..r} Σ_{j=-r..r} K(i+r, j+r) · P(y+r+i, x+r+j)
    P: border moduna göre padding uygulanmış görüntü. 4-komşu kernel için
    L = I(y-1,x) + I(y+1,x) + I(y,x-1) + I(y,x+1) - 4·I(y,x).
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    k = kernel.shape[0]
    r = k // 2
    padded = pad(img, r, border)  # float64 → uint8 taşması yok
    h, w = np.asarray(img).shape
    out = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for i in range(k):
                for j in range(k):
                    acc += kernel[i, j] * padded[y + i, x + j]
            out[y, x] = acc
    return out


def laplacian_vec(
    img: np.ndarray, kernel: np.ndarray = LAPLACIAN_4, border: str = "reflect101"
) -> np.ndarray:
    """`laplacian_loops` vektörize ikizi (aynı toplama sırası: önce satır i, sonra sütun j)."""
    kernel = np.asarray(kernel, dtype=np.float64)
    k = kernel.shape[0]
    padded = pad(img, k // 2, border)
    h, w = np.asarray(img).shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            out = out + kernel[i, j] * padded[i : i + h, j : j + w]
    return out


def sharpen(img: np.ndarray, lap: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Merkezi negatif kernel için keskinleştirme: S = I - c·L (float64)."""
    return np.asarray(img, dtype=np.float64) - c * lap


# ----------------------------------------------------------------------------- built-in
_CV_BORDERS = {
    "reflect101": "BORDER_REFLECT_101",
    "replicate": "BORDER_REPLICATE",
    "zero": "BORDER_CONSTANT",
}


def _cv_border(mode: str) -> int:
    import cv2

    return getattr(cv2, _CV_BORDERS[mode])


def mean_builtin(img_u8: np.ndarray, k: int = 3, border: str = "reflect101") -> np.ndarray:
    import cv2

    return cv2.blur(img_u8, (k, k), borderType=_cv_border(border))


def median_builtin(img_u8: np.ndarray, k: int = 3) -> np.ndarray:
    """cv2.medianBlur — border her zaman REPLICATE (değiştirilemez)."""
    import cv2

    return cv2.medianBlur(img_u8, k)


def gaussian_builtin(
    img_u8: np.ndarray, sigma: float, ksize: int | None = None, border: str = "reflect101"
) -> np.ndarray:
    import cv2

    ksize = ksize or gaussian_ksize(sigma)
    return cv2.GaussianBlur(
        img_u8, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=_cv_border(border)
    )


def laplacian_builtin(
    img: np.ndarray, kernel: np.ndarray = LAPLACIAN_4, border: str = "reflect101"
) -> np.ndarray:
    """cv2.filter2D float64'te, aynı kernel ve border."""
    import cv2

    return cv2.filter2D(
        np.asarray(img, dtype=np.float64), cv2.CV_64F, kernel, borderType=_cv_border(border)
    )


def laplacian_cv(img_u8: np.ndarray) -> np.ndarray:
    """cv2.Laplacian(ksize=1) → [[0,1,0],[1,-4,1],[0,1,0]], BORDER_REFLECT_101 (float64)."""
    import cv2

    return cv2.Laplacian(img_u8, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)


def laplacian_uint8_builtin(img_u8: np.ndarray) -> np.ndarray:
    """Taşma demosu: uint8 çıkışlı filter2D negatif değerleri 0'a doyurur (saturate)."""
    import cv2

    return cv2.filter2D(img_u8, -1, LAPLACIAN_4, borderType=cv2.BORDER_REFLECT_101)
