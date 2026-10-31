"""Q2 — Geometrik dönüşümler: inverse mapping ile manuel affine warp (Nearest Neighbor / Bilinear).

Koordinatlar: x = sütun, y = satır, piksel merkezleri tamsayı koordinatlarda (cv2.warpAffine ile aynı).
Forward affine A (2×3) giriş (x, y) → çıkış (x', y') eşler. Warp, her ÇIKIŞ pikseli için
A⁻¹ ile giriş koordinatını bulur (inverse mapping) — böylece çıkışta delik oluşmaz.
Görüntü dışına düşen örnekler `fill` (0) değerini alır (cv2 BORDER_CONSTANT gibi).
"""

from __future__ import annotations

import math

import numpy as np


def rotation_matrix(deg: float, shape: tuple[int, int], expand: bool = True):
    """Merkez etrafında `deg` derece döndürme (pozitif = ekranda saat yönünün tersi, cv2 kuralı).

    expand=True: çıkış tuvali döndürülmüş görüntünün tamamını içerecek şekilde büyütülür.
    Dönüş: (A 2×3, out_shape).
    """
    h, w = shape
    t = math.radians(deg)
    a, b = math.cos(t), math.sin(t)
    lin = np.array([[a, b], [-b, a]])
    if expand:
        w_out = int(round(w * abs(a) + h * abs(b)))
        h_out = int(round(w * abs(b) + h * abs(a)))
    else:
        h_out, w_out = h, w
    c_in = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    c_out = np.array([(w_out - 1) / 2.0, (h_out - 1) / 2.0])
    return np.hstack([lin, (c_out - lin @ c_in)[:, None]]), (h_out, w_out)


def shear_matrix(k: float, shape: tuple[int, int], axis: str = "x"):
    """Shear: axis='x' → x' = x + k·y ; axis='y' → y' = y + k·x. Tuval tüm sonucu içerir."""
    h, w = shape
    if axis == "x":
        lin = np.array([[1.0, k], [0.0, 1.0]])
        shift = np.array([max(0.0, -k * (h - 1)), 0.0])
        out = (h, w + int(math.ceil(abs(k) * (h - 1))))
    else:
        lin = np.array([[1.0, 0.0], [k, 1.0]])
        shift = np.array([0.0, max(0.0, -k * (w - 1))])
        out = (h + int(math.ceil(abs(k) * (w - 1))), w)
    return np.hstack([lin, shift[:, None]]), out


def _inverse(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lin_inv = np.linalg.inv(A[:, :2])
    return lin_inv, -lin_inv @ A[:, 2]


def source_coords(A: np.ndarray, out_shape: tuple[int, int], offset: tuple[float, float] = (0, 0)):
    """Her çıkış pikseli (+alt-piksel offset) için giriş koordinatları (sx, sy) — inverse mapping."""
    lin_inv, t_inv = _inverse(A)
    yy, xx = np.mgrid[0 : out_shape[0], 0 : out_shape[1]].astype(np.float64)
    xx += offset[0]
    yy += offset[1]
    sx = lin_inv[0, 0] * xx + lin_inv[0, 1] * yy + t_inv[0]
    sy = lin_inv[1, 0] * xx + lin_inv[1, 1] * yy + t_inv[1]
    return sx, sy


def _sample_nearest(img: np.ndarray, sx: np.ndarray, sy: np.ndarray, fill: float) -> np.ndarray:
    h, w = img.shape
    xi = np.floor(sx + 0.5).astype(np.intp)  # en yakın piksel merkezi
    yi = np.floor(sy + 0.5).astype(np.intp)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(sx.shape, fill, dtype=np.float64)
    out[valid] = img[yi[valid], xi[valid]]
    return out


def _sample_bilinear(img: np.ndarray, sx: np.ndarray, sy: np.ndarray, fill: float) -> np.ndarray:
    """4 komşu ağırlıklı ortalama; görüntü dışındaki komşular `fill` sayılır (BORDER_CONSTANT)."""
    h, w = img.shape
    p = np.pad(img, 1, mode="constant", constant_values=fill)  # p[y+1, x+1] = img[y, x]
    x0 = np.floor(sx)
    y0 = np.floor(sy)
    dx = sx - x0
    dy = sy - y0
    valid = (x0 >= -1) & (x0 <= w - 1) & (y0 >= -1) & (y0 <= h - 1)
    x0i = np.where(valid, x0, -1).astype(np.intp) + 1
    y0i = np.where(valid, y0, -1).astype(np.intp) + 1
    x1i = np.minimum(x0i + 1, w + 1)
    y1i = np.minimum(y0i + 1, h + 1)
    out = (
        (1 - dx) * (1 - dy) * p[y0i, x0i]
        + dx * (1 - dy) * p[y0i, x1i]
        + (1 - dx) * dy * p[y1i, x0i]
        + dx * dy * p[y1i, x1i]
    )
    return np.where(valid, out, fill)


def warp_affine(
    img: np.ndarray,
    A: np.ndarray,
    out_shape: tuple[int, int],
    interp: str = "bilinear",
    fill: float = 0.0,
) -> np.ndarray:
    """Manuel affine warp (inverse mapping). float64 döndürür."""
    f = np.asarray(img, dtype=np.float64)
    sx, sy = source_coords(A, out_shape)
    if interp == "nearest":
        return _sample_nearest(f, sx, sy, fill)
    if interp == "bilinear":
        return _sample_bilinear(f, sx, sy, fill)
    raise ValueError("interp 'nearest' veya 'bilinear' olmalı")


def warp_reference(
    img: np.ndarray, A: np.ndarray, out_shape: tuple[int, int], ss: int = 8, fill: float = 0.0
) -> np.ndarray:
    """Aliasing'siz referans: her çıkış pikselinde ss×ss alt-piksel NN örneğinin ortalaması (box filtre).

    Parça parça sabit (piecewise-constant) giriş görüntüsünün alan-ortalamalı ideal dönüşümüne yaklaşır.
    """
    f = np.asarray(img, dtype=np.float64)
    acc = np.zeros(out_shape, dtype=np.float64)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    for oy in offs:
        for ox in offs:
            sx, sy = source_coords(A, out_shape, (ox, oy))
            acc += _sample_nearest(f, sx, sy, fill)
    return acc / (ss * ss)


def valid_mask(A: np.ndarray, out_shape: tuple[int, int], in_shape: tuple[int, int]) -> np.ndarray:
    """Çıkışta giriş görüntüsünün içine düşen pikseller (NN anlamında)."""
    return warp_affine(np.ones(in_shape), A, out_shape, "nearest") > 0


def expected_gradient_angles(A: np.ndarray) -> tuple[float, float]:
    """Girişteki dikey/yatay kenarların gradyan yönleri (0°, 90°) dönüşüm sonrası nereye gider (°).

    Normal vektörler A⁻ᵀ ile dönüşür.
    """
    m = np.linalg.inv(A[:, :2]).T
    out = []
    for v in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        g = m @ v
        out.append(math.degrees(math.atan2(g[1], g[0])) % 180.0)
    return out[0], out[1]


# ----------------------------------------------------------------------------- built-in
def warp_builtin(img_u8: np.ndarray, A: np.ndarray, out_shape, interp: str = "bilinear"):
    import cv2

    flag = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR
    return cv2.warpAffine(
        img_u8,
        A.astype(np.float64),
        (out_shape[1], out_shape[0]),
        flags=flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
