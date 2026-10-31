"""Q3 — Sıfırdan Bilinear interpolation ile yeniden boyutlandırma.

`bilinear_resize_loops` raporda gösterilen sürümdür: açık iç içe for döngüleri ve 4 komşu
için ayrı ayrı yazılmış ağırlıklar. `bilinear_resize_vec` hız için vektörize ikizdir ve
testlerde loop sürümüne birebir (bit-exact) eşit olduğu doğrulanır.
"""

from __future__ import annotations

import math

import numpy as np


def output_shape(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    """Çıkış boyutu = round(boyut * scale) (half-up; ASSUMPTION 6)."""
    h, w = shape
    return math.floor(h * scale + 0.5), math.floor(w * scale + 0.5)


def bilinear_resize_loops(
    img: np.ndarray, scale: float = 1.7, convention: str = "half_pixel"
) -> np.ndarray:
    """Bilinear resize — RAPOR SÜRÜMÜ (iç içe for döngüleri). float64 döndürür.

    Koordinat eşlemesi (çıkış -> giriş, inverse mapping), oran = giriş/çıkış:
      half_pixel : src = (dst + 0.5) * oran - 0.5   (piksel merkezleri hizalı; OpenCV kuralı)
      naive      : src = dst * oran                 (piksel köşeleri hizalı)
    src [0, N-1] aralığına sıkıştırılır (kenar = replicate).

    Dört komşu ağırlığı (dx, dy ∈ [0,1)):
      out = (1-dx)(1-dy)·I(x0,y0) + dx(1-dy)·I(x1,y0) + (1-dx)dy·I(x0,y1) + dx·dy·I(x1,y1)
    """
    if convention not in ("half_pixel", "naive"):
        raise ValueError("convention 'half_pixel' veya 'naive' olmalı")
    src = np.asarray(img, dtype=np.float64)
    h_in, w_in = src.shape
    h_out, w_out = output_shape((h_in, w_in), scale)
    ratio_y = h_in / h_out
    ratio_x = w_in / w_out
    out = np.zeros((h_out, w_out), dtype=np.float64)

    for y in range(h_out):
        # --- dikey kaynak koordinatı ---
        if convention == "half_pixel":
            sy = (y + 0.5) * ratio_y - 0.5
        else:
            sy = y * ratio_y
        sy = min(max(sy, 0.0), h_in - 1.0)
        y0 = int(math.floor(sy))
        y1 = min(y0 + 1, h_in - 1)
        dy = sy - y0

        for x in range(w_out):
            # --- yatay kaynak koordinatı ---
            if convention == "half_pixel":
                sx = (x + 0.5) * ratio_x - 0.5
            else:
                sx = x * ratio_x
            sx = min(max(sx, 0.0), w_in - 1.0)
            x0 = int(math.floor(sx))
            x1 = min(x0 + 1, w_in - 1)
            dx = sx - x0

            # --- dört komşu ve ağırlıkları ---
            i00 = src[y0, x0]  # sol-üst    I(x0, y0)
            i10 = src[y0, x1]  # sağ-üst    I(x1, y0)
            i01 = src[y1, x0]  # sol-alt    I(x0, y1)
            i11 = src[y1, x1]  # sağ-alt    I(x1, y1)

            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy

            out[y, x] = w00 * i00 + w10 * i10 + w01 * i01 + w11 * i11

    return out


def _src_coords(n_in: int, n_out: int, convention: str):
    d = np.arange(n_out, dtype=np.float64)
    ratio = n_in / n_out
    s = (d + 0.5) * ratio - 0.5 if convention == "half_pixel" else d * ratio
    s = np.minimum(np.maximum(s, 0.0), n_in - 1.0)
    i0 = np.floor(s).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, s - i0


def bilinear_resize_vec(
    img: np.ndarray, scale: float = 1.7, convention: str = "half_pixel"
) -> np.ndarray:
    """`bilinear_resize_loops` ile birebir aynı aritmetik, NumPy broadcasting ile."""
    if convention not in ("half_pixel", "naive"):
        raise ValueError("convention 'half_pixel' veya 'naive' olmalı")
    src = np.asarray(img, dtype=np.float64)
    h_out, w_out = output_shape(src.shape, scale)
    y0, y1, dy = _src_coords(src.shape[0], h_out, convention)
    x0, x1, dx = _src_coords(src.shape[1], w_out, convention)
    dy = dy[:, None]
    dx = dx[None, :]
    i00 = src[y0[:, None], x0[None, :]]
    i10 = src[y0[:, None], x1[None, :]]
    i01 = src[y1[:, None], x0[None, :]]
    i11 = src[y1[:, None], x1[None, :]]
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy
    return w00 * i00 + w10 * i10 + w01 * i01 + w11 * i11


def resize_builtin(img_u8: np.ndarray, scale: float = 1.7) -> np.ndarray:
    """Karşılaştırma: cv2.resize INTER_LINEAR, aynı çıkış boyutu (dsize ile)."""
    import cv2

    h_out, w_out = output_shape(img_u8.shape, scale)
    return cv2.resize(img_u8, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
