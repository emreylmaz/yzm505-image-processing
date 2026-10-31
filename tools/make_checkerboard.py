"""MATLAB `checkerboard()` (varsayılan n=10, p=q=4) eşdeğerini uint8 TIFF olarak üretir.

MATLAB'da checkerboard.tif hazır dosya değildir; `I = checkerboard; imwrite(I,'checkerboard.tif')`
ile elde edilir (ASSUMPTION). Algoritma (Image Processing Toolbox):
  tile = [black white; white black]  (n×n kareler) → repmat(tile, p, q)  → 80×80
  sağ yarıdaki beyaz kareler 0.7 gri olur; im2uint8: round(0.7·255) = 179 (half away from zero).

Kullanım: uv run python tools/make_checkerboard.py <çıkış.tif>
"""

from __future__ import annotations

import sys

import numpy as np
import tifffile


def matlab_checkerboard(n: int = 10, p: int = 4, q: int = 4) -> np.ndarray:
    black = np.zeros((n, n))
    white = np.ones((n, n))
    tile = np.block([[black, white], [white, black]])
    img = np.tile(tile, (p, q))
    half = img.shape[1] // 2
    img[:, half:] *= 0.7
    return np.floor(img * 255.0 + 0.5).astype(np.uint8)  # im2uint8 (pozitif → half-up)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("kullanım: make_checkerboard.py <çıkış.tif>")
    img = matlab_checkerboard()
    tifffile.imwrite(sys.argv[1], img)
    print(sys.argv[1], img.shape, img.dtype, sorted(np.unique(img).tolist()))


if __name__ == "__main__":
    main()
