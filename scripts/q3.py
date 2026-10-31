"""Q3 — Manuel Bilinear resize (×1.7) + nested-loop Laplacian sharpening + built-in karşılaştırma.

Çıktılar: outputs/q3/*.png, outputs/q3/metrics.json
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from ip import filters as F
from ip import viz
from ip.io import OUTPUT_DIR, load_gray, require_data, save_png, to_uint8
from ip.metrics import compare, max_abs, mean_abs, write_metrics
from ip.resize import (
    bilinear_resize_loops,
    bilinear_resize_vec,
    output_shape,
    resize_builtin,
)

SCALE = 1.7
OUT = OUTPUT_DIR / "q3"
# Zoom ROI'leri (y, x, h, w) — orijinal görüntü koordinatlarında, sabit.
ROI_ORIG = (40, 100, 40, 40)  # kameramanın başı/kamera bölgesi
EXAMPLE_PIXEL = (100, 137)  # açık ağırlık hesabının raporlanacağı çıkış pikseli (y, x)


def worked_example(img: np.ndarray, y: int, x: int) -> dict:
    """Tek bir çıkış pikseli için koordinat eşlemesi ve 4 ağırlığın açık dökümü (rapor için)."""
    h_in, w_in = img.shape
    h_out, w_out = output_shape(img.shape, SCALE)
    ry, rx = h_in / h_out, w_in / w_out
    sy = min(max((y + 0.5) * ry - 0.5, 0.0), h_in - 1.0)
    sx = min(max((x + 0.5) * rx - 0.5, 0.0), w_in - 1.0)
    y0, x0 = math.floor(sy), math.floor(sx)
    y1, x1 = min(y0 + 1, h_in - 1), min(x0 + 1, w_in - 1)
    dy, dx = sy - y0, sx - x0
    nb = {
        "I(x0,y0)": int(img[y0, x0]),
        "I(x1,y0)": int(img[y0, x1]),
        "I(x0,y1)": int(img[y1, x0]),
        "I(x1,y1)": int(img[y1, x1]),
    }
    w = {
        "(1-dx)(1-dy)": (1 - dx) * (1 - dy),
        "dx(1-dy)": dx * (1 - dy),
        "(1-dx)dy": (1 - dx) * dy,
        "dx*dy": dx * dy,
    }
    val = sum(wv * nv for wv, nv in zip(w.values(), nb.values(), strict=True))
    return {
        "out_pixel_yx": [y, x],
        "ratio_y": ry,
        "ratio_x": rx,
        "src_y": sy,
        "src_x": sx,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "dx": dx,
        "dy": dy,
        "neighbors": nb,
        "weights": w,
        "weights_sum": sum(w.values()),
        "value_float": val,
        "value_uint8": int(to_uint8(np.array(val))),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1254 konsolu
    img = load_gray(require_data("cameraman.tif"))
    save_png(img, OUT / "original.png")
    m: dict = {"input": {"file": "data/cameraman.tif", "shape": list(img.shape)}, "scale": SCALE}

    # ------------------------------------------------------------------ 1) Bilinear resize
    t0 = time.perf_counter()
    up_loops = bilinear_resize_loops(img, SCALE, "half_pixel")  # RAPOR SÜRÜMÜ
    t_loops = time.perf_counter() - t0
    up_vec = bilinear_resize_vec(img, SCALE, "half_pixel")
    up_naive = bilinear_resize_vec(img, SCALE, "naive")
    up_cv = resize_builtin(img, SCALE)

    man_u8 = to_uint8(up_loops)
    naive_u8 = to_uint8(up_naive)
    h_out, w_out = up_loops.shape
    s_half = man_u8.astype(float) - up_cv  # işaretli uint8 farkları
    s_naive = naive_u8.astype(float) - up_cv
    d_half, dn = np.abs(s_half), np.abs(s_naive)
    m["resize"] = {
        "out_shape": [h_out, w_out],
        "ratio_x": img.shape[1] / w_out,
        "ratio_y": img.shape[0] / h_out,
        "loops_equals_vec_exact": bool(np.array_equal(up_loops, up_vec)),
        "loops_seconds": t_loops,
        "half_pixel_vs_cv2": compare(man_u8, up_cv),
        "naive_vs_cv2": compare(naive_u8, up_cv),
        "half_pixel_mean_signed_diff": float(np.mean(s_half)),
        "naive_mean_signed_diff": float(np.mean(s_naive)),
        "half_pixel_float_vs_cv2": {
            "max_abs_diff": max_abs(up_loops, up_cv),
            "mean_abs_diff": mean_abs(up_loops, up_cv),
        },
        "worked_example": worked_example(img, *EXAMPLE_PIXEL),
    }
    # naive farkın uzamsal dağılımı: kayma sağa/aşağı büyür → sağ-alt çeyrek vs sol-üst çeyrek
    q = h_out // 4
    m["resize"]["naive_diff_mean_topleft_quarter"] = float(dn[:q, :q].mean())
    m["resize"]["naive_diff_mean_bottomright_quarter"] = float(dn[-q:, -q:].mean())

    save_png(man_u8, OUT / "resized_manual.png")
    save_png(up_cv, OUT / "resized_cv2.png")
    save_png(naive_u8, OUT / "resized_manual_naive.png")
    viz.grid(
        [img, man_u8, up_cv, naive_u8],
        [
            f"Orijinal {img.shape[1]}×{img.shape[0]}",
            f"Manual Bilinear (half-pixel) {w_out}×{h_out}",
            "cv2.resize INTER_LINEAR",
            "Manual Bilinear (naive src=dst/s)",
        ],
        OUT / "fig_resize_compare.png",
        ncols=4,
        cell=3.4,
    )
    viz.heatmaps(
        [d_half, dn],
        [
            f"|manual(half-pixel) − cv2|  max={d_half.max():.0f}",
            f"|manual(naive) − cv2|  max={dn.max():.0f}",
        ],
        OUT / "fig_resize_diff_heatmap.png",
        suptitle="Q3 — Resize fark haritaları (uint8)",
    )
    viz.heatmap(
        d_half,
        OUT / "fig_resize_diff_half_pixel.png",
        "|manual(half-pixel) − cv2.resize| (0/1 ölçek)",
        cmap="gray",
        vmax=1,
    )
    # Zoom: aynı sahne bölgesi; çıkış ROI = orijinal ROI × ölçek
    y, x, hh, ww = ROI_ORIG
    roi_up = (round(y * SCALE), round(x * SCALE), round(hh * SCALE), round(ww * SCALE))
    viz.grid(
        [
            viz.draw_rois(img, [ROI_ORIG]),
            viz.zoom(img, ROI_ORIG, 6),
            viz.zoom(man_u8, roi_up, 4),
            viz.zoom(up_cv, roi_up, 4),
        ],
        [
            f"ROI (y,x,h,w)={ROI_ORIG}",
            "Orijinal ROI (NEAREST ×6)",
            "Manual ROI (NEAREST ×4)",
            "cv2 ROI (NEAREST ×4)",
        ],
        OUT / "fig_resize_zoom.png",
        ncols=4,
        cell=3.4,
    )

    # ------------------------------------------------------------------ 2) Laplacian sharpening
    t0 = time.perf_counter()
    lap = F.laplacian_loops(man_u8)  # RAPOR SÜRÜMÜ (float64, reflect101)
    t_lap = time.perf_counter() - t0
    lap_f2d = F.laplacian_builtin(man_u8)
    lap_cvl = F.laplacian_cv(man_u8)
    sharp_f = F.sharpen(man_u8, lap)  # float64, clip öncesi
    sharp_man = to_uint8(sharp_f)
    sharp_same_input_builtin = to_uint8(F.sharpen(man_u8, lap_f2d))
    # Tam built-in pipeline: cv2.resize → cv2.Laplacian → sharpen
    sharp_cv_pipeline = to_uint8(F.sharpen(up_cv, F.laplacian_cv(up_cv)))

    m["laplacian"] = {
        "kernel": F.LAPLACIAN_4,
        "border": "reflect101 (== cv2.BORDER_REFLECT_101)",
        "loops_seconds": t_lap,
        "loops_equals_vec_exact": bool(np.array_equal(lap, F.laplacian_vec(man_u8))),
        "loops_vs_filter2d_float": compare(lap, lap_f2d),
        "loops_vs_cv2_laplacian_ksize1": compare(lap, lap_cvl),
        "range": [float(lap.min()), float(lap.max())],
        "pct_negative": float(np.mean(lap < 0) * 100),
    }
    m["sharpen"] = {
        "formula": "S = I - c*L, c=1",
        "manual_vs_builtin_same_input": compare(sharp_man, sharp_same_input_builtin),
        "manual_pipeline_vs_cv2_pipeline": compare(sharp_man, sharp_cv_pipeline),
        "pct_clipped_low": float(np.mean(sharp_f < 0) * 100),
        "pct_clipped_high": float(np.mean(sharp_f > 255) * 100),
    }
    save_png(sharp_man, OUT / "sharpened_manual.png")
    save_png(sharp_cv_pipeline, OUT / "sharpened_cv2_pipeline.png")
    lap_vis = to_uint8(np.abs(lap) / np.abs(lap).max() * 255)
    viz.grid(
        [man_u8, lap_vis, sharp_man, sharp_cv_pipeline],
        [
            "Resize (manual)",
            "|Laplacian| (nested loops, normalize)",
            "Sharpened manual: I − L",
            "Sharpened built-in pipeline",
        ],
        OUT / "fig_laplacian_sharpen.png",
        ncols=4,
        cell=3.4,
    )
    viz.grid(
        [viz.zoom(man_u8, roi_up, 4), viz.zoom(sharp_man, roi_up, 4)],
        ["Resize ROI (×4)", "Sharpened ROI (×4)"],
        OUT / "fig_sharpen_zoom.png",
        ncols=2,
        cell=3.6,
    )
    d_pipe = np.abs(sharp_man.astype(float) - sharp_cv_pipeline)
    d_lap = np.abs(lap - lap_f2d)
    viz.heatmaps(
        [d_lap, d_pipe],
        [
            f"|L_loops − L_filter2D| (float)  max={d_lap.max():g}",
            f"|S_manual − S_cv2 pipeline|  max={d_pipe.max():.0f}",
        ],
        OUT / "fig_laplacian_diff_heatmap.png",
        suptitle="Q3 — Laplacian / sharpening fark haritaları",
    )

    # ------------------------------------------------------------------ 3) uint8 overflow demosu
    sat = F.laplacian_uint8_builtin(man_u8)  # cv2 uint8: negatifler 0'a doyar
    u = man_u8  # NumPy uint8 aritmetiği: mod 256 sarma (wrap-around)
    p = np.pad(u, 1, mode="reflect")
    wrap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - np.uint8(4) * u).astype(
        np.uint8
    )
    sharp_sat = to_uint8(man_u8.astype(float) - sat)
    m["overflow"] = {
        "uint8_saturated_vs_float_laplacian": compare(sat, lap),
        "uint8_wrapped_vs_float_laplacian": compare(wrap, lap),
        "sharpen_with_saturated_vs_float": compare(sharp_sat, sharp_man),
    }
    viz.grid(
        [
            to_uint8(np.clip(lap, -128, 127) + 128),
            sat,
            wrap,
            sharp_sat,
        ],
        [
            "Float L (+128 ofset, doğru)",
            "cv2.filter2D uint8 (negatif→0)",
            "NumPy uint8 (mod 256 wrap)",
            "I − L_sat (yanlış keskinleştirme)",
        ],
        OUT / "fig_overflow.png",
        ncols=4,
        cell=3.4,
    )

    # ------------------------------------------------------------------ 4) Border modu etkisi
    d_border = {
        mode: np.abs(F.laplacian_vec(man_u8, border=mode) - lap) for mode in ("replicate", "zero")
    }
    m["border_modes"] = {
        f"reflect101_vs_{mode}": {
            "max_abs_diff": float(d.max()),
            "n_pixels_differ": int(np.count_nonzero(d)),
            "n_inner_pixels_differ": int(np.count_nonzero(d[1:-1, 1:-1])),
        }
        for mode, d in d_border.items()
    }
    viz.heatmap(
        d_border["zero"][:60, :60],
        OUT / "fig_border_zero_vs_reflect.png",
        "|L(zero pad) − L(reflect101)| sol-üst 60×60",
    )

    write_metrics(m, OUT / "metrics.json")
    r = m["resize"]
    print(
        f"[q3] resize {img.shape}->{tuple(r['out_shape'])}  loops==vec:{r['loops_equals_vec_exact']}  "
        f"half_pixel vs cv2 max={r['half_pixel_vs_cv2']['max_abs_diff']:.0f} "
        f"psnr={r['half_pixel_vs_cv2']['psnr_db']:.2f}  naive max={r['naive_vs_cv2']['max_abs_diff']:.0f}\n"
        f"[q3] laplacian loops vs filter2D max={m['laplacian']['loops_vs_filter2d_float']['max_abs_diff']}"
    )


if __name__ == "__main__":
    main()
