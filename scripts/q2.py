"""Q2 — Shearing + Rotation (30°, NN ve Bilinear), Sobel & Canny, NN aliasing → false edges.

Çıktılar: outputs/q2/*.png, outputs/q2/metrics.json
"""

from __future__ import annotations

import sys

import numpy as np

from ip import edges as E
from ip import transforms as T
from ip import viz
from ip.io import OUTPUT_DIR, load_gray, require_data, save_png, to_uint8
from ip.metrics import compare, write_metrics

OUT = OUTPUT_DIR / "q2"
ANGLE = 30.0
SHEAR_K = 0.3  # ASSUMPTION 4: yatay shear katsayısı
SS = 8  # referans için alt-piksel örnekleme (8×8)
ZOOM = 8
# Sabit zoom ROI'leri (y, x, h, w) — döndürülmüş / shear'lı görüntü koordinatlarında.
ROI_ROT = [(20, 40, 24, 24), (56, 56, 24, 24)]
ROI_SHEAR = [(28, 30, 24, 24)]


def edge_analysis(img_u8, ref_u8, inner, expected_deg):
    """Bir görüntü için Sobel/Canny metrikleri; referansa (aliasing'siz) göre false edge ölçümü."""
    mag = E.sobel_mag(img_u8)
    ref_mag = E.sobel_mag(ref_u8)
    can = E.canny(img_u8) & inner
    ref_can = E.canny(ref_u8) & inner
    near_ref = E.dilate(ref_can, 1)
    near_me = E.dilate(can, 1)
    excess = np.clip(mag - ref_mag, 0, None)  # referansta olmayan FAZLA Sobel yanıtı = false edge enerjisi
    return {
        "canny_edge_pixels": int(can.sum()),
        "reference_canny_edge_pixels": int(ref_can.sum()),
        "canny_exact_mismatch_pct": float(np.mean(~ref_can[can]) * 100) if can.any() else 0.0,
        "canny_false_pct_1px": float(np.mean(~near_ref[can]) * 100) if can.any() else 0.0,
        "canny_missed_pct_1px": float(np.mean(~near_me[ref_can]) * 100) if ref_can.any() else 0.0,
        "sobel_mae_vs_reference": float(np.abs(mag - ref_mag)[inner].mean()),
        "sobel_excess_energy_pct": float(excess[inner].sum() / ref_mag[inner].sum() * 100)
        if ref_mag[inner].sum()
        else 0.0,
        "orientation": E.orientation_stats(np.where(inner, img_u8, 0), expected_deg)
        if expected_deg
        else None,
    }


def canny_overlay(can: np.ndarray, ref_can: np.ndarray) -> np.ndarray:
    """RGB: yeşil = referansla uyumlu kenar, kırmızı = false edge (referanstan >1 px uzak), mavi = kaçırılan."""
    near_ref = E.dilate(ref_can, 1)
    near_me = E.dilate(can, 1)
    rgb = np.zeros((*can.shape, 3), np.uint8)
    rgb[can & near_ref] = (0, 200, 0)
    rgb[can & ~near_ref] = (255, 0, 0)
    rgb[ref_can & ~near_me] = (60, 120, 255)
    return rgb


def check_roi(roi, shape):
    y, x, h, w = roi
    if y + h > shape[0] or x + w > shape[1]:
        raise ValueError(f"ROI {roi} görüntü {shape} dışına taşıyor — sabit ROI'leri güncelleyin")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1254 konsolu
    img = load_gray(require_data("checkerboard.tif"))
    m: dict = {
        "input": {
            "file": "data/checkerboard.tif",
            "shape": list(img.shape),
            "levels": sorted(int(v) for v in np.unique(img))[:10],
        },
        "angle_deg": ANGLE,
        "shear_k": SHEAR_K,
        "canny_thresholds": [E.CANNY_LO, E.CANNY_HI],
        "reference": f"{SS}×{SS} alt-piksel NN ortalaması (aliasing'siz box-filtre referansı)",
    }
    save_png(img, OUT / "original.png")

    A_rot, s_rot = T.rotation_matrix(ANGLE, img.shape, expand=True)
    A_sh, s_sh = T.shear_matrix(SHEAR_K, img.shape)
    A_id, s_id = T.rotation_matrix(0.0, img.shape)
    for r in ROI_ROT:
        check_roi(r, s_rot)
    for r in ROI_SHEAR:
        check_roi(r, s_sh)

    cases = {
        "original": (A_id, s_id, "nearest"),
        "shear_nn": (A_sh, s_sh, "nearest"),
        "shear_bilinear": (A_sh, s_sh, "bilinear"),
        "rot30_nn": (A_rot, s_rot, "nearest"),
        "rot30_bilinear": (A_rot, s_rot, "bilinear"),
    }
    imgs, refs, inners = {}, {}, {}
    m["cases"] = {}
    for name, (A, shape, interp) in cases.items():
        out = to_uint8(T.warp_affine(img, A, shape, interp))
        ref = to_uint8(T.warp_reference(img, A, shape, SS))
        valid = T.valid_mask(A, shape, img.shape)
        inner = ~E.dilate(~valid, 3)  # tuval sınırı (görüntü/arka plan kenarı) analiz dışı
        imgs[name], refs[name], inners[name] = out, ref, inner
        save_png(out, OUT / f"{name}.png")
        m["cases"][name] = {
            "shape": list(shape),
            "interp": interp,
            "matrix": A,
            "expected_gradient_deg": T.expected_gradient_angles(A),
            "manual_vs_cv2_warpAffine": compare(out, T.warp_builtin(img, A, shape, interp)),
            "vs_reference": compare(out[inner], ref[inner]),
        }

    vmax = max(float(E.sobel_mag(v).max()) for v in imgs.values())
    m["sobel_display_vmax"] = vmax
    for name, (A, *_r) in cases.items():
        m["cases"][name]["edges"] = edge_analysis(
            imgs[name], refs[name], inners[name], T.expected_gradient_angles(A)
        )
    save_png(refs["rot30_nn"], OUT / "rot30_reference.png")

    # ------------------------------------------------------------ figürler
    order = list(cases)
    titles = {
        "original": "Orijinal",
        "shear_nn": f"Shear k={SHEAR_K} (NN)",
        "shear_bilinear": f"Shear k={SHEAR_K} (Bilinear)",
        "rot30_nn": "Rotation 30° (NN)",
        "rot30_bilinear": "Rotation 30° (Bilinear)",
    }
    viz.grid(
        [imgs[k] for k in order] + [refs["rot30_nn"]],
        [titles[k] for k in order] + [f"Rotation 30° referans ({SS}×{SS} süper-örnekleme)"],
        OUT / "fig_transforms.png",
        ncols=3,
        cell=3.4,
    )
    viz.grid(
        [E.sobel_display(E.sobel_mag(imgs[k]), vmax) for k in order],
        [f"Sobel |∇| — {titles[k]}" for k in order],
        OUT / "fig_sobel.png",
        ncols=3,
        cell=3.4,
        suptitle=f"Sobel büyüklüğü (ortak ölçek, vmax={vmax:.0f})",
    )
    viz.grid(
        [(E.canny(imgs[k]) * 255).astype(np.uint8) for k in order],
        [f"Canny({E.CANNY_LO},{E.CANNY_HI}) — {titles[k]}" for k in order],
        OUT / "fig_canny.png",
        ncols=3,
        cell=3.4,
    )

    # Zoom: NN vs Bilinear vs referans × {görüntü, Sobel, Canny (renkli)}
    for tag, rois, nn_k, bl_k in (
        ("rot30", ROI_ROT, "rot30_nn", "rot30_bilinear"),
        ("shear", ROI_SHEAR, "shear_nn", "shear_bilinear"),
    ):
        ref = refs[nn_k]
        ref_can = E.canny(ref) & inners[nn_k]
        for i, roi in enumerate(rois):
            panels, labels = [], []
            for lab, im in (("NN", imgs[nn_k]), ("Bilinear", imgs[bl_k]), ("Referans", ref)):
                can = E.canny(im) & inners[nn_k]
                panels += [
                    viz.zoom(im, roi, ZOOM),
                    viz.zoom(E.sobel_display(E.sobel_mag(im), vmax), roi, ZOOM),
                    viz.zoom(canny_overlay(can, ref_can), roi, ZOOM),
                ]
                labels += [f"{lab}: görüntü", f"{lab}: Sobel |∇|", f"{lab}: Canny (kırmızı=false)"]
            viz.grid(
                panels,
                labels,
                OUT / f"fig_zoom_{tag}_roi{i + 1}.png",
                ncols=3,
                cell=3.0,
                suptitle=f"{titles[nn_k].split(' (')[0]} — ROI {roi} (NEAREST ×{ZOOM})",
            )
        viz.grid(
            [viz.draw_rois(imgs[nn_k], rois)],
            [f"{tag} ROI konumları"],
            OUT / f"fig_roi_map_{tag}.png",
            ncols=1,
            cell=4.0,
        )

    # Gradyan yönü histogramı (güçlü pikseller, büyüklük ağırlıklı)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (nn_k, bl_k) in zip(
        axes, (("rot30_nn", "rot30_bilinear"), ("shear_nn", "shear_bilinear")), strict=True
    ):
        exp = T.expected_gradient_angles(cases[nn_k][0])
        for k, st in ((nn_k, "-"), (bl_k, "--"), ("ref", ":")):
            im = refs[nn_k] if k == "ref" else imgs[k]
            gx, gy = E.sobel(im)
            mag = np.hypot(gx, gy) * inners[nn_k]
            ang = np.degrees(np.arctan2(gy, gx)) % 180
            h, edges_ = np.histogram(ang, bins=90, range=(0, 180), weights=mag)
            ax.plot(edges_[:-1] + 1, h / h.sum(), st, label="Referans" if k == "ref" else titles[k])
        for e in exp:
            ax.axvline(e, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel("gradyan yönü (°, mod 180)")
        ax.set_title(f"beklenen yönler: {exp[0]:.1f}°, {exp[1]:.1f}°", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Sobel gradyan yönü dağılımı (büyüklük ağırlıklı)", fontsize=10)
    fig.tight_layout()
    viz._save(fig, OUT / "fig_orientation_hist.png")

    write_metrics(m, OUT / "metrics.json")
    for k in order:
        e = m["cases"][k]["edges"]
        o = e["orientation"]
        print(
            f"[q2] {k:15s} canny={e['canny_edge_pixels']:5d}/ref {e['reference_canny_edge_pixels']:5d} "
            f"exactMis%={e['canny_exact_mismatch_pct']:5.1f} false1px%={e['canny_false_pct_1px']:4.1f} "
            f"sobelMAE={e['sobel_mae_vs_reference']:6.1f} excess%={e['sobel_excess_energy_pct']:5.1f} "
            f"yönSapma={o['mean_abs_dev_deg']:5.1f}° off10%={o['pct_off_axis_gt10deg']:5.1f}"
        )


if __name__ == "__main__":
    main()
