"""Çıktı doğrulaması: beklenen dosyalar, metrics.json geçerliliği, kritik değerler, data/ bütünlüğü.

Kullanım: uv run python tools/verify_outputs.py   (hata varsa exit 1)
"""

from __future__ import annotations

import json
import sys

from ip.io import DATA_DIR, OUTPUT_DIR, file_sha256

EXPECTED = {
    "q1": [f"{n}_{s}.png" for n in ("pout", "tire") for s in ("original", "stretch", "he")]
    + [
        f"{n}_fig_{f}.png"
        for n in ("pout", "tire")
        for f in (
            "enhance",
            "hist",
            "filters_stretch",
            "filters_he",
            "edge_zoom",
            "profile",
            "metrics",
            "saltpepper",
        )
    ],
    "q2": [
        "fig_transforms.png",
        "fig_sobel.png",
        "fig_canny.png",
        "fig_zoom_rot30_roi1.png",
        "fig_zoom_rot30_roi2.png",
        "fig_zoom_shear_roi1.png",
        "fig_orientation_hist.png",
    ],
    "q3": [
        "fig_resize_compare.png",
        "fig_resize_diff_heatmap.png",
        "fig_resize_zoom.png",
        "fig_laplacian_sharpen.png",
        "fig_laplacian_diff_heatmap.png",
        "fig_overflow.png",
        "resized_manual.png",
        "sharpened_manual.png",
    ],
}
INPUTS = ["pout.tif", "tire.tif", "checkerboard.tif", "cameraman.tif"]
HASH_FILE = OUTPUT_DIR / "data_hashes.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, note: str = "") -> None:
        rows.append((name, bool(ok), note))

    # data/ bütünlüğü: ilk çalıştırmada hash kaydedilir, sonrakilerde karşılaştırılır
    present = [f for f in INPUTS if (DATA_DIR / f).is_file()]
    check("data/ girişleri", len(present) == len(INPUTS), f"{len(present)}/{len(INPUTS)} mevcut")
    hashes = {f: file_sha256(DATA_DIR / f) for f in present}
    if HASH_FILE.is_file():
        old = json.loads(HASH_FILE.read_text(encoding="utf-8"))
        changed = [f for f in hashes if f in old and old[f] != hashes[f]]
        check("data/ değişmedi", not changed, ", ".join(changed) or "hash'ler aynı")
    elif hashes:
        HASH_FILE.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        check("data/ değişmedi", True, "ilk çalıştırma: hash'ler kaydedildi")

    metrics = {}
    for q, files in EXPECTED.items():
        missing = [f for f in files if not (OUTPUT_DIR / q / f).is_file()]
        check(
            f"{q} figürleri",
            not missing,
            f"eksik: {missing[:3]}" if missing else f"{len(files)} dosya",
        )
        mpath = OUTPUT_DIR / q / "metrics.json"
        try:
            metrics[q] = json.loads(mpath.read_text(encoding="utf-8"))
            check(f"{q} metrics.json", True, "geçerli JSON")
        except (OSError, json.JSONDecodeError) as e:
            check(f"{q} metrics.json", False, type(e).__name__)

    # Q1: iyileştirilmiş görüntüler tam [0,255]
    for n in ("pout", "tire"):
        for s in ("stretch", "he"):
            p = OUTPUT_DIR / "q1" / f"{n}_{s}.png"
            if p.is_file():
                import cv2

                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                check(
                    f"q1 {n}_{s} aralık",
                    img.min() == 0 and img.max() == 255,
                    f"min={img.min()} max={img.max()}",
                )

    if "q3" in metrics:
        r = metrics["q3"]["resize"]
        check("q3 loops == vec", r["loops_equals_vec_exact"])
        check(
            "q3 half-pixel vs cv2 ≤ 1",
            r["half_pixel_vs_cv2"]["max_abs_diff"] <= 1,
            f"max={r['half_pixel_vs_cv2']['max_abs_diff']}",
        )
        lap = metrics["q3"]["laplacian"]
        check("q3 Laplacian vs filter2D == 0", lap["loops_vs_filter2d_float"]["max_abs_diff"] == 0)
    if "q2" in metrics:
        c = metrics["q2"]["cases"]
        nn, bl = c["rot30_nn"]["edges"], c["rot30_bilinear"]["edges"]
        check(
            "q2 NN Sobel hatası > Bilinear",
            nn["sobel_mae_vs_reference"] > bl["sobel_mae_vs_reference"],
            f"{nn['sobel_mae_vs_reference']:.1f} vs {bl['sobel_mae_vs_reference']:.1f}",
        )
    if "q1" in metrics:
        for n in ("pout", "tire"):
            e = metrics["q1"][n]["enhance"]["he_vs_cv2_equalizeHist"]["max_abs_diff"]
            check(f"q1 {n} HE vs cv2 ≤ 1", e <= 1, f"max={e}")

    w = max(len(r[0]) for r in rows)
    print(f"| {'Kontrol'.ljust(w)} | Durum | Not")
    print(f"|{'-' * (w + 2)}|-------|-----")
    for name, ok, note in rows:
        print(f"| {name.ljust(w)} | {'✅' if ok else '❌'}    | {note}")
    bad = [r for r in rows if not r[1]]
    print(f"\n{'YEŞİL' if not bad else f'KIRMIZI: {len(bad)} kontrol başarısız'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
