"""Q1 — Manuel Contrast Stretching + Histogram Equalization, ardından Mean/Median/Gaussian filtreleri.

Her adım pout.tif ve tire.tif'e AYRI AYRI uygulanır.
Çıktılar: outputs/q1/<ad>_*.png, outputs/q1/metrics.json
"""

from __future__ import annotations

import sys

import numpy as np

from ip import edges as E
from ip import enhance as H
from ip import filters as F
from ip import viz
from ip.io import OUTPUT_DIR, load_gray, require_data, save_png, to_uint8
from ip.metrics import compare, psnr, write_metrics

OUT = OUTPUT_DIR / "q1"
IMAGES = {"pout": "pout.tif", "tire": "tire.tif"}
SIGMAS = (1.0, 2.0)
SP_PROB = 0.02  # ek deney: salt & pepper yoğunluğu (ASSUMPTION 2)
SEED = 505


def filter_bank(img_u8: np.ndarray) -> dict[str, np.ndarray]:
    """Tüm filtreler manual; border: mean/gaussian reflect101, median replicate."""
    bank = {
        "mean3": F.mean_filter(img_u8, 3),
        "mean5": F.mean_filter(img_u8, 5),
        "median3": F.median_filter(img_u8, 3),
        "median5": F.median_filter(img_u8, 5),
    }
    for s in SIGMAS:
        bank[f"gauss_s{s:g}"] = F.gaussian_filter(img_u8, s)
    return {k: to_uint8(v) for k, v in bank.items()}


LABELS = {
    "mean3": "Mean 3×3",
    "mean5": "Mean 5×5",
    "median3": "Median 3×3",
    "median5": "Median 5×5",
    "gauss_s1": "Gaussian σ=1 (7×7)",
    "gauss_s2": "Gaussian σ=2 (13×13)",
}


def score_filters(ref: np.ndarray, bank: dict[str, np.ndarray]) -> dict:
    """Aynı (referanstan çıkarılmış) kenar ve düz-bölge maskeleriyle metrikler."""
    emask, fmask = E.edge_mask(ref), E.flat_mask(ref)
    s0, n0 = E.edge_sharpness(ref, emask), E.flat_noise(ref, fmask)
    rows = {
        "unfiltered": {
            "edge_sharpness": s0,
            "laplacian_var": E.laplacian_variance(ref),
            "flat_noise": n0,
            "sharpness_retained_pct": 100.0,
            "noise_reduction_pct": 0.0,
        }
    }
    for k, img in bank.items():
        s, n = E.edge_sharpness(img, emask), E.flat_noise(img, fmask)
        rows[k] = {
            "edge_sharpness": s,
            "laplacian_var": E.laplacian_variance(img),
            "flat_noise": n,
            "sharpness_retained_pct": s / s0 * 100.0,
            "noise_reduction_pct": (1.0 - n / n0) * 100.0,
        }
    return {
        "edge_mask_pixels": int(emask.sum()),
        "flat_mask_pixels": int(fmask.sum()),
        "filters": rows,
    }


def verdict(rows: dict, sp: dict | None = None) -> dict:
    """Denge skoru = 2·S·N/(S+N) (S: keskinlik korunumu %, N: gürültü azalması %; F1 benzeri)."""
    score = {
        k: 2
        * v["sharpness_retained_pct"]
        * v["noise_reduction_pct"]
        / (v["sharpness_retained_pct"] + v["noise_reduction_pct"])
        for k, v in rows.items()
        if k != "unfiltered"
    }
    best = max(score, key=score.get)
    out = {
        "balance_score": score,
        "best": best,
        "best_label": LABELS[best],
        "best_sharpness_only": LABELS[max(score, key=lambda k: rows[k]["sharpness_retained_pct"])],
        "mean_vs_median": {
            f"k{k}": {
                "mean_sharpness_retained_pct": rows[f"mean{k}"]["sharpness_retained_pct"],
                "median_sharpness_retained_pct": rows[f"median{k}"]["sharpness_retained_pct"],
                "mean_noise_reduction_pct": rows[f"mean{k}"]["noise_reduction_pct"],
                "median_noise_reduction_pct": rows[f"median{k}"]["noise_reduction_pct"],
                "median_sharper": rows[f"median{k}"]["sharpness_retained_pct"]
                > rows[f"mean{k}"]["sharpness_retained_pct"],
            }
            for k in (3, 5)
        },
    }
    if sp:
        cand = {k: v["psnr_vs_clean_db"] for k, v in sp.items() if k in LABELS}
        out["saltpepper_best_psnr"] = LABELS[max(cand, key=cand.get)]
    return out


def salt_pepper(img_u8: np.ndarray, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.random(img_u8.shape)
    out = img_u8.copy()
    out[r < p / 2] = 0
    out[(r >= p / 2) & (r < p)] = 255
    return out


def process(name: str, fname: str) -> dict:
    img = load_gray(require_data(fname))
    m: dict = {"input": {"file": f"data/{fname}", "shape": list(img.shape)}}
    m["input"].update(min=int(img.min()), max=int(img.max()), mean=float(img.mean()))

    # ------------------------------------------------------------ 1) İyileştirme (manual)
    st = to_uint8(H.contrast_stretch(img))
    he = to_uint8(H.hist_equalize(img))
    save_png(img, OUT / f"{name}_original.png")
    save_png(st, OUT / f"{name}_stretch.png")
    save_png(he, OUT / f"{name}_he.png")

    def stats(x):
        return {
            "min": int(x.min()),
            "max": int(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "n_levels": int(np.count_nonzero(H.histogram(x))),
            "cdf_linearity_max_dev": H.cdf_linearity_max_dev(x),
        }

    m["enhance"] = {
        "original": stats(img),
        "stretch": stats(st),
        "he": stats(he),
        "he_vs_cv2_equalizeHist": compare(he, H.equalize_builtin_cv2(img)),
        "he_vs_skimage_equalize_hist": compare(he, H.equalize_builtin_skimage(img)),
    }
    viz.grid(
        [img, st, he],
        [
            f"Orijinal [{img.min()},{img.max()}]",
            f"Contrast Stretching [{st.min()},{st.max()}]",
            f"Histogram Equalization [{he.min()},{he.max()}]",
        ],
        OUT / f"{name}_fig_enhance.png",
        ncols=3,
        cell=3.6,
    )
    viz.histograms(
        [img, st, he],
        ["Orijinal", "Contrast Stretching", "Histogram Equalization"],
        OUT / f"{name}_fig_hist.png",
        title=f"{name}: histogram (gri) + CDF (kırmızı)",
    )

    # ------------------------------------------------------------ 2) Filtreler (her iki iyileştirmeye)
    m["filters"] = {}
    banks = {}
    for ename, ref in (("stretch", st), ("he", he)):
        bank = filter_bank(ref)
        banks[ename] = bank
        m["filters"][ename] = score_filters(ref, bank)
        viz.grid(
            [ref, *bank.values()],
            [f"{ename.upper()} (filtresiz)", *[LABELS[k] for k in bank]],
            OUT / f"{name}_fig_filters_{ename}.png",
            ncols=4,
            cell=3.2,
            suptitle=f"{name}: {ename} sonrası filtreler",
        )

    # ------------------------------------------------------------ 3) Mean vs Median — kenar odaklı kanıt (HE)
    bank = banks["he"]
    emask = E.edge_mask(he)
    roi = E.best_edge_roi(emask, size=40)
    m["edge_roi_yxhw"] = list(roi)
    m["edge_roi_note"] = (
        "HE görüntüsünün nesne kenar maskesinde (Canny(Gaussian σ=2)) kenar yoğunluğu en yüksek 40×40 pencere (otomatik, deterministik)"
    )
    viz.grid(
        [
            viz.draw_rois(he, [roi]),
            viz.zoom(he, roi, 6),
            viz.zoom(bank["mean3"], roi, 6),
            viz.zoom(bank["median3"], roi, 6),
            viz.zoom(E.sobel_display(E.sobel_mag(he)), roi, 6),
            viz.zoom(bank["mean5"], roi, 6),
            viz.zoom(bank["median5"], roi, 6),
            viz.zoom(bank["gauss_s2"], roi, 6),
        ],
        [
            f"HE + ROI {roi}",
            "HE ROI (NEAREST ×6)",
            "Mean 3×3",
            "Median 3×3",
            "HE |Sobel| ROI",
            "Mean 5×5",
            "Median 5×5",
            "Gaussian σ=2",
        ],
        OUT / f"{name}_fig_edge_zoom.png",
        ncols=4,
        cell=3.2,
        suptitle=f"{name}: kenar ROI'sinde Mean vs Median (HE sonrası)",
    )
    # Kenara dik çizgi profili: ROI merkez satırı boyunca
    y, x, h, w = roi
    row = y + h // 2
    m["profile_row"] = int(row)
    viz.profiles(
        {
            "HE (filtresiz)": he[row, x : x + w].astype(float),
            "Mean 5×5": bank["mean5"][row, x : x + w].astype(float),
            "Median 5×5": bank["median5"][row, x : x + w].astype(float),
            "Gaussian σ=2": bank["gauss_s2"][row, x : x + w].astype(float),
        },
        OUT / f"{name}_fig_profile.png",
        f"{name}: satır {row}, x∈[{x},{x + w}) yoğunluk profili",
        xlabel=f"x − {x}",
    )
    rows = m["filters"]["he"]["filters"]
    viz.bars(
        {
            LABELS.get(k, k): {
                "Keskinlik korunumu %": rows[k]["sharpness_retained_pct"],
                "Gürültü azalması %": rows[k]["noise_reduction_pct"],
            }
            for k in bank
        },
        OUT / f"{name}_fig_metrics.png",
        f"{name} (HE): kenar keskinliği vs düz bölge gürültüsü",
        "%",
    )

    # ------------------------------------------------------------ 4) Ek deney: salt & pepper
    noisy = salt_pepper(he, SP_PROB, SEED)
    nb = filter_bank(noisy)
    em = E.edge_mask(he)
    sp = {
        "noisy": {
            "psnr_vs_clean_db": psnr(noisy, he),
            "edge_sharpness": E.edge_sharpness(noisy, em),
        }
    }
    for k, v in nb.items():
        sp[k] = {"psnr_vs_clean_db": psnr(v, he), "edge_sharpness": E.edge_sharpness(v, em)}
    sp["clean_edge_sharpness"] = E.edge_sharpness(he, em)
    m["salt_pepper"] = {"p": SP_PROB, "seed": SEED, "results": sp}
    m["verdict"] = {
        "he": verdict(m["filters"]["he"]["filters"], sp),
        "stretch": verdict(m["filters"]["stretch"]["filters"]),
    }
    viz.grid(
        [noisy, nb["mean3"], nb["median3"], nb["gauss_s1"]]
        + [viz.zoom(z, roi, 6) for z in (noisy, nb["mean3"], nb["median3"], nb["gauss_s1"])],
        [
            f"HE + salt&pepper p={SP_PROB}",
            f"Mean 3×3  PSNR={sp['mean3']['psnr_vs_clean_db']:.1f}",
            f"Median 3×3  PSNR={sp['median3']['psnr_vs_clean_db']:.1f}",
            f"Gaussian σ=1  PSNR={sp['gauss_s1']['psnr_vs_clean_db']:.1f}",
            "zoom",
            "zoom",
            "zoom",
            "zoom",
        ],
        OUT / f"{name}_fig_saltpepper.png",
        ncols=4,
        cell=3.2,
        suptitle=f"{name}: ek deney — salt & pepper gürültüsü",
    )
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1254 konsolu
    metrics = {
        "sharpness_metric": (
            "SEÇİLMİŞ METRİK: nesne kenar maskesi içindeki ortalama Sobel gradyan büyüklüğü. "
            "Maske, filtrelenmemiş iyileştirilmiş görüntünün Gaussian σ=2 ile yumuşatılmış "
            "kopyasında Canny(50,150) + 3×3 dilate ile çıkarılır (gürültü kenar sayılmaz) ve tüm "
            "filtrelere aynı uygulanır. Gürültü: düz bölgede (yumuşatılmış Sobel alt %40, "
            "kenarlardan ≥3 px uzak) Laplacian yanıtının std'si."
        ),
        "borders": {"mean": "reflect101", "gaussian": "reflect101", "median": "replicate"},
    }
    for name, fname in IMAGES.items():
        metrics[name] = process(name, fname)
        r = metrics[name]["filters"]["he"]["filters"]
        print(
            f"[q1] {name}: stretch [{metrics[name]['enhance']['stretch']['min']},"
            f"{metrics[name]['enhance']['stretch']['max']}] he [{metrics[name]['enhance']['he']['min']},"
            f"{metrics[name]['enhance']['he']['max']}]  "
            + "  ".join(
                f"{k}: keskinlik%{v['sharpness_retained_pct']:.0f}/gürültü↓%{v['noise_reduction_pct']:.0f}"
                for k, v in r.items()
                if k != "unfiltered"
            )
        )
    write_metrics(metrics, OUT / "metrics.json")


if __name__ == "__main__":
    main()
