"""report/template.md → report/report.md

Yer tutucular:
  {{q3.resize.half_pixel_vs_cv2.max_abs_diff}}        metrics.json değeri (q1/q2/q3 kökleri)
  {{q3.resize.half_pixel_vs_cv2.psnr_db|.2f}}         biçim belirteci ile
  {{code:ip.resize.bilinear_resize_loops}}            inspect.getsource ile kod bloğu (rapor = gerçek kod)
  {{fig:q3/fig_resize_compare.png|Başlık}}            figür (dosya yoksa HATA)
  {{table:q1_filters:pout:he}}                        TABLES içindeki üreticiler
Çözümlenemeyen yer tutucu → exit 1 (sayı asla elle yazılmaz).
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

from ip.io import OUTPUT_DIR, ROOT

REPORT_DIR = Path(os.environ.get("IP_REPORT_DIR", ROOT / "report"))
TEMPLATE = ROOT / "report" / "template.md"
PH = re.compile(r"\{\{([^{}]+)\}\}")


def load_metrics() -> dict:
    return {
        q: json.loads((OUTPUT_DIR / q / "metrics.json").read_text(encoding="utf-8"))
        for q in ("q1", "q2", "q3")
    }


def lookup(m: dict, path: str):
    cur = m
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def fmt(v, spec: str | None) -> str:
    if spec and isinstance(v, (int, float)):
        return format(v, spec)
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"
    if isinstance(v, bool):
        return "evet" if v else "hayır"
    if isinstance(v, list) and v and all(isinstance(r, list) for r in v):
        return "[" + ", ".join("[" + ", ".join(f"{x:.4f}" for x in r) + "]" for r in v) + "]"
    if isinstance(v, list):
        return "×".join(str(x) for x in v) if all(isinstance(x, int) for x in v) else str(v)
    return str(v)


def code_block(dotted: str) -> str:
    mod, name = dotted.rsplit(".", 1)
    obj = getattr(importlib.import_module(mod), name)
    return f"```python\n{inspect.getsource(obj).rstrip()}\n```"


def figure(spec: str) -> str:
    rel, _, caption = spec.partition("|")
    path = OUTPUT_DIR / rel.strip()
    if not path.is_file():
        raise FileNotFoundError(f"figür yok: {path}")
    link = os.path.relpath(path, REPORT_DIR).replace(os.sep, "/")
    return f"![{caption.strip()}]({link})"


# ----------------------------------------------------------------------------- tablolar
def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header, rows) -> str:
    return "\n".join([_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)])


Q1_LABELS = {
    "unfiltered": "Filtresiz",
    "mean3": "Mean 3×3",
    "mean5": "Mean 5×5",
    "median3": "Median 3×3",
    "median5": "Median 5×5",
    "gauss_s1": "Gaussian σ=1",
    "gauss_s2": "Gaussian σ=2",
}


def t_q1_enhance(m, name):
    e = m["q1"][name]["enhance"]
    rows = []
    for k, lab in (
        ("original", "Orijinal"),
        ("stretch", "Contrast Stretching"),
        ("he", "Hist. Equalization"),
    ):
        s = e[k]
        rows.append(
            [
                lab,
                s["min"],
                s["max"],
                f"{s['mean']:.1f}",
                f"{s['std']:.1f}",
                s["n_levels"],
                f"{s['cdf_linearity_max_dev']:.3f}",
            ]
        )
    return _table(
        ["Görüntü", "min", "max", "ortalama", "std", "seviye sayısı", "CDF lineerlik sapması"], rows
    )


def t_q1_filters(m, name, enh):
    f = m["q1"][name]["filters"][enh]["filters"]
    score = m["q1"][name]["verdict"][enh]["balance_score"]
    rows = [
        [
            Q1_LABELS[k],
            f"{v['edge_sharpness']:.1f}",
            f"{v['sharpness_retained_pct']:.1f}",
            f"{v['flat_noise']:.2f}",
            f"{v['noise_reduction_pct']:.1f}",
            f"{v['laplacian_var']:.0f}",
            f"{score[k]:.1f}" if k in score else "—",
        ]
        for k, v in f.items()
    ]
    return _table(
        [
            "Filtre",
            "Kenar keskinliği",
            "Korunum %",
            "Düz bölge gürültüsü",
            "Gürültü azalması %",
            "Var(Laplacian)",
            "Denge skoru",
        ],
        rows,
    )


def t_q1_sp(m, name):
    r = m["q1"][name]["salt_pepper"]["results"]
    rows = [
        [
            Q1_LABELS.get(k, "Gürültülü (filtresiz)"),
            f"{v['psnr_vs_clean_db']:.2f}",
            f"{v['edge_sharpness']:.1f}",
        ]
        for k, v in r.items()
        if isinstance(v, dict)
    ]
    rows.append(["Temiz HE (referans)", "∞", f"{r['clean_edge_sharpness']:.1f}"])
    return _table(["Görüntü", "PSNR (dB, temiz HE'ye göre)", "Kenar keskinliği"], rows)


Q2_LABELS = {
    "original": "Orijinal",
    "shear_nn": "Shear (NN)",
    "shear_bilinear": "Shear (Bilinear)",
    "rot30_nn": "Rotation 30° (NN)",
    "rot30_bilinear": "Rotation 30° (Bilinear)",
}


def t_q2_edges(m):
    rows = []
    for k, c in m["q2"]["cases"].items():
        e, o = c["edges"], c["edges"]["orientation"]
        rows.append(
            [
                Q2_LABELS[k],
                f"{e['sobel_mae_vs_reference']:.1f}",
                f"{e['sobel_excess_energy_pct']:.1f}",
                f"{o['mean_abs_dev_deg']:.1f}",
                f"{o['pct_off_axis_gt10deg']:.1f}",
                f"{e['canny_edge_pixels']} / {e['reference_canny_edge_pixels']}",
                f"{e['canny_exact_mismatch_pct']:.1f}",
                f"{e['canny_missed_pct_1px']:.1f}",
            ]
        )
    return _table(
        [
            "Görüntü",
            "Sobel MAE (ref.)",
            "Sobel fazla enerji %",
            "Yön sapması (°)",
            "Yön >10° %",
            "Canny piksel / ref.",
            "Canny konum uyumsuzluğu %",
            "Kaçırılan ref. kenar % (1px)",
        ],
        rows,
    )


def t_q2_cv2(m):
    rows = []
    for k, c in m["q2"]["cases"].items():
        d = c["manual_vs_cv2_warpAffine"]
        rows.append(
            [
                Q2_LABELS[k],
                c["shape"][0],
                c["shape"][1],
                f"{d['max_abs_diff']:.0f}",
                f"{d['mean_abs_diff']:.3f}",
                f"{d['pct_pixels_differ']:.2f}",
            ]
        )
    return _table(["Görüntü", "H", "W", "max |Δ| (cv2)", "ortalama |Δ|", "farklı piksel %"], rows)


def _cmp_row(label, d):
    p = d["psnr_db"]
    return [
        label,
        f"{d['max_abs_diff']:g}",
        f"{d['mean_abs_diff']:.4f}",
        "∞" if p == "inf" else f"{p:.2f}",
        f"{d.get('pct_pixels_differ', 0):.2f}",
    ]


def t_q3_compare(m):
    r, lap, sh = m["q3"]["resize"], m["q3"]["laplacian"], m["q3"]["sharpen"]
    rows = [
        _cmp_row("Resize: manual (half-pixel) vs cv2.resize", r["half_pixel_vs_cv2"]),
        _cmp_row("Resize: manual (naive) vs cv2.resize", r["naive_vs_cv2"]),
        _cmp_row("Laplacian: loops vs cv2.filter2D (float64)", lap["loops_vs_filter2d_float"]),
        _cmp_row(
            "Laplacian: loops vs cv2.Laplacian(ksize=1)", lap["loops_vs_cv2_laplacian_ksize1"]
        ),
        _cmp_row("Sharpen: aynı giriş, manual vs built-in", sh["manual_vs_builtin_same_input"]),
        _cmp_row("Sharpen: tam pipeline manual vs cv2", sh["manual_pipeline_vs_cv2_pipeline"]),
        _cmp_row(
            "Taşma: uint8 doygun L vs float L",
            m["q3"]["overflow"]["uint8_saturated_vs_float_laplacian"],
        ),
        _cmp_row(
            "Taşma: uint8 wrap L vs float L",
            m["q3"]["overflow"]["uint8_wrapped_vs_float_laplacian"],
        ),
    ]
    return _table(
        ["Karşılaştırma", "max |Δ|", "ortalama |Δ|", "PSNR (dB)", "farklı piksel %"], rows
    )


def t_q3_example(m):
    w = m["q3"]["resize"]["worked_example"]
    nb, wt = list(w["neighbors"].items()), list(w["weights"].items())
    rows = [
        [nb[i][0], nb[i][1], wt[i][0], f"{wt[i][1]:.6f}", f"{wt[i][1] * nb[i][1]:.4f}"]
        for i in range(4)
    ]
    rows.append(
        [
            "**Toplam**",
            "",
            "",
            f"{w['weights_sum']:.6f}",
            f"**{w['value_float']:.4f}** → uint8 {w['value_uint8']}",
        ]
    )
    return _table(["Komşu", "Değer", "Ağırlık", "Ağırlık değeri", "Katkı"], rows)


def t_q3_border(m):
    rows = [
        [
            k.replace("reflect101_vs_", "reflect101 vs "),
            f"{v['max_abs_diff']:g}",
            v["n_pixels_differ"],
            v["n_inner_pixels_differ"],
        ]
        for k, v in m["q3"]["border_modes"].items()
    ]
    return _table(
        ["Border karşılaştırması", "max |Δ|", "farklı piksel", "iç bölgede farklı piksel"], rows
    )


# ----------------------------------------------------------------------------- veriye bağlı cümleler
# Sonuç cümleleri elle yazılmaz: sayılara göre burada üretilir (rapor ile veri çelişemez).
def a_he_cv2(m, name):
    d = m["q1"][name]["enhance"]["he_vs_cv2_equalizeHist"]
    if d["max_abs_diff"] == 0:
        return (
            "Manuel HE ile `cv2.equalizeHist` çıktıları **birebir aynıdır** (maksimum fark 0); "
            "iki uygulama aynı formülü kullanır."
        )
    return (
        f"Manuel HE ile `cv2.equalizeHist` arasındaki maksimum fark {d['max_abs_diff']:g} gri seviye "
        f"(farklı piksel oranı %{d['pct_pixels_differ']:.2f}); fark OpenCV'nin float32 ölçek katsayısı "
        "ve yuvarlama kuralından kaynaklanır."
    )


def a_builtin_filters(m):
    parts = []
    for name in ("pout", "tire"):
        b = m["q1"][name]["builtin_check"]
        parts.append(f"{name}: en fazla {max(v['max_abs_diff'] for v in b.values()):g}")
    return (
        "HE görüntüleri üzerinde manuel filtrelerin OpenCV karşılıklarından (aynı border) maksimum "
        "farkı — " + ", ".join(parts) + " gri seviye (ayrıntı: `metrics.json → builtin_check`)."
    )


def _mvm(m, n, k):
    return m["q1"][n]["verdict"]["he"]["mean_vs_median"][k]


def a_q1_conclusion(m):
    names, ks = ("pout", "tire"), ("k3", "k5")
    med_sharper = all(_mvm(m, n, k)["median_sharper"] for n in names for k in ks)
    mean_more_noise_red = all(
        _mvm(m, n, k)["median_noise_reduction_pct"] < _mvm(m, n, k)["mean_noise_reduction_pct"]
        for n in names
        for k in ks
    )
    v = {n: m["q1"][n]["verdict"]["he"] for n in names}
    sharp = {n: v[n]["best_sharpness_only"] for n in names}
    lines = []
    if med_sharper:
        lines.append(
            "Her iki görüntüde ve her iki pencere boyutunda (3×3, 5×5) **Median, Mean'e göre nesne "
            "kenarlarını daha iyi korumuştur** (Bölüm 1.7 sayıları, zoom ve profil figürleri)."
        )
    else:
        lines.append(
            "Median'ın Mean'e göre kenar korunumu üstünlüğü tüm görüntü/pencere kombinasyonlarında "
            "geçerli değildir; ayrıntı Bölüm 1.7'deki sayılardadır."
        )
    if mean_more_noise_red:
        lines.append(
            "Buna karşılık düz bölgelerdeki ince taneli (Gaussian benzeri) gürültüyü Mean daha güçlü "
            "bastırmıştır; Mean'in bu avantajı kenar bulanıklığı pahasına gelmektedir."
        )
    if sharp["pout"] == sharp["tire"]:
        lines.append(
            f"Yalnızca kenar keskinliği ölçütünde en iyi filtre her iki görüntüde **{sharp['pout']}** olmuştur."
        )
    else:
        lines.append(
            f"Yalnızca kenar keskinliği ölçütünde en iyi filtre pout'ta **{sharp['pout']}**, "
            f"tire'da **{sharp['tire']}** olmuştur."
        )
    lines.append(
        f"Gürültü azaltma–keskinlik denge skorunda pout'ta **{v['pout']['best_label']}**, tire'da "
        f"**{v['tire']['best_label']}** öndedir; impulsif gürültüde en yüksek PSNR pout'ta "
        f"**{v['pout']['saltpepper_best_psnr']}**, tire'da **{v['tire']['saltpepper_best_psnr']}** ile "
        "elde edilmiştir."
    )
    median_wins = med_sharper and all(
        "Median" in x for x in (*sharp.values(), *(v[n]["saltpepper_best_psnr"] for n in names))
    )
    if median_wins:
        lines.append(
            "**Cevap:** Sorunun ölçütü “gürültüyü temizlerken keskinliği korumak” olduğundan, hem "
            "kenar korunumunda hem impulsif gürültü temizliğinde önde olan **Median filtre** en uygun "
            "filtredir. Gaussian/Mean yalnızca düz bölge gürültüsü öncelikli olduğunda ve kenar kaybı "
            "kabul edilebildiğinde tercih edilebilir."
        )
    else:
        lines.append(
            "**Cevap:** Sonuç ölçüte bağlıdır; yukarıdaki sayılar hangi filtrenin hangi ölçütte önde "
            "olduğunu görüntü bazında göstermektedir."
        )
    return "\n\n".join(lines)


def a_sp_best(m):
    out = []
    for n in ("pout", "tire"):
        v = m["q1"][n]["verdict"]["he"]
        out.append(
            f"{n}: **{v['saltpepper_best_psnr']}** (ikinci {v['saltpepper_second']}, "
            f"fark {v['saltpepper_margin_db']:.2f} dB)"
        )
    return "İmpulsif gürültüde en yüksek PSNR — " + "; ".join(out) + "."


def a_q3_rounding(m):
    r = m["q3"]["resize"]
    f = r["half_pixel_float_vs_cv2"]["max_abs_diff"]
    s = r["half_pixel_mean_signed_diff"]
    if f < 1:
        msg = (
            f"Float manuel sonuç ile OpenCV `uint8` sonucu arasındaki maksimum fark {f:.3f}; bu değer "
            "1'den küçük olduğundan uint8 farkı son yuvarlama adımından kaynaklanır."
        )
    else:
        msg = (
            f"Float manuel sonuç ile OpenCV `uint8` sonucu arasındaki maksimum fark {f:.3f}; 1'i aştığı "
            "için fark yalnızca son yuvarlamayla açıklanamaz."
        )
    return (
        msg
        + f" İşaretli ortalama fark {s:+.4f} gri seviyedir (≈0 → sistematik parlaklık kayması yok)."
    )


def a_lap_exact(m):
    d = m["q3"]["laplacian"]["loops_vs_filter2d_float"]["max_abs_diff"]
    if d == 0:
        return (
            "Laplacian'da manuel sürüm ve OpenCV aynı `reflect101` modunu kullandığında çıktılar "
            "**bit düzeyinde aynıdır** (maksimum fark 0)."
        )
    return f"Laplacian'da aynı `reflect101` modu ile maksimum fark: {d:g} (kayan nokta toplama sırası)."


def a_q2_canny(m):
    c = m["q2"]["cases"]
    nn, bl = c["rot30_nn"]["edges"], c["rot30_bilinear"]["edges"]
    gap = nn["canny_exact_mismatch_pct"] - bl["canny_exact_mismatch_pct"]
    base = (
        f"Canny tarafında piksel konum uyumsuzluğu NN %{nn['canny_exact_mismatch_pct']:.1f}, Bilinear "
        f"%{bl['canny_exact_mismatch_pct']:.1f}; kenar piksel sayısı NN {nn['canny_edge_pixels']}, "
        f"Bilinear {bl['canny_edge_pixels']}, referans {nn['reference_canny_edge_pixels']} (kaçırılan "
        f"referans kenar oranı NN %{nn['canny_missed_pct_1px']:.1f}, Bilinear %{bl['canny_missed_pct_1px']:.1f})."
    )
    if abs(gap) < 2:
        base += (
            " Canny'nin inceltme ve histerezis adımları basamakları kısmen düzelttiği için Canny "
            "konum metriklerindeki fark küçüktür; aliasing'in ana kanıtı Sobel fazla enerjisi ve "
            "gradyan yönü sapmasıdır."
        )
    else:
        base += " Canny zinciri NN'de belirgin biçimde merdiven biçimini izlemektedir."
    return base


def a_q2_nn_cv2(m):
    d = m["q2"]["cases"]["rot30_nn"]["manual_vs_cv2_warpAffine"]
    if d["max_abs_diff"] == 0:
        return "NN dönüşümünde manuel sonuç ile `cv2.warpAffine` birebir aynıdır."
    return (
        f"NN satırındaki maksimum fark ({d['max_abs_diff']:g}) piksellerin yalnızca "
        f"%{d['pct_pixels_differ']:.2f} kadarında görülür: kaynak konumu iki piksel merkezinin "
        "ortasına (.5) çok yakın düşen noktalarda manuel `floor(x+0.5)` ile OpenCV'nin sabit noktalı "
        "yuvarlaması farklı komşuyu seçer; checkerboard'da komşular 0 ve 255 olabildiği için tek bir "
        "farklı seçim büyük fark üretir."
    )


AUTO = {
    "he_cv2": a_he_cv2,
    "builtin_filters": a_builtin_filters,
    "q1_conclusion": a_q1_conclusion,
    "sp_best": a_sp_best,
    "q3_rounding": a_q3_rounding,
    "lap_exact": a_lap_exact,
    "q2_canny": a_q2_canny,
    "q2_nn_cv2": a_q2_nn_cv2,
}


TABLES = {
    "q1_enhance": t_q1_enhance,
    "q1_filters": t_q1_filters,
    "q1_sp": t_q1_sp,
    "q2_edges": t_q2_edges,
    "q2_cv2": t_q2_cv2,
    "q3_compare": t_q3_compare,
    "q3_example": t_q3_example,
    "q3_border": t_q3_border,
}


def render(text: str, m: dict) -> tuple[str, list[str]]:
    errors: list[str] = []

    def sub(match: re.Match) -> str:
        key = match.group(1).strip()
        try:
            if key.startswith("code:"):
                return code_block(key[5:])
            if key.startswith("fig:"):
                return figure(key[4:])
            if key.startswith("auto:"):
                name, *args = key[5:].split(":")
                return AUTO[name](m, *args)
            if key.startswith("table:"):
                name, *args = key[6:].split(":")
                return TABLES[name](m, *args)
            path, _, spec = key.partition("|")
            return fmt(lookup(m, path.strip()), spec or None)
        except Exception as e:  # noqa: BLE001 — tüm hatalar raporlanır, sonra exit 1
            errors.append(f"{{{{{key}}}}} → {type(e).__name__}: {e}")
            return f"**[ÇÖZÜLEMEDİ: {key}]**"

    return PH.sub(sub, text), errors


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out, errors = render(TEMPLATE.read_text(encoding="utf-8"), load_metrics())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.md").write_text(out, encoding="utf-8")
    if errors:
        print("Çözülemeyen yer tutucular:\n  " + "\n  ".join(errors))
        return 1
    n_fig = out.count("![")
    print(f"report.md yazıldı: {REPORT_DIR / 'report.md'} ({len(out)} karakter, {n_fig} figür)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
