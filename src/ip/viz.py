"""Figür üretimi (matplotlib, Agg). Tüm figürler script'lerden üretilir; elle ekran görüntüsü yok."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ip.enhance import histogram  # noqa: E402

DPI = 150


def save_fig(fig, path: str | Path) -> Path:
    """Figürü DPI ile kaydeder ve kapatır."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    ncols: int = 3,
    suptitle: str | None = None,
    cell: float = 3.2,
    cmaps: Sequence[str] | None = None,
) -> Path:
    """Görüntü ızgarası; gri görüntüler [0,255] sabit ölçekle (otomatik kontrast YOK) gösterilir."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        img = images[i]
        cmap = cmaps[i] if cmaps else "gray"
        if img.ndim == 3:
            ax.imshow(img, interpolation="nearest")
        elif cmap == "gray":
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(titles[i], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return save_fig(fig, path)


def histograms(
    images: Sequence[np.ndarray], labels: Sequence[str], path: str | Path, title: str = ""
) -> Path:
    """Her görüntü için histogram (çubuk) + normalize CDF (çizgi)."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.8), squeeze=False)
    for ax, img, lab in zip(axes.flat, images, labels, strict=True):
        hist = histogram(img)
        ax.bar(np.arange(256), hist, width=1.0, color="0.35")
        ax.set_xlim(-2, 257)
        ax.set_title(f"{lab}\nmin={int(img.min())} max={int(img.max())}", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(np.cumsum(hist) / hist.sum(), color="tab:red", lw=1.2)
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save_fig(fig, path)


def zoom(img: np.ndarray, roi: tuple[int, int, int, int], scale: int = 8) -> np.ndarray:
    """ROI (y, x, h, w) kırp ve NEAREST ile tamsayı kat büyüt (piksel blokları görünür kalır)."""
    y, x, h, w = roi
    crop = img[y : y + h, x : x + w]
    return np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)


def draw_rois(
    img: np.ndarray, rois: Sequence[tuple[int, int, int, int]], colors=None
) -> np.ndarray:
    """Gri görüntü üzerine ROI dikdörtgenleri çiz (RGB uint8 döner)."""
    colors = colors or [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 160, 0)]
    rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
    for k, (y, x, h, w) in enumerate(rois):
        c = colors[k % len(colors)]
        y2, x2 = min(y + h, img.shape[0]) - 1, min(x + w, img.shape[1]) - 1
        rgb[y, x : x2 + 1] = c
        rgb[y2, x : x2 + 1] = c
        rgb[y : y2 + 1, x] = c
        rgb[y : y2 + 1, x2] = c
    return rgb


def _heat_panel(fig, ax, data, title, cmap, vmax, fontsize) -> None:
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def heatmap(
    data: np.ndarray, path: str | Path, title: str, cmap: str = "magma", vmax: float | None = None
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    _heat_panel(fig, ax, data, title, cmap, vmax, fontsize=10)
    fig.tight_layout()
    return save_fig(fig, path)


def heatmaps(
    datas: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    cmap: str = "magma",
    vmax: float | None = None,
    suptitle: str | None = None,
) -> Path:
    n = len(datas)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, d, t in zip(axes.flat, datas, titles, strict=True):
        _heat_panel(fig, ax, d, t, cmap, vmax, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return save_fig(fig, path)


def profiles(
    curves: dict[str, np.ndarray], path: str | Path, title: str, xlabel: str = "piksel"
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    styles = ["-", "--", "-.", ":", "-"]
    for k, (lab, c) in enumerate(curves.items()):
        ax.plot(c, styles[k % len(styles)], lw=1.4, label=lab, marker="." if k == 0 else None)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("yoğunluk")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, path)


def bars(groups: dict[str, dict[str, float]], path: str | Path, title: str, ylabel: str) -> Path:
    """groups = {grup_adı: {seri_adı: değer}} → gruplanmış çubuk grafik."""
    gnames = list(groups)
    snames = list(next(iter(groups.values())))
    x = np.arange(len(gnames))
    wd = 0.8 / len(snames)
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(gnames) * len(snames) / 2, 3.6))
    for i, s in enumerate(snames):
        ax.bar(x + i * wd - 0.4 + wd / 2, [groups[g][s] for g in gnames], wd, label=s)
    ax.set_xticks(x, gnames, fontsize=8, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, path)
