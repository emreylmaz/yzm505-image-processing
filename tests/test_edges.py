import numpy as np

from ip import edges as E
from ip import filters as F
from ip.io import to_uint8


def step_image(noise=0.0, seed=0):
    img = np.full((64, 64), 60.0)
    img[:, 32:] = 190.0
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return to_uint8(img)


def test_dilate_square():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    assert E.dilate(m, 1).sum() == 9


def test_edge_mask_ignores_noise_and_finds_step():
    img = step_image(noise=12)
    mask = E.edge_mask(img)
    cols = np.flatnonzero(mask.any(axis=0))
    assert cols.min() >= 28 and cols.max() <= 35  # yalnızca gerçek kenar çevresi
    raw = E.edge_mask(img, sigma=0)  # yumuşatmasız: gürültü de kenar sayılır
    assert raw.sum() > mask.sum()


def test_sharpness_ordering_median_vs_mean_on_step():
    img = step_image(noise=8)
    mask = E.edge_mask(img)
    s_med = E.edge_sharpness(to_uint8(F.median_filter(img, 5)), mask)
    s_mean = E.edge_sharpness(to_uint8(F.mean_filter(img, 5)), mask)
    assert s_med > s_mean  # median basamak kenarını korur, mean rampaya yayar


def test_flat_noise_drops_after_smoothing():
    img = step_image(noise=10)
    fm = E.flat_mask(img)
    assert E.flat_noise(to_uint8(F.gaussian_filter(img, 1.0)), fm) < E.flat_noise(img, fm)


def test_best_edge_roi_contains_edge():
    y, x, h, w = E.best_edge_roi(E.edge_mask(step_image()), size=16)
    assert x <= 32 < x + w


def test_orientation_stats_vertical_edge():
    st = E.orientation_stats(step_image(), expected_deg=(0.0,))
    assert st["mean_abs_dev_deg"] < 1.0 and st["pct_off_axis_gt10deg"] == 0.0
