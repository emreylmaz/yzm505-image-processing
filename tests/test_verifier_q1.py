"""Independent verifier checks for Q1 — expected values derived BY HAND (see comments).

Contrast stretch: s = (r - min) / (max - min) * 255
HE: T(k) = round((CDF(k) - CDF_min) / (N - CDF_min) * 255)
Box/median 3x3 with borders reflect101 (dcb|abcd|cba), replicate (aaa|abcd|ddd), zero.
"""

import numpy as np
import pytest

from ip.edges import dilate
from ip.enhance import contrast_stretch, equalization_lut, hist_equalize
from ip.filters import gaussian_filter, gaussian_kernel, mean_filter, median_filter

A = np.array([[1, 2, 3], [4, 10, 6], [7, 8, 20]], dtype=np.float64)
# Padded A (5x5):
# reflect101 rows [r1,r0,r1,r2,r1], each row [x1,x0,x1,x2,x1]:
#   [10,4,10,6,10] [2,1,2,3,2] [10,4,10,6,10] [8,7,8,20,8] [10,4,10,6,10]
# replicate:
#   [1,1,2,3,3] [1,1,2,3,3] [4,4,10,6,6] [7,7,8,20,20] [7,7,8,20,20]


# ---------------- contrast stretch ----------------
def test_contrast_stretch_minmax():
    img = np.array([[50, 75], [100, 60]], dtype=np.float64)
    # min=50, max=100 -> (r-50)*255/50 = (r-50)*5.1
    exp = np.array([[0.0, 127.5], [255.0, 51.0]])
    np.testing.assert_allclose(contrast_stretch(img), exp, atol=1e-9)


def test_contrast_stretch_percentile_clips():
    img = np.arange(0, 101, 10, dtype=np.float64).reshape(1, 11)  # 0,10,...,100
    # np.percentile (linear): p10 = 10, p90 = 90 -> s = (r-10)/80*255 clipped to [0,255]
    exp = np.clip((img - 10) / 80 * 255, 0, 255)  # 0->0, 20->31.875, 50->127.5, 100->255
    out = contrast_stretch(img, low_pct=10, high_pct=90)
    np.testing.assert_allclose(out, exp, atol=1e-9)
    assert out.min() == 0 and out.max() == 255


# ---------------- histogram equalization ----------------
def test_equalization_ties():
    img = np.array([[10, 10, 20], [20, 20, 30], [30, 40, 40]], dtype=np.uint8)
    # h: 10:2 20:3 30:2 40:2 ; CDF 2,5,7,9 ; CDF_min=2 ; denom 7
    # T(10)=0, T(20)=3/7*255=109.29->109, T(30)=5/7*255=182.14->182, T(40)=255
    lut = equalization_lut(img)
    assert lut.shape == (256,)
    exp = {10: 0, 20: 109, 30: 182, 40: 255}
    for k, v in exp.items():
        assert lut[k] == v
    out = hist_equalize(img)
    np.testing.assert_array_equal(out, np.array([[0, 0, 109], [109, 109, 182], [182, 255, 255]]))


def test_equalization_half_cases():
    img = np.array([[0, 0, 0], [1, 2, 2], [2, 3, 4]], dtype=np.uint8)
    # h: 0:3 1:1 2:3 3:1 4:1 ; CDF 3,4,7,8,9 ; CDF_min=3 ; denom 6
    # T(0)=0, T(1)=255/6=42.5, T(2)=4/6*255=170, T(3)=5/6*255=212.5, T(4)=255
    lut = equalization_lut(img)
    assert lut[0] == 0 and lut[2] == 170 and lut[4] == 255
    # Exact .5 ties: half-to-even (np.round) -> 42, 212 ; half-up -> 43, 213.
    assert (lut[1], lut[3]) in {(42, 212), (43, 213)}


# ---------------- mean / median ----------------
MEAN_CASES = [
    # border, {(y,x): expected}
    ("reflect101", {(0, 0): 53 / 9, (0, 1): 46 / 9, (1, 1): 61 / 9, (2, 2): 88 / 9}),
    ("replicate", {(0, 0): 26 / 9, (0, 1): 32 / 9, (1, 1): 61 / 9, (2, 2): 118 / 9}),
    ("zero", {(0, 0): 17 / 9, (0, 1): 26 / 9, (1, 1): 61 / 9, (2, 2): 44 / 9}),
]
MEDIAN_CASES = [
    # reflect101 (0,0): {1,2,2,4,4,10,10,10,10}->4 ; (0,1): {1,2,3,4,4,6,6,10,10}->4
    # (2,2): {6,6,8,8,10,10,10,10,20}->10 ; centre {1,2,3,4,6,7,8,10,20}->6
    ("reflect101", {(0, 0): 4, (0, 1): 4, (1, 1): 6, (2, 2): 10}),
    # replicate (0,0): {1,1,1,1,2,2,4,4,10}->2 ; (0,1): {1,1,2,2,3,3,4,6,10}->3
    # (2,2): {6,6,8,8,10,20,20,20,20}->10
    ("replicate", {(0, 0): 2, (0, 1): 3, (1, 1): 6, (2, 2): 10}),
    # zero (0,0): {0x5,1,2,4,10}->0 ; (0,1): {0,0,0,1,2,3,4,6,10}->2 ; (2,2): {0x5,6,8,10,20}->0
    ("zero", {(0, 0): 0, (0, 1): 2, (1, 1): 6, (2, 2): 0}),
]


@pytest.mark.parametrize("border,exp", MEAN_CASES)
def test_mean_filter_borders(border, exp):
    out = mean_filter(A, 3, border)
    assert out.shape == A.shape
    for (y, x), v in exp.items():
        assert out[y, x] == pytest.approx(v, abs=1e-9)


@pytest.mark.parametrize("border,exp", MEDIAN_CASES)
def test_median_filter_borders(border, exp):
    out = median_filter(A, 3, border)
    assert out.shape == A.shape
    for (y, x), v in exp.items():
        assert out[y, x] == pytest.approx(v, abs=1e-9)


def test_median_default_is_replicate():
    np.testing.assert_array_equal(median_filter(A, 3), median_filter(A, 3, "replicate"))


# ---------------- gaussian ----------------
# sigma=1: w(x)=exp(-x^2/2), x=-3..3 -> [.011109,.135335,.606531,1,...], sum 2.505950
W7 = np.exp(-(np.arange(-3, 4) ** 2) / 2.0)
W7 = W7 / W7.sum()  # centre .399050, +-1 .242036, +-2 .054006, +-3 .004433


def test_gaussian_kernel_sigma1_default():
    K = gaussian_kernel(1.0)
    assert K.shape == (7, 7)  # 2*ceil(3*1)+1
    assert K.sum() == pytest.approx(1.0, abs=1e-12)
    assert K[3, 3] == pytest.approx(0.399050**2, abs=1e-6)  # 0.159241
    assert K[0, 0] == pytest.approx(0.004433**2, abs=1e-8)
    np.testing.assert_allclose(K, np.outer(W7, W7), atol=1e-12)


def test_gaussian_kernel_sigma1_ksize3():
    w = np.array([np.exp(-0.5), 1.0, np.exp(-0.5)])
    w /= w.sum()  # [.274069, .451863, .274069]
    K = gaussian_kernel(1.0, 3)
    np.testing.assert_allclose(K, np.outer(w, w), atol=1e-12)
    assert K[1, 1] == pytest.approx(0.204180, abs=1e-6)


def test_gaussian_filter_impulse_and_constant():
    imp = np.zeros((13, 13))
    imp[6, 6] = 1.0
    out = gaussian_filter(imp, 1.0, border="zero")
    np.testing.assert_allclose(out[3:10, 3:10], np.outer(W7, W7), atol=1e-12)
    assert out.sum() == pytest.approx(1.0, abs=1e-12)
    const = np.full((5, 6), 42.0)
    for b in ("reflect101", "replicate"):
        np.testing.assert_allclose(gaussian_filter(const, 1.0, border=b), const, atol=1e-9)


# ---------------- dilate ----------------
def test_dilate_single_pixel_and_corner():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    exp = np.zeros((5, 5), dtype=bool)
    exp[1:4, 1:4] = True
    np.testing.assert_array_equal(dilate(m, 1).astype(bool), exp)
    assert dilate(m, 2).astype(bool).all()

    c = np.zeros((5, 5), dtype=bool)
    c[0, 0] = True
    exp = np.zeros((5, 5), dtype=bool)
    exp[:2, :2] = True  # no wrap-around to far edges
    np.testing.assert_array_equal(dilate(c, 1).astype(bool), exp)
