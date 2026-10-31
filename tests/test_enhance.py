import numpy as np
import pytest

from ip.enhance import (
    cdf_linearity_max_dev,
    contrast_stretch,
    equalization_lut,
    equalize_builtin_cv2,
    equalize_builtin_skimage,
    hist_equalize,
    histogram,
)
from ip.io import to_uint8
from ip.metrics import max_abs


@pytest.fixture
def low_contrast():
    rng = np.random.default_rng(1)
    return rng.normal(120, 12, size=(64, 64)).clip(80, 170).astype(np.uint8)


def test_contrast_stretch_known_range():
    img = np.arange(50, 101, dtype=np.uint8).reshape(1, -1)  # [50..100]
    out = contrast_stretch(img)
    assert out[0, 0] == 0.0 and out[0, -1] == 255.0
    assert np.allclose(out, (img.astype(float) - 50) / 50 * 255)  # lineer


def test_contrast_stretch_full_range(low_contrast):
    out = to_uint8(contrast_stretch(low_contrast))
    assert out.min() == 0 and out.max() == 255


def test_contrast_stretch_percentile_clips(low_contrast):
    out = contrast_stretch(low_contrast, 2, 98)
    assert out.min() == 0.0 and out.max() == 255.0


def test_contrast_stretch_constant_image():
    assert np.all(contrast_stretch(np.full((3, 3), 7, np.uint8)) == 0)


def test_hist_equalize_range_and_monotonic_lut(low_contrast):
    out = to_uint8(hist_equalize(low_contrast))
    assert out.min() == 0 and out.max() == 255
    lut = equalization_lut(low_contrast)
    assert np.all(np.diff(lut) >= 0)  # monoton artan


def test_hist_equalize_hand_computed():
    # 4 piksel: değerler 10,10,20,30 → h: 2,1,1; CDF: 2,3,4; CDF_min=2, N=4
    # T(10)=0, T(20)=round(1/2*255)=128 (127.5 half-up), T(30)=255
    img = np.array([[10, 10], [20, 30]], dtype=np.uint8)
    assert hist_equalize(img).tolist() == [[0.0, 0.0], [128.0, 255.0]]


def test_hist_equalize_cdf_approximately_linear(low_contrast):
    before = cdf_linearity_max_dev(low_contrast)
    after = cdf_linearity_max_dev(to_uint8(hist_equalize(low_contrast)))
    assert after < 0.05 < before


def test_hist_equalize_vs_cv2(low_contrast):
    ours = to_uint8(hist_equalize(low_contrast))
    assert max_abs(ours, equalize_builtin_cv2(low_contrast)) <= 1  # float32 ölçek / yuvarlama


def test_hist_equalize_vs_skimage_differs_by_cdf_min(low_contrast):
    # skimage CDF_min çıkarmaz → en koyu seviye 0'a gitmez; fark beklenir ve raporlanır
    from skimage import exposure

    sk_float = exposure.equalize_hist(low_contrast, nbins=256)
    assert sk_float.min() > 0  # = CDF_min / N (uint8'e yuvarlanınca 0 görünebilir)
    sk = equalize_builtin_skimage(low_contrast)
    assert max_abs(to_uint8(hist_equalize(low_contrast)), sk) > 0


def test_histogram_counts():
    assert histogram(np.array([[0, 0, 255]], np.uint8))[[0, 255]].tolist() == [2, 1]
