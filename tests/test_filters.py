import numpy as np
import pytest

from ip import filters as F
from ip.io import to_uint8
from ip.metrics import max_abs


# ----------------------------------------------------------------------------- padding
def test_pad_modes_match_definitions():
    row = np.array([[1.0, 2.0, 3.0]])
    assert F.pad(row, 2, "reflect101")[2].tolist() == [3, 2, 1, 2, 3, 2, 1]
    assert F.pad(row, 2, "replicate")[2].tolist() == [1, 1, 1, 2, 3, 3, 3]
    assert F.pad(row, 2, "zero")[2].tolist() == [0, 0, 1, 2, 3, 0, 0]


# ----------------------------------------------------------------------------- Laplacian
def test_laplacian_impulse_hand_computed():
    img = np.zeros((5, 5))
    img[2, 2] = 1.0
    lap = F.laplacian_loops(img)
    assert lap[2, 2] == -4.0
    assert lap[1, 2] == lap[3, 2] == lap[2, 1] == lap[2, 3] == 1.0
    assert lap.sum() == 0.0


def test_laplacian_linear_ramp_is_zero_inside(ramp4):
    assert np.all(F.laplacian_loops(ramp4)[1:-1, 1:-1] == 0.0)


@pytest.mark.parametrize("border", ["reflect101", "replicate", "zero"])
def test_laplacian_loops_equal_vec(rand_u8, border):
    assert np.array_equal(F.laplacian_loops(rand_u8, border=border), F.laplacian_vec(rand_u8, border=border))


@pytest.mark.parametrize("border", ["reflect101", "replicate", "zero"])
def test_laplacian_loops_vs_filter2d_exact(rand_u8, border):
    ours = F.laplacian_loops(rand_u8, border=border)
    assert max_abs(ours, F.laplacian_builtin(rand_u8, border=border)) == 0.0


def test_laplacian_vs_cv2_laplacian_ksize1(rand_u8):
    assert max_abs(F.laplacian_loops(rand_u8), F.laplacian_cv(rand_u8)) == 0.0


def test_uint8_laplacian_saturates(rand_u8):
    # uint8 filter2D negatifleri 0'a keser → float sonuçtan farklı (overflow gotcha)
    f = F.laplacian_loops(rand_u8)
    assert f.min() < 0
    assert max_abs(to_uint8(f), F.laplacian_uint8_builtin(rand_u8)) == 0.0  # aynı = doygunluk
    assert max_abs(f, F.laplacian_uint8_builtin(rand_u8)) > 0


# ----------------------------------------------------------------------------- mean / median / gaussian
def test_mean_hand_computed(ramp4):
    # 3×3 ortalama, lineer ramp'in iç noktasında merkez değeri verir
    assert F.mean_filter(ramp4, 3)[1, 1] == pytest.approx(ramp4[1, 1])


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("border", ["reflect101", "replicate"])
def test_mean_vs_builtin(rand_u8, k, border):
    ours = to_uint8(F.mean_filter(rand_u8, k, border))
    assert max_abs(ours, F.mean_builtin(rand_u8, k, border)) <= 1


@pytest.mark.parametrize("k", [3, 5])
def test_median_vs_builtin(rand_u8, k):
    ours = to_uint8(F.median_filter(rand_u8, k, "replicate"))
    assert max_abs(ours, F.median_builtin(rand_u8, k)) <= 1


def test_median_removes_salt_pepper():
    img = np.full((7, 7), 100.0)
    img[3, 3] = 255.0
    img[1, 5] = 0.0
    assert np.all(F.median_filter(img, 3) == 100.0)


def test_gaussian_kernel_properties():
    k = F.gaussian_kernel(1.0)
    assert k.shape == (7, 7)
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k.T) and k[3, 3] == k.max()


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_gaussian_vs_builtin(rand_u8, sigma):
    ours = to_uint8(F.gaussian_filter(rand_u8, sigma))
    assert max_abs(ours, F.gaussian_builtin(rand_u8, sigma)) <= 1
