import numpy as np
import pytest

from ip import transforms as T
from ip.io import to_uint8
from ip.metrics import max_abs, mean_abs


def test_rotate_zero_is_identity(rand_u8):
    A, shape = T.rotation_matrix(0, rand_u8.shape)
    assert shape == rand_u8.shape
    for interp in ("nearest", "bilinear"):
        assert np.array_equal(T.warp_affine(rand_u8, A, shape, interp), rand_u8.astype(float))


@pytest.mark.parametrize("n", [5, 6, 31])
def test_rotate_90_nearest_equals_rot90(n):
    img = np.arange(n * n, dtype=np.float64).reshape(n, n)
    A, shape = T.rotation_matrix(90, img.shape)
    assert shape == (n, n)
    # pozitif açı = ekranda saat yönünün tersi = np.rot90(k=1)
    assert np.array_equal(T.warp_affine(img, A, shape, "nearest"), np.rot90(img))


def test_shear_zero_is_identity(rand_u8):
    A, shape = T.shear_matrix(0.0, rand_u8.shape)
    assert np.array_equal(T.warp_affine(rand_u8, A, shape, "bilinear"), rand_u8.astype(float))


def test_shear_hand_computed():
    # x' = x + 0.5·y ; satır y=2 → 1 piksel sağa kayar
    img = np.arange(12, dtype=np.float64).reshape(3, 4)
    A, shape = T.shear_matrix(0.5, img.shape)
    assert shape == (3, 5)
    out = T.warp_affine(img, A, shape, "bilinear")
    assert out[0, :4].tolist() == img[0].tolist()
    assert out[2, 1:5].tolist() == img[2].tolist() and out[2, 0] == 0.0
    # satır 1: yarım piksel kayma → komşuların ortalaması (sınırda fill=0 ile)
    assert out[1, 1] == pytest.approx((img[1, 0] + img[1, 1]) / 2)


def test_inverse_mapping_has_no_holes():
    img = np.full((40, 40), 200.0)
    A, shape = T.rotation_matrix(30, img.shape)
    out = T.warp_affine(img, A, shape, "nearest")
    inner = T.valid_mask(A, shape, img.shape)
    assert np.all(out[inner] == 200.0)  # forward mapping'in aksine delik yok


def test_expand_canvas_size():
    A, shape = T.rotation_matrix(30, (80, 80))
    assert shape == (109, 109)  # 80·cos30 + 80·sin30 = 109.28


def test_bilinear_vs_cv2_warpaffine(smooth_u8):
    A, shape = T.rotation_matrix(30, smooth_u8.shape)
    ours = to_uint8(T.warp_affine(smooth_u8, A, shape, "bilinear"))
    cv = T.warp_builtin(smooth_u8, A, shape, "bilinear")
    m = T.valid_mask(A, shape, smooth_u8.shape)
    inner = m & np.roll(m, 2, 0) & np.roll(m, -2, 0) & np.roll(m, 2, 1) & np.roll(m, -2, 1)
    # cv2 alt-piksel konumunu 1/32'ye niceler (INTER_TAB_SIZE) → küçük farklar beklenir
    assert max_abs(ours[inner], cv[inner]) <= 2
    assert mean_abs(ours[inner], cv[inner]) < 0.5


def test_nearest_vs_cv2_mostly_equal(smooth_u8):
    A, shape = T.rotation_matrix(30, smooth_u8.shape)
    ours = to_uint8(T.warp_affine(smooth_u8, A, shape, "nearest"))
    cv = T.warp_builtin(smooth_u8, A, shape, "nearest")
    assert np.mean(ours != cv) < 0.02  # yalnızca .5 sınırına çok yakın örneklerde farklı


def test_reference_is_smooth_between_nn_and_bilinear():
    cb = np.kron(np.array([[0, 255], [255, 0]] * 2), np.ones((8, 8)))
    cb = np.tile(cb, (1, 2))
    A, shape = T.rotation_matrix(30, cb.shape)
    ref = T.warp_reference(cb, A, shape, ss=8)
    assert ref.min() >= 0 and ref.max() <= 255
    nn = T.warp_affine(cb, A, shape, "nearest")
    bl = T.warp_affine(cb, A, shape, "bilinear")
    m = T.valid_mask(A, shape, cb.shape)
    assert mean_abs(bl[m], ref[m]) < mean_abs(nn[m], ref[m])


def test_expected_gradient_angles():
    A, _ = T.rotation_matrix(30, (80, 80))
    a, b = T.expected_gradient_angles(A)
    assert a == pytest.approx(150.0) and b == pytest.approx(60.0)
    S, _ = T.shear_matrix(0.3, (80, 80))
    a, b = T.expected_gradient_angles(S)
    assert a == pytest.approx(180 - np.degrees(np.arctan(0.3))) and b == pytest.approx(90.0)
