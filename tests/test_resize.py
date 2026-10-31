import numpy as np
import pytest

from ip.io import to_uint8
from ip.metrics import max_abs
from ip.resize import (
    bilinear_resize_loops,
    bilinear_resize_vec,
    output_shape,
    resize_builtin,
)


def test_output_shape_non_integer_factor():
    assert output_shape((256, 256), 1.7) == (435, 435)  # 435.2 -> 435
    assert output_shape((10, 3), 1.5) == (15, 5)  # 4.5 -> 5 (half-up)


@pytest.mark.parametrize("conv", ["half_pixel", "naive"])
@pytest.mark.parametrize("scale", [1.7, 0.6, 2.0])
def test_loops_equal_vectorized_exact(rand_u8, conv, scale):
    a = bilinear_resize_loops(rand_u8, scale, conv)
    b = bilinear_resize_vec(rand_u8, scale, conv)
    assert a.shape == b.shape
    assert np.array_equal(a, b)  # bit-exact


def test_hand_computed_weights(ramp4):
    # 4×4 ramp, ×2 → 8×8, half_pixel: oran 0.5; çıkış (y=1,x=1) -> src = 1.5*0.5-0.5 = 0.25
    # dx=dy=0.25: 0.5625*I00 + 0.1875*I10 + 0.1875*I01 + 0.0625*I11 = 0.5625*0 + 0.1875*10 + 0.1875*40 + 0.0625*50
    out = bilinear_resize_loops(ramp4, 2.0, "half_pixel")
    assert out[1, 1] == pytest.approx(0.1875 * 10 + 0.1875 * 40 + 0.0625 * 50)
    # Lineer ramp'te bilinear tam lineer kalır: iç noktada değer = 10*(4*sy + sx)
    assert out[3, 5] == pytest.approx(10 * (4 * (3.5 * 0.5 - 0.5) + (5.5 * 0.5 - 0.5)))
    # naive: (y=2,x=3) -> src=(1.0,1.5) -> 10*(4*1 + 1.5)
    assert bilinear_resize_loops(ramp4, 2.0, "naive")[2, 3] == pytest.approx(55.0)


def test_constant_image_stays_constant():
    img = np.full((5, 7), 123.0)
    # ağırlık toplamı float'ta 1'e ancak ulp düzeyinde eşit → tolerans gerekli
    assert np.allclose(bilinear_resize_loops(img, 1.7), 123.0, rtol=0, atol=1e-9)
    assert np.all(to_uint8(bilinear_resize_loops(img, 1.7)) == 123)


def test_half_pixel_matches_cv2_within_1(smooth_u8, rand_u8):
    for img in (smooth_u8, rand_u8):
        ours = to_uint8(bilinear_resize_vec(img, 1.7, "half_pixel"))
        cv = resize_builtin(img, 1.7)
        assert ours.shape == cv.shape
        assert max_abs(ours, cv) <= 1  # fark: cv2 11-bit fixed-point ağırlıklar


def test_naive_convention_differs_from_cv2(smooth_u8):
    ours = to_uint8(bilinear_resize_vec(smooth_u8, 1.7, "naive"))
    assert max_abs(ours, resize_builtin(smooth_u8, 1.7)) > 1  # beklenen, açıklanabilir fark
