"""Independent verifier checks for Q3 — expected values derived BY HAND (see comments).

Bilinear: out = (1-dx)(1-dy)I(x0,y0) + dx(1-dy)I(x1,y0) + (1-dx)dy I(x0,y1) + dx dy I(x1,y1)
half_pixel: src=(dst+0.5)*ratio-0.5, naive: src=dst*ratio, ratio=N_in/N_out, src clamped to [0,N-1].
"""

import numpy as np
import pytest

from ip.filters import laplacian_loops, sharpen
from ip.io import to_uint8
from ip.resize import bilinear_resize_loops, bilinear_resize_vec

CHECKER = np.array([[0, 255], [255, 0]], dtype=np.float64)
RAMP = np.arange(16, dtype=np.float64).reshape(4, 4)  # I[y,x] = 4y + x
A = np.array([[1, 2, 3], [4, 10, 6], [7, 8, 20]], dtype=np.float64)

# (img, scale, convention, shape, {(y,x): expected})
RESIZE_CASES = [
    # checker x2 half_pixel: coords [0(clamp -0.25), .25, .75, 1(clamp 1.25)]
    (
        "checker",
        2.0,
        "half_pixel",
        (4, 4),
        {
            (0, 0): 0.0,
            (0, 1): 63.75,  # dx=.25: .75*0 + .25*255
            (1, 1): 95.625,  # w=(.5625,.1875,.1875,.0625) on (0,255,255,0)
            (1, 2): 159.375,  # dx=.75,dy=.25: w=(.1875,.5625,.0625,.1875)
            (0, 3): 255.0,  # sx clamped to 1, x0=x1=1
            (3, 3): 0.0,
        },
    ),
    # checker x2 naive: coords [0, .5, 1, 1(clamp 1.5)]
    (
        "checker",
        2.0,
        "naive",
        (4, 4),
        {
            (1, 1): 127.5,
            (0, 1): 127.5,
            (2, 1): 127.5,
            (3, 3): 0.0,
            (0, 3): 255.0,
        },
    ),
    # checker x1.7: N_out=floor(3.4+.5)=3, ratio=2/3; half_pixel coords [0, .5, 1(clamp 7/6)]
    (
        "checker",
        1.7,
        "half_pixel",
        (3, 3),
        {
            (0, 0): 0.0,
            (1, 1): 127.5,
            (0, 1): 127.5,
            (0, 2): 255.0,
            (2, 0): 255.0,
            (2, 2): 0.0,
        },
    ),
    # naive coords [0, 2/3, 1(clamp 4/3)]; (1,1): w=(1/9,2/9,2/9,4/9) -> 255*4/9
    (
        "checker",
        1.7,
        "naive",
        (3, 3),
        {
            (1, 1): 255 * 4 / 9,
            (0, 1): 170.0,
            (1, 0): 170.0,
            (2, 2): 0.0,
        },
    ),
    # ramp x2 half_pixel: 8x8, coords 0.5d-0.25 clamped -> out = 4*sy + sx (bilinear exact on affine)
    (
        "ramp",
        2.0,
        "half_pixel",
        (8, 8),
        {
            (0, 0): 0.0,
            (1, 1): 1.25,
            (4, 3): 8.25,
            (7, 7): 15.0,
            (0, 7): 3.0,
        },
    ),
    # ramp x1.7: N_out=floor(6.8+.5)=7, ratio=4/7; half_pixel coords
    # [0, 5/14, 13/14, 1.5, 29/14, 37/14, 3]
    (
        "ramp",
        1.7,
        "half_pixel",
        (7, 7),
        {
            (0, 0): 0.0,
            (3, 3): 7.5,  # x0=y0=1, dx=dy=.5: (5+6+9+10)/4
            (1, 2): 4 * 5 / 14 + 13 / 14,  # = 2.357142857
            (5, 4): 4 * 37 / 14 + 29 / 14,  # = 12.642857
            (0, 6): 3.0,
            (6, 6): 15.0,
        },
    ),
    # naive coords d*4/7, last clamped to 3
    (
        "ramp",
        1.7,
        "naive",
        (7, 7),
        {
            (1, 1): 5 * 4 / 7,
            (6, 0): 12.0,
            (5, 5): 5 * 20 / 7,
            (6, 6): 15.0,
        },
    ),
]


@pytest.mark.parametrize("name,scale,conv,shape,expected", RESIZE_CASES)
def test_bilinear_hand_values(name, scale, conv, shape, expected):
    img = CHECKER if name == "checker" else RAMP
    out = bilinear_resize_loops(img, scale, conv)
    assert out.shape == shape
    for (y, x), v in expected.items():
        assert out[y, x] == pytest.approx(v, abs=1e-9), (y, x)


@pytest.mark.parametrize("name,scale,conv,shape,expected", RESIZE_CASES)
def test_bilinear_vec_equals_loops(name, scale, conv, shape, expected):
    img = CHECKER if name == "checker" else RAMP
    np.testing.assert_array_equal(
        bilinear_resize_vec(img, scale, conv), bilinear_resize_loops(img, scale, conv)
    )


def test_bilinear_vec_equals_loops_random():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (13, 17)).astype(np.float64)
    for scale in (0.5, 1.3, 1.7, 2.0, 3.1):
        for conv in ("half_pixel", "naive"):
            np.testing.assert_array_equal(
                bilinear_resize_vec(img, scale, conv), bilinear_resize_loops(img, scale, conv)
            )


# L = up + down + left + right - 4c on A; out-of-range neighbours per border mode.
LAP_CASES = {
    "reflect101": {(0, 0): 8, (0, 1): 16, (2, 2): -52, (1, 1): -20, (1, 2): 19, (2, 0): -4},
    "replicate": {(0, 0): 4, (0, 1): 8, (2, 2): -26, (1, 1): -20, (1, 2): 15, (2, 0): -2},
    "zero": {(0, 0): 2, (0, 1): 6, (2, 2): -66, (1, 1): -20, (1, 2): 9, (2, 0): -16},
}


@pytest.mark.parametrize("border", list(LAP_CASES))
def test_laplacian_hand_values(border):
    out = laplacian_loops(A, border=border)
    for (y, x), v in LAP_CASES[border].items():
        assert out[y, x] == v, (border, y, x)


@pytest.mark.parametrize("border,corner", [("reflect101", -4), ("replicate", -2), ("zero", -4)])
def test_laplacian_corner_impulse_4x4(border, corner):
    img = np.zeros((4, 4))
    img[0, 0] = 1.0
    exp = np.zeros((4, 4))
    exp[0, 0] = corner
    exp[0, 1] = exp[1, 0] = 1.0
    np.testing.assert_array_equal(laplacian_loops(img, border=border), exp)


def test_laplacian_uint8_input_no_overflow():
    out = laplacian_loops(np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8))
    assert out[1, 1] == -1020 and out.dtype == np.float64


def test_sharpen_hand_values():
    lap = laplacian_loops(A)  # reflect101
    s1 = sharpen(A, lap, 1.0)
    assert s1[0, 0] == 1 - 8 and s1[1, 1] == 10 + 20 and s1[2, 2] == 20 + 52
    assert sharpen(A, lap, 0.5)[0, 0] == 1 - 4


def test_to_uint8():
    x = np.array([-5.0, 0.0, 0.4, 0.6, 1.49, 254.6, 255.0, 300.0])
    np.testing.assert_array_equal(to_uint8(x), np.array([0, 0, 0, 1, 1, 255, 255, 255]))
    assert to_uint8(x).dtype == np.uint8
