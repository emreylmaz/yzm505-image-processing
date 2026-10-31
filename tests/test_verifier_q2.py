"""Bağımsız doğrulayıcı (Q2): beklenen değerler küçük dizilerde ELLE türetildi.

I[y, x] = 3y + x (3×3). x = sütun, y = satır, piksel merkezleri tamsayı; A ileri eşleme, warp ters eşleme.
"""

import math

import numpy as np
import pytest

from ip.transforms import (
    expected_gradient_angles,
    rotation_matrix,
    shear_matrix,
    warp_affine,
    warp_reference,
)

I3 = np.arange(9.0).reshape(3, 3)
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))


def test_rotation_30_expand_80x80():
    # cx=cy=39.5; tuval round(80c+80s)=109; yeni merkez 54 → ek öteleme 14.5
    A, shp = rotation_matrix(30, (80, 80), True)
    tx = (1 - C30) * 39.5 - S30 * 39.5 + 14.5  # 0.041997
    ty = S30 * 39.5 + (1 - C30) * 39.5 + 14.5  # 39.541997
    np.testing.assert_allclose(A, [[C30, S30, tx], [-S30, C30, ty]], atol=1e-9)
    assert tuple(shp) == (109, 109)
    np.testing.assert_allclose(A @ [39.5, 39.5, 1], [54, 54], atol=1e-9)


def test_rotation_30_expand_nonsquare():
    # shape=(h=40, w=80): nw=round(80c+40s)=89, nh=round(80s+40c)=75
    A, shp = rotation_matrix(30, (40, 80), True)
    assert tuple(shp) == (75, 89)
    np.testing.assert_allclose(A @ [39.5, 19.5, 1], [44, 37], atol=1e-9)


def test_rotation_90_3x3():
    # x' = y, y' = 2 - x  (sağ-üst köşe → sol-üst: ekranda CCW)
    A, shp = rotation_matrix(90, (3, 3), True)
    np.testing.assert_allclose(A, [[0, 1, 0], [-1, 0, 2]], atol=1e-12)
    assert tuple(shp) == (3, 3)


@pytest.mark.parametrize("k", [0.5, 0.3])
def test_shear_matrix(k):
    A, shp = shear_matrix(k, (3, 3))
    np.testing.assert_allclose(A, [[1, k, 0], [0, 1, 0]])
    assert tuple(shp) == (3, 3 + math.ceil(k * 2))


def test_warp_rot90_both_interps():
    A, shp = rotation_matrix(90, (3, 3), True)
    exp = [[2, 5, 8], [1, 4, 7], [0, 3, 6]]  # out[y', x'] = I[x', 2-y']
    for m in ("nearest", "bilinear"):
        np.testing.assert_allclose(warp_affine(I3, A, shp, m), exp, atol=1e-9)


def test_warp_shear_bilinear():
    # k=0.5: sx = x' - 0.5 y'. Satır 1: (0+3)/2, (3+4)/2, (4+5)/2, (5+0)/2. Satır 2: sx=-1 → tamamen fill.
    A, shp = shear_matrix(0.5, (3, 3))
    exp = [[0, 1, 2, 0], [1.5, 3.5, 4.5, 2.5], [0, 6, 7, 8]]
    np.testing.assert_allclose(warp_affine(I3, A, shp, "bilinear"), exp, atol=1e-9)
    # k=0.3: satır 1 sx=x'-0.3, satır 2 sx=x'-0.6 (dışarıdaki komşu = 0)
    A, shp = shear_matrix(0.3, (3, 3))
    exp = [[0, 1, 2, 0], [2.1, 3.7, 4.7, 1.5], [2.4, 6.4, 7.4, 4.8]]
    np.testing.assert_allclose(warp_affine(I3, A, shp, "bilinear"), exp, atol=1e-9)


def test_warp_shear_nearest():
    # k=0.3: satır 1 sx∈{-0.3,0.7,1.7,2.7}→{0,1,2,dış}; satır 2 sx∈{-0.6,0.4,1.4,2.4}→{dış,0,1,2}
    A, shp = shear_matrix(0.3, (3, 3))
    exp = [[0, 1, 2, 0], [3, 4, 5, 0], [0, 6, 7, 8]]
    np.testing.assert_allclose(warp_affine(I3, A, shp, "nearest"), exp)


def test_warp_half_pixel_translation():
    A = np.array([[1.0, 0, 0.5], [0, 1, 0]])
    exp = [[0, 0.5, 1.5], [1.5, 3.5, 4.5], [3, 6.5, 7.5]]  # komşu ortalaması, ilk sütun fill ile
    np.testing.assert_allclose(warp_affine(I3, A, (3, 3), "bilinear"), exp, atol=1e-9)
    A3 = np.array([[1.0, 0, 0.3], [0, 1, 0]])
    np.testing.assert_allclose(warp_affine(I3, A3, (3, 3), "nearest"), I3)


def test_warp_reference():
    ident = np.array([[1.0, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(warp_reference(I3, ident, (3, 3), 8), I3)
    A = np.array([[1.0, 0, 0.5], [0, 1, 0]])
    for ss in (4, 8):
        r = warp_reference(I3, A, (3, 3), ss)
        np.testing.assert_allclose(r, [[0, 0.5, 1.5], [1.5, 3.5, 4.5], [3, 6.5, 7.5]], atol=1e-12)
        np.testing.assert_allclose(r * ss * ss, np.round(r * ss * ss), atol=1e-9)
    Ay = np.array([[1.0, 0, 0], [0, 1, 0.5]])
    np.testing.assert_allclose(
        warp_reference(I3, Ay, (3, 3), 4),
        [[0, 0.5, 1], [1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        atol=1e-12,
    )


def _mod180(a):
    return a % 180.0


def test_expected_gradient_angles():
    # 30° döndürme: A^{-T} = A; (1,0)→(c,-s)=-30°≡150°, (0,1)→(s,c)=60°
    a0, a90 = expected_gradient_angles(rotation_matrix(30, (80, 80), True)[0])
    assert _mod180(a0) == pytest.approx(150.0, abs=1e-9)
    assert _mod180(a90) == pytest.approx(60.0, abs=1e-9)
    # shear 0.3: A^{-T}=[[1,0],[-0.3,1]]; (1,0)→(1,-0.3)=-16.699°, (0,1)→90°
    a0, a90 = expected_gradient_angles(shear_matrix(0.3, (80, 80))[0])
    assert _mod180(a0) == pytest.approx(180 + math.degrees(math.atan2(-0.3, 1)), abs=1e-9)
    assert _mod180(a90) == pytest.approx(90.0, abs=1e-9)
