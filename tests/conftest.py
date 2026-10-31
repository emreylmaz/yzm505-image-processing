import numpy as np
import pytest


@pytest.fixture
def ramp4():
    """4×4 ramp: değer = 10·(y*4 + x)."""
    return np.arange(16, dtype=np.float64).reshape(4, 4) * 10.0


@pytest.fixture
def rand_u8():
    rng = np.random.default_rng(505)
    return rng.integers(0, 256, size=(37, 41), dtype=np.uint8)


@pytest.fixture
def smooth_u8():
    """Doğal görüntüye benzer pürüzsüz desen (resize karşılaştırmaları için)."""
    y, x = np.mgrid[0:48, 0:56]
    f = 127.5 + 100 * np.sin(x / 6.0) * np.cos(y / 9.0) + 0.5 * x
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)
