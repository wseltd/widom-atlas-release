"""Tests for widom_atlas.pbc.* (T017, T018, T019).

Covers wrap_frac, cart_to_frac, frac_to_cart, min_image_displacement,
min_image_distance, expand_27_images, collapse_to_primary.
"""

from __future__ import annotations

import time

import numpy as np

from widom_atlas.pbc.expansion import collapse_to_primary, expand_27_images
from widom_atlas.pbc.minimum_image import min_image_displacement, min_image_distance
from widom_atlas.pbc.wrap import cart_to_frac, frac_to_cart, wrap_frac

# --- T017 wrap / frac↔cart ---------------------------------------------------


def test_wrap_frac_idempotent() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(0.0, 5.0, (50, 3))
    once = wrap_frac(f)
    twice = wrap_frac(once)
    assert ((once >= 0.0) & (once < 1.0)).all()
    np.testing.assert_array_equal(once, twice)


def test_wrap_frac_handles_negative() -> None:
    f = np.array([[-0.1, -1.7, 0.2]])
    out = wrap_frac(f)
    np.testing.assert_allclose(out, [[0.9, 0.3, 0.2]], atol=1e-12)


def test_cart_frac_roundtrip_triclinic() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, 0.5, 7.0]])
    rng = np.random.default_rng(1)
    frac = rng.random((20, 3))
    cart = frac_to_cart(frac, cell)
    back = cart_to_frac(cart, cell)
    np.testing.assert_allclose(back, frac, atol=1e-10)


def test_frac_to_cart_orthorhombic() -> None:
    cell = np.diag([5.0, 6.0, 7.0])
    frac = np.array([[0.5, 0.5, 0.5]])
    cart = frac_to_cart(frac, cell)
    np.testing.assert_allclose(cart, [[2.5, 3.0, 3.5]], atol=1e-12)


# --- T018 minimum-image ------------------------------------------------------


def test_min_image_distance_orthorhombic_boundary() -> None:
    cell = np.eye(3) * 10.0
    a = np.array([0.05, 0.05, 0.05])
    b = np.array([0.95, 0.95, 0.95])
    d = float(min_image_distance(a, b, cell))
    assert abs(d - np.sqrt(3.0)) < 1e-10


def test_min_image_distance_triclinic_skewed() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [4.0, 5.0, 0.0], [3.0, 3.0, 5.0]])
    a = np.array([0.01, 0.01, 0.01])
    b = np.array([0.99, 0.99, 0.99])
    d = float(min_image_distance(a, b, cell))
    direct = float(np.linalg.norm((a - b) @ cell))
    assert d < direct


def test_min_image_displacement_broadcasting() -> None:
    cell = np.eye(3) * 10.0
    a = np.zeros((4, 3))
    b = np.array([[0.05, 0.05, 0.05]])
    disp = min_image_displacement(a, b, cell)
    assert disp.shape == (4, 3)


def test_min_image_zero_distance_self() -> None:
    cell = np.eye(3) * 8.0
    p = np.array([0.3, 0.4, 0.5])
    d = float(min_image_distance(p, p, cell))
    assert d == 0.0


def test_min_image_27image_fallback_skewed_cell() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [4.5, 5.0, 0.0], [4.5, 4.5, 5.0]])
    a = np.array([0.01, 0.01, 0.01])
    b = np.array([0.99, 0.99, 0.99])
    d = float(min_image_distance(a, b, cell))
    naive = float(np.linalg.norm((a - b) @ cell))
    assert d <= naive + 1e-9


# --- T019 27-image expansion + collapse --------------------------------------


def test_expand_27_images_shape() -> None:
    cell = np.eye(3) * 10.0
    frac = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    e = np.array([-1.0, -2.0])
    fe, ce, pi = expand_27_images(frac, e, cell)
    assert fe.shape == (54, 3)
    assert ce.shape == (54, 3)
    assert pi.shape == (54,)


def test_expand_27_images_parent_indices_correct() -> None:
    cell = np.eye(3) * 10.0
    frac = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    e = np.array([-1.0, -2.0])
    _, _, pi = expand_27_images(frac, e, cell)
    assert set(np.unique(pi).tolist()) == {0, 1}
    assert int((pi == 0).sum()) == 27
    assert int((pi == 1).sum()) == 27


def test_expand_27_images_energies_replicated() -> None:
    cell = np.eye(3) * 10.0
    frac = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    e = np.array([-1.0, -2.0])
    _, _, pi = expand_27_images(frac, e, cell)
    e_expanded = e[pi]
    assert e_expanded.shape == (54,)
    assert (e_expanded[pi == 0] == -1.0).all()
    assert (e_expanded[pi == 1] == -2.0).all()


def test_collapse_to_primary_merges_boundary_basin() -> None:
    cell = np.eye(3) * 10.0
    frac = np.array([[0.05, 0.5, 0.5], [0.95, 0.5, 0.5]])
    e = np.array([-1.0, -1.1])
    _, _, pi = expand_27_images(frac, e, cell)
    labels = np.full(54, -1, dtype=np.int64)
    labels[pi == 0] = 0
    labels[pi == 1] = 1
    merged = collapse_to_primary(labels, pi)
    assert merged.shape == (54,)


def test_expand_vectorised_no_python_loop_perf() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    frac = rng.random((n, 3))
    e = rng.normal(0.0, 1.0, n)
    cell = np.eye(3) * 10.0
    t0 = time.perf_counter()
    fe, ce, pi = expand_27_images(frac, e, cell)
    elapsed = time.perf_counter() - t0
    assert fe.shape == (n * 27, 3)
    assert elapsed < 1.0, f"expand_27_images took {elapsed:.3f}s on {n} points"
