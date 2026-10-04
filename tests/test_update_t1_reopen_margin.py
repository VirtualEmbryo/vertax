"""Length of the edge created by a periodic T1 transition (``reopen_margin`` of ``update_T1``).

A T1 transition replaces an edge shorter than ``min_dist_T1`` by a perpendicular edge of length
``(1 + reopen_margin) * min_dist_T1``. The default margin, 0.1, is the value every committed result
was produced with; passing it explicitly must change nothing, and another margin must give the new
edge the stated length.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from vertax import PbcMesh
from vertax.energy import energy_shape_factor_homo
from vertax.geo import get_length
from vertax.topo import update_T1

WIDTH, HEIGHT, N_CELLS = 4.0, 4.0, 16
MIN_DIST_T1 = 0.05


def _energy(v, h, f, _vp, _hp, fp):  # noqa: ANN001, ANN202
    return energy_shape_factor_homo(v, h, f, WIDTH, HEIGHT, jnp.atleast_1d(fp))


def _mesh_with_one_short_edge():  # noqa: ANN202
    """A Voronoi mesh in which one edge, between two non-triangular cells, is shrunk to 0.4 l_min."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=WIDTH, height=HEIGHT, random_key=5)
    v, he, fa = np.asarray(mesh.vertices).copy(), np.asarray(mesh.edges), np.asarray(mesh.faces)
    sides = np.bincount(he[:, 5], minlength=len(fa))
    idx = next(i for i in range(0, len(he), 2)
               if sides[he[i, 5]] >= 5 and sides[he[he[i, 2], 5]] >= 5)
    src, tgt = he[idx, 3], he[idx, 4]
    offset = he[idx, 6:8] * np.array([WIDTH, HEIGHT])
    a, b = v[src, :2], v[tgt, :2] + offset
    mid, u = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
    v[src, :2] = mid - 0.2 * MIN_DIST_T1 * u
    v[tgt, :2] = mid + 0.2 * MIN_DIST_T1 * u - offset
    return jnp.asarray(v), jnp.asarray(he), jnp.asarray(fa), idx


def _flip(t1_func):  # noqa: ANN001, ANN202
    v, he, fa, idx = _mesh_with_one_short_edge()
    n_v, n_he, n_f = v.shape[0], he.shape[0], fa.shape[0]
    out = t1_func(v, he, fa, WIDTH, HEIGHT, jnp.zeros(n_v), jnp.ones(n_he), jnp.full(n_f, 3.8), _energy,
                  MIN_DIST_T1, jnp.arange(n_v), jnp.arange(n_he), jnp.arange(n_f), False)
    return out, idx, he


def test_explicit_default_margin_changes_nothing() -> None:
    (v0, h0, f0), _, _ = _flip(update_T1)
    (v1, h1, f1), _, _ = _flip(functools.partial(update_T1, reopen_margin=0.1))
    np.testing.assert_array_equal(np.asarray(v0), np.asarray(v1))
    np.testing.assert_array_equal(np.asarray(h0), np.asarray(h1))
    np.testing.assert_array_equal(np.asarray(f0), np.asarray(f1))


@pytest.mark.parametrize("margin", [0.1, 1.0, 3.0])
def test_new_edge_has_the_stated_length(margin: float) -> None:
    (v, h, f), idx, he_before = _flip(functools.partial(update_T1, reopen_margin=margin))
    assert not np.array_equal(np.asarray(h[idx, :2]), np.asarray(he_before[idx, :2])), "the edge was not flipped"
    length = float(get_length(idx, v, h, f, WIDTH, HEIGHT))
    assert length == pytest.approx((1.0 + margin) * MIN_DIST_T1, rel=1e-5)
