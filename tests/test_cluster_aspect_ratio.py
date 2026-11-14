"""Aspect ratio of a bounded cluster from its area moments, and the two costs built on it.

Hand-computed references:

* a disk of radius rho: a regular N-gon inscribed in it, each side capped by an arc of half-angle
  pi/N, covers the disk exactly, so the area is pi rho^2, the central second moments are
  (pi rho^4 / 4) I and both aspect ratios are 1. This checks the circular caps.
* an ellipse of semi-axes a along x and b along y: a polygon of many vertices on it, with arcs so
  flat that their caps are negligible, has central second moments (pi a b / 4) diag(a^2, b^2) to
  O(1/N^2), so r_x = a/b. Turned by phi, r_x follows the closed form in the docstring of
  `aspect_ratio_along_x`, while the principal-axis ratio stays a/b.

Faces are traversed counterclockwise (`BoundedMesh` builds them so) and every cap lies to the right
of its directed chord. Each shape is also checked with the cell traversed clockwise, which the
library reads inside out: the polygon counts negatively and the caps lie inside it. For the disk,
the area is then ``pi rho^2 - 2 P`` (``P`` the polygon's area) and the moments stay isotropic about
the same centroid; for the ellipse, whose caps are negligible, every moment changes sign, so ``r_x``
is unchanged. On a generated mesh, the area must equal the sum of the cell
areas of `get_area_bounded`, and the gradients of both costs with respect to the vertices and the
arc angles must match central finite differences. All in float64, inside `enable_x64` so that the
rest of the suite keeps its own precision.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax import BoundedMesh
from vertax.cost import (
    aspect_ratio_along_x,
    aspect_ratio_principal,
    make_cost_aspect_ratio,
    make_cost_aspect_ratio_along_x,
)
from vertax.geo import get_area_bounded, get_cluster_moments_bounded


def _single_cell(points: np.ndarray, half_angle: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One cell whose every side is a free interface, in the half-edge layout of `BoundedMesh`.

    Edge k joins point k to point k+1. Its half-edge 2k lies in the cell (a surface half-edge: the
    vertices are in columns 5 and 6, offset by 2), and its twin 2k+1 belongs to no cell.
    """
    n = len(points)
    he = np.zeros((2 * n, 8), dtype=np.int32)
    he[:, 4] = 1
    he[:, 6] = 1
    for k in range(n):
        inside, outside = 2 * k, 2 * k + 1
        he[inside] = [2 * ((k - 1) % n), 2 * ((k + 1) % n), outside, 0, 1, k + 2, (k + 1) % n + 2, 0]
        he[outside, 2] = inside
    return jnp.asarray(points), jnp.full((n,), half_angle), jnp.asarray(he)


def _polygon(center, rx, ry, n, phi=0.0, clockwise=False):
    t = 2 * np.pi * np.arange(n) / n
    if clockwise:
        t = -t
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([center[0] + c * x - s * y, center[1] + s * x + c * y], axis=1)


@pytest.mark.parametrize("clockwise", [False, True])
def test_disk_from_polygon_and_caps(clockwise):
    with enable_x64():
        rho, n, center = 1.3, 7, (2.0, -1.0)
        vt, ang, he = _single_cell(_polygon(center, rho, rho, n, clockwise=clockwise), np.pi / n)
        area, centroid, s = get_cluster_moments_bounded(vt, ang, he)
        polygon = 0.5 * n * rho**2 * np.sin(2 * np.pi / n)
        np.testing.assert_allclose(float(area), np.pi * rho**2 - (2 * polygon if clockwise else 0.0), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(centroid), center, atol=1e-12)
        if clockwise:  # inside out: isotropic, of no closed form worth stating here
            np.testing.assert_allclose(float(s[0, 1]), 0.0, atol=1e-12)
            np.testing.assert_allclose(float(s[0, 0]), float(s[1, 1]), rtol=1e-12)
        else:
            np.testing.assert_allclose(np.asarray(s), np.pi * rho**4 / 4 * np.eye(2), atol=1e-12)
        np.testing.assert_allclose(float(aspect_ratio_along_x(vt, ang, he)), 1.0, rtol=1e-12)
        np.testing.assert_allclose(float(aspect_ratio_principal(vt, ang, he)), 1.0, rtol=1e-6)


@pytest.mark.parametrize("clockwise", [False, True])
@pytest.mark.parametrize("phi", [0.0, np.pi / 6])
def test_ellipse(clockwise, phi):
    with enable_x64():
        a, b, n = 2.0, 1.0, 4000
        vt, ang, he = _single_cell(_polygon((0.5, 3.0), a, b, n, phi, clockwise), 1e-6)
        area, _, s = get_cluster_moments_bounded(vt, ang, he)
        sense = -1.0 if clockwise else 1.0  # clockwise: read inside out, every moment changes sign
        np.testing.assert_allclose(float(area), sense * np.pi * a * b, rtol=1e-5)
        expected_x = np.sqrt(
            (a**2 * np.cos(phi) ** 2 + b**2 * np.sin(phi) ** 2) / (a**2 * np.sin(phi) ** 2 + b**2 * np.cos(phi) ** 2)
        )
        np.testing.assert_allclose(float(aspect_ratio_along_x(vt, ang, he)), expected_x, rtol=1e-5)
        if phi == 0.0:
            expected = sense * np.pi * a * b / 4 * np.diag([a**2, b**2])
            np.testing.assert_allclose(np.asarray(s), expected, rtol=1e-5, atol=1e-6)
        assert float(make_cost_aspect_ratio_along_x(2.5)(vt, ang, he)) == pytest.approx((expected_x - 2.5) ** 2)
        if not clockwise:  # the principal ratio assumes a positive area
            np.testing.assert_allclose(float(aspect_ratio_principal(vt, ang, he)), a / b, rtol=1e-5)
            assert float(make_cost_aspect_ratio(2.5)(vt, ang, he)) == pytest.approx((a / b - 2.5) ** 2, rel=1e-4)


def _generated_mesh(n_cells, seed):
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=n_cells**0.5, height=n_cells**0.5, random_key=seed)
    return (
        jnp.asarray(np.asarray(mesh.vertices, dtype=np.float64)),
        jnp.asarray(np.asarray(mesh.angles, dtype=np.float64)),
        mesh.edges,
        mesh.faces,
    )


@pytest.mark.parametrize("seed", [1, 2])
def test_area_matches_cell_areas(seed):
    with enable_x64():
        vt, ang, he, faces = _generated_mesh(12, seed)
        padded = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), vt])
        cells = jax.vmap(lambda f: get_area_bounded(f, padded, jnp.repeat(ang, 2), he, faces))(jnp.arange(faces.shape[0]))
        area, _, _ = get_cluster_moments_bounded(vt, ang, he)
        np.testing.assert_allclose(float(area), float(jnp.sum(cells)), rtol=1e-10)


@pytest.mark.parametrize("factory", [make_cost_aspect_ratio_along_x, make_cost_aspect_ratio])
def test_gradient_against_finite_differences(factory):
    with enable_x64():
        vt, ang, he, faces = _generated_mesh(12, 3)
        cost = factory(2.0)
        g_v, g_a = jax.grad(lambda v, a: cost(v, a, he, faces), argnums=(0, 1))(vt, ang)
        rng = np.random.default_rng(0)
        h = 1e-6
        for _ in range(6):
            dv = jnp.asarray(rng.standard_normal(vt.shape))
            da = jnp.asarray(rng.standard_normal(ang.shape))  # inner edges' angles do not enter: zero both ways
            fd = (cost(vt + h * dv, ang + h * da, he, faces) - cost(vt - h * dv, ang - h * da, he, faces)) / (2 * h)
            ad = jnp.sum(g_v * dv) + jnp.sum(g_a * da)
            np.testing.assert_allclose(float(ad), float(fd), rtol=1e-6, atol=1e-10)
