"""A free-interface cell whose chord polygon has turned over: area and energy gradient.

The cell of a bounded mesh is its chord polygon (the polygon of its vertices) plus a circular cap on
every free-interface edge, bulging away from the cell. Faces are traversed counterclockwise, so a
cap lies to the right of the directed chord of the cell's half-edge. A cell is a *crescent* when its
interior vertices have crossed the chord of its arc: the chord polygon, traversed counterclockwise,
then has a negative signed area, while the cell, cap included, is still a simple region of positive
area, ``cap - |polygon|``. Reading the polygon's sign as the cell's orientation instead turns the
polygon over and puts the cap on the other side of its chord, which gives ``cap + |polygon|``.

The fixture is a two-cell cluster with closed-form geometry::

          arc (half-angle alpha) over the chord C -> A
        A ----------- B ----------- C        crescent X: A -> B -> C -> arc -> A, with B inside the cap
         \\                         /
          \\  arcs (half-angle beta) /         cell Y:     A -> D -> C -> B -> A
           \\                     /
                      D

* area of X: ``cap(C, A, alpha) - triangle(A, B, C)``;
* area of Y: ``polygon(A, D, C, B) + cap(A, D, beta) + cap(D, C, beta)``;

The energy gradient is compared with central finite differences of the closed-form energy, not of
the library's. Each case
is checked on the configuration and on its mirror image (relabelled so that the faces stay
counterclockwise), turned and shifted, and from two different starting half-edges.

The convention itself is checked last: `BoundedMesh.from_seeds` builds every face counterclockwise,
and a relaxation with T1 transitions keeps the half-edge structure consistently oriented (every
interior edge is run in opposite senses by its two cells), so every cell keeps that orientation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import BoundedBilevelOptimizer, BoundedMesh
from vertax.energy import make_energy_line_tensions_bounded
from vertax.geo import get_area_bounded

K_AREAS = 40.0
TARGET_AREA = 0.6
ALPHA, BETA = 1.2, 0.5
TENSIONS = np.array([1.3, 1.7, 1.1, 1.5, 1.9])  # one per edge: A-B, B-C, arc C-A, arc A-D, arc D-C


@pytest.fixture(autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    with enable_x64():
        yield


def _points(mirror: bool, turn: float = 0.4, shift=(0.7, -2.3)) -> np.ndarray:
    """A, B, C, D, turned by ``turn`` and shifted. B lies inside the cap over C -> A."""
    a, b, c, d = np.array([-1.0, 0.0]), np.array([0.25, 0.35]), np.array([1.0, 0.0]), np.array([-0.2, -1.0])
    if mirror:  # reflect x -> -x and swap A and C, so that both cells stay counterclockwise
        a, b, c, d = (np.array([-p[0], p[1]]) for p in (c, b, a, d))
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    return np.stack([a, b, c, d]) @ rot.T + np.asarray(shift)


def _tables(start_x: int, start_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-edges, arc half-angles (one per edge) and faces of the two-cell cluster.

    Edge k is the pair of half-edges (2k, 2k+1); the even one lies in a cell. Vertex ids are offset
    by 2 (rows 0 and 1 of the padded vertex table are reserved), as in `BoundedMesh`.
    """
    A, B, C, D = 2, 3, 4, 5
    he = np.zeros((10, 8), dtype=np.int32)
    he[:, 4] = 1
    he[:, 6] = 1
    #           prev next twin  in-src in-tgt  out-src out-tgt face
    he[0] = [4, 2, 1, A, B, 0, 1, 0]  # A -> B, cell X
    he[2] = [0, 4, 3, B, C, 0, 1, 0]  # B -> C, cell X
    he[4] = [2, 0, 5, 0, 1, C, A, 0]  # C -> A, free interface of X
    he[6] = [1, 8, 7, 0, 1, A, D, 1]  # A -> D, free interface of Y
    he[8] = [6, 3, 9, 0, 1, D, C, 1]  # D -> C, free interface of Y
    he[3] = [8, 1, 2, C, B, 0, 1, 1]  # C -> B, cell Y
    he[1] = [3, 6, 0, B, A, 0, 1, 1]  # B -> A, cell Y
    for k in (5, 7, 9):  # outer half-edges of the free interfaces belong to no cell
        he[k, 2] = k - 1
    angles = np.array([1.0, 1.0, ALPHA, BETA, BETA])  # interior edges: unused (zero chord)
    faces = np.array([[start_x, 0], [start_y, 0]], dtype=np.int32)
    return he, angles, faces


def _cross(p, q) -> float:
    return float(p[0] * q[1] - p[1] * q[0])


def _polygon_area(pts) -> float:
    """Signed area of a polygon, counterclockwise positive."""
    return 0.5 * sum(_cross(p, q) for p, q in zip(pts, np.roll(pts, -1, axis=0)))


def _cap_area(p0, p1, alpha) -> float:
    """Area of the circular cap of half-angle alpha over the chord p0 -> p1."""
    s, c = np.sin(alpha), np.cos(alpha)
    return float(np.linalg.norm(p1 - p0) ** 2 * (alpha - s * c) / (4 * s**2))


def _expected(pts) -> dict:
    a, b, c, d = pts
    tri_abc = _polygon_area(np.stack([a, b, c]))
    assert tri_abc < 0, "fixture: the chord polygon of X must have turned over"
    return {
        "area_x": _cap_area(c, a, ALPHA) + tri_abc,
        "area_y": _polygon_area(np.stack([a, d, c, b])) + _cap_area(a, d, BETA) + _cap_area(d, c, BETA),
        "polygon_x": tri_abc,
    }


def _true_energy(pts, angles) -> float:
    """Closed-form energy: sum of tension x length, plus (K/2) sum over cells of (area - A0)^2."""
    a, b, c, d = pts
    alpha, beta_ad, beta_dc = angles[2], angles[3], angles[4]

    def arc(p0, p1, al):
        return np.linalg.norm(p1 - p0) * al / np.sin(al)

    def cap(p0, p1, al):
        return np.linalg.norm(p1 - p0) ** 2 * (al - np.sin(al) * np.cos(al)) / (4 * np.sin(al) ** 2)

    def shoelace(*ps):
        return 0.5 * sum(_cross(p, q) for p, q in zip(ps, ps[1:] + ps[:1]))

    lengths = [np.linalg.norm(b - a), np.linalg.norm(c - b), arc(c, a, alpha), arc(a, d, beta_ad), arc(d, c, beta_dc)]
    area_x = shoelace(a, b, c) + cap(c, a, alpha)
    area_y = shoelace(a, d, c, b) + cap(a, d, beta_ad) + cap(d, c, beta_dc)
    area_term = (area_x - TARGET_AREA) ** 2 + (area_y - TARGET_AREA) ** 2
    return float(np.dot(TENSIONS, lengths) + 0.5 * K_AREAS * area_term)


CASES = [(mirror, start_x, start_y) for mirror in (False, True) for start_x, start_y in ((0, 6), (4, 3))]


@pytest.mark.parametrize(("mirror", "start_x", "start_y"), CASES)
def test_crescent_cell_area(mirror, start_x, start_y) -> None:
    pts = _points(mirror)
    he, angles, faces = _tables(start_x, start_y)
    padded = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), jnp.asarray(pts)])
    rep = jnp.repeat(jnp.asarray(angles), 2)
    got = [float(get_area_bounded(jnp.asarray(f), padded, rep, jnp.asarray(he), jnp.asarray(faces))) for f in (0, 1)]
    ref = _expected(pts)
    unsigned = ref["area_x"] - 2 * ref["polygon_x"]
    np.testing.assert_allclose(got[0], ref["area_x"], rtol=1e-12,
                               err_msg=f"crescent area; cap + |polygon| would be {unsigned}")
    np.testing.assert_allclose(got[1], ref["area_y"], rtol=1e-12)


@pytest.mark.parametrize(("mirror", "start_x", "start_y"), CASES)
def test_crescent_energy_gradient_against_finite_differences(mirror, start_x, start_y) -> None:
    pts = _points(mirror)
    he, angles, faces = _tables(start_x, start_y)
    energy = make_energy_line_tensions_bounded(K_AREAS)
    he_j, faces_j = jnp.asarray(he), jnp.asarray(faces)

    def lib(v, a):
        tensions, targets = jnp.asarray(TENSIONS), jnp.full(2, TARGET_AREA)
        return energy(v, a, he_j, faces_j, None, None, None, jnp.zeros(1), tensions, targets)

    assert float(lib(jnp.asarray(pts), jnp.asarray(angles))) == pytest.approx(_true_energy(pts, angles), rel=1e-12)
    g_v, g_a = jax.grad(lib, argnums=(0, 1))(jnp.asarray(pts), jnp.asarray(angles))
    rng = np.random.default_rng(7)
    h = 1e-6
    for _ in range(6):
        dv = rng.standard_normal(pts.shape)
        da = rng.standard_normal(angles.shape) * np.array([0, 0, 1, 1, 1])  # only free arcs carry an angle
        fd = (_true_energy(pts + h * dv, angles + h * da) - _true_energy(pts - h * dv, angles - h * da)) / (2 * h)
        ad = float(jnp.sum(g_v * dv) + jnp.sum(g_a * da))
        np.testing.assert_allclose(ad, fd, rtol=1e-7, atol=1e-9)


def _signed_chord_polygons(mesh: BoundedMesh) -> np.ndarray:
    he = np.asarray(mesh.edges)
    vt = np.vstack([np.zeros((2, 2)), np.asarray(mesh.vertices, dtype=np.float64)])
    src, tgt = he[:, 3] + he[:, 5], he[:, 4] + he[:, 6] - 1
    in_cell = src != 0
    cross = vt[src, 0] * vt[tgt, 1] - vt[tgt, 0] * vt[src, 1]
    return np.bincount(he[in_cell, 7], weights=cross[in_cell], minlength=mesh.nb_faces) / 2


def _orientation_defects(he: np.ndarray) -> int:
    """Half-edges that break a consistently oriented structure (0 for a valid mesh)."""
    src, tgt = he[:, 3] + he[:, 5], he[:, 4] + he[:, 6] - 1
    h = np.where(src != 0)[0]
    nxt, twin = he[h, 1], he[h, 2]
    interior = he[h, 3] != 0
    return int(np.sum(tgt[h] != src[nxt]) + np.sum(he[nxt, 7] != he[h, 7]) + np.sum(he[twin, 2] != h)
               + np.sum(src[h[interior]] != tgt[twin[interior]]))


@pytest.mark.parametrize("seed", [1, 2])
def test_faces_are_built_counterclockwise_and_t1_keeps_the_orientation(seed) -> None:
    for n_cells in (6, 12, 18, 36):
        mesh = BoundedMesh.from_random_seeds(n_cells, n_cells**0.5, n_cells**0.5, seed)
        assert (_signed_chord_polygons(mesh) > 0).all(), f"{n_cells} cells, seed {seed}: a face is built clockwise"

    mesh = BoundedMesh.from_random_seeds(12, 12**0.5, 12**0.5, seed)
    rng = np.random.default_rng(seed)
    mesh.vertices_params = jnp.asarray([0.0])
    mesh.edges_params = jnp.asarray(rng.random(mesh.nb_edges) * 20 - 10)  # wide tension spread: T1 transitions
    mesh.faces_params = jnp.asarray([TARGET_AREA])
    before = np.asarray(mesh.edges).copy()
    bop = BoundedBilevelOptimizer()
    bop.min_dist_T1 = 0.05
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.max_nb_iterations = 300
    bop.tolerance = 0.0
    bop.loss_function_inner = make_energy_line_tensions_bounded(K_AREAS, sigmoid_tensions=True)
    bop.inner_optimization(mesh=mesh)
    after = np.asarray(mesh.edges)
    assert (before != after).any(), "no T1 transition happened; the check below would not discriminate"
    assert _orientation_defects(after) == 0
