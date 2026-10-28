"""`cost.cost_areas` follows the periodic outer-cost protocol and can be used as an outer loss.

On 0.2.1 it raised `TypeError` at its first call: it called `geo.get_area` without the face-walk
bound `max_iter`. Its arguments were also not in the 12-argument order in which the optimizers call
an outer cost (selections before targets), and it could not take `None` selections, which is what
the optimizers pass when no subset is selected.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_areas

N_CELLS = 20
SIDE = float(np.sqrt(N_CELLS))


def _meshes() -> tuple[PbcMesh, PbcMesh]:
    """A target mesh and the same mesh with its vertices moved (same topology, other areas)."""
    target = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=SIDE, height=SIDE, random_key=0)
    current = PbcMesh.copy_mesh(target)
    shift = np.random.default_rng(0).normal(scale=0.05, size=np.shape(target.vertices))
    current.vertices = target.vertices + jnp.asarray(shift, dtype=target.vertices.dtype)
    return current, target


def _expected(current: PbcMesh, target: PbcMesh, faces: np.ndarray) -> float:
    """Mean squared area difference over `faces`, from `PbcMesh.get_area`."""
    faces = jnp.asarray(faces)
    return float(jnp.mean((current.get_area(faces) - target.get_area(faces)) ** 2))


def test_cost_areas_in_the_outer_cost_protocol() -> None:
    """Positional call in the protocol order: mesh, box, targets, selections, image."""
    current, target = _meshes()
    faces = np.arange(current.nb_faces)
    value = cost_areas(
        current.vertices, current.edges, current.faces, current.width, current.height,
        target.vertices, target.edges, target.faces,
        None, None, jnp.asarray(faces), jnp.array([]),
    )
    expected = _expected(current, target, faces)
    assert expected > 0, "the moved mesh must have other areas, or the test proves nothing"
    np.testing.assert_allclose(float(value), expected, rtol=1e-6)


def test_cost_areas_as_the_outer_loss_of_a_periodic_optimizer() -> None:
    """Through `compute_outer_loss`, with no selection (`None`) and with a subset of faces."""
    current, target = _meshes()
    optimizer = PbcBilevelOptimizer()
    optimizer.loss_function_outer = cost_areas
    optimizer.vertices_target = target.vertices
    optimizer.edges_target = target.edges
    optimizer.faces_target = target.faces

    every_face = np.arange(current.nb_faces)
    np.testing.assert_allclose(optimizer.compute_outer_loss(current), _expected(current, target, every_face),
                               rtol=1e-6)
    some_faces = [0, 3, 7]
    np.testing.assert_allclose(optimizer.compute_outer_loss(current, only_on_faces=some_faces),
                               _expected(current, target, np.asarray(some_faces)), rtol=1e-6)
    assert optimizer.compute_outer_loss(target) == 0.0


def test_cost_areas_has_a_finite_gradient_in_the_vertices() -> None:
    """It is an outer cost, so the hypergradient estimators differentiate it."""
    current, target = _meshes()
    g = jax.grad(cost_areas)(
        current.vertices, current.edges, current.faces, current.width, current.height,
        target.vertices, target.edges, target.faces, None, None, None, jnp.array([]),
    )
    assert g.shape == current.vertices.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0
