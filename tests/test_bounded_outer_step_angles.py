"""After a bounded outer step, `mesh.angles` belongs to the same equilibrium as `mesh.vertices`.

The automatic-differentiation, equilibrium-propagation and implicit-differentiation estimators of
`BoundedBilevelOptimizer` replace `mesh.vertices`, `mesh.edges` and `mesh.faces` by the equilibrium
of their inner solve (the free one, for EP). On 0.2.1 they did not replace `mesh.angles`, so the mesh
was left with the vertices of one state and the arc angles of another, and the relaxation that
`bilevel_optimization` runs next started from that mixed state. The adjoint-state estimator leaves
the whole state as it was, which is consistent and is pinned here too.

The reference equilibria are computed with the library's own relaxations, called directly with the
arguments the estimators use.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from vertax import BoundedBilevelOptimizer, BoundedMesh
from vertax.cost import cost_ratio
from vertax.energy import make_energy_line_tensions_bounded
from vertax.method_enum import BilevelOptimizationMethod
from vertax.opt_bounded import (
    _loss_ep_static_bounded,
    bilevel_opt_bounded,
    inner_eq_prop_bounded,
    inner_opt_bounded,
)

N_CELLS = 6  # small: bounded implicit differentiation runs one MINRES solve per parameter
SIDE = float(np.sqrt(N_CELLS))
BETA = 0.01
INNER_ITERATIONS = 30
ENERGY = make_energy_line_tensions_bounded(10.0)
# Built once: every relaxation is compiled for its solver object, so a new one per test recompiles.
INNER_SOLVER = optax.sgd(0.01)
OUTER_SOLVER = optax.sgd(1e-3)
METHODS = list(BilevelOptimizationMethod)
AD = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
EP = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
AS = BilevelOptimizationMethod.ADJOINT_STATE


def _mesh() -> BoundedMesh:
    mesh = BoundedMesh.from_random_seeds(nb_seeds=N_CELLS, width=SIDE, height=SIDE, random_key=1)
    rng = np.random.default_rng(1)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.asarray(1.0 + 0.1 * rng.standard_normal(mesh.nb_angles), dtype=mesh.vertices.dtype)
    mesh.faces_params = jnp.ones(mesh.nb_faces)
    return mesh


def _optimizer(method: BilevelOptimizationMethod) -> BoundedBilevelOptimizer:
    optimizer = BoundedBilevelOptimizer()
    optimizer.inner_solver = INNER_SOLVER
    optimizer.outer_solver = OUTER_SOLVER
    optimizer.bilevel_optimization_method = method
    optimizer.loss_function_inner = ENERGY
    optimizer.loss_function_outer = cost_ratio
    optimizer.min_dist_T1 = 0.025
    optimizer.max_nb_iterations = INNER_ITERATIONS
    optimizer.tolerance = 0.0
    optimizer.patience = INNER_ITERATIONS
    optimizer.beta = BETA
    return optimizer


def _relax(optimizer: BoundedBilevelOptimizer, state: tuple, params: tuple) -> tuple:
    """`inner_opt_bounded` as the estimators and `inner_optimization` call it."""
    (v, a, h, f), _ = inner_opt_bounded(
        *state, *params, ENERGY, optimizer.inner_solver, optimizer.min_dist_T1, INNER_ITERATIONS, 0.0,
        INNER_ITERATIONS, None, None, None, optimizer._update_T1_func, cap_vertex_step=False,
    )
    return v, a, h, f


def _estimator_state(method: BilevelOptimizationMethod, optimizer: BoundedBilevelOptimizer, mesh: BoundedMesh) -> tuple:
    """The (vertices, angles, edges, faces) the estimator should leave on the mesh."""
    state = (mesh.vertices, mesh.angles, mesh.edges, mesh.faces)
    params = (mesh.vertices_params, mesh.edges_params, mesh.faces_params)
    if method is AS:
        return state
    if method is EP:  # the free solve of the centered estimator, at -beta
        (v, a, h, f), _ = inner_eq_prop_bounded(
            *state, *params, _loss_ep_static_bounded, optimizer.vertices_target, optimizer.angles_target,
            optimizer.edges_target, optimizer.faces_target, ENERGY, cost_ratio, optimizer.inner_solver,
            optimizer.min_dist_T1, INNER_ITERATIONS, 0.0, INNER_ITERATIONS, None, None, None,
            optimizer.image_target, -BETA, update_T1_func=optimizer._update_T1_func, cap_vertex_step=False,
        )
        return v, a, h, f
    return _relax(optimizer, state, params)


def _assert_state(mesh: BoundedMesh, expected: tuple, what: str) -> None:
    v, a, h, f = expected
    np.testing.assert_array_equal(np.asarray(mesh.edges), np.asarray(h), err_msg=f"{what}: edges")
    np.testing.assert_array_equal(np.asarray(mesh.faces), np.asarray(f), err_msg=f"{what}: faces")
    # The relaxations are recomputed here outside the estimator (for AD, outside `value_and_grad`), so
    # they agree to float32 rounding, not to the bit. The arc angles move by ~1e-2 in this relaxation.
    np.testing.assert_allclose(np.asarray(mesh.vertices), np.asarray(v), rtol=0, atol=1e-5 * SIDE,
                               err_msg=f"{what}: vertices")
    np.testing.assert_allclose(np.asarray(mesh.angles), np.asarray(a), rtol=0, atol=1e-5,
                               err_msg=f"{what}: angles")


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_outer_step_leaves_angles_of_the_same_equilibrium(method: BilevelOptimizationMethod) -> None:
    optimizer, mesh = _optimizer(method), _mesh()
    expected = _estimator_state(method, optimizer, mesh)
    if method is not AS:
        moved = np.abs(np.asarray(expected[1]) - np.asarray(mesh.angles)).max()
        assert moved > 1e-3, "the inner solve must move the angles, or the test proves nothing"
    optimizer.outer_optimization(mesh)
    _assert_state(mesh, expected, method.name)


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_bilevel_optimization_relaxes_from_the_estimator_equilibrium(method: BilevelOptimizationMethod) -> None:
    """`bilevel_optimization` = outer step, then a relaxation from the state the outer step left."""
    optimizer, mesh = _optimizer(method), _mesh()
    after_outer = _estimator_state(method, optimizer, mesh)
    optimizer.bilevel_optimization(mesh)
    new_params = (mesh.vertices_params, mesh.edges_params, mesh.faces_params)  # the relaxation keeps them
    _assert_state(mesh, _relax(optimizer, after_outer, new_params), method.name)


@pytest.mark.parametrize("method", [AD, EP], ids=lambda m: m.name)
def test_functional_bilevel_opt_bounded_relaxes_from_the_estimator_equilibrium(
    method: BilevelOptimizationMethod,
) -> None:
    """Same contract for the functional wrapper `opt_bounded.bilevel_opt_bounded`."""
    optimizer, mesh = _optimizer(method), _mesh()
    after_outer = _estimator_state(method, optimizer, mesh)
    (v, a, h, f, vp, hp, fp), _ = bilevel_opt_bounded(
        mesh.vertices, mesh.angles, mesh.edges, mesh.faces,
        mesh.vertices_params, mesh.edges_params, mesh.faces_params,
        optimizer.vertices_target, optimizer.angles_target, optimizer.edges_target, optimizer.faces_target,
        ENERGY, cost_ratio, INNER_SOLVER, OUTER_SOLVER, optimizer.min_dist_T1, INNER_ITERATIONS, 0.0,
        INNER_ITERATIONS, None, None, None, optimizer.image_target, BETA, method, optimizer._update_T1_func,
    )
    result = BoundedMesh.copy_mesh(mesh)
    result.vertices, result.angles, result.edges, result.faces = v, a, h, f
    _assert_state(result, _relax(optimizer, after_outer, (vp, hp, fp)), method.name)
