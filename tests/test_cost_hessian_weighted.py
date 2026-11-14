"""The Hessian-weighted cost C_W and the staged schedule against the definitions Figure 3 was run with.

`tests/fixtures/hessian_weighted_cost_reference.npz` holds the normalized Hessian, C_W and its gradient
computed by the functions of the Figure 3 inference script on two mesh pairs of the synthetic E2
benchmark (`make_hessian_weighted_cost_reference.py` beside it): instance 1008, whose initial mesh has
the target topology, and instance 1000, one T1 transition away from it. The library must give the
same numbers to the rounding of float64: the Hessian to a few units in the last place of its unit
spectral norm, the cost and its gradient to a relative 1e-13.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import PbcBilevelOptimizer, PbcMesh, cost_v2v, make_energy_line_tensions, topological_distance
from vertax.cost import hessian_at_target, make_cost_hessian_weighted, switch_cost_at_target_topology

FIXTURES = Path(__file__).parent / "fixtures"
K_AREAS = 10.0
EPS = np.finfo(np.float64).eps


@pytest.fixture(autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    with enable_x64():
        yield


@pytest.fixture(scope="module")
def reference() -> dict:
    with np.load(FIXTURES / "hessian_weighted_cost_reference.npz") as data:
        return dict(data)


def _pair(instance: int) -> tuple[PbcMesh, PbcMesh]:
    with enable_x64():
        return (PbcMesh.load_mesh(str(FIXTURES / f"mesh_{instance}.npz")),
                PbcMesh.load_mesh(str(FIXTURES / f"mesh_target_{instance}.npz")))


def _energy(mesh: PbcMesh):  # noqa: ANN202
    assert K_AREAS == 10.0
    e2 = make_energy_line_tensions(K_AREAS)
    return lambda v, h, f, _vp, hp, fp: e2(v, h, f, mesh.width, mesh.height, hp, fp)


CASES = [(1008, "uniform"), (1008, "target"), (1000, "uniform")]


@pytest.mark.parametrize(("instance", "tensions"), CASES)
def test_hessian_matches_figure3(reference, instance, tensions) -> None:
    _, target = _pair(instance)
    edges_params = None if tensions == "uniform" else jnp.asarray(target.edges_params, dtype=jnp.float64)
    h = np.asarray(hessian_at_target(target, _energy(target), edges_params))
    ref = reference[f"H_{tensions}_{instance}"]
    assert h.dtype == np.float64
    assert np.max(np.abs(h - ref)) <= 4 * EPS
    assert np.max(np.abs(np.linalg.eigvalsh(h))) == pytest.approx(1.0, abs=8 * EPS)


@pytest.mark.parametrize(("instance", "tensions"), CASES)
def test_cost_and_gradient_match_figure3(reference, instance, tensions) -> None:
    mesh, target = _pair(instance)
    edges_params = None if tensions == "uniform" else jnp.asarray(target.edges_params, dtype=jnp.float64)
    cost = make_cost_hessian_weighted(target, _energy(target), edges_params)

    def at(v):  # noqa: ANN001, ANN202
        return cost(v, mesh.edges, mesh.faces, mesh.width, mesh.height, target.vertices, target.edges, target.faces,
                    None, None, None, None)

    value, grad = jax.value_and_grad(at)(jnp.asarray(mesh.vertices, dtype=jnp.float64))
    suffix = f"_{instance}" if tensions == "uniform" else f"_target_tensions_{instance}"
    np.testing.assert_allclose(float(value), float(reference["cost" + suffix]), rtol=1e-13, atol=0)
    ref_grad = reference["grad" + suffix]
    assert np.max(np.abs(np.asarray(grad) - ref_grad)) <= 1e-13 * np.max(np.abs(ref_grad))
    assert float(at(jnp.asarray(target.vertices, dtype=jnp.float64))) == 0.0


def test_cost_is_invariant_to_whole_periods() -> None:
    mesh, target = _pair(1008)
    cost = make_cost_hessian_weighted(target, _energy(target))
    v = jnp.asarray(mesh.vertices, dtype=jnp.float64)
    shifted = v.at[3].add(jnp.array([mesh.width, -2 * mesh.height]))
    args = (mesh.edges, mesh.faces, mesh.width, mesh.height, target.vertices, target.edges, target.faces)
    assert float(cost(shifted, *args)) == pytest.approx(float(cost(v, *args)), rel=1e-12)


def test_vertex_selection_keeps_the_force_residual_of_the_selected_vertices() -> None:
    mesh, target = _pair(1000)
    cost = make_cost_hessian_weighted(target, _energy(target))
    args = (mesh.vertices, mesh.edges, mesh.faces, mesh.width, mesh.height, target.vertices, target.edges, target.faces)
    n = mesh.vertices.shape[0]
    every = float(cost(*args, jnp.arange(n), jnp.arange(2 * n), jnp.arange(20), None))
    assert every == float(cost(*args))  # the optimizer passes every index when nothing is selected
    first, rest = jnp.arange(n // 2), jnp.arange(n // 2, n)
    assert float(cost(*args, first, None, None, None)) + float(cost(*args, rest, None, None, None)) == pytest.approx(
        every, rel=1e-12)


def _optimizer(target: PbcMesh) -> PbcBilevelOptimizer:
    bop = PbcBilevelOptimizer()
    bop.min_dist_T1 = 0.05
    bop.max_nb_iterations = 6
    bop.tolerance = 0.0
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.adam(learning_rate=1e-3)
    bop.beta = 0.01
    bop.loss_function_inner = _energy(target)
    bop.loss_function_outer = cost_v2v
    bop.vertices_target = target.vertices.copy()
    bop.edges_target = target.edges.copy()
    bop.faces_target = target.faces.copy()
    return bop


def test_staged_switch_waits_for_the_target_topology() -> None:
    mesh, target = _pair(1000)
    assert topological_distance(mesh.vertices, mesh.edges, mesh.faces, mesh.width, mesh.height,
                                target.vertices, target.edges, target.faces) == 2
    bop = _optimizer(target)
    cost_w = make_cost_hessian_weighted(target, bop.loss_function_inner)
    assert switch_cost_at_target_topology(bop, mesh, cost_w) is False
    assert bop.loss_function_outer is cost_v2v
    assert bop.update_T1

    # The target itself has the target topology: the switch is made, once.
    assert switch_cost_at_target_topology(bop, target, cost_w) is True
    assert bop.loss_function_outer is cost_w
    assert not bop.update_T1
    assert switch_cost_at_target_topology(bop, mesh, cost_w) is True  # stays switched


def test_staged_switch_at_the_target_topology_and_one_epoch() -> None:
    mesh, target = _pair(1008)
    bop = _optimizer(target)
    cost_w = make_cost_hessian_weighted(target, bop.loss_function_inner)
    assert switch_cost_at_target_topology(bop, mesh, cost_w) is True
    assert bop.loss_function_outer is cost_w
    assert not bop.update_T1
    mesh.faces_params = target.faces_params.copy()
    bop.bilevel_optimization(mesh)
    assert np.isfinite(np.asarray(mesh.edges_params)).all()
