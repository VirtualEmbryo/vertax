"""`minimize` gives line-search solvers the energy along the array they optimize, for every target.

A line-search solver (`gd_linesearch`, `cg_linesearch`, `optax.lbfgs()`) re-evaluates the energy
through the `value_fn` that `minimize` passes to `solver.update`, together with the current `params`.
On 0.2.1 both were chosen by branches numbered 0, 2, 3, 4, 5, 6 while `OptimizationTarget` is
numbered 0 to 5, so for every target other than `VERTICES` the solver received another array as
`params` and a `value_fn` that substituted its trial point into another argument of the energy.
The gradient itself was right.

The energy here is a weighted quadratic in the optimized array alone, so its minimum is known.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from vertax import PbcMesh
from vertax.cg import gd_linesearch
from vertax.opt import OptimizationTarget, minimize
from vertax.topo import do_not_update_T1

N_CELLS = 20
SIDE = float(np.sqrt(N_CELLS))
FLOAT_TARGETS = [
    OptimizationTarget.VERTICES,
    OptimizationTarget.VERTEX_PARAMETERS,
    OptimizationTarget.EDGE_PARAMETERS,
    OptimizationTarget.FACE_PARAMETERS,
]


def _problem(target: OptimizationTarget) -> tuple[tuple, object, np.ndarray]:
    """Mesh tables, an energy minimal at `optimum` in the target array, and that optimum."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=SIDE, height=SIDE, random_key=0)
    rng = np.random.default_rng(0)
    arrays = [mesh.vertices, mesh.edges, mesh.faces,
              jnp.ones(mesh.nb_vertices), jnp.ones(mesh.nb_half_edges), jnp.ones(mesh.nb_faces)]
    x0 = arrays[target.value]
    weights = jnp.asarray(rng.uniform(0.5, 2.0, size=x0.shape), dtype=x0.dtype)
    shift = 0.1 if target is OptimizationTarget.VERTICES else 1.0  # vertices stay inside the box
    optimum = x0 + jnp.asarray(shift * rng.uniform(-1.0, 1.0, size=x0.shape), dtype=x0.dtype)

    def energy(*tables):  # noqa: ANN002, ANN202
        return 0.5 * jnp.sum(weights * (tables[target.value] - optimum) ** 2)

    return tuple(arrays), energy, np.asarray(optimum)


def _minimize(target: OptimizationTarget, solver: optax.GradientTransformation, iterations: int) -> np.ndarray:
    (v, h, f, vp, hp, fp), energy, _ = _problem(target)
    tables, _ = minimize(
        v, h, f, SIDE, SIDE, vp, hp, fp, energy, solver, 1e-6, iterations_max=iterations, tolerance=0.0,
        patience=iterations, optimization_target=target, update_t1_func=do_not_update_T1,
    )
    return np.asarray(tables[target.value])


@pytest.mark.parametrize("target", FLOAT_TARGETS, ids=lambda t: t.name)
def test_value_fn_evaluates_the_energy_at_the_solver_params(target: OptimizationTarget) -> None:
    """A spy solver checks, at each step, that `params` and `value_fn` refer to the optimized array."""
    (v, h, f, vp, hp, fp), energy, _ = _problem(target)
    arrays = [v, h, f, vp, hp, fp]
    seen: list[tuple[float, float, float]] = []

    def record(shape_ok, at_params, at_probe) -> None:  # noqa: ANN001
        seen.append((float(shape_ok), float(at_params), float(at_probe)))

    def update(updates, state, params=None, *, value, grad, value_fn, **_extra):  # noqa: ANN001, ANN202, ARG001
        probe = params + 0.01
        swapped = [*arrays]
        swapped[target.value] = probe
        jax.debug.callback(
            record,
            jnp.asarray(params.shape == updates.shape, dtype=jnp.float32),
            value_fn(params) - value,  # the solver's own point: must give the current energy
            value_fn(probe) - energy(*swapped),  # a trial point: must move the optimized array only
        )
        return jax.tree.map(jnp.zeros_like, updates), state

    spy = optax.GradientTransformationExtraArgs(lambda _params: optax.EmptyState(), update)
    minimize(v, h, f, SIDE, SIDE, vp, hp, fp, energy, spy, 1e-6, iterations_max=3, tolerance=0.0, patience=3,
             optimization_target=target, update_t1_func=do_not_update_T1)

    assert len(seen) == 3
    scale = float(energy(*arrays))
    for shape_ok, at_params, at_probe in seen:
        assert shape_ok == 1.0, "params handed to the solver is not the optimized array"
        assert abs(at_params) <= 1e-6 * scale, "value_fn(params) is not the current energy"
        assert abs(at_probe) <= 1e-6 * scale, "value_fn does not evaluate the energy along the optimized array"


@pytest.mark.parametrize("target", FLOAT_TARGETS[1:], ids=lambda t: t.name)
def test_gd_linesearch_reaches_the_minimum_of_a_quadratic(target: OptimizationTarget) -> None:
    """Steepest descent with a Wolfe line search finds the known minimum in the parameters.

    The vertices are left out: `minimize` wraps them into the periodic box, which moves the minimum
    of this quadratic for vertices near the box edge, and they are covered by the other tests.
    """
    _, _, optimum = _problem(target)
    x = _minimize(target, gd_linesearch(), iterations=200)
    # float32: the energy stops resolving the remaining step near 1e-4 of the parameter scale.
    np.testing.assert_allclose(x, optimum, rtol=0, atol=1e-3 * np.abs(optimum).max())
