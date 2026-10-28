"""`BoundedBilevelOptimizer` honours `params_regularizer` and `ep_variant`, and refuses `persist_outer_state`.

On 0.2.1 the bounded optimizer passed neither `params_regularizer` nor `ep_variant` to its
estimators, so a bounded run ignored both without a word: the regularizer was never added to the
hypergradient (its docstring says it is honoured by all estimator paths), and the bounded EP
estimate was always centered. `persist_outer_state=True` was ignored the same way.

The outer solver is `optax.sgd`, so one outer step moves the parameters by exactly
`-learning_rate * gradient` and the gradient can be read back from the step.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from vertax import BoundedBilevelOptimizer, BoundedMesh
from vertax.cost import cost_ratio
from vertax.energy import make_energy_line_tensions_bounded
from vertax.method_enum import BilevelOptimizationMethod, EPVariant
from vertax.opt_bounded import inner_eq_prop_bounded

N_CELLS = 6  # small: bounded implicit differentiation runs one MINRES solve per parameter
SIDE = float(np.sqrt(N_CELLS))
LEARNING_RATE = 1e-3
BETA = 0.01
INNER_ITERATIONS = 30
ENERGY = make_energy_line_tensions_bounded(10.0)
# Built once: every relaxation is compiled for its solver object, so a new one per test recompiles.
INNER_SOLVER = optax.sgd(0.01)
OUTER_SOLVER = optax.sgd(LEARNING_RATE)
METHODS = list(BilevelOptimizationMethod)
PARAMS = ("vertices_params", "edges_params", "faces_params")


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


def _outer_gradient(optimizer: BoundedBilevelOptimizer) -> dict[str, np.ndarray]:
    """The gradient applied by one outer step, read back from the sgd step."""
    mesh = _mesh()
    before = {name: np.asarray(getattr(mesh, name), dtype=np.float64) for name in PARAMS}
    optimizer.outer_optimization(mesh)
    return {name: (before[name] - np.asarray(getattr(mesh, name), dtype=np.float64)) / LEARNING_RATE
            for name in PARAMS}


def _close(a: np.ndarray, b: np.ndarray, rtol: float) -> bool:
    """Equal within `rtol` of the larger of the two arrays' scales (float32 parameters)."""
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    return bool(np.abs(a - b).max() <= rtol * scale)


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_params_regularizer_is_added_to_the_bounded_hypergradient(method: BilevelOptimizationMethod) -> None:
    """With `R = 0.5 * c * sum(edges_params**2)`, the applied gradient grows by `c * edges_params`."""
    c = 3.0

    def regularizer(_vp, hp, _fp):  # noqa: ANN001, ANN202
        return 0.5 * c * jnp.sum(hp**2)

    plain = _outer_gradient(_optimizer(method))
    optimizer = _optimizer(method)
    optimizer.params_regularizer = regularizer
    regularized = _outer_gradient(optimizer)

    expected = c * np.asarray(_mesh().edges_params, dtype=np.float64)
    # The regularizer term is O(1) and the hypergradient is computed identically in both runs, so
    # the difference is the regularizer gradient up to the float32 rounding of the parameter step.
    np.testing.assert_allclose(regularized["edges_params"] - plain["edges_params"], expected,
                               rtol=0, atol=1e-3 * np.abs(expected).max())
    for name in ("vertices_params", "faces_params"):
        np.testing.assert_array_equal(regularized[name], plain[name], err_msg=name)


def test_one_sided_ep_variant_is_honoured_by_the_bounded_optimizer() -> None:
    """ONE_SIDED: free solve at beta = 0, nudged at +beta, divided by beta (not the centered form)."""
    centered = _outer_gradient(_optimizer(BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION))
    optimizer = _optimizer(BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION)
    optimizer.ep_variant = EPVariant.ONE_SIDED
    one_sided = _outer_gradient(optimizer)
    assert not _close(one_sided["edges_params"], centered["edges_params"], 1e-3), (
        "ONE_SIDED gave the centered estimate: ep_variant is ignored")

    # The one-sided estimate from its definition, with the library's own EP relaxation.
    mesh = _mesh()

    def equilibrium(beta: float) -> tuple:
        (v, a, h, f), _ = inner_eq_prop_bounded(
            mesh.vertices, mesh.angles, mesh.edges, mesh.faces,
            mesh.vertices_params, mesh.edges_params, mesh.faces_params,
            _loss_ep, optimizer.vertices_target, optimizer.angles_target, optimizer.edges_target,
            optimizer.faces_target, ENERGY, cost_ratio, optimizer.inner_solver, optimizer.min_dist_T1,
            INNER_ITERATIONS, 0.0, INNER_ITERATIONS, None, None, None, optimizer.image_target, beta,
            cap_vertex_step=False,
        )
        return v, a, h, f

    def params_gradient(state: tuple) -> tuple:
        v, a, h, f = state
        # The nudge beta * L_out does not depend on the parameters, so only the energy contributes.
        return jax.grad(ENERGY, argnums=(7, 8, 9))(
            v, a, h, f, None, None, None, mesh.vertices_params, mesh.edges_params, mesh.faces_params)

    free, nudged = params_gradient(equilibrium(0.0)), params_gradient(equilibrium(BETA))
    for k, name in enumerate(PARAMS):
        expected = (np.asarray(nudged[k], np.float64) - np.asarray(free[k], np.float64)) / BETA
        assert _close(one_sided[name], expected, 1e-3), name


def _loss_ep(v, a, h, f, vp, hp, fp, vt, at, ht, ft, L_in, L_out, sv, sh, sf, image, beta):  # noqa: ANN001, ANN202, PLR0913
    """The EP loss `L_in + beta * L_out`, in the signature `inner_eq_prop_bounded` expects."""
    return L_in(v, a, h, f, sv, sh, sf, vp, hp, fp) + beta * L_out(v, a, h, f, vt, at, ht, ft, sv, sh, sf, image)


def test_persist_outer_state_is_refused_by_the_bounded_optimizer() -> None:
    """The bounded EP path has no persistent outer state: say so rather than run without it."""
    optimizer = _optimizer(BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION)
    optimizer.persist_outer_state = True
    with pytest.raises(ValueError, match="persist_outer_state"):
        optimizer.outer_optimization(_mesh())
