"""The area stiffness of the shape-factor energy E1, as an explicit and trainable parameter.

`make_energy_shape_factor(lambda_area)` builds

    E1 = sum_alpha lambda_area (A_alpha - 1)^2 + (P_alpha - p0_alpha)^2 ,

and must reduce to `energy_shape_factor_hetero` at `lambda_area = 1`. For the stiffness to be
inferred rather than assumed it must also be differentiable, including when it is carried as a
positive outer parameter `lambda_area = exp(kappa)` and reached only through the inner solve.

`float64` is enabled per-module and scoped.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import energy_shape_factor_hetero, make_energy_shape_factor

N_CELLS = 20
L = math.sqrt(N_CELLS)
# A preferred perimeter below the rigidity transition (~3.81), so the ground state is not a
# zero-energy manifold and the area term is not trivially zero at equilibrium.
P0_MEAN = 3.6


@pytest.fixture(autouse=True)
def _x64():  # noqa: PT004
    with enable_x64():
        yield


@pytest.fixture()
def random_mesh():
    """A random periodic mesh, with vertices jittered so no cell sits at area 1 by construction."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=L, height=L, random_key=321)
    rng = np.random.default_rng(7)
    v = jnp.asarray(mesh.vertices, jnp.float64)
    v = v.at[:, :2].add(0.03 * jnp.asarray(rng.standard_normal((v.shape[0], 2))))
    p0 = jnp.asarray(P0_MEAN + 0.2 * rng.standard_normal(mesh.nb_faces))
    return v, jnp.asarray(mesh.edges), jnp.asarray(mesh.faces), p0


def _energy(lambda_area, v, he, fa, p0):
    return make_energy_shape_factor(lambda_area)(v, he, fa, L, L, jnp.arange(fa.shape[0]), p0)


def test_factory_at_unit_stiffness_equals_the_existing_energy(random_mesh) -> None:
    """At `lambda_area = 1` the factory is `energy_shape_factor_hetero`, value and gradient."""
    v, he, fa, p0 = random_mesh
    faces = jnp.arange(fa.shape[0])
    old = energy_shape_factor_hetero(v, he, fa, L, L, faces, p0)
    new = _energy(1.0, v, he, fa, p0)
    assert float(new) == pytest.approx(float(old), rel=1e-14, abs=0.0)

    g_old = jax.grad(lambda x: energy_shape_factor_hetero(x, he, fa, L, L, faces, p0))(v)
    g_new = jax.grad(lambda x: _energy(1.0, x, he, fa, p0))(v)
    np.testing.assert_allclose(np.asarray(g_new), np.asarray(g_old), rtol=1e-13, atol=1e-15)

    # the default is the unit stiffness
    default = make_energy_shape_factor()(v, he, fa, L, L, faces, p0)
    assert float(default) == float(new)


def test_energy_is_linear_in_the_area_stiffness(random_mesh) -> None:
    """`E1(lambda) = E_perimeter + lambda * E_area`: affine in `lambda`, with slope the area term.

    The slope is checked against the area term computed independently, so a factor applied to the
    wrong term -- or to both -- would fail here and not only in the finite-difference test.
    """
    from vertax.energy import MAX_EDGES_IN_ANY_FACE
    from vertax.geo import get_area

    v, he, fa, p0 = random_mesh
    e0 = float(_energy(0.0, v, he, fa, p0))
    e1 = float(_energy(1.0, v, he, fa, p0))
    areas = jax.vmap(lambda f: get_area(f, v, he, fa, L, L, MAX_EDGES_IN_ANY_FACE))(
        jnp.arange(fa.shape[0]))
    area_term = float(jnp.sum((areas - 1.0) ** 2))
    assert area_term > 1e-3, "fixture must have cells away from unit area"
    assert e1 - e0 == pytest.approx(area_term, rel=1e-12)

    for lam in (0.1, 0.37, 2.5, 10.0):
        assert float(_energy(lam, v, he, fa, p0)) == pytest.approx(e0 + lam * area_term, rel=1e-12)


def test_gradient_in_the_area_stiffness_matches_central_difference(random_mesh) -> None:
    """`dE1/dlambda` and `dE1/dkappa` with `lambda = exp(kappa)` agree with central differences.

    The energy is linear in `lambda`, so its difference quotient is exact up to rounding; the
    log-parameterised form is not, and is the one an inference would use to keep the stiffness
    positive.
    """
    v, he, fa, p0 = random_mesh
    lam0, h = 0.3, 1e-4

    g = float(jax.grad(lambda lam: _energy(lam, v, he, fa, p0))(lam0))
    fd = float(_energy(lam0 + h, v, he, fa, p0) - _energy(lam0 - h, v, he, fa, p0)) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-6)

    kappa0 = math.log(lam0)
    e_of_kappa = lambda k: _energy(jnp.exp(k), v, he, fa, p0)  # noqa: E731
    g = float(jax.grad(e_of_kappa)(kappa0))
    fd = float(e_of_kappa(kappa0 + h) - e_of_kappa(kappa0 - h)) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-6)


def test_log_stiffness_hypergradient_through_an_unrolled_inner_solve(random_mesh) -> None:
    """As an outer parameter, `kappa = log(lambda_area)` reaches the outer cost only through the
    equilibrium, and its hypergradient must still match a central difference.

    The inner solve is unrolled gradient descent, which is what automatic differentiation of the
    bilevel problem differentiates.
    """
    v0, he, fa, p0 = random_mesh
    faces = jnp.arange(fa.shape[0])
    lr, n_steps = 0.01, 30

    def relax(kappa):
        energy = make_energy_shape_factor(jnp.exp(kappa))
        grad_x = jax.grad(lambda x: energy(x, he, fa, L, L, faces, p0))

        def step(x, _):
            return x - lr * grad_x(x), None

        x, _ = jax.lax.scan(step, v0, None, length=n_steps)
        return x

    reference = relax(jnp.asarray(0.0))              # geometry relaxed at lambda_area = 1

    def outer(kappa):
        return jnp.sum((relax(kappa) - reference) ** 2)

    kappa0, h = math.log(0.3), 1e-4
    g = float(jax.grad(outer)(kappa0))
    fd = float(outer(kappa0 + h) - outer(kappa0 - h)) / (2 * h)
    assert abs(g) > 1e-10, "the hypergradient must not vanish identically"
    assert g == pytest.approx(fd, rel=1e-6)


def test_log_stiffness_is_trained_by_the_bilevel_optimizer() -> None:
    """End to end: a `kappa` stored in `vertices_params[0]` is updated by the bilevel optimizer.

    This is the same slot and parameterisation used elsewhere for the E2 stiffness as a joint-K
    parameter, so it confirms the factory composes with the optimizer's parameter plumbing.
    """
    mesh = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=L, height=L, random_key=322)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices).at[0].set(math.log(0.3))
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    mesh.faces_params = jnp.full(mesh.nb_faces, P0_MEAN)
    target = PbcMesh.copy_mesh(mesh)

    def energy(v, h, f, vp, _hp, fp):
        return make_energy_shape_factor(jnp.exp(vp[0]))(v, h, f, L, L, jnp.arange(len(f)), fp)

    # a target that is NOT the starting geometry, relaxed at lambda_area = 1
    relax = jax.grad(lambda x: make_energy_shape_factor(1.0)(
        x, target.edges, target.faces, L, L, jnp.arange(target.nb_faces), target.faces_params))
    x = jnp.asarray(target.vertices, jnp.float64)
    for _ in range(50):
        x = x - 0.01 * relax(x)

    bop = PbcBilevelOptimizer()
    bop.bilevel_optimization_method = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
    bop.loss_function_inner = energy
    bop.loss_function_outer = cost_v2v
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.sgd(learning_rate=1e-2)
    bop.max_nb_iterations = 5
    bop.update_T1 = False
    bop.vertices_target, bop.edges_target, bop.faces_target = x, target.edges, target.faces

    before = float(mesh.vertices_params[0])
    bop.bilevel_optimization(mesh, perform_energy_check=False)
    after = float(mesh.vertices_params[0])

    assert math.isfinite(after)
    assert after != before, "kappa received no update: the gradient did not reach it"
    # every other vertex-parameter slot is unused by the energy and must receive no update
    assert float(jnp.max(jnp.abs(jnp.asarray(mesh.vertices_params)[1:]))) == 0.0
