"""T1 counts carried in the loop state of ``vertax.opt.minimize`` (``count_t1=True``).

The counts are part of the scan carry, so they are exact under ``jit``, ``vmap`` and ``jax.grad`` and
counting changes nothing the loop computes. Checked here:

* tables, loss history and gradient are bit-identical with and without counting;
* under the forced rule every presented edge is flipped; under the energy-checked rule kept <= presented;
* differentiating through a relaxation that undergoes T1 transitions returns the same counts as the
  forward relaxation, through ``grad(..., has_aux=True)``;
* under ``vmap`` the counts of each batch member equal those of its own relaxation;
* ``recording_t1_statistics`` collects the counts of the relaxation that the automatic-differentiation
  hypergradient unrolls and of the two solves of the equilibrium-propagation hypergradient, without
  changing either result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.energy import make_energy_line_tensions
from vertax.opt import OptimizationTarget, minimize, outer_eq_prop, outer_opt, recording_t1_statistics
from vertax.cost import cost_v2v
from vertax.topo import update_T1

WIDTH = HEIGHT = 4.0
MIN_DIST = 0.05
E2 = make_energy_line_tensions(10.0)


def _energy(v, h, f, _vp, hp, fp):  # noqa: ANN001, ANN202
    return E2(v, h, f, WIDTH, HEIGHT, hp, fp)


@pytest.fixture(scope="module")
def tissue():  # noqa: ANN201
    mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
    return mesh


def _relax(mesh, hp, check: bool, count: bool, n: int = 400):  # noqa: ANN001, ANN202
    v = jnp.asarray(mesh.vertices, jnp.float64)
    return minimize(v, jnp.asarray(mesh.edges), jnp.asarray(mesh.faces), WIDTH, HEIGHT, jnp.zeros(v.shape[0]), hp,
                    jnp.ones(mesh.nb_faces), _energy, optax.sgd(0.01), MIN_DIST, iterations_max=n, tolerance=0.0,
                    optimization_target=OptimizationTarget.VERTICES, update_t1_func=update_T1,
                    perform_energy_check=check, count_t1=count)


def _tensions(mesh, scale: float = 0.6):  # noqa: ANN001, ANN202
    rng = np.random.default_rng(0)
    return jnp.asarray(1.0 + scale * rng.random(mesh.nb_half_edges), jnp.float64)


@pytest.mark.parametrize("check", [False, True])
def test_counting_changes_nothing_and_counts_are_consistent(tissue, check: bool) -> None:  # noqa: ANN001
    with enable_x64():
        hp = _tensions(tissue)
        (t0, (h0, s0)) = _relax(tissue, hp, check, False)
        (t1, (h1, s1, counts)) = _relax(tissue, hp, check, True)
    for a, b in zip(t0, t1):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(h0), np.asarray(h1))
    c = np.asarray(counts)
    assert c[:, 1].sum() > 0, "the test tissue no longer undergoes T1 transitions"
    assert np.all(c[:, 1] <= c[:, 0])
    if not check:
        np.testing.assert_array_equal(c[:, 1], c[:, 0])


def test_counts_survive_reverse_mode_differentiation(tissue) -> None:  # noqa: ANN001
    with enable_x64():
        hp = _tensions(tissue)
        target = jnp.asarray(tissue.vertices, jnp.float64)

        def loss(lam, count):  # noqa: ANN001, ANN202
            v = jnp.asarray(tissue.vertices, jnp.float64)
            out = minimize(v, jnp.asarray(tissue.edges), jnp.asarray(tissue.faces), WIDTH, HEIGHT,
                           jnp.zeros(v.shape[0]), lam, jnp.ones(tissue.nb_faces), _energy, optax.sgd(0.01),
                           MIN_DIST, iterations_max=200, tolerance=0.0,
                           optimization_target=OptimizationTarget.VERTICES, update_t1_func=update_T1,
                           count_t1=count)
            val = jnp.sum((out[0][0] - target) ** 2)
            return (val, out[1][2]) if count else val

        g_plain = jax.grad(lambda lam: loss(lam, False))(hp)
        g_counted, counts = jax.grad(lambda lam: loss(lam, True), has_aux=True)(hp)
        forward = np.asarray(_relax(tissue, hp, False, True, n=200)[1][2])
    np.testing.assert_array_equal(np.asarray(g_plain), np.asarray(g_counted))
    np.testing.assert_array_equal(np.asarray(counts), forward)
    assert forward[:, 1].sum() > 0


def test_counts_under_vmap(tissue) -> None:  # noqa: ANN001
    with enable_x64():
        lams = jnp.stack([_tensions(tissue, 0.6), _tensions(tissue, 0.3)])
        batched = jax.vmap(lambda lam: _relax(tissue, lam, False, True, n=150)[1][2])(lams)
        single = [np.asarray(_relax(tissue, lam, False, True, n=150)[1][2]) for lam in lams]
    for b, s in zip(np.asarray(batched), single):
        np.testing.assert_array_equal(b, s)


def test_recorder_collects_the_automatic_differentiation_relaxation(tissue) -> None:  # noqa: ANN001
    with enable_x64():
        hp = _tensions(tissue)
        v = jnp.asarray(tissue.vertices, jnp.float64)
        args = (v, jnp.asarray(tissue.edges), jnp.asarray(tissue.faces), WIDTH, HEIGHT, jnp.zeros(v.shape[0]), hp,
                jnp.ones(tissue.nb_faces), v, jnp.asarray(tissue.edges), jnp.asarray(tissue.faces), _energy,
                cost_v2v, optax.sgd(0.01), optax.sgd(1e-3), MIN_DIST, 200, 0.0, 5, None, None, None, None, update_T1)
        plain = outer_opt(*args)
        with recording_t1_statistics() as rec:
            recorded = outer_opt(*args)
        forward = np.asarray(_relax(tissue, hp, False, True, n=200)[1][2])
    for a, b in zip(plain, recorded):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    ad = [r for r in rec if r["source"] == "automatic differentiation"]
    assert len(ad) == 1
    np.testing.assert_array_equal(ad[0]["counts"], forward[: len(ad[0]["counts"])])
    assert ad[0]["counts"][:, 1].sum() > 0


def _ep_args(tissue, hp, beta: float = 1e-3):  # noqa: ANN001, ANN202
    v = jnp.asarray(tissue.vertices, jnp.float64)
    edges, faces = jnp.asarray(tissue.edges), jnp.asarray(tissue.faces)
    return (v, edges, faces, WIDTH, HEIGHT, jnp.zeros(v.shape[0]), hp, jnp.ones(tissue.nb_faces), v, edges, faces,
            _energy, cost_v2v, optax.sgd(0.01), optax.sgd(1e-3), MIN_DIST, 200, 0.0, 5, None, None, None, None,
            beta)


def test_recorder_collects_both_equilibrium_propagation_solves(tissue) -> None:  # noqa: ANN001
    with enable_x64():
        hp = _tensions(tissue)
        args = _ep_args(tissue, hp)
        plain = outer_eq_prop(*args)
        with recording_t1_statistics() as rec:
            recorded = outer_eq_prop(*args)
        again = outer_eq_prop(*args)
    ep = [r for r in rec if r["source"] == "equilibrium propagation"]
    assert len(ep) == 2, "the free and the nudged solve must each be recorded"
    for r in ep:
        assert r["counts"].shape[1] == 2
        assert r["counts"][:, 1].sum() > 0, "the test tissue no longer undergoes T1 transitions in the EP solves"
        assert np.all(r["counts"][:, 1] <= r["counts"][:, 0])
    # counting changes nothing the solves compute, and leaves the uncounted path as it was
    for a, b, c in zip(plain, recorded, again):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))
