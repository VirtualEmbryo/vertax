"""Reset of the line tension of an edge created by a T1 transition in a bounded mesh (`T1TensionReset`).

In a bounded mesh an edge keeps its index, and so its tension, through a T1. An edge whose tension
breaks the triangle inequality at its vertices (here 3 against neighbours at 1) collapses; under the
forced T1 rule the new edge inherits the same tension, collapses again, and the edge flips on almost
every relaxation step. With the reset the new edge starts at the mean tension of the other edges of its
kind, and the flipping stops.

What is checked:

* with the option off nothing changes (the library's bounded regression tests check the numbers; here,
  with the option on and no T1 firing, the state and parameters equal those without the option, bit
  for bit);
* the reset value is the mean tension of the other edges of the new edge's kind, for raw and for
  sigmoid tensions;
* an edge that flip-flops under the forced rule flips once and stays open with the reset;
* automatic differentiation through a relaxation in which a reset happens agrees with central finite
  differences of the same relaxation;
* the estimators apply their outer update to the post-reset parameters and report the resets.

A property of the bounded T1 routine, measured here and stated in the flip-flop test: the new edge is
created at ``min_dist_T1 + 1e-3``, so a single relaxation step can shorten it back below the threshold.
Of the eight flip-flopping edges of this cluster, one flips once and stays (the edge tested), and seven
flip back on the very next step, after which the restored edge, now at the mean tension, stays open.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import BilevelOptimizationMethod, BoundedBilevelOptimizer, BoundedMesh, T1TensionReset
from vertax.energy import make_energy_line_tensions_bounded
from vertax.opt_bounded import cost_ad_bounded, inner_opt_bounded
from vertax.topo import update_T1_bounded



# float64 for this module only: `jax.config.update("jax_enable_x64", True)` at import would switch it on
# for the whole pytest process and change tests whose references are float32 (see
# test_bounded_t1_repair_guard.py). Module scope, so the module-scoped cluster is built in float64 too.
@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_this_module():  # noqa: PT004
    with enable_x64():
        yield

K_AREAS = 40.0
TARGET_AREA = 0.6
MIN_DIST_T1 = 0.025
STEP = 2e-3
HIGH_TENSION = 3.0  # breaks the triangle inequality against two neighbours at tension 1
FLIP_FLOP_EDGE = 4  # interior edge of the 12-cell cluster below that flip-flops without the reset

ENERGY = make_energy_line_tensions_bounded(K_AREAS)
SOLVER = optax.sgd(STEP)


def _relax(state, tensions, steps, **kwargs):  # noqa: ANN001, ANN003, ANN202
    vt, at, he, fa = state
    return inner_opt_bounded(
        vt, at, he, fa, jnp.zeros(1), tensions, jnp.full(fa.shape[0], TARGET_AREA), ENERGY, SOLVER, MIN_DIST_T1,
        steps, 0.0, 5, **kwargs,
    )


@pytest.fixture(scope="module")
def cluster():  # noqa: ANN201
    """A 12-cell cluster relaxed at uniform tension 1 (energy-checked T1), float64."""
    side = math.sqrt(12)
    m = BoundedMesh.from_random_seeds(nb_seeds=12, width=side, height=side, random_key=1)
    state = (jnp.asarray(np.asarray(m.vertices, np.float64)), jnp.asarray(np.asarray(m.angles, np.float64)),
             jnp.asarray(m.edges), jnp.asarray(m.faces))
    n_edges = state[2].shape[0] // 2
    state, _ = _relax(state, jnp.ones(n_edges), 6000, perform_energy_check=True)
    return state


def _interior(he: jnp.ndarray) -> np.ndarray:
    he = np.asarray(he)
    return (he[0::2, 3] != 0) & (he[1::2, 3] != 0)


def _length(state, e: int) -> float:  # noqa: ANN001
    vt, he = np.asarray(state[0]), np.asarray(state[2])
    a = he[2 * e]
    return float(np.linalg.norm(vt[a[4] + a[6] - 1 - 2] - vt[a[3] + a[5] - 2]))


def _cells(he: jnp.ndarray, e: int) -> tuple[int, int]:
    he = np.asarray(he)
    return tuple(sorted((int(he[2 * e, 7]), int(he[2 * e + 1, 7]))))


def test_without_any_t1_the_option_changes_nothing(cluster) -> None:  # noqa: ANN001
    n_edges = cluster[2].shape[0] // 2
    tensions = jnp.linspace(1.0, 1.3, n_edges)  # within the triangle inequality: no edge collapses
    off, _ = _relax(cluster, tensions, 300)
    on, _, resets = _relax(cluster, tensions, 300, tension_reset=T1TensionReset())
    assert int(resets.flips.sum()) == 0
    for a, b in zip(off, on, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(resets.he_params), np.asarray(tensions))


@pytest.mark.parametrize("sigmoid", [False, True])
def test_reset_value_is_the_mean_tension_of_the_other_edges_of_its_kind(cluster, sigmoid: bool) -> None:  # noqa: ANN001
    vt, at, he, fa = cluster
    n_edges = he.shape[0] // 2
    interior = _interior(he)
    e = FLIP_FLOP_EDGE
    assert interior[e]
    # shorten edge e below the threshold about its midpoint, so the next T1 call flips it
    h = np.asarray(he)
    s, t = h[2 * e, 3] - 2, h[2 * e, 4] - 2
    v = np.asarray(vt).copy()
    mid, d = (v[s] + v[t]) / 2, (v[t] - v[s]) / np.linalg.norm(v[t] - v[s])
    v[s], v[t] = mid - 0.4 * MIN_DIST_T1 * d, mid + 0.4 * MIN_DIST_T1 * d
    params = jnp.asarray(np.random.default_rng(0).normal(0.0, 1.0, n_edges) if sigmoid else
                         np.random.default_rng(0).uniform(1.0, 2.0, n_edges))
    energy = make_energy_line_tensions_bounded(K_AREAS, sigmoid_tensions=sigmoid)
    out = update_T1_bounded(jnp.asarray(v), at, he, fa, jnp.zeros(1), params, jnp.full(fa.shape[0], TARGET_AREA),
                            energy, MIN_DIST_T1, None, None, None, False,
                            tension_reset=T1TensionReset(sigmoid_tensions=sigmoid))
    new_he, new_params, flips = out[2], np.asarray(out[4]), np.asarray(out[5])
    assert flips[e] == 1 and flips.sum() == 1
    assert _cells(new_he, e) != _cells(he, e)
    others = (_interior(new_he) == _interior(new_he)[e]) & (np.arange(n_edges) != e)
    p = np.asarray(params)
    tension = 1 + 1 / (1 + np.exp(-p)) if sigmoid else p
    expected_tension = tension[others].mean()
    new_tension = 1 + 1 / (1 + np.exp(-new_params[e])) if sigmoid else new_params[e]
    assert new_tension == pytest.approx(expected_tension, rel=1e-12)
    keep = np.arange(n_edges) != e
    assert np.array_equal(new_params[keep], p[keep])


def test_flip_flopping_edge_flips_once_and_stays(cluster) -> None:  # noqa: ANN001
    n_edges = cluster[2].shape[0] // 2
    assert _interior(cluster[2])[FLIP_FLOP_EDGE]
    tensions = jnp.ones(n_edges).at[FLIP_FLOP_EDGE].set(HIGH_TENSION)

    def forced_without_reset(vt, at, he, fa, vp, hp, fp, L_in, d, sv, sh, sf, check, tension_reset=None):  # noqa: ANN001, ANN202, ARG001, N803
        """The forced rule with tensions left as they are, counting the transitions (the dynamics without the reset)."""
        out = update_T1_bounded(vt, at, he, fa, vp, hp, fp, L_in, d, sv, sh, sf, False, tension_reset=T1TensionReset())
        return (*out[:4], hp, out[5])

    steps = 1500
    state_off, _, count_off = _relax(cluster, tensions, steps, update_T1_func=forced_without_reset,
                                     tension_reset=T1TensionReset())
    state_ref, _ = _relax(cluster, tensions, steps)  # the library's forced rule without the option
    for a, b in zip(state_off, state_ref, strict=True):  # the counting wrapper does not change the dynamics
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert int(count_off.flips[FLIP_FLOP_EDGE]) > 100, "the edge no longer flip-flops without the reset"
    assert _length(state_off, FLIP_FLOP_EDGE) < 1.1 * MIN_DIST_T1

    state_on, _, resets = _relax(cluster, tensions, steps, tension_reset=T1TensionReset())
    assert int(resets.flips[FLIP_FLOP_EDGE]) == 1
    assert _cells(state_on[2], FLIP_FLOP_EDGE) != _cells(cluster[2], FLIP_FLOP_EDGE)
    assert _length(state_on, FLIP_FLOP_EDGE) > 5 * MIN_DIST_T1
    assert float(resets.he_params[FLIP_FLOP_EDGE]) == pytest.approx(1.0, rel=1e-12)


def test_ad_through_a_reset_agrees_with_finite_differences(cluster) -> None:  # noqa: ANN001
    n_edges = cluster[2].shape[0] // 2
    rng = np.random.default_rng(3)
    tensions = jnp.asarray(rng.uniform(1.0, 1.2, n_edges)).at[FLIP_FLOP_EDGE].set(HIGH_TENSION)
    vt, at, he, fa = cluster
    steps = 300
    a0 = jnp.full(fa.shape[0], TARGET_AREA)

    def outer(vt_, at_, he_, fa_, *_):  # noqa: ANN001, ANN202
        return jnp.sum((vt_ - jnp.mean(vt_, axis=0)) ** 2 * jnp.array([1.0, -1.0]))

    def run(p):  # noqa: ANN001, ANN202
        return cost_ad_bounded(vt, at, he, fa, jnp.zeros(1), p, a0, None, None, None, None, ENERGY, outer, SOLVER,
                               MIN_DIST_T1, steps, 0.0, 5, tension_reset=T1TensionReset())

    (value, aux), grad = jax.value_and_grad(run, has_aux=True)(tensions)
    resets = aux[4]  # aux: vertices, angles, edges, faces of the equilibrium, then the resets
    assert int(resets.flips[FLIP_FLOP_EDGE]) >= 1, "no reset happens in the differentiated relaxation"
    grad = np.asarray(grad)
    fd = np.zeros(n_edges)
    d = 1e-6
    for k in range(n_edges):
        plus, aux_p = run(tensions.at[k].add(d))
        minus, aux_m = run(tensions.at[k].add(-d))
        assert np.array_equal(np.asarray(aux_p[4].flips), np.asarray(resets.flips))
        assert np.array_equal(np.asarray(aux_m[4].flips), np.asarray(resets.flips))
        fd[k] = (float(plus) - float(minus)) / (2 * d)
    assert np.linalg.norm(grad - fd) / np.linalg.norm(fd) < 1e-6
    # the reset makes the new edge's tension depend on every other interior edge: their derivatives are
    # not those of a relaxation with the reset value held fixed
    interior = _interior(he)
    assert np.count_nonzero(np.abs(fd[interior]) > 1e-8) > interior.sum() // 2


@pytest.mark.parametrize(
    "method",
    [BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION, BilevelOptimizationMethod.ADJOINT_STATE,
     BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION],
)
def test_estimators_update_the_post_reset_parameters(cluster, method) -> None:  # noqa: ANN001
    n_edges = cluster[2].shape[0] // 2
    mesh = BoundedMesh.from_random_seeds(nb_seeds=12, width=math.sqrt(12), height=math.sqrt(12), random_key=1)
    mesh.vertices, mesh.angles, mesh.edges, mesh.faces = cluster
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.ones(n_edges).at[FLIP_FLOP_EDGE].set(HIGH_TENSION)
    mesh.faces_params = jnp.full(mesh.nb_faces, TARGET_AREA)
    bop = BoundedBilevelOptimizer()
    bop.min_dist_T1 = MIN_DIST_T1
    bop.max_nb_iterations = 300
    bop.tolerance = 0.0
    bop.inner_solver = SOLVER
    bop.outer_solver = optax.sgd(0.0)  # no step: the stored parameters are the ones the update was applied to
    bop.loss_function_inner = ENERGY
    bop.loss_function_outer = lambda vt, *_: jnp.sum(vt[:, 0] ** 2)
    bop.bilevel_optimization_method = method
    bop.beta = 1e-3
    bop.t1_tension_reset = T1TensionReset()
    bop.outer_optimization(mesh)
    report = bop.last_t1_resets
    assert report is not None
    assert all(int(f[FLIP_FLOP_EDGE]) == 1 for f in report.flips)
    assert report.nudged_mismatch == 0
    assert np.array_equal(np.asarray(mesh.edges_params), np.asarray(report.he_params))
    assert float(mesh.edges_params[FLIP_FLOP_EDGE]) == pytest.approx(1.0, rel=1e-12)
    assert _cells(mesh.edges, FLIP_FLOP_EDGE) != _cells(cluster[2], FLIP_FLOP_EDGE)
