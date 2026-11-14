"""Length of the edge created by a bounded T1 transition (``reopen_margin`` of ``update_T1_bounded``).

By default the bounded T1 routine gives the new edge the length ``min_dist_T1 + 1e-3`` about the old
edge's midpoint, the value every committed bounded result was produced with; with ``reopen_margin`` the
length is ``(1 + reopen_margin) * min_dist_T1``, as for the periodic ``update_T1``. At the default, 1.04
``min_dist_T1`` for ``min_dist_T1 = 0.025``, one relaxation step can shorten the new edge below the
threshold again, so it flips straight back.

Checked here, on a 12-cell cluster in which one interior edge at tension 3 among edges at 1 collapses
(the forced rule with the tension reset, `T1TensionReset`):

* omitting the keyword and passing ``reopen_margin=None`` give identical results (the library's bounded
  regression tests check the default against committed numbers);
* the new edge has the stated length, by default and with a margin;
* at the default, the edge flips back on the step after its first flip; reopened at 1.5 or 2
  ``min_dist_T1`` it does not flip on the next step. Edge 7 then stays in its new topology. Edge 1 still
  returns a few steps later: once its tension is reset to that of its neighbours nothing holds the new
  junction open, which is a property of this single-edge test, not of the reopen length.
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import BoundedMesh, T1TensionReset
from vertax.energy import make_energy_line_tensions_bounded
from vertax.opt_bounded import inner_opt_bounded
from vertax.topo import update_T1_bounded



# float64 for this module only: `jax.config.update("jax_enable_x64", True)` at import would switch it on
# for the whole pytest process and change tests whose references are float32 (see
# test_bounded_t1_repair_guard.py). Module scope, so the module-scoped cluster is built in float64 too.
@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_this_module():  # noqa: PT004
    with enable_x64():
        yield

MIN_DIST_T1 = 0.025
TARGET_AREA = 0.6
ENERGY = make_energy_line_tensions_bounded(40.0)
SOLVER = optax.sgd(2e-3)


@pytest.fixture(scope="module")
def cluster():  # noqa: ANN201
    """A 12-cell cluster relaxed at uniform tension 1 (energy-checked T1), float64."""
    side = math.sqrt(12)
    m = BoundedMesh.from_random_seeds(nb_seeds=12, width=side, height=side, random_key=1)
    state = (jnp.asarray(np.asarray(m.vertices, np.float64)), jnp.asarray(np.asarray(m.angles, np.float64)),
             jnp.asarray(m.edges), jnp.asarray(m.faces))
    n_edges = state[2].shape[0] // 2
    state, _ = inner_opt_bounded(*state, jnp.zeros(1), jnp.ones(n_edges), jnp.full(12, TARGET_AREA), ENERGY, SOLVER,
                                 MIN_DIST_T1, 6000, 0.0, 5, perform_energy_check=True)
    return state


def _length(vt, he, e: int) -> float:  # noqa: ANN001
    vt, he = np.asarray(vt), np.asarray(he)
    a = he[2 * e]
    return float(np.linalg.norm(vt[a[4] + a[6] - 1 - 2] - vt[a[3] + a[5] - 2]))


def _shortened(cluster, e: int):  # noqa: ANN001, ANN202
    """``cluster`` with interior edge ``e`` shortened to 0.8 ``min_dist_T1`` about its midpoint."""
    vt, at, he, fa = cluster
    h = np.asarray(he)
    s, t = h[2 * e, 3] - 2, h[2 * e, 4] - 2
    v = np.asarray(vt).copy()
    mid, d = (v[s] + v[t]) / 2, (v[t] - v[s]) / np.linalg.norm(v[t] - v[s])
    v[s], v[t] = mid - 0.4 * MIN_DIST_T1 * d, mid + 0.4 * MIN_DIST_T1 * d
    return jnp.asarray(v), at, he, fa


def _t1(state, **kwargs):  # noqa: ANN001, ANN003, ANN202
    vt, at, he, fa = state
    n_edges = he.shape[0] // 2
    return update_T1_bounded(vt, at, he, fa, jnp.zeros(1), jnp.ones(n_edges), jnp.full(fa.shape[0], TARGET_AREA),
                             ENERGY, MIN_DIST_T1, None, None, None, False, **kwargs)


def test_default_is_unchanged(cluster) -> None:  # noqa: ANN001
    state = _shortened(cluster, 4)
    for a, b in zip(_t1(state), _t1(state, reopen_margin=None), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize(("margin", "expected"), [(None, MIN_DIST_T1 + 1e-3), (0.5, 1.5 * MIN_DIST_T1),
                                                  (1.0, 2.0 * MIN_DIST_T1)])
def test_new_edge_has_the_stated_length(cluster, margin, expected: float) -> None:  # noqa: ANN001
    state = _shortened(cluster, 4)
    vt, _, he, _ = _t1(state, reopen_margin=margin)
    assert not np.array_equal(np.asarray(he), np.asarray(state[2])), "the edge was not flipped"
    assert _length(vt, he, 4) == pytest.approx(expected, rel=1e-12)


def _flip_steps(cluster, e: int, margin: float | None, steps: int) -> list[int]:  # noqa: ANN001
    """Steps at which edge ``e``, at tension 3 among edges at 1, flips (forced rule, tension reset)."""
    n_edges = cluster[2].shape[0] // 2
    t1 = functools.partial(update_T1_bounded, reopen_margin=margin)
    tensions, state, flips = jnp.ones(n_edges).at[e].set(3.0), cluster, []
    for k in range(steps):
        state, _, resets = inner_opt_bounded(*state, jnp.zeros(1), tensions, jnp.full(12, TARGET_AREA), ENERGY, SOLVER,
                                             MIN_DIST_T1, 1, 0.0, 5, update_T1_func=t1, perform_energy_check=False,
                                             tension_reset=T1TensionReset())
        tensions = resets.he_params
        if int(resets.flips[e]):
            flips.append(k)
    return flips


@pytest.mark.parametrize("edge", [7, 1])
def test_reopened_edge_does_not_flip_back_on_the_next_step(cluster, edge: int) -> None:  # noqa: ANN001
    default = _flip_steps(cluster, edge, None, 200)
    assert len(default) == 2 and default[1] == default[0] + 1, "the setup no longer flips back on the next step"
    for margin in (0.5, 1.0):
        flips = _flip_steps(cluster, edge, margin, 200)
        assert flips[0] == default[0] or abs(flips[0] - default[0]) <= 2
        assert all(b - a > 1 for a, b in zip(flips, flips[1:], strict=False))
        if edge == 7:
            assert len(flips) == 1
