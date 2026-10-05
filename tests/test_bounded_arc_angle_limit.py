"""Free boundary arcs of a bounded mesh stay between a chord and a half circle at every relaxation step.

The arc half-angle limit (``vertax.topo.ARC_HALF_ANGLE_MIN``, ``ARC_HALF_ANGLE_MAX``) used to be applied
only inside the T1 routine, so a relaxation with T1 transitions disabled could push an arc past a half
circle, where the arc energy is no longer smooth. Here cells are given a target area three times their
area, so every free arc bulges outward, and the relaxation runs with T1 transitions disabled.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import optax

from vertax import BoundedMesh
from vertax.energy import make_energy_line_tensions_bounded
from vertax.opt_bounded import inner_opt_bounded
from vertax.topo import ARC_HALF_ANGLE_MAX, ARC_HALF_ANGLE_MIN, do_not_update_T1_bounded, update_T1_bounded


def _relax(t1_func, n_iter: int = 200):  # noqa: ANN001, ANN202
    """Half-angles of the free arcs after every step (single-step calls; sgd carries no state)."""
    n_cells = 20
    side = math.sqrt(n_cells)
    m = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=3)
    vt, he, fa = jnp.asarray(m.vertices), jnp.asarray(m.edges), jnp.asarray(m.faces)
    at = jnp.asarray(m.angles)
    free = np.unique(np.nonzero(np.asarray(he)[:, 5] != 0)[0] // 2)
    at = at.at[free].set(ARC_HALF_ANGLE_MAX - 1e-3)        # start every free arc close to the limit
    energy = make_energy_line_tensions_bounded(10.0)
    tension, a0, solver = jnp.ones(at.shape[0]), jnp.full(m.nb_faces, 3.0), optax.sgd(0.01)
    history = []
    for _ in range(n_iter):
        (vt, at, he, fa), _hist = inner_opt_bounded(
            vt, at, he, fa, jnp.zeros(1), tension, a0, energy, solver, 0.005, iterations_max=1,
            tolerance=0.0, patience=5, update_T1_func=t1_func, perform_energy_check=False)
        history.append(np.asarray(at)[free])
    return np.array(history)


def test_arcs_stay_within_the_limit_without_t1() -> None:
    a = _relax(do_not_update_T1_bounded)
    assert a.max() <= ARC_HALF_ANGLE_MAX + 1e-7
    assert a.min() >= ARC_HALF_ANGLE_MIN - 1e-7
    assert np.any(np.abs(a - ARC_HALF_ANGLE_MAX) < 1e-6), "the setup no longer pushes any arc to the limit"


def test_arcs_stay_within_the_limit_with_t1() -> None:
    a = _relax(update_T1_bounded)
    assert a.max() <= ARC_HALF_ANGLE_MAX + 1e-7
