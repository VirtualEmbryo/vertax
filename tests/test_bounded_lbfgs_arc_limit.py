"""A line-search solver on a bounded mesh whose free arcs start on their half-circle limit.

The relaxation clips every arc half-angle to ``[ARC_HALF_ANGLE_MIN, ARC_HALF_ANGLE_MAX]`` after each
update, but the line search used to score its trial steps with the angles unclipped. With an arc on its
limit and the energy pulling it past, the step the line search accepted was then no descent step: the
energy rose, and on a 100-cell tissue the relaxation stalled with a residual force on the arc's ends.

Here the free arcs start on the limit, under a line tension of 0.2 against 1 inside, which pulls them
past it, and ``optax.lbfgs()`` or ``gd_linesearch`` relaxes the tissue with T1 transitions disabled. Over
the first 25 steps, before any edge shrinks near zero length (where the length is not smooth), the
energy must never rise. Without the clip in the line search, both raised it within those steps.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import BoundedMesh
from vertax.cg import gd_linesearch
from vertax.energy import make_energy_line_tensions_bounded
from vertax.opt_bounded import inner_opt_bounded
from vertax.topo import ARC_HALF_ANGLE_MAX, do_not_update_T1_bounded


def _relax(solver, n_iter: int = 25):  # noqa: ANN001, ANN202
    """(energy history, free arcs on the upper limit pulled past it at the start)."""
    n_cells = 20
    side = math.sqrt(n_cells)
    m = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=3)
    vt, he, fa = jnp.asarray(m.vertices), jnp.asarray(m.edges), jnp.asarray(m.faces)
    free = np.unique(np.nonzero(np.asarray(he)[:, 5] != 0)[0] // 2)
    at = jnp.asarray(m.angles).at[free].set(ARC_HALF_ANGLE_MAX)     # every free arc on the limit
    energy = make_energy_line_tensions_bounded(10.0)
    vp, tension, a0 = jnp.zeros(1), jnp.ones(at.shape[0]).at[free].set(0.2), jnp.full(m.nb_faces, 1.0)
    sv, sh, sf = jnp.arange(vt.shape[0]), jnp.arange(he.shape[0]), jnp.arange(fa.shape[0])
    ga = jax.grad(lambda a: energy(vt, a, he, fa, sv, sh, sf, vp, tension, a0))(at)
    n_pulled = int((np.asarray(ga)[free] < 0).sum())
    # No vertex step cap: it serves the T1 detector, off here, and it too moves the point a step
    # reaches away from the one the line search scored, which would mix a second effect in.
    _tables, hist = inner_opt_bounded(
        vt, at, he, fa, vp, tension, a0, energy, solver, 0.005, iterations_max=n_iter, tolerance=0.0,
        patience=5, update_T1_func=do_not_update_T1_bounded, perform_energy_check=False, cap_vertex_step=False)
    return np.asarray(hist), n_pulled


@pytest.mark.parametrize("make_solver", [optax.lbfgs, gd_linesearch], ids=["lbfgs", "gd_linesearch"])
def test_line_search_never_raises_the_energy_against_the_arc_limit(make_solver) -> None:  # noqa: ANN001
    with enable_x64():
        hist, n_pulled = _relax(make_solver())
    assert n_pulled > 0, "the setup no longer pulls any arc past its limit"
    rises = np.diff(hist)
    assert rises.max() <= 1e-12 * np.abs(hist).max(), f"the energy rose by up to {rises.max():.3e}"
