"""A line-search solver under ``cap_vertex_step``: the line search scores the step as the update caps it.

The relaxation caps every vertex displacement at ``MAX_VERTEX_STEP_FACTOR * min_dist_T1`` after each
update, but the line search used to score its trial steps uncapped. When the cap shortened the accepted
step, the point the tissue reached was not the one the line search had scored: its energy was never
evaluated, and the cap, applied vertex by vertex, changes the direction of the step as well as its
length, so the step need not lower the energy.

Here a periodic and a bounded tissue, far from equilibrium, are relaxed by ``optax.lbfgs()``,
``cg_linesearch`` or ``gd_linesearch`` with the cap on, a ``min_dist_T1`` small enough for the cap to
bite, and T1 transitions disabled. Over the first 25 steps the energy must never rise. Without the cap
in the line search, ``optax.lbfgs()`` raised it on the bounded tissue, by 8.6 on an energy of 156.

``optax.lbfgs()`` on the periodic tissue is left out: it raises the energy with or without the cap.
When ``update_pbc`` wraps a vertex back into the box, the position difference L-BFGS stores for its
curvature pairs jumps by a box width, and the next direction is no descent direction.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

from vertax import BoundedMesh, PbcMesh
from vertax.cg import cg_linesearch, gd_linesearch
from vertax.energy import make_energy_line_tensions, make_energy_line_tensions_bounded
from vertax.opt import MAX_VERTEX_STEP_FACTOR, OptimizationTarget, minimize
from vertax.opt_bounded import inner_opt_bounded
from vertax.topo import do_not_update_T1, do_not_update_T1_bounded

SOLVERS = {"lbfgs": optax.lbfgs, "cg_linesearch": cg_linesearch, "gd_linesearch": gd_linesearch}
MIN_DIST_T1 = 0.01
N_ITER = 25

WIDTH = HEIGHT = 4.0
E2 = make_energy_line_tensions(10.0)


def _pbc_energy(v, h, f, _vp, hp, fp):  # noqa: ANN001, ANN202
    return E2(v, h, f, WIDTH, HEIGHT, hp, fp)


def _relax_periodic(solver, cap: bool, n_iter: int):  # noqa: ANN001, ANN202
    """(energy history, largest vertex displacement over the run, in units of min_dist_T1)."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=3)
    v = jnp.asarray(mesh.vertices)
    (v_out, *_), (hist, *_) = minimize(
        v, jnp.asarray(mesh.edges), jnp.asarray(mesh.faces), WIDTH, HEIGHT, jnp.zeros(v.shape[0]),
        jnp.ones(mesh.nb_half_edges), jnp.full(mesh.nb_faces, 1.0), _pbc_energy, solver, MIN_DIST_T1,
        iterations_max=n_iter, tolerance=0.0, patience=5, optimization_target=OptimizationTarget.VERTICES,
        update_t1_func=do_not_update_T1, perform_energy_check=False, cap_vertex_step=cap)
    d = np.asarray(v_out)[:, :2] - np.asarray(v)[:, :2]
    d = (d + 0.5 * np.array([WIDTH, HEIGHT])) % np.array([WIDTH, HEIGHT]) - 0.5 * np.array([WIDTH, HEIGHT])
    return np.asarray(hist), np.linalg.norm(d, axis=1).max() / MIN_DIST_T1


def _relax_bounded(solver, cap: bool, n_iter: int):  # noqa: ANN001, ANN202
    """(energy history, largest vertex displacement over the run, in units of min_dist_T1)."""
    n_cells = 20
    side = math.sqrt(n_cells)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=1)
    vt, at = jnp.asarray(mesh.vertices), jnp.asarray(mesh.angles)
    tension = jnp.asarray(1.0 + 0.5 * np.random.default_rng(1).random(at.shape[0]))
    (vt_out, *_), hist = inner_opt_bounded(
        vt, at, jnp.asarray(mesh.edges), jnp.asarray(mesh.faces), jnp.zeros(1), tension,
        jnp.full(mesh.nb_faces, 1.5), make_energy_line_tensions_bounded(10.0), solver, MIN_DIST_T1,
        iterations_max=n_iter, tolerance=0.0, patience=5, update_T1_func=do_not_update_T1_bounded,
        perform_energy_check=False, cap_vertex_step=cap)
    d = np.asarray(vt_out)[:, :2] - np.asarray(vt)[:, :2]
    return np.asarray(hist), np.linalg.norm(d, axis=1).max() / MIN_DIST_T1


RELAX = {"periodic": _relax_periodic, "bounded": _relax_bounded}
# optax.lbfgs() on the periodic tissue: see the module docstring
CASES = [(kind, name) for kind in RELAX for name in SOLVERS if (kind, name) != ("periodic", "lbfgs")]


@pytest.mark.parametrize(("kind", "name"), CASES, ids=[f"{name}-{kind}" for kind, name in CASES])
def test_capped_line_search_never_raises_the_energy(kind: str, name: str) -> None:
    relax = RELAX[kind]
    with enable_x64():
        _hist, first_step = relax(SOLVERS[name](), False, 1)
        hist, _step = relax(SOLVERS[name](), True, N_ITER)
    assert first_step > 2 * MAX_VERTEX_STEP_FACTOR, "the uncapped first step stays under the cap: the setup is stale"
    rises = np.diff(hist)
    assert rises.max() <= 1e-12 * np.abs(hist).max(), f"the energy rose by up to {rises.max():.3e}"
