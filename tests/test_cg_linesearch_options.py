"""Nonlinear CG in the minimisation loop: float64 type stability and the line-search start option.

Two properties of ``vertax.cg`` that relaxation drivers rely on:

* under ``jax_enable_x64`` the zoom line search returns an integer counter whose width differs from
  the one it was initialised with; ``vertax.opt.minimize`` selects between two solver states with
  ``lax.cond``, which requires identical types, so the solver must keep its integer dtypes. It does
  so without changing any value;
* ``linesearch_initial_guess`` defaults to ``"keep"`` (the behaviour every committed result was
  produced with) and ``"one"`` is a different, still descending, trajectory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.cg import cg_linesearch, gd_linesearch
from vertax.energy import energy_shape_factor_homo
from vertax.opt import OptimizationTarget, minimize
from vertax.topo import update_T1

WIDTH, HEIGHT, N_CELLS = 4.0, 4.0, 16
MIN_DIST_T1 = 0.005


def _energy(v, h, f, _vp, _hp, fp):  # noqa: ANN001, ANN202
    return energy_shape_factor_homo(v, h, f, WIDTH, HEIGHT, jnp.atleast_1d(fp))


def _relax(solver, dtype, n_iter: int = 40):  # noqa: ANN001, ANN202
    mesh = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=WIDTH, height=HEIGHT, random_key=3)
    v = jnp.asarray(mesh.vertices, dtype=dtype)
    fp = jnp.full(mesh.nb_faces, 3.7, dtype=dtype)
    (v_out, _h, _f, _a, _b, _c), (hist, _n) = minimize(
        v, jnp.asarray(mesh.edges), jnp.asarray(mesh.faces), WIDTH, HEIGHT,
        jnp.zeros(v.shape[0], dtype), jnp.ones(mesh.nb_half_edges, dtype), fp, _energy, solver,
        MIN_DIST_T1, iterations_max=n_iter, tolerance=0.0, patience=5,
        optimization_target=OptimizationTarget.VERTICES, update_t1_func=update_T1,
        perform_energy_check=True, cap_vertex_step=True)
    return np.asarray(v_out), np.asarray(hist)


@pytest.mark.parametrize("factory", [cg_linesearch, gd_linesearch])
def test_line_search_solvers_run_in_float64_inside_minimize(factory) -> None:  # noqa: ANN001
    with enable_x64():
        v, hist = _relax(factory(), jnp.float64)
    assert v.dtype == np.float64
    assert np.all(np.isfinite(hist))
    assert hist[-1] < hist[0]


def test_integer_state_keeps_its_dtype_across_updates() -> None:
    with enable_x64():
        solver = cg_linesearch()
        params = jnp.ones(3, jnp.float64)

        def f(p):  # noqa: ANN001, ANN202
            return jnp.sum((p - 2.0) ** 2)

        state = solver.init(params)
        g = jax.grad(f)(params)
        _upd, new_state = solver.update(g, state, params, value=f(params), grad=g, value_fn=f)
        before = [x.dtype for x in jax.tree.leaves(state)]
        after = [x.dtype for x in jax.tree.leaves(new_state)]
    assert before == after


def test_default_line_search_start_is_keep() -> None:
    v_default, h_default = _relax(cg_linesearch(), jnp.float32)
    v_keep, h_keep = _relax(cg_linesearch(linesearch_initial_guess="keep"), jnp.float32)
    np.testing.assert_array_equal(v_default, v_keep)
    np.testing.assert_array_equal(h_default, h_keep)


def test_unit_line_search_start_descends_on_its_own_path() -> None:
    v_keep, h_keep = _relax(cg_linesearch(), jnp.float32)
    v_one, h_one = _relax(cg_linesearch(linesearch_initial_guess="one"), jnp.float32)
    assert h_one[-1] < h_one[0]
    assert not np.array_equal(v_keep, v_one)
