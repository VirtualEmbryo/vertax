"""L-BFGS in the bounded minimisation loop, and the opt-in solver reset after a T1 transition.

* ``optax.lbfgs()`` relaxes a small bounded tissue with T1 transitions to a force residual (on the
  vertices and the arc angles) below 1e-10 in float64 through ``vertax.opt_bounded.inner_opt_bounded``,
  with ``reset_solver_on_t1=True``;
* ``reset_solver_on_t1`` is off by default, and with the topology fixed it changes nothing (the reset
  fires only when a T1 transition changed the topology).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import enable_x64

from vertax import BoundedMesh
from vertax.cg import cg_linesearch
from vertax.energy import make_energy_line_tensions_bounded
from vertax.opt_bounded import inner_opt_bounded
from vertax.topo import do_not_update_T1_bounded, update_T1_bounded

N_CELLS = 20
_ENERGY = make_energy_line_tensions_bounded(10.0)


def _relax(solver, t1_func=update_T1_bounded, n_iter: int = 2000, **kw):  # noqa: ANN001, ANN202
    side = math.sqrt(N_CELLS)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=N_CELLS, width=side, height=side, random_key=3)
    v = jnp.asarray(mesh.vertices, dtype=jnp.float64)
    a = jnp.asarray(mesh.angles, dtype=jnp.float64)
    he, fa = jnp.asarray(mesh.edges), jnp.asarray(mesh.faces)
    vp, tension, a0 = jnp.zeros(1), jnp.ones(a.shape[0]), jnp.full(mesh.nb_faces, 1.0)
    (v2, a2, h2, f2), _hist = inner_opt_bounded(
        v, a, he, fa, vp, tension, a0, _ENERGY, solver, 0.005, iterations_max=n_iter, tolerance=0.0, patience=5,
        update_T1_func=t1_func, perform_energy_check=True, cap_vertex_step=True, **kw)
    sv, sh, sf = jnp.arange(v2.shape[0]), jnp.arange(h2.shape[0]), jnp.arange(f2.shape[0])
    gv, ga = jax.grad(_ENERGY, argnums=(0, 1))(v2, a2, h2, f2, sv, sh, sf, vp, tension, a0)
    gnorm = float(jnp.sqrt(jnp.sum(gv**2) + jnp.sum(ga**2)))
    return np.asarray(v2), np.asarray(a2), np.asarray(h2), gnorm, not np.array_equal(np.asarray(he), np.asarray(h2))


def test_lbfgs_converges_with_t1_and_reset() -> None:
    with enable_x64():
        _v, _a, _h, gnorm, topology_changed = _relax(optax.lbfgs(), reset_solver_on_t1=True)
    assert topology_changed, "the test tissue no longer undergoes T1 transitions"
    assert gnorm < 1e-10


def test_reset_is_off_by_default() -> None:
    with enable_x64():
        a = _relax(cg_linesearch(), n_iter=300)
        b = _relax(cg_linesearch(), n_iter=300, reset_solver_on_t1=False)
    for x, y in zip(a[:3], b[:3], strict=True):
        np.testing.assert_array_equal(x, y)


def test_reset_changes_nothing_without_topology_changes() -> None:
    with enable_x64():
        a = _relax(optax.lbfgs(), t1_func=do_not_update_T1_bounded, n_iter=300)
        b = _relax(optax.lbfgs(), t1_func=do_not_update_T1_bounded, n_iter=300, reset_solver_on_t1=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
