"""L-BFGS in the minimisation loop, and the opt-in solver reset after a T1 transition.

* ``optax.lbfgs()`` relaxes a small periodic tissue with T1 transitions to a force residual below
  1e-10 in float64 through ``vertax.opt.minimize``, with ``reset_solver_on_t1=True``;
* ``reset_solver_on_t1`` is off by default, and with the topology fixed it changes nothing (the reset
  fires only when a T1 transition changed the topology).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.cg import cg_linesearch
from vertax.energy import energy_shape_factor_homo
from vertax.opt import OptimizationTarget, minimize
from vertax.rheology import t1_event_count, topology_signature
from vertax.topo import do_not_update_T1, update_T1

WIDTH = HEIGHT = 4.0


def _energy(v, h, f, _vp, _hp, fp):  # noqa: ANN001, ANN202
    return energy_shape_factor_homo(v, h, f, WIDTH, HEIGHT, jnp.atleast_1d(fp))


def _relax(solver, t1_func=update_T1, n_iter: int = 2000, **kw):  # noqa: ANN001, ANN202
    mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
    v = jnp.asarray(mesh.vertices, dtype=jnp.float64)
    he, fa = jnp.asarray(mesh.edges), jnp.asarray(mesh.faces)
    fp = jnp.full(mesh.nb_faces, 3.7)
    (v2, h2, f2, *_), (hist, _n) = minimize(
        v, he, fa, WIDTH, HEIGHT, jnp.zeros(v.shape[0]), jnp.ones(mesh.nb_half_edges), fp, _energy, solver, 0.05,
        iterations_max=n_iter, tolerance=0.0, optimization_target=OptimizationTarget.VERTICES,
        update_t1_func=t1_func, perform_energy_check=True, cap_vertex_step=True, **kw)
    g = jax.grad(lambda x: _energy(x, h2, f2, None, None, fp))(v2)
    return np.asarray(v2), np.asarray(h2), float(jnp.linalg.norm(g)), t1_event_count(topology_signature(he),
                                                                                         topology_signature(h2))


def test_lbfgs_converges_with_t1_and_reset() -> None:
    with enable_x64():
        _v, _h, gnorm, events = _relax(optax.lbfgs(), reset_solver_on_t1=True)
    assert events > 0, "the test tissue no longer undergoes T1 transitions"
    assert gnorm < 1e-10


def test_reset_is_off_by_default() -> None:
    with enable_x64():
        a = _relax(cg_linesearch(), n_iter=300)
        b = _relax(cg_linesearch(), n_iter=300, reset_solver_on_t1=False)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_reset_changes_nothing_without_topology_changes() -> None:
    with enable_x64():
        a = _relax(optax.lbfgs(), t1_func=do_not_update_T1, n_iter=300)
        b = _relax(optax.lbfgs(), t1_func=do_not_update_T1, n_iter=300, reset_solver_on_t1=True)
    np.testing.assert_array_equal(a[0], b[0])
