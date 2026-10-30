"""``optax.lbfgs()`` on a periodic tissue whose vertices wrap across the box.

``scale_by_lbfgs`` stores the positions it was last called at and builds its next curvature pair from the
difference with the new ones. ``update_pbc`` wraps a vertex that left the box back inside, a jump by a box
width that the solver never took; the pair used to include it, the next direction was no descent
direction, and the energy rose at the step after every wrap. The loop now shifts the stored positions by
the same jump.

Here a periodic tissue, far from equilibrium, is relaxed by ``optax.lbfgs()`` with T1 transitions disabled
and no vertex step cap. Vertices wrap from the first step on; over the first 25 steps the energy must never
rise.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.energy import make_energy_line_tensions
from vertax.opt import OptimizationTarget, minimize
from vertax.topo import do_not_update_T1

WIDTH = HEIGHT = 4.0
E2 = make_energy_line_tensions(10.0)


def _energy(v, h, f, _vp, hp, fp):  # noqa: ANN001, ANN202
    return E2(v, h, f, WIDTH, HEIGHT, hp, fp)


def _relax(n_iter: int):  # noqa: ANN202
    """(energy history, initial and final vertex positions)."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=3)
    v = jnp.asarray(mesh.vertices)
    (v_out, *_), (hist, *_) = minimize(
        v, jnp.asarray(mesh.edges), jnp.asarray(mesh.faces), WIDTH, HEIGHT, jnp.zeros(v.shape[0]),
        jnp.ones(mesh.nb_half_edges), jnp.full(mesh.nb_faces, 1.0), _energy, optax.lbfgs(), 0.01,
        iterations_max=n_iter, tolerance=0.0, patience=5, optimization_target=OptimizationTarget.VERTICES,
        update_t1_func=do_not_update_T1, perform_energy_check=False, cap_vertex_step=False)
    return np.asarray(hist), np.asarray(v), np.asarray(v_out)


def test_lbfgs_never_raises_the_energy_when_vertices_wrap() -> None:
    with enable_x64():
        _hist, v0, v1 = _relax(1)
        hist, _v0, _v = _relax(25)
    n_wrapped = int((np.abs(v1 - v0).max(axis=1) > WIDTH / 2).sum())
    assert n_wrapped > 0, "no vertex wraps across the box in the first step: the setup is stale"
    rises = np.diff(hist)
    assert rises.max() <= 1e-12 * np.abs(hist).max(), f"the energy rose by up to {rises.max():.3e}"
