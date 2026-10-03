"""Regression test for the periodic bilevel pipeline, in two tiers.

Why elementwise vertex comparison is the wrong test
----------------------------------------------------
An `assert_allclose(pbc_mesh.vertices, ref.vertices, rtol=0.001)` against a stored mesh (plus the
same on the integer `edges`/`faces` tables) is the wrong design, independently of any environment
problem:

* Raw vertex coordinates under periodic boundary conditions are not a well-defined quantity to
  compare elementwise: the same physical vertex may be stored at a different periodic
  representative, and the T1 swap repair legitimately *relabels* vertices, so vertex `k` in one run
  need not be vertex `k` in another.
* The trajectory is chaotic in the strict sense: T1 events are discrete branch decisions, so a
  last-bit arithmetic difference can flip one and separate the trajectories. Float32 makes that far
  more likely, which is why this test requires float64 throughout.

The two tiers
-------------
1. **In-environment determinism** (bit-exact, no stored artifact). Runs the pipeline twice in the
   same process and requires bit-identical output. This catches genuine nondeterminism - the thing a
   regression test should catch - without being fragile across machines.
2. **Cross-environment physical invariants** (stored, with justified tolerances). Compares scalars
   that are physically meaningful and representation-independent, never raw coordinates.

`jax_enable_x64` is enabled here: float32 is what makes the trajectory chaotic enough for a 1-ULP
difference to flip a T1 decision, and the project's standing rules require float64 for anything we
intend to interpret.

The superseded `tests/reference_result_test_inverse_modeling.npz` is left in place (results are not
deleted) but is no longer read. Regenerate the invariants with:
`python tests/test_inverse_modeling.py --regenerate`
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from jax.experimental import enable_x64

import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import pytest
from jax import vmap

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import topological_distance, cost_v2v
from vertax.geo import get_area, get_length
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    from jax import Array

INVARIANTS_PATH = Path(__file__).parent / "reference_invariants_test_inverse_modeling.json"

# Tolerances for tier 2. Justification: identical seeds and library versions reproduce these to
# float64 precision, but across machines/XLA versions a flipped T1 decision can move them. 1% is
# comfortably inside the ~8% mesh-to-mesh spread of these metrics while still catching a factor-2
# error, a sign error, or a lost term. `d_topo` is compared exactly because it is discrete: if it
# differs, the run took a different topological branch and that must fail loudly rather than be
# absorbed by a tolerance.
RTOL_SCALAR = 1e-2
RTOL_GRAD_NORM = 0.5  # weak convergence signal, nearly blind to soft modes; order-of-magnitude only


# `jax_enable_x64` is confined to this module rather than set at import scope.
# `jax.config.update("jax_enable_x64", True)` is PROCESS-WIDE, so calling it here directly would
# leak float64 into every test pytest imports afterwards, silently breaking
# `test_forward_modeling.py::test_forward_modeling_for_regressions`, whose reference is float32 over
# a chaotic relaxation: running it after a module that enables x64 gives 399/400 elements
# mismatched -- a different basin, not drift. `jax.experimental.enable_x64` restores the previous
# setting on exit (verified: the flag reads False again after the block), so the confinement is
# real and the suite stops depending on collection order.
@pytest.fixture(autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    with enable_x64():
        yield


def _run_pipeline() -> tuple[PbcMesh, dict[str, float]]:
    """Run the reference bilevel pipeline and return (final mesh, physical invariants).

    The physics below is preserved verbatim from the original test: same seeds, same energies,
    same optimizer settings, same 2 outer epochs.
    """
    n_cells = 20
    width = math.sqrt(n_cells)
    height = width
    epochs = 2
    MAX_EDGES_IN_ANY_FACE = 20
    K_AREAS = 20  # this test's own convention; asserted so it can never drift silently
    assert K_AREAS == 20, "this regression fixture is pinned to K_areas=20"

    pbc_mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
    pbc_mesh.vertices_params = jnp.asarray([0.0 for _ in range(pbc_mesh.nb_vertices)])

    mu_tensions = 1.2
    std_tensions = 0.1
    key = jax.random.PRNGKey(1)
    he_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(pbc_mesh.nb_edges,))
    pbc_mesh.edges_params = jnp.repeat(he_params, 2)
    pbc_mesh.faces_params = jnp.asarray([1.0 for _ in range(pbc_mesh.nb_faces)])
    he_params_reference = he_params[0]

    def area_part(face: Array, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        return (a - face_param) ** 2

    def hedge_part(he: Array, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        return he_param * get_length(he, vertTable, heTable, faceTable, width, height)

    def energy(
        vertTable: Array, heTable: Array, faceTable: Array, _vert_params: Array, he_params: Array, face_params: Array
    ) -> Array:
        areas_part = vmap(lambda f, p: area_part(f, p, vertTable, heTable, faceTable))(
            jnp.arange(len(faceTable)), face_params
        )
        hedges_part = vmap(lambda e, p: hedge_part(e, p, vertTable, heTable, faceTable))(
            jnp.arange(2, len(heTable)), he_params[2:]
        )
        return (
            (2 * he_params_reference * get_length(0, vertTable, heTable, faceTable, width, height))
            + jnp.sum(hedges_part)
            + (0.5 * K_AREAS) * jnp.sum(areas_part)
        )

    bop = PbcBilevelOptimizer()
    bop.update_T1 = True
    bop.min_dist_T1 = 0.005
    bop.max_nb_iterations = 1000
    bop.tolerance = 1e-4
    bop.patience = 5
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)
    bop.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION

    bop.loss_function_inner = energy
    bop.inner_optimization(mesh=pbc_mesh)

    pbc_mesh_target = PbcMesh.copy_mesh(pbc_mesh)
    key = jax.random.PRNGKey(2)
    he_params_target = mu_tensions + std_tensions * jax.random.normal(
        key, shape=(pbc_mesh_target.nb_edges,)
    )
    he_params_reference_target = he_params_target[0]
    pbc_mesh_target.vertices_params = jnp.asarray([0.0])
    pbc_mesh_target.edges_params = jnp.repeat(he_params_target, 2)
    pbc_mesh_target.faces_params = jnp.asarray([1.0 for _ in range(pbc_mesh_target.nb_faces)])
    bop.inner_optimization(mesh=pbc_mesh_target)

    bop.vertices_target = pbc_mesh_target.vertices.copy()
    bop.edges_target = pbc_mesh_target.edges.copy()
    bop.faces_target = pbc_mesh_target.faces.copy()
    del pbc_mesh_target

    fixed_areas_target = vmap(
        lambda f: get_area(
            f, bop.vertices_target, bop.edges_target, bop.faces_target, width, height, MAX_EDGES_IN_ANY_FACE
        )
    )(jnp.arange(pbc_mesh.nb_faces))

    def energy_v2(
        vertTable: Array, heTable: Array, faceTable: Array, _vert_params: Array, he_params: Array, _face_params: Array
    ) -> Array:
        areas_part = vmap(
            lambda f: (
                get_area(f, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
                - fixed_areas_target[f]
            )
            ** 2
        )(jnp.arange(len(faceTable)))
        hedges_part = vmap(lambda e, p: hedge_part(e, p, vertTable, heTable, faceTable))(
            jnp.arange(2, len(heTable)), he_params[2:]
        )
        return (
            (2 * he_params_reference_target * get_length(0, vertTable, heTable, faceTable, width, height))
            + jnp.sum(hedges_part)
            + (0.5 * K_AREAS) * jnp.sum(areas_part)
        )

    bop.loss_function_inner = energy_v2
    bop.loss_function_outer = cost_v2v
    for _ in range(epochs + 1):
        bop.bilevel_optimization(pbc_mesh)

    # ---- physical invariants (representation-independent) ----
    outer_cost = float(
        cost_v2v(
            pbc_mesh.vertices, pbc_mesh.edges, pbc_mesh.faces, pbc_mesh.width, pbc_mesh.height,
            bop.vertices_target, bop.edges_target, bop.faces_target,
        )
    )
    inner_energy = float(
        energy_v2(
            pbc_mesh.vertices, pbc_mesh.edges, pbc_mesh.faces,
            pbc_mesh.vertices_params, pbc_mesh.edges_params, pbc_mesh.faces_params,
        )
    )
    n_v = pbc_mesh.vertices.shape[0]

    def _e_flat(vflat: Array) -> Array:
        return energy_v2(
            vflat.reshape(n_v, 2), pbc_mesh.edges, pbc_mesh.faces,
            pbc_mesh.vertices_params, pbc_mesh.edges_params, pbc_mesh.faces_params,
        )

    grad_norm_E = float(
        jnp.linalg.norm(jax.grad(_e_flat)(jnp.asarray(pbc_mesh.vertices, jnp.float64).ravel()))
    )
    d_topo = float(
        topological_distance(
            pbc_mesh.vertices, pbc_mesh.edges, pbc_mesh.faces, pbc_mesh.width, pbc_mesh.height,
            bop.vertices_target, bop.edges_target, bop.faces_target,
        )
    )
    # Tension recovery against the target tensions (per canonical edge = every second half-edge).
    lam_r = np.asarray(pbc_mesh.edges_params, dtype=float)[::2]
    lam_t = np.asarray(jnp.repeat(he_params_target, 2), dtype=float)[::2]
    spread = float(np.linalg.norm(lam_t - lam_t.mean()))
    invariants = {
        "grad_norm_E": grad_norm_E,          # reported first, per standing rules
        "outer_cost_v2v": outer_cost,
        "inner_energy": inner_energy,
        "d_topo": d_topo,
        "err_over_spread": float(np.linalg.norm(lam_r - lam_t) / spread) if spread > 0 else float("nan"),
        "pearson": float(np.corrcoef(lam_r, lam_t)[0, 1]),
    }
    return pbc_mesh, invariants


def test_in_environment_determinism() -> None:
    """Tier 1: two runs in the same environment must agree bit-exactly.

    No stored artifact, so this is machine-independent. It is the check that actually catches a
    regression in determinism (uninitialised state, iteration-order dependence, nondeterministic
    reductions), which is what the old raw-coordinate comparison was failing to do.
    """
    mesh_a, inv_a = _run_pipeline()
    mesh_b, inv_b = _run_pipeline()
    np.testing.assert_array_equal(np.asarray(mesh_a.vertices), np.asarray(mesh_b.vertices))
    np.testing.assert_array_equal(np.asarray(mesh_a.edges), np.asarray(mesh_b.edges))
    np.testing.assert_array_equal(np.asarray(mesh_a.faces), np.asarray(mesh_b.faces))
    assert inv_a == inv_b, f"invariants not bit-identical across runs: {inv_a} vs {inv_b}"


@pytest.mark.skipif(not INVARIANTS_PATH.exists(), reason="reference invariants not generated yet")
def test_physical_invariants() -> None:
    """Tier 2: physical invariants must match the stored reference within stated tolerances."""
    _, inv = _run_pipeline()
    ref = json.loads(INVARIANTS_PATH.read_text())["invariants"]

    # Discrete: must match exactly, a different value means a different topological branch.
    assert inv["d_topo"] == ref["d_topo"], (
        f"topological distance changed: {inv['d_topo']} vs reference {ref['d_topo']} -- the run took "
        f"a different T1 branch; this is not absorbable by a tolerance"
    )
    assert inv["grad_norm_E"] == pytest.approx(ref["grad_norm_E"], rel=RTOL_GRAD_NORM)
    for k in ("outer_cost_v2v", "inner_energy", "err_over_spread", "pearson"):
        assert inv[k] == pytest.approx(ref[k], rel=RTOL_SCALAR), (
            f"{k}: {inv[k]} vs reference {ref[k]} (rtol={RTOL_SCALAR})"
        )


def _regenerate() -> None:
    """Write the tier-2 reference invariants for the current environment."""
    import platform

    _, inv = _run_pipeline()
    payload = {
        "generated_on": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "x64": True,
        },
        "note": (
            "Physical invariants only. Raw vertex coordinates are deliberately NOT stored: they are "
            "not comparable across periodic representatives, vertex relabelling by the T1 swap "
            "repair, or machines. See the module docstring."
        ),
        "invariants": inv,
    }
    INVARIANTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {INVARIANTS_PATH}")
    for k, v in inv.items():
        print(f"  {k} = {v!r}")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        test_in_environment_determinism()
        test_physical_invariants()
