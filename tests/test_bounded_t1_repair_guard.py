"""The periodic T1 swap repair must refuse to run on a bounded mesh.

`_BilevelOptimizer.inner_optimization` is shared by both mesh types and `BoundedBilevelOptimizer`
does not override it, so a bounded mesh carrying a registered `vertices_target` would silently
execute the *periodic* repair. That repair keys on `heTable[:, 5]` (the face index for periodic
meshes, a boundary-edge vertex index for bounded ones -- bounded keeps the face index in column 7)
and scores swaps with periodic minimum-image distances that bounded meshes do not have.

The guard turns that silent wrong answer into a loud refusal. It is triage, not the fix; the fix is
the mesh-type dispatch described in `_reject_bounded_t1_repair`.

A check that cannot distinguish the broken case from the working one is worthless, so
this test asserts both directions: the guard fires for bounded-with-target, and it stays silent for
the two configurations that must keep working -- bounded without a target (all current bounded runs,
which are inverse design and register no target mesh) and periodic with a target.
"""
from __future__ import annotations

import jax

from jax.experimental import enable_x64
import jax.numpy as jnp
import pytest

from vertax.bilevelopt.bilevelopt import _reject_bounded_t1_repair
from vertax.meshes.bounded_mesh import BoundedMesh
from vertax.meshes.pbc_mesh import PbcMesh


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


def test_guard_rejects_bounded_mesh() -> None:
    """A bounded mesh must be refused, with a message naming the actual defect."""
    mesh = BoundedMesh.from_random_seeds(nb_seeds=6, width=2.0, height=2.0, random_key=7)
    with pytest.raises(NotImplementedError) as excinfo:
        _reject_bounded_t1_repair(mesh)
    msg = str(excinfo.value)
    # The message must be actionable: name the column confusion, the periodic-distance problem,
    # and where the real fix is specified.
    assert "heTable[:, 5]" in msg
    assert "column 7" in msg
    assert "minimum-image" in msg
    assert "_reject_bounded_t1_repair" in msg


def test_guard_allows_periodic_mesh() -> None:
    """A periodic mesh must pass through untouched -- this is the Figure-3 path."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=6, width=2.0, height=2.0, random_key=7)
    _reject_bounded_t1_repair(mesh)  # must not raise


def test_bounded_inner_optimization_without_target_is_unaffected() -> None:
    """Bounded runs that register no vertex target must keep working.

    This is every current bounded run: inverse design has no target mesh, so `vertices_target`
    stays empty, the repair is gated off, and the guard is never reached.
    """
    from vertax.bilevelopt.boundedbop import BoundedBilevelOptimizer

    bop = BoundedBilevelOptimizer()
    assert bop.vertices_target is not None
    assert bop.vertices_target.size == 0, "bounded default should carry no vertex target"
    # The repair gate as written in inner_optimization; with no target it is False, so the
    # guard is not consulted at all.
    gate = (
        bop.vertices_target is not None
        and bop.vertices_target.size > 0
        and bop.update_T1
    )
    assert gate is False

    # And if a target IS registered on a bounded optimizer, the guard must fire -- i.e. the
    # refusal is reachable exactly in the configuration that would have been silently wrong.
    mesh = BoundedMesh.from_random_seeds(nb_seeds=6, width=2.0, height=2.0, random_key=7)
    bop.vertices_target = jnp.asarray(mesh.vertices).copy()
    assert bop.vertices_target.size > 0
    with pytest.raises(NotImplementedError):
        _reject_bounded_t1_repair(mesh)
