"""`get_area_bounded` returns the signed chord polygon plus the arc caps.

With ``s`` the accumulated shoelace sum and ``a`` the accumulated arc correction of a face, the
area is ``0.5 * (s + a)``:

* faces are traversed counterclockwise -- `BoundedMesh.from_seeds` orders every face so, and T1
  transitions keep the half-edge structure consistently oriented -- so ``0.5 * s`` is the signed
  area of the chord polygon;
* ``0.5 * a`` is the area of the circular caps between each free-interface chord and its arc. It is
  non-negative by construction: ``a = sum(chordlen**2 * f(angle))`` with
  ``f(angle) = (angle - sin*cos) / (2 * sin**2) > 0`` on the range angles are clipped to
  (``topo.py``: 0.017 to pi/2 - 0.001). Every arc bulges to the right of its directed chord, away
  from the cell, so its cap always adds.

``s < 0`` therefore does not mean a clockwise face: it means that the cell's interior vertices have
crossed the chord of its arc. Such a cell is a crescent when ``s + a > 0`` (area ``cap - |polygon|``)
and a fold when ``s + a <= 0``. The spellings ``0.5 * (abs(s) + a)`` and ``0.5 * abs(s + a)`` both
differ from the area there: the first reads the cap on the polygon's side of the chord, the second
hides a fold.

The fixture is discriminating: face 3 of the stored relaxed mesh
(``fixtures/bounded_mesh_with_turned_over_cell.npz``) has turned over (it was counterclockwise at
construction, ``s`` = +0.17, and is ``s`` = -0.72 after the relaxation), so the test first asserts
that such a face exists.
"""
from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from jax.experimental import enable_x64
import jax.numpy as jnp

from vertax.geo import _compute_contributions, get_area_bounded

FIXTURE = Path(__file__).parent / "fixtures" / "bounded_mesh_with_turned_over_cell.npz"


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


def _load():
    """The fixture's tables in the layout `get_area_bounded` reads, as the bounded energy builds it.

    The vertex table gets the two reserved leading rows (half-edges store vertex ids offset by 2),
    and the arc half-angles, stored one per edge, are repeated for the two half-edges of each edge.
    """
    with np.load(FIXTURE, allow_pickle=True) as z:
        vertices = np.asarray(z["vertices"], float)
        return (
            jnp.asarray(np.vstack([[[0.0, 0.0], [1.0, 1.0]], vertices]), jnp.float64),
            jnp.asarray(np.repeat(np.asarray(z["angles"], float), 2), jnp.float64),
            jnp.asarray(np.asarray(z["edges"])),
            jnp.asarray(np.asarray(z["faces"])),
        )


def _shoelace_and_arc(face: int, vt, at, ht) -> tuple[float, float]:
    """Accumulate (shoelace, arc) exactly as get_area_bounded does, walking `next` (col 1)."""
    ht_np = np.asarray(ht)
    start = face
    res = np.asarray(_compute_contributions(jnp.asarray(start), vt, at, ht), dtype=float)
    cur = start
    for _ in range(11):
        cur = int(ht_np[cur, 1])
        if cur == start:
            break
        res = res + np.asarray(_compute_contributions(jnp.asarray(cur), vt, at, ht), dtype=float)
    return float(res[0]), float(res[1])


def _faces_with_valid_ring(vt, ht, ft):
    """Faces whose half-edge ring stays inside the (padded) vertex table: a guard on the fixture."""
    ht_np, ft_np, n_v = np.asarray(ht), np.asarray(ft), int(np.asarray(vt).shape[0])
    ok = []
    for f in range(ft_np.shape[0]):
        start = int(ft_np[f, 0])
        hes, cur = [start], start
        for _ in range(11):
            cur = int(ht_np[cur, 1])
            if cur == start:
                break
            hes.append(cur)
        idx = [int(ht_np[h, 3]) + int(ht_np[h, 5]) for h in hes] + [
            int(ht_np[h, 4]) + int(ht_np[h, 6]) - 1 for h in hes
        ]
        if all(0 <= i < n_v for i in idx):
            ok.append((f, start))
    return ok


def test_fixture_contains_a_turned_over_face() -> None:
    """Guard: if no chord polygon had turned over, the area test below could not discriminate."""
    vt, at, ht, ft = _load()
    turned = [f for f, start in _faces_with_valid_ring(vt, ht, ft)
              if _shoelace_and_arc(start, vt, at, ht)[0] < 0]
    assert turned, "fixture has no turned-over chord polygon; the area test below would not discriminate"


def test_area_is_signed_polygon_plus_caps() -> None:
    """Area must be signed chord polygon + arc caps for EVERY face, including turned-over ones."""
    vt, at, ht, ft = _load()
    checked = discriminating = 0
    for f, start in _faces_with_valid_ring(vt, ht, ft):
        s, a = _shoelace_and_arc(start, vt, at, ht)
        got = float(get_area_bounded(jnp.asarray(f), vt, at, ht, ft))
        assert got == pytest.approx(0.5 * (s + a), rel=1e-9, abs=1e-12), (
            f"face {f}: area {got} != 0.5 (s + a) = {0.5 * (s + a)}; "
            f"0.5 (|s| + a) would give {0.5 * (abs(s) + a)} (s={s}, a={a})"
        )
        assert got >= 0.5 * s - 1e-12, f"face {f}: arc caps reduced the area ({got} < {0.5 * s})"
        checked += 1
        if s < 0 < a:
            discriminating += 1
    assert checked == ft.shape[0], f"only {checked} of {ft.shape[0]} faces have a valid ring"
    assert discriminating > 0, "no turned-over face with a non-zero arc term was exercised"
