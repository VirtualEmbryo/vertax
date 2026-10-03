"""`get_area_bounded` must add the arc caps, never subtract them.

Two spellings of the final combination existed in the tree, with ``s`` the accumulated shoelace sum
and ``a`` the accumulated arc correction::

    old:  0.5 * abs(s + a)          new:  0.5 * (abs(s) + abs(a))

They agree only when ``sign(s) == sign(a)``, and they disagree on real meshes by up to 117x.

Why the new form is the correct one:

* ``0.5 * abs(s)`` is the chord-polygon area. The sign of ``s`` encodes the *traversal orientation*
  of the face, not anything physical: for the stored bounded meshes roughly 45% of faces are
  clockwise (verified by comparing ``s`` against an independent signed-area computation over the
  same vertex ring -- ``s`` is exactly twice it, and is negative exactly when the ring is
  clockwise). An area must not depend on traversal direction.
* ``0.5 * a`` is the total area of the circular segments between each boundary chord and its arc.
  It is non-negative by construction: ``a = sum(chordlen**2 * f(angle))`` with
  ``f(angle) = (angle - sin*cos) / (2 * sin**2)``, and ``f > 0`` on the whole range angles are
  clipped to (``topo.py``: ``clip(angTable, 0.017, pi/2 - 0.001)``, applied on every inner
  iteration). The implementation therefore cannot represent an inward-bulging arc at all.

So the cell area is (chord polygon) + (arc caps), i.e. the new form. The old form subtracts the
caps whenever the face happens to be traversed clockwise, returning ``abs(segments - polygon)``,
which is not an area and can even vanish.

This test is deliberately built to *discriminate*: it asserts the area is never below the
chord-polygon area, and it first asserts that the fixture actually contains a clockwise face, so
the discriminating case is exercised. Any orientation-blind check -- or any fixture with only
counter-clockwise faces -- passes both spellings, which is how the defect went unnoticed.
"""
from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from jax.experimental import enable_x64
import jax.numpy as jnp

from vertax.geo import _compute_contributions, get_area_bounded

FIXTURE = Path(__file__).parent / "reference_result_test_forward_modeling_bounded.npz"


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
    with np.load(FIXTURE, allow_pickle=True) as z:
        return (
            jnp.asarray(np.asarray(z["vertices"], float), jnp.float64),
            jnp.asarray(np.asarray(z["angles"], float), jnp.float64),
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
    """Faces whose half-edge ring stays inside the vertex table.

    At least one stored face (face 17 of this fixture) walks off the end of the vertex table; that
    is a separate data-integrity problem and is excluded here rather than silently masked.
    """
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


def test_fixture_contains_a_clockwise_face() -> None:
    """Guard: if every face were counter-clockwise this test suite could not discriminate."""
    vt, at, ht, ft = _load()
    cw = [f for f, start in _faces_with_valid_ring(vt, ht, ft)
          if _shoelace_and_arc(start, vt, at, ht)[0] < 0]
    assert cw, "fixture has no clockwise face; the area test below would not discriminate"


def test_arc_caps_are_added_not_subtracted() -> None:
    """Area must be chord-polygon + arc caps for EVERY face, including clockwise ones."""
    vt, at, ht, ft = _load()
    checked = discriminating = 0
    for f, start in _faces_with_valid_ring(vt, ht, ft):
        s, a = _shoelace_and_arc(start, vt, at, ht)
        polygon, caps = 0.5 * abs(s), 0.5 * abs(a)
        got = float(get_area_bounded(jnp.asarray(f), vt, at, ht, ft))
        assert got == pytest.approx(polygon + caps, rel=1e-9, abs=1e-12), (
            f"face {f}: area {got} != polygon {polygon} + caps {caps}; "
            f"0.5*abs(s+a) would give {0.5 * abs(s + a)} (s={s}, a={a})"
        )
        assert got >= polygon - 1e-12, f"face {f}: arc caps reduced the area ({got} < {polygon})"
        checked += 1
        if s < 0 < a:
            discriminating += 1
    assert checked > 0, "no face with a valid ring was checked"
    assert discriminating > 0, "no clockwise face with a non-zero arc term was exercised"
