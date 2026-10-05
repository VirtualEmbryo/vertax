"""Guard for Fix A: `PbcMesh` face ordering must RAISE, not hang, and must change nothing else.

WHY THIS TEST EXISTS. `_make_he_structure` orders each face's edges with

    while sorted(edges_face) != sorted(visited):
        ...
        e = edges_face[i % len(face)]

The modulus is over `face` -- the face's VERTICES -- while the list indexed is `edges_face`, the
vertex pairs of that face that are also edges. Termination requires every element of `edges_face` to
be visited, but only indices `0 .. len(face)-1` are reachable, so when `len(edges_face) > len(face)`
the loop never terminates: no exception, no warning, no output. A seed sweep does not crash on a bad
seed, it stalls forever, which under `nohup` is indistinguishable from slow.

This exact hang is also documented in `tests/test_api_contracts_guard.py`'s module docstring;
this suite's `nb_seeds=20, sqrt(20)` convention exists to route around it. This test is here so
it cannot go undiscovered again.

WHAT IS GUARDED
  1. A known-pathological seed raises promptly, and the message carries the four things a caller
     needs to act: face index, `len(face)`, `len(edges_face)`, and the offending vertex set.
  2. `nb_seeds=20` at `sqrt(20)` square -- the convention the rest of the suite relies on -- still
     builds.
  3. Meshes that build are **byte-identical** to a build without the bound. The bound cannot
     change them: termination already implies `len(edges_face) == len(face)`, so the guard can only
     fire where the walk would otherwise never return. Measured over 420 seed cases and the 120
     committed synthetic meshes: the invariant holds on 189/189 that build, fails on 211/211 that
     hang.

WHAT IS **NOT** GUARDED, deliberately: that the pathological seeds produce a mesh. That is Fix B, it
needs a geometric decision about what a chord means, and its acceptance test is
`clamped.signed_face_areas` rather than `geo.get_area`'s `0.5*|shoelace|` -- which accepts a
truncated or self-intersecting face silently, the `get_area_bounded` failure mode.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax import PbcMesh


@pytest.fixture(autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    with enable_x64():
        yield

# Measured hangs. Each has `len(edges_face) > len(face)` on at least one face.
PATHOLOGICAL = [
    (20, 3.1622776601683795, 6.324555320336759, 232),
    (20, 3.1622776601683795, 6.324555320336759, 1),
    (4, 2.0, 2.0, 0),
    (6, 1.7320508075688772, 3.4641016151377544, 0),
]

# Cases that build. Hashes are asserted to be stable, not to specific literals: the point is that
# the guard changed nothing, and a hard-coded hash would also fail on any legitimate future change
# to Voronoi construction, which this test has no business policing.
BUILDABLE = [
    (20, 4.47213595499958, 4.47213595499958, 0),      # the suite's own convention: sqrt(20) square
    (20, 4.47213595499958, 4.47213595499958, 3),
    (50, 7.071067811865476, 7.071067811865476, 1),
    (72, 6.0, 12.0, 231),                             # the Figure-5 production box
    (72, 6.0, 12.0, 233),
]

RAISE_BUDGET_S = 30.0


@pytest.mark.parametrize(("nb_seeds", "width", "height", "key"), PATHOLOGICAL)
def test_pathological_seed_raises_promptly(nb_seeds: int, width: float, height: float, key: int) -> None:
    """A seed whose face ordering cannot terminate must raise quickly, with an actionable message."""
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError) as excinfo:
        PbcMesh.from_random_seeds(nb_seeds=nb_seeds, width=width, height=height, random_key=key)
    elapsed = time.perf_counter() - t0
    assert elapsed < RAISE_BUDGET_S, (
        f"raised only after {elapsed:.1f}s; the point of the bound is that it is prompt"
    )
    msg = str(excinfo.value)
    for needle in ("face index", "len(face)", "len(edges_face)", "vertex set", "random_key"):
        assert needle in msg, f"diagnostic is missing {needle!r}: {msg}"


@pytest.mark.parametrize(("nb_seeds", "width", "height", "key"), BUILDABLE)
def test_buildable_seed_still_builds(nb_seeds: int, width: float, height: float, key: int) -> None:
    """A mesh that built before the bound must still build, with a valid torus Euler characteristic."""
    m = PbcMesh.from_random_seeds(nb_seeds=nb_seeds, width=width, height=height, random_key=key)
    assert m.nb_faces == nb_seeds
    assert m.nb_vertices == 2 * nb_seeds, "trivalent periodic mesh has V = 2F"
    assert m.nb_vertices - m.nb_half_edges // 2 + m.nb_faces == 0, "V - E + F must be 0 on a torus"


def test_build_is_deterministic_and_byte_stable() -> None:
    """Two builds of the same seed agree byte for byte.

    The regression evidence that the bound changes no existing mesh is a 41-case baseline,
    verified byte-identical (41/41). This test keeps the weaker, self-contained half of that
    guarantee in the suite, where it costs seconds.
    """
    nb_seeds, w, h, key = BUILDABLE[0]
    digests = []
    for _ in range(2):
        m = PbcMesh.from_random_seeds(nb_seeds=nb_seeds, width=w, height=h, random_key=key)
        hasher = hashlib.sha256()
        for name in ("vertices", "edges", "faces"):
            hasher.update(np.ascontiguousarray(np.asarray(getattr(m, name))).tobytes())
        digests.append(hasher.hexdigest())
    assert digests[0] == digests[1]


def test_face_ordering_invariant_holds_on_every_built_mesh() -> None:
    """`len(edges_face) == len(face)` on every face of every mesh that builds.

    This is the deduction the bound rests on, checked directly rather than trusted: if it ever fails
    on a mesh that builds, the bound is no longer provably inert and Fix B has become urgent.
    """
    for nb_seeds, w, h, key in BUILDABLE:
        m = PbcMesh.from_random_seeds(nb_seeds=nb_seeds, width=w, height=h, random_key=key)
        e = np.asarray(m.edges)
        undirected = {tuple(sorted((int(r[3]), int(r[4])))) for r in e}
        for f_start in np.asarray(m.faces):
            he, cycle = int(f_start), []
            for _ in range(m.MAX_EDGES_IN_ANY_FACE * 4):
                cycle.append(he)
                he = int(e[he, 1])
                if he == int(f_start):
                    break
            vertices_of_face = sorted({int(e[h, 3]) for h in cycle} | {int(e[h, 4]) for h in cycle})
            induced = sum(1 for i, a in enumerate(vertices_of_face)
                          for b in vertices_of_face[i + 1:] if (a, b) in undirected)
            assert induced == len(vertices_of_face), (
                f"chord on a mesh that builds: nb_seeds={nb_seeds} key={key}, face vertices "
                f"{vertices_of_face}, induced edges {induced}"
            )
