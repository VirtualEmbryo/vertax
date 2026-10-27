"""Loading a bounded mesh refuses clockwise faces and warns about turned-over faces.

Cell areas read every face of a bounded mesh as counterclockwise. A mesh built outside VertAX with
clockwise faces would be read inside out, with negative areas and no error, so every constructor
and loader checks the orientation.

The sign of one face's chord polygon cannot separate the two cases: it is negative for a clockwise
face and also for a crescent, a legitimate cell whose interior vertices have crossed the chord of
its free arc. The sum of the signed chord polygons of all faces can, because every interior edge is
run once in each sense and the sum is the signed area of the polygon through the outer chords. It
depends on the surface vertices only, so a crescent leaves it positive, and it changes sign when
every face is reversed.

A clockwise mesh is made here by reflecting the vertices of a mesh VertAX built (x -> width - x) and
keeping its tables, which reverses the sense of every face. A crescent mesh is the two-cell cluster
of `test_bounded_crescent_cell.py`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from test_bounded_crescent_cell import _points, _tables
from vertax import BoundedMesh
from vertax.meshes.bounded_mesh import _signed_chord_polygons

TESTS = Path(__file__).parent
FIXTURE_WITH_FOLD = TESTS / "fixtures" / "bounded_mesh_with_turned_over_cell.npz"
STORED_BOUNDED_MESHES = sorted(TESTS.glob("reference_result_test_*_bounded.npz"))


def _clockwise(mesh: BoundedMesh) -> BoundedMesh:
    """The mirror image of a mesh, with the same tables: every face is now clockwise."""
    mirrored = BoundedMesh.copy_mesh(mesh)
    mirrored.vertices = jnp.stack([mesh.width - mesh.vertices[:, 0], mesh.vertices[:, 1]], axis=1)
    return mirrored


def _crescent_cluster(mirror: bool = False) -> BoundedMesh:
    """The two-cell cluster whose first cell is a crescent (chord polygon turned over)."""
    he, angles, faces = _tables(0, 6)
    mesh = BoundedMesh.create_empty()
    mesh.vertices = jnp.asarray(_points(mirror, shift=(2.0, 3.0)), dtype=jnp.float32)
    mesh.edges = jnp.asarray(he)
    mesh.faces = jnp.asarray(faces)
    mesh.angles = jnp.asarray(angles, dtype=jnp.float32)
    mesh.width, mesh.height = 6.0, 6.0
    mesh.vertices_params = jnp.asarray([0.0])
    mesh.edges_params = jnp.ones(5)
    mesh.faces_params = jnp.asarray([0.6])
    return mesh


def _round_trip(mesh: BoundedMesh, tmp_path: Path, kind: str) -> BoundedMesh:
    # a built mesh carries no parameters, and the text loader refuses an empty parameter file
    mesh.vertices_params = jnp.asarray([0.0])
    mesh.edges_params = jnp.ones(mesh.nb_edges)
    mesh.faces_params = jnp.asarray([0.6])
    if kind == "npz":
        mesh.save_mesh(str(tmp_path / "mesh.npz"))
        return BoundedMesh.load_mesh(str(tmp_path / "mesh.npz"))
    mesh.save_mesh_txt(str(tmp_path / "txt"))
    return BoundedMesh.load_mesh_txt(str(tmp_path / "txt"))


@pytest.mark.parametrize("kind", ["npz", "txt"])
@pytest.mark.parametrize(("n_cells", "seed"), [(3, 1), (6, 2), (12, 3), (20, 1), (36, 1)])
def test_built_and_reloaded_meshes_pass_without_warning(n_cells, seed, tmp_path, kind) -> None:
    side = n_cells**0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the constructor and the loader neither raise nor warn
        mesh = BoundedMesh.from_random_seeds(n_cells, side, side, seed)
        assert (_signed_chord_polygons(mesh) > 0).all()
        reloaded = _round_trip(mesh, tmp_path, kind)
    np.testing.assert_allclose(np.asarray(reloaded.vertices), np.asarray(mesh.vertices), rtol=1e-6)


@pytest.mark.parametrize("kind", ["npz", "txt"])
@pytest.mark.parametrize(("n_cells", "seed"), [(6, 2), (20, 1), (36, 1)])
def test_clockwise_mesh_is_refused(n_cells, seed, tmp_path, kind) -> None:
    side = n_cells**0.5
    mirrored = _clockwise(BoundedMesh.from_random_seeds(n_cells, side, side, seed))
    assert (_signed_chord_polygons(mirrored) < 0).all()  # the fixture is clockwise
    # Without the check the mesh is read inside out: negative areas, no error.
    assert float(mirrored.get_area(jnp.arange(mirrored.nb_faces)).min()) < 0
    with pytest.raises(ValueError, match="clockwise"):
        _round_trip(mirrored, tmp_path, kind)


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("kind", ["npz", "txt"])
def test_crescent_loads_with_a_warning_and_its_mirror_image_is_refused(mirror, tmp_path, kind) -> None:
    mesh = _crescent_cluster(mirror)
    polygons = _signed_chord_polygons(mesh)
    assert polygons[0] < 0 < polygons.sum()  # a turned-over cell in a mesh whose outer polygon is positive
    with pytest.warns(UserWarning, match=r"1 face\(s\) have a turned-over chord polygon \(face 0\)"):
        reloaded = _round_trip(mesh, tmp_path, kind)
    area = float(reloaded.get_area(jnp.arange(2))[0])
    assert area > 0  # a crescent is a sound cell: cap - |polygon| > 0

    # The same cluster read the other way round (every face reversed) has a negative outer polygon.
    other = _clockwise(_crescent_cluster(mirror))
    with pytest.raises(ValueError, match="clockwise"):
        _round_trip(other, tmp_path, kind)


def test_folded_cell_of_a_stored_mesh_warns() -> None:
    with pytest.warns(UserWarning, match=r"1 face\(s\) have a turned-over chord polygon \(face 3\)"):
        mesh = BoundedMesh.load_mesh(str(FIXTURE_WITH_FOLD))
    assert float(mesh.get_area(jnp.arange(mesh.nb_faces))[3]) < 0  # it is a fold, not a crescent


@pytest.mark.parametrize("path", STORED_BOUNDED_MESHES, ids=lambda p: p.name)
def test_stored_regression_meshes_pass_without_warning(path) -> None:
    assert STORED_BOUNDED_MESHES, "no stored bounded mesh found"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = BoundedMesh.load_mesh(str(path))
    assert (_signed_chord_polygons(mesh) > 0).all()
