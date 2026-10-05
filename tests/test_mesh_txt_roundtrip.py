"""`save_mesh_txt` -> `load_mesh_txt` must preserve the physical parameter arrays.

WHY THIS EXISTS. `edges_params` and `faces_params` are PHYSICAL PARAMETERS, never indices, so they
must be read as float: reading them with `dtype=np.int64` silently truncates them -- an E2 tension
field of mean 1 and spread 0.1 comes back as all-ones, the entire inferred signal replaced by a
constant, with no error raised.

`save_mesh_txt` writes the INDEX tables through `np.savetxt`, i.e. in float notation
(`0.000000000000000000e+00`). Reading those back with `dtype=np.int64` makes numpy parse an integer
via a float, which raises a DeprecationWarning -- and this project sets `filterwarnings = ["error"]`,
so the loader instead reads as float and casts, which is also compatible with any file written this
way.

The fixture is a real committed mesh rather than a hand-built one, so the index tables are
well-formed and the test exercises the same path an external caller that reads geometry but never
touches `faces_params` would take.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vertax.meshes.pbc_mesh import PbcMesh

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mesh_1008.npz"

# mean 1, spread ~0.1 -- an E2 tension field. Integer truncation sends every entry to 1.
RNG = np.random.default_rng(20260823)


@pytest.fixture()
def mesh_with_known_params():
    """A real mesh carrying parameter fields that integer truncation would destroy."""
    m = PbcMesh.load_mesh(str(FIXTURE))
    tensions = 1.0 + 0.1 * RNG.standard_normal(m.nb_half_edges)
    shape_factors = 3.95 + 0.2 * RNG.standard_normal(m.nb_faces)
    vertex_params = 0.5 * RNG.standard_normal(m.nb_vertices)
    m.edges_params = np.asarray(tensions)
    m.faces_params = np.asarray(shape_factors)
    m.vertices_params = np.asarray(vertex_params)
    return m, tensions, shape_factors, vertex_params


def test_parameter_arrays_survive_a_text_roundtrip(mesh_with_known_params, tmp_path) -> None:
    """The regression itself: float parameters must not be truncated to integers."""
    m, tensions, shape_factors, vertex_params = mesh_with_known_params
    m.save_mesh_txt(str(tmp_path))
    back = PbcMesh.load_mesh_txt(str(tmp_path))
    # the mesh classes hold float32, so compare at single precision rather than to the bit
    np.testing.assert_allclose(np.asarray(back.edges_params), tensions, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(back.faces_params), shape_factors, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(back.vertices_params), vertex_params, rtol=1e-6)


def test_the_roundtrip_preserves_variation_not_merely_magnitude(mesh_with_known_params,
                                                                tmp_path) -> None:
    """Guard against the specific failure, not merely against a loose tolerance.

    Integer truncation of a mean-1 spread-0.1 field yields a CONSTANT array whose entries are each
    within 1.0 of the truth, so a test with a generous absolute tolerance would pass on the broken
    code. Asserting that the spread and the number of distinct values survive is something
    truncation cannot fake.
    """
    m, tensions, _, _ = mesh_with_known_params
    m.save_mesh_txt(str(tmp_path))
    got = np.asarray(PbcMesh.load_mesh_txt(str(tmp_path)).edges_params)
    assert got.std() == pytest.approx(tensions.std(), rel=1e-5), (
        "the tension field lost its variation -- the integer-truncation signature")
    assert np.unique(got).size == tensions.size, "distinct tensions collapsed onto shared values"


def test_geometry_and_connectivity_survive_the_roundtrip(mesh_with_known_params, tmp_path) -> None:
    """The index tables must come back as integers, and unchanged.

    This is the half of the loader that the float-notation write nearly broke: the tables are
    written as floats and must be read back to exactly the same integers.
    """
    m, *_ = mesh_with_known_params
    m.save_mesh_txt(str(tmp_path))
    back = PbcMesh.load_mesh_txt(str(tmp_path))
    np.testing.assert_array_equal(np.asarray(back.edges), np.asarray(m.edges))
    np.testing.assert_array_equal(np.asarray(back.faces), np.asarray(m.faces))
    np.testing.assert_allclose(np.asarray(back.vertices), np.asarray(m.vertices), rtol=1e-6)
    assert np.asarray(back.edges).dtype.kind == "i", "connectivity must load as an integer type"
    assert back.width == pytest.approx(float(m.width))
    assert back.height == pytest.approx(float(m.height))
