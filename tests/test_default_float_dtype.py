"""Meshes and relaxations follow JAX's default float dtype.

Without ``jax_enable_x64`` the default float dtype is ``float32`` and every number must be
bit-identical to what version 0.2.0 produced, because 0.2.0 hard-coded ``float32`` in the mesh
constructors. With ``jax_enable_x64`` the default is ``float64`` and the vertices, the arc angles and
everything the relaxation returns must stay ``float64``.

The x64 flag is read once per process, so the x64 cases run in a subprocess started with
``JAX_ENABLE_X64=1``. The reference arrays were produced by the unmodified 0.2.0 code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from vertax import BoundedBilevelOptimizer, BoundedMesh, PbcBilevelOptimizer, PbcMesh
from vertax.energy import energy_shape_factor_homo, make_energy_line_tensions_bounded

TESTS_DIR = Path(__file__).resolve().parent
REFERENCE = TESTS_DIR / "reference_result_test_default_dtype.npz"

N_CELLS_PERIODIC = 20
N_CELLS_BOUNDED = 20
N_EDGES_BOUNDED = (N_CELLS_BOUNDED - 1) * 3
RELAXATION_ITERATIONS = 30
BOUNDED_ENERGY = make_energy_line_tensions_bounded(40.0, sigmoid_tensions=True)


def build_periodic_mesh() -> PbcMesh:
    """A periodic mesh of 20 cells in a square box of area 20, built from fixed seeds."""
    side = float(np.sqrt(N_CELLS_PERIODIC))
    mesh = PbcMesh.from_random_seeds(nb_seeds=N_CELLS_PERIODIC, width=side, height=side, random_key=1)
    mesh.vertices_params = jnp.asarray([0.0])
    mesh.edges_params = jnp.asarray([0.0])
    mesh.faces_params = jnp.asarray([3.7])
    return mesh


def relax_periodic_mesh(mesh: PbcMesh) -> None:
    """One short relaxation of a periodic mesh with plain gradient descent."""
    width, height = float(mesh.width), float(mesh.height)
    optimizer = PbcBilevelOptimizer()
    optimizer.inner_solver = optax.sgd(learning_rate=0.01)
    optimizer.min_dist_T1 = 0.2
    optimizer.max_nb_iterations = RELAXATION_ITERATIONS
    optimizer.tolerance = 1e-12
    optimizer.patience = RELAXATION_ITERATIONS
    optimizer.loss_function_inner = lambda v, h, f, _vp, _hp, fp: energy_shape_factor_homo(v, h, f, width, height, fp)
    optimizer.inner_optimization(mesh=mesh)


def build_bounded_mesh() -> BoundedMesh:
    """A bounded mesh of 20 cells built from fixed seeds, with random edge tensions."""
    side = float(np.sqrt(N_CELLS_BOUNDED))
    mesh = BoundedMesh.from_random_seeds(nb_seeds=N_CELLS_BOUNDED, width=side, height=side, random_key=1)
    rng = np.random.default_rng(seed=1)
    mesh.vertices_params = jnp.asarray([0.0])
    mesh.edges_params = jnp.asarray(rng.random(N_EDGES_BOUNDED) * 20 - 10)
    mesh.faces_params = jnp.asarray([0.6])
    return mesh


def relax_bounded_mesh(mesh: BoundedMesh) -> None:
    """One short relaxation of a bounded mesh with plain gradient descent."""
    optimizer = BoundedBilevelOptimizer()
    optimizer.min_dist_T1 = 0.025
    optimizer.inner_solver = optax.sgd(learning_rate=0.01)
    optimizer.max_nb_iterations = RELAXATION_ITERATIONS
    optimizer.tolerance = 1e-12
    optimizer.patience = RELAXATION_ITERATIONS
    optimizer.loss_function_inner = BOUNDED_ENERGY
    optimizer.inner_optimization(mesh=mesh)


def mask_from_voronoi_cells() -> np.ndarray:
    """A label image of 20 Voronoi cells on a 100 x 100 grid, the input of ``PbcMesh.from_mask``.

    Only the dtype of the mesh built from it is checked: ``from_mask`` does not produce a valid
    mesh from every label image, and this one is a seed for which it builds one.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed=0)
    seeds = rng.random((20, 2)) * 100
    yy, xx = np.mgrid[0:100, 0:100]
    _, label = cKDTree(seeds).query(np.stack([xx.ravel(), yy.ravel()], axis=1))
    return (label.reshape(100, 100) + 1).astype(np.int32)


def collect_arrays() -> dict[str, np.ndarray]:
    """Constructor outputs and relaxed vertices of both modes, as numpy arrays."""
    out: dict[str, np.ndarray] = {}
    periodic = build_periodic_mesh()
    out["periodic_vertices_initial"] = np.asarray(periodic.vertices)
    out["periodic_edges"] = np.asarray(periodic.edges)
    out["periodic_faces"] = np.asarray(periodic.faces)
    relax_periodic_mesh(periodic)
    out["periodic_vertices_relaxed"] = np.asarray(periodic.vertices)

    bounded = build_bounded_mesh()
    out["bounded_vertices_initial"] = np.asarray(bounded.vertices)
    out["bounded_angles_initial"] = np.asarray(bounded.angles)
    out["bounded_edges"] = np.asarray(bounded.edges)
    out["bounded_faces"] = np.asarray(bounded.faces)
    relax_bounded_mesh(bounded)
    out["bounded_vertices_relaxed"] = np.asarray(bounded.vertices)
    out["bounded_angles_relaxed"] = np.asarray(bounded.angles)
    return out


def probe_dtypes() -> dict[str, object]:
    """Dtypes (and whether float32 would have lost digits) of every float array a user receives."""
    report: dict[str, object] = {"default_float": str(jnp.zeros(1).dtype)}

    periodic = build_periodic_mesh()
    report["periodic_vertices_initial"] = str(periodic.vertices.dtype)
    report["periodic_initial_beyond_float32"] = bool(
        np.any(np.asarray(periodic.vertices) != np.asarray(periodic.vertices).astype(np.float32))
    )
    relax_periodic_mesh(periodic)
    report["periodic_vertices_relaxed"] = str(periodic.vertices.dtype)

    from_seeds = PbcMesh.from_seeds(jnp.asarray(np.random.default_rng(2).random((15, 2)) * 4), 4.0, 4.0)
    report["periodic_from_seeds_vertices"] = str(from_seeds.vertices.dtype)

    from_mask = PbcMesh.from_mask(mask_from_voronoi_cells())
    report["periodic_from_mask_vertices"] = str(from_mask.vertices.dtype)

    bounded = build_bounded_mesh()
    report["bounded_vertices_initial"] = str(bounded.vertices.dtype)
    report["bounded_angles_initial"] = str(bounded.angles.dtype)
    report["bounded_initial_beyond_float32"] = bool(
        np.any(np.asarray(bounded.vertices) != np.asarray(bounded.vertices).astype(np.float32))
    )
    relax_bounded_mesh(bounded)
    report["bounded_vertices_relaxed"] = str(bounded.vertices.dtype)
    report["bounded_angles_relaxed"] = str(bounded.angles.dtype)
    return report


def test_default_precision_is_bit_identical_to_version_0_2_0() -> None:
    """Without x64 the constructors and one relaxation of each mode reproduce 0.2.0 exactly."""
    assert not jax.config.jax_enable_x64, "this test checks the default (float32) configuration"
    reference = np.load(REFERENCE)
    now = collect_arrays()
    assert set(now) == set(reference.files)
    for name, array in now.items():
        assert array.dtype == reference[name].dtype, name
        np.testing.assert_array_equal(array, reference[name], err_msg=name)


def test_default_precision_is_float32() -> None:
    """Without x64 every mesh float array is float32, from every constructor."""
    assert not jax.config.jax_enable_x64
    report = probe_dtypes()
    floats = {k: v for k, v in report.items() if isinstance(v, str)}
    assert set(floats.values()) == {"float32"}, floats


@pytest.fixture(scope="module")
def x64_report() -> dict[str, object]:
    """The dtype report of ``probe_dtypes`` computed in a fresh process with ``JAX_ENABLE_X64=1``."""
    package_root = Path(sys.modules["vertax"].__file__).resolve().parent.parent
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(package_root), str(TESTS_DIR), env.get("PYTHONPATH", "")])
    code = "import json, test_default_float_dtype as t; print(json.dumps(t.probe_dtypes()))"
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_x64_default_float_is_float64(x64_report: dict[str, object]) -> None:
    """The subprocess really ran with x64 enabled."""
    assert x64_report["default_float"] == "float64"


def test_x64_periodic_mesh_is_float64_before_and_after_relaxation(x64_report: dict[str, object]) -> None:
    """Periodic vertices are float64 from every constructor and stay float64 through a relaxation."""
    assert x64_report["periodic_vertices_initial"] == "float64"
    assert x64_report["periodic_from_seeds_vertices"] == "float64"
    assert x64_report["periodic_from_mask_vertices"] == "float64"
    assert x64_report["periodic_vertices_relaxed"] == "float64"
    assert x64_report["periodic_initial_beyond_float32"], "float64 vertices carry digits float32 cannot hold"


def test_x64_bounded_mesh_is_float64_before_and_after_relaxation(x64_report: dict[str, object]) -> None:
    """Bounded vertices and arc angles are float64 and stay float64 through a relaxation."""
    assert x64_report["bounded_vertices_initial"] == "float64"
    assert x64_report["bounded_angles_initial"] == "float64"
    assert x64_report["bounded_vertices_relaxed"] == "float64"
    assert x64_report["bounded_angles_relaxed"] == "float64"
    assert x64_report["bounded_initial_beyond_float32"], "float64 vertices carry digits float32 cannot hold"
