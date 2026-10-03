"""V0 — invariance and wrap safety of the bulk-driven shear rheometer.

These are the checks that gate everything else in the Figure-5 rheometry path. They are
**configuration-independent identities**, so they need no inner solve and run in seconds: the
drive's correctness does not depend on the mesh being relaxed.

WHY EACH ONE EXISTS. The drive is a potential linear in the vertex coordinates, and the periodic
solver re-wraps coordinates at every inner iteration (`geo.update_pbc`). That combination has exactly
one real hazard: a term linear in the *raw* coordinate jumps by `a*c_i*W` when a vertex crosses the
seam, which would inject spurious work into every solve and surface as a plausible-looking modulus
rather than as an error. Everything below exists to make that impossible to miss.

Meshes use the suite's `nb_seeds=20`, `sqrt(20)` square convention. That is not arbitrary: at 20
cells in a 1:2 box, 4 of 12 random keys hit the `PbcMesh` face-ordering defect, while the square
box at 20 cells hit 0 of 12.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.energy import energy_shape_factor_homo
from vertax.geo import update_pbc
from vertax.rheology import (
    compliance,
    drive_drift,
    hessian_lambda_max,
    make_drive_energy,
    make_shear_mode,
    min_image_delta_x,
    mode_amplitude,
    modulus_from_compliance,
    require_float64,
    target_compliance,
)

N_CELLS = 20
SIDE = math.sqrt(20.0)
SEED = 231
P0 = 3.60
A0 = 0.02
MACHINE = 1e-12


@pytest.fixture(scope="module", autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    """Scoped to this module, matching `mesh`/`energy`'s scope so x64 is on before they build."""
    with enable_x64():
        yield


@pytest.fixture(scope="module")
def mesh():
    """A periodic mesh in float64. `PbcMesh.from_seeds` builds float32; the cast is not cosmetic."""
    m = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=SIDE, height=SIDE, random_key=SEED)
    m.vertices = jnp.asarray(m.vertices, dtype=jnp.float64)
    m.vertices_params = jnp.zeros(m.nb_vertices, dtype=jnp.float64)
    m.edges_params = jnp.zeros(m.nb_half_edges, dtype=jnp.float64)
    m.faces_params = jnp.full(m.nb_faces, P0, dtype=jnp.float64)
    return m


@pytest.fixture(scope="module")
def energy():
    """E1 in the six-argument inner-loss signature; width/height are closed over (static_argnums)."""

    def e(v, h, f, _vp, _hp, fp):  # noqa: ANN001, ANN202
        return energy_shape_factor_homo(v, h, f, SIDE, SIDE, jnp.atleast_1d(jnp.asarray(fp)))

    return e


@pytest.fixture(scope="module")
def mode(mesh):  # noqa: ANN001
    """A transverse n=1 mode referenced on the mesh as given."""
    return make_shear_mode(mesh.vertices, SIDE, SIDE, n=1)


def test_require_float64_passes_under_x64() -> None:
    """The guard must accept the configuration these tests run in."""
    require_float64()


def test_v0a_mean_removal(mode) -> None:  # noqa: ANN001
    """Without mean removal the drive carries a net force and the translation mode is unbounded."""
    assert abs(float(jnp.sum(mode.c))) < MACHINE


def test_v0a_mean_removal_is_enforced(mesh) -> None:  # noqa: ANN001
    """`make_shear_mode` must refuse a mode whose weights do not sum to zero."""
    with pytest.raises(ValueError, match="net force"):
        make_shear_mode(mesh.vertices, SIDE, SIDE, n=1, mean_tol=0.0)


def test_v0b_wrap_identity_and_half_open_interval() -> None:
    """The wrap is invariant under a lattice translation at EVERY point, including the tie.

    The obvious spelling `d - L*round(d/L)` is not: `round` is half-to-even, so it maps both `+-L/2`
    to themselves, the interval comes out closed, and the two representatives at exactly `d = L/2`
    differ by one lattice vector. That failed V0b at `|delta| = L` and is why the implementation uses
    `floor(d/L + 1/2)`.
    """
    xs = jnp.array([0.1, 1.0, SIDE - 0.1, SIDE / 2.0, -SIDE / 2.0, 0.0, 0.37 * SIDE])
    zeros = jnp.zeros_like(xs)
    base = min_image_delta_x(xs, zeros, SIDE)
    shifted = min_image_delta_x(xs + SIDE, zeros, SIDE)
    assert float(jnp.max(jnp.abs(base - shifted))) < 1e-14
    assert bool(jnp.all(base >= -SIDE / 2 - 1e-15))
    assert bool(jnp.all(base < SIDE / 2))


def test_v0c_wrap_derivative_is_exactly_one() -> None:
    """`floor` carries a zero JVP, so the wrap's derivative is exactly 1 and no stop_gradient is
    needed anywhere in the module."""
    g = jax.grad(lambda x: min_image_delta_x(x, jnp.array(0.0), SIDE))
    for probe in (0.1, SIDE / 2 - 1e-9, SIDE / 2 + 1e-9, SIDE - 1e-9, -SIDE / 2 + 1e-9):
        assert float(g(jnp.array(probe))) == 1.0


def test_v0d_lattice_translation_invariance(mesh, energy, mode) -> None:  # noqa: ANN001
    """Translating the whole mesh by a lattice vector changes no observable."""
    drive = make_drive_energy(energy, mode, A0)
    e0 = float(drive(mesh.vertices, mesh.edges, mesh.faces, None, None, mesh.faces_params))
    u0 = float(mode_amplitude(mesh.vertices, mode))

    shifted = mesh.vertices + jnp.array([SIDE, SIDE])
    vt, ht, ft = update_pbc(shifted, mesh.edges, mesh.faces, SIDE, SIDE)
    mode_t = make_shear_mode(vt, SIDE, SIDE, n=1)
    drive_t = make_drive_energy(energy, mode_t, A0)
    e1 = float(drive_t(vt, ht, ft, None, None, mesh.faces_params))
    u1 = float(mode_amplitude(vt, mode_t))

    assert abs(u1 - u0) < MACHINE
    assert abs(e1 - e0) < 1e-9 * max(1.0, abs(e0))
    assert float(jnp.max(jnp.abs(jnp.sort(mode_t.c) - jnp.sort(mode.c)))) < 1e-13


def test_v0e_seam_crossing_changes_nothing(mesh, energy, mode) -> None:  # noqa: ANN001
    """Two representations of the same physical configuration must agree exactly.

    This is the hazard the minimum image exists for: `update_pbc` re-wraps a vertex that leaves the
    box, and a drive written on the raw coordinate would jump by `a*c_i*W`.
    """
    drive = make_drive_energy(energy, mode, A0)
    i = int(jnp.argmin(mesh.vertices[:, 0]))
    pushed = mesh.vertices.at[i, 0].add(-2.0 * float(mesh.vertices[i, 0]) - 1e-3)
    wrapped, hw, fw = update_pbc(pushed, mesh.edges, mesh.faces, SIDE, SIDE)
    assert abs(float(wrapped[i, 0]) - float(pushed[i, 0])) > SIDE / 2, "the vertex must have wrapped"

    u_pre = float(mode_amplitude(pushed, mode))
    u_post = float(mode_amplitude(wrapped, mode))
    e_pre = float(drive(pushed, mesh.edges, mesh.faces, None, None, mesh.faces_params))
    e_post = float(drive(wrapped, hw, fw, None, None, mesh.faces_params))
    g_pre = jax.grad(lambda x: drive(x, mesh.edges, mesh.faces, None, None, mesh.faces_params))(pushed)
    g_post = jax.grad(lambda x: drive(x, hw, fw, None, None, mesh.faces_params))(wrapped)

    assert abs(u_post - u_pre) < MACHINE
    assert abs(e_post - e_pre) < 1e-9 * max(1.0, abs(e_pre))
    assert float(jnp.max(jnp.abs(g_pre - g_post))) < 1e-9


def test_v0f_compliance_is_independent_of_the_reference(mesh, mode) -> None:  # noqa: ANN001
    """`J` depends on the weights, never on the reference positions.

    `dx(+) - dx(-) = x(+) - x(-)` exactly, so shifting the reference cannot move `J`. Checked with
    synthetic driven states, so the identity is tested rather than a solver.
    """
    disp = 1e-3 * jnp.asarray(mode.c)[:, None] * jnp.array([1.0, 0.0])
    v_plus, v_minus = mesh.vertices + disp, mesh.vertices - disp
    j_ref = float(compliance(mode_amplitude(v_plus, mode), mode_amplitude(v_minus, mode), A0))

    off = 0.37 * SIDE
    mode_off = make_shear_mode(mesh.vertices.at[:, 0].add(off), SIDE, SIDE, n=1)
    j_off = float(compliance(mode_amplitude(v_plus.at[:, 0].add(off), mode_off),
                             mode_amplitude(v_minus.at[:, 0].add(off), mode_off), A0))
    assert abs(j_ref - j_off) < 1e-10 * max(1.0, abs(j_ref))


def test_v0g_drive_does_not_enter_the_hessian(mesh, energy, mode) -> None:  # noqa: ANN001
    """The drive is linear in X, so `a0` cannot move the SGD stability bound `eta*lambda_max < 2`."""
    bare = hessian_lambda_max(energy, mesh.vertices, mesh.edges, mesh.faces,
                              None, None, mesh.faces_params)
    driven = hessian_lambda_max(make_drive_energy(energy, mode, A0), mesh.vertices, mesh.edges,
                                mesh.faces, None, None, mesh.faces_params)
    assert abs(bare - driven) < 1e-12 * abs(bare)


def test_v0h_drive_force_is_exact(mesh, energy, mode) -> None:  # noqa: ANN001
    """The force is exactly `a*c_i` along the polarisation axis and zero on the other, with zero net."""
    drive = make_drive_energy(energy, mode, A0)
    g_bare = jax.grad(lambda x: energy(x, mesh.edges, mesh.faces, None, None, mesh.faces_params))(
        mesh.vertices)
    g_tot = jax.grad(lambda x: drive(x, mesh.edges, mesh.faces, None, None, mesh.faces_params))(
        mesh.vertices)
    force = g_bare - g_tot
    want = jnp.zeros_like(force).at[:, 0].set(A0 * mode.c)
    assert float(jnp.max(jnp.abs(force - want))) < 1e-14
    assert abs(float(jnp.sum(force[:, 0]))) < 1e-13
    assert abs(float(jnp.sum(force[:, 1]))) < 1e-15


def test_v0i_uncentred_weights_would_carry_a_net_force(mesh, mode) -> None:  # noqa: ANN001
    """Negative control: mean removal is load-bearing, not bookkeeping."""
    raw = jnp.cos(mode.q * mesh.vertices[:, 1])
    net_raw = abs(float(A0 * jnp.sum(raw)))
    net_centred = abs(float(A0 * jnp.sum(mode.c)))
    assert net_raw > 1e4 * max(net_centred, 1e-18)


def test_mode_rejects_bad_arguments(mesh) -> None:  # noqa: ANN001
    """`n = 0` is a rigid translation, and the axes must be 0 or 1."""
    with pytest.raises(ValueError, match="rigid translation"):
        make_shear_mode(mesh.vertices, SIDE, SIDE, n=0)
    with pytest.raises(ValueError, match="mod_axis and pol_axis"):
        make_shear_mode(mesh.vertices, SIDE, SIDE, n=1, mod_axis=2)


def test_modulus_and_target_are_inverse(mode) -> None:  # noqa: ANN001
    """`target_compliance` must invert `modulus_from_compliance`, and refuse a zero target."""
    g = 0.5
    j = target_compliance(g, mode)
    assert abs(float(modulus_from_compliance(j, mode)) / g - 1.0) < 1e-12
    with pytest.raises(ValueError, match="degenerate fluid-side case"):
        target_compliance(0.0, mode)


def test_drive_energy_rejects_a_sliced_vertex_table(mesh, energy, mode) -> None:  # noqa: ANN001
    """A partial `selected_verts` would mis-pair weights with vertices; it must raise, not broadcast.

    `opt._jit_minimize` calls the inner loss on `vt[selected_verts]`, so this is the one silent
    failure mode of the construction.
    """
    drive = make_drive_energy(energy, mode, A0)
    with pytest.raises(ValueError, match="mis-pair"):
        drive(mesh.vertices[:-2], mesh.edges, mesh.faces, None, None, mesh.faces_params)


def test_drive_drift_is_zero_on_the_reference_state(mesh, mode) -> None:  # noqa: ANN001
    """The drift diagnostic must vanish where the weights were built."""
    assert drive_drift(mesh.vertices, mode) < 1e-13
