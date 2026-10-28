"""Regression and property tests for `vertax.clamped`.

WHY THIS EXISTS. The clamped-patch mechanics layer was promoted out of a local analysis script, and
its correctness was established by bit-exact comparison against that script on 25 crops (925
comparisons). That comparison cannot be repeated once the script is gone, so the properties it
covered are pinned here against a stored reference over one small committed patch.

THE FIXTURE. `tests/fixtures/clamped_patch_demo_003_pos2.npz`, 8 KiB: 140 vertices, 30 faces, 65
cell-cell edges, 36 free vertices. Small, and it exercises the interesting paths — under the
reference tension field four edges collapse to ~3e-7, which is a blocked T1 and is exactly the
situation the residual bookkeeping exists for.

`float64` is required and is enabled per-module, scoped, rather than at import: a library must not
flip a process-wide flag, and neither must a test.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax.clamped import (
    ClampedPatch,
    best_K,
    clamped_edges_are_inert,
    load_mesh,
    newton_polish,
    patch_geometry,
    r_eq,
    relax,
    signed_face_areas,
    split_gradient,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "clamped_patch_demo_003_pos2.npz"
K_SHOW = 5.0


@pytest.fixture(autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    """Residuals here reach 1e-6; float32 cannot represent the comparisons below."""
    with enable_x64():
        yield


@pytest.fixture()
def reference() -> dict:
    return dict(np.load(HERE / "reference_result_test_clamped.npz", allow_pickle=False))


@pytest.fixture()
def patch() -> ClampedPatch:
    return load_mesh(FIXTURE)


def _energy(patch: ClampedPatch, gamma, eps: float = 0.0):  # noqa: ANN001, ANN202
    """E2 on the patch. Defined HERE, not in `vertax.clamped`, which ships no energy by design."""
    gamma = jnp.asarray(gamma, jnp.float64)

    def E(x, K):  # noqa: ANN001, ANN202
        A, _, L = patch_geometry(x, patch, eps)
        return 0.5 * K * jnp.sum((A - 1.0) ** 2) + jnp.sum(gamma * L)
    return E


# ---------------------------------------------------------------------------------------
# 1. Regression against the stored reference
# ---------------------------------------------------------------------------------------
def test_patch_loads_with_the_expected_shape(patch: ClampedPatch, reference: dict) -> None:
    assert patch.nf == int(reference["nf"])
    assert patch.n_free == int(reference["n_free"])
    assert len(patch.EA) == int(reference["ncc"])
    assert patch.n_free == len(patch.free_idx)
    assert len(patch.PA) == len(patch.PB) == len(patch.PF)


def test_geometry_matches_reference_exactly(patch: ClampedPatch, reference: dict) -> None:
    """Zero tolerance: the promotion was validated as bit-exact and must stay that way."""
    xp = jnp.asarray(reference["xp"])
    A, P, L = (np.asarray(t) for t in patch_geometry(xp, patch))
    np.testing.assert_array_equal(A, reference["areas"])
    np.testing.assert_array_equal(P, reference["perims"])
    np.testing.assert_array_equal(L, reference["lens"])


def test_gradient_and_k_split_match_reference(patch: ClampedPatch, reference: dict,
                                              exact_references: bool) -> None:
    """The gradient, its K split, the residual and `best_K` against the stored reference.

    With ``VERTAX_EXACT_REFERENCES=1``, on the machine that produced the reference, bit for bit.
    Elsewhere within the rounding of the computation, as measured on Linux against the macOS
    reference (the gradient, `u`, `v` and `r_eq` are identical there, and `best_K` differs by 1 ulp):

    * the gradient, `u` and `v` within 1e-14 of the gradient's scale, the few ulps (about 45 `eps`)
      that the sums compiled by XLA for another CPU can change, as in
      `test_the_k_split_is_exact_not_a_fit`; an elementwise `rtol` would fail on the near-zero
      components;
    * `r_eq`, a norm over the `2 N_free` free components, within the error of the gradient plus
      `2 N_free` `eps` of its value;
    * `best_K`, recomputed from the stored `u` and `v` so that only its own arithmetic is compared:
      `K = -(u.v)/(u.u)` over the free components, two `np.dot` products that BLAS sums in a
      machine-dependent order, within the bound `n eps (sum |u_i v_i| / |u.v| + 1) |K|` of a
      reordered sum of `n` terms; the residual within the error that this `K` and the rounding of
      its norm give.
    """
    xp = jnp.asarray(reference["xp"])
    E = _energy(patch, reference["gamma"])
    g = np.asarray(jax.grad(E, argnums=0)(xp, K_SHOW))
    u, v = split_gradient(E, patch, xp)
    if exact_references:
        np.testing.assert_array_equal(g, reference["grad"])
        np.testing.assert_array_equal(u, reference["u"])
        np.testing.assert_array_equal(v, reference["v"])
        assert r_eq(g, patch) == float(reference["r_eq"])
        np.testing.assert_array_equal(np.array(best_K(u, v, patch)), reference["best_K"])
        return

    eps = np.finfo(np.float64).eps
    g_atol = 1e-14 * np.abs(reference["grad"]).max()
    np.testing.assert_allclose(g, reference["grad"], rtol=0, atol=g_atol)
    np.testing.assert_allclose(u, reference["u"], rtol=0, atol=1e-14 * np.abs(reference["u"]).max())
    np.testing.assert_allclose(v, reference["v"], rtol=0, atol=1e-14 * np.abs(reference["v"]).max())
    n_free_components = 2 * patch.n_free
    r_ref = float(reference["r_eq"])
    assert abs(r_eq(g, patch) - r_ref) <= g_atol + n_free_components * eps * r_ref

    uf, vf = (np.asarray(reference[k])[patch.free_idx].ravel() for k in ("u", "v"))
    K_ref, res_ref = (float(x) for x in reference["best_K"])
    K, res = best_K(reference["u"], reference["v"], patch)
    K_bound = uf.size * eps * (np.abs(uf * vf).sum() / abs(uf @ vf) + 1.0) * abs(K_ref)
    assert abs(K - K_ref) <= K_bound
    res_bound = (K_bound * np.linalg.norm(uf)
                 + uf.size * eps * (abs(K_ref) * np.linalg.norm(uf) + np.linalg.norm(vf))) / np.sqrt(uf.size)
    assert abs(res - res_ref) <= res_bound


def test_relaxation_matches_reference(patch: ClampedPatch, reference: dict) -> None:
    """L-BFGS through scipy, so a tolerance rather than exact equality.

    L-BFGS stops when the energy no longer decreases in float64 (`ftol` is below `eps`), so the
    stopping point is set by the rounding of the energy, and another machine stops elsewhere on the
    same flat bottom. MEASURED, Linux against the macOS reference: positions within 4.9e-6, energy
    within 1e-12 relative, `min_area` within 6e-7 relative, and the smoothed residual 7.1e-6 against
    2.9e-5.

    * The energy is second order in the position error, so it keeps its tolerance, 1e-8 relative.
    * `min_area` is first order in it: 1e-5 relative, 15 times the measured difference.
    * The smoothed residual at the stop is the floor that the energy's rounding allows, about
      `sqrt(2 lambda eps E)` for a curvature `lambda`, large (about 1/eps_smooth) at the four collapsed
      edges, so its value is not reproducible and is not compared with the reference. It is checked to
      be below 1e-4 of the residual at the observed configuration, its scale, three times above the
      larger of the two measured values: a relaxation that has not converged stays far above that.
    """
    E0 = _energy(patch, reference["gamma"])
    Es = _energy(patch, reference["gamma"], eps=1e-6)
    out = relax(Es, patch, K_SHOW, E_exact=E0)
    assert out["success"]
    assert out["E"] == pytest.approx(float(reference["relax_E"]), rel=1e-8)
    assert out["r_eq_smoothed"] < 1e-4 * out["r_eq_obs"], (
        f"smoothed residual {out['r_eq_smoothed']:.3e} against {out['r_eq_obs']:.3e} at the observed "
        f"configuration (reference {float(reference['relax_r_eq_smoothed']):.3e})")
    assert out["min_area"] == pytest.approx(float(reference["relax_min_area"]), rel=1e-5)


# ---------------------------------------------------------------------------------------
# 2. Properties the module documents, checked rather than trusted
# ---------------------------------------------------------------------------------------
def test_the_k_split_is_exact_not_a_fit(patch: ClampedPatch, reference: dict) -> None:
    """`grad E = K u + v` must hold at ARBITRARY K, since the energy is affine in K.

    Exact in exact arithmetic, and the test is at K values far outside the {2, 5, 10} band to make
    sure it is a decomposition and not a fit. In float64 it is exact to a few ulps rather than to
    the bit, because the reconstruction evaluates `K*(g1 - g0) + g0` while the direct gradient
    evaluates the fused expression: MEASURED relative error 0 at K=0, 1.9e-17 at K=1, and
    1.1e-15 to 2.1e-15 at K = 2.5, 7 and 100. The bound below is set an order above that, and
    against the gradient's own SCALE -- an elementwise `rtol` would fail on the components that
    are near zero, which is a property of those components and not of the decomposition.
    """
    xp = jnp.asarray(reference["xp"])
    E = _energy(patch, reference["gamma"])
    u, v = split_gradient(E, patch, xp)
    for K in (0.0, 1.0, 2.5, 7.0, 100.0):
        direct = np.asarray(jax.grad(E, argnums=0)(xp, K))
        err = np.abs(K * u + v - direct).max()
        assert err <= 1e-14 * max(np.abs(direct).max(), 1.0), (
            f"K={K}: reconstruction error {err:.3e} exceeds a few ulps of the gradient scale")


def test_residual_ignores_clamped_vertices(patch: ClampedPatch, reference: dict) -> None:
    """r_eq must not move when the gradient at a CLAMPED vertex changes.

    This is the whole reason the residual is restricted: a clamped vertex carries an unknown
    reaction force from unimaged material, and including it would report the clamp instead of the
    mechanics.
    """
    g = np.array(reference["grad"], copy=True)
    before = r_eq(g, patch)
    clamped = np.setdiff1d(np.arange(len(g)), patch.free_idx)
    assert len(clamped) > 0, "fixture must have clamped vertices for this test to mean anything"
    g[clamped] += 1e3                      # an enormous fake reaction force
    assert r_eq(g, patch) == before

    g2 = np.array(reference["grad"], copy=True)
    g2[patch.free_idx[0]] += 1e-3          # a tiny change on a FREE vertex must register
    assert r_eq(g2, patch) != before


def test_areas_are_signed_not_absolute(patch: ClampedPatch, reference: dict) -> None:
    """Reflecting the patch must flip every area sign.

    `vertax.geo.get_area` returns `|shoelace|` and would give the same answer for both, which is
    the divergence this module documents: an absolute area cannot detect an inverted cell.
    """
    x = jnp.asarray(reference["xp"])
    A = np.asarray(signed_face_areas(x, patch))
    A_reflected = np.asarray(signed_face_areas(x * jnp.array([1.0, -1.0]), patch))
    assert (A > 0).all(), "the fixture should be counter-clockwise throughout"
    np.testing.assert_allclose(A_reflected, -A, rtol=1e-12)
    assert (A_reflected < 0).all(), "a reflected patch must read as inverted, not as unchanged"


def test_clamped_edges_exert_no_force_on_free_vertices() -> None:
    """The claim that lets the energy drop the tension on cell-background edges."""
    out = clamped_edges_are_inert(FIXTURE)
    assert out["max_abs_diff_free"] < 1e-12, "dropping clamped edges changed a free-vertex force"
    assert out["max_abs_diff_clamped"] > 1e-3, (
        "the dropped term must act SOMEWHERE, namely on the clamped rows; if it does not, the "
        "test is comparing two identical edge sets and proves nothing")


def test_smoothing_converges_where_the_exact_energy_stalls(patch: ClampedPatch,
                                                           reference: dict) -> None:
    """The documented reason `smoothed_norm` exists, as a measurement.

    Under the reference tension field four edges collapse towards zero length. Those are blocked
    T1s: their force does not vanish with their length, so they leave an irreducible residual that
    is a property of the configuration and not of the solver. The test therefore checks the
    residual EXCLUDING those vertices, which is the number that says whether the solve converged.
    """
    E0 = _energy(patch, reference["gamma"])
    Es = _energy(patch, reference["gamma"], eps=1e-6)
    out = relax(Es, patch, K_SHOW, E_exact=E0)

    assert out["n_edge_lt_1e3"] == int(reference["relax_n_edge_short"]) > 0, (
        "fixture should exercise the blocked-T1 path")
    assert out["r_eq_excl_T1"] < 1e-4, "excluding blocked T1s, the solve must have converged"
    assert out["r_eq"] > 10 * out["r_eq_excl_T1"], (
        "the blocked T1s must dominate the raw residual; if they do not, this fixture no longer "
        "demonstrates the distinction the residual bookkeeping exists for")


def test_newton_polish_returns_a_full_configuration(patch: ClampedPatch,
                                                    reference: dict) -> None:
    """Polishing must move only free coordinates and leave the clamped ones exactly alone."""
    E0 = _energy(patch, reference["gamma"])
    x_start = np.asarray(reference["relax_x"])
    xp = newton_polish(E0, patch, K_SHOW, x_start, iters=5)
    assert xp.shape == x_start.shape
    clamped = np.setdiff1d(np.arange(len(xp)), patch.free_idx)
    np.testing.assert_array_equal(xp[clamped], x_start[clamped])


def test_relax_holds_clamped_vertices_at_their_observed_positions(patch: ClampedPatch,
                                                                  reference: dict) -> None:
    E0 = _energy(patch, reference["gamma"])
    out = relax(_energy(patch, reference["gamma"], eps=1e-6), patch, K_SHOW, E_exact=E0)
    clamped = np.setdiff1d(np.arange(len(patch.x0)), patch.free_idx)
    np.testing.assert_array_equal(out["x"][clamped], patch.x0[clamped])


def test_best_K_minimises_the_free_residual(patch: ClampedPatch, reference: dict) -> None:
    """Closed form, so it must beat a scan; and it is a diagnostic, not an inference."""
    xp = jnp.asarray(reference["xp"])
    E = _energy(patch, reference["gamma"])
    u, v = split_gradient(E, patch, xp)
    K_star, res_star = best_K(u, v, patch)
    for K in (K_star - 0.5, K_star - 0.1, K_star + 0.1, K_star + 0.5):
        assert r_eq(K * u + v, patch) >= res_star - 1e-15
