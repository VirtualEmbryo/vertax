"""Validation for the differentiable-microscopy image model, before any inference uses it.

Includes a check that half-edge twins render identically.

`float64` is enabled per-module and scoped: a library must not flip a process-wide flag, and
neither must a test.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax.cost import _sinc, make_cost_dmic, render_interface_image
from vertax.meshes.pbc_mesh import PbcMesh

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mesh_1008.npz"
RES = 64
SIGMA = 0.30          # physical units; the box is ~4.47 wide with mean cell diameter 1
# Narrower than SIGMA: the background-suppression test below needs a target with an actual dark
# background, which the shared width blurs away (darkest pixel is 69% of brightest at SIGMA).
SIGMA_DARK = 0.18     # widest width, from a 0.01-step scan, with darkest/brightest < 5%


@pytest.fixture(autouse=True)
def _x64():  # noqa: PT004
    with enable_x64():
        yield


@pytest.fixture()
def mesh():
    m = PbcMesh.load_mesh(str(FIXTURE))
    return (jnp.asarray(m.vertices, jnp.float64), jnp.asarray(m.edges),
            jnp.asarray(m.faces), float(m.width), float(m.height))


# -------------------------------------------------------------------------------------
# 1. the closed-form segment transform against numerical quadrature
# -------------------------------------------------------------------------------------
def test_segment_transform_matches_numerical_quadrature() -> None:
    """`integral_e exp(-i xi.gamma) ds = l exp(-i xi.(a+b)/2) sinc(xi.(b-a)/2)`, exactly.

    This is the one piece of mathematics the whole renderer rests on, so it is checked against a
    brute-force quadrature rather than assumed. Several `xi` are used, including one nearly
    parallel to the segment (where `sinc` is far from 1) and one orthogonal to it (where the
    argument vanishes and `sinc` must return exactly 1).
    """
    a = jnp.array([0.31, -0.72], jnp.float64)
    b = jnp.array([1.43, 0.55], jnp.float64)
    d = b - a
    ell = jnp.linalg.norm(d)
    n_quad = 2_000_001                       # Simpson needs an odd number of points
    t = jnp.linspace(0.0, 1.0, n_quad, dtype=jnp.float64)

    for xi in (jnp.array([2.0, 3.0]), jnp.array([11.0, 0.0]),
               d / ell * 7.0,                                   # parallel
               jnp.array([-d[1], d[0]]) / ell * 7.0,            # orthogonal
               jnp.array([0.0, 0.0])):                          # DC
        gamma = a[None, :] + t[:, None] * d[None, :]
        integrand = jnp.exp(-1j * (gamma @ xi)) * ell
        quad = jnp.trapezoid(integrand, t)
        closed = ell * jnp.exp(-1j * jnp.dot(xi, 0.5 * (a + b))) * _sinc(0.5 * jnp.dot(xi, d))
        assert abs(complex(quad - closed)) < 1e-9 * float(ell), (
            f"xi={xi}: quadrature {quad} vs closed form {closed}")


# -------------------------------------------------------------------------------------
# 2. normalisation
# -------------------------------------------------------------------------------------
def test_image_integrates_to_one(mesh) -> None:
    """The `1/|Gamma|` normalisation must make the image integrate to exactly 1 over the box.

    This is the discrete Parseval statement: the DC coefficient alone fixes the total, so it holds
    to rounding, not merely to quadrature accuracy.
    """
    v, he, _, W, H = mesh
    img = render_interface_image(v, he, W, H, RES, SIGMA)
    total = float(jnp.sum(img)) * (W / RES) * (H / RES)
    assert total == pytest.approx(1.0, rel=1e-12)


def test_normalisation_is_invariant_to_a_uniform_edge_weight(mesh) -> None:
    """Scaling every `w_e` by a constant must change nothing -- `1/|Gamma|` cancels it.

    This is why iterating `heTable`, which visits every internal edge twice, is harmless on a
    confluent periodic tissue: the doubling IS `w_e = 2` and it divides out.
    """
    v, he, _, W, H = mesh
    a = render_interface_image(v, he, W, H, RES, SIGMA)
    b = render_interface_image(v, he, W, H, RES, SIGMA,
                               edge_weights=jnp.full(he.shape[0], 3.7, jnp.float64))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)


# -------------------------------------------------------------------------------------
# 3. periodicity and translation
# -------------------------------------------------------------------------------------
def test_translating_the_mesh_rolls_the_image(mesh) -> None:
    """Shifting the mesh by a whole number of pixels must roll the image by that many pixels.

    Stronger than the lattice-vector test the plan asks for, and it is what licenses comparing the
    WHOLE frame instead of a central crop: there is no seam to avoid.

    The shift is applied WITHOUT wrapping the coordinates back into the box. Wrapping would move
    vertices across the boundary and so invalidate the per-half-edge periodic offsets stored in
    `heTable[:, 6:8]`, which are computed for the original positions -- an edge that did not wrap
    would start to, and the render would legitimately change. The renderer needs no wrapping: a
    translation enters only through `exp(-i xi . mid)`, and `xi . L = 2 pi k`, so the result is
    periodic regardless.
    """
    v, he, _, W, H = mesh
    shift_px = RES // 8
    dx = W * shift_px / RES
    v_shift = v.at[:, 0].add(dx)
    img = np.asarray(render_interface_image(v, he, W, H, RES, SIGMA))
    got = np.asarray(render_interface_image(v_shift, he, W, H, RES, SIGMA))
    np.testing.assert_allclose(got, np.roll(img, shift_px, axis=1), rtol=1e-10, atol=1e-12)


def test_a_full_lattice_translation_changes_nothing(mesh) -> None:
    v, he, _, W, H = mesh
    img = render_interface_image(v, he, W, H, RES, SIGMA)
    moved = render_interface_image(v.at[:, 0].add(W), he, W, H, RES, SIGMA)
    np.testing.assert_allclose(np.asarray(moved), np.asarray(img), rtol=1e-10, atol=1e-12)


def test_half_edge_twins_render_identically(mesh) -> None:
    """Each half-edge and its twin must produce the SAME field.

    The previous real-space renderer measured pixel offsets from the segment's base point, so the
    two half-edges of an edge resolved different periodic images and disagreed by up to half the
    peak intensity on every edge. In the closed form the transform is even in `(b-a)`, and a
    lattice shift of the midpoint contributes `exp(-i 2 pi k) = 1`, so the asymmetry is gone --
    which is what makes the doubling an exact factor rather than an artefact.
    """
    v, he, _, W, H = mesh
    pairs = np.asarray(he[:, 3:5])
    seen, first, twin = {}, [], []
    for i, (s, t) in enumerate(pairs):
        k = (min(s, t), max(s, t))
        if k in seen:
            first.append(seen[k]); twin.append(i)
        else:
            seen[k] = i
    assert len(first) > 0, "fixture must contain twinned half-edges"
    a = render_interface_image(v, he, W, H, RES, SIGMA, selected_hes=jnp.asarray(first))
    b = render_interface_image(v, he, W, H, RES, SIGMA, selected_hes=jnp.asarray(twin))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


# -------------------------------------------------------------------------------------
# 4. self-consistency: the property the old cost could not satisfy
# -------------------------------------------------------------------------------------
def test_cost_and_gradient_vanish_at_a_perfect_fit(mesh) -> None:
    """A mesh scored against its own render must give exactly zero cost AND zero gradient.

    The previous cost was an unsquared L2 norm, so the perfect fit was a KINK: measured, its
    gradient converged to 278 rather than to 0, and no fixed step size could settle there. This is
    the test it could not pass.
    """
    v, he, fa, W, H = mesh
    target = render_interface_image(v, he, W, H, RES, SIGMA)
    cost = make_cost_dmic(SIGMA)

    value = cost(v, he, fa, W, H, None, None, None, None, None, None, target)
    assert float(value) == pytest.approx(0.0, abs=1e-24)

    grad = jax.grad(lambda x: cost(x, he, fa, W, H, None, None, None, None, None, None, target))(v)
    assert float(jnp.max(jnp.abs(grad))) < 1e-12, "gradient must vanish at the optimum"


def test_cost_is_positive_away_from_the_optimum(mesh) -> None:
    """Guard the test above against passing for the wrong reason (e.g. a cost that is always 0)."""
    v, he, fa, W, H = mesh
    target = render_interface_image(v, he, W, H, RES, SIGMA)
    cost = make_cost_dmic(SIGMA)
    moved = v.at[0, 0].add(0.05)
    assert float(cost(moved, he, fa, W, H, None, None, None, None, None, None, target)) > 1e-12


# -------------------------------------------------------------------------------------
# 5. the gradient itself
# -------------------------------------------------------------------------------------
def test_gradient_matches_central_differences(mesh) -> None:
    """AD against central finite differences on a handful of coordinates.

    Evaluated away from the optimum, where the gradient is non-trivial. `float64` and a step of
    1e-6 put the truncation and rounding errors both near 1e-10.
    """
    v, he, fa, W, H = mesh
    target = render_interface_image(v, he, W, H, RES, SIGMA)
    x0 = v.at[:, :2].add(0.02)                       # displace so the cost is not at its minimum
    cost = make_cost_dmic(SIGMA)

    def f(x):
        return cost(x, he, fa, W, H, None, None, None, None, None, None, target)

    g = np.asarray(jax.grad(f)(x0))
    eps = 1e-6
    rng = np.random.default_rng(0)
    picks = [(int(i), int(j)) for i, j in
             zip(rng.integers(0, x0.shape[0], 6), rng.integers(0, 2, 6))]
    for i, j in picks:
        plus = f(x0.at[i, j].add(eps))
        minus = f(x0.at[i, j].add(-eps))
        fd = float((plus - minus) / (2 * eps))
        assert g[i, j] == pytest.approx(fd, rel=2e-5, abs=1e-11), f"coordinate ({i},{j})"


# -------------------------------------------------------------------------------------
# sigma_psf is an explicit, physical parameter
# -------------------------------------------------------------------------------------
def test_sigma_psf_is_a_real_parameter_and_is_in_physical_units(mesh) -> None:
    """Two things at once: `sigma_psf` is in physical units, and it actually affects the render.

    If `sigma_psf` were specified in PIXELS rather than physical units, the physical problem would
    change with the resolution. Here `sigma_psf` is physical, so a render at
    two resolutions must agree where they are comparable -- and a different `sigma_psf` must give a
    genuinely different image, so the parameter is not being ignored.
    """
    v, he, _, W, H = mesh
    coarse = np.asarray(render_interface_image(v, he, W, H, 32, SIGMA))
    fine = np.asarray(render_interface_image(v, he, W, H, 64, SIGMA))
    # every second pixel of the fine render samples the same physical points as the coarse one
    np.testing.assert_allclose(fine[::2, ::2], coarse, rtol=2e-2, atol=2e-2)

    other = np.asarray(render_interface_image(v, he, W, H, 32, SIGMA * 2))
    assert np.abs(other - coarse).max() > 1e-3, "sigma_psf must actually affect the image"


# -------------------------------------------------------------------------------------
# 7. the intensity-weighted objective
# -------------------------------------------------------------------------------------
def test_intensity_weighting_is_off_by_default_and_does_not_change_the_optimum(mesh) -> None:
    """The flag must be additive: the unweighted path is untouched, and the minimiser moves nowhere.

    The weight is a fixed field built from the target image alone, so it cannot shift the location
    of the optimum -- only the metric in which departures from it are measured. Both properties are
    load-bearing: the first means no recorded dmic result changes, the second means a weighted run
    and an unweighted run are estimating the same thing.
    """
    v, he, fa, W, H = mesh
    target = render_interface_image(v, he, W, H, RES, SIGMA)
    plain = make_cost_dmic(SIGMA)
    weighted = make_cost_dmic(SIGMA, intensity_weighted=True)

    def call(cost, x):
        return cost(x, he, fa, W, H, None, None, None, None, None, None, target)

    # default is off: bit-identical to the unweighted cost
    moved = v.at[:, :2].add(0.02)
    assert float(call(make_cost_dmic(SIGMA, intensity_weighted=False), moved)) == float(
        call(plain, moved))

    # vanishes at the optimum, with a vanishing gradient, exactly as the unweighted form does
    assert float(call(weighted, v)) == pytest.approx(0.0, abs=1e-24)
    g = jax.grad(lambda x: call(weighted, x))(v)
    assert float(jnp.max(jnp.abs(g))) < 1e-12, "weighted gradient must vanish at the optimum"

    # positive away from it, and genuinely a different objective
    assert float(call(weighted, moved)) > 1e-12
    assert float(call(weighted, moved)) != float(call(plain, moved))


def test_intensity_weighting_suppresses_background_and_respects_the_floor(mesh) -> None:
    """The weight must actually be the image, pixel by pixel, and the floor must lift it.

    A discrepancy placed where the target image is dark should count for far less under the
    weighted objective than the same discrepancy at an interface. `weight_floor` trades that
    property away deliberately, so it must measurably raise the background contribution.
    """
    v, he, fa, W, H = mesh
    target = np.asarray(render_interface_image(v, he, W, H, RES, SIGMA_DARK))
    tgt = jnp.asarray(target)

    assert target.min() < 0.05 * target.max(), \
        "fixture must have an actual dark background: darkest pixel is not below 5% of brightest"

    dark = np.unravel_index(np.argmin(target), target.shape)
    bright = np.unravel_index(np.argmax(target), target.shape)
    # Small enough that the bump leaves its own pixel weight (target + weight_floor * max(target))
    # essentially unchanged -- at 0.1 * max it was raising its own weight enough to lift the
    # dark/bright cost ratio to ~0.13, above the 0.05 threshold this test asserts.
    bump = float(target.max()) * 1e-3

    def loss_with_bump(px, floor):
        perturbed = tgt.at[px].add(bump)          # perturb the OBSERVATION, not the mesh
        cost = make_cost_dmic(SIGMA_DARK, intensity_weighted=True, weight_floor=floor)
        return float(cost(v, he, fa, W, H, None, None, None, None, None, None, perturbed))

    assert loss_with_bump(dark, 0.0) < 0.05 * loss_with_bump(bright, 0.0), \
        "a discrepancy in background must cost far less than the same one at an interface"
    assert loss_with_bump(dark, 0.05) > loss_with_bump(dark, 0.0), \
        "weight_floor must lift the background contribution"
