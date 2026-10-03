"""The pixel-footprint render mode of `render_interface_image`.

A camera pixel records the MEAN of the blurred interface density over its square footprint; the
default render POINT-SAMPLES it at the pixel centre. With `pixel_footprint=True` the spectrum is
multiplied by the transform of the box of side `Delta`, `sinc(xi_x Delta/2) sinc(xi_y Delta/2)`, and
summed over the alias orders `xi + 2 pi m / Delta` that sampling folds onto each frequency.

The reference is the brute-force definition of a footprint average: render at `s` times the
resolution, with the fine samples placed at the midpoints of an `s x s` subdivision of each coarse
pixel, and average each block. With an even `s` no fine sample falls on a coarse centre, so the
mesh is translated by `(s - 1) / 2` fine pixels to centre the blocks; the render is periodic, so a
translation needs no wrapping (see `test_dmic.py`, translation test).

That block mean is a midpoint quadrature of the pixel integral, so it carries its own error, falling
as `1 / s^2`: about 1.4e-3 at `s = 8` and 3e-4 at `s = 16` for a width of half a pixel. The
Richardson combination `(4 r_16 - r_8) / 3` removes the leading term and is the sharper reference.

`float64` is enabled per-module and scoped, as in `test_dmic.py`.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from vertax.cost import cost_dmic, make_cost_dmic, render_interface_image
from vertax.meshes.pbc_mesh import PbcMesh

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mesh_1008.npz"
RES = 32                  # coarse resolution; the 16x reference is then 512 px per axis
TOLERANCE = 1e-3          # relative L2 between the footprint render and a block-mean reference
EXTRAPOLATED_TOLERANCE = 1e-5   # against the Richardson-extrapolated reference


@pytest.fixture(autouse=True)
def _x64():  # noqa: PT004
    with enable_x64():
        yield


@pytest.fixture()
def mesh():
    m = PbcMesh.load_mesh(str(FIXTURE))
    return (jnp.asarray(m.vertices, jnp.float64), jnp.asarray(m.edges),
            jnp.asarray(m.faces), float(m.width), float(m.height))


def _supersampled_footprint_average(v, he, W, H, res, sigma_psf, factor):
    """Mean of `factor x factor` point samples per pixel, centred on the pixel."""
    shift = jnp.asarray([(factor - 1) / 2 * W / res / factor, (factor - 1) / 2 * H / res / factor])
    fine = np.asarray(render_interface_image(v.at[:, :2].add(shift), he, W, H, factor * res,
                                             sigma_psf))
    return fine.reshape(res, factor, res, factor).mean(axis=(1, 3))


def _relative_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _sigma_physical(sigma_px, W):
    return sigma_px * W / RES


@pytest.mark.parametrize(
    ("sigma_px", "factor"),
    [
        # (0.5, 8) is left out: the 8x block mean is itself 1.4e-3 from the pixel average there
        (0.5, 16), (1.0, 8), (1.0, 16), (4.0, 8), (4.0, 16),
    ],
)
def test_footprint_render_equals_the_supersampled_pixel_average(mesh, sigma_px, factor) -> None:
    """With the option on, the render is the block mean of `factor x factor` sub-pixel samples to
    within 0.1 % relative L2."""
    v, he, _, W, H = mesh
    sigma = _sigma_physical(sigma_px, W)
    footprint = np.asarray(render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True))
    reference = _supersampled_footprint_average(v, he, W, H, RES, sigma, factor)
    assert _relative_l2(footprint, reference) < TOLERANCE


@pytest.mark.parametrize("sigma_px", [0.5, 1.0, 4.0])
def test_footprint_render_equals_the_extrapolated_pixel_average(mesh, sigma_px) -> None:
    """Against the reference with its `1/s^2` quadrature error removed, the render agrees to 1e-5,
    including at half a pixel, where the alias orders carry several per cent of the image."""
    v, he, _, W, H = mesh
    sigma = _sigma_physical(sigma_px, W)
    footprint = np.asarray(render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True))
    r8 = _supersampled_footprint_average(v, he, W, H, RES, sigma, 8)
    r16 = _supersampled_footprint_average(v, he, W, H, RES, sigma, 16)
    assert _relative_l2(footprint, (4.0 * r16 - r8) / 3.0) < EXTRAPOLATED_TOLERANCE


@pytest.mark.parametrize("sigma_px", [0.5, 1.0, 4.0])
def test_supersampled_reference_has_converged(mesh, sigma_px) -> None:
    """The 16x reference is within half the tolerance of the extrapolated one, so the comparison
    above measures the renderer and not the reference."""
    v, he, _, W, H = mesh
    sigma = _sigma_physical(sigma_px, W)
    r8 = _supersampled_footprint_average(v, he, W, H, RES, sigma, 8)
    r16 = _supersampled_footprint_average(v, he, W, H, RES, sigma, 16)
    assert _relative_l2(r16, (4.0 * r16 - r8) / 3.0) < 0.5 * TOLERANCE


def test_folding_the_alias_orders_is_what_makes_half_a_pixel_exact(mesh) -> None:
    """At 0.5 px the footprint factor on the image's own frequency grid alone is several per cent
    from the pixel average; the default alias orders bring it below the tolerance."""
    v, he, _, W, H = mesh
    sigma = _sigma_physical(0.5, W)
    reference = _supersampled_footprint_average(v, he, W, H, RES, sigma, 16)
    unfolded = np.asarray(render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True,
                                                 footprint_alias_orders=0))
    folded = np.asarray(render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True))
    assert _relative_l2(unfolded, reference) > 0.02
    assert _relative_l2(folded, reference) < TOLERANCE


@pytest.mark.parametrize("sigma_px", [0.5, 1.0, 4.0])
def test_footprint_render_is_closer_to_the_pixel_average_than_the_point_sample(mesh, sigma_px) -> None:
    """At every width the option at least halves the distance to the pixel average left by point
    sampling."""
    v, he, _, W, H = mesh
    sigma = _sigma_physical(sigma_px, W)
    reference = _supersampled_footprint_average(v, he, W, H, RES, sigma, 16)
    footprint = np.asarray(render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True))
    point = np.asarray(render_interface_image(v, he, W, H, RES, sigma))
    assert _relative_l2(footprint, reference) < 0.5 * _relative_l2(point, reference)


def test_footprint_render_still_integrates_to_one(mesh) -> None:
    """The box transform is 1 at `xi = 0`, so the normalisation is untouched."""
    v, he, _, W, H = mesh
    img = render_interface_image(v, he, W, H, RES, _sigma_physical(1.0, W), pixel_footprint=True)
    assert float(jnp.sum(img)) * (W / RES) * (H / RES) == pytest.approx(1.0, rel=1e-12)


def test_option_off_is_bit_identical_to_the_default(mesh) -> None:
    """`pixel_footprint=False` is the default path, for the renderer and for both image costs."""
    v, he, fa, W, H = mesh
    sigma = _sigma_physical(1.0, W)
    for kwargs in ({}, {"normalise": False}, {"selected_hes": jnp.arange(0, he.shape[0], 2)}):
        a = np.asarray(render_interface_image(v, he, W, H, RES, sigma, **kwargs))
        b = np.asarray(render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=False,
                                              **kwargs))
        assert np.array_equal(a, b)

    target = render_interface_image(v, he, W, H, RES, sigma)
    moved = v.at[:, :2].add(0.02)

    def value_and_grad(cost):
        f = lambda x: cost(x, he, fa, W, H, None, None, None, None, None, None, target)  # noqa: E731
        return float(f(moved)), np.asarray(jax.grad(f)(moved))

    for kwargs in ({}, {"intensity_weighted": True, "weight_floor": 0.01}):
        c0, g0 = value_and_grad(make_cost_dmic(sigma, **kwargs))
        c1, g1 = value_and_grad(make_cost_dmic(sigma, pixel_footprint=False, **kwargs))
        assert c0 == c1
        assert np.array_equal(g0, g1)

    f0 = cost_dmic(moved, he, fa, W, H, None, None, None, None, None, None, target, sigma_psf=sigma)
    f1 = cost_dmic(moved, he, fa, W, H, None, None, None, None, None, None, target, sigma_psf=sigma,
                   pixel_footprint=False)
    assert float(f0) == float(f1)


@pytest.mark.parametrize("sigma_px", [0.5, 4.0])
def test_objective_and_gradient_vanish_at_the_true_geometry_with_footprint(mesh, sigma_px) -> None:
    """A target rendered with the footprint is fitted exactly by the footprint model, and the
    option changes the objective away from the optimum."""
    v, he, fa, W, H = mesh
    sigma = _sigma_physical(sigma_px, W)
    target = render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True)
    cost = make_cost_dmic(sigma, pixel_footprint=True)

    def call(c, x):
        return c(x, he, fa, W, H, None, None, None, None, None, None, target)

    assert float(call(cost, v)) == pytest.approx(0.0, abs=1e-24)
    g = jax.grad(lambda x: call(cost, x))(v)
    assert float(jnp.max(jnp.abs(g))) < 1e-12, "gradient must vanish at the optimum"

    moved = v.at[:, :2].add(0.02)
    assert float(call(cost, moved)) > 1e-12
    assert float(call(cost, moved)) != float(call(make_cost_dmic(sigma), moved))


@pytest.mark.parametrize("sigma_px", [0.5, 4.0])
@pytest.mark.parametrize("intensity_weighted", [False, True])
def test_footprint_cost_gradient_matches_central_differences(mesh, sigma_px,
                                                             intensity_weighted) -> None:
    """AD through the footprint render, alias orders included, against central differences."""
    v, he, fa, W, H = mesh
    sigma = _sigma_physical(sigma_px, W)
    target = render_interface_image(v, he, W, H, RES, sigma, pixel_footprint=True)
    x0 = v.at[:, :2].add(0.02)
    cost = make_cost_dmic(sigma, intensity_weighted=intensity_weighted, pixel_footprint=True)

    def f(x):
        return cost(x, he, fa, W, H, None, None, None, None, None, None, target)

    g = np.asarray(jax.grad(f)(x0))
    eps = 1e-6
    rng = np.random.default_rng(0)
    picks = [(int(i), int(j)) for i, j in
             zip(rng.integers(0, x0.shape[0], 6), rng.integers(0, 2, 6))]
    for i, j in picks:
        fd = float((f(x0.at[i, j].add(eps)) - f(x0.at[i, j].add(-eps))) / (2 * eps))
        assert g[i, j] == pytest.approx(fd, rel=2e-5, abs=1e-11), f"coordinate ({i},{j})"


def test_alias_orders_follow_the_width(mesh) -> None:
    """None folds only where the PSF leaves content above the pixel Nyquist frequency, and an
    explicit order reproduces the default when it names the same order."""
    v, he, _, W, H = mesh
    wide = _sigma_physical(4.0, W)
    a = np.asarray(render_interface_image(v, he, W, H, RES, wide, pixel_footprint=True))
    b = np.asarray(render_interface_image(v, he, W, H, RES, wide, pixel_footprint=True,
                                          footprint_alias_orders=0))
    assert np.array_equal(a, b)
    narrow = _sigma_physical(0.5, W)
    c = np.asarray(render_interface_image(v, he, W, H, RES, narrow, pixel_footprint=True))
    d = np.asarray(render_interface_image(v, he, W, H, RES, narrow, pixel_footprint=True,
                                          footprint_alias_orders=2))
    assert np.array_equal(c, d)
