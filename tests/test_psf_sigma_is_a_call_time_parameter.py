"""The PSF width is a parameter of the cost, supplied at the call, and differentiable.

Two things are pinned here, and they are different.

If `sigma_psf` were module state instead of a call-time argument -- a covariance matrix and a
precomputed Fourier-domain PSF, both module level, mutated from outside -- the consumer,
`_gaussian_blur_line_segments`, being `@jit`, would read the PSF as a GLOBAL and capture it as a
constant at trace time: changing the module attribute after the first call would have no effect
and raise nothing. Code that rendered once and then "changed sigma" would keep rendering at the
old width.

The width may also become trainable alongside the mechanical parameters, so the gradient with
respect to it must exist and be non-trivial. That is asserted rather than assumed, so the
requirement cannot silently regress.
"""

import math

import jax
import jax.numpy as jnp
import pytest

import vertax.cost as vc
from vertax import LEGACY_PSF_SIGMA, cost_mesh2image, make_cost_mesh2image
from vertax.meshes.pbc_mesh import PbcMesh


@pytest.fixture(scope="module")
def case():
    """A small periodic mesh, its half-edge selection, and a target rendered from itself."""
    n = 12
    side = math.sqrt(n)
    mesh = PbcMesh.from_random_seeds(nb_seeds=n, width=side, height=side, random_key=7)
    sel = jnp.arange(mesh.edges.shape[0])
    target = _render(mesh, side, sel, LEGACY_PSF_SIGMA)
    return mesh, side, sel, target


def _render(mesh, side, sel, sigma):
    wh = jnp.asarray([side, side])
    start = mesh.vertices[mesh.edges[sel, 3], :2] * 2 / wh
    end = ((mesh.vertices[mesh.edges[sel, 4], :2]
            + jnp.stack([mesh.edges[sel, 6], mesh.edges[sel, 7]], axis=-1) * wh) * 2 / wh)
    x = jnp.stack((start, end), axis=1).transpose(1, 2, 0) - 1
    img = vc._gaussian_blur_line_segments(x, sigma).real
    return img / img.sum()


def _cost(mesh, side, sel, target, sigma):
    return cost_mesh2image(mesh.vertices, mesh.edges, mesh.faces, side, side,
                           None, None, None, None, sel, None, target, sigma)


def test_psf_width_actually_changes_the_render_after_the_first_trace(case):
    """Rendering twice at two widths must actually produce two different images.

    A jitted consumer that captures a module-level width bakes it in on the first trace, so a
    second render at a different width would silently reuse the first. Here the two must differ.
    """
    mesh, side, sel, _target = case
    narrow = _render(mesh, side, sel, LEGACY_PSF_SIGMA)
    wide = _render(mesh, side, sel, LEGACY_PSF_SIGMA * 4.0)
    assert not jnp.allclose(narrow, wide), "widening the PSF did not change the render"
    # a wider PSF spreads the same total intensity, so the peak must drop
    assert float(wide.max()) < float(narrow.max())


def test_cost_is_minimised_at_the_width_the_target_was_rendered_with(case):
    """Self-consistency: the cost against a target rendered from the same mesh must vanish.

    Stated relatively rather than against an absolute floor, because this suite runs in the
    default precision and the floor is set by it, not by the cost. What carries the meaning is
    that the matched width is orders below a mismatched one.
    """
    mesh, side, sel, target = case
    matched = abs(float(_cost(mesh, side, sel, target, LEGACY_PSF_SIGMA)))
    mismatched = abs(float(_cost(mesh, side, sel, target, LEGACY_PSF_SIGMA * 4.0)))
    assert matched < 1e-5, f"cost at the matched width is {matched:.3e}"
    assert matched < mismatched / 100.0


def test_gradient_with_respect_to_sigma_is_finite_and_non_zero(case):
    """The trainability requirement. Evaluated away from the optimum, where the gradient exists.

    At the width the target was rendered with, the cost is at a minimum and the derivative is
    legitimately zero, so the check is made at a displaced width.
    """
    mesh, side, sel, target = case
    g = jax.grad(lambda s: _cost(mesh, side, sel, target, s))(LEGACY_PSF_SIGMA * 1.3)
    assert jnp.isfinite(g), "gradient with respect to sigma_psf is not finite"
    assert abs(float(g)) > 0.0, "gradient with respect to sigma_psf is identically zero"


def test_factory_matches_the_direct_call(case):
    """`make_cost_mesh2image` is the fixed-width protocol form of the same function."""
    mesh, side, sel, target = case
    bound = make_cost_mesh2image(LEGACY_PSF_SIGMA * 1.7)
    direct = float(_cost(mesh, side, sel, target, LEGACY_PSF_SIGMA * 1.7))
    viaf = float(bound(mesh.vertices, mesh.edges, mesh.faces, side, side,
                       None, None, None, None, sel, None, target))
    # the two compile separately, so agreement is to the working precision, not to the bit
    assert abs(viaf - direct) <= 1e-6 * max(abs(direct), 1e-30)


def test_sigma_psf_has_no_module_level_state(case):
    """The PSF width is a call argument, not a module global: none of these names may exist."""
    for name in ("sigma", "mu", "fg", "_fourier_gaussian"):
        assert not hasattr(vc, name), f"vertax.cost.{name} must not exist: sigma_psf is a call-time argument, not module state"
