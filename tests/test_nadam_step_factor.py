"""The reset-Adam(nesterov) effective step factor, verified against optax rather than derived.

WHY THIS IS A TEST AND NOT AN ANALYSIS SCRIPT. It establishes a property of the optimiser this
project's outer loop relies on: with the outer state reset every epoch, every update is a first
update, so from zero state NAdam's Dozat formulation gives

    delta_theta = -eta * (1 + beta1 / (1 + beta1)) * sign(g)

i.e. 1.4737 * eta at beta1 = 0.9. That is a DERIVED factor, and optax's `nesterov` implementation
need not match Dozat's paper. So it is measured: apply ONE update from a fresh state with g = 1 and
read the step magnitude. beta1 is swept to test the functional FORM rather than one number -- a
coincidence at 0.9 would not survive 0.5, 0.8 and 0.99.

Two consequences the outer loop depends on, asserted here so they cannot drift silently:

* the step is SCALE-BLIND -- sign-like, independent of |g| over six decades, which is why the outer
  update behaves as a fixed-size signed step and why the epoch budget and step size trade off the
  way they do;
* `nesterov=False` gives exactly 1.0, so an arm run without Nesterov takes a 32 % SMALLER effective
  step at the same nominal learning rate. Any comparison between those two arms is confounded by
  step size unless this is accounted for.

Verification harnesses that establish `src/`-adjacent behaviour belong in the released suite, not
in the unreleased analysis tree.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.experimental import enable_x64

ETA = 0.001
BETA1S = (0.5, 0.8, 0.9, 0.99)

# Adam divides by `sqrt(v_hat) + eps` with `eps = 1e-8` by default, so with g = 1 the factor comes
# out as 1/(1 + 1e-8) rather than exactly 1. The tolerance below is therefore set BY THAT EPSILON,
# not chosen as a round number: 1e-6 is two orders above the eps artefact and many orders below any
# change of formulation worth catching. An earlier version of this test asserted `abs=1e-9`, which
# is tighter than the algorithm's own epsilon and failed on correct behaviour.
ADAM_EPS = 1e-8
TOL = 1e-6


@pytest.fixture(autouse=True)
def _enable_x64_for_this_module():  # noqa: PT004
    """float64 for a factor asserted at 1e-9; scoped so it cannot leak to other modules."""
    with enable_x64():
        yield


def _first_step(eta: float, beta1: float, nesterov: bool, g: float) -> float:
    """One update from a fresh (zero) state -- exactly what a per-epoch reset produces."""
    opt = optax.adam(learning_rate=eta, b1=beta1, nesterov=nesterov)
    p = jnp.zeros((1,), jnp.float64)
    grad = jnp.full((1,), g, jnp.float64)
    upd, _ = opt.update(grad, opt.init(p), p)
    return float(np.asarray(upd)[0])


@pytest.mark.parametrize("beta1", BETA1S)
def test_nesterov_step_factor_matches_dozat(beta1: float) -> None:
    """The measured factor must equal 1 + beta1/(1+beta1) at every beta1, not just at 0.9."""
    factor = abs(_first_step(ETA, beta1, True, 1.0)) / ETA
    assert factor == pytest.approx(1.0 + beta1 / (1.0 + beta1), rel=TOL)


@pytest.mark.parametrize("beta1", BETA1S)
def test_plain_adam_step_factor_is_exactly_one(beta1: float) -> None:
    assert abs(_first_step(ETA, beta1, False, 1.0)) / ETA == pytest.approx(1.0, rel=TOL)


def test_first_step_is_scale_blind_where_adam_eps_is_negligible() -> None:
    """The magnitude must not depend on |g|: this is what makes the update a signed step.

    Bounded to `|g| >= 1e-2`. The deviation from perfect scale-blindness is set by Adam's
    denominator, `|g| / (|g| + eps)`, so it is of relative size `eps / |g|` -- 1e-6 at `|g| = 1e-2`
    and 1e-2 at `|g| = 1e-6`. The bound here is therefore derived from that ratio rather than
    chosen; the test below pins the scaling itself.
    """
    steps = [abs(_first_step(ETA, 0.9, True, g)) for g in (1e0, 1e-1, 1e-2)]
    assert max(steps) / min(steps) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("g", [1e-2, 1e-4, 1e-6])
def test_scale_blindness_deviation_scales_as_adam_eps_over_gradient(g: float) -> None:
    """MEASURED: the step shortfall is `eps / |g|`, so it reaches ~1 % at `|g| = 1e-6`.

    Recorded as behaviour rather than dismissed as noise. It matters when interpreting a run whose
    hypergradient has decayed towards 1e-6: the update there is no longer a pure signed step but a
    shrinking one, so an apparent plateau at very small gradient is partly an optimiser artefact
    rather than a property of the problem.
    """
    ref = abs(_first_step(ETA, 0.9, True, 1.0))
    shortfall = 1.0 - abs(_first_step(ETA, 0.9, True, g)) / ref
    assert shortfall == pytest.approx(ADAM_EPS / g, rel=0.05)


def test_nesterov_takes_a_larger_step_than_plain_adam() -> None:
    """Quantifies the confound between a nesterov and a non-nesterov arm at equal nominal eta."""
    ratio = abs(_first_step(ETA, 0.9, True, 1.0) / _first_step(ETA, 0.9, False, 1.0))
    assert ratio == pytest.approx(1.4737, abs=1e-3)
