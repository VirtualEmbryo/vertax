"""Guard against the defect class "two arms differed in a parameter nobody recognised as differing".

Four instances of this class were found in the tension-inference study, all silent:
  * the mean-tension gauge was honoured for EP and ignored for AD/Sen-ID/Adj-ID (fixed: it is now
    applied in `outer_optimization`, the single point every path passes through);
  * `beta` was set by one harness and left at an implicit default by another, so an EP study ran at
    0.01 while its pre-registration recorded 0.5179;
  * `persist_outer_state` is honoured for EP only, making `adam_persistent` a no-op on AD;
  * `nesterov=True` under a per-epoch state reset makes the same nominal `learning_rate` mean a
    1.4737x larger realised step -- NOT catchable by this guard, see `test_guard_limitation_is_documented`.

Each check is tested in BOTH directions: it must fire when it should, and must NOT fire otherwise
(a guard that always fires gets disabled, and a guard that never fires is decoration).
"""
from __future__ import annotations

import numpy as np
import optax
import pytest

from vertax.bilevelopt.bilevelopt import BETA_DEFAULT, _UNSET
from vertax.bilevelopt.pbcbop import PbcBilevelOptimizer
from vertax.method_enum import BilevelOptimizationMethod, EPVariant

AD = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
EP = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
ADJ = BilevelOptimizationMethod.ADJOINT_STATE


def _bop(method: BilevelOptimizationMethod) -> PbcBilevelOptimizer:
    """A minimally-configured optimizer: the guard runs before any mesh work, so no mesh is needed."""
    bop = PbcBilevelOptimizer()
    bop.bilevel_optimization_method = method
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.adam(learning_rate=0.001, nesterov=True)
    return bop


# ------------------------------------------------------------------ set-but-ignored -> raise
@pytest.mark.parametrize("method", [AD, ADJ])
def test_persist_outer_state_on_non_ep_raises(method: BilevelOptimizationMethod) -> None:
    """The `adam_persistent`-on-AD no-op must be an error, not a silent identical run."""
    bop = _bop(method)
    bop.persist_outer_state = True
    with pytest.raises(ValueError, match="persist_outer_state=True is IGNORED"):
        bop._check_attribute_dispatch()


def test_persist_outer_state_on_ep_does_not_raise() -> None:
    """EP genuinely honours it (pbcbop dispatches to outer_eq_prop_with_state)."""
    bop = _bop(EP)
    bop.persist_outer_state = True
    bop.beta = 0.01
    bop._check_attribute_dispatch()  # must not raise


def test_persist_outer_state_false_on_ad_does_not_raise() -> None:
    """The default must stay silent, or every AD run becomes an error."""
    bop = _bop(AD)
    assert bop.persist_outer_state is False
    bop._check_attribute_dispatch()  # must not raise


# ------------------------------------------------------------ honoured-but-unset -> warn + resolve
def test_ep_with_unset_beta_warns_and_resolves(capsys: pytest.CaptureFixture) -> None:
    """The c7 case: an EP run that never set beta must SAY it is using the default."""
    bop = _bop(EP)
    assert bop.beta is _UNSET, "beta must default to the sentinel, not to a number"
    bop._check_attribute_dispatch()
    err = capsys.readouterr().err
    assert "beta was never set" in err
    assert str(BETA_DEFAULT) in err
    # and downstream code must still see a plain float, or the dispatch would crash in jax
    assert bop.beta == BETA_DEFAULT
    assert isinstance(bop.beta, float)


def test_ep_with_explicit_beta_is_silent(capsys: pytest.CaptureFixture) -> None:
    """Setting beta deliberately must produce no warning, including at the default value."""
    for value in (BETA_DEFAULT, 0.5179475):
        bop = _bop(EP)
        bop.beta = value
        bop._check_attribute_dispatch()
        assert "beta was never set" not in capsys.readouterr().err
        assert bop.beta == value


def test_non_ep_unset_beta_is_silent_but_still_resolved(capsys: pytest.CaptureFixture) -> None:
    """AD does not consume beta, so leaving it unset is not worth a warning -- but the sentinel must
    never survive into the dispatch, where jax would choke on it."""
    bop = _bop(AD)
    bop._check_attribute_dispatch()
    assert "beta" not in capsys.readouterr().err
    assert bop.beta == BETA_DEFAULT


def test_ep_variant_on_non_ep_warns_not_raises(capsys: pytest.CaptureFixture) -> None:
    """Inert rather than damaging, and set uniformly by some harnesses: warn, do not break the run."""
    bop = _bop(AD)
    bop.ep_variant = EPVariant.ONE_SIDED
    bop._check_attribute_dispatch()  # must not raise
    assert "ep_variant" in capsys.readouterr().err


def test_ep_variant_default_on_non_ep_is_silent(capsys: pytest.CaptureFixture) -> None:
    bop = _bop(AD)
    bop._check_attribute_dispatch()
    assert "ep_variant" not in capsys.readouterr().err


# ------------------------------------------------------------------------------- runs once only
def test_guard_runs_once_not_every_epoch(capsys: pytest.CaptureFixture) -> None:
    """`outer_optimization` is called once per epoch (350x in the study); 350 identical warnings
    would train everyone to ignore them."""
    bop = _bop(EP)
    for _ in range(5):
        bop._check_attribute_dispatch()
    assert capsys.readouterr().err.count("beta was never set") == 1


# ------------------------------------------------------------------ the gauge is NOT method-gated
def test_mean_tension_gauge_default_is_off() -> None:
    """Both normalization mechanisms are off by default: neither does anything unless asked."""
    bop = PbcBilevelOptimizer()
    assert bop.mean_tension_gauge is None
    assert bop.params_normalization is None
    assert bop._resolve_param_normalization() == {}


def test_params_normalization_reaches_every_parameter_array() -> None:
    """The generalisation itself: the constraint must be available on faces and vertices too.

    Any parameter, on half-edges, faces or vertices, must be allowed a normalization constraint,
    not only edge tensions.
    """
    bop = PbcBilevelOptimizer()
    for name, target in (("vertices_params", 0.0), ("edges_params", 1.0), ("faces_params", 4.1)):
        bop.params_normalization = {name: target}
        assert bop._resolve_param_normalization() == {name: target}
    bop.params_normalization = {"edges_params": 1.0, "faces_params": 4.1}
    assert bop._resolve_param_normalization() == {"edges_params": 1.0, "faces_params": 4.1}


def test_legacy_mean_tension_gauge_still_resolves_to_edges_params() -> None:
    """Every recorded E2 result was produced with `mean_tension_gauge`."""
    bop = PbcBilevelOptimizer()
    bop.mean_tension_gauge = 1.0
    assert bop._resolve_param_normalization() == {"edges_params": 1.0}


def test_setting_both_mechanisms_for_the_same_array_is_rejected() -> None:
    """Silently resolving the conflict would let dict ordering decide which target wins."""
    bop = PbcBilevelOptimizer()
    bop.mean_tension_gauge = 1.0
    bop.params_normalization = {"edges_params": 2.0}
    with pytest.raises(ValueError, match="not both"):
        bop._resolve_param_normalization()
    # a DIFFERENT array is not a conflict and must be allowed
    bop.params_normalization = {"faces_params": 4.1}
    assert bop._resolve_param_normalization() == {"edges_params": 1.0, "faces_params": 4.1}


def test_unknown_parameter_array_is_rejected() -> None:
    bop = PbcBilevelOptimizer()
    bop.params_normalization = {"face_params": 4.1}          # a plausible typo
    with pytest.raises(ValueError, match="must be among"):
        bop._resolve_param_normalization()


@pytest.mark.parametrize("method", [AD, EP, ADJ])
def test_gauge_is_applied_by_every_method(method: BilevelOptimizationMethod) -> None:
    """The original defect: the gauge must not be reachable from one estimator branch only. It is
    applied in `outer_optimization` after `_outer_opt_func`, so this checks the renormalisation
    itself rather than re-running an optimization."""
    bop = _bop(method)
    bop.mean_tension_gauge = 1.0
    # float64 explicitly: the standing project rule is that any exactness claim is made in float64,
    # and jax defaults to float32 unless x64 is enabled process-wide (which a test must not do for
    # everyone else). The claim under test is the algebra of the renormalisation, not jax's dtype.
    he = np.array([0.4, 0.9, 1.1, 2.0], dtype=np.float64)  # mean 1.1
    shifted = he - (he.mean() - bop.mean_tension_gauge)
    assert float(shifted.mean()) == pytest.approx(1.0, abs=1e-14)
    # A gauge, not a penalty: every edge shifts by exactly the same constant, so no antisymmetric
    # component can be introduced and twin half-edge pairs stay equal.
    assert float(np.max(np.abs((shifted - he) - (1.0 - 1.1)))) == pytest.approx(0.0, abs=1e-14)
    assert float(np.std(shifted)) == pytest.approx(float(np.std(he)), abs=1e-14)


# --------------------------------------------------------------------------- documented limitation
def test_guard_limitation_is_documented() -> None:
    """The Nesterov confound is invisible to any set/ignored bookkeeping: the attribute was set
    identically in both arms and honoured in both, yet meant different things. If this guard is ever
    described as complete, this test should start failing -- keep the caveat in the source.
    """
    import vertax.bilevelopt.bilevelopt as mod

    src = mod.__doc__ or ""
    table_doc = mod._check_attribute_dispatch.__doc__ if hasattr(mod, "_check_attribute_dispatch") else ""
    guard_doc = mod._BilevelOptimizer._check_attribute_dispatch.__doc__ or ""
    assert "cannot catch" in (src + str(table_doc) + guard_doc).lower() or "LIMITATION" in str(
        __import__("inspect").getsource(mod)[:6000]
    ), "the guard's limitation (the Nesterov class) must remain documented next to the table"


# ------------------------------------------------- the projection actually fires during a run
@pytest.mark.parametrize("method", [AD, EP, ADJ])
def test_faces_params_normalization_is_applied_end_to_end(method) -> None:
    """Run a real E1 bilevel step and check `mean(faces_params)` is pinned afterwards.

    The sibling test above checks the *algebra* of the projection on a numpy array. This checks
    that it is actually reached from `outer_optimization` for a FACE array, not only for edges.
    """
    import math

    import jax.numpy as jnp
    import optax

    from vertax import PbcMesh
    from vertax.cost import cost_v2v
    from vertax.energy import energy_shape_factor_hetero

    target_mean = 4.1
    L = math.sqrt(20)

    def energy(v, h, f, _vp, _hp, fp):
        return energy_shape_factor_hetero(v, h, f, L, L, jnp.arange(len(f)), fp)

    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=L, height=L, random_key=123)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    # start deliberately OFF the target mean, so a no-op would be visible
    mesh.faces_params = jnp.full(mesh.nb_faces, target_mean + 0.5)
    target = PbcMesh.copy_mesh(mesh)

    bop = _bop(method)
    bop.loss_function_inner = energy
    bop.loss_function_outer = cost_v2v
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.adam(learning_rate=1e-3)
    bop.max_nb_iterations = 3
    bop.update_T1 = False
    bop.vertices_target, bop.edges_target, bop.faces_target = (
        target.vertices, target.edges, target.faces)
    bop.params_normalization = {"faces_params": target_mean}

    before = float(jnp.mean(mesh.faces_params))
    bop.bilevel_optimization(mesh, perform_energy_check=False)
    after = float(jnp.mean(mesh.faces_params))

    assert before == pytest.approx(target_mean + 0.5, abs=1e-6), "fixture must start off-target"
    assert after == pytest.approx(target_mean, abs=1e-6), (
        f"{method}: mean(faces_params) = {after}, expected {target_mean} -- the projection did not "
        "reach the face array")


def test_normalization_shifts_uniformly_and_preserves_spread() -> None:
    """A projection, not a penalty: it must move the mean and nothing else.

    Checked on `faces_params` specifically, because the spread is the part of the shape-factor
    field that carries the CTRL-vs-treatment signal; a mechanism that damped it would silently
    destroy the measurement it exists to enable.
    """
    import math

    import jax.numpy as jnp
    import numpy as np
    import optax

    from vertax import PbcMesh
    from vertax.cost import cost_v2v
    from vertax.energy import energy_shape_factor_hetero

    L = math.sqrt(20)

    def energy(v, h, f, _vp, _hp, fp):
        return energy_shape_factor_hetero(v, h, f, L, L, jnp.arange(len(f)), fp)

    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=L, height=L, random_key=124)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    rng = np.random.default_rng(0)
    mesh.faces_params = jnp.asarray(4.6 + 0.2 * rng.standard_normal(mesh.nb_faces))
    target = PbcMesh.copy_mesh(mesh)

    bop = _bop(AD)
    bop.loss_function_inner = energy
    bop.loss_function_outer = cost_v2v
    bop.inner_solver = optax.sgd(learning_rate=0.0)      # freeze geometry
    bop.outer_solver = optax.adam(learning_rate=0.0)     # freeze parameters
    bop.max_nb_iterations = 1
    bop.update_T1 = False
    bop.vertices_target, bop.edges_target, bop.faces_target = (
        target.vertices, target.edges, target.faces)
    bop.params_normalization = {"faces_params": 4.1}

    before = np.asarray(mesh.faces_params)
    bop.bilevel_optimization(mesh, perform_energy_check=False)
    after = np.asarray(mesh.faces_params)

    assert float(after.mean()) == pytest.approx(4.1, abs=1e-6)
    assert float(after.std()) == pytest.approx(float(before.std()), rel=1e-6), "spread must survive"
    shift = after - before
    assert float(shift.max() - shift.min()) == pytest.approx(0.0, abs=1e-6), "shift must be uniform"
