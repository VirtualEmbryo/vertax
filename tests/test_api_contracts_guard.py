"""Guards for API contracts that had no dedicated regression test.

Each test here locks in one specific claim about `examples/` and `tests/`:

* `make_energy_line_tensions(_bounded)` require `K_areas` explicitly (no hidden default that
  could silently reintroduce the old hardcoded-20 hazard; see `energy.py` module docstring).
* `PbcMesh.load_mesh` must hand back `MAX_EDGES_IN_ANY_FACE` as a hashable Python `int`, not a
  0-d numpy array or `np.integer` (both of which are usable in most places but break anything
  that hashes or `static_argnums`-jits on it).
* `min_image_sq_dist` / `cost_v2v` must return a finite gradient when a vertex coincides exactly
  with its target -- the equivalent `(min_k sqrt(u_k))**2` spelling is `0/0 -> NaN` there (see
  `geo.py::min_image_sq_dist` docstring).
* The mean-tension gauge (`mean_tension_gauge`) actually holds `mean(edges_params) == target`
  after a real `outer_optimization` step, for every estimator path -- not just the renormalisation
  algebra in isolation (already covered by `test_attribute_dispatch_guard.py`).

All tests here are expected to PASS; see
`test_make_energy_line_tensions_energy_is_callable_on_a_real_mesh` for a case that also pins the
returned energy value, not merely that the call does not raise.

**On mesh sizes:** every mesh built below uses `nb_seeds=20` with `width = height = sqrt(nb_seeds)`
(the convention used throughout the rest of the test suite, e.g. `test_n_bilevel_opt.py`, and the
threshold below which the library itself prints `"[n_cells < 20] initial condition may not work as
expected"`). This is required, not cosmetic: `PbcMesh.from_random_seeds` can **hang indefinitely**
(60s+, not merely slow, confirmed with a hard `timeout`) for
some seed-count/random-key/box combinations below `nb_seeds=20` -- observed for `nb_seeds in
{4, 5, 6}` at `width=height=2.0` (all keys tried) and for `nb_seeds=10` at `width=height=sqrt(10)`
with `random_key=3` specifically (`random_key=0` and `2` at the same size returned in <0.1s, so
this is seed-dependent, not a fixed size cutoff). `nb_seeds=20` at `width=height=sqrt(20)` was
verified fast (<0.1s) for `random_key in {0..5}` and is what every test below now uses. The exact
trigger condition is not characterised beyond "small `nb_seeds`"; fixing it would be a
`src/vertax` change, out of scope for these guards.
"""
from __future__ import annotations

import importlib
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import grad

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import make_energy_line_tensions, make_energy_line_tensions_bounded
from vertax.geo import get_area, get_length, min_image_sq_dist
from vertax.method_enum import BilevelOptimizationMethod

AD = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
EP = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
ADJ = BilevelOptimizationMethod.ADJOINT_STATE


# --------------------------------------------------------------------- K_areas explicitness


def test_make_energy_line_tensions_requires_explicit_k_areas() -> None:
    """No hidden default: `make_energy_line_tensions()` must fail loudly, not silently pick 20."""
    with pytest.raises(TypeError):
        make_energy_line_tensions()  # type: ignore[call-arg]


def test_make_energy_line_tensions_bounded_requires_explicit_k_areas() -> None:
    """Same guarantee for the bounded factory."""
    with pytest.raises(TypeError):
        make_energy_line_tensions_bounded()  # type: ignore[call-arg]


def test_make_energy_line_tensions_energy_is_callable_on_a_real_mesh() -> None:
    """The factory's whole point is to be used as `loss_function_inner`; it must actually run.

    REGRESSION GUARD. `_hedge_part` must pass exactly 6 positional arguments to `get_length` --
    unlike `get_area`, a half-edge needs no max-iterations bound -- since 7 arguments raises
    `TypeError: get_length() takes 6 positional arguments but 7 were given`.

    This also checks the returned VALUE, not merely that the call does not raise. A signature fix
    could plausibly be "fixed" by passing the wrong argument, so pinning the energy is what makes
    this a real guard: the factory must reproduce the documented convention
    ``E_2 = 1/2 sum_he L_he l_he + K/2 sum_a (A_a - A0_a)^2``, verified here against an
    independent reimplementation rather than against a hardcoded number, so the assertion stays
    valid if the reference mesh ever changes.
    """
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=20.0**0.5, height=20.0**0.5, random_key=0)
    he_params = jnp.ones(mesh.nb_edges * 2)
    face_params = jnp.ones(mesh.nb_faces)

    energy_fn = make_energy_line_tensions(10.0)
    value = energy_fn(
        mesh.vertices, mesh.edges, mesh.faces, mesh.width, mesh.height, he_params, face_params
    )
    assert jnp.isfinite(value)

    # The factory must agree with an independent implementation of the same convention, so that a
    # future "fix" to the argument list cannot pass by calling get_length with the wrong arguments.
    reference = _make_energy(mesh.width, mesh.height, 10.0)(
        mesh.vertices, mesh.edges, mesh.faces, None, he_params, face_params
    )
    assert jnp.allclose(value, reference, rtol=1e-10, atol=0.0), (value, reference)

    # It must also be differentiable w.r.t. the tensions -- the only reason it exists.
    dE_dlambda = grad(
        lambda p: energy_fn(
            mesh.vertices, mesh.edges, mesh.faces, mesh.width, mesh.height, p, face_params
        )
    )(he_params)
    assert jnp.all(jnp.isfinite(dE_dlambda)), dE_dlambda


# --------------------------------------------------------------------- MAX_EDGES_IN_ANY_FACE


def test_load_mesh_max_edges_in_any_face_is_hashable_python_int(tmp_path) -> None:  # noqa: ANN001
    """`load_mesh` must hand back a plain, hashable Python int (not a 0-d array / np.integer).

    A 0-d numpy array is not hashable at all, and even a hashable `np.integer` is a different type
    from what the rest of the codebase (and any `functools.lru_cache`/`static_argnums` use) expects.
    """
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=20.0**0.5, height=20.0**0.5, random_key=3)
    path = str(tmp_path / "mesh.npz")
    mesh.save_mesh(path)
    loaded = PbcMesh.load_mesh(path)

    assert type(loaded.MAX_EDGES_IN_ANY_FACE) is int
    hash(loaded.MAX_EDGES_IN_ANY_FACE)  # must not raise
    assert loaded.MAX_EDGES_IN_ANY_FACE == mesh.MAX_EDGES_IN_ANY_FACE


# --------------------------------------------------------------------- coincident-vertex gradients


def test_min_image_sq_dist_grad_is_finite_when_points_coincide() -> None:
    """The `(min sqrt(u))**2` spelling was `0/0 -> NaN` here; the current one must not be."""
    p = jnp.array([1.0, 1.0])
    q = jnp.array([1.0, 1.0])  # exact coincidence
    g = grad(lambda a: min_image_sq_dist(a, q, 5.0, 5.0))(p)
    assert jnp.all(jnp.isfinite(g)), g


def test_cost_v2v_grad_is_finite_when_mesh_equals_target() -> None:
    """`cost_v2v` must not blow up to NaN gradients when a run starts exactly at its target."""
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=20.0**0.5, height=20.0**0.5, random_key=2)
    target_vertices = mesh.vertices  # identical -> every vertex coincides with its target

    def loss(vertices: jnp.ndarray) -> jnp.ndarray:
        return cost_v2v(
            vertices,
            mesh.edges,
            mesh.faces,
            mesh.width,
            mesh.height,
            target_vertices,
            mesh.edges,
            mesh.faces,
        )

    g = grad(loss)(mesh.vertices)
    assert jnp.all(jnp.isfinite(g)), g


# --------------------------------------------------------------------- mean-tension gauge, end to end


def _make_energy(width: float, height: float, k_areas: float):  # noqa: ANN202
    """A small, known-correct E2 energy, written independently of `make_energy_line_tensions`.

    Kept independent on purpose: it is the reference that
    `test_make_energy_line_tensions_energy_is_callable_on_a_real_mesh` checks the factory against,
    so it must not be refactored to call the factory. Note the deliberately different signature --
    `width`/`height` are closed over here, whereas the factory takes them positionally.
    """

    def energy(
        vertTable: jnp.ndarray,
        heTable: jnp.ndarray,
        faceTable: jnp.ndarray,
        _vert_params: jnp.ndarray,
        he_params: jnp.ndarray,
        face_params: jnp.ndarray,
    ) -> jnp.ndarray:
        def area_part(face: jnp.ndarray, face_param: jnp.ndarray) -> jnp.ndarray:
            a = get_area(face, vertTable, heTable, faceTable, width, height, 20)
            return (a - face_param) ** 2

        def hedge_part(he: jnp.ndarray, he_param: jnp.ndarray) -> jnp.ndarray:
            return he_param * get_length(he, vertTable, heTable, faceTable, width, height)

        from jax import vmap

        areas = vmap(area_part)(jnp.arange(len(faceTable)), face_params)
        hedges = vmap(hedge_part)(jnp.arange(len(heTable)), he_params)
        return (0.5 * jnp.sum(hedges)) + (0.5 * k_areas) * jnp.sum(areas)

    return energy


@pytest.mark.parametrize("method", [AD, EP, ADJ])
def test_mean_tension_gauge_holds_after_outer_step(method: BilevelOptimizationMethod) -> None:
    """Integration-level check: after a real `outer_optimization` step, `mean(edges_params)` sits at
    the gauge target for every estimator path -- not just the renormalisation algebra in isolation
    (`test_attribute_dispatch_guard.py::test_gauge_is_applied_by_every_method`), which is a necessary
    but not sufficient check that the wiring is correct end to end.
    """
    n_cells = 20
    width = height = float(jnp.sqrt(n_cells))
    mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = 1.2 + 0.1 * jnp.asarray(np.random.default_rng(0).standard_normal(mesh.nb_edges * 2))
    mesh.faces_params = jnp.ones(mesh.nb_faces)

    target = PbcMesh.copy_mesh(mesh)
    target.edges_params = mesh.edges_params + 0.3  # different tensions, same topology

    energy = _make_energy(width, height, k_areas=10.0)

    bop = PbcBilevelOptimizer()
    bop.loss_function_inner = energy
    bop.loss_function_outer = cost_v2v
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.adam(learning_rate=0.001, nesterov=True)
    bop.max_nb_iterations = 20
    bop.tolerance = 1e-4
    bop.patience = 5
    bop.bilevel_optimization_method = method
    if method is EP:
        bop.beta = 0.01
    bop.vertices_target = target.vertices
    bop.edges_target = target.edges
    bop.faces_target = target.faces
    bop.mean_tension_gauge = 1.0

    bop.outer_optimization(mesh)

    got_mean = float(jnp.mean(mesh.edges_params))
    assert got_mean == pytest.approx(1.0, abs=1e-6), (
        f"mean(edges_params) = {got_mean} after outer_optimization({method.name}); "
        "the post-step gauge renormalisation did not hold"
    )


# ------------------------------------------------------------------ plot_mesh must not block
def test_plot_mesh_does_not_display_by_default() -> None:
    """`plot_mesh` must default to `show=False` and must return its handles.

    A library function that calls `plt.show()` by default blocks under an interactive backend, so a
    headless or CI run hangs indefinitely rather than failing. That is exactly what happened here:
    `tests/test_plot.py` sat for over an hour before it was noticed, and only because a separate
    fix made the module collectable for the first time.

    The repair was three things together -- default `show=False`, stop closing the figure, return
    `(fig, ax)` -- because flipping the default alone would have left the 34 call sites that rely
    on it producing nothing at all, silently. This test pins all three.

    It also replaces a `conftest.py` that forced the Agg backend. That was a workaround for this
    defect; with the defect fixed the suite passes without it, and a contract test is the better
    guard because it names the property instead of suppressing the symptom.
    """
    import inspect

    from vertax import plot_mesh

    sig = inspect.signature(plot_mesh)
    assert sig.parameters["show"].default is False, (
        "plot_mesh must not display by default; a blocking library call hangs headless runs")

    for impl in ("_plot_pbc_mesh", "_plot_bounded_mesh"):
        fn = getattr(importlib.import_module("vertax.meshes.plot"), impl)
        assert inspect.signature(fn).parameters["show"].default is False, (
            f"{impl} still defaults to show=True; the dispatcher default alone is not enough")

    src = inspect.getsource(importlib.import_module("vertax.meshes.plot"))
    assert src.count("return fig, _ax") == 2, (
        "both implementations must return their handles, or `show=False` makes them produce "
        "nothing the caller can use")


# ------------------------------------------------------- shear rheometry: the two loss signatures
def _rheo_mesh():  # noqa: ANN202
    """A float64 periodic mesh, built with this suite's own `nb_seeds=20, sqrt(20)` convention.

    **Must be called inside `jax.experimental.enable_x64()`.** This module runs in the package
    default (float32) and its other guards pin energies against references computed there, so
    flipping `jax_enable_x64` globally here would silently change what they compare. The rheometry
    entry points refuse to run outside float64 (`rheology.require_float64`), which is deliberate --
    every conditioning, Hessian and equilibrium number in that path is float64 by standing rule -- so
    the four tests below scope x64 to themselves with the context manager rather than to the module.

    That convention is load-bearing rather than arbitrary: at 20 cells in a 1:2 box, 4 of 12 random
    keys hit the `PbcMesh` face-ordering defect, while the square box at 20 cells hit 0 of 12.
    The cast to float64 is a no-op under x64 since 0.2.1 (meshes follow the default float dtype); before
    that `PbcMesh.from_seeds` built `vertices` in float32 whatever the setting.
    """
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=20.0**0.5, height=20.0**0.5, random_key=0)
    mesh.vertices = jnp.asarray(mesh.vertices, jnp.float64)
    return mesh



#
# `opt._jit_minimize` calls the inner loss POSITIONALLY with six arguments, and the outer loss with
# twelve. Both of the rheometry factories return closures that must satisfy those exactly, and the
# failure mode of getting it wrong is not a TypeError at import: it is a driven solve that never
# runs, or an outer cost silently evaluated on the wrong argument. Signature drift is what this
# module exists to catch.
def test_make_drive_energy_matches_the_inner_loss_signature() -> None:
    """The drive must accept exactly `(vertTable, heTable, faceTable, vp, hp, fp)`."""
    import inspect

    from jax.experimental import enable_x64

    from vertax import make_drive_energy, make_shear_mode
    from vertax.opt import InnerLossFunction  # noqa: F401  (documents the contract being pinned)

    def base(v, h, f, vp, hp, fp):  # noqa: ANN001, ANN202
        return jnp.sum(v)

    with enable_x64():
        mesh = _rheo_mesh()
        mode = make_shear_mode(mesh.vertices, mesh.width, mesh.height, n=1)
        params = list(inspect.signature(make_drive_energy(base, mode, 1e-3)).parameters)
    assert len(params) == 6, f"inner loss takes 6 positional arguments, got {len(params)}: {params}"


def test_make_compliance_cost_matches_the_outer_loss_signature() -> None:
    """The compliance cost must accept exactly the twelve-argument outer-loss signature."""
    import inspect

    from jax.experimental import enable_x64

    from vertax import make_compliance_cost, make_shear_mode
    from vertax.cost import cost_v2v

    with enable_x64():
        mesh = _rheo_mesh()
        mode = make_shear_mode(mesh.vertices, mesh.width, mesh.height, n=1)
        cost = make_compliance_cost(mode, j_target=1.0, a0=1e-3, partner_amplitude=0.0, sign=1)
    n_cost = len(inspect.signature(cost).parameters)
    n_ref = len(inspect.signature(cost_v2v).parameters)
    assert n_cost == n_ref == 12, (
        f"outer loss takes {n_ref} arguments (cost_v2v); the compliance cost takes {n_cost}")


def test_rheobop_does_not_import_from_analysis() -> None:
    """A `src` module must not depend on an analysis script.

    The CG configuration this problem needs lives outside `src/`, and it is
    injected through `solver_factory` precisely so the library does not import it directly. If that
    ever inverts, the package becomes unimportable outside this working tree.
    """
    import importlib
    import inspect

    # No mesh and no x64 needed: this is a source-level contract. Parse the module rather than
    # grepping it: a text grep would also match the module's own DOCSTRING, which
    # explains in prose why it must not import from analysis/ and therefore contains both words.
    import ast

    tree = ast.parse(inspect.getsource(importlib.import_module("vertax.bilevelopt.rheobop")))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offending += [a.name for a in node.names if a.name.split(".")[0] == "analysis"]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "analysis":
                offending.append(node.module)
    assert not offending, f"rheobop imports from analysis/: {offending}"


def test_shear_optimizer_rejects_silent_defaults() -> None:
    """`beta` must be positive and `n_sweeps` must stay in 1..4.

    The second is not fussiness. Raising `n_sweeps` past 4 hides a non-contracting alternation instead
    of reporting it, which is the failure the plan explicitly says to surface rather than absorb.
    """
    import optax
    import pytest
    from jax.experimental import enable_x64

    from vertax import ShearRheologyOptimizer, make_shear_mode

    with enable_x64():
        mesh = _rheo_mesh()
        mode = make_shear_mode(mesh.vertices, mesh.width, mesh.height, n=1)
    kw = {"energy_theta": lambda *a: jnp.asarray(0.0), "mode": mode, "a0": 1e-3, "j_target": 1.0,
          "solver_factory": lambda: optax.sgd(1e-3), "width": mesh.width, "height": mesh.height}
    for bad_beta in (0.0, -1e-6):
        with pytest.raises(ValueError, match="beta"):
            ShearRheologyOptimizer(beta=bad_beta, **kw)
    for bad_sweeps in (0, 5):
        with pytest.raises(ValueError, match="n_sweeps"):
            ShearRheologyOptimizer(beta=1e-6, n_sweeps=bad_sweeps, **kw)
