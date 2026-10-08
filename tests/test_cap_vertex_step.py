"""``cap_vertex_step`` on the bilevel optimizers: automatic for line-search solvers, passed to every relaxation.

A line-search solver chooses its own step and can move a vertex across whole cells in one iteration,
which defeats the incremental T1 detector and can turn a cell inside out; the area energy cannot push the
cell back, because ``get_area`` and ``get_area_bounded`` return absolute values. Capping each per-step
vertex displacement at ``MAX_VERTEX_STEP_FACTOR * min_dist_T1`` prevents it. Checked here:

* the automatic choice (``cap_vertex_step = None``) is on for line-search solvers (``gd_linesearch``,
  ``cg_linesearch``, ``optax.lbfgs()``) and off for fixed-step ones (``optax.adam``, ``optax.sgd``), on both
  optimizers; an explicit ``True`` or ``False`` always wins;
* the choice reaches every relaxation an optimizer runs: the inner optimization and the inner solves of the
  four hypergradient estimators (automatic differentiation, equilibrium propagation, implicit
  differentiation, adjoint state), on both optimizers, and for equilibrium propagation with persistent
  outer-optimizer state;
* regression, on a small bounded tissue relaxed by ``gd_linesearch``: without the cap the first step moves a
  vertex by tens of ``min_dist_T1`` and turns cells inside out; with the automatic choice no vertex moves by
  more than the cap and no cell is inverted.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import optax
import pytest

import vertax.opt
import vertax.opt_bounded
from vertax import BoundedBilevelOptimizer, BoundedMesh, PbcBilevelOptimizer, PbcMesh
from vertax.cg import cg_linesearch, gd_linesearch
from vertax.cost import cost_ratio, cost_v2v
from vertax.energy import make_energy_line_tensions, make_energy_line_tensions_bounded
from vertax.method_enum import BilevelOptimizationMethod
from vertax.opt import MAX_VERTEX_STEP_FACTOR

SOLVERS = {
    "gd_linesearch": (lambda: gd_linesearch(), True),
    "cg_linesearch": (lambda: cg_linesearch(), True),
    "lbfgs": (lambda: optax.lbfgs(), True),
    "adam": (lambda: optax.adam(1e-3), False),
    "sgd": (lambda: optax.sgd(0.01), False),
}
OPTIMIZERS = {"periodic": PbcBilevelOptimizer, "bounded": BoundedBilevelOptimizer}


def _optimizer(kind: str, solver):  # noqa: ANN001, ANN202
    optimizer = OPTIMIZERS[kind]()
    # an automatic-differentiation optimizer refuses a line-search solver; the choice does not depend on it
    optimizer.bilevel_optimization_method = BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION
    optimizer.inner_solver = solver
    return optimizer


@pytest.mark.parametrize("kind", OPTIMIZERS)
@pytest.mark.parametrize("name", SOLVERS)
def test_automatic_choice_follows_the_solver(kind: str, name: str) -> None:
    make_solver, expected = SOLVERS[name]
    optimizer = _optimizer(kind, make_solver())
    assert optimizer.cap_vertex_step is None
    assert optimizer._resolve_cap_vertex_step() is expected  # noqa: SLF001


@pytest.mark.parametrize("kind", OPTIMIZERS)
@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_value_wins(kind: str, name: str, explicit: bool) -> None:
    optimizer = _optimizer(kind, SOLVERS[name][0]())
    optimizer.cap_vertex_step = explicit
    assert optimizer._resolve_cap_vertex_step() is explicit  # noqa: SLF001


def test_summary_states_the_choice() -> None:
    optimizer = _optimizer("bounded", gd_linesearch())
    assert "Cap vertex step: auto (-> True)" in optimizer.self_summary()
    optimizer.cap_vertex_step = False
    assert "Cap vertex step: False (-> False)" in optimizer.self_summary()


# ---- the choice reaches every relaxation ---------------------------------------------------------------------
PBC_WIDTH = PBC_HEIGHT = 4.0
E2 = make_energy_line_tensions(10.0)


def _pbc_energy(v, h, f, _vp, hp, fp):  # noqa: ANN001, ANN202
    return E2(v, h, f, PBC_WIDTH, PBC_HEIGHT, hp, fp)


def _periodic_problem(method: BilevelOptimizationMethod, persist: bool):  # noqa: ANN202
    mesh = PbcMesh.from_random_seeds(nb_seeds=9, width=PBC_WIDTH, height=PBC_HEIGHT, random_key=2)
    optimizer = PbcBilevelOptimizer()
    optimizer.inner_solver = optax.sgd(0.01)
    optimizer.outer_solver = optax.adam(1e-3)
    optimizer.bilevel_optimization_method = method
    optimizer.max_nb_iterations = 3
    optimizer.tolerance = 0.0
    optimizer.min_dist_T1 = 0.05
    optimizer.update_T1 = True
    optimizer.beta = 0.01
    optimizer.persist_outer_state = persist
    optimizer.loss_function_inner = _pbc_energy
    optimizer.loss_function_outer = cost_v2v
    optimizer.vertices_target = jnp.asarray(mesh.vertices)
    optimizer.edges_target = jnp.asarray(mesh.edges)
    optimizer.faces_target = jnp.asarray(mesh.faces)
    mesh.vertices_params = jnp.zeros(1)
    mesh.edges_params = jnp.ones(mesh.nb_half_edges)
    mesh.faces_params = jnp.ones(mesh.nb_faces)
    return optimizer, mesh


def _bounded_problem(method: BilevelOptimizationMethod, persist: bool):  # noqa: ANN202, ARG001
    n_cells = 6
    side = math.sqrt(n_cells)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=1)
    optimizer = BoundedBilevelOptimizer()
    optimizer.inner_solver = optax.sgd(0.01)
    optimizer.outer_solver = optax.adam(1e-3)
    optimizer.bilevel_optimization_method = method
    optimizer.max_nb_iterations = 3
    optimizer.tolerance = 0.0
    optimizer.min_dist_T1 = 0.05
    optimizer.update_T1 = True
    optimizer.beta = 0.01
    optimizer.loss_function_inner = make_energy_line_tensions_bounded(10.0)
    optimizer.loss_function_outer = cost_ratio
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.ones((n_cells - 1) * 3)
    mesh.faces_params = jnp.ones(mesh.nb_faces)
    return optimizer, mesh


CASES = [
    ("periodic", _periodic_problem, method, False) for method in BilevelOptimizationMethod
] + [("periodic", _periodic_problem, BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION, True)] + [
    ("bounded", _bounded_problem, method, False) for method in BilevelOptimizationMethod
]
CASE_IDS = [f"{kind}-{method.name.lower()}{'-persistent' if persist else ''}" for kind, _, method, persist in CASES]


class _Spy:
    """Records the ``cap_vertex_step`` handed to every compiled relaxation loop."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.seen: list[object] = []
        for module, name in ((vertax.opt, "_jit_minimize"), (vertax.opt_bounded, "_minimize_bounded")):
            monkeypatch.setattr(module, name, self._wrap(getattr(module, name)))

    def _wrap(self, function):  # noqa: ANN001, ANN202
        def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            self.seen.append(kwargs.get("cap_vertex_step", "not passed"))
            return function(*args, **kwargs)

        return spy


def _check_every_relaxation(monkeypatch, build, method, persist: bool, value: bool) -> None:  # noqa: ANN001
    optimizer, mesh = build(method, persist)
    optimizer.cap_vertex_step = value
    spy = _Spy(monkeypatch)
    optimizer.outer_optimization(mesh)
    assert spy.seen, "no relaxation ran"
    assert all(seen is value for seen in spy.seen), spy.seen
    spy.seen.clear()
    optimizer.inner_optimization(mesh, perform_energy_check=False)
    assert spy.seen == [value]


@pytest.mark.parametrize(("kind", "build", "method", "persist"), CASES, ids=CASE_IDS)
def test_every_relaxation_receives_the_choice(  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch, kind: str, build, method, persist: bool  # noqa: ANN001
) -> None:
    _check_every_relaxation(monkeypatch, build, method, persist, True)


@pytest.mark.parametrize("build", [_periodic_problem, _bounded_problem])
def test_every_relaxation_receives_an_explicit_false(monkeypatch: pytest.MonkeyPatch, build) -> None:  # noqa: ANN001
    _check_every_relaxation(monkeypatch, build, BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION, False, False)


@pytest.mark.parametrize("kind", OPTIMIZERS)
def test_automatic_choice_reaches_the_relaxation(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    build = _periodic_problem if kind == "periodic" else _bounded_problem
    optimizer, mesh = build(BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION, False)
    spy = _Spy(monkeypatch)
    optimizer.inner_optimization(mesh, perform_energy_check=False)
    assert spy.seen == [False]  # sgd: automatic means off


# ---- regression: an uncapped line-search step turns cells inside out ------------------------------------------
MIN_DIST_T1 = 0.01


def _signed_areas(mesh: BoundedMesh) -> np.ndarray:
    """Shoelace sum of the chord polygon of every face, with the sign of its traversal orientation.

    The orientation differs from face to face (``get_area_bounded`` takes the absolute value), so a cell is
    inverted when this sum changes sign between two states of the same face.
    """
    v, he, fa = np.asarray(mesh.vertices), np.asarray(mesh.edges), np.asarray(mesh.faces)
    n = len(v)
    out = []
    for face in range(fa.shape[0]):
        start = int(fa[face, 0])
        h, total = start, 0.0
        for _ in range(40):
            a = int(np.clip(he[h, 3] + he[h, 5], -n, n - 1))
            b = int(np.clip(he[h, 4] + he[h, 6] - 1, -n, n - 1))
            total += v[a, 0] * v[b, 1] - v[b, 0] * v[a, 1]
            h = int(he[h, 1])
            if h == start:
                break
        out.append(0.5 * total)
    return np.array(out)


def _first_step(cap: bool | None, seed: int):  # noqa: ANN202
    n_cells = 20
    side = math.sqrt(n_cells)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=seed)
    rng = np.random.default_rng(seed)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.asarray(1.0 + 0.5 * rng.random((n_cells - 1) * 3))
    mesh.faces_params = jnp.full(mesh.nb_faces, 1.5)
    optimizer = BoundedBilevelOptimizer()
    optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    optimizer.beta = 0.01
    optimizer.inner_solver = gd_linesearch()
    optimizer.loss_function_inner = make_energy_line_tensions_bounded(10.0)
    optimizer.min_dist_T1 = MIN_DIST_T1
    optimizer.max_nb_iterations = 1
    optimizer.tolerance = 0.0
    optimizer.update_T1 = True
    optimizer.cap_vertex_step = cap
    before, areas_before = np.asarray(mesh.vertices).copy(), _signed_areas(mesh)
    optimizer.inner_optimization(mesh, perform_energy_check=False)
    step = np.abs(np.asarray(mesh.vertices) - before).max() / MIN_DIST_T1
    inverted = int((np.sign(_signed_areas(mesh)) != np.sign(areas_before)).sum())
    return step, inverted


@pytest.mark.parametrize("seed", [1, 5])
def test_uncapped_line_search_step_inverts_cells_and_the_automatic_cap_prevents_it(seed: int) -> None:
    step_off, inverted_off = _first_step(False, seed)
    step_auto, inverted_auto = _first_step(None, seed)
    assert step_off > 2 * MAX_VERTEX_STEP_FACTOR, "the uncapped step no longer overshoots: the setup is stale"
    assert inverted_off >= 1, "the uncapped step no longer inverts a cell: the setup is stale"
    assert step_auto <= MAX_VERTEX_STEP_FACTOR * (1 + 1e-4)
    assert inverted_auto == 0
