"""Bi-level optimizers abstract class."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pandas
from jax import Array, lax

from vertax.meshes.mesh import Mesh
from vertax.meshes.plot import plot_mesh, save_simple_xy_graph
from vertax.method_enum import BilevelOptimizationMethod, EPVariant

__all__ = ["_BilevelOptimizer"]


class _Unset:
    """Sentinel for "the caller never assigned this attribute".

    Distinguishing "unset" from "set to the default value" is the whole point: a silent default is
    indistinguishable from a deliberate choice in a results file, and that is how an entire EP study
    came to run at beta=0.01 while its pre-registration said 0.5179.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"

    def __bool__(self) -> bool:
        return False


_UNSET = _Unset()

BETA_DEFAULT = 0.01
"""Value `beta` resolves to when an EP run leaves it unset. Applied with a warning, never silently."""

# Which optional attributes each bilevel method actually CONSUMES.
#
# Kept here, next to the guard that reads it, and cross-referenced from the dispatch `match`
# statements in `pbcbop.py` / `boundedbop.py`: if you add a branch there that starts or stops
# honouring one of these, update this table in the same commit.
#
# The recurring defect this exists to catch: two runs differ in a parameter nobody recognised as
# differing, because one code path reads it and another silently drops it. Four instances found in
# this project -- the mean-tension gauge (honoured for EP only), `beta` (set by one harness, unset by
# another), `persist_outer_state` (honoured for EP only), and `ep_variant`.
#
# LIMITATION, stated explicitly so this table is not mistaken for full coverage: it can only detect
# set-but-ignored and honoured-but-unset. It CANNOT detect an attribute that is set identically in
# two arms and honoured in both, yet means different things -- the `nesterov=True` case, where the
# same nominal `learning_rate` denotes a 1.4737x larger realised step under a per-epoch state reset.
# That class needs a functional probe of the realised step (see
# `tests/test_nadam_step_factor.py`), not a bookkeeping check.
_EP_ONLY = ("persist_outer_state", "ep_variant", "beta")


class _BilevelOptimizer:
    """Abstract base class for Bi-level optimizers.

    Use the specialezed corresponding ones:

    - to optimize a `PbcMesh` use a `PbcBilevelOptimizer`,
    - to optimize  a `BoundedMesh` use a `BoundedBilevelOptimizer`.
    """

    def __init__(self) -> None:
        """Initialize shared parameters and hyper-parameters between Bi-level optimizers."""
        self._bilevel_optimization_method: BilevelOptimizationMethod = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
        """Which bilevel optimization method to use. Defaults to Equilibrium Propagation."""
        self._inner_solver: optax.GradientTransformation = optax.sgd(learning_rate=0.01)
        """Which inner solver to use. Defaults to `optax.sgd(learning_rate=0.01)`."""
        self.outer_solver: optax.GradientTransformation = optax.adam(learning_rate=0.0001, nesterov=True)
        """Which outer solver to use. Defaults to `optax.adam(learning_rate=0.0001, nesterov=True)`"""
        self.loss_function_inner: Callable | None = None
        """Define your inner loss function."""
        self.loss_function_outer: Callable | None = None
        """Define your outer loss function."""
        self.params_regularizer: Callable | None = None
        """Optional regularization term on the parameters themselves, as
        `(vert_params, he_params, face_params) -> scalar`.

        Its gradient is added to the outer gradients *before* the outer solver's update, so the
        term is optimized jointly with the outer loss rather than applied as a separate step.
        This is the only place such a term can act: `loss_function_outer` is called without the
        parameters, and a parameter-only term added to `loss_function_inner` cancels exactly in
        equilibrium propagation's `(grad_nudged - grad_free)` difference. Defaults to None.

        Honoured by ALL estimator paths.

        CAUTION: a parameter-space penalty is mis-scaled against a *filtered* hypergradient. If the
        inner solve is truncated or preconditioned, the data gradient carries the filter factor and
        `grad R(theta)` does not, so the effective weight scales as 1/f_k. Do not use this to fix a
        gauge -- use `mean_tension_gauge`. See `outer_optimization`."""
        self.ep_variant: EPVariant = EPVariant.CENTERED
        """Which EP finite-difference form to use (EQUILIBRIUM_PROPAGATION only).
        CENTERED (default) is second-order in beta; ONE_SIDED is first-order and is
        provided for ablation. See `vertax.method_enum.EPVariant`."""
        self.params_normalization: dict[str, float] | None = None
        """Mean-value normalization constraints, per parameter array, or None for no constraint.

        A dict from parameter-array name to the target mean, e.g.
        ``{"faces_params": 4.1}`` for E1 shape factors or ``{"edges_params": 1.0}`` for E2 line
        tensions. Applied as an additive **projection** after every outer step:
        ``p <- p - (mean(p) - target)``. Valid keys are ``vertices_params``, ``edges_params`` and
        ``faces_params``. Off by default.

        **This is a normalization CONSTRAINT, not a gauge**, and the distinction is load-bearing:
        a gauge transformation leaves the physics invariant; a uniform
        shift of these parameters does **not**. For E2 tensions, ``M 1 != 0``: measured on the
        benchmark, ``||M.1|| = 1.40-1.72`` against ``||M.Lambda|| = 1.04-1.55``, so a shift of size
        ``std(Lambda)`` produces 10-14% of the tension force. For E1 shape factors the same holds --
        a uniform increase in ``p0`` makes every cell prefer a longer perimeter, which moves the
        equilibrium. The uniform direction is **near**-null, not null: least observable by a wide
        margin, which is precisely why it must be supplied from outside the data.

        **It is an ASSUMPTION, not bookkeeping.** Fixing the mean injects the true value of one
        direction -- the one that would need ~100x more data to determine than the best-conditioned
        one -- and it is known only because the benchmark was constructed with that mean. Legitimate
        and standard, but it must be stated. **On real images there is no ground-truth mean, so the
        corresponding error metric must be CENTRED** (subtract the mean from both the recovered and
        the reference field) rather than normalised against an assumed level.

        The additive convention is deliberate. The force-inference literature normalises
        *multiplicatively* (``p -> a p``), which for E2 is equivalent to ``K -> K/a`` and so trades
        against the area stiffness instead of being near-invariant. State which convention is used;
        a reader from that literature will assume the other one.

        Generalises `mean_tension_gauge`, which reached ``edges_params`` only and is why E1's
        ``faces_params`` were never constrained. Setting both is an
        error.
        """
        self.mean_tension_gauge: float | None = None
        """Target for the mean line tension, fixed by parameter renormalisation after every outer
        step (`Lambda <- Lambda - (mean(Lambda) - target)`), or None to leave the gauge free.

        The additive freedom of E2 line tensions is the least observable direction in the problem
        and carries no information, so it is fixed by convention rather than inferred. Set to 1.0
        for E2 drivers. Leave None for E1 (per-cell shape factors have no analogous freedom) and
        for bounded inverse-design runs. Prefer this to a `params_regularizer` penalty: see
        `outer_optimization` for why a penalty is mis-scaled against a filtered hypergradient."""
        self.persist_outer_state: bool = False
        """If True, the outer optimizer's moment/accumulator state is carried across bilevel
        epochs instead of being re-initialised at every call.  Only honoured by the
        EQUILIBRIUM_PROPAGATION method in ``PbcBilevelOptimizer``; setting it with any other
        method, or with a ``BoundedBilevelOptimizer``, raises ``ValueError`` at the first outer
        step.  Defaults to False (current behaviour)."""
        self._outer_opt_state: object = None
        """Cached outer optimizer state.  Initialised on the first call when
        ``persist_outer_state=True``; reset to None whenever ``persist_outer_state`` is set
        to False.  Not intended to be set directly by user code."""

        self.custom_metrics: dict[str, tuple[Callable[[Any, Any], float], list[float], list[float]]] = {}
        """Define custom metrics that can be saved during optimization.
        The function must take two arguments : a mesh and a bilevel optimizer ; and return a float."""

        self.max_nb_iterations: int = 1000
        """Maximum number of iterations during an optimization step."""
        self.tolerance: float = 1e-4
        """Below this level, we consider that the loss is stagnating."""
        self.patience: int = 5
        """Maximum number of consecutive stagnating loss before we stop."""

        self.min_dist_T1: float = 0.005
        """Threshold to perform T1 transitions."""
        self.cap_vertex_step: bool | None = None
        """Cap each per-step vertex displacement at ``MAX_VERTEX_STEP_FACTOR * min_dist_T1``.

        ``None`` (the default) means AUTO: on for line-search solvers, off for fixed-step ones. Set
        ``True`` or ``False`` to force it either way; an explicit value always wins. The choice is
        passed to every relaxation the optimizer runs, including the inner solves of every
        hypergradient estimator.

        Line-search solvers (``gd_linesearch``, ``cg_linesearch``, ``optax.lbfgs()``) pick their own
        step and can take one of order 1: a vertex then crosses whole cells in a single iteration,
        which defeats the incremental T1 detector and can turn a cell inside out (its boundary winding
        flips). Nothing in the periodic energy objects to that, because ``get_area`` takes the absolute
        value of the shoelace sum: an inverted cell scores like a sound one of equal area, so the fold
        is never pushed back out (``get_area_bounded`` did the same before 0.2.1; it now returns the
        signed area, which is not positive for a folded bounded cell). On the 20-cell tissue of the mitotic-spindle
        notebook, the first ``gd_linesearch`` step moves a vertex by 124 ``min_dist_T1`` on cells about
        0.75 wide and leaves a cell with signed area -0.056; with the cap the same step is
        ``5 min_dist_T1`` and no cell is inverted (measured by Michael Corbeau).

        Fixed-step solvers (``optax.sgd``, ``optax.adam``) are left alone because the cap is not
        inert for them: ``sgd(0.01)`` with ``min_dist_T1 = 0.005`` gives a cap of 0.025, which
        bites. Their results are therefore bit-identical to those of a library without the option."""
        self._update_T1: bool = False
        self._update_T1_func: Callable | None = None  # value set by _set_update_T1_func

        # These values will be set in the init function of child classes
        self._inner_opt_func: Callable[[Mesh, bool, Array | None, Array | None, Array | None], list[float]] | None = (
            None
        )
        self._outer_opt_func: Callable[[Mesh, Array | None, Array | None, Array | None], None] | None = None

        self.update_T1 = True  # Force the setting of update T1 func
        """Perform T1 transitions if necessary."""

        # Targets
        self.vertices_target = jnp.array([])
        """Vertices table of a target mesh if any."""
        self.edges_target = jnp.array([])
        """Edges table of a target mesh if any."""
        self.faces_target = jnp.array([])
        """Faces table of a target mesh if any."""
        # Those attributes are not always used (depends on the bilevel_optimization_method)
        self.image_target: Array = jnp.array([])
        """Passed to the outer cost as its last argument, ``image_target``; no cost of the library uses it."""
        self.beta: Any = _UNSET
        """β parameter used for Equilibrium Propagation.

        Defaults to the sentinel `_UNSET`, not to a number, so that an EP run which never assigns it
        produces a WARNING instead of silently adopting a default: an implicit default can differ
        from the intended value by a large factor with nothing reporting the discrepancy.
        Resolved to `BETA_DEFAULT` by `_check_attribute_dispatch`
        before any dispatch, so downstream code always sees a float."""

    def _set_update_T1_func(self, b: bool) -> None:  # noqa: N802
        """Set the _update_T1_func callable with respect to whether it is needed or not."""
        raise NotImplementedError

    def _resolve_cap_vertex_step(self) -> bool:
        """Turn ``cap_vertex_step`` into a decision.

        ``None`` means auto: cap line-search solvers, leave fixed-step ones alone. A line-search
        solver is recognised by the ``ScaleByZoomLinesearchState`` in its own state (see
        ``_has_zoom_linesearch_state``), whatever it is called. An explicit ``True`` or ``False``
        always wins.
        """
        if self.cap_vertex_step is not None:
            return bool(self.cap_vertex_step)
        return _has_zoom_linesearch_state(self._inner_solver)

    @property
    def update_T1(self) -> bool:  # noqa: N802
        """Whether to process T1 topological operations or not."""
        return self._update_T1

    @update_T1.setter
    def update_T1(self, value: bool) -> None:  # noqa: N802
        self._update_T1 = value
        self._set_update_T1_func(value)

    @property
    def inner_solver(self) -> optax.GradientTransformation:
        """Optax inner solver used."""
        return self._inner_solver

    @inner_solver.setter
    def inner_solver(self, new_inner_solver: optax.GradientTransformation) -> None:
        if (
            self.bilevel_optimization_method == BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
            and _is_ad_incompatible_solver(new_inner_solver)
        ):
            msg = f"[ WARNING ] A line-search inner solver (cg_linesearch, gd_linesearch, LBFGS, ...) \
is incompatible with the current optimization \
method ({BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION}). As such the inner solver was not changed. \
Current inner solver still is {self.inner_solver}. Change the optimization method if you want \
to use a line-search inner solver."
            print(msg, file=sys.stderr)
        else:
            self._inner_solver = new_inner_solver

    @property
    def bilevel_optimization_method(self) -> BilevelOptimizationMethod:
        """The bilevel optimization method currently used."""
        return self._bilevel_optimization_method

    @bilevel_optimization_method.setter
    def bilevel_optimization_method(self, method: BilevelOptimizationMethod) -> None:
        if method == BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION and _is_ad_incompatible_solver(
            self.inner_solver
        ):
            msg = f"[ WARNING ] A line-search inner solver is incompatible with the \
{BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION} method. As such the method was not changed. \
Current method still is {self.bilevel_optimization_method}. Change the inner solver if you want \
to use the {BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION} optimization method."
            print(msg, file=sys.stderr)
        else:
            self._bilevel_optimization_method = method

    def compute_outer_loss(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> float:
        """Get the result of self.loss_function_outer called with the correct arguments.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    def compute_inner_loss(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> float:
        """Get the result of self.loss_function_inner called with the correct arguments.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    def inner_optimization(
        self,
        mesh: Mesh,
        perform_energy_check: bool = True,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the inner loss function.

        Args:
            mesh (Mesh): The mesh to act on.
            perform_energy_check (bool = True): Allow T1 only if the energy is decreased by doing so.
                                                       Should be true if forward mode is used only.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: Histor of loss values during optimization.
        """
        # To be defined by child classes
        if self._inner_opt_func is None:
            msg = "The inner function was not initialized."
            raise AttributeError(msg)
        else:
            old_edges = mesh.edges.copy()  # Copy previous state to check for permutation later

            loss_history = self._inner_opt_func(
                mesh,
                perform_energy_check,
                *self._selection_to_jax_arrays(only_on_vertices, only_on_edges, only_on_faces),
            )

            # Optional permutation test if T1 happened.
            if self.vertices_target is not None and self.vertices_target.size > 0 and self.update_T1:
                _reject_bounded_t1_repair(mesh)
                perm = _build_t1_repair_perm(
                    mesh.vertices, self.vertices_target, old_edges, mesh.edges, mesh.width, mesh.height
                )
                mesh.vertices, mesh.edges = _apply_perm_to_state(perm, mesh.vertices, mesh.edges)

            return loss_history

    def _check_attribute_dispatch(self) -> None:
        """Fail loudly when an attribute is set but the selected method will ignore it.

        Runs once per optimizer instance (`outer_optimization` is called every epoch; this must not
        warn 350 times). Two directions are checked:

        * **set-but-ignored -> raise.** Assigning `persist_outer_state=True` on a non-EP method is a
          no-op: only the EQUILIBRIUM_PROPAGATION branch threads the outer optimizer state
          (`opt.py:outer_eq_prop_with_state`; there is no AD/ID equivalent). Silently ignoring it
          turns an intended reset-vs-persistent A/B into a comparison of two identical arms, which
          reports a false null rather than an error.
        * **honoured-but-unset -> warn and resolve.** An EP run that never assigns `beta` gets
          `BETA_DEFAULT` with a warning, so "the default was used" appears in the log instead of
          being indistinguishable from a deliberate choice.

        `ep_variant` set to a non-default on a non-EP method warns rather than raises: it is
        genuinely inert there and some harnesses set it uniformly across estimators.

        See the `_EP_ONLY` table above for what this can and (importantly) cannot catch.
        """
        if getattr(self, "_dispatch_checked", False):
            return
        self._dispatch_checked = True

        is_ep = self._bilevel_optimization_method == BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION

        if not is_ep and self.persist_outer_state:
            msg = (
                f"persist_outer_state=True is IGNORED by "
                f"{self._bilevel_optimization_method.name}: only EQUILIBRIUM_PROPAGATION threads "
                f"the outer optimizer state across epochs, so this run would be identical to "
                f"persist_outer_state=False and any A/B against it would compare two identical "
                f"arms. Either use EQUILIBRIUM_PROPAGATION, or -- if the intent is a scale-aware "
                f"outer optimizer rather than persistence -- use a stateless optimizer such as "
                f"optax.sgd, for which the per-epoch reset is a no-op by construction."
            )
            raise ValueError(msg)

        if not is_ep and self.ep_variant is not EPVariant.CENTERED:
            print(
                f"[vertax] WARNING: ep_variant={self.ep_variant.name} is ignored by "
                f"{self._bilevel_optimization_method.name} (EP-only attribute).",
                file=sys.stderr,
            )

        if is_ep and self.beta is _UNSET:
            print(
                f"[vertax] WARNING: beta was never set; EQUILIBRIUM_PROPAGATION is using the "
                f"default beta={BETA_DEFAULT}. Set it explicitly and record it with your results.",
                file=sys.stderr,
            )
        if self.beta is _UNSET:
            self.beta = BETA_DEFAULT

    def outer_optimization(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> None:
        """Optimize the mesh for the outer loss function.

        Args:
            mesh (Mesh): The mesh to act on.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.
        """
        self._check_attribute_dispatch()

        # To be defined by child classes
        if self._outer_opt_func is None:
            msg = "The outer function was not initialized."
            raise AttributeError(msg)
        else:
            self._outer_opt_func(mesh, *self._selection_to_jax_arrays(only_on_vertices, only_on_edges, only_on_faces))

        # Mean-value normalization by PARAMETER RENORMALISATION, applied after the outer step, for
        # every estimator path (this method is the single point they all pass through).
        #
        # Why renormalisation and not a penalty, and not a gradient projection:
        #
        # * A penalty `w*(mean(Lambda)-target)**2` is silently MIS-SCALED whenever the
        #   hypergradient is filtered or preconditioned. Truncating the inner solve at k steps
        #   multiplies the data gradient by the Landweber factor f_k, but `grad R(theta)` is a
        #   direct parameter-space gradient and is NOT filtered, so their ratio scales as 1/f_k --
        #   with no error and no warning, this can substantially degrade recovery quality at small
        #   k while leaving large k untouched. This is general to bilevel optimisation with
        #   truncated or preconditioned inner solves, not specific to VertAX. It also explains why
        #   a naive `w=1` penalty works for EP specifically: EP's hypergradient is unfiltered, so
        #   no rescaling correction is needed there -- but not for the other paths, whose
        #   hypergradient is filtered by the truncated inner solve.
        # * Projecting the GRADIENT does not fix the mean either, because Adam rescales
        #   per-coordinate (g/sqrt(v)) and, with the per-epoch state reset, reduces to
        #   -eta*sign(g); neither operation preserves a mean-zero update.
        #
        # Subtracting a constant is exact, optimiser-agnostic, and preserves twin symmetry
        # (both half-edges of a pair shift identically), so it cannot introduce an antisymmetric
        # component. Off by default: for E2, fixing `mean(Lambda)` is a normalization CONSTRAINT
        # and an ASSUMPTION supplied from outside the data -- a uniform shift DOES change the
        # force (`M 1 != 0`), so this is not a physically-free direction, merely the least
        # observable one. E1 shape factors are per-cell and observable, with no analogous
        # assumption needed.
        for name, target in self._resolve_param_normalization().items():
            arr = jnp.asarray(getattr(mesh, name))
            setattr(mesh, name, arr - (jnp.mean(arr) - target))

    _PARAM_ARRAYS = ("vertices_params", "edges_params", "faces_params")

    def _resolve_param_normalization(self) -> dict[str, float]:
        """Merge `params_normalization` and the older `mean_tension_gauge` into one mapping.

        `mean_tension_gauge` is kept working as an alias because every recorded E2 result was
        produced with it; it is exactly `{"edges_params": value}`. Setting both is rejected rather
        than silently resolved, because the two would disagree about `edges_params` and the loser
        would be decided by dict ordering.
        """
        spec = dict(self.params_normalization or {})
        if self.mean_tension_gauge is not None:
            if "edges_params" in spec:
                msg = ("Set either mean_tension_gauge or params_normalization['edges_params'], "
                       "not both -- they would disagree about the same array.")
                raise ValueError(msg)
            spec["edges_params"] = self.mean_tension_gauge
        bad = set(spec) - set(self._PARAM_ARRAYS)
        if bad:
            msg = (f"params_normalization keys must be among {self._PARAM_ARRAYS}; got "
                   f"{sorted(bad)}.")
            raise ValueError(msg)
        return spec

    def bilevel_optimization(
        self,
        mesh: Mesh,
        perform_energy_check: bool = False,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the loss functions given.

        Args:
            mesh (Mesh): The mesh to act on.
            perform_energy_check (bool = True): Allow T1 only if the energy is decreased by doing so.
                                                       Should be true if forward mode is used only.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: History of loss values during optimization.
        """
        self.outer_optimization(mesh, only_on_vertices, only_on_edges, only_on_faces)
        # Do not perform energy check in BO
        return self.inner_optimization(mesh, perform_energy_check, only_on_vertices, only_on_edges, only_on_faces)

    def do_n_bilevel_optimization(  # noqa: C901
        self,
        nb_epochs: int,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
        pre_inner_optimization: bool = False,
        report_every: int = 0,
        also_report_to_stdout: bool = False,
        save_plotmesh_every: int = 0,
        save_mesh_data_every: int = 0,
        save_folder: str = ".",
    ) -> None:
        """Optimize the mesh for the loss function given.

        Args:
            nb_epochs (int): The number of bilevel optimization steps to perform.
            mesh (Mesh): The mesh to act on.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.
            pre_inner_optimization (bool = False): Whether to perform an optional initial inner optimization first.
            report_every (int = 0): If strictly positive, report current state (costs, metrics, time) at this frequency.
            also_report_to_stdout (bool = False): Whether to also report to stdout (print).
            save_plotmesh_every (int = 0): If positive, save the plot of the mesh at this frequency.
            save_mesh_data_every (int = 0): If positive, save the mesh data at this frequency.
            save_folder: (str = "."): Base folder where to save data if needed.
        """
        save_folder_path = Path(save_folder)
        save_folder_path.mkdir(parents=True, exist_ok=True)
        mesh_save_plot_folder = save_folder_path / "meshes_plot"
        mesh_save_data_folder = save_folder_path / "meshes_data"
        outer_cost_graph_filename = save_folder_path / "outer_cost_over_time.png"
        inner_cost_graph_filename = save_folder_path / "inner_cost_over_time.png"
        summary_filename = save_folder_path / "summary.csv"
        parameters_filename = save_folder_path / "hyper-parameters.txt"

        with parameters_filename.open("w") as f:
            f.write(self.self_summary())

        all_epoch_data = []
        epoch_data = []
        if save_plotmesh_every > 0:
            mesh_save_plot_folder.mkdir(parents=True, exist_ok=True)
            fig, _ = plot_mesh(mesh, show=False, save=True, save_path=str(mesh_save_plot_folder / "mesh_epoch_0.png"))
            plt.close(fig)
        if save_mesh_data_every > 0:
            mesh_save_data_folder.mkdir(parents=True, exist_ok=True)
            mesh.save_mesh(path=str(mesh_save_data_folder / "mesh_epoch_0.npz"))

        x_epochs: list[float] = []
        y_outer_costs: list[float] = []
        y_inner_costs: list[float] = []
        if report_every > 0:
            epoch_data = [0]
            outer_cost = self.compute_outer_loss(mesh, only_on_vertices, only_on_edges, only_on_faces)
            inner_cost = self.compute_inner_loss(mesh)
            x_epochs.append(0)
            y_outer_costs.append(outer_cost)
            y_inner_costs.append(inner_cost)
            epoch_data.extend([outer_cost, inner_cost, 0, 0])  # the zeros are time and delta time
            # Reset and put initial metrics
            for metric_name in self.custom_metrics:
                self.custom_metrics[metric_name][1].clear()
                self.custom_metrics[metric_name][2].clear()
                self.custom_metrics[metric_name][1].append(0.0)
                metric = self.custom_metrics[metric_name][0](mesh, self)
                self.custom_metrics[metric_name][2].append(metric)
                epoch_data.append(metric)
            all_epoch_data.append(epoch_data)
            pandas.DataFrame(
                all_epoch_data,
                columns=[
                    "Epoch",
                    "Outer cost",
                    "Inner cost",
                    "Time (s)",
                    "Delta Time (s)",
                    *(metric_name for metric_name in self.custom_metrics),
                ],  # ty:ignore[invalid-argument-type]
            ).to_csv(summary_filename)

            if also_report_to_stdout:
                msg = "First epoch may be a bit long."
                msg += f" Initial state: Outer cost = {outer_cost}, Inner cost = {inner_cost}"
                for metric_name in self.custom_metrics:
                    msg += f", {metric_name} = {self.custom_metrics[metric_name][2][-1]}"
                print(msg)

        if pre_inner_optimization:
            self.inner_optimization(mesh, False, only_on_vertices, only_on_edges, only_on_faces)

        t_initial = perf_counter()
        for epoch in range(1, nb_epochs + 1):
            epoch_data = [epoch]
            t_begin_of_epoch = perf_counter()

            self.bilevel_optimization(mesh, False, only_on_vertices, only_on_edges, only_on_faces)
            t_end_of_epoch = perf_counter()
            total_time = t_end_of_epoch - t_initial
            delta_time = t_end_of_epoch - t_begin_of_epoch

            if save_plotmesh_every > 0 and epoch % save_plotmesh_every == 0:
                fig, _ = plot_mesh(
                    mesh, show=False, save=True, save_path=str(mesh_save_plot_folder / f"mesh_epoch_{epoch}.png")
                )
                plt.close(fig)
            if save_mesh_data_every > 0 and epoch % save_mesh_data_every == 0:
                mesh.save_mesh(path=str(mesh_save_data_folder / f"mesh_epoch_{epoch}.npz"))

            if report_every > 0 and epoch % report_every == 0:
                # Compute results data for this step
                outer_cost = self.compute_outer_loss(mesh, only_on_vertices, only_on_edges, only_on_faces)
                inner_cost = self.compute_inner_loss(mesh)
                x_epochs.append(epoch)
                y_outer_costs.append(outer_cost)
                y_inner_costs.append(inner_cost)
                epoch_data.extend([outer_cost, inner_cost, total_time, delta_time])  # the zeros are time and delta time
                for metric_name in self.custom_metrics:
                    self.custom_metrics[metric_name][1].append(epoch)
                    metric = self.custom_metrics[metric_name][0](mesh, self)
                    self.custom_metrics[metric_name][2].append(metric)
                    epoch_data.append(metric)

                all_epoch_data.append(epoch_data)
                pandas.DataFrame(
                    all_epoch_data,
                    columns=[
                        "Epoch",
                        "Outer cost",
                        "Inner cost",
                        "Time (s)",
                        "Delta Time (s)",
                        *(metric_name for metric_name in self.custom_metrics),
                    ],  # ty:ignore[invalid-argument-type]
                ).to_csv(summary_filename)

                # Report in log.
                if also_report_to_stdout:
                    msg = f"\nEpoch {epoch}/{nb_epochs} : Outer cost = {outer_cost}, Inner cost = {inner_cost}"
                    for metric_name in self.custom_metrics:
                        msg += f", {metric_name} = {self.custom_metrics[metric_name][2][-1]}"
                    msg += f", Time = {total_time}s (+{delta_time}s)"
                    print(msg)

                # Update graphs
                save_simple_xy_graph(
                    str(outer_cost_graph_filename),
                    x_epochs,
                    y_outer_costs,
                    "Outer cost over time",
                    "Epoch",
                    "Outer cost",
                )
                save_simple_xy_graph(
                    str(inner_cost_graph_filename),
                    x_epochs,
                    y_inner_costs,
                    "Inner cost over time",
                    "Epoch",
                    "Inner cost",
                )
                for metric_name in self.custom_metrics:
                    graph_name = save_folder_path / (metric_name + ".png")
                    save_simple_xy_graph(
                        str(graph_name),
                        self.custom_metrics[metric_name][1],
                        self.custom_metrics[metric_name][2],
                        f"{metric_name} over time",
                        "Epoch",
                        metric_name,
                    )

    def self_summary(self) -> str:
        """Give a summary of hyper-parameters of the optimizer."""
        summary = f"Bi-level optimization method: {self.bilevel_optimization_method.value}\n"
        if self.loss_function_inner is not None:
            summary += f"Inner loss function: {self.loss_function_inner.__name__}\n"  # ty:ignore[unresolved-attribute]
        else:
            summary += "Inner loss function: None\n"
        if self.loss_function_outer is not None:
            summary += f"Outer loss function: {self.loss_function_outer.__name__}\n"  # ty:ignore[unresolved-attribute]
        else:
            summary += "Outer loss function: None\n"
        summary += f"Max number of iterations: {self.max_nb_iterations}\n"
        summary += f"Tolerance: {self.tolerance}\n"
        summary += f"Patience: {self.patience}\n"
        summary += f"Minimum distance for T1: {self.min_dist_T1}\n"
        _cap = "auto" if self.cap_vertex_step is None else str(self.cap_vertex_step)
        summary += f"Cap vertex step: {_cap} (-> {self._resolve_cap_vertex_step()})\n"
        summary += f"Update T1: {self.update_T1}\n"
        if self._update_T1_func is not None:
            summary += f"Update T1 function: {self._update_T1_func.__name__}\n"  # ty:ignore[unresolved-attribute]
        else:
            summary += "Update T1 function: None\n"
        summary += f"Beta: {self.beta}"
        return summary

    def add_custom_metric(self, name: str, function: Callable[[Any, Any], float]) -> None:
        """Add a custom metric to the metrics to save when performing n bilevel optimizations.

        The function must take two arguments : a mesh and a bilevel optimizer.
        """
        self.custom_metrics[name] = (function, [], [])

    def remove_custom_metric(self, name: str) -> None:
        """Remove a custom metric from the metrics to save when performing n bilevel optimizations."""
        if name in self.custom_metrics:
            del self.custom_metrics[name]

    def _selection_to_jax_arrays(
        self,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> tuple[Array | None, Array | None, Array | None]:
        selected_vertices, selected_edges, selected_faces = None, None, None
        if only_on_vertices is not None:
            selected_vertices = jnp.array(only_on_vertices)
        if only_on_edges is not None:
            selected_edges = jnp.array(only_on_edges)
        if only_on_faces is not None:
            selected_faces = jnp.array(only_on_faces)
        return selected_vertices, selected_edges, selected_faces


def _reject_bounded_t1_repair(mesh: Any) -> None:  # noqa: ANN401
    """Refuse to run the periodic T1 swap repair on a bounded mesh.

    ``_BilevelOptimizer.inner_optimization`` is shared by both mesh types and
    ``BoundedBilevelOptimizer`` does not override it, so a bounded mesh with a registered vertex
    target would silently execute the *periodic* repair. That repair is wrong for bounded meshes
    in two independent ways:

    * it detects T1 events as ``heTable[:, 5]`` changing. Column 5 is the **face index** for a
      periodic mesh, but for a bounded mesh it is a **boundary-edge source-vertex index** (see
      ``geo.get_chord_length``); the bounded face index lives in column 7. The detector therefore
      keys on the wrong quantity entirely.
    * it scores candidate swaps with ``_periodic_sq_dist``, i.e. the minimum over the nine
      periodic images. Bounded meshes have no periodic images, and their ``width``/``height`` are
      a bounding box rather than a period.

    This is currently unreachable -- no bounded run registers ``vertices_target``, because bounded
    work is inverse *design* (elongation, salt-and-pepper fates), which has no target mesh to match
    and hence no vertex index-correspondence problem for the repair to solve. This guard exists so
    that the first run which does register one fails loudly instead of returning a wrong answer.

    The fix (not implemented) is to dispatch on mesh type: column 7 plus a plain squared distance
    for bounded, column 5 plus ``geo.min_image_sq_dist`` for periodic -- preferably one function
    taking a mesh-type parameter rather than two copies.
    """
    from vertax.meshes.bounded_mesh import BoundedMesh

    if isinstance(mesh, BoundedMesh):
        msg = (
            "The T1 swap repair is not implemented for bounded meshes, and the shared periodic "
            "implementation is incorrect for them: it detects T1 events via heTable[:, 5], which "
            "is the face index for a periodic mesh but a boundary-edge vertex index for a bounded "
            "one (bounded stores the face index in column 7), and it scores swaps with periodic "
            "minimum-image distances that a bounded mesh does not have. Refusing rather than "
            "silently returning a wrong permutation. Either leave vertices_target unset (bounded "
            "runs are inverse design and do not need it) or implement the mesh-type dispatch "
            "described in the docstring of _reject_bounded_t1_repair."
        )
        raise NotImplementedError(msg)


def _periodic_sq_dist(p: Array, q: Array, width: float, height: float) -> Array:
    """Squared distance under PBC between two 2D points (cols 0,1).

    Mirrors `cost_v2v`'s convention: take the minimum over the 9 image shifts
    (center + 8 surrounding sectors) of the target.
    """
    # TODO : abstract, make a PBC and bounded version
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    candidates = jnp.array(
        [
            dx * dx + dy * dy,
            (dx - width) * (dx - width) + dy * dy,
            (dx + width) * (dx + width) + dy * dy,
            dx * dx + (dy - height) * (dy - height),
            dx * dx + (dy + height) * (dy + height),
            (dx - width) * (dx - width) + (dy - height) * (dy - height),
            (dx - width) * (dx - width) + (dy + height) * (dy + height),
            (dx + width) * (dx + width) + (dy - height) * (dy - height),
            (dx + width) * (dx + width) + (dy + height) * (dy + height),
        ]
    )
    return jnp.min(candidates)


def _build_t1_repair_perm(
    vertTable_after: Array,
    vertTable_target: Array,
    heTable_before: Array,
    heTable_after: Array,
    width: float,
    height: float,
) -> Array:
    """Build a vertex-index permutation that swaps T1-flipped pairs when it lowers v2v cost.

    A half-edge whose face index (col 5) changed across `inner_opt` was part of a T1: only
    `update_T1` ever writes that column (`update_pbc` does not). For each such half-edge,
    cols (3, 4) give the (a, b) vertex pair to test. Each T1 is reported twice (edge + twin);
    the swap-only-if-better filter is idempotent, so duplicates are harmless.
    """
    perm = jnp.arange(vertTable_after.shape[0], dtype=jnp.int32)

    t1_mask = heTable_before[:, 5] != heTable_after[:, 5]
    # jax.debug.print("t1_mask = {x}", x=t1_mask.any(), ordered=True)
    pair_a = heTable_before[:, 3].astype(jnp.int32)
    pair_b = heTable_before[:, 4].astype(jnp.int32)

    def body(i: int, perm: Array) -> Array:
        a = pair_a[i]
        b = pair_b[i]
        ta = vertTable_target[perm[a]]
        tb = vertTable_target[perm[b]]
        pa = vertTable_after[a]
        pb = vertTable_after[b]
        cost_no_swap = _periodic_sq_dist(pa, ta, width, height) + _periodic_sq_dist(pb, tb, width, height)
        cost_swap = _periodic_sq_dist(pa, tb, width, height) + _periodic_sq_dist(pb, ta, width, height)
        do_swap = t1_mask[i] & (cost_swap < cost_no_swap)
        new_a_val = jnp.where(do_swap, perm[b], perm[a])
        new_b_val = jnp.where(do_swap, perm[a], perm[b])
        return perm.at[a].set(new_a_val).at[b].set(new_b_val)

    return lax.fori_loop(0, heTable_before.shape[0], body, perm)


def _apply_perm_to_state(
    perm: Array,
    vertTable: Array,
    heTable: Array,
) -> tuple[Array, Array]:
    """Reorder vertTable rows and relabel vertex indices stored in heTable cols 3, 4.

    heTable cols 0, 1, 2, 5 reference half-edges/faces and are unaffected.
    Self-canceling for any cost that reads positions by half-edge (only cost_v2v sees the swap).
    """
    new_vertTable = vertTable[perm]
    src = perm[heTable[:, 3].astype(jnp.int32)].astype(heTable.dtype)
    tgt = perm[heTable[:, 4].astype(jnp.int32)].astype(heTable.dtype)
    new_heTable = heTable.at[:, 3].set(src).at[:, 4].set(tgt)
    return new_vertTable, new_heTable


def _has_zoom_linesearch_state(solver: optax.GradientTransformation) -> bool:
    """True if `solver`'s state contains a zoom (Wolfe) line-search sub-state.

    Every line-search solver — optax.lbfgs as well as cg_linesearch /
    gd_linesearch — chains in ``optax.scale_by_zoom_linesearch``, whose state is a
    ``ScaleByZoomLinesearchState``. That is the reliable signature: optax.lbfgs has
    a generic repr with no "linesearch" marker in its name, and its own
    ``ScaleByLBFGSState`` is buried alongside other sub-states, so we look for the
    line-search state itself rather than a specific solver class.

    The state is only realised by ``init``, so we build it on a dummy scalar
    parameter. ``init`` is pure and cheap; any failure (a solver that rejects the
    probe) simply means "not a line-search solver" for our purposes.
    """
    try:
        state = solver.init(jnp.zeros(()))
    except Exception:
        return False
    substates = state if isinstance(state, tuple) else (state,)
    return any(s.__class__.__name__ == "ScaleByZoomLinesearchState" for s in substates)


def _is_ad_incompatible_solver(solver: optax.GradientTransformation) -> bool:
    """True if `solver` cannot be reverse-mode differentiated through.

    Line-search inner solvers run an inner ``lax.while_loop`` with data-dependent
    bounds (the Wolfe/zoom search), which reverse-mode AD cannot differentiate
    through — so they are incompatible with the AUTOMATIC_DIFFERENTIATION bilevel
    method. Two detection paths, because no single one covers every solver:

    - ``cg_linesearch``, ``gd_linesearch`` (and future ones) carry a "linesearch"
      marker in their factory name, so a cheap name check catches them.
    - ``optax.lbfgs`` has a generic repr with no such marker, so it is identified by
      the ``ScaleByZoomLinesearchState`` in its optimiser state.
    """
    if "linesearch" in str(solver):
        return True
    return _has_zoom_linesearch_state(solver)
