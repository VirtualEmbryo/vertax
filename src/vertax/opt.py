"""Optimization methods for VertAX."""

import os
from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, grad, jacfwd, jit, jvp, lax
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator, minres

import contextlib

from vertax.geo import get_length, update_pbc
from vertax.method_enum import BilevelOptimizationMethod, EPVariant
from vertax.topo import update_T1

# Per-step cap on how far a single vertex may move during inner relaxation,
# expressed as a MULTIPLE of min_dist_T1 rather than as an absolute length.
#
# The incremental T1 detector fires when an edge shrinks below min_dist_T1, and it
# assumes vertices move by comparable amounts each iteration. Line-search solvers
# (cg_linesearch, gd_linesearch, lbfgs) choose their own step size and can take
# steps of order 1, letting a vertex jump across whole cells in one iteration and
# producing tangled arcs / spurious-looking cells.
#
# Tying the cap to min_dist_T1 makes it scale-free: the same setting behaves the
# same on a unit-sized mesh and on one a hundred times larger, whereas an absolute
# cap would throttle the big mesh and do nothing on the small one. The factor is
# how many T1 thresholds a vertex may cross per iteration; 5 keeps steps on the
# detector's scale while leaving the solvers room to make progress.
#
# Under PBC a vertex leaving the box is wrapped back inside by update_pbc, so an
# uncapped step is not merely large but ambiguous: the wrap makes a jump across
# the box indistinguishable from a short move the other way. Capping the step
# keeps displacements well below the box size, where the wrap stays unambiguous.
#
# Set VERTAX_MAX_VERTEX_STEP_FACTOR to override; a large value effectively
# disables the cap and restores the previous unbounded behaviour. This mirrors
# MAX_VERTEX_STEP_FACTOR in opt_bounded, and reads the same variable so a single
# setting governs both mesh types.
MAX_VERTEX_STEP_FACTOR = float(os.environ.get("VERTAX_MAX_VERTEX_STEP_FACTOR", "5.0"))


class OptimizationTarget(Enum):
    """What the minimize function will try to optimize."""

    VERTICES = 0
    EDGES = 1
    FACES = 2
    VERTEX_PARAMETERS = 3
    EDGE_PARAMETERS = 4
    FACE_PARAMETERS = 5


InnerLossFunction = Callable[[Array, Array, Array, Array, Array, Array], Array]
OuterLossFunction = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ],
    Array,
]
LossEPFunction = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        InnerLossFunction,
        OuterLossFunction,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        float,
    ],
    Array,
]

UpdateT1Func = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        InnerLossFunction,
        float,
        Array,
        Array,
        Array,
        bool,
    ],
    tuple[Array, Array, Array],
]


# ---- T1 statistics ----------------------------------------------------------------------------------
# Counts of T1 decisions are carried in the loop state of ``_jit_minimize`` (``count_t1=True``), so they
# are exact under ``jit``, ``vmap`` and ``jax.grad``. ``recording_t1_statistics`` collects them from every
# relaxation the library runs inside the bilevel estimators, including the relaxation that automatic
# differentiation unrolls (returned there as an auxiliary output of the differentiated cost).
_T1_RECORDER: list | None = None


@contextlib.contextmanager
def recording_t1_statistics():  # noqa: ANN201
    """Collect the per-iteration T1 counts of every relaxation run inside the block.

    Yields a list to which one record is appended per relaxation, in call order: a dict with
    ``counts``, an integer array of shape ``(iterations, 2)`` holding, per iteration, the edges presented
    to the T1 detector (no longer than ``min_dist_T1``, neither cell a triangle) and the flips kept, and
    ``source``, the library function that ran the relaxation. With T1 transitions enabled, rejected
    flips are presented minus kept, provided the flips of one iteration share no cell (see
    ``minimize``, ``count_t1``). Relaxations traced inside a caller's own ``jit``/``vmap``/``grad``
    cannot be recorded this way: call ``minimize(..., count_t1=True)`` there and return the counts.
    """
    global _T1_RECORDER  # noqa: PLW0603
    previous, _T1_RECORDER = _T1_RECORDER, []
    try:
        yield _T1_RECORDER
    finally:
        _T1_RECORDER = previous


def _record_t1(counts: Array, steps: Array, source: str) -> None:
    if _T1_RECORDER is None or isinstance(counts, jax.core.Tracer) or isinstance(steps, jax.core.Tracer):
        return
    _T1_RECORDER.append(dict(source=source, counts=np.asarray(counts)[: int(steps)]))


def _with_incoming_integer_dtypes(new_state, old_state):  # noqa: ANN001, ANN202
    """Cast an integer leaf of ``new_state`` to the dtype of the same leaf of ``old_state`` where they differ.

    Under ``jax_enable_x64`` the zoom line search of ``optax.lbfgs()`` returns an int64 step counter
    where it was initialised with int32, and ``lax.cond`` in the loop requires identical types. The
    comparison is made at trace time, so a solver whose dtypes already match gets no extra operation.
    """
    def cast(new, old):  # noqa: ANN001, ANN202
        if hasattr(new, "dtype") and jnp.issubdtype(new.dtype, jnp.integer) and new.dtype != old.dtype:
            return new.astype(old.dtype)
        return new

    return jax.tree.map(cast, new_state, old_state)


###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


@partial(jit, static_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22))
def _jit_minimize(  # noqa: C901
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    selected_verts: Array,
    selected_hes: Array,
    selected_faces: Array,
    width: float,
    height: float,
    L_in: InnerLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    optimization_target: OptimizationTarget = OptimizationTarget.VERTICES,
    update_t1_func: UpdateT1Func = update_T1,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
    reset_solver_on_t1: bool = False,
    count_t1: bool = False,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], tuple[Array, ...]]:
    x0: Array = [vertTable, heTable, faceTable, vert_params, he_params, face_params][optimization_target.value]
    sel: Array = [
        selected_verts,
        selected_hes,
        selected_faces,
        selected_verts,
        selected_hes,
        selected_faces,
    ][optimization_target.value]
    argnums = optimization_target.value
    opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = L_in(
        vertTable[selected_verts],
        heTable[selected_hes],
        faceTable[selected_faces],
        vert_params[selected_verts],
        he_params[selected_hes],
        face_params[selected_faces],
    )
    L_in_list = jnp.zeros((iterations_max,)).at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)

    # Define the inner step function for lax.scan
    # The inner function is defined *without* @jit because it's compiled by lax.scan/jit on minimize
    def scan_step(  # noqa: C901
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], i: int
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], None]:
        # Unpack the carry structure:
        (
            vt,
            ht,
            ft,
            vp,
            hp,
            fp,
            opt_state,
            prev_L_values,
            stagnation_count,
            step_count,
            should_stop,
            L_list,
        ) = carry[:12]

        # Determine if we are still running (i.e., not stopped)
        is_running = jnp.logical_not(should_stop)
        # Compute loss
        L_current = L_in(
            vt[selected_verts],
            ht[selected_hes],
            ft[selected_faces],
            vp[selected_verts],
            hp[selected_hes],
            fp[selected_faces],
        )

        # Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)

        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)

        # Gradient wrt chosen argnums
        grads = grad(L_in, argnums=optimization_target.value)(vt, ht, ft, vp, hp, fp)

        # value_fn re-evaluates L_in as a function of the solver's params only
        # (everything else held fixed at the current step). Line-search solvers
        # (cg_linesearch, sgd_linesearch, lbfgs) need it; adam/sgd ignore the extra arguments.
        # The branches follow OptimizationTarget: 0 vertices, 1 edges, 2 faces, 3 to 5 their parameters.
        if argnums == 0:
            params_cur = vt

            def value_fn(p: Array) -> Array:
                return L_in(
                    p[selected_verts],
                    ht[selected_hes],
                    ft[selected_faces],
                    vp[selected_verts],
                    hp[selected_hes],
                    fp[selected_faces],
                )
        elif argnums == 1:
            params_cur = ht

            def value_fn(p: Array) -> Array:
                return L_in(
                    vt[selected_verts],
                    p[selected_hes],
                    ft[selected_faces],
                    vp[selected_verts],
                    hp[selected_hes],
                    fp[selected_faces],
                )
        elif argnums == 2:
            params_cur = ft

            def value_fn(p: Array) -> Array:
                return L_in(
                    vt[selected_verts],
                    ht[selected_hes],
                    p[selected_faces],
                    vp[selected_verts],
                    hp[selected_hes],
                    fp[selected_faces],
                )
        elif argnums == 3:
            params_cur = vp

            def value_fn(p: Array) -> Array:
                return L_in(
                    vt[selected_verts],
                    ht[selected_hes],
                    ft[selected_faces],
                    p[selected_verts],
                    hp[selected_hes],
                    fp[selected_faces],
                )
        elif argnums == 4:
            params_cur = hp

            def value_fn(p: Array) -> Array:
                return L_in(
                    vt[selected_verts],
                    ht[selected_hes],
                    ft[selected_faces],
                    vp[selected_verts],
                    p[selected_hes],
                    fp[selected_faces],
                )
        else:  # argnums == 5
            params_cur = fp

            def value_fn(p: Array) -> Array:
                return L_in(
                    vt[selected_verts],
                    ht[selected_hes],
                    ft[selected_faces],
                    vp[selected_verts],
                    hp[selected_hes],
                    p[selected_faces],
                )

        # Optimizer update. Forward the line-search args unconditionally:
        # first-order solvers (adam/sgd) ignore them, line-search solvers use them.
        updates: Array
        new_opt_state: Array
        updates, new_opt_state = solver.update(  # ty:ignore[invalid-assignment]
            grads,
            opt_state,
            params_cur,
            value=L_current,  # ty:ignore[unknown-argument]
            grad=grads,  # ty:ignore[unknown-argument]
            value_fn=value_fn,  # ty:ignore[unknown-argument]
        )
        new_opt_state = _with_incoming_integer_dtypes(new_opt_state, opt_state)

        # Apply updates to the chosen array on selected indices
        updates_sel = updates.at[sel].get()

        # Cap the per-step vertex move so large line-search steps (cg_linesearch,
        # gd_linesearch) stay on the scale the incremental T1 detector expects.
        # The cap is relative to min_dist_T1, so it follows the mesh's own scale.
        #
        # This clips PER VERTEX: each vertex keeps its own direction, but a vertex
        # already under the cap is left alone while a faster one is pulled back to
        # exactly max_step. Relative magnitudes between vertices are therefore not
        # preserved -- steps of 0.03 and 3.0 both come out at the cap -- so the
        # descent direction of the mesh as a whole IS altered. This is a local
        # geometric guard, not a line-search safeguard: the goal is to stop any
        # single vertex from crossing whole cells and breaking the incremental T1
        # detector, which is a per-vertex condition.
        #
        # Scaling every vertex by one shared factor would preserve the global
        # descent direction, but a single fast vertex would then throttle the whole
        # mesh. That trade was decided in favour of per-vertex clipping.
        #
        # Vertices only: the other targets (half-edge / face parameters) are not
        # displacements, so a length-scale cap would be meaningless for them.
        # Columns 0/1 of vertTable are x/y, as in the bounded case, so the norm
        # is taken over axis 1.
        #
        # Opt-in via cap_vertex_step: only line-search solvers (cg_linesearch,
        # gd_linesearch, lbfgs) pick their own step size and need it. Applying it
        # unconditionally would silently alter fixed-step solvers whenever the cap
        # falls below their step -- sgd(lr=0.01) with min_dist_T1=0.005 gives a cap
        # of 0.025, which does bite and shifts the result.
        if argnums == 0 and cap_vertex_step:
            max_step = MAX_VERTEX_STEP_FACTOR * min_dist_T1
            vnorm = jnp.linalg.norm(updates_sel, axis=1, keepdims=True)
            vscale = jnp.minimum(1.0, max_step / jnp.maximum(vnorm, 1e-12))
            updates_sel = updates_sel * vscale

        # Conditional application of optimization update and state
        arrays = [vt, ht, ft, vp, hp, fp]
        k = optimization_target.value
        new_optim = arrays[k].at[sel].set(arrays[k][sel] + updates_sel)
        arrays[k] = lax.cond(is_running, lambda: new_optim, lambda: arrays[k])
        vt, ht, ft, vp, hp, fp = arrays

        # new_vt_optim = vt.at[sel].set(vt[sel] + updates_sel)
        # vt = lax.cond(is_running, lambda: new_vt_optim, lambda: vt)

        opt_state = lax.cond(is_running, lambda: new_opt_state, lambda: opt_state)
        stagnation_count = lax.cond(is_running, lambda: new_stagnation_count, lambda: stagnation_count)
        should_stop = new_should_stop  # should_stop is a flag for the *next* iteration

        # Geometry updates (must also be conditional/masked)
        new_vt_pbc, new_ht_pbc, new_ft_pbc = update_pbc(vt, ht, ft, width, height)
        vt = lax.cond(is_running, lambda: new_vt_pbc, lambda: vt)
        ht = lax.cond(is_running, lambda: new_ht_pbc, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_pbc, lambda: ft)

        ht_before_t1 = ht
        if count_t1:
            # Edges presented to the T1 detector at this iteration, measured on the tables it receives:
            # one half-edge per edge, no longer than min_dist_T1, neither of its two cells a triangle.
            even = jnp.arange(0, ht.shape[0], 2)
            he_even, twin_even = ht[even], ht[ht[even, 2]]
            lengths = jax.vmap(lambda i: get_length(i, vt, ht, ft, width, height))(even)
            admissible = (ht[he_even[:, 0], 0] != he_even[:, 1]) & (ht[twin_even[:, 0], 0] != twin_even[:, 1])
            n_presented = jnp.sum((lengths <= min_dist_T1) & admissible).astype(jnp.int32)
        new_vt_T1, new_ht_T1, new_ft_T1 = update_t1_func(
            vt,
            ht,
            ft,
            width,
            height,
            vp,
            hp,
            fp,
            L_in,
            min_dist_T1,
            selected_verts,
            selected_hes,
            selected_faces,
            perform_energy_check,
        )
        vt = lax.cond(is_running, lambda: new_vt_T1, lambda: vt)
        ht = lax.cond(is_running, lambda: new_ht_T1, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_T1, lambda: ft)

        if count_t1:
            # A kept flip re-links both half-edges of the flipped edge (previous and next both change);
            # the neighbouring half-edges change one link each.
            n_kept = jnp.sum((ht[even, 0] != ht_before_t1[even, 0]) & (ht[even, 1] != ht_before_t1[even, 1]))
            t1_counts = carry[12].at[i].set(
                jnp.where(is_running, jnp.stack([n_presented, n_kept.astype(jnp.int32)]), 0))

        # A T1 transition changes the mesh the solver state describes: CG's conjugate direction and
        # the curvature pairs of L-BFGS refer to the old topology and vertex positions. Opt-in reset
        # of the solver state to its initial value at the new positions whenever the topology has
        # changed; integer and float leaves keep their dtypes so that the state type is stable.
        if reset_solver_on_t1 and argnums == 0:
            topology_changed = is_running & jnp.any(ht != ht_before_t1)
            fresh = jax.tree.map(lambda new, old: new.astype(old.dtype), solver.init(vt), opt_state)
            opt_state = lax.cond(topology_changed, lambda: fresh, lambda: opt_state)

        # Shift prev_L_values
        new_prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        new_prev_L_values = new_prev_L_values.at[0].set(L_current)
        prev_L_values = lax.cond(is_running, lambda: new_prev_L_values, lambda: prev_L_values)

        # Update history and step count
        L_list = L_list.at[i].set(L_current)
        step_count = i + 1

        new_carry = (
            vt,
            ht,
            ft,
            vp,
            hp,
            fp,
            opt_state,
            prev_L_values,
            stagnation_count,
            step_count,
            should_stop,
            L_list,
        )
        if count_t1:
            new_carry = (*new_carry, t1_counts)
        return new_carry, None

    # Call lax.scan
    init_carry = (
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        opt_state,
        prev_L_values,
        stagnation_count,
        step_count,
        should_stop,
        L_in_list,
    )

    if count_t1:
        init_carry = (*init_carry, jnp.zeros((iterations_max, 2), dtype=jnp.int32))

    final_state, _ = lax.scan(scan_step, init_carry, xs=jnp.arange(iterations_max))  # ty:ignore[invalid-argument-type]

    # Unpack and return results (for slicing outside jit)
    vt_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state[:12]
    if count_t1:
        return (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f, final_state[12])  # ty:ignore[invalid-return-type]
    return (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f)  # ty:ignore[invalid-return-type]


def minimize(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    optimization_target: OptimizationTarget = OptimizationTarget.VERTICES,
    update_t1_func: UpdateT1Func = update_T1,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
    reset_solver_on_t1: bool = False,
    count_t1: bool = False,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], tuple[Array, ...]]:
    """General minimization function.

    Supported solvers: any ``optax.GradientTransformation``. Fixed-step solvers (``optax.sgd``,
    ``optax.adam``) use the gradient only. Line-search solvers -- ``vertax.cg.cg_linesearch``
    (nonlinear conjugate gradient), ``vertax.cg.gd_linesearch`` (steepest descent with a Wolfe line
    search) and ``optax.lbfgs()`` -- also need the energy and a callable to re-evaluate it, which
    this loop passes to ``solver.update`` as ``value``, ``grad`` and ``value_fn`` at every step. Use
    ``cap_vertex_step=True`` with them, and ``reset_solver_on_t1=True`` with ``optax.lbfgs()``,
    whose curvature pairs otherwise outlive the topology they were measured on. A solver whose state
    holds integers needs a fixed integer dtype under ``jax_enable_x64`` (``cg_linesearch`` and
    ``gd_linesearch`` do; ``optax.lbfgs()`` does).

    Build the solver once and pass the same object to every call. The relaxation loop is compiled
    for each solver object (the solver is a static argument of the compiled loop, compared by
    identity), so a new solver at every call, e.g. ``minimize(..., optax.sgd(0.01), ...)`` in a loop,
    recompiles at every call and keeps every compiled copy in the jit cache. The same holds for
    ``L_in`` and ``update_t1_func``, and for the solvers given to the bilevel optimizers. Measured on
    a 20-cell periodic mesh, 50 iterations, CPU: 5.7 s for the first call, then 0.15 s per call with
    the same solver; 4.7 to 5.0 s per call with a new ``optax.sgd`` each time, the cache growing by
    one entry per call.

    Args:
        cap_vertex_step: Clip each per-step vertex displacement to
            MAX_VERTEX_STEP_FACTOR * min_dist_T1. Intended for line-search solvers
            (cg_linesearch, gd_linesearch, lbfgs), which choose their own step size
            and can otherwise move a vertex across whole cells in one iteration.
            Off by default, so fixed-step solvers (adam, sgd) are untouched.
        reset_solver_on_t1: Re-initialise the solver state at the new vertex positions after
            every iteration whose T1 step changed the topology (vertex optimisation only). Off by
            default.
        count_t1: Also return, as a third element of the history, an integer array of shape
            ``(iterations_max, 2)``: per iteration, the edges presented to the T1 detector and the
            flips kept (zero after the loop has stopped). Carried in the loop state, so exact under
            ``jit``, ``vmap`` and ``jax.grad``; the relaxation itself is unchanged. Off by default.
            "Presented" is exact. "Kept" is a count of edges whose two half-edges both had their
            previous and next links changed, which equals the number of flips only when the flips of one
            iteration share no cell: where neighbouring flips share a cell (possible under the forced
            rule, which flips every presented edge), a presented edge that was not itself flipped can
            have both links changed by its neighbours and be counted, so "kept" can differ from the
            flips performed and rejected flips (presented minus kept) are then unreliable.
    """
    # ensure width and height are hashable...
    width = float(width)
    height = float(height)
    iterations_max = int(iterations_max)
    patience = int(patience)
    min_dist_T1 = float(min_dist_T1)
    tolerance = float(tolerance)

    selected_verts = jnp.arange(vertTable.shape[0]) if selected_verts is None else jnp.array(selected_verts)
    selected_hes = jnp.arange(heTable.shape[0]) if selected_hes is None else jnp.array(selected_hes)
    selected_faces = jnp.arange(faceTable.shape[0]) if selected_faces is None else jnp.array(selected_faces)

    iterations_max = int(iterations_max)
    patience = int(patience)

    result = jax.block_until_ready(
        _jit_minimize(
            vertTable=vertTable,
            heTable=heTable,
            faceTable=faceTable,
            vert_params=vert_params,
            he_params=he_params,
            face_params=face_params,
            selected_verts=selected_verts,
            selected_hes=selected_hes,
            selected_faces=selected_faces,
            width=width,
            height=height,
            L_in=L_in,
            solver=solver,
            min_dist_T1=min_dist_T1,
            iterations_max=iterations_max,
            tolerance=tolerance,
            patience=patience,
            optimization_target=optimization_target,
            update_t1_func=update_t1_func,
            perform_energy_check=perform_energy_check,
            cap_vertex_step=cap_vertex_step,
            reset_solver_on_t1=reset_solver_on_t1,
            count_t1=count_t1 or _T1_RECORDER is not None,
        )
    )
    if count_t1 or _T1_RECORDER is not None:
        tables, (hist, steps, counts) = result
        _record_t1(counts, steps, "minimize")
        return (tables, (hist, steps, counts)) if count_t1 else (tables, (hist, steps))
    return result


def inner_opt(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    update_t1_func: UpdateT1Func = update_T1,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
) -> tuple[tuple[Array, Array, Array], Array]:
    """Inner optimization function.

    Args:
        cap_vertex_step: Clip each per-step vertex displacement to
            MAX_VERTEX_STEP_FACTOR * min_dist_T1, as in `minimize`. Forwarded unchanged.
    """
    # Use the general minimize function with VERTICES (optimize vertTable)
    (vt_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = minimize(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver,
        min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        optimization_target=OptimizationTarget.VERTICES,
        update_t1_func=update_t1_func,
        perform_energy_check=perform_energy_check,
        cap_vertex_step=cap_vertex_step,
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()

    # Slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]

    # Return updated arrays and loss history
    return (vt_f, ht_f, ft_f), final_L_list


def cost_ad(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    image_target: Array | None = None,
    update_t1_func: UpdateT1Func = update_T1,
    return_t1_counts: bool = False,
    cap_vertex_step: bool = False,
) -> Array:
    """Automatic differentiation cost function.

    With ``return_t1_counts`` it returns ``(cost, (counts, steps))``, the T1 counts of the unrolled
    relaxation (see ``minimize(count_t1=True)``) as an auxiliary output for ``grad(..., has_aux=True)``;
    the relaxation is the same as ``inner_opt``'s.
    """
    if return_t1_counts:
        (vertTable, heTable, faceTable, *_), (_hist, steps, counts) = minimize(
            vertTable, heTable, faceTable, width, height, vert_params, he_params, face_params, L_in, solver_inner,
            min_dist_T1, iterations_max=iterations_max, tolerance=tolerance, patience=patience,
            selected_verts=selected_verts, selected_hes=selected_hes, selected_faces=selected_faces,
            optimization_target=OptimizationTarget.VERTICES, update_t1_func=update_t1_func, count_t1=True,
            cap_vertex_step=cap_vertex_step)
        return L_out(vertTable, heTable, faceTable, width, height, vertTable_target, heTable_target,
                     faceTable_target, selected_verts, selected_hes, selected_faces, image_target), (counts, steps)
    (vertTable, heTable, faceTable), _loss = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_t1_func,
        cap_vertex_step=cap_vertex_step,
    )

    loss_out_value = L_out(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_out_value


def outer_opt(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    update_t1_func: UpdateT1Func = update_T1,
    params_regularizer: Callable | None = None,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array]:
    """Outer optimization for Automatic differentiation method."""
    recording = _T1_RECORDER is not None
    cost = partial(cost_ad, return_t1_counts=True) if recording else cost_ad
    hypergradient = grad(cost, argnums=(5, 6, 7), has_aux=recording)(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        update_t1_func,
        cap_vertex_step=cap_vertex_step,
    )
    if recording:
        (grad_verts, grad_hes, grad_faces), (counts, steps) = hypergradient
        _record_t1(counts, steps, "automatic differentiation")
    else:
        grad_verts, grad_hes, grad_faces = hypergradient

    # The mean-tension gauge (and any parameter-space regulariser) applies to EVERY estimator,
    # not only EP. The total objective is C(x*(theta)) + R(theta), so
    # grad_theta = hypergradient + grad R(theta). This is simpler than the EP case: there is no
    # cancellation to work around, because R depends on theta alone.
    if params_regularizer is not None:
        reg_verts, reg_hes, reg_faces = grad(params_regularizer, argnums=(0, 1, 2))(
            vert_params, he_params, face_params
        )
        grad_verts = grad_verts + reg_verts
        grad_hes = grad_hes + reg_hes
        grad_faces = grad_faces + reg_faces

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    new_vert_params: Array = updated_params["vert_params"]  # type: ignore
    new_he_params: Array = updated_params["he_params"]  # type: ignore
    new_face_params: Array = updated_params["face_params"]  # type: ignore

    return new_vert_params, new_he_params, new_face_params


#############################
## EQUILIBTIUM PROPAGATION ##
#############################


def _loss_ep_static(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
) -> Array:
    loss_inner = L_in(vertTable, heTable, faceTable, vert_params, he_params, face_params)
    loss_outer = L_out(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )
    return loss_inner + (beta * loss_outer)


@partial(jit, static_argnums=(6, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27))
def _minimize_ep(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,  # 0, 1, 2
    vert_params: Array,
    he_params: Array,
    face_params: Array,  # 3, 4, 5
    loss_fn: LossEPFunction,  # 6 [STATIC]
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,  # 7, 8, 9
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,  # 10, 11 [STATIC]
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,  # 12, 13, 14
    image_target: Array | None,
    beta: float,  # 15, 16
    solver: optax.GradientTransformation,  # 17 [STATIC]
    min_dist_T1: float,  # 18
    iterations_max: int = 1000,
    tolerance: float = 1e-3,
    patience: int = 5,  # 19, 20, 21 [STATIC]
    width: float = 1,
    height: float = 1,
    update_t1_func: UpdateT1Func = update_T1,  # 22, 23, 24 : [STATIC]
    perform_energy_check: bool = False,  # 25 static
    count_t1: bool = False,  # 26 static
    cap_vertex_step: bool = False,  # 27 static
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], tuple[Array, ...]]:
    # 1. Helper: Bind the fixed arguments to the loss function.
    def loss_evaluated(vt: Array, ht: Array, ft: Array, vp: Array, hp: Array, fp: Array) -> Array:
        return loss_fn(
            vt,
            ht,
            ft,
            width,
            height,
            vp,
            hp,
            fp,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
            beta,
        )

    # 2. Ensure static parameters are concrete integers
    width = float(width)
    height = float(height)
    iterations_max = int(iterations_max)
    patience = int(patience)
    min_dist_T1 = float(min_dist_T1)
    tolerance = float(tolerance)

    selected_verts = jnp.arange(vertTable.shape[0]) if selected_verts is None else jnp.array(selected_verts)
    selected_hes = jnp.arange(heTable.shape[0]) if selected_hes is None else jnp.array(selected_hes)
    selected_faces = jnp.arange(faceTable.shape[0]) if selected_faces is None else jnp.array(selected_faces)

    iterations_max = int(iterations_max)
    patience = int(patience)

    result = _jit_minimize(
        vertTable=vertTable,
        heTable=heTable,
        faceTable=faceTable,
        vert_params=vert_params,
        he_params=he_params,
        face_params=face_params,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        width=width,
        height=height,
        L_in=loss_evaluated,
        solver=solver,
        min_dist_T1=min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        optimization_target=OptimizationTarget.VERTICES,
        update_t1_func=update_t1_func,
        perform_energy_check=perform_energy_check,
        count_t1=count_t1,
        cap_vertex_step=cap_vertex_step,
    )
    # With ``count_t1`` the history is ``(hist, steps, counts)``: the counts leave the jitted function as
    # an output, so the caller (outside any trace) can record them.
    return result


def inner_eq_prop(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    loss_ep_static: LossEPFunction,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-3,
    patience: int = 5,
    update_t1_func: UpdateT1Func = update_T1,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
) -> tuple[tuple[Array, Array, Array], Array]:
    """Wrapper to call the JIT-compiled minimize function and handle post-processing (slicing history)."""
    # Call the JIT-compiled minimize function (defined in previous step)
    # We pass all targets and loss components so they can be bound
    # inside the static wrapper.
    count_t1 = _T1_RECORDER is not None
    (vt_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array, *counts) = _minimize_ep(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        loss_ep_static,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
        solver,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        width,
        height,
        update_t1_func,
        perform_energy_check=perform_energy_check,
        count_t1=count_t1,
        cap_vertex_step=cap_vertex_step,
    )
    if count_t1:
        _record_t1(counts[0], step_f_array, "equilibrium propagation")

    # Convert the JAX array step_f to a Python integer for slicing
    step_f = step_f_array.item()

    # Slice the loss history to remove padding zeros
    final_L_list = L_hist_full[:step_f]

    # Return updated geometry and the valid loss history
    return (vt_f, ht_f, ft_f), final_L_list


def _compute_ep_grads(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    update_t1_func: UpdateT1Func = update_T1,
    params_regularizer: Callable | None = None,
    ep_variant: EPVariant = EPVariant.CENTERED,
    cap_vertex_step: bool = False,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Compute EP parameter gradients and return (params_dict, grads_dict).

    This is the gradient-computation core shared by ``outer_eq_prop`` and
    ``outer_eq_prop_with_state``.  It performs the free and nudged inner solves,
    takes their difference, adds any ``params_regularizer`` gradient, and returns the
    parameter and gradient dicts ready for an optax ``solver_outer.update`` call.

    The estimator is selected by ``ep_variant`` (see :class:`EPVariant`):

    * ``CENTERED`` (default, recommended) -- solve at ``-beta`` and ``+beta``, divide by
      ``2*beta``. Second order: the leading O(beta) bias cancels, leaving O(beta**2).
    * ``ONE_SIDED`` -- solve at ``0`` (true free equilibrium) and ``+beta``, divide by
      ``beta``. First order: O(beta) bias. Provided for comparison/ablation only.

    Convention note: ``beta`` here is the *half-span* -- perturbations are placed at
    ``+/-beta``, i.e. a total span of ``2*beta``, with **no conversion factor**; no
    factor of 2 applies anywhere.

    Returns:
        params: dict with keys ``vert_params``, ``he_params``, ``face_params``.
        grads:  same keys, EP gradients (+ regularizer if provided).
    """
    ep_variant = EPVariant(ep_variant)
    if ep_variant is EPVariant.CENTERED:
        beta_free = -beta       # symmetric negative nudge
        denominator = 2.0 * beta
    else:  # EPVariant.ONE_SIDED
        beta_free = 0.0         # true (unnudged) free equilibrium
        denominator = beta
    # The free solve and the free *gradient* must use the SAME beta_free, otherwise the
    # estimator is an inconsistent hybrid -- e.g. a free solve at 0 paired with a
    # 1/(2*beta) normalisation, i.e. neither centered nor one-sided.
    (vertTable_free, heTable_free, faceTable_free), _ = inner_eq_prop(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta_free,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        update_t1_func,
        cap_vertex_step=cap_vertex_step,
    )

    # Nudged solve: beta>0
    (vertTable_nudged, heTable_nudged, faceTable_nudged), _ = inner_eq_prop(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        update_t1_func,
        cap_vertex_step=cap_vertex_step,
    )

    grad_ep = grad(_loss_ep_static, argnums=(5, 6, 7))

    grad_loss_ep_free_verts, grad_loss_ep_free_hes, grad_loss_ep_free_faces = grad_ep(
        vertTable_free,
        heTable_free,
        faceTable_free,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=beta_free,
    )

    grad_loss_ep_nudged_verts, grad_loss_ep_nudged_hes, grad_loss_ep_nudged_faces = grad_ep(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_verts = (1.0 / denominator) * ((grad_loss_ep_nudged_verts) - (grad_loss_ep_free_verts))
    grad_hes = (1.0 / denominator) * ((grad_loss_ep_nudged_hes) - (grad_loss_ep_free_hes))
    grad_faces = (1.0 / denominator) * ((grad_loss_ep_nudged_faces) - (grad_loss_ep_free_faces))

    # A term on the parameters alone cancels in the (nudged - free) difference above, so it has to
    # be differentiated separately and added in here, before the outer solver sees the gradients.
    if params_regularizer is not None:
        reg_verts, reg_hes, reg_faces = grad(params_regularizer, argnums=(0, 1, 2))(
            vert_params, he_params, face_params
        )
        grad_verts = grad_verts + reg_verts
        grad_hes = grad_hes + reg_hes
        grad_faces = grad_faces + reg_faces

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    return params, grads


def outer_eq_prop(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    update_t1_func: UpdateT1Func = update_T1,
    params_regularizer: Callable | None = None,
    ep_variant: EPVariant = EPVariant.CENTERED,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array]:
    """Outer optimization for equilibrium propagation method.

    `params_regularizer(vert_params, he_params, face_params) -> scalar`, if given, has its
    gradient added to the equilibrium-propagation gradients before the outer solver's update, so
    it is optimized jointly with the outer loss.

    The outer-optimizer state is re-initialised every call (current / default behaviour).
    For persistent state across epochs use ``outer_eq_prop_with_state``.
    """
    params, grads = _compute_ep_grads(
        vertTable, heTable, faceTable,
        width, height,
        vert_params, he_params, face_params,
        vertTable_target, heTable_target, faceTable_target,
        L_in, L_out,
        solver_inner,
        min_dist_T1, iterations_max, tolerance, patience,
        selected_verts, selected_hes, selected_faces,
        image_target, beta,
        update_t1_func, params_regularizer, ep_variant, cap_vertex_step,
    )

    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


def outer_eq_prop_with_state(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    opt_state: optax.OptState | None,
    update_t1_func: UpdateT1Func = update_T1,
    params_regularizer: Callable | None = None,
    ep_variant: EPVariant = EPVariant.CENTERED,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array, optax.OptState]:
    """Outer optimization for equilibrium propagation with persistent optimizer state.

    Identical to ``outer_eq_prop`` except:
    - Accepts ``opt_state``: if ``None``, initialises the state (first call semantics);
      otherwise uses the supplied state (persistent-Adam / any stateful optax transform).
    - Returns ``(vert_params, he_params, face_params, updated_opt_state)`` so the caller
      can thread the state across epochs.

    This function is intended to be called from ``PbcBilevelOptimizer._outer_opt`` when
    ``persist_outer_state=True``.  All other callers should use ``outer_eq_prop``.
    """
    params, grads = _compute_ep_grads(
        vertTable, heTable, faceTable,
        width, height,
        vert_params, he_params, face_params,
        vertTable_target, heTable_target, faceTable_target,
        L_in, L_out,
        solver_inner,
        min_dist_T1, iterations_max, tolerance, patience,
        selected_verts, selected_hes, selected_faces,
        image_target, beta,
        update_t1_func, params_regularizer, ep_variant, cap_vertex_step,
    )

    if opt_state is None:
        opt_state = solver_outer.init(params)
    updates, new_opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params, new_opt_state


###########################
## IMPLICIT DIFFERENTION ##
###########################


def outer_implicit(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    update_t1_func: UpdateT1Func = update_T1,
    params_regularizer: Callable | None = None,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array]:
    """Outer optimization for implicit differentiation method."""

    def L_in_flatten(  # noqa: N802
        vertTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2)
        return L_in(vertTable_tmp, heTable, faceTable, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), _ = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_t1_func,
        cap_vertex_step=cap_vertex_step,
    )

    vert_flat_eq = vertTable_eq.flatten()

    # H = d^2 L_in / d vertTable^2
    G_op = grad(L_in_flatten, argnums=0)

    dtype = vert_flat_eq.dtype

    def matvec(v: ArrayLike) -> Array:
        v_casted = jnp.asarray(v, dtype=dtype)
        _, Hv = jvp(
            lambda vfe: G_op(vfe, heTable_eq, faceTable_eq, vert_params, he_params, face_params),
            (vert_flat_eq,),
            (v_casted,),
        )
        return Hv

    nv = len(vert_flat_eq)
    H_np = LinearOperator((nv, nv), matvec=matvec)  # type: ignore

    # Compute cross-derivatives (dL_in / d vertTable d param)
    crossderivative_verts_op = jacfwd(G_op, argnums=3)
    crossderivative_hes_op = jacfwd(G_op, argnums=4)
    crossderivative_faces_op = jacfwd(G_op, argnums=5)

    crossderivative_verts = crossderivative_verts_op(
        vert_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = crossderivative_hes_op(
        vert_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = crossderivative_faces_op(
        vert_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    # Convert JAX arrays to NumPy arrays for use with scipy.sparse.linalg.minres
    # H_np = np.asarray(H) # Replaced by linear operator
    cd_verts_np = np.asarray(crossderivative_verts)
    cd_hes_np = np.asarray(crossderivative_hes)
    cd_faces_np = np.asarray(crossderivative_faces)

    # MINRES solves H * X = B. We want X = -H^{-1} * B, so we solve H * X = -B

    b_verts = -cd_verts_np
    b_verts_np = np.asarray(b_verts)
    N_rhs = b_verts.shape[1]

    Lambda_solutions = []
    # Loop over the columns (right-hand sides)
    for i in range(N_rhs):
        b_vector = b_verts_np[:, i]
        # Solve H_np * Lambda_i = b_vector
        Lambda_i_np, _ = minres(H_np, b_vector)
        Lambda_solutions.append(Lambda_i_np)
    L_in_derivative_verts_np = jnp.stack(Lambda_solutions, axis=1)

    b_hes = -cd_hes_np
    b_hes_np = np.asarray(b_hes)
    N_rhs = b_hes.shape[1]

    Lambda_solutions = []
    # Loop over the columns (right-hand sides)
    for i in range(N_rhs):
        b_vector = b_hes_np[:, i]
        # Solve H_np * Lambda_i = b_vector
        Lambda_i_np, _ = minres(H_np, b_vector)
        Lambda_solutions.append(Lambda_i_np)
    L_in_derivative_hes_np = jnp.stack(Lambda_solutions, axis=1)

    b_faces = -cd_faces_np

    b_faces_np = np.asarray(b_faces)
    N_rhs = b_faces_np.shape[1]

    Lambda_solutions = []
    # Loop over the columns (right-hand sides)
    for i in range(N_rhs):
        b_vector = b_faces_np[:, i]
        # Solve H_np * Lambda_i = b_vector
        Lambda_i_np, _ = minres(H_np, b_vector)
        Lambda_solutions.append(Lambda_i_np)
    L_in_derivative_faces_np = jnp.stack(Lambda_solutions, axis=1)

    # Convert solutions back to JAX arrays for the final gradient calculation
    L_in_derivative_verts = jnp.asarray(L_in_derivative_verts_np)
    L_in_derivative_hes = jnp.asarray(L_in_derivative_hes_np)
    L_in_derivative_faces = jnp.asarray(L_in_derivative_faces_np)

    # The term (dL_out / d vertTable) evaluated at the equilibrium
    dL_out_dvert = grad(L_out, argnums=0)(
        vertTable_eq,
        heTable_eq,
        faceTable_eq,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    ).flatten()

    # dL_out / d param = (dL_in_d param)^T @ (dL_out / d vertTable)
    grad_verts = L_in_derivative_verts.T @ dL_out_dvert
    grad_hes = L_in_derivative_hes.T @ dL_out_dvert
    grad_faces = L_in_derivative_faces.T @ dL_out_dvert

    # The mean-tension gauge (and any parameter-space regulariser) applies to EVERY estimator,
    # not only EP. The total objective is C(x*(theta)) + R(theta), so
    # grad_theta = hypergradient + grad R(theta). This is simpler than the EP case: there is no
    # cancellation to work around, because R depends on theta alone.
    if params_regularizer is not None:
        reg_verts, reg_hes, reg_faces = grad(params_regularizer, argnums=(0, 1, 2))(
            vert_params, he_params, face_params
        )
        grad_verts = grad_verts + reg_verts
        grad_hes = grad_hes + reg_hes
        grad_faces = grad_faces + reg_faces

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


##########################
## ADJOINT STATE METHOD ##
##########################


def outer_adjoint_state(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    update_t1_func: UpdateT1Func = update_T1,
    params_regularizer: Callable | None = None,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array]:
    """Outer optimization for adjoint state method."""

    def L_in_flatten(  # noqa: N802
        vertTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2)
        return L_in(vertTable_tmp, heTable, faceTable, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), _L_in_value = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_t1_func,
        cap_vertex_step=cap_vertex_step,
    )

    vertTable_eq_flat = vertTable_eq.flatten()

    G_op = grad(L_in_flatten, argnums=0)

    dtype = vertTable_eq_flat.dtype

    def matvec(v: ArrayLike) -> Array:
        v_casted = jnp.asarray(v, dtype=dtype)
        _, Hv = jvp(
            lambda vfe: G_op(vfe, heTable_eq, faceTable_eq, vert_params, he_params, face_params),
            (vertTable_eq_flat,),
            (v_casted,),
        )
        return Hv

    nv = len(vertTable_eq_flat)
    H_np = LinearOperator((nv, nv), matvec=matvec)  # type: ignore
    gradout = grad(L_out, argnums=0)(
        vertTable_eq,
        heTable_eq,
        faceTable_eq,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    ).flatten()

    # Convert JAX arrays to NumPy arrays for SciPy function call
    gradout_np = np.asarray(gradout)

    # Call SciPy MINRES. H is symmetric and possibly indefinite, so MINRES is suitable.
    Lambda_np, _ = minres(H_np, gradout_np)

    # Convert the result back to a JAX array
    Lambda = jnp.asarray(Lambda_np)

    crossderivative_verts = jacfwd(G_op, argnums=3)(
        vertTable_eq_flat, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = jacfwd(G_op, argnums=4)(
        vertTable_eq_flat, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = jacfwd(G_op, argnums=5)(
        vertTable_eq_flat, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    grad_verts = -Lambda @ crossderivative_verts
    grad_hes = -Lambda @ crossderivative_hes
    grad_faces = -Lambda @ crossderivative_faces

    # The mean-tension gauge (and any parameter-space regulariser) applies to EVERY estimator,
    # not only EP. The total objective is C(x*(theta)) + R(theta), so
    # grad_theta = hypergradient + grad R(theta). This is simpler than the EP case: there is no
    # cancellation to work around, because R depends on theta alone.
    if params_regularizer is not None:
        reg_verts, reg_hes, reg_faces = grad(params_regularizer, argnums=(0, 1, 2))(
            vert_params, he_params, face_params
        )
        grad_verts = grad_verts + reg_verts
        grad_hes = grad_hes + reg_hes
        grad_faces = grad_faces + reg_faces

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}

    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


#############
## WRAPPER ##
#############


def bilevel_opt(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None = None,
    beta: float = 0.01,
    method: BilevelOptimizationMethod = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], Array]:
    """Bilevel optimization for PBC meshes."""
    match method:
        case BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION:
            vert_params, he_params, face_params = outer_opt(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
                update_t1_func,
            )

        case BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION:
            vert_params, he_params, face_params = outer_eq_prop(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
                beta,
                update_t1_func,
            )

        case BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION:
            vert_params, he_params, face_params = outer_implicit(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
                update_t1_func,
            )

        case BilevelOptimizationMethod.ADJOINT_STATE:
            vert_params, he_params, face_params = outer_adjoint_state(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
                update_t1_func,
            )

        case _:
            msg = f"Method not recognized. Must be a BilevelOptimizationMethod. Got {method}."
            raise ValueError(msg)

    (vertTable, heTable, faceTable), cost = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_t1_func,
    )

    return (vertTable, heTable, faceTable, vert_params, he_params, face_params), cost
