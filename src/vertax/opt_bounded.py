"""Optimization methods for VertAX (bounded version)."""

import os
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, grad, jacfwd, jit, jvp, lax, value_and_grad
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator, minres

from vertax.method_enum import BilevelOptimizationMethod, EPVariant
from vertax.topo import ARC_HALF_ANGLE_MAX, ARC_HALF_ANGLE_MIN, update_T1_bounded

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
# Set VERTAX_MAX_VERTEX_STEP_FACTOR to override; a large value effectively
# disables the cap and restores the previous unbounded behaviour.
MAX_VERTEX_STEP_FACTOR = float(os.environ.get("VERTAX_MAX_VERTEX_STEP_FACTOR", "5.0"))

InnerLossFunctionBounded = Callable[
    [Array, Array, Array, Array, Array | None, Array | None, Array | None, Array, Array, Array], Array
]
OuterLossFunction = Callable[
    [
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ],
    Array,
]

UpdateT1FuncBounded = Callable[
    [Array, Array, Array, Array, Array, Array, Array, InnerLossFunctionBounded, float, Array, Array, Array, bool],
    tuple[Array, Array, Array, Array],
]

LossEPFunction = Callable[
    [
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        InnerLossFunctionBounded,
        OuterLossFunction,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        float,
    ],
    Array,
]
###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


@partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
def _minimize_bounded(  # noqa: C901
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunctionBounded,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    argnums: int = 0,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    iterations_max = int(iterations_max)
    patience = int(patience)

    # --- select the target array and indices for the optimizer ---
    if argnums == 0:
        x0, sel = vertTable, selected_verts
    elif argnums == 2:
        x0, sel = heTable, selected_hes
    elif argnums == 3:
        x0, sel = faceTable, selected_faces
    elif argnums == 4:
        x0, sel = vert_params, selected_verts
    elif argnums == 5:
        x0, sel = he_params, selected_hes
    elif argnums == 6:
        x0, sel = face_params, selected_faces
    else:
        msg = "argnums must be in {0,2,3,4,5,6}"
        raise ValueError(msg)

    # --- check if the target is float-type before differentiating ---
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        msg = (
            f"Cannot differentiate argnums={argnums}: "
            f"target array has dtype {x0.dtype}. "
            f"Use a float array (vertTable, vert_params, he_params, face_params) only."
        )
        raise TypeError(msg)

    jacnums = argnums
    if argnums == 0:
        opt_state = solver.init((vertTable, angTable))
        jacnums = [0, 1]
    else:
        opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = L_in(
        vertTable,
        angTable,
        heTable,
        faceTable,
        selected_verts,
        selected_hes,
        selected_faces,
        vert_params,
        he_params,
        face_params,
    )
    L_in_list = jnp.zeros((iterations_max,)).at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)

    @jit
    def scan_step(  # noqa: C901
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], i: int
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], None]:
        vt, at, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list = carry

        is_running = jnp.logical_not(should_stop)

        # Compute loss
        L_current = L_in(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)

        # Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)
        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)

        # Gradient wrt chosen argnums
        grads = grad(L_in, argnums=jacnums)(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)

        # value_fn re-evaluates L_in as a function of the solver's params only
        # (everything else held fixed at the current step). Line-search solvers
        # (cg_linesearch, sgd_linesearch, lbfgs) need it; adam/sgd ignore the extra arguments.
        if argnums == 0:
            params_cur = (vt, at)

            def value_fn(p: tuple[Array, Array]) -> Array:
                vt_p, at_p = p
                return L_in(vt_p, at_p, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)
        elif argnums == 2:
            params_cur = ht

            def value_fn(p: Array) -> Array:
                return L_in(vt, at, p, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)
        elif argnums == 3:
            params_cur = ft

            def value_fn(p: Array) -> Array:
                return L_in(vt, at, ht, p, selected_verts, selected_hes, selected_faces, vp, hp, fp)
        elif argnums == 4:
            params_cur = vp

            def value_fn(p: Array) -> Array:
                return L_in(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, p, hp, fp)
        elif argnums == 5:
            params_cur = hp

            def value_fn(p: Array) -> Array:
                return L_in(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, p, fp)
        else:  # argnums == 6
            params_cur = fp

            def value_fn(p: Array) -> Array:
                return L_in(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, p)

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
        # updates, new_opt_state = solver.update(grads, opt_state)

        # Apply updates to the chosen array on selected indices
        if argnums == 0:
            # Cap the per-step vertex move so large line-search steps (cg_linesearch,
            # gd_linesearch) stay on the scale the incremental T1 detector expects.
            # The cap is relative to min_dist_T1, so it follows the mesh's own scale.
            # Angles ride along unscaled.
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
            # Opt-in via cap_vertex_step: only line-search solvers choose their own
            # step size and need it. Fixed-step solvers happened to be unaffected on
            # the 20-cell bench (their steps sat below the cap), but that is a
            # property of those settings, not a guarantee -- sgd(lr=0.01) with
            # min_dist_T1=0.005 gives a cap of 0.025 that does bite. Gating it keeps
            # them untouched by construction rather than by coincidence.
            vstep = updates[0].at[sel].get()
            if cap_vertex_step:
                max_step = MAX_VERTEX_STEP_FACTOR * min_dist_T1
                vnorm = jnp.linalg.norm(vstep, axis=1, keepdims=True)
                vscale = jnp.minimum(1.0, max_step / jnp.maximum(vnorm, 1e-12))
                vstep = vstep * vscale
            new_vt = vt.at[sel].set(vt[sel] + vstep)
            # Keep every arc between a chord and a half circle at every step. The T1 routine applies
            # the same limit when it is called, but with T1 transitions disabled nothing else does,
            # and an arc that crosses a half circle leaves the smooth branch of the energy. Clipping
            # twice is idempotent, so relaxations with T1 transitions are unchanged.
            new_at = jnp.clip(at + updates[1], ARC_HALF_ANGLE_MIN, ARC_HALF_ANGLE_MAX)
            vt = lax.cond(is_running, lambda: new_vt, lambda: vt)
            at = lax.cond(is_running, lambda: new_at, lambda: at)
        elif argnums == 1:
            new_ht = ht.at[sel].set(ht[sel] + updates.at[sel].get())
            ht = lax.cond(is_running, lambda: new_ht, lambda: ht)
        elif argnums == 2:
            new_ft = ft.at[sel].set(ft[sel] + updates.at[sel].get())
            ft = lax.cond(is_running, lambda: new_ft, lambda: ft)
        elif argnums == 3:
            new_vp = vp.at[sel].set(vp[sel] + updates.at[sel].get())
            vp = lax.cond(is_running, lambda: new_vp, lambda: vp)
        elif argnums == 4:
            new_hp = hp.at[sel].set(hp[sel] + updates.at[sel].get())
            hp = lax.cond(is_running, lambda: new_hp, lambda: hp)
        elif argnums == 5:
            new_fp = fp.at[sel].set(fp[sel] + updates.at[sel].get())
            fp = lax.cond(is_running, lambda: new_fp, lambda: fp)

        opt_state = lax.cond(is_running, lambda: new_opt_state, lambda: opt_state)
        stagnation_count = lax.cond(is_running, lambda: new_stagnation_count, lambda: stagnation_count)
        should_stop = new_should_stop

        # Geometry updates
        new_vt_T1, new_at_T1, new_ht_T1, new_ft_T1 = update_T1_func(
            vt,
            at,
            ht,
            ft,
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
        at = lax.cond(is_running, lambda: new_at_T1, lambda: at)
        ht = lax.cond(is_running, lambda: new_ht_T1, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_T1, lambda: ft)

        # Shift prev_L_values
        new_prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        new_prev_L_values = new_prev_L_values.at[0].set(L_current)
        prev_L_values = lax.cond(is_running, lambda: new_prev_L_values, lambda: prev_L_values)

        L_list = L_list.at[i].set(L_current)
        step_count = i + 1

        return (
            vt,
            at,
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
        ), None

    init_carry = (
        vertTable,
        angTable,
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

    final_state, _ = lax.scan(scan_step, init_carry, xs=jnp.arange(iterations_max))

    # Unpack and return results (for slicing outside JIT)
    vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state

    return (vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f)


def inner_opt_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunctionBounded,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
) -> tuple[tuple[Array, Array, Array, Array], Array]:
    """Inner optimization for a bounded mesh.

    Args:
        cap_vertex_step: Clip each per-step vertex displacement to
            MAX_VERTEX_STEP_FACTOR * min_dist_T1. Intended for line-search solvers
            (cg_linesearch, gd_linesearch, lbfgs), which choose their own step size
            and can otherwise move a vertex across whole cells in one iteration.
            Off by default, so fixed-step solvers (adam, sgd) are untouched.
    """
    # Use the general minimize function with argnums=0 (optimize vertTable)
    (vt_f, at_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = _minimize_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
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
        argnums=0,
        update_T1_func=update_T1_func,
        perform_energy_check=perform_energy_check,
        cap_vertex_step=cap_vertex_step,
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()
    # Now slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]
    # Return updated arrays and loss history
    return (vt_f, at_f, ht_f, ft_f), final_L_list


def _sq_dist(p: Array, q: Array) -> Array:
    """Squared distance between 2D points (cols 0,1)."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _build_t1_repair_perm(
    vertTable_after: Array,
    vertTable_target: Array | None,
    heTable_before: Array,
    heTable_after: Array,
) -> Array:
    """Build a vertex-index permutation that swaps T1-flipped pairs when it lowers v2v cost.

    A half-edge whose face index (col 7) changed across `inner_opt` was part of a T1: only
    `update_T1` ever writes that column. For each such half-edge,
    cols (3, 4) give the (a, b) vertex pair to test (T1 should only happen on inside edges).
    Each T1 is reported twice (edge + twin);
    the swap-only-if-better filter is idempotent, so duplicates are harmless.
    """
    perm = jnp.arange(vertTable_after.shape[0], dtype=jnp.int32)
    # jax.debug.print("perm={x}", x=perm, ordered=True)
    # TODO : replace with jax.cond ?

    # jax.debug.print("vertTableAfter={x}", x=vertTable_after, ordered=True)
    if vertTable_target is None:
        return perm

    t1_mask = heTable_before[:, 7] != heTable_after[:, 7]
    pair_a = heTable_before[:, 3].astype(jnp.int32)
    pair_b = heTable_before[:, 4].astype(jnp.int32)

    def body(i: int, perm: Array) -> Array:
        a = pair_a[i]
        b = pair_b[i]
        ta = vertTable_target[perm[a]]
        tb = vertTable_target[perm[b]]
        pa = vertTable_after[a]
        pb = vertTable_after[b]
        cost_no_swap = _sq_dist(pa, ta) + _sq_dist(pb, tb)
        cost_swap = _sq_dist(pa, tb) + _sq_dist(pb, ta)
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


def _add_params_regularizer_grads(
    grad_verts: Array,
    grad_hes: Array,
    grad_faces: Array,
    params_regularizer: Callable | None,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
) -> tuple[Array, Array, Array]:
    """Add the gradient of `params_regularizer(vert_params, he_params, face_params)`, if any, to a hypergradient.

    The regularizer depends on the parameters alone, so its gradient is added to the hypergradient of
    every estimator before the outer solver's update, as in the periodic estimators (`vertax.opt`).
    """
    if params_regularizer is None:
        return grad_verts, grad_hes, grad_faces
    reg_verts, reg_hes, reg_faces = grad(params_regularizer, argnums=(0, 1, 2))(vert_params, he_params, face_params)
    return grad_verts + reg_verts, grad_hes + reg_hes, grad_faces + reg_faces


def cost_ad_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
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
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    cap_vertex_step: bool = False,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Automatic differentiation cost function."""
    # heTable_before = heTable

    (vertTable, angTable, heTable, faceTable), _ = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
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
        update_T1_func,
        cap_vertex_step=cap_vertex_step,
    )

    # perm = _build_t1_repair_perm(
    #     vertTable,
    #     vertTable_target,
    #     heTable_before,
    #     heTable,
    # )
    # vertTable, heTable = _apply_perm_to_state(perm, vertTable, heTable)

    loss_out_value = L_out(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_out_value, (vertTable, angTable, heTable, faceTable)


def outer_opt_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    image_target: Array | None = None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    params_regularizer: Callable | None = None,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Outer optimization for a bounded mesh.

    Returns the tables (vertices, angles, edges, faces) of the inner equilibrium and the updated
    parameters.
    """
    ((_, (vertTable, angTable, heTable, faceTable)), (grad_verts, grad_hes, grad_faces)) = value_and_grad(
        cost_ad_bounded, argnums=(4, 5, 6), has_aux=True
    )(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
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
        update_T1_func,
        cap_vertex_step=cap_vertex_step,
    )

    grad_verts, grad_hes, grad_faces = _add_params_regularizer_grads(
        grad_verts, grad_hes, grad_faces, params_regularizer, vert_params, he_params, face_params
    )
    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params


#############################
## EQUILIBTIUM PROPAGATION ##
#############################


def _loss_ep_static_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
) -> Array:
    loss_inner_value = L_in(
        vertTable,
        angTable,
        heTable,
        faceTable,
        selected_verts,
        selected_hes,
        selected_faces,
        vert_params,
        he_params,
        face_params,
    )

    loss_outer_value = L_out(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_inner_value + (beta * loss_outer_value)


@partial(jit, static_argnums=(7, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27))
def _minimize_ep_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    loss_fn: LossEPFunction,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    image_target: Array | None = None,
    beta: float = 0.001,
    argnums: int = 0,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    def loss_evaluated(
        vt: Array,
        at: Array,
        ht: Array,
        ft: Array,
        selected_verts: Array,
        selected_hes: Array,
        selected_faces: Array,
        vp: Array,
        hp: Array,
        fp: Array,
    ) -> Array:
        return loss_fn(
            vt,
            at,
            ht,
            ft,
            vp,
            hp,
            fp,
            vertTable_target,
            angTable_target,
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

    return _minimize_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        loss_evaluated,
        solver,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        argnums,
        update_T1_func,
        perform_energy_check=perform_energy_check,
        cap_vertex_step=cap_vertex_step,
    )


def inner_eq_prop_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    loss_ep_static_bounded: LossEPFunction,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
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
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    perform_energy_check: bool = False,
    cap_vertex_step: bool = False,
) -> tuple[tuple[Array, Array, Array, Array], Array]:
    """Inner optimization function for equilibrium propagation only and bounded meshes."""
    (vt_f, at_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = _minimize_ep_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        loss_ep_static_bounded,
        vertTable_target,
        angTable_target,
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
        beta,
        argnums=0,
        update_T1_func=update_T1_func,
        perform_energy_check=perform_energy_check,
        cap_vertex_step=cap_vertex_step,
    )

    step_f = step_f_array.item()
    final_L_list = L_hist_full[:step_f]
    return (vt_f, at_f, ht_f, ft_f), final_L_list


def outer_eq_prop_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
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
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    params_regularizer: Callable | None = None,
    ep_variant: EPVariant = EPVariant.CENTERED,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Outer optimization for equilibrium propagation.

    `ep_variant` selects the estimator, as in the periodic `vertax.opt._compute_ep_grads`: CENTERED
    (default) solves at `-beta` and `+beta` and divides by `2 * beta`; ONE_SIDED solves at `0` and
    `+beta` and divides by `beta`. The returned tables are those of the free solve.
    `params_regularizer(vert_params, he_params, face_params) -> scalar`, if given, has its gradient
    added to the EP gradient before the outer solver's update.

    Returns the tables (vertices, angles, edges, faces) of the free solve and the updated parameters.
    """
    ep_variant = EPVariant(ep_variant)
    if ep_variant is EPVariant.CENTERED:
        beta_free = -beta
        denominator = 2.0 * beta
    else:  # EPVariant.ONE_SIDED
        beta_free = 0.0
        denominator = beta
    # heTable_before = heTable

    (vertTable_free, angTable_free, heTable_free, faceTable_free), _loss_free = inner_eq_prop_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static_bounded,
        vertTable_target,
        angTable_target,
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
        beta=beta_free,
        update_T1_func=update_T1_func,
        cap_vertex_step=cap_vertex_step,
    )

    # perm_free = _build_t1_repair_perm(
    #     vertTable_free,
    #     vertTable_target,
    #     heTable_before,
    #     heTable_free,
    # )
    # vertTable_free, heTable_free = _apply_perm_to_state(perm_free, vertTable_free, heTable_free)

    (vertTable_nudged, angTable_nudged, heTable_nudged, faceTable_nudged), _loss_nudged = inner_eq_prop_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static_bounded,
        vertTable_target,
        angTable_target,
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
        beta,
        update_T1_func,
        cap_vertex_step=cap_vertex_step,
    )

    # perm_nudged = _build_t1_repair_perm(
    #     vertTable_nudged,
    #     vertTable_target,
    #     heTable_before,
    #     heTable_nudged,
    # )
    # vertTable_nudged, heTable_nudged = _apply_perm_to_state(perm_nudged, vertTable_nudged, heTable_nudged)

    grad_ep = grad(_loss_ep_static_bounded, argnums=(4, 5, 6))

    grad_loss_ep_free_verts, grad_loss_ep_free_hes, grad_loss_ep_free_faces = grad_ep(
        vertTable_free,
        angTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
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
        angTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
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

    grad_verts, grad_hes, grad_faces = _add_params_regularizer_grads(
        grad_verts, grad_hes, grad_faces, params_regularizer, vert_params, he_params, face_params
    )
    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vertTable_free, angTable_free, heTable_free, faceTable_free, vert_params, he_params, face_params


###########################
## IMPLICIT DIFFERENTION ##
###########################


def outer_implicit_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
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
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    params_regularizer: Callable | None = None,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Outer optimization for implicit differentiation method.

    Returns the tables (vertices, angles, edges, faces) of the inner equilibrium and the updated
    parameters.
    """

    def L_in_flatten(  # noqa: N802
        vertangTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertangTable_flatten[: -heTable.shape[0] // 2].reshape(
            (len(vertangTable_flatten) - heTable.shape[0] // 2) // 2, 2
        )
        angTable_tmp = vertangTable_flatten[-heTable.shape[0] // 2 :]
        return L_in(
            vertTable_tmp,
            angTable_tmp,
            heTable,
            faceTable,
            selected_verts,
            selected_hes,
            selected_faces,
            vert_params,
            he_params,
            face_params,
        )

    # heTable_before = heTable

    (vertTable_eq, angTable_eq, heTable_eq, faceTable_eq), _L_in_value = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
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
        update_T1_func,
        cap_vertex_step=cap_vertex_step,
    )

    # perm = _build_t1_repair_perm(
    #     vertTable_eq,
    #     vertTable_target,
    #     heTable_before,
    #     heTable_eq,
    # )
    # vertTable_eq, heTable_eq = _apply_perm_to_state(perm, vertTable_eq, heTable_eq)
    ################################################################
    ################            Start mod           ################
    ################################################################

    vertang_flat_eq = jnp.concatenate((vertTable_eq.flatten(), angTable_eq))

    # H = d^2 L_in / d vertangTable^2
    G_op = grad(L_in_flatten, argnums=0)

    dtype = vertang_flat_eq.dtype

    def matvec(v: ArrayLike) -> Array:
        v_casted = jnp.asarray(v, dtype=dtype)
        _, Hv = jvp(
            lambda vfe: G_op(vfe, heTable_eq, faceTable_eq, vert_params, he_params, face_params),
            (vertang_flat_eq,),
            (v_casted,),
        )
        return Hv

    nv = len(vertang_flat_eq)
    H_np = LinearOperator((nv, nv), matvec=matvec)  # type: ignore

    # Compute cross-derivatives (dL_in / d vertangTable d param)
    crossderivative_verts_op = jacfwd(G_op, argnums=3)
    crossderivative_hes_op = jacfwd(G_op, argnums=4)
    crossderivative_faces_op = jacfwd(G_op, argnums=5)

    crossderivative_verts = crossderivative_verts_op(
        vertang_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = crossderivative_hes_op(
        vertang_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = crossderivative_faces_op(
        vertang_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
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

    # The term (dL_out / d vertangTable) evaluated at the equilibrium
    dL_out_dvertang = jnp.concatenate(
        [
            grad(L_out, argnums=0)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            )[:, :2].flatten(),
            grad(L_out, argnums=1)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            ),
        ]
    )

    # dL_out / d param = (dL_in_d param)^T @ (dL_out / d vertangTable)
    grad_verts = L_in_derivative_verts.T @ dL_out_dvertang
    grad_hes = L_in_derivative_hes.T @ dL_out_dvertang
    grad_faces = L_in_derivative_faces.T @ dL_out_dvertang

    grad_verts, grad_hes, grad_faces = _add_params_regularizer_grads(
        grad_verts, grad_hes, grad_faces, params_regularizer, vert_params, he_params, face_params
    )
    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vertTable_eq, angTable_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params


##########################
## ADJOINT STATE METHOD ##
##########################


def outer_adjoint_state_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
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
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
    params_regularizer: Callable | None = None,
    cap_vertex_step: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Outer optimization for adjoint state method.

    Returns the tables (vertices, angles, edges, faces) it was given, unchanged, and the updated
    parameters.
    """

    def L_in_flatten(  # noqa: N802
        vertangTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertangTable_flatten[: -heTable.shape[0] // 2].reshape(
            (len(vertangTable_flatten) - heTable.shape[0] // 2) // 2, 2
        )
        angTable_tmp = vertangTable_flatten[-heTable.shape[0] // 2 :]
        return L_in(
            vertTable_tmp,
            angTable_tmp,
            heTable,
            faceTable,
            selected_verts,
            selected_hes,
            selected_faces,
            vert_params,
            he_params,
            face_params,
        )

    # heTable_before = heTable

    (vertTable_eq, angTable_eq, heTable_eq, faceTable_eq), _L_in_value = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
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
        update_T1_func,
        cap_vertex_step=cap_vertex_step,
    )

    # jax.debug.print("vertTableAfter={x}", x=vertTable_eq, ordered=True)
    # perm = _build_t1_repair_perm(
    #     vertTable_eq,
    #     vertTable_target,
    #     heTable_before,
    #     heTable_eq,
    # )
    # vertTable_eq, heTable_eq = _apply_perm_to_state(perm, vertTable_eq, heTable_eq)

    vertTable_eq_flat = jnp.concatenate((vertTable_eq.flatten(), angTable_eq))

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

    gradout = jnp.concatenate(
        [
            grad(L_out, argnums=0)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            )[:, :2].flatten(),
            grad(L_out, argnums=1)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            ),
        ]
    )

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

    grad_verts, grad_hes, grad_faces = _add_params_regularizer_grads(
        grad_verts, grad_hes, grad_faces, params_regularizer, vert_params, he_params, face_params
    )
    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}

    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params


#############
## WRAPPER ##
#############


def bilevel_opt_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
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
    beta: float = 0.001,
    optimization_method: BilevelOptimizationMethod = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], Array]:
    """Bilevel optimization for bounded meshes."""
    match optimization_method:
        case BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION:
            vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params = outer_opt_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
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
            )
        case BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION:
            vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params = outer_eq_prop_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
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
                update_T1_func,
            )
        case BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION:
            vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params = outer_implicit_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
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
                update_T1_func,
            )
        case BilevelOptimizationMethod.ADJOINT_STATE:
            vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params = outer_adjoint_state_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
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
                update_T1_func,
            )
        case _:
            msg = f"{optimization_method} is not implemented for bounded meshes."
            raise ValueError(msg)
    (vertTable, angTable, heTable, faceTable), cost = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
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
        update_T1_func,
    )

    return (vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params), cost
