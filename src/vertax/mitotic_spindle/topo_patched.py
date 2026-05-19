"""
Patched version of update_T1_bounded (VertAX) compatible with post-division
meshes produced by divide_cell.

The only change from the original is in body_fun: edge pairs where both
half-edges have face_id=0 are skipped. These are boundary ghost edges whose
topology is not compatible with T1 flip operations.
"""

from __future__ import annotations

import jax
import jax.lax
import jax.numpy as jnp
from jax import Array

ROT_ANGLE = -jnp.pi / 2
SINA = jnp.sin(ROT_ANGLE)
COSA = jnp.cos(ROT_ANGLE)


def update_T1_bounded_safe(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in,
    min_dist_T1: float,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
) -> tuple[Array, Array, Array, Array]:
    """update_T1_bounded patched to skip boundary ghost edges."""
    num_edges = heTable.shape[0] // 2
    angTable = jnp.clip(angTable, 0.017, jnp.pi / 2 - 0.001)
    scale = (min_dist_T1 + 10 ** (-3)) / 2

    def true_fun4(stacked_data):
        (idx, idx_twin, v_idx_source, v_idx_target, x1, y1, x2, y2,
         he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face,
         vertTable_, angTable_, heTable_, faceTable_) = stacked_data

        he_prev_twin = heTable_.at[he_prev, 2].get()
        he_twin_next_twin = heTable_.at[he_twin_next, 2].get()
        he_twin_next_twin_next = heTable_.at[he_twin_next_twin, 1].get()
        he_twin_next_twin_face = heTable_.at[he_twin_next_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_twin_prev, 1].set(he_twin_next)
        heTable_ = heTable_.at[he_prev_twin, 0].set(idx_twin)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 5].set(v_idx_source)
        heTable_ = heTable_.at[he_twin_next, 0].set(he_twin_prev)
        heTable_ = heTable_.at[he_twin_next, 3].set(v_idx_target)
        heTable_ = heTable_.at[he_twin_next_twin, 1].set(idx_twin)
        heTable_ = heTable_.at[he_twin_next_twin, 4].set(v_idx_target)
        heTable_ = heTable_.at[idx, 0].set(0)
        heTable_ = heTable_.at[idx, 1].set(0)
        heTable_ = heTable_.at[idx, 3].set(0)
        heTable_ = heTable_.at[idx, 4].set(1)
        heTable_ = heTable_.at[idx, 7].set(0)
        heTable_ = heTable_.at[idx_twin, 0].set(he_twin_next_twin)
        heTable_ = heTable_.at[idx_twin, 1].set(he_twin_next_twin_next)
        heTable_ = heTable_.at[idx_twin, 3].set(0)
        heTable_ = heTable_.at[idx_twin, 4].set(1)
        heTable_ = heTable_.at[idx_twin, 5].set(v_idx_target)
        heTable_ = heTable_.at[idx_twin, 6].set(v_idx_source)
        heTable_ = heTable_.at[idx_twin, 7].set(he_twin_next_twin_face)
        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        faceTable_ = faceTable_.at[he_twin_face, 0].set(he_twin_next)
        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)
        angTable_ = angTable_.at[idx_twin // 2].set(0.017)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun4(stacked_data):
        (idx, idx_twin, v_idx_source, v_idx_target, x1, y1, x2, y2,
         he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face,
         vertTable_, angTable_, heTable_, faceTable_) = stacked_data

        he_prev_twin = heTable_.at[he_prev, 2].get()
        he_twin_prev_twin = heTable_.at[he_twin_prev, 2].get()
        he_next_twin = heTable_.at[he_next, 2].get()
        he_twin_next_twin = heTable_.at[he_twin_next, 2].get()
        he_next_twin_next = heTable_.at[he_next_twin, 1].get()
        he_next_twin_face = heTable_.at[he_next_twin, 7].get()
        he_twin_next_twin_next = heTable_.at[he_twin_next_twin, 1].get()
        he_twin_next_twin_face = heTable_.at[he_twin_next_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_twin_prev, 1].set(he_twin_next)
        heTable_ = heTable_.at[he_prev_twin, 0].set(idx_twin)
        heTable_ = heTable_.at[he_twin_prev_twin, 0].set(idx)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 3].set(v_idx_source)
        heTable_ = heTable_.at[he_twin_next, 0].set(he_twin_prev)
        heTable_ = heTable_.at[he_twin_next, 3].set(v_idx_target)
        heTable_ = heTable_.at[he_next_twin, 1].set(idx)
        heTable_ = heTable_.at[he_next_twin, 4].set(v_idx_source)
        heTable_ = heTable_.at[he_twin_next_twin, 1].set(idx_twin)
        heTable_ = heTable_.at[he_twin_next_twin, 4].set(v_idx_target)
        heTable_ = heTable_.at[idx, 0].set(he_next_twin)
        heTable_ = heTable_.at[idx, 1].set(he_next_twin_next)
        heTable_ = heTable_.at[idx, 7].set(he_next_twin_face)
        heTable_ = heTable_.at[idx_twin, 0].set(he_twin_next_twin)
        heTable_ = heTable_.at[idx_twin, 1].set(he_twin_next_twin_next)
        heTable_ = heTable_.at[idx_twin, 7].set(he_twin_next_twin_face)
        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        faceTable_ = faceTable_.at[he_twin_face, 0].set(he_twin_next)
        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)
        return vertTable_, angTable_, heTable_, faceTable_

    def true_fun3(stacked_data):
        (idx, idx_twin, v_idx_source, v_idx_target, x1, y1, x2, y2,
         he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face,
         vertTable_, angTable_, heTable_, faceTable_) = stacked_data

        he_twin_prev_twin = heTable_.at[he_twin_prev, 2].get()
        he_next_twin = heTable_.at[he_next, 2].get()
        he_next_twin_next = heTable_.at[he_next_twin, 1].get()
        he_next_twin_face = heTable_.at[he_next_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_twin_prev, 1].set(he_twin_next)
        heTable_ = heTable_.at[he_twin_prev_twin, 0].set(idx)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 3].set(v_idx_source)
        heTable_ = heTable_.at[he_twin_next, 0].set(he_twin_prev)
        heTable_ = heTable_.at[he_twin_next, 5].set(v_idx_target)
        heTable_ = heTable_.at[he_next_twin, 1].set(idx)
        heTable_ = heTable_.at[he_next_twin, 4].set(v_idx_source)
        heTable_ = heTable_.at[idx, 0].set(he_next_twin)
        heTable_ = heTable_.at[idx, 1].set(he_next_twin_next)
        heTable_ = heTable_.at[idx, 3].set(0)
        heTable_ = heTable_.at[idx, 4].set(1)
        heTable_ = heTable_.at[idx, 5].set(v_idx_source)
        heTable_ = heTable_.at[idx, 6].set(v_idx_target)
        heTable_ = heTable_.at[idx, 7].set(he_next_twin_face)
        heTable_ = heTable_.at[idx_twin, 0].set(0)
        heTable_ = heTable_.at[idx_twin, 1].set(0)
        heTable_ = heTable_.at[idx_twin, 3].set(0)
        heTable_ = heTable_.at[idx_twin, 4].set(1)
        heTable_ = heTable_.at[idx_twin, 7].set(0)
        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        faceTable_ = faceTable_.at[he_twin_face, 0].set(he_twin_next)
        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)
        angTable_ = angTable_.at[idx // 2].set(0.017)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun3(stacked_data):
        _, _, _, _, _, _, _, _, _, he_next, _, _, _, _, _, _, heTable_, _ = stacked_data
        externalize2 = heTable_.at[he_next, 3].get() == 0
        return jax.lax.cond(externalize2, true_fun4, false_fun4, stacked_data)

    def true_fun2(stacked_data):
        (idx, idx_twin, v_idx_source, v_idx_target, x1, y1, x2, y2,
         he_prev, he_next, _, _, he_face, _, vertTable_, angTable_, heTable_, faceTable_) = stacked_data

        he_prev_twin = heTable_.at[he_prev, 2].get()
        he_next_twin = heTable_.at[he_next, 2].get()
        he_next_twin_next = heTable_.at[he_next_twin, 1].get()
        he_prev_twin_prev = heTable_.at[he_prev_twin, 0].get()
        he_next_twin_face = heTable_.at[he_next_twin, 7].get()
        he_prev_twin_face = heTable_.at[he_prev_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_prev_twin, 0].set(idx_twin)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 3].set(v_idx_source)
        heTable_ = heTable_.at[he_next_twin_next, 0].set(idx)
        heTable_ = heTable_.at[he_next_twin, 1].set(idx)
        heTable_ = heTable_.at[he_next_twin, 4].set(v_idx_source)
        heTable_ = heTable_.at[he_prev_twin_prev, 1].set(idx_twin)
        heTable_ = heTable_.at[he_prev_twin_prev, 6].set(v_idx_target)
        heTable_ = heTable_.at[idx, 0].set(he_next_twin)
        heTable_ = heTable_.at[idx, 1].set(he_next_twin_next)
        heTable_ = heTable_.at[idx, 3].set(v_idx_source)
        heTable_ = heTable_.at[idx, 4].set(v_idx_target)
        heTable_ = heTable_.at[idx, 5].set(0)
        heTable_ = heTable_.at[idx, 6].set(1)
        heTable_ = heTable_.at[idx, 7].set(he_next_twin_face)
        heTable_ = heTable_.at[idx_twin, 0].set(he_prev_twin_prev)
        heTable_ = heTable_.at[idx_twin, 1].set(he_prev_twin)
        heTable_ = heTable_.at[idx_twin, 3].set(v_idx_target)
        heTable_ = heTable_.at[idx_twin, 4].set(v_idx_source)
        heTable_ = heTable_.at[idx_twin, 7].set(he_prev_twin_face)
        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)
        angTable_ = angTable_.at[idx // 2].set(1.0)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun2(stacked_data):
        _, _, _, _, _, _, _, _, he_prev, _, _, _, _, _, _, _, heTable_, _ = stacked_data
        externalize1 = heTable_.at[he_prev, 3].get() == 0
        return jax.lax.cond(externalize1, true_fun3, false_fun3, stacked_data)

    def true_fun1(stacked_data):
        (x1, y1, x2, y2, cx, cy, x1_norm, x2_norm, _, _, v_source, v_target,
         _, _, _, _, _, _, vertTable_, angTable_, heTable_, faceTable_) = stacked_data
        sx1 = (x1 - cx) * scale / x1_norm + cx
        sy1 = (y1 - cy) * scale / x1_norm + cy
        sx2 = (x2 - cx) * scale / x2_norm + cx
        sy2 = (y2 - cy) * scale / x2_norm + cy
        vertTable_ = vertTable_.at[v_source - 2, 0].set(sx1)
        vertTable_ = vertTable_.at[v_source - 2, 1].set(sy1)
        vertTable_ = vertTable_.at[v_target - 2, 0].set(sx2)
        vertTable_ = vertTable_.at[v_target - 2, 1].set(sy2)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun1(stacked_data):
        (x1, y1, x2, y2, cx, cy, x1_norm, x2_norm, idx, idx_twin, v_source, v_target,
         he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face,
         vertTable_, angTable_, heTable_, faceTable_) = stacked_data
        rx1 = cx + COSA * (x1 - cx) - SINA * (y1 - cy)
        ry1 = cy + SINA * (x1 - cx) + COSA * (y1 - cy)
        rx2 = cx + COSA * (x2 - cx) - SINA * (y2 - cy)
        ry2 = cy + SINA * (x2 - cx) + COSA * (y2 - cy)
        nx1 = (rx1 - cx) * scale / x1_norm + cx
        ny1 = (ry1 - cy) * scale / x1_norm + cy
        nx2 = (rx2 - cx) * scale / x2_norm + cx
        ny2 = (ry2 - cy) * scale / x2_norm + cy
        internalize = heTable_.at[idx, 3].get() == 0
        return jax.lax.cond(
            internalize, true_fun2, false_fun2,
            (idx, idx_twin, v_source, v_target, nx1, ny1, nx2, ny2,
             he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face,
             vertTable_, angTable_, heTable_, faceTable_),
        )

    def true_fun0(stacked_data):
        x1, y1, x2, y2, idx, v_source, v_target, vertTable_, angTable_, heTable_, faceTable_ = stacked_data
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        x1_norm = jnp.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        x2_norm = jnp.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        he = heTable_[idx]
        idx_twin = he[2]
        he_twin = heTable_[idx_twin]
        he_prev = he[0]; he_next = he[1]
        he_twin_prev = he_twin[0]; he_twin_next = he_twin[1]
        he_face = he[7]; he_twin_face = he_twin[7]
        emergency = jnp.logical_and(heTable_.at[he_prev, 3].get() == 0, heTable_.at[he_next, 3].get() == 0)
        emergency = jnp.logical_or(emergency, heTable_.at[he_next, 1].get() == he_prev)
        emergency = jnp.logical_or(emergency, heTable_.at[he_twin_next, 1].get() == he_twin_prev)
        emergency = jnp.logical_or(emergency, jnp.sum(heTable_[:, 5] != 0) == 2)
        return jax.lax.cond(
            emergency, true_fun1, false_fun1,
            (x1, y1, x2, y2, cx, cy, x1_norm, x2_norm, idx, idx_twin, v_source, v_target,
             he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face,
             vertTable_, angTable_, heTable_, faceTable_),
        )

    def false_fun0(stacked_data):
        _, _, _, _, _, _, _, vertTable_, angTable_, heTable_, faceTable_ = stacked_data
        return vertTable_, angTable_, heTable_, faceTable_

    def body_fun(i, stacked_data):
        vertTable_, angTable_, heTable_, faceTable_ = stacked_data

        selector = (heTable_.at[2 * i, 3].get() + heTable_.at[2 * i, 5].get()) == 0
        idx = 2 * i + selector.astype("int32")

        # PATCH: skip boundary ghost pairs (face_id=0 on both sides).
        # These are twins of surface edges and cannot undergo a T1 flip
        # without corrupting the topology.
        face_a = heTable_.at[2 * i, 7].get()
        face_b = heTable_.at[2 * i + 1, 7].get()
        is_ghost_pair = jnp.logical_and(face_a == 0, face_b == 0)

        v_source = heTable_.at[idx, 3].get() + heTable_.at[idx, 5].get()
        v_target = heTable_.at[idx, 4].get() + heTable_.at[idx, 6].get() - 1
        x0 = vertTable_.at[v_source - 2, 0].get()
        y0 = vertTable_.at[v_source - 2, 1].get()
        x1 = vertTable_.at[v_target - 2, 0].get()
        y1 = vertTable_.at[v_target - 2, 1].get()
        dist = jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Trigger T1 only if the edge is short AND not a ghost edge
        T1_trigger = jnp.logical_and(dist < min_dist_T1, ~is_ghost_pair)

        return jax.lax.cond(
            T1_trigger, true_fun0, false_fun0,
            (x0, y0, x1, y1, idx, v_source, v_target, vertTable_, angTable_, heTable_, faceTable_),
        )

    return jax.lax.fori_loop(0, num_edges, body_fun, (vertTable, angTable, heTable, faceTable))
