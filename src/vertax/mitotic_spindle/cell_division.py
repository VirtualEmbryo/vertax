"""
JAX-compiled cell division (ported from bounded_division).

Public API:
  divide_cell(mesh, face_id, c, u) -> (BoundedMesh, new_face_id)

    mesh    : BoundedMesh before division
    face_id : index of the cell to divide
    c       : spindle centre (unused — bounded_division recomputes centroid internally)
    u       : spindle axis unit vector [ux, uy]

The axis angle alpha = atan2(uy, ux) is passed to cell_division() as divAxes.
The new BoundedMesh is built from the padded tables returned by cell_division().
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, jit

from vertax import BoundedMesh


# ---------------------------------------------------------------------------
# Geometry helpers (verbatim from bounded_division)
# ---------------------------------------------------------------------------

@jit
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@jit
def get_intersection(he: Array, curv: Array, centroid: Array, axis: Array,
                     vertTable_: Array, angTable_: Array, heTable_: Array) -> Array:
    d = jnp.array([jnp.cos(axis), jnp.sin(axis)])

    def line_inter(_):
        p1 = vertTable_.at[heTable_.at[he, 3].get() - 2, :].get()
        p2 = vertTable_.at[heTable_.at[he, 4].get() - 2, :].get()
        v1 = p1 - centroid
        v2 = p2 - centroid
        c1 = cross2d(d, v1)
        c2 = cross2d(d, v2)
        crosses_line = c1 * c2 <= 0.0

        def compute_intersection(_):
            v = p2 - p1
            denom = cross2d(d, v)

            def no_solution(_):
                return jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.int32(0)

            def solve(_):
                t = cross2d(p1 - centroid, v) / denom
                point = centroid + t * d
                return point, jnp.array([0.0, 0.0]), jnp.int32(1)

            return jax.lax.cond(jnp.abs(denom) < 1e-8, no_solution, solve, operand=None)

        def skip(_):
            return jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.int32(0)

        return jax.lax.cond(crosses_line, compute_intersection, skip, operand=None)

    def circle_inter(_):
        p1 = vertTable_.at[heTable_.at[he, 5].get() - 2, :].get()
        p2 = vertTable_.at[heTable_.at[he, 6].get() - 2, :].get()
        alpha = angTable_.at[he // 2].get()
        v = p2 - p1
        L = jnp.linalg.norm(v)
        m = 0.5 * (p1 + p2)
        r = L / (2 * jnp.sin(alpha))
        h = jnp.sqrt(jnp.maximum(r**2 - (L/2)**2, 0.0))
        perp = jnp.array([-v[1], v[0]]) / L
        center = m + h * perp
        cos_alpha = jnp.cos(2 * alpha)
        m0 = centroid - center
        b = 2.0 * jnp.dot(d, m0)
        c_val = jnp.dot(m0, m0) - r**2
        disc = b**2 - 4.0 * c_val

        def no_intersection(_):
            return jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.int32(0)

        def solve(_):
            sqrt_disc = jnp.sqrt(disc)
            t1 = (-b - sqrt_disc) * 0.5
            t2 = (-b + sqrt_disc) * 0.5
            i1 = centroid + t1 * d
            i2 = centroid + t2 * d
            s_vec = p1 - center
            t_vec = p2 - center

            def on_arc(p):
                p_vec = p - center
                cond1 = jnp.dot(s_vec, p_vec) >= r * r * cos_alpha
                cond2 = jnp.dot(t_vec, p_vec) >= r * r * cos_alpha
                side_p = cross2d(v, p - p1)
                side_c = cross2d(v, center - p1)
                cond3 = side_p * side_c <= 0.0
                return cond1 & cond2 & cond3

            valid1 = on_arc(i1)
            valid2 = on_arc(i2)
            no_cross = jnp.array([0.0, 0.0])
            i1 = jnp.where(valid1, i1, no_cross)
            i2 = jnp.where(valid2, i2, no_cross)
            count = valid1.astype(jnp.int32) + valid2.astype(jnp.int32)
            return i1, i2, count

        return jax.lax.cond(disc <= 0.0, no_intersection, solve, operand=None)

    return jax.lax.cond(curv, circle_inter, line_inter, operand=None)


@jit
def centroid_contributions(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    v_source = heTable.at[he, 3].get() + heTable.at[he, 5].get()
    v_target = heTable.at[he, 4].get() + heTable.at[he, 6].get() - 1
    p_source = vertTable.at[v_source - 2].get()
    p_target = vertTable.at[v_target - 2].get()
    ang = angTable.at[he // 2].get()
    mask = heTable.at[he, 5].get() != 0
    chordlen = jnp.sqrt(jnp.sum((p_source - p_target) ** 2))
    area_contribution = p_source[0] * p_target[1] - p_target[0] * p_source[1]
    twice_midpoint = (p_source + p_target)
    centroid_contribution = area_contribution * twice_midpoint
    sin_ang = jnp.sin(ang)
    sin_ang_2 = sin_ang ** 2
    arc_area = (chordlen ** 2) * (ang - sin_ang * jnp.cos(ang)) / (4 * sin_ang_2) * mask
    edge_vector = p_target - p_source
    d_midpoint_to_center = jnp.sqrt(1 / sin_ang_2 - 1) / 2
    vector_midpoint_to_center = jnp.array([-edge_vector[1], edge_vector[0]])
    center = twice_midpoint / 2 + vector_midpoint_to_center * d_midpoint_to_center
    d_center_to_centroid = 2 * sin_ang_2 / (3 * (2 * ang - jnp.sin(2 * ang)))
    arc_centroid = center - vector_midpoint_to_center * d_center_to_centroid
    weighted_arc_centroid = arc_centroid * arc_area
    return area_contribution, centroid_contribution, arc_area * mask, weighted_arc_centroid


@jit
def get_centroid(start_he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    poly_area, poly_centroid, area, centroid = centroid_contributions(start_he, vertTable, angTable, heTable)
    initial_carry = (start_he, poly_area, poly_centroid, area, centroid, False)
    xs = jnp.arange(16)

    def scan_body(carry, i):
        previous_he, poly_a, poly_c, total_a, total_c, has_stopped = carry
        current_he = heTable.at[previous_he, 1].get()
        is_start_node = (current_he == start_he)
        p_a, p_c, t_a, t_c = centroid_contributions(current_he, vertTable, angTable, heTable)
        has_stopped = has_stopped | is_start_node
        new_poly_a = jnp.where(has_stopped, poly_a, poly_a + p_a)
        new_poly_c = jnp.where(has_stopped, poly_c, poly_c + p_c)
        new_total_a = jnp.where(has_stopped, total_a, total_a + t_a)
        new_total_c = jnp.where(has_stopped, total_c, total_c + t_c)
        return (current_he, new_poly_a, new_poly_c, new_total_a, new_total_c, has_stopped), 0.0

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)
    _, poly_area, poly_centroid, area, centroid, _ = final_carry
    signed_poly_area = poly_area / 2
    poly_centroid = jnp.where(signed_poly_area > 1e-6, poly_centroid / 6 / signed_poly_area, jnp.array([0.0, 0.0]))
    poly_area = jnp.abs(signed_poly_area)
    return (centroid + poly_centroid * poly_area) / (area + poly_area)


# ---------------------------------------------------------------------------
# Core division (verbatim from bounded_division)
# ---------------------------------------------------------------------------

def cell_division(vertTable: Array,
                  angTable: Array,
                  heTable: Array,
                  faceTable: Array,
                  divOrders: Array,
                  divAxes: Array) -> Array:
    num_faces   = faceTable.shape[0]
    num_verts   = vertTable.shape[0]
    num_edges   = angTable.size
    num_divides = divOrders.size
    padded_vertTable = jnp.pad(vertTable, ((0, 2 * num_divides), (0, 0)))
    padded_angTable = jnp.pad(angTable, (0, 3 * num_divides))
    padded_heTable = jnp.pad(heTable, ((0, 6 * num_divides), (0, 0)))
    padded_faceTable = jnp.pad(faceTable, ((0, num_divides), (0, 0)))
    new_he_idxes = jnp.zeros(2 * num_divides, dtype='int32')
    new_face_idxes = jnp.zeros(num_divides, dtype='int32')

    def update_face_col(he_i, he_f, heTable_, face):
        initial_carry = (heTable_, he_i, he_i == he_f)
        xs = jnp.arange(16)

        def scan_body(carry, i):
            new_heTable, previous_he, has_stopped = carry
            new_heTable = jax.lax.cond(has_stopped, lambda t: t, lambda t: t.at[previous_he, 7].set(face), new_heTable)
            next_he = new_heTable.at[previous_he, 1].get()
            is_final_node = (next_he == he_f)
            has_stopped = has_stopped | is_final_node
            return (new_heTable, next_he, has_stopped), 0.0

        final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)
        final_table, _, _ = final_carry
        return final_table

    def true_fun1(stacked_data):
        vertTable_, angTable_, heTable_, faceTable_, he_idxes_, face_idxes_, count_divides, div_face, final_inter1, final_inter2, final_he1, _, _, _ = stacked_data

        new_vert1 = num_verts + 2 * count_divides
        new_vert2 = new_vert1 + 1
        vertTable_ = vertTable_.at[new_vert1, 0].set(final_inter1[0])
        vertTable_ = vertTable_.at[new_vert1, 1].set(final_inter1[1])
        vertTable_ = vertTable_.at[new_vert2, 0].set(final_inter2[0])
        vertTable_ = vertTable_.at[new_vert2, 1].set(final_inter2[1])
        source1 = heTable_.at[final_he1, 3].get() + heTable_.at[final_he1, 5].get()
        target1 = heTable_.at[final_he1, 4].get() + heTable_.at[final_he1, 6].get() - 1
        p_source = vertTable_.at[source1 - 2].get()
        p_target = vertTable_.at[target1 - 2].get()
        test_dir = final_inter1 - p_source
        test1 = cross2d(test_dir, p_target - p_source)
        test2 = cross2d(test_dir, final_inter2 - p_source)
        ordering = test1 * test2 > 0
        len1 = jnp.linalg.norm(p_source - jnp.where(ordering, final_inter1, final_inter2))
        len2 = jnp.linalg.norm(p_target - jnp.where(ordering, final_inter2, final_inter1))
        leni = jnp.linalg.norm(final_inter1 - final_inter2)
        total_len = len1 + len2 + leni
        ratio11 = len1 / total_len
        ratio12 = len2 / total_len
        ratioi = leni / total_len
        angle1 = angTable_.at[final_he1 // 2].get()
        angTable_ = angTable_.at[num_edges + 3 * count_divides].set(1.0)
        angTable_ = angTable_.at[num_edges + 3 * count_divides + 1].set(angle1 * ratio11)
        angTable_ = angTable_.at[num_edges + 3 * count_divides + 2].set(angle1 * ratio12)
        angTable_ = angTable_.at[final_he1 // 2].set(angle1 * ratioi)
        new_vert1_off = new_vert1 + 2
        new_vert2_off = new_vert2 + 2
        near_source = jnp.where(ordering, new_vert1_off, new_vert2_off)
        near_target = jnp.where(ordering, new_vert2_off, new_vert1_off)
        prev_base_prev1 = heTable_.at[final_he1, 0].get()
        prev_base_next1 = heTable_.at[final_he1, 1].get()
        new_he0 = 2 * num_edges + 6 * count_divides
        new_he1 = new_he0 + 1
        new_he2 = new_he1 + 1
        new_he3 = new_he2 + 1
        new_he4 = new_he3 + 1
        new_he5 = new_he4 + 1
        new_face = num_faces + count_divides
        heTable_ = heTable_.at[final_he1, 0].set(new_he0)
        heTable_ = heTable_.at[final_he1, 1].set(new_he0)
        heTable_ = heTable_.at[final_he1, 5].set(near_source)
        heTable_ = heTable_.at[final_he1, 6].set(near_target)
        heTable_ = heTable_.at[final_he1, 7].set(new_face)
        heTable_ = heTable_.at[new_he0, 0].set(final_he1)
        heTable_ = heTable_.at[new_he0, 1].set(final_he1)
        heTable_ = heTable_.at[new_he0, 2].set(new_he1)
        heTable_ = heTable_.at[new_he0, 3].set(near_target)
        heTable_ = heTable_.at[new_he0, 4].set(near_source)
        heTable_ = heTable_.at[new_he0, 6].set(1)
        heTable_ = heTable_.at[new_he0, 7].set(new_face)
        heTable_ = heTable_.at[new_he1, 0].set(new_he2)
        heTable_ = heTable_.at[new_he1, 1].set(new_he4)
        heTable_ = heTable_.at[new_he1, 2].set(new_he0)
        heTable_ = heTable_.at[new_he1, 3].set(near_source)
        heTable_ = heTable_.at[new_he1, 4].set(near_target)
        heTable_ = heTable_.at[new_he1, 6].set(1)
        heTable_ = heTable_.at[new_he1, 7].set(div_face)
        heTable_ = heTable_.at[new_he2, 0].set(prev_base_prev1)
        heTable_ = heTable_.at[new_he2, 1].set(new_he1)
        heTable_ = heTable_.at[new_he2, 2].set(new_he3)
        heTable_ = heTable_.at[new_he2, 4].set(1)
        heTable_ = heTable_.at[new_he2, 5].set(source1)
        heTable_ = heTable_.at[new_he2, 6].set(near_source)
        heTable_ = heTable_.at[new_he2, 7].set(div_face)
        heTable_ = heTable_.at[new_he3, 2].set(new_he2)
        heTable_ = heTable_.at[new_he3, 4].set(1)
        heTable_ = heTable_.at[new_he3, 6].set(1)
        heTable_ = heTable_.at[new_he4, 0].set(new_he1)
        heTable_ = heTable_.at[new_he4, 1].set(prev_base_next1)
        heTable_ = heTable_.at[new_he4, 2].set(new_he5)
        heTable_ = heTable_.at[new_he4, 4].set(1)
        heTable_ = heTable_.at[new_he4, 5].set(near_target)
        heTable_ = heTable_.at[new_he4, 6].set(target1)
        heTable_ = heTable_.at[new_he4, 7].set(div_face)
        heTable_ = heTable_.at[new_he5, 2].set(new_he4)
        heTable_ = heTable_.at[new_he5, 4].set(1)
        heTable_ = heTable_.at[new_he5, 6].set(1)
        heTable_ = heTable_.at[prev_base_prev1, 1].set(new_he2)
        heTable_ = heTable_.at[prev_base_next1, 0].set(new_he4)
        faceTable_ = faceTable_.at[div_face, 0].set(new_he1)
        faceTable_ = faceTable_.at[new_face, 0].set(final_he1)
        he_idxes_ = he_idxes_.at[2 * count_divides].set(final_he1)
        he_idxes_ = he_idxes_.at[2 * count_divides + 1].set(final_he1)
        face_idxes_ = face_idxes_.at[count_divides].set(div_face)
        return vertTable_, angTable_, heTable_, faceTable_, he_idxes_, face_idxes_, count_divides + 1

    def false_fun1(stacked_data):
        vertTable_, angTable_, heTable_, faceTable_, he_idxes_, face_idxes_, count_divides, div_face, final_inter1, final_inter2, final_he1, final_he2, final_curv1, final_curv2 = stacked_data

        new_vert1 = num_verts + 2 * count_divides
        new_vert2 = new_vert1 + 1
        vertTable_ = vertTable_.at[new_vert1, 0].set(final_inter1[0])
        vertTable_ = vertTable_.at[new_vert1, 1].set(final_inter1[1])
        vertTable_ = vertTable_.at[new_vert2, 0].set(final_inter2[0])
        vertTable_ = vertTable_.at[new_vert2, 1].set(final_inter2[1])
        source1 = heTable_.at[final_he1, 3].get() + heTable_.at[final_he1, 5].get()
        target1 = heTable_.at[final_he1, 4].get() + heTable_.at[final_he1, 6].get() - 1
        len11 = jnp.linalg.norm(final_inter1 - vertTable_.at[source1 - 2].get())
        len12 = jnp.linalg.norm(final_inter1 - vertTable_.at[target1 - 2].get())
        ratio11 = len11 / (len11 + len12)
        ratio12 = 1 - ratio11
        angle1 = angTable_.at[final_he1 // 2].get()
        source2 = heTable_.at[final_he2, 3].get() + heTable_.at[final_he2, 5].get()
        target2 = heTable_.at[final_he2, 4].get() + heTable_.at[final_he2, 6].get() - 1
        len21 = jnp.linalg.norm(final_inter2 - vertTable_.at[source2 - 2].get())
        len22 = jnp.linalg.norm(final_inter2 - vertTable_.at[target2 - 2].get())
        ratio21 = len21 / (len21 + len22)
        ratio22 = 1 - ratio21
        angle2 = angTable_.at[final_he2 // 2].get()
        angTable_ = angTable_.at[num_edges + 3 * count_divides].set(1.0)
        angTable_ = angTable_.at[num_edges + 3 * count_divides + 1].set(jnp.where(final_curv1, angle1 * ratio11, 1.0))
        angTable_ = angTable_.at[num_edges + 3 * count_divides + 2].set(jnp.where(final_curv2, angle2 * ratio21, 1.0))
        angTable_ = angTable_.at[final_he1 // 2].set(jnp.where(final_curv1, angle1 * ratio12, 1.0))
        angTable_ = angTable_.at[final_he2 // 2].set(jnp.where(final_curv2, angle2 * ratio22, 1.0))
        new_vert1_off = new_vert1 + 2
        new_vert2_off = new_vert2 + 2
        he1_twin = heTable_.at[final_he1, 2].get()
        he2_twin = heTable_.at[final_he2, 2].get()
        prev_base_prev1 = heTable_.at[final_he1, 0].get()
        prev_base_next1 = heTable_.at[final_he1, 1].get()
        prev_twin_next1 = heTable_.at[he1_twin, 1].get()
        prev_base_prev2 = heTable_.at[final_he2, 0].get()
        prev_base_next2 = heTable_.at[final_he2, 1].get()
        prev_twin_next2 = heTable_.at[he2_twin, 1].get()
        new_he0 = 2 * num_edges + 6 * count_divides
        new_he1 = new_he0 + 1
        new_he2 = new_he1 + 1
        new_he3 = new_he2 + 1
        new_he4 = new_he3 + 1
        new_he5 = new_he4 + 1
        new_face = num_faces + count_divides
        twin_face1 = heTable_.at[he1_twin, 7].get()
        twin_face2 = heTable_.at[he2_twin, 7].get()
        consecutive1 = prev_base_next1 == final_he2
        consecutive2 = prev_base_next2 == final_he1
        heTable_ = heTable_.at[final_he1, 0].set(new_he0)
        heTable_ = heTable_.at[final_he1, 1].set(jnp.where(consecutive1, new_he5, prev_base_next1))
        heTable_ = heTable_.at[final_he1, 3].set(jnp.where(final_curv1, 0, new_vert1_off))
        heTable_ = heTable_.at[final_he1, 5].set(jnp.where(final_curv1, new_vert1_off, 0))
        heTable_ = heTable_.at[he1_twin, 1].set(jnp.where(final_curv1, 0, new_he2))
        heTable_ = heTable_.at[he1_twin, 4].set(jnp.where(final_curv1, 1, new_vert1_off))
        heTable_ = heTable_.at[final_he2, 0].set(new_he1)
        heTable_ = heTable_.at[final_he2, 1].set(jnp.where(consecutive2, new_he3, prev_base_next2))
        heTable_ = heTable_.at[final_he2, 3].set(jnp.where(final_curv2, 0, new_vert2_off))
        heTable_ = heTable_.at[final_he2, 5].set(jnp.where(final_curv2, new_vert2_off, 0))
        heTable_ = heTable_.at[final_he2, 7].set(new_face)
        heTable_ = heTable_.at[he2_twin, 1].set(jnp.where(final_curv2, 0, new_he4))
        heTable_ = heTable_.at[he2_twin, 4].set(jnp.where(final_curv2, 1, new_vert2_off))
        heTable_ = heTable_.at[new_he0, 0].set(new_he5)
        heTable_ = heTable_.at[new_he0, 1].set(final_he1)
        heTable_ = heTable_.at[new_he0, 2].set(new_he1)
        heTable_ = heTable_.at[new_he0, 3].set(new_vert2_off)
        heTable_ = heTable_.at[new_he0, 4].set(new_vert1_off)
        heTable_ = heTable_.at[new_he0, 6].set(1)
        heTable_ = heTable_.at[new_he0, 7].set(div_face)
        heTable_ = heTable_.at[new_he1, 0].set(new_he3)
        heTable_ = heTable_.at[new_he1, 1].set(final_he2)
        heTable_ = heTable_.at[new_he1, 2].set(new_he0)
        heTable_ = heTable_.at[new_he1, 3].set(new_vert1_off)
        heTable_ = heTable_.at[new_he1, 4].set(new_vert2_off)
        heTable_ = heTable_.at[new_he1, 6].set(1)
        heTable_ = heTable_.at[new_he1, 7].set(new_face)
        heTable_ = heTable_.at[new_he2, 0].set(jnp.where(final_curv1, 0, he1_twin))
        heTable_ = heTable_.at[new_he2, 1].set(jnp.where(final_curv1, 0, prev_twin_next1))
        heTable_ = heTable_.at[new_he2, 2].set(new_he3)
        heTable_ = heTable_.at[new_he2, 3].set(jnp.where(final_curv1, 0, new_vert1_off))
        heTable_ = heTable_.at[new_he2, 4].set(jnp.where(final_curv1, 1, source1))
        heTable_ = heTable_.at[new_he2, 6].set(1)
        heTable_ = heTable_.at[new_he2, 7].set(twin_face1)
        heTable_ = heTable_.at[new_he3, 0].set(prev_base_prev1)
        heTable_ = heTable_.at[new_he3, 1].set(new_he1)
        heTable_ = heTable_.at[new_he3, 2].set(new_he2)
        heTable_ = heTable_.at[new_he3, 3].set(jnp.where(final_curv1, 0, source1))
        heTable_ = heTable_.at[new_he3, 4].set(jnp.where(final_curv1, 1, new_vert1_off))
        heTable_ = heTable_.at[new_he3, 5].set(jnp.where(final_curv1, source1, 0))
        heTable_ = heTable_.at[new_he3, 6].set(jnp.where(final_curv1, new_vert1_off, 1))
        heTable_ = heTable_.at[new_he3, 7].set(new_face)
        heTable_ = heTable_.at[new_he4, 0].set(jnp.where(final_curv2, 0, he2_twin))
        heTable_ = heTable_.at[new_he4, 1].set(jnp.where(final_curv2, 0, prev_twin_next2))
        heTable_ = heTable_.at[new_he4, 2].set(new_he5)
        heTable_ = heTable_.at[new_he4, 3].set(jnp.where(final_curv2, 0, new_vert2_off))
        heTable_ = heTable_.at[new_he4, 4].set(jnp.where(final_curv2, 1, source2))
        heTable_ = heTable_.at[new_he4, 6].set(1)
        heTable_ = heTable_.at[new_he4, 7].set(twin_face2)
        heTable_ = heTable_.at[new_he5, 0].set(prev_base_prev2)
        heTable_ = heTable_.at[new_he5, 1].set(new_he0)
        heTable_ = heTable_.at[new_he5, 2].set(new_he4)
        heTable_ = heTable_.at[new_he5, 3].set(jnp.where(final_curv2, 0, source2))
        heTable_ = heTable_.at[new_he5, 4].set(jnp.where(final_curv2, 1, new_vert2_off))
        heTable_ = heTable_.at[new_he5, 5].set(jnp.where(final_curv2, source2, 0))
        heTable_ = heTable_.at[new_he5, 6].set(jnp.where(final_curv2, new_vert2_off, 1))
        heTable_ = heTable_.at[new_he5, 7].set(div_face)
        heTable_ = heTable_.at[prev_base_prev1, 1].set(new_he3)
        heTable_ = heTable_.at[prev_base_prev2, 1].set(new_he5)
        heTable_ = heTable_.at[jnp.where(final_curv1, he1_twin, prev_twin_next1), 0].set(jnp.where(final_curv1, 0, new_he2))
        heTable_ = heTable_.at[jnp.where(final_curv2, he2_twin, prev_twin_next2), 0].set(jnp.where(final_curv2, 0, new_he4))
        heTable_ = update_face_col(heTable_.at[final_he2, 1].get(), new_he3, heTable_, new_face)
        faceTable_ = faceTable_.at[div_face, 0].set(final_he1)
        faceTable_ = faceTable_.at[new_face, 0].set(final_he2)
        he_idxes_ = he_idxes_.at[2 * count_divides].set(final_he1)
        he_idxes_ = he_idxes_.at[2 * count_divides + 1].set(final_he2)
        face_idxes_ = face_idxes_.at[count_divides].set(div_face)
        return vertTable_, angTable_, heTable_, faceTable_, he_idxes_, face_idxes_, count_divides + 1

    def true_fun0(stacked_data):
        _, _, _, _, _, _, _, _, _, _, final_he1, final_he2, _, _ = stacked_data
        return jax.lax.cond(final_he1 == final_he2, true_fun1, false_fun1, stacked_data)

    def false_fun0(stacked_data):
        vertTable_, angTable_, heTable_, faceTable_, he_params_, face_params_, count_divides, _, _, _, _, _, _, _ = stacked_data
        return vertTable_, angTable_, heTable_, faceTable_, he_params_, face_params_, count_divides

    def body_fun(i, stacked_data):
        vertTable_, angTable_, heTable_, faceTable_, he_idxes_, face_idxes_, count_divides = stacked_data
        face = divOrders.at[i].get()
        start_he = faceTable_.at[face, 0].get()
        centroid = get_centroid(start_he, vertTable_, angTable_, heTable_)
        axis = divAxes.at[i].get()
        curv1 = heTable_.at[start_he, 3].get() == 0
        curv2 = False
        inter1, inter2, init_count = get_intersection(start_he, curv1, centroid, axis, vertTable_, angTable_, heTable_)
        he1 = jnp.where(init_count > 0, start_he, jnp.int32(-1))
        inter1 = jnp.where(init_count == 2, inter1, inter1 + inter2)
        initial_carry = (start_he, False, init_count, inter1, inter2, he1, he1, curv1, curv2)
        xs = jnp.arange(16)

        def scan_body(carry, j):
            previous_he, has_stopped, prev_count, prev_inter1, prev_inter2, prev_he1, prev_he2, prev_curv1, prev_curv2 = carry
            current_he = heTable_.at[previous_he, 1].get()
            is_start_node = (current_he == start_he)
            current_curv = heTable_.at[current_he, 3].get() == 0
            curr_inter1, curr_inter2, curr_count = get_intersection(current_he, current_curv, centroid, axis, vertTable_, angTable_, heTable_)
            has_stopped = has_stopped | is_start_node
            curr_exist = curr_count > 0
            two_found = curr_count == 2
            update = curr_exist & ~has_stopped
            first_not_found = prev_he1 < 0
            update1 = update & first_not_found
            update2 = update & (~first_not_found | two_found)
            true_inter1 = jnp.where(two_found, curr_inter1, curr_inter1 + curr_inter2)
            true_inter2 = jnp.where(two_found, curr_inter2, curr_inter1 + curr_inter2)
            new_inter1 = jnp.where(update1, true_inter1, prev_inter1)
            new_inter2 = jnp.where(update2, true_inter2, prev_inter2)
            new_he1 = jnp.where(update1, current_he, prev_he1)
            new_he2 = jnp.where(update2, current_he, prev_he2)
            new_curv1 = jnp.where(update1, current_curv, prev_curv1)
            new_curv2 = jnp.where(update2, current_curv, prev_curv2)
            new_count = jnp.where(has_stopped, prev_count, prev_count + curr_count)
            return (current_he, has_stopped, new_count, new_inter1, new_inter2, new_he1, new_he2, new_curv1, new_curv2), 0.0

        final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)
        _, _, final_count, final_inter1, final_inter2, final_he1, final_he2, final_curv1, final_curv2 = final_carry

        return jax.lax.cond(final_count == 2, true_fun0, false_fun0, (vertTable_, angTable_, heTable_, faceTable_, he_idxes_, face_idxes_, count_divides, face, final_inter1, final_inter2, final_he1, final_he2, final_curv1, final_curv2))

    return jax.lax.fori_loop(0, num_divides, body_fun, (padded_vertTable, padded_angTable, padded_heTable, padded_faceTable, new_he_idxes, new_face_idxes, 0))


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def divide_cell(mesh: BoundedMesh, face_id: int, c: np.ndarray, u: np.ndarray
                ) -> tuple['BoundedMesh', int]:
    """Divide cell face_id along axis u and return (new_mesh, new_face_id).

    c is accepted for API compatibility but ignored — bounded_division
    recomputes the centroid internally via get_centroid().

    u is the spindle axis; the cleavage line is PERPENDICULAR to it (as in real
    cytokinesis), so the cut is made along the direction orthogonal to u.

    Raises ValueError if cell_division finds fewer than 2 intersections
    (degenerate geometry — caller should try another cell).
    """
    # Cleavage line ⟂ spindle axis: cut along perp(u) = (-u_y, u_x).
    cut = np.array([-float(u[1]), float(u[0])])
    alpha = float(np.arctan2(cut[1], cut[0]))

    vT = jnp.array(mesh.vertices, dtype=jnp.float64)
    aT = jnp.array(mesh.angles, dtype=jnp.float64)
    hT = jnp.array(mesh.edges, dtype=jnp.int32)
    fT = jnp.array(mesh.faces, dtype=jnp.int32)

    divOrders = jnp.array([face_id], dtype=jnp.int32)
    divAxes   = jnp.array([alpha],   dtype=jnp.float64)
    # Cast tables to int32 to avoid int32/int64 conflicts inside cell_division (x64 mode)
    hT = hT.astype(jnp.int32)
    fT = fT.astype(jnp.int32)

    padded_vT, padded_aT, padded_hT, padded_fT, he_idxes, face_idxes, n_done = cell_division(
        vT, aT, hT, fT, divOrders, divAxes
    )

    n_done = int(n_done)
    if n_done == 0:
        raise ValueError(f"cell_division found no valid cut for face {face_id} (degenerate geometry)")

    # The original face stays at face_id; the new daughter face is appended at nb_faces.
    new_face_id = mesh.nb_faces

    # Trim padding: cell_division pads for num_divides=1
    num_verts_new  = mesh.nb_vertices + 2
    num_edges_new  = mesh.nb_edges    + 3
    num_hes_new    = num_edges_new * 2
    num_faces_new  = mesh.nb_faces    + 1

    new_mesh = BoundedMesh.__new__(BoundedMesh)
    BoundedMesh.__init__(new_mesh)

    new_mesh.vertices = jnp.array(padded_vT[:num_verts_new])
    new_mesh.angles   = jnp.array(padded_aT[:num_edges_new])
    new_mesh.edges    = jnp.array(padded_hT[:num_hes_new], dtype=jnp.int32)
    new_mesh.faces    = jnp.array(padded_fT[:num_faces_new], dtype=jnp.int32)
    new_mesh.width    = mesh.width
    new_mesh.height   = mesh.height

    # Propagate params: extend arrays for new elements
    # Vertices: 2 new vertices with mean params
    vp = np.array(mesh.vertices_params)
    mean_vp = float(np.mean(vp)) if len(vp) > 0 else 0.0
    new_vp = np.append(vp, [mean_vp, mean_vp])
    new_mesh.vertices_params = jnp.array(new_vp)

    # Edges: 3 new edges — division wall (index 0 = first new edge) gets low tension,
    # the two truncated halves inherit proportional angles set by cell_division
    ep = np.array(mesh.edges_params)
    mean_ep = float(np.mean(ep)) if len(ep) > 0 else 0.0
    # Division wall: minimum tension (sigmoid(-10)+1 ≈ 1.0)
    # Two edge segments: mean tension
    new_ep = np.append(ep, [-10.0, mean_ep, mean_ep])
    new_mesh.edges_params = jnp.array(new_ep)

    # Faces: 1 new face; copy the divided face's params for both daughter cells
    fp = np.array(mesh.faces_params)
    parent_fp = float(fp[face_id]) if len(fp) > face_id else (float(np.mean(fp)) if len(fp) > 0 else 0.6)
    new_fp = np.append(fp, [parent_fp])
    new_mesh.faces_params = jnp.array(new_fp)

    return new_mesh, new_face_id
