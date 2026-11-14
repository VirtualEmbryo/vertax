"""Cost functions collection, used for outer optimization.

A cost function can be any user-defined function but it has to respect a strict signature.

For a `PbcMesh` and `PbcBilevelOptimizer`, the cost function must have the following signature:

    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,

and return an Array (or float).

The names can vary and you can give default parameters. But the number and type of parameters is important.
You don't have to use every parameters but they all have to be here.
An unused parameters can of course also have the type None.

Same for `BoundedMesh` and `BoundedBilevelOptimizer`, but with a slightly different signature:

    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vertTable_target: Array,
    angTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,

and return an Array (or float).

Hopefully the variable names are self-explanatory.

You can create a function with this signature exactly that uses also locally-accessible external variable if you want.
"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.lax import fori_loop

from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from vertax.energy import MAX_EDGES_IN_ANY_FACE
from vertax.geo import get_area, get_cluster_moments_bounded, min_image_sq_dist

TARGET_RATIO = 2.0

##################
# COST FUNCTIONS #
##################


@partial(jit, static_argnums=(3, 4))
def cost_v2v(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost vertex to vertex. Compare the positions of given vertices to target vertices (`PbcMesh`)."""
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    def squared_distance(v: Array, vertTable: Array, vertTable_target: Array, width: float, height: float) -> Array:
        """Nearest-periodic-image squared distance for vertex ``v`` -- the C_v2v summand.

        Delegates to `vertax.geo.min_image_sq_dist`, which documents why this must not be
        respelled as a min over nine scalars, nor as ``(min sqrt(u))**2``. The
        same helper is used by the periodic T1 swap-repair criterion so the two cannot drift.
        """
        return min_image_sq_dist(vertTable[v], vertTable_target[v], width, height)

    def mapped_fn(v: Array) -> Array:
        return squared_distance(v, vertTable, vertTable_target, width, height)

    distances = vmap(mapped_fn)(selected_verts)

    return (1.0 / (2 * len(distances))) * jnp.sum(distances)


@partial(jit, static_argnums=(3, 4))
def cost_areas(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost areas : compare the areas of the given mesh versus the areas of the target mesh (`PbcMesh`).

    Mean over the selected faces (all faces if `selected_faces` is None) of the squared difference
    between the area of a face and the area of the same face in the target mesh, which must have
    the same faces.
    """
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    def mapped_fn(f: Array) -> Array:
        return (
            get_area(f, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
            - get_area(f, vertTable_target, heTable_target, faceTable_target, width, height, MAX_EDGES_IN_ANY_FACE)
        ) ** 2

    difference = vmap(mapped_fn)(selected_faces)
    return (1.0 / len(difference)) * jnp.sum(difference)


@jit
def cost_ratio(
    vertTable: Array,
    _angTable: Array | None = None,
    _heTable: Array | None = None,
    _faceTable: Array | None = None,
    _vertTable_target: Array | None = None,
    _angTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost that nudges a `BoundedMesh` to elongate along one axis while narrowing along the orthogonal axis."""
    # Compute pairwise squared distances by broadcasting
    # diff shape: (n, n, d)
    diff = vertTable[:, None, :] - vertTable[None, :, :]
    sq_dists = jnp.sum(diff**2, axis=-1)  # (n, n)
    distances = jnp.sqrt(sq_dists + 1e-12)  # small eps for numerical stability

    # For each row, get index of farthest point
    max_distances = jnp.max(distances, axis=1)
    max_idx_1 = jnp.argmax(max_distances)  # index of row with largest max distance
    long_axis = max_distances[max_idx_1]  # longest distance (already excludes diagonal)
    max_idx_2 = jnp.argmax(distances[max_idx_1])  # the farthest point from max_idx_1

    max_point_1 = vertTable[max_idx_1]
    max_point_2 = vertTable[max_idx_2]
    long_axis_vector = max_point_2 - max_point_1
    long_axis_vector = long_axis_vector / (jnp.linalg.norm(long_axis_vector) + 1e-12)

    # Short axis (perpendicular)
    short_axis_vector = jnp.array([long_axis_vector[1], -long_axis_vector[0]])

    # Project points onto short axis: signed distances
    # If vertTable is (n,2) this works. If higher dim, modify accordingly.
    signed_distances_to_long_axis = jnp.dot(vertTable - max_point_1, short_axis_vector)

    short_axis = jnp.max(signed_distances_to_long_axis) - jnp.min(signed_distances_to_long_axis)

    # Numerically stabilize division
    ratio = long_axis / (short_axis + 1e-12)
    return (ratio - TARGET_RATIO) ** 2


def aspect_ratio_along_x(vertTable: Array, angTable: Array, heTable: Array) -> Array:
    r"""Aspect ratio of a bounded cluster along the fixed x axis, $r_x=\sqrt{S_{xx}/S_{yy}}$.

    $S_{xx}$ and $S_{yy}$ are the central second moments of the region the cluster
    covers, cells and circular caps of the free interface included
    (`vertax.geo.get_cluster_moments_bounded`). For an ellipse of semi-axes ``a`` along x and ``b``
    along y, $r_x=a/b$. The axis is fixed: the same ellipse turned by an angle $\phi$
    has $r_x^2=(a^2\cos^2\phi+b^2\sin^2\phi)/(a^2\sin^2\phi+b^2\cos^2\phi)$, which is
    largest at $\phi=0$, so elongating the cluster along any other direction gives a smaller
    $r_x$. Smooth in the vertex positions and arc angles.
    """
    _, _, s = get_cluster_moments_bounded(vertTable, angTable, heTable)
    return jnp.sqrt(s[0, 0] / s[1, 1])


def aspect_ratio_principal(vertTable: Array, angTable: Array, heTable: Array) -> Array:
    r"""Aspect ratio of a bounded cluster along its own principal axes, $\sqrt{\lambda_1/\lambda_2}$.

    $\lambda_1\ge\lambda_2$ are the eigenvalues of the central second-moment matrix of the
    region the cluster covers (`vertax.geo.get_cluster_moments_bounded`), so the value does not
    depend on the orientation of the cluster, and the direction of elongation is left free. For an
    ellipse of semi-axes ``a >= b`` it is ``a/b`` at any orientation. Smooth except at an exactly
    isotropic cluster, where the two eigenvalues cross; a term of relative size 1e-12 under the
    square root keeps the gradient finite there.
    """
    _, _, s = get_cluster_moments_bounded(vertTable, angTable, heTable)
    trace = s[0, 0] + s[1, 1]
    split = jnp.sqrt((s[0, 0] - s[1, 1]) ** 2 + 4 * s[0, 1] ** 2 + (1e-12 * trace) ** 2)
    return jnp.sqrt((trace + split) / (trace - split))


def make_cost_aspect_ratio_along_x(target_ratio: float) -> Callable:
    r"""Outer cost $\mathcal C=(r_x-r_\mathrm{target})^2$ for a `BoundedMesh`, with the axis of elongation fixed to x.

    ``r_x`` is `aspect_ratio_along_x`. Use this cost when the direction of elongation is part of the
    design; with `make_cost_aspect_ratio` the direction is free and can drift during the design.
    The returned function has the bounded outer-cost signature.
    """
    target = float(target_ratio)

    def cost_aspect_ratio_along_x(
        vertTable: Array,
        angTable: Array,
        heTable: Array,
        _faceTable: Array | None = None,
        _vertTable_target: Array | None = None,
        _angTable_target: Array | None = None,
        _heTable_target: Array | None = None,
        _faceTable_target: Array | None = None,
        _selected_verts: Array | None = None,
        _selected_hes: Array | None = None,
        _selected_faces: Array | None = None,
        _image_target: Array | None = None,
    ) -> Array:
        return (aspect_ratio_along_x(vertTable, angTable, heTable) - target) ** 2

    return cost_aspect_ratio_along_x


def make_cost_aspect_ratio(target_ratio: float) -> Callable:
    r"""Outer cost $\mathcal C=(r-r_\mathrm{target})^2$ for a `BoundedMesh`, invariant to rotation.

    ``r`` is `aspect_ratio_principal`, so the cluster may elongate along any direction. Unlike
    `cost_ratio`, which measures the largest vertex distance and changes non-smoothly when the
    farthest pair of vertices changes, this cost is smooth. The returned function has the bounded
    outer-cost signature.
    """
    target = float(target_ratio)

    def cost_aspect_ratio(
        vertTable: Array,
        angTable: Array,
        heTable: Array,
        _faceTable: Array | None = None,
        _vertTable_target: Array | None = None,
        _angTable_target: Array | None = None,
        _heTable_target: Array | None = None,
        _faceTable_target: Array | None = None,
        _selected_verts: Array | None = None,
        _selected_hes: Array | None = None,
        _selected_faces: Array | None = None,
        _image_target: Array | None = None,
    ) -> Array:
        return (aspect_ratio_principal(vertTable, angTable, heTable) - target) ** 2

    return cost_aspect_ratio


@jit
def cost_checkerboard(
    vertTable: Array,
    _angTable: Array,
    heTable: Array,
    faceTable: Array,
    _vertTable_target: Array | None = None,
    _angTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost that nudges a `BoundedMesh` with different fated cells to avoid having neighboring cells with same fate.

    This leads to a checkerboard pattern when there is 2 fates.
    """

    def body_fun(i: int, current_len: Array) -> Array:
        idx = 2 * i
        he_twin = heTable[idx, 2]
        he_face = heTable[idx, 7]
        he_fate = faceTable[he_face, 1]
        he_face_twin = heTable[he_twin, 7]
        he_fate_twin = faceTable[he_face_twin, 1]
        v_source = heTable[idx, 3] - 2
        v_target = heTable[idx, 4] - 2
        pos_source = vertTable[v_source]
        pos_target = vertTable[v_target]
        return current_len + jnp.where(
            jnp.logical_and(he_fate == he_fate_twin, heTable[idx, 3] > 0), jnp.sum((pos_target - pos_source) ** 2), 0.0
        )

    n_edges = heTable.shape[0] // 2
    return fori_loop(0, n_edges, body_fun, 0.0)





def topological_distance(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> int:
    r"""Discrete topological distance ``d_topo`` between two meshes: differing cell contacts.

    Builds the dual (cell) adjacency matrix of each mesh, ``A[a, b] = 1`` iff cells ``a`` and ``b``
    share a junction, puts the two cell labellings into correspondence with the Hungarian
    algorithm (cell labels are arbitrary), and counts the cell--cell contacts on which the two
    meshes then disagree, each unordered pair once:

    $$
    d_{topo} = \sum_{a<b} \bigl| A^{(1)}_{ab} - A^{(2)}_{\sigma(a)\sigma(b)} \bigr| .
    $$

    An elementary T1 transition destroys exactly one contact and creates exactly one other, so it
    changes **two** upper-triangle entries. Therefore ``d_topo`` is always **even**, and
    ``d_topo / 2`` is the number of T1 events separating the two topologies. (Verified against the
    Figure-3 synthetic dataset, whose levels 0, 2, 4, 6, 8, 10 this function reproduces exactly.)

    **This is a measurement, not an objective.** It returns a Python ``int`` and provides no
    gradient, so it cannot be assigned to ``loss_function_outer``. Use it to quantify topological
    distance and to time phase switches in a staged schedule. Separately, any purely topological or
    edge-level cost is a strict information loss for edge-tension inference relative to a
    vertex-level cost such as `cost_v2v` -- the cell-adjacency Laplacian has a large null
    space -- so topology should not be used to infer tensions in the first place.
    """

    def build_dual_adj_np(heTable: Array, n_faces: int) -> NDArray:
        face = heTable[:, 5]
        twin = heTable[:, 2]

        face_twin = face[twin]
        mask = (face != face_twin).astype(float)

        A = np.zeros((n_faces, n_faces))

        A[face, face_twin] += mask
        A[face_twin, face] += mask

        A = np.clip(A, 0.0, 1.0)
        return A

    n1 = len(faceTable)
    n2 = len(faceTable_target)

    if n1 != n2:
        msg = "This version requires same number of faces"
        raise ValueError(msg)

    n = n1

    A1 = build_dual_adj_np(heTable, n)
    A2 = build_dual_adj_np(heTable_target, n)

    # --- Step 1: node matching via Hungarian ---
    # cost between nodes = L1 difference of adjacency rows
    C = np.sum(np.abs(A1[:, None, :] - A2[None, :, :]), axis=-1)

    row_ind, col_ind = linear_sum_assignment(C)

    # permutation: face i in A1 ↔ face col_ind[i] in A2
    P = np.zeros((n, n))
    P[row_ind, col_ind] = 1.0

    # --- Step 2: permute A2 ---
    A2_perm = P @ A2 @ P.T

    # --- Step 3: count edge mismatches ---
    diff = np.abs(A1 - A2_perm)

    # count each edge once
    diff_upper = np.triu(diff, k=1)

    loss = int(np.sum(diff_upper))

    return loss
