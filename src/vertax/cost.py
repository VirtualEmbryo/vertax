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
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.lax import fori_loop
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from vertax.energy import MAX_EDGES_IN_ANY_FACE
from vertax.geo import get_area, get_cluster_moments_bounded, min_image_sq_dist

if TYPE_CHECKING:
    from vertax.bilevelopt.pbcbop import PbcBilevelOptimizer
    from vertax.meshes.pbc_mesh import PbcMesh

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


#############################
# HESSIAN-WEIGHTED COST C_W #
#############################


def hessian_at_target(
    target: "PbcMesh",
    energy: Callable,
    edges_params: Array | None = None,
) -> Array:
    r"""Hessian of the inner energy at the target geometry, divided by its spectral norm.

    The Hessian $\mathbf H=\nabla^2_{\mathbf X}\mathcal E$ is taken with respect to the flattened
    vertex positions $\mathbf X=(x_1,y_1,x_2,y_2,\dots)$ of ``target``, at its own topology and box,
    with the target's ``vertices_params`` and ``faces_params`` (for $\mathcal E_2$, the target
    areas $A^0$). It is symmetrized, $(\mathbf H+\mathbf H^{\mathsf T})/2$, and divided by its
    spectral norm $\max_i|\lambda_i(\mathbf H)|$, so that its largest eigenvalue in magnitude is 1.

    Args:
        target: The target mesh. Its vertex positions are converted to JAX's default float dtype;
            use `float64` (`jax_enable_x64`) for anything that depends on the small eigenvalues.
        energy: The inner energy with the signature of `PbcBilevelOptimizer.loss_function_inner`,
            ``energy(vertTable, heTable, faceTable, vert_params, he_params, face_params)``.
        edges_params: The edge parameters at which the Hessian is evaluated. ``None`` (the default)
            sets every one of them to 1: for a line-tension energy every tension is one, so the
            weighting uses the observed geometry and nothing of the tensions being inferred.

    Returns:
        The normalized Hessian, shape ``(2 * nb_vertices, 2 * nb_vertices)``.
    """
    dtype = jnp.result_type(float)
    vertices = jnp.asarray(target.vertices, dtype=dtype)
    shape = vertices.shape
    if edges_params is None:
        edges_params = jnp.ones(len(target.edges_params), dtype=dtype)
    vert_params = jnp.asarray(target.vertices_params, dtype=dtype)
    face_params = jnp.asarray(target.faces_params, dtype=dtype)

    def energy_of_flat_positions(x: Array) -> Array:
        return energy(x.reshape(shape), target.edges, target.faces, vert_params, edges_params, face_params)

    hess = jax.hessian(energy_of_flat_positions)(vertices.ravel())
    hess = 0.5 * (hess + hess.T)
    return hess / jnp.max(jnp.abs(jnp.linalg.eigvalsh(hess)))


def make_cost_hessian_weighted(
    target: "PbcMesh",
    energy: Callable,
    edges_params: Array | None = None,
) -> Callable:
    r"""Hessian-weighted vertex cost $\mathcal C_W$ for a `PbcMesh`, in the periodic outer-cost signature.

    $$
    \mathcal C_W=\tfrac12\bigl\lVert\widehat{\mathbf H}\,(\mathbf X-\mathbf X_{\text{target}})\bigr\rVert^2 ,
    $$

    where $\widehat{\mathbf H}$ is `hessian_at_target` (by default with every edge parameter set to
    1) and each vertex displacement $\mathbf X-\mathbf X_{\text{target}}$ is taken to the nearest
    periodic image of the target vertex, as in `cost_v2v`. The vertex cost `cost_v2v` is
    $\tfrac12\lVert\mathbf H^{+}\mathbf M\,\delta\boldsymbol\theta\rVert^2$ to first order in the
    parameter error $\delta\boldsymbol\theta$ ($\mathbf M$, the derivative of the forces with respect
    to the parameters), so the soft modes of the tissue, with small Hessian eigenvalues, amplify any
    geometric error. With the Hessian $\mathbf H$ at the true parameters in place of
    $\widehat{\mathbf H}$, the same expansion gives $\tfrac12\lVert\mathbf M\,\delta\boldsymbol\theta\rVert^2$,
    the force residual, which the soft modes do not amplify; $\widehat{\mathbf H}$ approximates
    $\mathbf H$ up to its normalization, without using the parameters being inferred. The
    normalization multiplies the cost by a constant and keeps its scale comparable to that of
    `cost_v2v`.

    $\widehat{\mathbf H}$ is computed once, from ``target``, and belongs to its topology: the cost
    compares vertex $i$ of the mesh with vertex $i$ of the target, and is meant for a mesh with the
    target topology, kept fixed (``update_T1 = False``). From another topology, use the staged
    schedule of `switch_cost_at_target_topology`.

    Args:
        target: The target mesh; the cost compares with the target vertices the optimizer passes,
            which are normally ``target.vertices``.
        energy: The inner energy, in the signature of `PbcBilevelOptimizer.loss_function_inner`.
        edges_params: The edge parameters of the Hessian; ``None`` sets them all to 1.

    Returns:
        ``cost_hessian_weighted(vertTable, heTable, faceTable, width, height, vertTable_target,
        heTable_target, faceTable_target, selected_verts, selected_hes, selected_faces,
        image_target)``, for `PbcBilevelOptimizer.loss_function_outer`. With ``selected_verts``,
        the sum runs over the rows of $\widehat{\mathbf H}(\mathbf X-\mathbf X_{\text{target}})$
        that belong to the selected vertices, the components of the force residual there; the
        displacement of every vertex still enters through $\widehat{\mathbf H}$. The edge and face
        selections and ``image_target`` are not used. The normalized Hessian is the attribute
        ``hessian`` of the returned function.

    Example:
        ```python
        optimizer.loss_function_outer = make_cost_hessian_weighted(target, optimizer.loss_function_inner)
        optimizer.update_T1 = False
        ```
    """
    hessian = hessian_at_target(target, energy, edges_params)

    def cost_hessian_weighted(
        vertTable: Array,
        heTable: Array,  # noqa: ARG001
        faceTable: Array,  # noqa: ARG001
        width: float,
        height: float,
        vertTable_target: Array,
        heTable_target: Array | None = None,  # noqa: ARG001
        faceTable_target: Array | None = None,  # noqa: ARG001
        selected_verts: Array | None = None,
        _selected_hes: Array | None = None,
        _selected_faces: Array | None = None,
        _image_target: Array | None = None,
    ) -> Array:
        d = vertTable - vertTable_target
        dx = jnp.stack(
            [d[:, 0] - width * jnp.round(d[:, 0] / width), d[:, 1] - height * jnp.round(d[:, 1] / height)], axis=1
        ).ravel()
        hdx = hessian @ dx
        if selected_verts is not None:
            hdx = hdx.reshape(-1, 2)[selected_verts]
        return 0.5 * jnp.sum(hdx * hdx)

    cost_hessian_weighted.hessian = hessian  # type: ignore[attr-defined]
    return cost_hessian_weighted


def switch_cost_at_target_topology(optimizer: "PbcBilevelOptimizer", mesh: "PbcMesh", cost: Callable) -> bool:
    r"""Second stage of the staged schedule: at the first match with the target topology, freeze it and change cost.

    The staged schedule infers parameters from a mesh whose topology differs from the target's. It
    minimizes a cost without a fixed metric, normally `cost_v2v`, with T1 transitions active, until
    the topology of the mesh first matches that of the target ($d_{\text{topo}}=0$,
    `topological_distance`); it then disables T1 transitions and minimizes ``cost``, normally
    `make_cost_hessian_weighted`, whose Hessian belongs to the target topology and changes
    discontinuously at a T1 transition. Switching at the first match matters: under `cost_v2v` the
    topology can match and unmatch again, so a rule that waits for several matching epochs switches
    late or never.

    Call it after every epoch. It compares ``mesh`` with the optimizer's target
    (``vertices_target``, ``edges_target``, ``faces_target``); once the switch has been made, it
    leaves the optimizer as it is and returns ``True`` without computing the distance again. To cap
    the first stage at a number of epochs, make the switch yourself when the cap is reached
    (``optimizer.update_T1 = False``; ``optimizer.loss_function_outer = cost``).

    Args:
        optimizer: The `PbcBilevelOptimizer` that runs the inference, with its first-stage cost set.
        mesh: The mesh being optimized, after the epoch.
        cost: The cost of the second stage.

    Returns:
        ``True`` if the second stage is in effect (the switch happened now or earlier), ``False`` if
        the mesh does not have the target topology yet.

    Example:
        ```python
        optimizer.loss_function_outer = cost_v2v
        cost_w = make_cost_hessian_weighted(target, optimizer.loss_function_inner)
        for epoch in range(n_epochs):
            optimizer.bilevel_optimization(mesh)
            switch_cost_at_target_topology(optimizer, mesh, cost_w)
        ```
    """
    if optimizer.loss_function_outer is cost and not optimizer.update_T1:
        return True
    distance = topological_distance(
        mesh.vertices,
        mesh.edges,
        mesh.faces,
        mesh.width,
        mesh.height,
        optimizer.vertices_target,
        optimizer.edges_target,
        optimizer.faces_target,
    )
    if distance != 0:
        return False
    optimizer.update_T1 = False
    optimizer.loss_function_outer = cost
    return True
