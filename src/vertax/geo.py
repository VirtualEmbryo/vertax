"""Geometric functions over a 2D mesh."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap


# The nine periodic images of a point: itself, plus translations by +/-width and/or +/-height.
# Order is preserved from the original cost_v2v implementation.
PBC_IMAGE_OFFSETS = (
    (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
)


def min_image_sq_dist(p: Array, q: Array, width: float, height: float) -> Array:
    """Squared distance from ``p`` to the **nearest periodic image** of ``q`` (2D, PBC).

    Under periodic boundary conditions two points can be close through a box face while far
    apart in stored coordinates, so the distance is measured to the nearest of the nine images
    of ``q`` (itself plus translations by +/-width and/or +/-height). This makes the result
    independent of which image representative happens to be stored, which is why both the
    vertex-to-vertex cost and the periodic T1 swap-repair criterion must use it.

    Implemented as one vectorised reduction over a ``(9, 2)`` offset array.

    **Do not rewrite this as ``jnp.min`` over a Python list of nine scalars.** Under ``jit``
    (inside ``vmap`` + ``grad``) XLA emitted NaN gradients for that form on real meshes -- 2 of
    24 saved dataset meshes -- whereas this form is NaN-free jitted, unjitted, and in the
    degenerate case ``p == q`` exactly. The equivalent ``(min_k sqrt(u_k))**2`` spelling is
    defective: its AD path is ``0/0 -> NaN`` when ``p == q``.
    """
    offsets = jnp.asarray(
        [[a * width, b * height] for a, b in PBC_IMAGE_OFFSETS], dtype=p.dtype
    )
    delta = p[:2] - q[:2]
    return jnp.min(jnp.sum((delta - offsets) ** 2, axis=1))


@partial(jit, static_argnums=(3, 4))
def _sum_edges(
    face: Array, heTable: Array, faceTable: Array, fun: Callable[[Array, Array], Array], max_iter: int
) -> Array:
    """Sums edge contributions for a face using lax.scan.

    Running for a fixed number of iterations (max_iter).
    """
    start_he = faceTable.at[face].get()

    # 1. Process the first edge (start_he)
    # This computes the first contribution and sets the initial state
    res_0 = fun(start_he, jnp.array([0.0, 0.0, 0.0]))

    # 2. Setup initial carry for the scan
    # The scan will process the (max_iter - 1) *remaining* edges
    # state = (last_he, cumulative_res, has_stopped_flag)
    initial_carry = (start_he, res_0, False)

    # 3. Create a dummy array for scan to iterate over
    # It will run (max_iter - 1) times
    xs = jnp.arange(max_iter - 1)

    def scan_body(carry: tuple[Array, Array, Array | bool], _: int) -> tuple[tuple[Array, Array, Array | bool], float]:
        """The body of the lax.scan.

        `carry` is the state from the *previous* iteration.
        `i` is the current loop index (from xs).
        """
        previous_he, previous_res, has_stopped = carry

        # Get the *next* half-edge in the face loop
        current_he = heTable.at[previous_he, 1].get()

        # Check if this new edge is the one we started with
        is_start_node = current_he == start_he

        # Latch the 'has_stopped' flag. Once it's True, it stays True.
        has_stopped = has_stopped | is_start_node

        # Calculate the contribution of the *current* edge,
        # using the *previous* cumulative result (for offsets)
        contribution = fun(current_he, previous_res)

        # Accumulate the result,
        # but only if the 'has_stopped' flag is not set.
        new_res = jax.lax.select(
            has_stopped,
            previous_res,  # If stopped, result is unchanged
            previous_res + contribution,  # If running, accumulate
        )

        # The new carry state for the *next* iteration
        new_carry = (current_he, new_res, has_stopped)

        # scan requires a 'y' output, we don't need it
        return new_carry, 0.0

    # Run the scan
    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)  # type: ignore

    # The final result is the cumulative 'res' from the final state
    _, final_res, _ = final_carry
    return final_res


@jit
def get_length(he: Array, vertTable: Array, heTable: Array, _faceTable: Array, width: float, height: float) -> Array:
    """Get the lengths of given half-edges."""
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()

    x0, y0 = vertTable.at[v_source, :2].get()  # source vertex
    he_offset_x1 = heTable.at[he, 6].get() * width  # target offset
    he_offset_y1 = heTable.at[he, 7].get() * height
    x1, y1 = vertTable.at[v_target, :2].get() + jnp.array([he_offset_x1, he_offset_y1])  # target vertex

    length = jnp.hypot(x1 - x0, y1 - y0)
    return length


@jit
def get_length_with_offset(
    he: Array, vertTable: Array, heTable: Array, _faceTable: Array, width: float, height: float
) -> Array:
    """Get the length and associated offsets for given half-edges."""
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0, y0 = vertTable.at[v_source, :2].get()  # source vertex
    he_offset_x1 = heTable.at[he, 6].get() * width  # target offset
    he_offset_y1 = heTable.at[he, 7].get() * height
    x1, y1 = vertTable.at[v_target, :2].get() + jnp.array([he_offset_x1, he_offset_y1])  # target vertex

    length = jnp.hypot(x1 - x0, y1 - y0)
    return jnp.array([length, he_offset_x1, he_offset_y1])


@partial(jit, static_argnums=(4, 5, 6))
def get_perimeter(
    face: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float, max_iter: int
) -> Array:
    """Get the perimeters of given faces."""

    def fun(he: Array, _: Array) -> Array:
        return get_length_with_offset(he, vertTable, heTable, faceTable, width, height)

    return _sum_edges(face, heTable, faceTable, fun, max_iter)[0]


@jit
def _compute_numerator(he: Array, res: Array, vertTable: Array, heTable: Array, width: float, height: float) -> Array:
    x_offset, y_offset = res.at[1].get(), res.at[2].get()
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get() + x_offset  # source
    y0 = vertTable.at[v_source, 1].get() + y_offset
    he_offset_x1 = heTable.at[he, 6].get() * width  # offset target
    he_offset_y1 = heTable.at[he, 7].get() * height
    x1 = vertTable.at[v_target, 0].get() + x_offset + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + y_offset + he_offset_y1

    return jnp.array([(x0 * y1) - (x1 * y0), he_offset_x1, he_offset_y1])


@partial(jit, static_argnums=(4, 5, 6))
def get_area(
    face: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float, max_iter: int
) -> Array:
    """Get area of given faces (using shoelace formula)."""

    def fun(he: Array, res: Array) -> Array:
        return _compute_numerator(he, res, vertTable, heTable, width, height)

    return 0.5 * jnp.abs(_sum_edges(face, heTable, faceTable, fun, max_iter)[0])


@jit
def _update_he(
    he: Array, vertTable: Array, heTable: Array, width: float, height: float
) -> tuple[Array, Array, Array, Array]:
    v_idx_target = heTable.at[he, 4].get()
    v_x = vertTable.at[v_idx_target, 0].get()
    v_y = vertTable.at[v_idx_target, 1].get()
    offset_x_target = jnp.where(v_x < 0.0, -1, jnp.where(v_x > width, +1, 0))
    offset_y_target = jnp.where(v_y < 0.0, -1, jnp.where(v_y > height, +1, 0))

    v_idx_source = heTable.at[he, 3].get()
    v_x = vertTable.at[v_idx_source, 0].get()
    v_y = vertTable.at[v_idx_source, 1].get()
    offset_x_source = jnp.where(v_x < 0.0, -1, jnp.where(v_x > width, +1, 0))
    offset_y_source = jnp.where(v_y < 0.0, -1, jnp.where(v_y > height, +1, 0))

    return offset_x_target, offset_y_target, offset_x_source, offset_y_source


@jit
def _move_vertex_inside(v: Array, vertTable: Array, width: float, height: float) -> tuple[Array, Array]:
    v_x = vertTable.at[v, 0].get()
    v_y = vertTable.at[v, 1].get()
    v_x = jnp.where(v_x < 0.0, v_x + width, jnp.where(v_x > width, v_x - width, v_x))
    v_y = jnp.where(v_y < 0.0, v_y + height, jnp.where(v_y > height, v_y - height, v_y))

    return v_x, v_y


@jit
def update_pbc(
    vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> tuple[Array, Array, Array]:
    """Updating vertices positions and offsets for periodic boundary conditions."""

    def mapped_offsets(he: Array) -> tuple[Array, Array, Array, Array]:
        return _update_he(he, vertTable, heTable, width, height)

    offset_x_target, offset_y_target, offset_x_source, offset_y_source = vmap(mapped_offsets)(jnp.arange(len(heTable)))
    heTable = heTable.at[:, 6].add(+offset_x_target - offset_x_source)
    heTable = heTable.at[:, 7].add(+offset_y_target - offset_y_source)

    def mapped_vertices(v: Array) -> tuple[Array, Array]:
        return _move_vertex_inside(v, vertTable, width, height)

    v_x, v_y = vmap(mapped_vertices)(jnp.arange(len(vertTable)))
    vertTable = vertTable.at[:, 0].set(v_x)
    vertTable = vertTable.at[:, 1].set(v_y)

    return vertTable, heTable, faceTable


def select_verts_hes_faces_inside(
    vertTable: Array, heTable: Array, faceTable: Array, x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[Array, Array, Array]:
    """Get a selection of all vertices, edges and faces for faces that lie totally inside the bounds."""
    selected_faces = []
    selected_hes = jnp.array([], dtype=int)
    selected_verts = jnp.array([], dtype=int)

    for face in range(len(faceTable)):
        start_he = faceTable.at[face].get()
        he = start_he

        hes_idxs = []
        verts_idxs = []
        all_inside = True  # flag to check if all vertices are inside L_box_inner

        while True:
            v_source = heTable.at[he, 3].get()
            vert_x, vert_y = vertTable.at[v_source, 0].get(), vertTable.at[v_source, 1].get()

            # check if the vertex is outside the inner box
            if not ((y_min <= vert_y <= y_max) and (x_min <= vert_x <= x_max)):
                all_inside = False
                break

            hes_idxs.append(he)
            verts_idxs.append(v_source)

            he = heTable.at[he, 1].get()
            if he == start_he:
                break

        if all_inside:
            selected_faces.append(face)
            selected_hes = jnp.concatenate((selected_hes, jnp.array(hes_idxs)))
            selected_verts = jnp.concatenate((selected_verts, jnp.array(verts_idxs)))

    # unique elements in each array
    selected_verts = jnp.unique(selected_verts)
    selected_hes = jnp.unique(selected_hes)
    selected_faces = jnp.unique(jnp.array(selected_faces))

    return selected_verts, selected_hes, selected_faces


# ==========
# Bounded
# ==========
@jit
def get_edge_length(he: Array, vertTable: Array, heTable: Array) -> Array:
    """For a bounded mesh, get the length of straight edges, and 0 for boundary edges."""
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    mask = v_source != 0
    x0 = vertTable.at[v_source, 0].get()
    y0 = vertTable.at[v_source, 1].get()
    x1 = vertTable.at[v_target, 0].get()
    y1 = vertTable.at[v_target, 1].get()

    return jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * mask


@jit
def get_chord_length(he: Array, vertTable: Array, heTable: Array) -> Array:
    """For a bounded mesh, get the chord lengths of boundary edges, and 0 for non-boundary edges."""
    v_source = heTable.at[he, 5].get()
    v_target = heTable.at[he, 6].get()
    mask = v_source != 0
    x0 = vertTable.at[v_source, 0].get()
    y0 = vertTable.at[v_source, 1].get()
    x1 = vertTable.at[v_target, 0].get()
    y1 = vertTable.at[v_target, 1].get()

    return jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * mask


@jit
def get_surface_length(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    """For a bounded mesh, get the lengths of boundary edges, and 0 for non-boundary edges."""
    chordlen = get_chord_length(he, vertTable, heTable)
    angle = angTable.at[he].get()

    return chordlen / jnp.sinc(angle / jnp.pi)


@jit
def _area_factor_between_arc_and_chord(angle: Array) -> Array:
    sinang = jnp.sin(angle)
    return (angle - sinang * jnp.cos(angle)) / (2 * sinang**2)


@jit
def _compute_contributions(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    v_source = heTable.at[he, 3].get() + heTable.at[he, 5].get()
    v_target = heTable.at[he, 4].get() + heTable.at[he, 6].get() - 1
    x0 = vertTable.at[v_source, 0].get()
    y0 = vertTable.at[v_source, 1].get()
    x1 = vertTable.at[v_target, 0].get()
    y1 = vertTable.at[v_target, 1].get()

    angle = angTable.at[he].get()
    chordlen = get_chord_length(he, vertTable, heTable)

    return jnp.array([x0 * y1 - x1 * y0, (chordlen**2) * _area_factor_between_arc_and_chord(angle)])


@jit
def get_area_bounded(face: Array, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array) -> Array:
    """Get the areas of given faces."""
    start_he = faceTable.at[face, 0].get()
    res_0 = _compute_contributions(start_he, vertTable, angTable, heTable)
    initial_carry = (start_he, res_0, False)
    xs = jnp.arange(11)

    def scan_body(carry: tuple[Array, Array, Array | bool], _: int) -> tuple[tuple[Array, Array, Array], float]:
        previous_he, previous_res, has_stopped = carry
        current_he = heTable.at[previous_he, 1].get()
        is_start_node = current_he == start_he
        has_stopped = has_stopped | is_start_node
        contribution = _compute_contributions(current_he, vertTable, angTable, heTable)
        new_res = jax.lax.select(
            has_stopped,
            previous_res,  # If stopped, result is unchanged
            previous_res + contribution,  # If running, accumulate
        )
        return (current_he, new_res, has_stopped), 0.0

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)  # type: ignore
    _, final_res, _ = final_carry

    # final_res is [shoelace_sum, arc_correction_sum]; the area is half their sum, both signed.
    # Faces are traversed counterclockwise (`BoundedMesh` builds them so and T1 transitions keep it),
    # and every arc bulges to the right of its directed chord, away from the cell, so its cap always
    # adds (the arc sum is positive: angles are clipped to [0.017, pi/2 - 0.001]). The shoelace sum is the
    # signed chord polygon. It is negative when the cell's interior vertices have crossed the chord
    # of its arc (a crescent: the area is cap - |polygon|, still positive), and the area itself is
    # negative only when the cell has folded. Taking abs of the shoelace sum would read a crescent
    # as polygon and cap on the same side of the chord, cap + |polygon|, and hide a fold.
    return 0.5 * jnp.sum(final_res)


@jit
def get_any_length(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    """Get bounded length for both surface and inside edges."""
    selector = heTable.at[he, 3].get() != 0
    flat_len = get_edge_length(he, vertTable, heTable)
    curved_len = get_surface_length(he, vertTable, angTable, heTable)
    return jnp.where(selector, flat_len, curved_len)


@jit
def get_perimeter_bounded(face: Array, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array) -> Array:
    """Get the perimeters of given faces."""
    start_he = faceTable.at[face, 0].get()
    res_0 = get_any_length(start_he, vertTable, angTable, heTable)
    initial_carry = (start_he, res_0, False)
    xs = jnp.arange(11)

    def scan_body(carry: tuple[Array, Array, Array | bool], _i: int) -> tuple[tuple[Array, Array, Array | bool], float]:
        previous_he, previous_res, has_stopped = carry
        current_he = heTable.at[previous_he, 1].get()
        is_start_node = current_he == start_he
        has_stopped = has_stopped | is_start_node
        contribution = get_any_length(current_he, vertTable, angTable, heTable)
        new_res = jax.lax.select(
            has_stopped,
            previous_res,  # If stopped, result is unchanged
            previous_res + contribution,  # If running, accumulate
        )
        return (current_he, new_res, has_stopped), 0.0

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)  # type: ignore
    _, final_res, _ = final_carry
    return final_res


# Gauss-Legendre rule for the moments of a circular cap (see `_cap_moments`). The integrands are
# analytic on [-alpha, alpha] with alpha <= pi/2, so 16 nodes reach machine precision in float64.
_CAP_QUADRATURE_NODES, _CAP_QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _cap_moments(chord: Array, angle: Array) -> tuple[Array, Array, Array, Array]:
    """Area moments of the circular cap between a chord and its arc, in the chord's own frame.

    The frame has its origin at the chord midpoint, ``u`` along the normal on the side of the arc and
    ``w`` along the chord. ``angle`` is the half-angle the arc subtends at its centre, as stored in
    ``angTable``. The cap is parametrised by the polar angle ``phi`` in ``[-angle, angle]``: a point of
    the arc is at ``w = R sin(phi)``, and the cap's height above the chord there is
    ``R (cos(phi) - cos(angle)) = 2 R sin((angle + phi)/2) sin((angle - phi)/2)``. That product form
    has no cancellation for small angles, unlike the closed forms (sector minus triangle), which lose
    most of their digits in float32 when the arc is nearly flat.

    Returns ``(area, int u dA, int u^2 dA, int w^2 dA)``; ``int w dA`` and ``int u w dA`` vanish by
    symmetry.
    """
    radius = chord / (2 * jnp.sin(angle))
    phi = angle * _CAP_QUADRATURE_NODES
    jac = angle * _CAP_QUADRATURE_WEIGHTS * radius * jnp.cos(phi)
    height = 2 * radius * jnp.sin(0.5 * (angle + phi)) * jnp.sin(0.5 * (angle - phi))
    w = radius * jnp.sin(phi)
    return (
        jnp.sum(jac * height),
        jnp.sum(jac * height**2 / 2),
        jnp.sum(jac * height**3 / 3),
        jnp.sum(jac * w**2 * height),
    )


@jit
def get_cluster_moments_bounded(vertTable: Array, angTable: Array, heTable: Array) -> tuple[Array, Array, Array]:
    """Area, centroid and central second-moment matrix of the whole region covered by a bounded mesh.

    The region is the union of the cells, each with its circular caps on the free interface, exactly
    the region whose cell areas `get_area_bounded` returns. Its moments are computed by Green's
    theorem: every half-edge of a cell contributes the moments of the triangle it spans with the
    origin, and every free interface adds the moments of its circular cap, which always lies outside
    its cell. Faces are traversed counterclockwise (`BoundedMesh` builds them so and T1 transitions
    keep it), so the triangles are signed by the traversal and each cap lies to the right of its
    directed chord. The orientation is not read from the sign of a cell's chord polygon, which turns
    over in a crescent (a cell whose interior vertices have crossed the chord of its arc) while the
    cell and its cap keep their side.

    Args:
        vertTable: vertex positions as stored in `BoundedMesh.vertices`, shape ``(nb_vertices, 2)``
            (without the two reserved rows the energies prepend).
        angTable: one arc half-angle per edge, as stored in `BoundedMesh.angles`.
        heTable: the half-edge table, as stored in `BoundedMesh.edges`.

    Returns:
        ``(area, centroid, second_moments)``, where ``second_moments[i, j]`` is
        ``int (x_i - c_i)(x_j - c_j) dA`` over the region, ``c`` its centroid. For an ellipse of
        semi-axes ``a`` along x and ``b`` along y, ``second_moments = (pi a b / 4) diag(a^2, b^2)``.
    """
    # Moments about a reference point inside the cluster, so the centring subtraction does not cancel
    # digits on a mesh far from the origin. Any reference gives the same central moments.
    ref = jnp.mean(vertTable, axis=0)
    vt = jnp.vstack([jnp.zeros((2, 2), dtype=vertTable.dtype), vertTable - ref])

    source = heTable[:, 3] + heTable[:, 5]
    target = heTable[:, 4] + heTable[:, 6] - 1
    in_a_cell = source != 0  # the outer half-edge of a free interface belongs to no cell
    p0 = vt[source]
    p1 = vt[target]
    x0, y0, x1, y1 = p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]

    # Polygon part: Green's theorem, triangle (origin, p0, p1) per half-edge of a cell.
    s = (x0 * y1 - x1 * y0) * in_a_cell
    area = jnp.sum(s) / 2
    first = jnp.stack([jnp.sum(s * (x0 + x1)), jnp.sum(s * (y0 + y1))]) / 6
    sxx = jnp.sum(s * (x0**2 + x0 * x1 + x1**2)) / 12
    syy = jnp.sum(s * (y0**2 + y0 * y1 + y1**2)) / 12
    sxy = jnp.sum(s * (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1)) / 24

    # Caps of the free interfaces. The surface half-edge is the one of the pair that lies in a cell.
    is_surface = heTable[:, 5] != 0
    angle = jnp.clip(angTable[jnp.arange(heTable.shape[0]) // 2], 1e-6, jnp.pi / 2)
    d = p1 - p0
    chord = jnp.sqrt(jnp.sum(d**2, axis=1) + 1e-30)
    t = d / chord[:, None]
    # Outward normal: to the right of the directed chord, the cell being traversed counterclockwise.
    n = jnp.stack([t[:, 1], -t[:, 0]], axis=1)
    m = 0.5 * (p0 + p1)
    cap_a, cap_u, cap_uu, cap_ww = vmap(_cap_moments)(chord, angle)
    cap_a, cap_u, cap_uu, cap_ww = (q * is_surface for q in (cap_a, cap_u, cap_uu, cap_ww))

    area = area + jnp.sum(cap_a)
    first = first + jnp.sum(cap_a[:, None] * m + cap_u[:, None] * n, axis=0)
    second = jnp.array([[sxx, sxy], [sxy, syy]])
    second = second + jnp.einsum("k,ki,kj->ij", cap_a, m, m)
    second = second + jnp.einsum("k,ki,kj->ij", cap_u, m, n) + jnp.einsum("k,ki,kj->ij", cap_u, n, m)
    second = second + jnp.einsum("k,ki,kj->ij", cap_uu, n, n) + jnp.einsum("k,ki,kj->ij", cap_ww, t, t)

    centroid_rel = first / area
    central = second - area * jnp.outer(centroid_rel, centroid_rel)
    return area, centroid_rel + ref, central
