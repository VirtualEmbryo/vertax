"""
Geometry utilities for BoundedMesh (VertAX).

Contains:
  - get_face_polygon      : discretised polygon of a cell boundary
  - get_cell_segments     : boundary decomposed into segments with tensions
  - polygon_centroid      : centroid via the Green-Stokes formula
  - _inertia_axis         : principal inertia axis (spindle initialisation)
"""

import numpy as np
import jax
import jax.numpy as jnp

from vertax import BoundedMesh
from vertax.meshes.plot import _get_face_vertices_bounded, _get_arc_with_n_points


def get_face_polygon(face_id: int, mesh: BoundedMesh) -> np.ndarray:
    """Return the discretised polygon of cell face_id's boundary.

    Delegates to VertAX (_get_face_vertices_bounded), which handles surface
    arcs: each arc is sampled into intermediate points so the returned contour
    is a closed polyline.

    Returns an (N, 2) array of 2-D coordinates.
    """
    pts = _get_face_vertices_bounded(mesh, face_id)
    # _get_face_vertices_bounded closes the polygon by duplicating the first vertex
    # at the end; drop it so polygon_centroid receives an open contour.
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def get_cell_segments(face_id: int, mesh: BoundedMesh
                      ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Decompose the cell boundary into elementary straight segments.

    Each segment is a rectilinear portion of the boundary, associated with the
    tension of its parent edge. Curved surface arcs are subdivided into
    sub-segments via _get_arc_with_n_points; each sub-segment inherits the
    tension of the original edge.

    Used by the force model: microtubules are traced to the first segment hit,
    and the segment tension Λ is the coefficient in dF = Λ · L · ê · dθ.

    Returns:
        v0s      : (N, 2) — source endpoint of each segment
        v1s      : (N, 2) — target endpoint of each segment
        tensions : (N,)   — tension Λ associated with each segment
    """
    heTable   = np.array(mesh.edges)
    faceTable = np.array(mesh.faces)
    verts     = np.array(mesh.vertices)
    # Physical junction tension Λ = sigmoid(edges_params) + 1 ∈ [1, 2], the SAME
    # transform used everywhere else (energy_line_tensions_bounded, the tension colour scale):
    # edges_params stores the raw, unbounded parameter. Using it directly as Λ was a
    # bug — it let Λ be negative (microtubules pushing) and span huge ratios, which
    # pulled the spindle far off the cell's long axis (breaking Hertwig's rule).
    tensions  = np.asarray(jax.nn.sigmoid(jnp.asarray(mesh.edges_params)) + 1.0)
    angles    = np.array(mesh.angles)

    # Walk the face's half-edge loop in next order
    start_he = int(faceTable[face_id, 0])
    he = start_he
    v0s, v1s, tens = [], [], []
    first = True
    while first or he != start_he:
        first = False
        # VertAX encoding: src = src_i + src_s - 2, tgt = tgt_i + tgt_s - 3
        # (indices 0 and 1 are reserved as sentinels)
        src_id = int(heTable[he, 3] + heTable[he, 5] - 2)
        tgt_id = int(heTable[he, 4] + heTable[he, 6] - 3)
        tension = float(tensions[he // 2])

        if src_id >= 0 and tgt_id >= 0:
            p0  = verts[src_id]
            p1  = verts[tgt_id]
            ang = float(angles[he // 2])
            # Surface edge: src_i == 0 (the inner column holds sentinel 0)
            is_surface = (int(heTable[he, 3]) == 0)

            if is_surface and ang > 0.01:
                # Circular arc: discretise into sub-segments so that rays can
                # correctly intersect the real curvature of the membrane
                arc_pts = [p0] + _get_arc_with_n_points(ang, p0, p1) + [p1]
                for a, b in zip(arc_pts[:-1], arc_pts[1:]):
                    v0s.append(a); v1s.append(b); tens.append(tension)
            else:
                # Inner edge or negligible arc: straight segment
                v0s.append(p0); v1s.append(p1); tens.append(tension)

        he = int(heTable[he, 1])  # advance to next half-edge

    return (jnp.array(v0s, dtype=jnp.float64),
            jnp.array(v1s, dtype=jnp.float64),
            jnp.array(tens, dtype=jnp.float64))


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Compute the centroid of a polygon via the Green-Stokes formula.

    For a polygon with vertices (xᵢ, yᵢ), the signed area and centroid are:

        A  = ½ Σᵢ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
        cx = 1/(6A) · Σᵢ (xᵢ + xᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
        cy = 1/(6A) · Σᵢ (yᵢ + yᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)

    Indices are cyclic (last vertex connects back to first).
    This formula exactly integrates ∫∫ x dA / A via Green's theorem.
    """
    x, y   = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)  # next vertices (cyclic)
    cross  = x * yn - xn * y                  # cross term xᵢyᵢ₊₁ − xᵢ₊₁yᵢ
    A  = 0.5 * np.sum(cross)
    cx = np.sum((x + xn) * cross) / (6.0 * A)
    cy = np.sum((y + yn) * cross) / (6.0 * A)
    return np.array([cx, cy])


def _inertia_axis(pts: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute the principal inertia axis of the filled polygon.

    The surface inertia tensor (relative to the centroid) is:

        I = [[Ixx, -Ixy],
             [-Ixy, Iyy]]

    where Ixx = ∫∫ y² dA, Iyy = ∫∫ x² dA, Ixy = ∫∫ xy dA.

    For a polygon, these integrals have closed-form expressions via Green:

        Ixx = 1/12 · Σᵢ (yᵢ² + yᵢyᵢ₊₁ + yᵢ₊₁²)(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)
        Iyy = 1/12 · Σᵢ (xᵢ² + xᵢxᵢ₊₁ + xᵢ₊₁²)(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)
        Ixy = 1/24 · Σᵢ (xᵢyᵢ₊₁ + 2xᵢyᵢ + 2xᵢ₊₁yᵢ₊₁ + xᵢ₊₁yᵢ)(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)

    The principal axis is the eigenvector for the smallest eigenvalue of I
    (direction of maximum elongation). Used as Newton-Raphson initialisation.
    """
    p = pts - centroid           # centred coordinates
    x, y   = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross  = x * yn - xn * y    # cross term (same as in polygon_centroid)
    Ixx = np.sum((y**2 + y*yn + yn**2) * cross) / 12.0
    Iyy = np.sum((x**2 + x*xn + xn**2) * cross) / 12.0
    Ixy = np.sum((x*yn + 2*x*y + 2*xn*yn + xn*y) * cross) / 24.0
    # np.linalg.eigh returns eigenvalues in ascending order;
    # vecs[:, 0] is the eigenvector for the smallest eigenvalue → max elongation axis.
    _, vecs = np.linalg.eigh(np.array([[Iyy, -Ixy], [-Ixy, Ixx]]))
    return vecs[:, 0]
