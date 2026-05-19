"""
Half-edge table surgery to divide a cell.

The division line is perpendicular to the spindle axis u and passes through
the centre c. It cuts exactly 2 edges of the cell boundary.
The half-edge topology is updated manually:
  - cut edges are truncated and completed with new half-edges,
  - tensions redistributed proportionally to lengths,
  - two new twin half-edges (hAB, hBA) form the division wall.

Contains:
  - _seg_line_intersect    : line / segment intersection
  - _arc_line_intersect    : line / circular arc intersection
  - _walk_face_half_edges  : ordered list of half-edges around a face
  - _get_he_endpoints      : (src, tgt) coordinates of a half-edge
  - divide_cell            : public API — divides the cell and returns the new mesh
"""

import numpy as np
import jax.numpy as jnp

from vertax import BoundedMesh
from vertax.meshes.plot import _get_arc_with_n_points


def _seg_line_intersect(p0: np.ndarray, p1: np.ndarray,
                        origin: np.ndarray, normal: np.ndarray
                        ) -> tuple[bool, float, np.ndarray]:
    """Test whether segment [p0, p1] intersects the division line.

    The line is defined implicitly by: (x − origin) · normal = 0.
    Signed distances of endpoints to the line:
        d0 = (p0 − origin) · normal
        d1 = (p1 − origin) · normal

    If d0 and d1 have the same sign, the segment is entirely on one side: no
    intersection. Otherwise, the intersection parameter t ∈ [0, 1] is:
        t = d0 / (d0 − d1)

    and the intersection point is p0 + t · (p1 − p0).

    Returns (intersects, t, point).
    """
    d0 = float(np.dot(p0 - origin, normal))
    d1 = float(np.dot(p1 - origin, normal))
    if d0 * d1 > 0:           # same sign → same side → no intersection
        return False, 0.0, p0
    if abs(d1 - d0) < 1e-12:  # segment nearly parallel to the line
        return False, 0.0, p0
    t = d0 / (d0 - d1)
    return True, t, p0 + t * (p1 - p0)


def _arc_line_intersect(p0: np.ndarray, p1: np.ndarray, angle: float,
                        origin: np.ndarray, normal: np.ndarray,
                        n_arc: int = 200
                        ) -> tuple[bool, float, np.ndarray]:
    """Intersection of the division line with a circular arc.

    The arc connects p0 to p1 with a given inscribed angle (VertAX parametrisation).
    It is discretised into n_arc sub-segments and _seg_line_intersect is applied
    to each until the first intersection is found.

    t_arc ∈ [0, 1] is the fractional position of the intersection along the arc
    (0 = near p0, 1 = near p1):
        t_arc = (i + t_seg) / n_arc
    where i is the sub-segment index and t_seg ∈ [0,1] the local fraction.

    Intersections too close to the endpoints (t_arc < 1e-4 or > 1-1e-4) are
    excluded: they correspond to a line passing through an existing vertex,
    handled separately by the 'vertex_tgt'/'vertex_src' cases.

    Returns (intersects, t_arc, point_on_arc).
    """
    arc_pts = np.array([p0] + _get_arc_with_n_points(angle, p0, p1, nb_draw_points=n_arc) + [p1])
    for i in range(len(arc_pts) - 1):
        hit, t_seg, pt = _seg_line_intersect(arc_pts[i], arc_pts[i + 1], origin, normal)
        if hit and -1e-9 <= t_seg <= 1 + 1e-9:
            t_arc = (i + t_seg) / (len(arc_pts) - 1)
            if 1e-4 < t_arc < 1 - 1e-4:
                return True, t_arc, pt
    return False, 0.0, p0


def _walk_face_half_edges(face_id: int, heTable: np.ndarray,
                          faceTable: np.ndarray) -> list[int]:
    """Return the ordered list of half-edges around the face, following next pointers."""
    start = int(faceTable[face_id, 0])
    he, hes = start, []
    first = True
    while first or he != start:
        first = False
        hes.append(he)
        he = int(heTable[he, 1])
    return hes


def _get_he_endpoints(he: int, heTable: np.ndarray,
                      verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (src, tgt) coordinates of a half-edge.

    VertAX two-column encoding:
      - Inner edge  : src_i ≥ 2, src_s = 0  → src_idx = src_i - 2
      - Surface edge: src_i = 0, src_s ≥ 2  → src_idx = src_s - 2
    Unified identity: src_idx = src_i + src_s - 2.
    Similarly for tgt: tgt_idx = tgt_i + tgt_s - 3.
    """
    src_inner, src_surf = int(heTable[he, 3]), int(heTable[he, 5])
    tgt_inner, tgt_surf = int(heTable[he, 4]), int(heTable[he, 6])
    src_idx = src_inner + src_surf - 2
    tgt_idx = tgt_inner + tgt_surf - 3
    return verts[src_idx], verts[tgt_idx]


def divide_cell(mesh: BoundedMesh, face_id: int,
                c: np.ndarray, u: np.ndarray) -> tuple[BoundedMesh, bool]:
    """Divide cell face_id into two along the line perpendicular to u through c.

    Half-edge topology surgery algorithm:

    1. CUT DETECTION
       Walk the face's half-edges and find the 2 edges (or vertices) crossed
       by the division line. Four cases:
         - 'edge'       : line cuts the interior of an edge → new vertex
         - 'arc'        : line cuts a surface arc → new vertex
         - 'vertex_tgt' : line passes exactly through a vertex → reuse it
         - 'vertex_src' : same (detected from the other side of the edge)

    2. NEW VERTEX CREATION
       For each 'edge' or 'arc' cut, insert a new vertex at the intersection.
       Its VertAX index is nV_orig + n + 2 (offset +2 for sentinels).

    3. EDGE TRUNCATION
       For each cut edge ('split' case):
         - heX is truncated: its tgt changes from the old vertex to new vX.
         - If the edge is inner, its twin heX.twin is also truncated:
           its tgt changes to vX (from the neighbouring face).
         - A new half-edge heX2 completes the remaining portion of heX
           (from vX to the old tgt of heX).
         - A new half-edge heX2t is the twin of heX2 (in the neighbouring face
           or boundary ghost if it is a surface edge).
       Tensions are redistributed proportionally to relative length:
         Λ(heX)  ← Λ_orig · t      (kept portion)
         Λ(heX2) ← Λ_orig · (1-t)  (created portion)

    4. WALL CREATION
       Two new twin half-edges:
         hAB : vA → vB, belongs to face_id  (mother cell)
         hBA : vB → vA, belongs to new_face (daughter cell)
       Their tension is initialised to the mean tissue tension.

    5. PREV/NEXT/FACE REWIRING
       Existing half-edges of the face are distributed between the two new
       faces (face_id and new_face) in two loops:
         loop_1 : half-edges between heA and heB → new_face
         loop_2 : half-edges between heB and heA → face_id
       prev/next pointers and face_id fields are updated so each loop is
       consistent (each next points to the next half-edge in the same face).

    Returns the new BoundedMesh and a boolean indicating whether an existing
    vertex was reused (in which case T1 transitions may be needed).
    """
    # Extract tables as mutable NumPy arrays (copy — original mesh is not modified)
    heT  = np.array(mesh.edges,           dtype=np.int32)
    faceT= np.array(mesh.faces,           dtype=np.int32)
    vertT= np.array(mesh.vertices,        dtype=np.float32)
    angT = np.array(mesh.angles,          dtype=np.float32)
    epar = np.array(mesh.edges_params,    dtype=np.float32)
    vpar = np.array(mesh.vertices_params, dtype=np.float32)
    fpar = np.array(mesh.faces_params,    dtype=np.float32)

    if np.any(np.isnan(c)) or np.any(np.isnan(u)):
        raise ValueError("Spindle contains NaN — find_spindle diverged for this cell.")

    # Division line equation: (x − c) · u = 0, so u is the line normal.
    div_normal = u

    def _cut_point(cut):
        kind = cut[0]
        if kind in ('edge', 'arc'):
            return np.array(cut[3])
        he_ = cut[1]
        src_i, src_s = int(heT[he_, 3]), int(heT[he_, 5])
        tgt_i, tgt_s = int(heT[he_, 4]), int(heT[he_, 6])
        if kind == 'vertex_src':
            return vertT[src_i + src_s - 2]
        return vertT[tgt_i + tgt_s - 3]

    def _detect_cuts(origin, normal):
        """Detect intersections of the line (origin, normal) with the face boundary.

        For non-convex cells where the infinite line crosses > 2 edges, keep only
        the 2 intersections closest to origin along the line direction (perpendicular
        to normal), one from each side. These are the actual entry/exit points of
        the division line.
        """
        VERTEX_TOL = 0.03
        # Direction along the division line (perpendicular to normal)
        along = np.array([-normal[1], normal[0]])

        hes_ = _walk_face_half_edges(face_id, heT, faceT)
        cuts_ = []
        hit_vertices_ = set()
        for he in hes_:
            p0, p1 = _get_he_endpoints(he, heT, vertT)
            is_surf_he = (int(heT[he, 3]) == 0 and int(heT[he, 5]) >= 2)
            ang_he = float(angT[he // 2])
            if is_surf_he and ang_he > 0.01:
                hit, t_arc, pt_arc = _arc_line_intersect(p0, p1, ang_he, origin, normal)
                if hit:
                    cuts_.append(('arc', he, t_arc, pt_arc))
            else:
                hit, t, pt = _seg_line_intersect(p0, p1, origin, normal)
                if not hit:
                    continue
                if t > 1 - VERTEX_TOL:
                    tgt_i, tgt_s = int(heT[he, 4]), int(heT[he, 6])
                    vtx = tgt_i + tgt_s - 3
                    if vtx not in hit_vertices_:
                        hit_vertices_.add(vtx)
                        cuts_.append(('vertex_tgt', he, vtx))
                elif t < VERTEX_TOL:
                    src_i, src_s = int(heT[he, 3]), int(heT[he, 5])
                    vtx = src_i + src_s - 2
                    if vtx not in hit_vertices_:
                        hit_vertices_.add(vtx)
                        cuts_.append(('vertex_src', he, vtx))
                else:
                    cuts_.append(('edge', he, t, pt))

        # If the line passes exactly through a vertex (1 cut), duplicate it:
        # the vertex acts as both entry and exit — one cut per adjacent half-edge.
        if len(cuts_) == 1 and cuts_[0][0].startswith('vertex'):
            kind, he_hit, vtx = cuts_[0]
            # Find the other half-edge that shares this vertex on the other side
            hes_list = _walk_face_half_edges(face_id, heT, faceT)
            for he2 in hes_list:
                if he2 == he_hit:
                    continue
                src_i2, src_s2 = int(heT[he2, 3]), int(heT[he2, 5])
                tgt_i2, tgt_s2 = int(heT[he2, 4]), int(heT[he2, 6])
                vtx_src2 = src_i2 + src_s2 - 2
                vtx_tgt2 = tgt_i2 + tgt_s2 - 3
                if vtx_src2 == vtx:
                    cuts_.append(('vertex_src', he2, vtx))
                    break
                if vtx_tgt2 == vtx:
                    cuts_.append(('vertex_tgt', he2, vtx))
                    break

        # If > 2 cuts (non-convex cell): keep one cut from each side of origin,
        # taking the closest on each side along the division line direction.
        if len(cuts_) > 2:
            def proj_along(cut):
                pt = _cut_point(cut)
                return float(np.dot(pt - origin, along))
            pos = [(proj_along(ct), ct) for ct in cuts_]
            neg_side = [(s, ct) for s, ct in pos if s <= 0]
            pos_side = [(s, ct) for s, ct in pos if s > 0]
            best = []
            if neg_side:
                best.append(max(neg_side, key=lambda x: x[0])[1])
            if pos_side:
                best.append(min(pos_side, key=lambda x: x[0])[1])
            cuts_ = best

        return cuts_

    # ── 1. Cut detection ─────────────────────────────────────────────────
    cuts = _detect_cuts(c, div_normal)

    if len(cuts) != 2:
        raise ValueError(
            f"Division line intersects {len(cuts)} edge(s)/vertex(ices) "
            f"(expected 2). Try another cell."
        )

    # Local helpers to read/write src/tgt columns of the half-edge table

    def _tgt_cols(he):
        """Return (tgt_i, tgt_s) of half-edge he before any modification."""
        return int(heT[he, 4]), int(heT[he, 6])

    def _src_cols(he):
        """Return (src_i, src_s) of half-edge he before any modification."""
        return int(heT[he, 3]), int(heT[he, 5])

    def _set_tgt(he, v):
        """Write target vertex v into half-edge he.

        v is the VertAX index (offset +2 included), so real_vertex = v - 2.
        Encoding depends on edge type:
          - Inner   (heT[he,3] != 0): tgt_i = v, tgt_s = 1  → tgt = v+1-3 = v-2 ✓
          - Surface (heT[he,3] == 0): tgt_i = 1, tgt_s = v  → tgt = 1+v-3 = v-2 ✓
        """
        if int(heT[he, 3]) != 0:
            heT[he, 4] = v; heT[he, 6] = 1
        else:
            heT[he, 4] = 1; heT[he, 6] = v

    def _set_src(he, v):
        """Write source vertex v into half-edge he.

        v is the VertAX index (offset +2 included), so real_vertex = v - 2.
        Encoding depends on edge type:
          - Inner   (heT[he,3] != 0): src_i = v, src_s = 0  → src = v+0-2 = v-2 ✓
          - Surface (heT[he,3] == 0): src_i = 0, src_s = v  → src = 0+v-2 = v-2 ✓
        """
        if int(heT[he, 3]) != 0:
            heT[he, 3] = v; heT[he, 5] = 0
        else:
            heT[he, 3] = 0; heT[he, 5] = v

    # ── 2. New vertex creation and cut info collection ────────────────────
    nV_orig = len(vertT)
    new_verts = []
    cut_info = []

    for cut in cuts:
        if cut[0] == 'vertex_tgt':
            _, heX, vtx_idx = cut
            vX = vtx_idx + 2  # indice VertAX (décalage +2)
            cut_info.append({'he': heX, 'v': vX, 'split': False,
                             'he2': None, 'he2t': None})
        elif cut[0] == 'vertex_src':
            _, heX, vtx_idx = cut
            vX = vtx_idx + 2
            # vertex_src is equivalent to vertex_tgt on the previous half-edge:
            # use he_prev so the "entry half-edge" into the vertex is consistent.
            he_prev = int(heT[heX, 0])
            cut_info.append({'he': he_prev, 'v': vX, 'split': False,
                             'he2': None, 'he2t': None})
        elif cut[0] == 'arc':
            _, heX, t_arc, ptX = cut
            vX = nV_orig + len(new_verts) + 2
            new_verts.append(ptX.astype(np.float32))
            twin = int(heT[heX, 2])
            ang_orig = float(angT[heX // 2])
            # Arc cut at fraction t_arc from p0:
            #   ang_A = angle of portion p0 → vX  (inherited by truncated heX)
            #   ang_B = angle of portion vX → p1  (inherited by new heX2)
            # Angles are proportional to the arc fraction because the inscribed
            # angle is directly tied to arc length on a fixed circle.
            ang_A = ang_orig * t_arc
            ang_B = ang_orig * (1.0 - t_arc)
            # Tensions redistributed proportionally to arc length:
            ep_orig = float(epar[heX // 2])
            cut_info.append({'he': heX, 'v': vX, 'split': True,
                             'he2': None, 'he2t': None, 'pt': ptX,
                             '_ang':      ang_A,
                             '_ang_he2':  ang_B,
                             '_ep':       ep_orig * t_arc,
                             '_ep_he2':   ep_orig * (1.0 - t_arc),
                             '_tgt_i':    _tgt_cols(heX)[0],
                             '_tgt_s':    _tgt_cols(heX)[1],
                             '_twin':     twin,
                             '_is_surf':  True,
                             # Pour un arc de surface, le twin est un fantôme de bord :
                             # il n'y a pas de face voisine, donc pas de twin_tgt à sauvegarder.
                             '_twin_src_i': 0,   # surface arc: no real neighbouring face
                             '_twin_src_s': 0,
                             '_twin_next':  0,
                             '_twin_face':  0})
        else:  # 'edge' — coupe intérieure d'une arête droite
            _, heX, t, ptX = cut
            vX = nV_orig + len(new_verts) + 2
            new_verts.append(ptX.astype(np.float32))
            twin = int(heT[heX, 2])
            is_surf = (int(heT[heX, 3]) == 0)
            # Tensions redistribuées au prorata de la longueur relative t :
            #   Λ(heX)  ← Λ_orig · t       (portion [src, vX])
            #   Λ(heX2) ← Λ_orig · (1 − t) (portion [vX, tgt_orig])
            ep_orig = float(epar[heX // 2])
            cut_info.append({'he': heX, 'v': vX, 'split': True,
                             'he2': None, 'he2t': None, 'pt': ptX,
                             '_ang':      float(angT[heX // 2]),
                             '_ang_he2':  float(angT[heX // 2]),
                             '_ep':       ep_orig * t,
                             '_ep_he2':   ep_orig * (1.0 - t),
                             '_tgt_i':    _tgt_cols(heX)[0],
                             '_tgt_s':    _tgt_cols(heX)[1],
                             '_twin':     twin,
                             '_is_surf':  is_surf,
                             # Save twin's tgt BEFORE mutation: after _set_tgt(twin, vX)
                             # this info is lost. New half-edge heX2t must go from vX
                             # to that original tgt.
                             '_twin_tgt_i': _tgt_cols(twin)[0],
                             '_twin_tgt_s': _tgt_cols(twin)[1],
                             '_twin_next':  int(heT[twin, 1]),
                             '_twin_face':  int(heT[twin, 7])})

    # Append new vertices to the table
    if new_verts:
        vertT = np.vstack([vertT] + [v.reshape(1, 2) for v in new_verts])
        vpar  = np.append(vpar, [0.0] * len(new_verts)).astype(np.float32)

    nHE = len(heT)
    new_face_id = len(faceT)

    # ── 3. Allocate indices for new half-edges ───────────────────────────
    he_idx = nHE
    for ci in cut_info:
        if ci['split']:
            ci['he2']  = he_idx;     he_idx += 1
            ci['he2t'] = he_idx;     he_idx += 1

    ciA, ciB = cut_info[0], cut_info[1]
    hAB = he_idx;     he_idx += 1  # vA → vB (appartient à face_id)
    hBA = he_idx                   # vB → vA (appartient à new_face_id)

    vA = ciA['v']
    vB = ciB['v']
    heA = ciA['he']
    heB = ciB['he']

    def collect_until(start, stop):
        """Collect half-edges from start up to (but not including) stop, following next."""
        loop, he = [], start
        while he != stop:
            loop.append(he)
            he = int(heT[he, 1])
        return loop

    # If heA is split, the portion [vA → old_tgt_heA] becomes ciA's he2,
    # and loop_1 must include it (starting from heA_next, wired after heA2
    # in new_face). Otherwise loop_1 starts at next(heA).
    # Same symmetric logic for loop_2 / heB.
    heA_next = int(heT[heA, 1])
    heB_next = int(heT[heB, 1])

    # loop_1: half-edges between heA and heB → assigned to new_face
    # If ciA is a split, ciA's he2 is inserted between heA and loop_1:
    # loop_1 runs from heA_next up to heB (excluded).
    loop_1 = collect_until(heA_next, heB)

    # loop_2: half-edges between heB and heA → assigned to face_id
    # If ciB is a split, ciB's he2 is inserted between heB and loop_2:
    # loop_2 runs from heB_next up to heA (excluded).
    loop_2 = collect_until(heB_next, heA)

    # ── 4. Truncate cut edges ────────────────────────────────────────────
    for ci in cut_info:
        if ci['split']:
            _set_tgt(ci['he'], ci['v'])
            if not ci['_is_surf']:
                _set_tgt(ci['_twin'], ci['v'])

    # ── 5. Build new half-edges ──────────────────────────────────────────
    rows = []
    ang_new = []
    ep_new  = []

    first_1 = loop_1[0] if loop_1 else heB
    last_1  = loop_1[-1] if loop_1 else (ciA['he2'] if ciA['split'] else heA)
    first_2 = loop_2[0] if loop_2 else heA
    last_2  = loop_2[-1] if loop_2 else (ciB['he2'] if ciB['split'] else heB)

    for ci in cut_info:
        if not ci['split']:
            continue
        heX2  = ci['he2']
        heX2t = ci['he2t']
        heX   = ci['he']
        twin  = ci['_twin']
        vX    = ci['v']
        ti, ts = ci['_tgt_i'], ci['_tgt_s']
        is_surf = ci['_is_surf']

        if is_surf:
            rows.append([0, 0, heX2t, 0, ti, vX, ts, 0])
            rows.append([0, 0, heX2, 0, 1, 0, 1, 0])
            ang_new.append(ci['_ang_he2'])
            ep_new.append(ci['_ep_he2'])
            heT[twin, 2] = heX2t
            angT[heX // 2] = ci['_ang']
            epar[heX // 2] = ci['_ep']
        else:
            rows.append([0, 0, heX2t, vX, ti, 0, ts, 0])
            twin_tgt_i = ci['_twin_tgt_i']
            twin_tgt_s = ci['_twin_tgt_s']
            twin_next  = ci['_twin_next']
            twin_face  = ci['_twin_face']
            rows.append([twin, twin_next, heX2, vX, twin_tgt_i, 0, twin_tgt_s, twin_face])
            ang_new.append(ci['_ang_he2'])
            ep_new.append(ci['_ep_he2'])
            epar[heX // 2] = ci['_ep']
            heT[twin, 1]      = heX2t
            heT[twin_next, 0] = heX2t

    rows.append([heA, 0, hBA, vA, vB, 0, 1, face_id])
    rows.append([heB, 0, hAB, vB, vA, 0, 1, new_face_id])
    ang_new.append(1.0)
    ep_new.append(float(np.mean(epar)))

    R = np.array(rows, dtype=np.int32)
    heT = np.vstack([heT, R])

    # ── 6. Rewire prev/next of new half-edges ────────────────────────────
    #
    # After division, face_id contains the loop:
    #   heA → hAB → [heB2 →] loop_2 → heA   (face_id loop)
    # and new_face contains:
    #   heB → hBA → [heA2 →] loop_1 → heB   (new_face loop)
    #
    # heA2 (resp. heB2) only exists if ciA (resp. ciB) is a split.
    # If so, it is inserted between hBA→loop_1 (resp. hAB→loop_2).

    heA2_idx = ciA['he2'] if ciA['split'] else None
    heB2_idx = ciB['he2'] if ciB['split'] else None

    # First effective element of each loop in each face:
    #   face_id  : hAB → heB2 (if split) → loop_2[0] (if any) → heA
    #   new_face : hBA → heA2 (if split) → loop_1[0] (if any) → heB

    # What immediately follows hAB in face_id:
    after_hAB = heB2_idx if heB2_idx is not None else (loop_2[0] if loop_2 else heA)
    # What immediately follows hBA in new_face:
    after_hBA = heA2_idx if heA2_idx is not None else (loop_1[0] if loop_1 else heB)
    # What precedes heA in face_id:
    before_heA = loop_2[-1] if loop_2 else (heB2_idx if heB2_idx is not None else hAB)
    # What precedes heB in new_face:
    before_heB = loop_1[-1] if loop_1 else (heA2_idx if heA2_idx is not None else hBA)

    # Wire hAB (face_id): prev=heA, next=after_hAB
    heT[hAB, 0] = heA
    heT[hAB, 1] = after_hAB

    # Wire hBA (new_face): prev=heB, next=after_hBA
    heT[hBA, 0] = heB
    heT[hBA, 1] = after_hBA

    # heA closes the face_id loop: next=hAB, prev=before_heA
    heT[heA, 1] = hAB
    heT[heA, 0] = before_heA

    # heB closes the new_face loop: next=hBA, prev=before_heB
    heT[heB, 1] = hBA
    heT[heB, 0] = before_heB

    # heA2 (if present) in new_face: prev=hBA, next=loop_1[0] or heB
    if heA2_idx is not None:
        heT[heA2_idx, 0] = hBA
        heT[heA2_idx, 1] = loop_1[0] if loop_1 else heB
        heT[heA2_idx, 7] = new_face_id

    # heB2 (if present) in face_id: prev=hAB, next=loop_2[0] or heA
    if heB2_idx is not None:
        heT[heB2_idx, 0] = hAB
        heT[heB2_idx, 1] = loop_2[0] if loop_2 else heA
        heT[heB2_idx, 7] = face_id

    # Junction at the endpoints of loop_1 (new_face)
    if loop_1:
        prev_of_first_1 = heA2_idx if heA2_idx is not None else hBA
        heT[loop_1[0],  0] = prev_of_first_1
        heT[loop_1[-1], 1] = heB

    # Junction at the endpoints of loop_2 (face_id)
    if loop_2:
        prev_of_first_2 = heB2_idx if heB2_idx is not None else hAB
        heT[loop_2[0],  0] = prev_of_first_2
        heT[loop_2[-1], 1] = heA

    # ── 7. Update face_id fields ─────────────────────────────────────────
    face_id_hes     = ([heA, hAB]
                       + ([heB2_idx] if heB2_idx is not None else [])
                       + loop_2)
    new_face_id_hes = ([heB, hBA]
                       + ([heA2_idx] if heA2_idx is not None else [])
                       + loop_1)
    for he in face_id_hes:
        heT[he, 7] = face_id
    for he in new_face_id_hes:
        heT[he, 7] = new_face_id

    # ── 8. Update angle, tension, and face tables ────────────────────────
    angT = np.append(angT, ang_new).astype(np.float32)
    epar = np.append(epar, ep_new).astype(np.float32)

    faceT[face_id, 0] = heA
    fate = int(faceT[face_id, 1]) if faceT.shape[1] > 1 else 0
    faceT = np.vstack([faceT, np.array([[heB, fate]], dtype=np.int32)])
    fpar  = np.append(fpar, [float(fpar[face_id])]).astype(np.float32)

    # ── 9. Assemble the new BoundedMesh ─────────────────────────────────
    new_mesh = BoundedMesh._create()
    new_mesh.vertices        = jnp.array(vertT)
    new_mesh.edges           = jnp.array(heT)
    new_mesh.faces           = jnp.array(faceT)
    new_mesh.angles          = jnp.array(angT)
    new_mesh.width           = mesh.width
    new_mesh.height          = mesh.height
    new_mesh.vertices_params = jnp.array(vpar)
    new_mesh.edges_params    = jnp.array(epar)
    new_mesh.faces_params    = jnp.array(fpar)

    n_split = sum(1 for ci in cut_info if ci['split'])
    print(f"Cell {face_id} divided → faces {face_id} and {new_face_id} "
          f"({n_split} new vertex(ices))")
    return new_mesh, n_split < 2
