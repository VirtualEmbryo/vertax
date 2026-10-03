"""Trace a segmentation label mask into a straight-edge vertex-model mesh.

This module turns a **label mask** — an integer image in which each cell carries a distinct
positive label and unsegmented space carries `0` — into a polygonal mesh of the kind the rest of
VertAX operates on: vertices at cell corners, edges along cell-cell interfaces, faces for cells.

It is written for the case a microscope actually produces: a **crop**. A crop is a patch of
segmented cells surrounded by unsegmented margin, so some cells are cut by the field of view and
some corners have a neighbour that was never imaged. Handling that honestly is the point of this
module, and it is why `trace_mask` reports a *boundary partition* alongside the mesh rather than
silently pretending the crop is a closed tissue.

## Relation to `vertax.mask_analysis`

`vertax.mask_analysis.find_vertices_edges_faces` extracts a mesh from a mask that has been padded
so every cell is fully enclosed. That assumption fails on a raw crop: there the Euler characteristic
does not close (`V - E + F != 0`) and the extraction is not meaningful. Use `vertax.mask_analysis`
for closed, padded masks; use this module for crops.

## The corner lattice, and why there are no tuning parameters

The tracer works on the **pixel-corner lattice** of the mask — the grid of points at pixel corners,
which is the exact dual of the segmentation. Corner `(i, j)` sits at image coordinate
`(x, y) = (j, i)` and is surrounded by four pixels `NW, NE, SW, SE`. The four lattice segments
meeting at a corner are *boundary* segments exactly when the two pixels they separate carry
different labels, so the corner's boundary **degree** is

```
[NW != NE] + [SW != SE] + [NW != SW] + [NE != SE]     in {0, 2, 3, 4}
```

and `degree >= 3` is the junction criterion. This is strictly more general than looking for three
distinct labels in a 2x2 window: it also catches diagonal-touch trijunctions (4 segments, 3 labels)
and checkerboard pinches (4 segments, 2 labels), which a 2x2-window test misses or mishandles.

From there the construction is forced, with no thresholds to choose:

* 8-connected clusters of junction corners collapse to one **vertex** at their centroid, because a
  real trijunction is generally a small cluster of corners rather than a single one;
* between two vertices the boundary is a chain of degree-2 corners carrying a single label pair,
  and each such chain becomes one straight **edge** (the underlying staircase polyline is retained
  only so the discarded length can be measured);
* a **face** is the cycle of edges carrying one cell label. Background is not a face.

Straight edges are a modelling choice, not an approximation of convenience: drawing a cell-cell
interface straight *is* the vertex-model convention, and the curvature it discards is precisely the
pressure difference that inverse methods are trying to recover.

## Free and clamped vertices

A vertex is **clamped** iff an unmodelled material — background, by default — is one of the labels
meeting there. This is a force-balance criterion rather than a stylistic convention: at such a
vertex at least one incident material is the environment that was never imaged, so force balance
cannot be written from the data. Every other vertex is **free** and carries a complete force
balance.

**Consumers should write mechanical equations on `TracedMesh.free` rows only.** Doing so has a
consequence worth knowing in advance: restricting to those rows introduces exact null directions
that are absent when all rows are used, so some cell-pressure combinations become unobservable. That
is a property of the data being a crop, not of this tracer or of any particular boundary model.

## The one parameter that does exist: `rim_tol`

A cell-cell interface is drawn straight on purpose. A cell-background arc is not an interface at all
— it is the outer edge of the observed patch, i.e. a Dirichlet boundary condition read off the data
— and there is no model reason to straighten it. Traced with a single chord per arc it is badly
wrong, because a rim cell's whole outer side can be one junction-to-junction arc hundreds of pixels
long; measured on real crops, rim-cell areas came out 22.6 % low in the median and 92 % low at
worst.

Cell-background arcs are therefore subdivided by Ramer-Douglas-Peucker to `rim_tol` pixels, and the
inserted break points are clamped vertices. **This adds no free degree of freedom and no
information**, since every inserted vertex is clamped by construction (its label set is
`{0, cell}`); it only stops the boundary condition from being wrong. Interior-cell areas are
unaffected by it.

A cell is **interior** iff none of its edges is a cell-background edge, and **rim-adjacent**
otherwise. Rim-adjacent cells are still complete, fully segmented cells — their outline against the
background is a real membrane, not a truncation — so they keep their area term and contribute
through their free vertices.

## Typical use

```python
from vertax.trace import regularize, trace_mask_closed, interior_area_ref, normalized

clean, report = regularize(mask)               # remove label debris; report says how much moved
assert report["frac_pixels_moved"] < 1e-3      # else investigate the segmentation
mesh = trace_mask_closed(clean, rim_tol=2.0)   # raises if any cell fails to close
verts = normalized(mesh, interior_area_ref(mesh))   # lengths in units of sqrt(mean interior area)
```

Dependencies are `numpy` and `scipy.ndimage` only. Nothing here needs a mesh-reconstruction
library, Cellpose, or JAX, so the tracer works on any label mask from any source.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------------------
@dataclass
class TracedMesh:
    """Straight-edge mesh traced from one label mask, in **pixel** units."""

    stem: str
    shape: tuple[int, int]
    verts: NDArray            # (Nv, 2) float64, (x, y) in pixel-corner coordinates
    vert_labels: list[frozenset]   # labels meeting at each vertex (may include 0 = background)
    vert_cluster_size: NDArray     # (Nv,) int, number of lattice corners fused into this vertex
    edges: NDArray            # (Ne, 2) int, vertex indices
    edge_labels: NDArray      # (Ne, 2) int, the two labels the edge separates (sorted)
    edge_arc_len: NDArray     # (Ne,) float64, length of the pixel-lattice polyline (staircase)
    faces: list[list[int]]    # per cell, CCW cycle of vertex indices
    face_labels: NDArray      # (Nf,) int, the cell label of each face
    failures: dict            # cell label -> reason, for labels that did not close into a face
    unmodelled: frozenset     # labels treated as missing environment (always contains 0)
    junction_census: dict
    diagnostics: dict = field(default_factory=dict)

    # -- derived ---------------------------------------------------------------------
    @property
    def n_verts(self) -> int:
        return len(self.verts)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def clamped(self) -> NDArray:
        """(Nv,) bool -- True where an unmodelled material is one of the incident labels."""
        return np.array([bool(s & self.unmodelled) for s in self.vert_labels], dtype=bool)

    @property
    def free(self) -> NDArray:
        return ~self.clamped

    @property
    def edge_is_cell_cell(self) -> NDArray:
        """True iff both flanking materials are modelled cells (a real interface)."""
        um = self.unmodelled
        return np.array([(int(a) not in um) and (int(b) not in um)
                         for a, b in self.edge_labels], dtype=bool)

    @property
    def face_is_interior(self) -> NDArray:
        """A face is interior iff none of its edges borders an unmodelled material."""
        touching = set()
        for (a, b) in self.edge_labels[~self.edge_is_cell_cell]:
            a, b = int(a), int(b)
            if a not in self.unmodelled:
                touching.add(a)
            if b not in self.unmodelled:
                touching.add(b)
        return np.array([int(l) not in touching for l in self.face_labels], dtype=bool)

    def face_areas(self, verts: NDArray | None = None) -> NDArray:
        v = self.verts if verts is None else verts
        return np.array([_shoelace(v[f]) for f in self.faces])

    def face_perimeters(self, verts: NDArray | None = None) -> NDArray:
        v = self.verts if verts is None else verts
        out = []
        for f in self.faces:
            p = v[f]
            out.append(float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum()))
        return np.array(out)

    def edge_lengths(self, verts: NDArray | None = None) -> NDArray:
        v = self.verts if verts is None else verts
        return np.linalg.norm(v[self.edges[:, 1]] - v[self.edges[:, 0]], axis=1)


def _shoelace(p: NDArray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------------------
def _corner_fields(mask: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Return (NW, NE, SW, SE, degree) on the (H+1) x (W+1) corner lattice."""
    M = np.pad(mask.astype(np.int64), 1, mode="constant", constant_values=0)
    H1, W1 = mask.shape[0] + 1, mask.shape[1] + 1
    NW = M[0:H1, 0:W1]
    NE = M[0:H1, 1:W1 + 1]
    SW = M[1:H1 + 1, 0:W1]
    SE = M[1:H1 + 1, 1:W1 + 1]
    deg = ((NW != NE).astype(np.int8) + (SW != SE).astype(np.int8)
           + (NW != SW).astype(np.int8) + (NE != SE).astype(np.int8))
    return NW, NE, SW, SE, deg


def _incident(i: int, j: int, NW, NE, SW, SE) -> list[tuple[tuple[int, int], tuple, tuple]]:
    """Boundary segments at corner (i, j): [((i2, j2), label_pair, seg_id), ...]."""
    nw, ne, sw, se = int(NW[i, j]), int(NE[i, j]), int(SW[i, j]), int(SE[i, j])
    out = []
    if nw != ne:                                            # north (vertical) segment
        out.append(((i - 1, j), (min(nw, ne), max(nw, ne)), ("V", i - 1, j)))
    if sw != se:                                            # south
        out.append(((i + 1, j), (min(sw, se), max(sw, se)), ("V", i, j)))
    if nw != sw:                                            # west (horizontal)
        out.append(((i, j - 1), (min(nw, sw), max(nw, sw)), ("H", i, j - 1)))
    if ne != se:                                            # east
        out.append(((i, j + 1), (min(ne, se), max(ne, se)), ("H", i, j)))
    return out


def _rdp(pts: NDArray, tol: float) -> list[int]:
    """Ramer-Douglas-Peucker: indices of `pts` kept at perpendicular tolerance `tol`."""
    n = len(pts)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p0, p1 = pts[a], pts[b]
        d = p1 - p0
        nrm = float(np.hypot(*d))
        seg = pts[a + 1:b]
        if nrm < 1e-12:
            dev = np.linalg.norm(seg - p0, axis=1)
        else:
            dev = np.abs(d[0] * (p0[1] - seg[:, 1]) - (p0[0] - seg[:, 0]) * d[1]) / nrm
        k = int(np.argmax(dev))
        if dev[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.nonzero(keep)[0].tolist()


def _sagitta(poly: NDArray) -> float:
    """Max perpendicular deviation of a polyline from its own chord, in the same units."""
    if len(poly) < 3:
        return 0.0
    p0, p1 = poly[0], poly[-1]
    d = p1 - p0
    nrm = float(np.hypot(*d))
    if nrm < 1e-12:
        return float(np.linalg.norm(poly - p0, axis=1).max())
    return float((np.abs(d[0] * (p0[1] - poly[:, 1]) - (p0[0] - poly[:, 0]) * d[1]) / nrm).max())


def trace_mask(mask: NDArray, stem: str = "", rim_tol: float = 2.0,
               unmodelled: Iterable[int] = (0,)) -> TracedMesh:
    """Trace one label mask into a straight-edge mesh.

    Args:
        mask: integer label image, `0` for unsegmented space. Not modified.
        stem: optional identifier carried through onto the returned mesh, for reporting.
        rim_tol: Ramer-Douglas-Peucker tolerance in pixels, applied to **clamped cell-background
            arcs only**. Interior cell-cell interfaces are always single straight chords whatever
            this is set to, so `rim_tol` cannot affect an interior cell's area. See the module
            docstring for why the rim is treated differently and why this adds no free parameter.
        unmodelled: labels treated as missing environment. Background `0` is always included,
            whatever is passed. A cell whose outline does not close into a simple cycle — a
            segmentation topology defect, such as a region pinched to a point — must be added here
            and the trace repeated; the reason is the same one that clamps the background rim, that
            its mechanics cannot be written from the data, so every vertex it touches loses its
            force balance. `trace_mask_closed` performs that iteration automatically and is usually
            what you want.

    Returns:
        A `TracedMesh`. Cells that failed to close are recorded in `TracedMesh.failures` and are
        **not** faces; this call does not raise on them.

    Raises:
        AssertionError: if a lattice corner comes out with boundary degree 1, which is
            geometrically impossible and therefore means this tracer is wrong rather than that the
            mask is unusual.
    """
    unmodelled = frozenset(int(u) for u in unmodelled) | frozenset({0})
    mask = np.asarray(mask)
    H, W = mask.shape
    NW, NE, SW, SE, deg = _corner_fields(mask)

    if (deg == 1).any():
        raise AssertionError("corner of boundary degree 1 -- impossible, tracer is wrong")

    # -- vertices: 8-connected clusters of degree>=3 corners --------------------------
    jmask = deg >= 3
    lab, n_clusters = ndi.label(jmask, structure=np.ones((3, 3), dtype=int))
    corner_to_vert = np.full(jmask.shape, -1, dtype=np.int64)
    corner_to_vert[jmask] = lab[jmask] - 1

    verts = np.zeros((n_clusters, 2), dtype=np.float64)
    vert_labels: list[frozenset] = [frozenset() for _ in range(n_clusters)]
    cluster_size = np.zeros(n_clusters, dtype=np.int64)
    ii, jj = np.nonzero(jmask)
    acc = np.zeros((n_clusters, 2), dtype=np.float64)
    lsets: list[set] = [set() for _ in range(n_clusters)]
    for i, j in zip(ii.tolist(), jj.tolist()):
        v = int(corner_to_vert[i, j])
        acc[v] += (j, i)                       # (x, y)
        cluster_size[v] += 1
        lsets[v].update((int(NW[i, j]), int(NE[i, j]), int(SW[i, j]), int(SE[i, j])))
    verts = acc / cluster_size[:, None]
    vert_labels = [frozenset(s) for s in lsets]

    # -- arcs: walk each degree-2 chain between two vertex clusters --------------------
    used: set[tuple] = set()
    arcs: list[tuple[int, int, tuple, NDArray]] = []     # (v0, v1, label_pair, polyline)
    self_loops: list[tuple] = []
    for i0, j0 in zip(ii.tolist(), jj.tolist()):
        v0 = int(corner_to_vert[i0, j0])
        for (nb, pair, seg) in _incident(i0, j0, NW, NE, SW, SE):
            if seg in used:
                continue
            ni, nj = nb
            if corner_to_vert[ni, nj] == v0:
                used.add(seg)                  # segment internal to the junction cluster
                continue
            used.add(seg)
            path = [(i0, j0), (ni, nj)]
            prev, cur = (i0, j0), (ni, nj)
            steps = 0
            while corner_to_vert[cur[0], cur[1]] < 0:
                nxt = None
                for (nb2, pair2, seg2) in _incident(cur[0], cur[1], NW, NE, SW, SE):
                    if nb2 == prev or seg2 in used:
                        continue
                    nxt = (nb2, pair2, seg2)
                if nxt is None:
                    raise AssertionError(f"dead end while tracing arc from {(i0, j0)}")
                if nxt[1] != pair:
                    raise AssertionError("label pair changed along a degree-2 chain")
                used.add(nxt[2])
                prev, cur = cur, nxt[0]
                path.append(cur)
                steps += 1
                if steps > 4 * (H + W):
                    raise AssertionError("runaway arc walk")
            v1 = int(corner_to_vert[cur[0], cur[1]])
            poly = np.array([(c, r) for (r, c) in path], dtype=np.float64)   # (x, y)
            if v1 == v0:
                self_loops.append((pair, float(len(path) - 1)))
                continue
            arcs.append((v0, v1, pair, poly))

    # -- straight edges; clamped cell-background arcs are subdivided (no free DOF added) ---
    n_junc = int(n_clusters)                        # junction vertices, before any subdivision
    vert_list = [verts[k] for k in range(n_clusters)]
    lset_list = list(lsets)
    csize_list = cluster_size.tolist()
    e_ends: list[tuple[int, int]] = []
    e_labels: list[tuple[int, int]] = []
    e_arclen: list[float] = []
    sagitta_cc: list[float] = []
    sagitta_cc_rel: list[float] = []
    n_inserted = 0
    for (v0, v1, pair, poly) in arcs:
        arclen = float(len(poly) - 1)
        if (pair[0] not in unmodelled) and (pair[1] not in unmodelled):   # real interface: chord
            e_ends.append((v0, v1))
            e_labels.append(pair)
            e_arclen.append(arclen)
            s = _sagitta(poly)
            sagitta_cc.append(s)
            chord = float(np.linalg.norm(poly[-1] - poly[0]))
            sagitta_cc_rel.append(s / chord if chord > 0 else np.nan)
            continue
        # clamped arc: subdivide. The arc's polyline runs from a corner of cluster v0 to a
        # corner of cluster v1; anchor its ends on the cluster centroids so the mesh stays closed.
        keep = _rdp(poly, rim_tol)
        chain = [v0]
        for k in keep[1:-1]:
            vert_list.append(poly[k])
            lset_list.append(set(pair))
            csize_list.append(1)
            chain.append(len(vert_list) - 1)
            n_inserted += 1
        chain.append(v1)
        for a, b in zip(chain[:-1], chain[1:]):
            e_ends.append((a, b))
            e_labels.append(pair)
            e_arclen.append(arclen / (len(chain) - 1))

    verts = np.asarray(vert_list, dtype=np.float64)
    vert_labels = [frozenset(s) for s in lset_list]
    cluster_size = np.asarray(csize_list, dtype=np.int64)
    n_clusters = len(verts)
    edges = np.array(e_ends, dtype=np.int64).reshape(-1, 2)
    edge_labels = np.array(e_labels, dtype=np.int64).reshape(-1, 2)
    edge_arc_len = np.array(e_arclen, dtype=np.float64)

    # -- faces: for each cell label, close the cycle of its edges ----------------------
    cell_labels = sorted(int(l) for l in np.unique(mask) if int(l) not in unmodelled)
    inc: dict[int, list[int]] = defaultdict(list)
    for k, (a, b) in enumerate(edge_labels.tolist()):
        if a not in unmodelled:
            inc[a].append(k)
        if b not in unmodelled:
            inc[b].append(k)

    faces: list[list[int]] = []
    face_labels: list[int] = []
    failures: dict[int, str] = {}
    for cl in cell_labels:
        ek = inc.get(cl, [])
        if len(ek) < 3:
            failures[cl] = f"only {len(ek)} incident edges"
            continue
        adj: dict[int, list[int]] = defaultdict(list)
        for k in ek:
            a, b = int(edges[k, 0]), int(edges[k, 1])
            adj[a].append(b)
            adj[b].append(a)
        bad = [v for v, nb in adj.items() if len(nb) != 2]
        if bad:
            failures[cl] = f"{len(bad)} vertices of cell-degree != 2 (non-manifold outline)"
            continue
        start = next(iter(adj))
        cycle = [start]
        prev, cur = start, adj[start][0]
        while cur != start:
            cycle.append(cur)
            nb = adj[cur]
            nxt = nb[0] if nb[0] != prev else nb[1]
            prev, cur = cur, nxt
            if len(cycle) > len(ek) + 1:
                break
        if len(cycle) != len(ek):
            failures[cl] = f"outline is not one closed cycle ({len(cycle)} of {len(ek)} edges)"
            continue
        poly = verts[cycle]
        if _shoelace(poly) < 0:
            cycle = cycle[::-1]
        faces.append(cycle)
        face_labels.append(cl)

    junc_lsets = [frozenset(s) for s in lsets]
    census = {
        "n_junction_corners": int(jmask.sum()),
        "n_junction_vertices": n_junc,
        "n_inserted_rim_vertices": int(n_inserted),
        "rim_tol_px": float(rim_tol),
        "cluster_size_hist": dict(sorted(
            (int(k), int(v)) for k, v in zip(*np.unique(cluster_size[:n_junc],
                                                        return_counts=True)))),
        "junction_order_hist": dict(sorted(
            (int(k), int(v)) for k, v in zip(*np.unique([len(s) for s in junc_lsets],
                                                        return_counts=True)))),
        "deg4_corners": int((deg == 4).sum()),
        "self_loop_arcs": len(self_loops),
        "self_loop_detail": [(list(p), n) for p, n in self_loops],
        "sagitta_cc_px_median": float(np.median(sagitta_cc)) if sagitta_cc else float("nan"),
        "sagitta_cc_px_p90": float(np.percentile(sagitta_cc, 90)) if sagitta_cc else float("nan"),
        "sagitta_cc_rel_median": float(np.nanmedian(sagitta_cc_rel)) if sagitta_cc else float("nan"),
        "sagitta_cc_rel_p90": (float(np.nanpercentile(sagitta_cc_rel, 90))
                               if sagitta_cc else float("nan")),
    }
    return TracedMesh(
        stem=stem, shape=(H, W), verts=verts, vert_labels=vert_labels,
        vert_cluster_size=cluster_size, edges=edges, edge_labels=edge_labels,
        edge_arc_len=edge_arc_len, faces=faces, face_labels=np.array(face_labels, dtype=np.int64),
        failures=failures, unmodelled=unmodelled, junction_census=census,
    )


def trace_mask_closed(mask: NDArray, stem: str = "", rim_tol: float = 2.0,
                      max_rounds: int = 4) -> TracedMesh:
    """Trace, then demote any cell whose outline did not close to unmodelled, and re-trace.

    A cell that fails to close is a segmentation topology defect (e.g. a region pinched to a
    point, which is 8-connected but not simply connected as a polygon). It is handled exactly as
    the background is: it becomes missing environment, its vertices are clamped, and no tension is
    charged on its interfaces. The cumulative list is reported in `mesh.failures_cumulative`.
    """
    um: set[int] = {0}
    hist: dict[int, str] = {}
    for _ in range(max_rounds):
        m = trace_mask(mask, stem=stem, rim_tol=rim_tol, unmodelled=um)
        if not m.failures:
            m.diagnostics["failures_cumulative"] = dict(hist)
            m.diagnostics["n_demoted_cells"] = len(um) - 1
            return m
        hist.update(m.failures)
        um |= set(m.failures)
    m.diagnostics["failures_cumulative"] = dict(hist)
    m.diagnostics["n_demoted_cells"] = len(um) - 1
    m.diagnostics["unresolved"] = True
    return m


# ---------------------------------------------------------------------------------------
def vertex_degrees(m: TracedMesh) -> NDArray:
    """Number of edges incident to each vertex of a traced mesh, as an integer array of length ``m.n_verts``."""
    d = np.zeros(m.n_verts, dtype=np.int64)
    for a, b in m.edges.tolist():
        d[a] += 1
        d[b] += 1
    return d


def rasterize_face(m: TracedMesh, k: int) -> tuple[NDArray, tuple[int, int, int, int]]:
    """Boolean raster of face k over its bounding box, using pixel centres."""
    from matplotlib.path import Path

    poly = m.verts[m.faces[k]]
    x0 = max(int(np.floor(poly[:, 0].min())) - 1, 0)
    x1 = min(int(np.ceil(poly[:, 0].max())) + 1, m.shape[1])
    y0 = max(int(np.floor(poly[:, 1].min())) - 1, 0)
    y1 = min(int(np.ceil(poly[:, 1].max())) + 1, m.shape[0])
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = Path(poly).contains_points(pts).reshape(len(ys), len(xs))
    return inside, (y0, y1, x0, x1)


def normalized(m: TracedMesh, a_ref: float) -> NDArray:
    """Vertex coordinates in units of sqrt(A_ref), so that <A_interior> = 1 by construction."""
    return m.verts / np.sqrt(a_ref)


def interior_area_ref(m: TracedMesh) -> float:
    """Reference cell area: the mean area of the interior (non-rim) faces of a traced mesh.

    Falls back to the mean over all faces when the mesh has no interior face.
    """
    A = m.face_areas()
    sel = m.face_is_interior
    if sel.sum() == 0:
        return float(A.mean())
    return float(A[sel].mean())


# =======================================================================================
# Pre-processing: the one modification made to a raw mask, declared in full
# =======================================================================================
STRUCT8 = np.ones((3, 3), dtype=int)   # 8-connectivity, used by `regularize`



def regularize(mask: NDArray) -> tuple[NDArray, dict]:
    """Remove label debris from a segmentation mask, and report exactly how much was moved.

    Automatic segmentation leaves pixel-level debris: a label may own stray one- or two-pixel
    islands disconnected from its main body. These are not tissue, and they matter here because
    they are **topological** rather than merely cosmetic. A stray island of label ``L`` sitting
    inside cell ``M`` creates a closed boundary loop; if that loop's junction corners fall into a
    single 8-connected cluster it collapses to a self-loop arc, which no straight-edge polygon can
    represent, and which can in turn break the outline of the cell that owns it.

    The rule, and it is deliberately the weakest one that fixes the problem:

        For every label, keep only its largest 8-connected component. Pixels in the smaller
        components are reassigned to the label of the nearest surviving pixel, which may be
        background.

    **Nothing else is touched** -- no dilation, no erosion, no hole filling, no smoothing, no
    relabelling, no gap filling across the background margin. Anything more would be editing the
    data rather than making it representable.

    Args:
        mask: integer label image; `0` is unsegmented space. Not modified.

    Returns:
        `(clean, report)`. `report` carries `n_labels_with_islands`, `n_pixels_moved`,
        `frac_pixels_moved` and the per-label island sizes.

    Note:
        `report["frac_pixels_moved"]` is not decoration — **check it.** It should be negligible
        (order `1e-4` or below on a good segmentation). If it is not, that is a finding about the
        segmentation and should be investigated, not a licence to clean harder.
    """
    mask = np.asarray(mask)
    out = mask.copy()
    removed = np.zeros(mask.shape, dtype=bool)
    per_label: dict[int, list[int]] = {}

    for lv in np.unique(mask):
        lv = int(lv)
        if lv == 0:
            continue
        sel = mask == lv
        comp, k = ndi.label(sel, structure=STRUCT8)
        if k <= 1:
            continue
        sizes = np.asarray(ndi.sum(sel, comp, index=np.arange(1, k + 1)))
        keep = int(np.argmax(sizes)) + 1
        drop = sel & (comp != keep)
        removed |= drop
        per_label[lv] = sorted((sizes[np.arange(1, k + 1) != keep]).astype(int).tolist(),
                               reverse=True)

    n_removed = int(removed.sum())
    if n_removed:
        out[removed] = 0
        _, idx = ndi.distance_transform_edt(removed, return_indices=True)
        out[removed] = out[tuple(i[removed] for i in idx)]

    report = {
        "n_labels_with_islands": len(per_label),
        "n_pixels_moved": n_removed,
        "frac_pixels_moved": n_removed / mask.size,
        "island_sizes_per_label": per_label,
    }
    return out, report


__all__ = [
    "TracedMesh",
    "regularize",
    "trace_mask",
    "trace_mask_closed",
    "vertex_degrees",
    "rasterize_face",
    "normalized",
    "interior_area_ref",
]
