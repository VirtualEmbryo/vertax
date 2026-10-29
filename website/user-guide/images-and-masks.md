# Meshes from images and masks

VertAX has two routes from a segmentation to a mesh, and they give different kinds of mesh. Neither
gives a working periodic mesh of a real tissue on main.

| Route | Input | Output | Status |
| --- | --- | --- | --- |
| `PbcMesh.from_mask`, `PbcMesh.from_image` | a label mask with no holes, or an image | a periodic `PbcMesh` | unsupported for real masks |
| `vertax.trace` | a label mask, possibly a crop with unsegmented margin | a `TracedMesh`, a patch with a boundary | tested against a stored reference |

## Periodic meshes from a mask: `from_mask` and `from_image`

[`PbcMesh.from_mask`][vertax.meshes.pbc_mesh.PbcMesh.from_mask]`(mask)` pads the mask on every side
with mirror images of itself, extracts vertices, edges and faces from the padded mask with
`vertax.mask_analysis.find_vertices_edges_faces`, and keeps one periodic copy. The periodic box is
twice the size of the mask in each direction, because of the mirroring.
[`PbcMesh.from_image`][vertax.meshes.pbc_mesh.PbcMesh.from_image]`(image)` first segments the image
with Cellpose (`vertax.mask_analysis.mask_from_image`: a Gaussian blur, the `cyto` model, then the
background is filled by growing the labels), then calls `from_mask`.

Their docstrings mark both as unsupported for real masks. The junction extraction keeps a triple of
labels only where exactly one 2 x 2 window of pixels carries it, while a real junction is usually
spread over several windows: real junctions are lost and spurious short cycles appear. The docstrings
report, for `examples/image.tif`, $V - E + F = 55$ ("a correct planar mesh gives 0") and 472 faces
for 64 segmented cells. The notebook `examples/pbc_from_image.ipynb`
carries the same warning.

## Tracing a label mask: `vertax.trace`

[`vertax.trace`](../api/trace.md) turns a label mask, an integer image where each cell has its own
positive label and unsegmented space is 0, into a mesh with straight edges. It is written for crops
of microscopy images, where some cells are cut by the field of view.

- Vertices are found on the lattice of pixel corners: a corner where three or more boundary segments
  meet is a junction, and neighboring junction corners are merged into one vertex at their centroid.
- An edge between two cells is a straight chord between two vertices. An edge between a cell and the
  background follows the outline, simplified with the Ramer-Douglas-Peucker algorithm to a tolerance
  of `rim_tol` pixels (2 by default).
- A face is the cycle of edges of one cell label, listed counterclockwise. The background is not a face.
- A vertex is **clamped** when the background (or another label declared unmodelled) is one of the
  labels that meet there, and **free** otherwise. Only free vertices have a complete force balance;
  mechanical equations should be written for them only.

The functions are:

- [`regularize`][vertax.trace.regularize]`(mask)` keeps, for each label, its largest 8-connected
  piece, and gives the pixels of the other pieces to the label of the nearest remaining pixel, which
  may be the background. It returns the
  cleaned mask and a report; its docstring asks you to check that `report["frac_pixels_moved"]` is
  negligible.
- [`trace_mask`][vertax.trace.trace_mask]`(mask, stem="", rim_tol=2.0, unmodelled=(0,))` traces the
  mask. Cells whose outline does not close into a simple cycle are listed in `failures` and are not
  faces.
- [`trace_mask_closed`][vertax.trace.trace_mask_closed]`(mask, stem="", rim_tol=2.0, max_rounds=4)`
  traces, declares the cells that failed as unmodelled, and traces again, up to `max_rounds` times.
  The cells removed this way are in `mesh.diagnostics["failures_cumulative"]`; if some remain after
  the last round, `mesh.diagnostics["unresolved"]` is `True`. It does not raise.
- [`interior_area_ref`][vertax.trace.interior_area_ref]`(mesh)` is the mean area of the faces that
  touch no background, and [`normalized`][vertax.trace.normalized]`(mesh, a_ref)` the vertex
  coordinates in units of $\sqrt{a_\text{ref}}$.

The [`TracedMesh`][vertax.trace.TracedMesh] it returns is in pixel units: `verts` holds $(x, y)$
coordinates, `edges` pairs of vertex indices, `faces` lists of vertex indices, and the properties
`free`, `clamped`, `edge_is_cell_cell` and `face_is_interior` give the boundary partition.

```python
import numpy as np
import tifffile

from vertax.trace import interior_area_ref, normalized, regularize, trace_mask_closed

mask = tifffile.imread("tests/trace_fixture_demo_003_pos2_mask.tif")
print(mask.shape, mask.dtype, "labels:", len(np.unique(mask)) - 1)

clean, report = regularize(mask)
print("fraction of pixels moved:", report["frac_pixels_moved"])

traced = trace_mask_closed(clean, rim_tol=2.0)
print(f"{traced.n_verts} vertices ({int(traced.free.sum())} free), "
      f"{traced.n_edges} edges ({int(traced.edge_is_cell_cell.sum())} between two cells), "
      f"{traced.n_faces} faces ({int(traced.face_is_interior.sum())} interior)")
print("cells declared unmodelled:", traced.diagnostics["failures_cumulative"])

verts = normalized(traced, interior_area_ref(traced))
print("mean interior area after normalization:",
      float(np.mean(traced.face_areas(verts)[traced.face_is_interior])))
```

The mask read here is a test fixture of the repository; the path is relative to the root of the clone.

A traced mesh is a patch with a boundary, not a periodic mesh, and VertAX has no function that turns
it into a `PbcMesh` or a `BoundedMesh`. Its mechanics are those of [clamped patches](clamped-patches.md).
