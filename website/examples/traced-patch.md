# Traced patch

From a segmentation mask to a relaxed patch: the mask of a cropped tissue is traced into a mesh with
free and clamped vertices, and a patch of the same crop is relaxed with its clamped vertices held in
place. The data are the test fixtures of the repository, `tests/trace_fixture_demo_003_pos2_mask.tif`
and `tests/fixtures/clamped_patch_demo_003_pos2.npz`; the paths are relative to the root of the
clone. The steps follow `tests/test_trace.py`, `tests/test_clamped.py` and the README.

## Tracing

```python
import numpy as np
import tifffile

from vertax.mask_analysis import find_vertices_edges_faces
from vertax.trace import regularize, trace_mask, trace_mask_closed

mask = tifffile.imread("tests/trace_fixture_demo_003_pos2_mask.tif")
clean, report = regularize(mask)
print(f"{len(np.unique(mask)) - 1} labels; {report['n_pixels_moved']} pixel(s) moved by regularize")

traced = trace_mask_closed(clean, rim_tol=2.0)
print(f"{traced.n_verts} vertices, {int(traced.free.sum())} free; {traced.n_faces} faces, "
      f"{int(traced.face_is_interior.sum())} interior")
```

`rim_tol` simplifies the outline against the background only. It changes the number of vertices on
the rim, not the area of an interior cell:

```python
by_tolerance = {tol: trace_mask(clean, rim_tol=tol) for tol in (1.0, 2.0, 8.0)}
for tol, m in by_tolerance.items():
    interior = dict(zip(m.face_labels[m.face_is_interior].tolist(), m.face_areas()[m.face_is_interior]))
    print(f"rim_tol {tol}: {m.n_verts} vertices, total interior area {sum(interior.values()):.1f} px^2")
```

Declaring a cell unmodelled removes it from the faces and clamps the vertices around it. This costs
free vertices only when the cell was inside the patch:

```python
base = trace_mask(clean, rim_tol=2.0)
rim_label = int(base.face_labels[~base.face_is_interior][0])
interior_label = int(base.face_labels[base.face_is_interior][0])
for label in (rim_label, interior_label):
    m = trace_mask(clean, rim_tol=2.0, unmodelled=(label,))
    print(f"cell {label} unmodelled: {m.n_faces} faces, {int(m.free.sum())} free vertices "
          f"(instead of {base.n_faces} and {int(base.free.sum())})")
```

The extraction used by `PbcMesh.from_mask` does not close on such a crop, which is why
`vertax.trace` exists:

```python
verts, edges, faces = find_vertices_edges_faces(mask)
print("mask_analysis on the raw crop: V - E + F =", len(verts) - len(edges) + len(faces))
```

## Relaxing the patch

The patch file holds the same crop in the format of `vertax.clamped.load_mesh`. The energy is
written by the caller; here, unit tensions on the cell-cell edges and an area term of stiffness
`K = 5` towards area 1.

```python
import jax.numpy as jnp
from jax.experimental import enable_x64

from vertax.clamped import clamped_edges_are_inert, load_mesh, patch_geometry, relax, signed_face_areas

with enable_x64():
    patch = load_mesh("tests/fixtures/clamped_patch_demo_003_pos2.npz")
    print(f"{len(patch.x0)} vertices, {patch.n_free} free, {patch.nf} faces")
    print("edges along unimaged space act on free vertices:",
          clamped_edges_are_inert("tests/fixtures/clamped_patch_demo_003_pos2.npz")["max_abs_diff_free"])

    gamma = jnp.ones(len(patch.EA))

    def E(x, K):
        A, P, L = patch_geometry(x, patch)
        return 0.5 * K * jnp.sum((A - 1.0) ** 2) + jnp.sum(gamma * L)

    out = relax(E, patch, K=5.0)
    clamped = np.setdiff1d(np.arange(len(patch.x0)), patch.free_idx)
    print("clamped vertices moved:", float(np.abs(out["x"][clamped] - patch.x0[clamped]).max()))
    print(f"residual: observed {out['r_eq_obs']:.2e}, relaxed {out['r_eq']:.2e}")
    print("smallest signed area:", out["min_area"], "; inverted faces:", out["n_inverted"])
    print("signed areas of the first faces:", np.round(np.asarray(signed_face_areas(jnp.asarray(out['x']), patch))[:5], 3))
```

`r_eq_obs` is the residual of the observed configuration: how far the tissue, as imaged, is from the
balance of this energy. `r_eq` is the residual after relaxation, over the free vertices only.
