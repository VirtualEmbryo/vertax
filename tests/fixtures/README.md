# Test fixtures

Small, explicitly-committed inputs for the released test suite.

## `mesh_1008.npz`, `mesh_target_1008.npz`

One initial/target mesh pair from the synthetic `E2` benchmark at `d_topo = 0`, 3.6 KiB each.

**That dataset is gitignored and stays gitignored.** It is 120 files and has been swept into the
index three times by directory-level `git add`. These two files are added by explicit path, which is
the rule that was broken then and still stands. Do not add the directory.

They exist so that verification harnesses which establish `src/` behaviour can live in the released
test suite instead of in `analysis/`, where they are not shipped and cannot run for anyone else.

`PbcMesh.load_mesh` reads them directly; `20` faces, `K_areas = 10`, `float64` analysis.

## `bounded_mesh_with_turned_over_cell.npz`

A relaxed 20-cell bounded mesh (`BoundedMesh.save_mesh` format, float32) in which the chord polygon of
face 3 has turned over: the face was counterclockwise at construction (signed polygon +0.085) and is
-0.360 after the relaxation, while its arc caps add 0.219, so its signed area is negative and its
outline crosses itself. It is the discriminating case of `test_area_bounded_groundtruth.py`, which
asserts that the cell area is the signed chord polygon plus the caps. It was produced by the relaxation
of `test_forward_modeling_bounded.py` under an energy that read the area of such a cell as
`|polygon| + caps`, which cannot resist the fold; the energy now reads the signed area, so that
relaxation no longer produces it, and the file is kept as a fixed input rather than regenerated.
