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
