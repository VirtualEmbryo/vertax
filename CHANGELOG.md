# Changelog

## 0.2.1 — 2026-10-22

Bug-fix release, with two fixes.

### Fixed: mesh constructors follow JAX's default float dtype

- `PbcMesh` (`from_seeds`, `from_random_seeds`, `from_mask`, `from_image`) and `BoundedMesh`
  (`from_seeds`, `from_random_seeds`) built their `vertices` (and, for `BoundedMesh`, their `angles`)
  as `float32` whatever `jax_enable_x64` was set to, so a user who enabled x64 still relaxed a
  `float32` mesh. Both now use JAX's default float dtype: `float32` by default, `float64` under
  `jax_enable_x64`. Integer index tables are unchanged.
- Docstrings and notes that told users to cast `PbcMesh.vertices` to `float64` after construction
  (`rheology`, `clamped`, `require_float64`'s error message) are updated.
- New regression tests: `tests/test_default_float_dtype.py`, with reference arrays from 0.2.0.

**Results without `jax_enable_x64` are unchanged; users who enabled x64 without casting the mesh now get
float64, as intended.** The mesh constructors and one relaxation of each mode are bit-identical to 0.2.0
without x64 (checked by the new tests).

### Fixed: area of a bounded cell whose chord polygon has turned over

- `get_area_bounded` (and with it `BoundedMesh.get_area`, the bounded energy and the energy check of the
  bounded T1 update) returned `0.5 (|polygon| + caps)` for a cell, where *polygon* is the signed area of
  the polygon through the cell's vertices and *caps* the area between each free-interface chord and its
  arc. `BoundedMesh` builds every face counterclockwise and T1 transitions keep it, so the sign of the
  polygon is not the cell's orientation: it turns over when a free-interface cell's interior vertices
  cross the chord of its arc while the arc keeps the cell open (a crescent). Such a cell was given
  cap + |polygon| instead of cap − |polygon|, and a folded cell was hidden. The area is now the signed
  polygon plus the caps: a crescent has area cap − |polygon|, and a folded cell has a non-positive area.
- Because the area now relies on the counterclockwise convention, `BoundedMesh.from_seeds`, `load_mesh`
  and `load_mesh_txt` check it. A mesh whose faces are clockwise raises `ValueError` (it would be read
  inside out, with negative areas and no error); a mesh with a face whose chord polygon has turned over,
  a crescent or a fold, emits a `UserWarning`. A mesh built and saved by VertAX is never refused.
- New tests: `tests/test_bounded_crescent_cell.py` (closed-form area and energy gradient of a crescent,
  on the configuration and its mirror image) and `tests/test_bounded_mesh_orientation_guard.py`.
  `tests/test_area_bounded_groundtruth.py` now asserts the signed area, on a stored mesh with a folded cell.

**Results are unchanged, bit for bit, for any bounded mesh without a turned-over cell** (checked on the two
stored bounded regression meshes: the other 19 cell areas are identical). The two bounded regression
references (`tests/reference_result_test_{forward,inverse}_modeling_bounded.npz`) held a folded cell that
the unsigned area could not resist, and are regenerated: that relaxation no longer folds the cell.

## 0.2.0 — changes since the merge base `ed504f2`

### Solver and T1 options (defaults unchanged)

- Nonlinear CG: integer state keeps its dtype under `jax_enable_x64` (fix);
  `linesearch_initial_guess="one"` (opt-in; default `"keep"`) starts every line
  search from a unit step.
- `update_T1(..., reopen_margin=0.1)`: length of the reopened edge as a keyword
  (default unchanged, `1.1 * min_dist_T1`).
- Bounded mode: the arc half-angle limit is applied after every relaxation step,
  also with T1 transitions disabled (fix).
- `minimize(..., reset_solver_on_t1=True)`: opt-in reset of the solver state after
  a T1 transition (default `False`); `optax.lbfgs()` now runs under
  `jax_enable_x64` (fix).
- `minimize(..., count_t1=True)`: opt-in T1 counts in the loop state, exact under
  `jit`, `vmap` and `grad` (default `False`); `recording_t1_statistics()` collects
  them from the bilevel estimators.

### Vertex step cap and T1 counts

- `cap_vertex_step` on `PbcBilevelOptimizer` and `BoundedBilevelOptimizer`
  (default `None` = automatic): the per-step vertex displacement is capped at
  `5 * min_dist_T1` for line-search inner solvers (`gd_linesearch`,
  `cg_linesearch`, `optax.lbfgs()`, recognised by the line-search state they
  carry) and left uncapped for fixed-step ones (`optax.sgd`, `optax.adam`, whose
  results are unchanged). An explicit `True` or `False` always wins. The choice
  reaches every relaxation the optimizer runs, including the inner solves of the
  automatic-differentiation, equilibrium-propagation, implicit-differentiation
  and adjoint-state hypergradients. **Behaviour change for line-search users**:
  an uncapped line-search step can move a vertex across whole cells and turn a
  cell inside out, which `get_area` and `get_area_bounded` (absolute values) do
  not penalise; set `cap_vertex_step = False` to keep the old behaviour. Original
  work by Michael Corbeau.
- Fix: `recording_t1_statistics()` now also records the two solves of the
  equilibrium-propagation hypergradient (they were dropped).
- Note: the "kept" count of `minimize(..., count_t1=True)` is exact only when the
  flips of one iteration share no cell; under the forced T1 rule neighbouring
  flips can make it differ from the flips performed, and presented minus kept is
  then unreliable.

### New modules

- **`vertax.trace`**: traces a segmentation label mask, including a raw crop,
  into a straight-edge vertex mesh, with a free/clamped vertex partition and
  mask regularisation.
- **`vertax.clamped`**: force balance on a cropped tissue patch with clamped rim
  vertices, using signed areas; `relax`, `newton_polish`, residual and
  gradient-splitting utilities.
- **`vertax.rheology`** and **`vertax.bilevelopt.rheobop`**: bulk-driven shear
  rheometry on periodic tissues (cosine drive mode, compliance, modulus) and
  `ShearRheologyOptimizer` for inverse design against a target modulus.

### Image-based cost

- `cost_dmic` rewritten to the published interface-density model;
  `make_cost_dmic`.
- `render_interface_image`: optional intensity normalisation, pixel-footprint
  averaging (exact pixel average), and the PSF width as a call-time parameter.

### Energies and gradient estimators

- `make_energy_line_tensions(K_areas)` and
  `make_energy_line_tensions_bounded(K_areas, ...)` replace the former
  fixed-stiffness energies; `K_areas` is always explicit.
- New shape-factor energy with area stiffness.
- Centered equilibrium propagation corrected (free phase at `-beta`); the
  `2x beta` convention is gone from docstrings.
- Mean-tension normalization constraint (post-step projection) generalised to
  any parameter array via `params_normalization`; `mean_tension_gauge` remains
  as an alias. `params_regularizer` is honoured on every estimator path.

### Costs

- `cost_v2v` fixed to the published formula; `d_ias` renamed `d_topo`; the
  former IAS cost family and the dead topology costs are removed.

### Fixes

- `get_area_bounded` adds the arc caps on clockwise faces.
- Periodic T1 swap repair refused on bounded meshes.
- `load_mesh_txt` no longer truncates parameter arrays to integers.
- `PbcMesh` raises instead of hanging in periodic face ordering.
- `plot_mesh` does not display or close the figure by default and returns
  `(fig, ax)`.
- `from_mask` and `from_image` are marked unsupported for real masks.

### Tests, environment and documentation

- Test suite runnable, order-independent and green; periodic regression test in
  two tiers; long-running tests carry the `long` marker and are deselected by
  default (`pytest -m long`); `pytest-timeout` added to the dev group.
- Pinned environment (`uv.lock`, `requirements.txt`, Python 3.12).
- API documentation regenerated with pdoc 16; tension inference guide updated
  for the mean-tension normalization constraint; README section on the new
  modules.
