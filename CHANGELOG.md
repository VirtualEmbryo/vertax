# Changelog

## 0.4.0 — <date to set when tagging>

The library of the VertAX paper: the Hessian-weighted cost, bounded T1 tension reset and aspect-ratio costs enter, the image costs leave, and the line-search and L-BFGS fixes are included. A minor rather than a patch release: removing the image costs removes public functions, and the fixes change bounded and line-search results.

One library for the paper, its figure code and the documentation: the features that the VertAX paper's figure code uses and that only its branch had are merged in, the image costs leave the library, and the Hessian-weighted cost of the paper enters it. Without the new options, results are those of 0.3.0 with the fixes below.

### Added: the Hessian-weighted cost $\mathcal C_W$ and the staged schedule

- `hessian_at_target(target, energy, edges_params=None)`: the Hessian of the inner energy at the target geometry, every edge parameter set to one by default, symmetrized and divided by its spectral norm.
- `make_cost_hessian_weighted(target, energy, edges_params=None)`: $\mathcal C_W=\tfrac12\lVert\widehat{\mathbf H}(\mathbf X-\mathbf X_{\text{target}})\rVert^2$ in the periodic outer-cost signature, with displacements to the nearest periodic image as in `cost_v2v`.
- `switch_cost_at_target_topology(optimizer, mesh, cost)`: called after every epoch, it disables T1 transitions and sets the outer cost at the first epoch where the mesh has the target topology, the second stage of the staged schedule ($\mathcal C_{\text{v2v}}$, then $\mathcal C_W$).
- New test: `tests/test_cost_hessian_weighted.py`, against the values of the code that defined $\mathcal C_W$ for Figure 3 of the paper, on two mesh pairs of the synthetic benchmark.

### Added: from the paper's branch

- `T1TensionReset` (bounded meshes, off by default): the edge created by a T1 transition starts at the mean tension of the other edges of its kind instead of inheriting the tension of the edge it replaces. `BoundedBilevelOptimizer.t1_tension_reset` and `last_t1_resets`; with the option, `update_T1_bounded` returns the post-reset parameters and the transitions per edge, and the bounded outer functions return a `T1ResetReport` as an eighth value. New test: `tests/test_t1_tension_reset_bounded.py`.
- `reopen_margin`, a keyword of `update_T1_bounded`: length of the reopened edge, `(1 + reopen_margin) * min_dist_T1`, as for the periodic `update_T1`; default unchanged (`min_dist_T1 + 1e-3`). New test: `tests/test_bounded_t1_reopen_margin.py`.
- `get_cluster_moments_bounded` (area, centroid and central second moments of a bounded cluster, arc caps included), `aspect_ratio_along_x`, `aspect_ratio_principal`, and the outer costs `make_cost_aspect_ratio_along_x` and `make_cost_aspect_ratio`. New test: `tests/test_cluster_aspect_ratio.py`; `tests/test_bounded_crescent_cell.py` also checks the cluster moments of a crescent cell.

### Removed: the image costs

- `cost_dmic`, `make_cost_dmic`, `render_interface_image`, `cost_mesh2image`, `make_cost_mesh2image`, `erode_mask_for_psf` and the constant `LEGACY_PSF_SIGMA`, with their tests, their README section and their website pages. They are kept on the branch `image-cost`.
- The outer-cost protocol keeps its last argument, `image_target`, which no library cost now uses.

### Fixed since 0.3.0

- The line search scores the vertex steps capped and the bounded arc angles clipped, as the update leaves them.
- L-BFGS: the stored positions follow the vertices that `update_pbc` wraps across the periodic box.
- Bounded meshes: opt-in reset of the solver state after a T1 transition (`reset_solver_on_t1`); `update_T1_bounded` skips the triangular-twin-face emergency when the twin half-edge is an arc.
- The bounded inverse regression runs on a clipped-SGD outer step; reference regenerated on Linux.

## 0.3.0 — 2026-10-28

Four fixes, docstring corrections and test tolerances for references made on another machine. A minor
rather than a patch release: the bounded-angles fix changes bounded bilevel results and the return value of
the four bounded outer functions.

### Fixed: a bounded outer step leaves the arc angles of the estimator's equilibrium on the mesh

- The automatic-differentiation, equilibrium-propagation and implicit-differentiation estimators of
  `BoundedBilevelOptimizer` replace `mesh.vertices`, `mesh.edges` and `mesh.faces` by the equilibrium of
  their inner solve (the free one, for EP), but kept the old `mesh.angles`: the mesh was left with the
  vertices of one state and the arc angles of another, and the relaxation that `bilevel_optimization`
  and `do_n_bilevel_optimization` run next started from that mixed state. The functional
  `opt_bounded.bilevel_opt_bounded` did the same. The angles now go with the other tables.
- API: `outer_opt_bounded`, `outer_eq_prop_bounded`, `outer_implicit_bounded` and
  `outer_adjoint_state_bounded` return `(vertices, angles, edges, faces, vertex, edge and face
  parameters)`, seven arrays instead of six. The adjoint-state estimator returns the state it was given,
  unchanged, as before.
- New test: `tests/test_bounded_outer_step_angles.py`.

**Bounded bilevel results change** for the AD, EP and ID estimators whenever a relaxation follows an outer
step on the same mesh (`bilevel_optimization`, `do_n_bilevel_optimization`, `bilevel_opt_bounded`).
The parameter update of an outer step is unchanged, and so is a run that restores the mesh state after
`outer_optimization` and relaxes from it. The bounded inverse regression reference
(`tests/reference_result_test_inverse_modeling_bounded.npz`) is regenerated: final energy 51.766 (was
51.643), outer cost 0.5616 (was 0.5760). Periodic meshes are not affected.

### Fixed: `BoundedBilevelOptimizer` honours `params_regularizer` and `ep_variant`, and refuses `persist_outer_state`

- The bounded optimizer passed neither `params_regularizer` nor `ep_variant` to its estimators: a bounded
  run ignored the regularizer, whose docstring says it is honoured by all estimator paths, and its EP
  estimate was always centered. Both are now passed to the four bounded estimators, as in the periodic
  ones; `EPVariant.ONE_SIDED` solves the free problem at `beta = 0` and divides by `beta`.
- `persist_outer_state=True`, which the bounded EP path ignored, now raises `ValueError` at the first
  outer step: only `PbcBilevelOptimizer` threads the outer optimizer state across epochs.
- New test: `tests/test_bounded_bilevel_options.py`.

**This fix alone leaves results with the defaults (no regularizer, `CENTERED`, `persist_outer_state=False`)
unchanged, bit for bit.**

### Fixed: `cost_areas` can be used as an outer cost

- `cost_areas` called `geo.get_area` without its face-walk bound and raised `TypeError` at its first call.
  Its arguments were also not in the 12-argument order in which the optimizers call an outer cost; they
  now are (targets before selections), `width` and `height` are static as in `cost_v2v`, and
  `selected_faces=None` means every face.
- New test: `tests/test_cost_areas.py`.

### Fixed: `minimize` with a line-search solver and a target other than the vertices

- The `params` and `value_fn` that `minimize` hands to `solver.update` were chosen by branches numbered
  0, 2, 3, 4, 5, 6 while `OptimizationTarget` is numbered 0 to 5. For every target other than `VERTICES`
  a line-search solver (`gd_linesearch`, `cg_linesearch`, `optax.lbfgs()`) received another array and
  re-evaluated the energy along it; with the usual table shapes the call raised `TypeError`. Fixed-step
  solvers do not use `value_fn` and were not affected.
- New test: `tests/test_minimize_value_fn_targets.py`.

**Results are unchanged for `OptimizationTarget.VERTICES`**, the only target VertAX itself uses (the branch
is unchanged).

### Documentation

Docstrings that said something the code does not do, corrected without a change of behaviour:
`BoundedMesh.faces` is `(nb_faces, 2)`, a half-edge of the face and the cell's fate;
`make_energy_line_tensions` returns a 7-argument function that has to be wrapped as an inner loss;
`BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION` is reverse mode (`jax.grad`), not forward mode;
`trace_mask_closed` does not raise, and its record is in `mesh.diagnostics`; no VertAX function writes
the file that `clamped.load_mesh` reads; `save_mesh_txt` and `load_mesh_txt` also write and read the
parameter tables and the constants. `minimize` now says to build the solver once and reuse it: the
relaxation loop is compiled for each solver object, so a new solver at every call recompiles and keeps
every compiled copy.

### Tests

Four tests compared floating-point results bit for bit with references made on a Mac and failed on
Linux x86-64, off in the last bits (`test_default_float_dtype`, `test_trace`, and two in `test_clamped`).
They now compare within a tolerance derived from the computation (float32 ulps of the array's scale,
or the rounding bound of a reordered sum), and integer and exact lattice quantities stay exact. Set
`VERTAX_EXACT_REFERENCES=1` on the machine that produced the references to compare them bit for bit
(new fixture `exact_references` in `tests/conftest.py`). Three of these tests are renamed, without
`exactly` or `bit_identical` in their names.

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
