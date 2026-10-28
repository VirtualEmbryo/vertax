# Release notes

What each release changes for a user, and what you may have to change in your own code. The
[changelog](changelog.md) has the full list.

## 0.2.1 (2026-10-22)

A bug-fix release, with two fixes.

**Meshes follow JAX's default float dtype.** The constructors of `PbcMesh` (`from_seeds`,
`from_random_seeds`, `from_mask`, `from_image`) and of `BoundedMesh` (`from_seeds`,
`from_random_seeds`) used to build `float32` vertices (and angles) even with `jax_enable_x64`
enabled. They now use JAX's default float dtype: `float32` by default, `float64` under
`jax_enable_x64`. If your code cast the vertices to `float64` after building a mesh, the cast is no
longer needed. Without x64, results are unchanged. See [Numerical precision](user-guide/precision.md).

**The area of a bounded cell is signed.** A free-interface cell whose interior vertices cross the
chord of its arc (a crescent) used to be given the area of the cap plus that of the chord polygon;
it now gets the cap minus the polygon, and a folded cell gets a non-positive area. This changes the
area in `get_area_bounded`, `BoundedMesh.get_area`, the bounded energy and the energy check of the
bounded T1 update. Results are unchanged, bit for bit, for bounded meshes without such a cell.

**Bounded meshes are checked when built or loaded.** `BoundedMesh.from_seeds`, `load_mesh` and
`load_mesh_txt` raise a `ValueError` for a mesh whose faces are clockwise, and emit a `UserWarning`
for a face whose chord polygon has turned over. A mesh built and saved by VertAX is never refused.

## 0.2.0

Changes since the merge base `ed504f2`.

### What you may have to change

- The line-tension energies are built by `make_energy_line_tensions(K_areas)` and
  `make_energy_line_tensions_bounded(K_areas, ...)`. They replace the former energies with a fixed
  area stiffness, and `K_areas` has no default.
- `plot_mesh` no longer displays or closes the figure by default, and returns `(fig, ax)`. Pass
  `show=True` to display it.
- With a line-search inner solver (`gd_linesearch`, `cg_linesearch`, `optax.lbfgs()`), each vertex
  step is now capped at `5 * min_dist_T1` by default. Set `cap_vertex_step = False` on the optimizer
  to get the previous behaviour. Fixed-step solvers (`optax.sgd`, `optax.adam`) are not affected.
- `cost_v2v` follows the published formula, and `cost_dmic` the published interface-density model,
  so their values differ from earlier versions. The former IAS cost family and the unused topology
  costs are removed, and `d_ias` is renamed `d_topo`.
- Centered equilibrium propagation was corrected: the free phase is solved at `-beta`.

### New

- The modules `vertax.trace` (mesh from a label mask), `vertax.clamped` (force balance on a cropped
  patch) and `vertax.rheology` with `ShearRheologyOptimizer` (shear rheometry of periodic tissues).
- `params_normalization` on the bilevel optimizers, a mean-value constraint on any parameter array;
  `mean_tension_gauge` remains as an alias for the edge parameters.
- `make_energy_shape_factor(lambda_area)`, the shape-factor energy with an area stiffness.
- `make_cost_dmic`, and options of `render_interface_image`: intensity normalization, pixel-footprint
  averaging and the PSF width as a call-time parameter.
- Opt-in options of the relaxation: `minimize(..., reset_solver_on_t1=True)`,
  `minimize(..., count_t1=True)` and `recording_t1_statistics()`, `update_T1(..., reopen_margin=...)`,
  and `linesearch_initial_guess="one"` for `cg_linesearch` and `gd_linesearch`.

### Fixes

- `get_area_bounded` adds the arc caps on clockwise faces.
- The periodic T1 swap repair is refused on bounded meshes.
- `load_mesh_txt` no longer truncates parameter arrays to integers.
- `PbcMesh` raises an error instead of hanging when it cannot order the edges of a face.
- `PbcMesh.from_mask` and `PbcMesh.from_image` are marked unsupported for real masks.
- Nonlinear CG keeps the dtype of its integer state under `jax_enable_x64`; `optax.lbfgs()` runs under
  `jax_enable_x64`; the arc half-angle limit of bounded meshes is applied after every relaxation
  step, also without T1 transitions.
