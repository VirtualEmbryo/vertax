# Changelog

## Unreleased — changes since the merge base `ed504f2`

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
