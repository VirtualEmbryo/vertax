# Changelog

## Unreleased — changes since the merge base `ed504f2`

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
