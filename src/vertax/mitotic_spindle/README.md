# Mitotic Spindle Simulation

Simulation of successive mitotic cell divisions on a 2D epithelial tissue using VertAX.

## Differences with `bounded_division`

### 1. Physical spindle model (our contribution)

`bounded_division` takes the division axis as an **input parameter** — it does not determine it.

We implement a full biophysical model (`spindle_model.py`) with three selectable methods (`method` parameter of `find_spindle`):

**`method='newton'` (default)** — full 3-DOF equilibrium:
- Each centrosome emits microtubules in all directions θ ∈ [0, 2π]
- Elementary force: dF(θ) = Λ(θ) · L(θ) · ê(θ) · dθ
- Spindle equilibrium (c, φ) found by solving F₁ + F₂ = 0, M₁ + M₂ = 0 via Newton-Raphson with JAX automatic differentiation
- Initialisation: geometric centroid + principal inertia axis
- Fallback to (centroid, inertia axis) if Newton diverges or places the centre outside the cell

**`method='pdf'`** — Boltzmann orientation density (Thery et al.):
- The orientational potential is V̄(α) = −∫₀^α T̄(α') dα', where T̄(α) is the net torque at orientation α with the spindle centred at the geometric centroid
- The probability density p(α) = p₀ exp(−V̄(α)/C) is computed over α ∈ [0, π] (trapezoidal integration, 180 samples)
- The division axis is the **mode** of p(α) — the most probable orientation
- C (`pdf_C`) is a noise parameter: small → peaked near equilibrium, large → nearly uniform

**`method='pdf_sample'`** — stochastic draw from p(α):
- Same density as `'pdf'`, but the division axis is **randomly sampled** from p(α)
- Models the stochastic variability observed experimentally (thermal noise + nuclear friction)

### 2. Cell division: direct port of `bounded_division` with a `BoundedMesh` wrapper

`cell_division.py` embeds `bounded_division`'s JAX-compiled `cell_division()` verbatim, plus a `divide_cell(mesh, face_id, c, u)` wrapper.

The wrapper is necessary because `bounded_division`'s `cell_division()` operates on raw JAX arrays `(vertTable, angTable, heTable, faceTable, divOrders, divAxes)`, while our pipeline works with `BoundedMesh` objects and a spindle vector `u` rather than an angle. The wrapper:
1. Extracts the raw tables from the `BoundedMesh`.
2. Converts the spindle vector `u` to an angle `alpha = atan2(uy, ux)`.
3. Calls `cell_division()` and trims the padded output arrays.
4. Reconstructs a new `BoundedMesh` with propagated `edges_params`, `faces_params`, and `vertices_params`.

Two bugs in `bounded_division` are fixed in the process:
- **`div_face` index bug**: `body_fun` passed the loop index `i` as `div_face` instead of the actual `face` id — causing `faceTable[0]` to be overwritten on every single-cell division. Fixed by passing `face` instead.
- **`jax.lax.select` dtype conflict**: in JAX x64 mode, Python integer literals are `int64` while computed half-edge indices are `int32`. `lax.select` requires identical dtypes; all `jax.lax.select` calls replaced by `jnp.where`, which auto-promotes.

### 3. Arc intersection detection

`bounded_division`'s `get_intersection()` handles both straight edges and circular arcs in a unified JAX-compiled function. The algebraic approach:
- Reconstruct the circle from chord and inscribed angle
- Solve line-circle intersection algebraically (quadratic)
- Filter candidate points via dot-product angular constraints (no arccos)

### 4. Post-division T1 transitions

`bounded_division` provides `update_T1_div` (`topo_div.py`), a specialised T1 transition function for the post-division context with 5 edge-case handlers for degenerate topologies.

We use the standard `update_T1_bounded` from VertAX main.

### 5. Area homogenisation after division (our contribution)

`bounded_division` has no post-division relaxation strategy specific to area homogenisation.

VertAX `cost.py` provides `cost_areas` which minimises `Σ (area_i - area_i_target)²` against a **reference mesh** — designed for inverse inference, not post-division homogenisation.

We implement `outer_loss_homogenise` (`energy_adaptive.py`) which minimises the **variance of current cell areas** with no external reference:

```
L_out = K · mean((area_i - mean_area)²) / mean_area²
```

This drives all cells toward a uniform size after each division, using the bilevel outer optimisation loop of VertAX.

### 6. Adaptive target area (our contribution)

The standard VertAX inner energy uses a fixed `TARGET_AREA` constant.

We implement `energy_adaptive` which reads the target area from `face_params[0]`, updated after each division to `initial_total_area / nb_faces`. This ensures the mechanical equilibrium target scales correctly as the number of cells grows.

### 7. Outer loss clipping (our contribution)

After each outer optimisation epoch, `face_params` (area targets) are clipped to ±50% around the current mean target:

```python
mesh_opt.faces_params = jnp.clip(mesh_opt.faces_params, target * 0.5, target * 1.5)
```

This prevents the bilevel outer optimiser from pushing individual area targets to degenerate values that would collapse cells.

## Pipeline

```
find_spindle          →  spindle axis: Newton-Raphson | PDF mode | PDF sample
divide_cell           →  JAX-compiled topology surgery (bounded_division port)
inner_optimization    →  mechanical relaxation (Adam warmup-cosine, peak lr 8e-4)
bilevel_optimization  →  area homogenisation (outer loss on he_params, face_params)
```

The `spindle_method` parameter of `run_n_divisions` selects the axis-finding strategy for all divisions in a run.

## Files

| File | Role |
|------|------|
| `spindle_model.py` | Microtubule force model, Newton-Raphson solver, PDF orientation density |
| `cell_division.py` | Half-edge topology surgery for cell division |
| `energy_adaptive.py` | Adaptive inner energy + outer homogenisation loss |
| `mesh_utils.py` | Polygon geometry utilities (centroid, inertia axis) |
| `spindle_analysis.py` | Main pipeline: n successive divisions with relaxation |
| `visualization.py` | Plotting utilities |
| `tutorial.ipynb` | Step-by-step notebook |
