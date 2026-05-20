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

### 2. Arc intersection detection

Both use the same **exact algebraic** approach (port of `circle_inter` from `bounded_division`):
- Reconstruct circle from chord and inscribed angle
- Solve line-circle intersection algebraically (quadratic)
- Filter candidate points via dot-product angular constraints (no arccos)

Our `_arc_line_intersect` is a NumPy port of `bounded_division`'s JAX `circle_inter`.

### 3. Quasi-coincident vertex detection (our contribution)

`bounded_division` has no mechanism to detect when the division line passes close to an existing vertex.

We introduce `VERTEX_TOL = 0.03` in `_detect_cuts`: if the intersection parameter t is within VERTEX_TOL of 0 or 1, the existing vertex is reused instead of creating a new one. This reduces interstitial artefacts after relaxation.

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

## Pipeline

```
find_spindle          →  spindle axis: Newton-Raphson | PDF mode | PDF sample
divide_cell           →  half-edge topology surgery (NumPy)
inner_optimization    →  mechanical relaxation (Adam with cosine decay, lr ≤ 5e-4)
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
