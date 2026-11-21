# Line-search solvers

Two experiments from the tests of the line-search solvers. The first, from
`tests/test_cap_vertex_step.py`, shows why the step of each vertex is capped when the inner solver
chooses its own step length. The second, from `tests/test_minimize_lbfgs_t1_reset.py`, relaxes a
tissue with L-BFGS and T1 transitions to a force residual below $10^{-10}$.

## One uncapped step can fold cells

A bounded cluster is relaxed for a single iteration of `gd_linesearch`, steepest descent with a
line search, once with `cap_vertex_step = False` and once with the default, `None`, which caps
line-search solvers. The largest vertex displacement is measured in units of `min_dist_T1`, and the
cells whose signed area is not positive afterwards are folded.

```python
import math

import jax.numpy as jnp
import numpy as np

from vertax import BilevelOptimizationMethod, BoundedBilevelOptimizer, BoundedMesh
from vertax.cg import gd_linesearch
from vertax.energy import make_energy_line_tensions_bounded

MIN_DIST_T1 = 0.01
energy = make_energy_line_tensions_bounded(10.0)
solver = gd_linesearch()


def first_step(cap, seed):
    n_cells = 20
    side = math.sqrt(n_cells)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=seed)
    rng = np.random.default_rng(seed)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.asarray(1.0 + 0.5 * rng.random(mesh.nb_edges))
    mesh.faces_params = jnp.full(mesh.nb_faces, 1.5)

    optimizer = BoundedBilevelOptimizer()
    optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    optimizer.inner_solver = solver
    optimizer.loss_function_inner = energy
    optimizer.min_dist_T1 = MIN_DIST_T1
    optimizer.max_nb_iterations = 1
    optimizer.tolerance = 0.0
    optimizer.cap_vertex_step = cap

    before = np.asarray(mesh.vertices).copy()
    optimizer.inner_optimization(mesh, perform_energy_check=False)
    step = np.abs(np.asarray(mesh.vertices) - before).max() / MIN_DIST_T1
    folded = int((mesh.get_area(jnp.arange(mesh.nb_faces)) <= 0).sum())
    return step, folded


for seed in (1, 5):
    for cap in (False, None):
        step, folded = first_step(cap, seed)
        print(f"seed {seed}, cap_vertex_step={cap}: largest step {step:6.1f} x min_dist_T1, {folded} folded cell(s)")
```

In the run that checked this page, the uncapped step moved a vertex by 70 and 74 times
`min_dist_T1` for the two seeds, and folded one cell for seed 5; with the cap, the largest step was
5 times `min_dist_T1` and no cell folded. The test counts a cell as inverted when the sign of its
vertex polygon changes, a broader criterion that also catches seed 1. The cap is `5 * min_dist_T1`
per vertex and per iteration. The optimizer uses equilibrium
propagation here only because automatic differentiation refuses a line-search inner solver; this
experiment runs no outer step.

<figure markdown="span">
  ![The cluster of seed 5 after one step of gd_linesearch, without the cap, with long vertex steps and a folded cell circled, and with the cap, with short steps](../figures/example-vertex-step-cap.svg)
  <figcaption markdown="span">One iteration of `gd_linesearch` on the cluster of seed 5: without the cap one vertex moves by 74 × `min_dist_T1` and one cell folds (circled); with it, no step exceeds 5 × `min_dist_T1`; dashed: the cluster before the step.</figcaption>
</figure>

## L-BFGS with T1 transitions

`optax.lbfgs()` keeps curvature information from previous iterations. After a T1 transition this
information describes another topology, and `reset_solver_on_t1=True` resets it. This relaxation
calls `minimize` directly, in `float64`, with the energy check of the T1 transitions on and the step
cap on.

```python
import jax
import optax
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.energy import energy_shape_factor_homo
from vertax.opt import minimize
from vertax.rheology import t1_event_count, topology_signature

WIDTH = HEIGHT = 4.0


def shape_energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, WIDTH, HEIGHT, face_params)


with enable_x64():
    tissue = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
    vp, hp, fp = jnp.zeros(tissue.nb_vertices), jnp.ones(tissue.nb_half_edges), jnp.full(tissue.nb_faces, 3.7)
    (v, he, fa, *_), (history, steps) = minimize(
        tissue.vertices, tissue.edges, tissue.faces, WIDTH, HEIGHT, vp, hp, fp, shape_energy, optax.lbfgs(), 0.05,
        iterations_max=2000, tolerance=0.0, perform_energy_check=True, cap_vertex_step=True,
        reset_solver_on_t1=True)
    residual = jnp.linalg.norm(jax.grad(shape_energy)(v, he, fa, vp, hp, fp))
    events = t1_event_count(topology_signature(tissue.edges), topology_signature(he))
    print(f"{int(steps)} iterations, {events} T1 event(s), force residual {float(residual):.1e}")
```

The test asserts at least one T1 event and a residual below $10^{-10}$. This tissue has 16 cells, so
its construction prints a warning about small cell numbers.
