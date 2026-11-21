# Tension inference

This example follows the notebook `examples/inverse_modelling_example.ipynb`. A periodic tissue with
random line tensions is relaxed; a target is made by relaxing the same tissue with other random
tensions; then the tensions of the first tissue are changed, by equilibrium propagation, until its
equilibrium vertices match the target's. Everything is in `float64`.

## The tissue, the energy and the optimizer

The energy is the line-tension energy $E_2$ with `K_areas = 10` and a target area of 1 for every
cell. Tensions are stored per half-edge, with the same value on the two half-edges of an edge.

```python
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import make_energy_line_tensions

n_cells = 20
width = height = math.sqrt(n_cells)
K_AREAS = 10.0
e2 = make_energy_line_tensions(K_AREAS)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)


def random_tensions(seed, n_edges, mean=1.2, std=0.1):
    """One tension per edge, repeated on its two half-edges."""
    return jnp.repeat(mean + std * jax.random.normal(jax.random.PRNGKey(seed), (n_edges,)), 2)


mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = random_tensions(1, mesh.nb_edges)
mesh.faces_params = jnp.ones(mesh.nb_faces)

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.update_T1 = True
optimizer.min_dist_T1 = 0.005
optimizer.max_nb_iterations = 1000
optimizer.tolerance = 1e-4
optimizer.patience = 5

optimizer.inner_optimization(mesh)
```

## The target

```python
target = PbcMesh.copy_mesh(mesh)
target.edges_params = random_tensions(2, mesh.nb_edges)
optimizer.inner_optimization(target)

optimizer.vertices_target = target.vertices
optimizer.edges_target = target.edges
optimizer.faces_target = target.faces
```

## The outer problem

The cost is the vertex-to-vertex distance. Adding a constant to every tension hardly changes the
equilibrium, so the mean tension is fixed after every outer step by `mean_tension_gauge`. Its value
here is the mean of the target tensions, which is known only because the target was made by the
script; the notebook fixes it at 1.0.

```python
optimizer.loss_function_outer = cost_v2v
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.beta = 0.01
optimizer.outer_solver = optax.adam(learning_rate=2e-3, nesterov=True)
optimizer.mean_tension_gauge = float(target.edges_params.mean())


def report(epoch):
    estimate, truth = mesh.edges_params[::2], target.edges_params[::2]  # one value per edge
    pearson = float(jnp.corrcoef(estimate, truth)[0, 1])
    centered_error = jnp.linalg.norm((estimate - estimate.mean()) - (truth - truth.mean()))
    spread = jnp.linalg.norm(truth - truth.mean())
    print(f"epoch {epoch:2d}: cost {optimizer.compute_outer_loss(mesh):.3e}, "
          f"Pearson {pearson:+.3f}, error/spread {float(centered_error / spread):.3f}")


report(0)
for epoch in range(1, 21):
    optimizer.bilevel_optimization(mesh)
    if epoch % 5 == 0:
        report(epoch)
```

`error/spread` is the recovery error $\varepsilon$ of the paper: the norm of the difference between
the recovered and the true tensions, both centered, divided by the spread of the true tensions. The
practical guide `docs/tension_inference_guide.md` of the repository says what to report with it.
Twenty epochs are far from enough to recover the tensions: the runs of the paper use hundreds.

<figure markdown="span">
  ![Left, the cost decreasing over 20 epochs; right, recovered against target tensions at epoch 0 and epoch 20, scattered far from the diagonal](../figures/example-tension-inference.svg)
  <figcaption markdown="span">The script of this page: over 20 epochs the cost falls from $3.2 \times 10^{-3}$ to $1.0 \times 10^{-3}$ (left), while the tensions stay far from the target ones (right; Pearson 0.04, then 0.34; error/spread 1.33, then 1.10).</figcaption>
</figure>

## Other costs for tension inference

This example uses the vertex-to-vertex cost $\mathcal C_{\text{v2v}}$. The paper also uses the
Hessian-weighted cost $\mathcal C_W$, which compensates for the amplification of the geometric error by
the soft modes of the tissue, and a staged schedule for a starting mesh whose topology differs from
the target's ($d_{\mathrm{T1}} > 0$). Both are described, with a runnable example, in
[Costs](../user-guide/costs.md#hessian-weighted-vertex-cost-and-the-staged-schedule).
