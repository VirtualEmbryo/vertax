# Bounded elongation

An inverse design on a bounded cluster: the tensions are changed so that the cluster elongates. It
follows the notebook `examples/inverse_modelling_example_bounded.ipynb`, with the settings of the
regression test `tests/test_inverse_modeling_bounded.py`.

## The cluster and its relaxation

The energy is the bounded line-tension energy with `K_areas = 40` and tensions passed through a
sigmoid, so that each tension stays between 1 and 2 whatever the outer steps do to `edges_params`.

```python
import math

import jax.numpy as jnp
import numpy as np
import optax

from vertax import BilevelOptimizationMethod, BoundedBilevelOptimizer, BoundedMesh, EdgePlot, plot_mesh
from vertax.cost import cost_ratio
from vertax.energy import make_energy_line_tensions_bounded

n_cells = 20
side = math.sqrt(n_cells)
mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=1)
rng = np.random.default_rng(seed=1)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.asarray(rng.random(mesh.nb_edges) * 20 - 10)
mesh.faces_params = jnp.full(mesh.nb_faces, 0.6)

optimizer = BoundedBilevelOptimizer()
optimizer.loss_function_inner = make_energy_line_tensions_bounded(40.0, sigmoid_tensions=True)
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.min_dist_T1 = 0.025
optimizer.max_nb_iterations = 1000
optimizer.tolerance = 1e-6
optimizer.patience = 5
optimizer.inner_optimization(mesh)

fig, ax = plot_mesh(mesh, edge_plot=EdgePlot.EDGE_PARAMETER, title="Before")
fig.savefig("elongation_before.png")
```

## The design

[`cost_ratio`][vertax.cost.cost_ratio] is $(r - 2)^2$, where $r$ is the longest distance between two
vertices divided by the width of the cluster across that direction. It uses the vertex positions
only. There is no target mesh, so `vertices_target` stays empty.

```python
optimizer.loss_function_outer = cost_ratio
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.beta = 0.01
optimizer.outer_solver = optax.adam(learning_rate=0.01, nesterov=True)

print(f"epoch 0: cost_ratio {float(cost_ratio(mesh.vertices)):.4f}")
for epoch in range(1, 6):
    optimizer.bilevel_optimization(mesh)
    print(f"epoch {epoch}: cost_ratio {float(cost_ratio(mesh.vertices)):.4f}")

fig, ax = plot_mesh(mesh, edge_plot=EdgePlot.EDGE_PARAMETER, title="After 5 epochs")
fig.savefig("elongation_after.png")
```

`cost_ratio(mesh.vertices)` can be called with the vertices alone, because its other eleven
arguments have defaults. The notebook runs ten epochs with a smaller learning rate, `1e-4`; the test
runs three with `0.01`.
