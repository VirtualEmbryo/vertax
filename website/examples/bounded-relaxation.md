# Bounded relaxation

A forward simulation of a cluster of 20 cells with a curved outline, with the settings of the
regression test `tests/test_forward_modeling_bounded.py`: the line-tension energy with
`K_areas = 40`, tensions passed through a sigmoid, and a target area of 0.6 for every cell.

## The cluster

`BoundedMesh.from_random_seeds` keeps exactly `nb_seeds` cells. The outline is made of circular arcs
whose half-angles are degrees of freedom, relaxed together with the vertices.

```python
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from vertax import BoundedBilevelOptimizer, BoundedMesh, EdgePlot, FacePlot, plot_mesh
from vertax.energy import make_energy_line_tensions_bounded

n_cells = 20
side = math.sqrt(n_cells)
mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=1)
print(f"{mesh.nb_faces} cells, {mesh.nb_edges} edges, {mesh.nb_vertices} vertices")
```

## The energy

With `sigmoid_tensions=True`, the tension of edge $e$ is $1 + \operatorname{sigmoid}(\lambda_e)$,
where $\lambda_e$ is read from `edges_params`, so every tension lies between 1 and 2. There is one
value per edge.

```python
rng = np.random.default_rng(seed=1)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.asarray(rng.random(mesh.nb_edges) * 20 - 10)
mesh.faces_params = jnp.full(mesh.nb_faces, 0.6)

energy = make_energy_line_tensions_bounded(40.0, sigmoid_tensions=True)


def energy_and_residual(mesh):
    """The energy and the norm of its gradient with respect to the vertex positions."""
    def f(vertices):
        return energy(vertices, mesh.angles, mesh.edges, mesh.faces, None, None, None,
                      mesh.vertices_params, mesh.edges_params, mesh.faces_params)
    return float(f(mesh.vertices)), float(jnp.linalg.norm(jax.grad(f)(mesh.vertices)))


print("before: energy %.4f, residual %.4f" % energy_and_residual(mesh))
```

## Relaxation

```python
optimizer = BoundedBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.min_dist_T1 = 0.025
optimizer.max_nb_iterations = 1000
optimizer.tolerance = 1e-6
optimizer.patience = 5

history = optimizer.inner_optimization(mesh)
print(f"after {len(history)} iterations: energy %.4f, residual %.4f" % energy_and_residual(mesh))

areas = mesh.get_area(jnp.arange(mesh.nb_faces))
print(f"cell areas: min {float(areas.min()):.3f}, mean {float(areas.mean()):.3f}, max {float(areas.max()):.3f}")
print(f"arc half-angles: min {float(mesh.angles.min()):.3f}, max {float(mesh.angles.max()):.3f}")

fig, ax = plot_mesh(mesh, face_plot=FacePlot.FATES, edge_plot=EdgePlot.EDGE_PARAMETER,
                    edge_parameters_name="edges_params", title="Relaxed cluster")
fig.savefig("bounded_relaxation.png")
```

`FacePlot.FATES` colors the cells by their fate marker, the second column of `mesh.faces`. The cell
areas are signed: a cell with a non-positive area has folded.
