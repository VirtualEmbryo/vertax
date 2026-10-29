# Periodic relaxation

A forward simulation: 100 cells in a periodic box relax under the shape-factor energy with T1
transitions. The script then checks the result the way the regression test
`tests/test_forward_modeling.py` does: not by comparing vertex coordinates, which change with the
precision and the machine, but by the energy, the force residual and the topology.

## The tissue and the energy

The box has area 100, so the mean cell area is 1. Every cell has the preferred perimeter
$p_0 = 3.7$. The threshold `min_dist_T1 = 0.2` is the one of the test.

```python
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from vertax import FacePlot, PbcBilevelOptimizer, PbcMesh, plot_mesh
from vertax.energy import energy_shape_factor_homo

n_cells = 100
width = height = math.sqrt(n_cells)
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(mesh.nb_faces, 3.7)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)


optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.min_dist_T1 = 0.2
optimizer.max_nb_iterations = 1000
optimizer.tolerance = 1e-4
optimizer.patience = 5
```

## Measures that do not depend on coordinates

```python
def energy_and_residual(mesh):
    """The energy and the norm of its gradient with respect to the vertex positions."""
    args = (mesh.edges, mesh.faces, mesh.vertices_params, mesh.edges_params, mesh.faces_params)
    value = energy(mesh.vertices, *args)
    gradient = jax.grad(energy)(mesh.vertices, *args)
    return float(value), float(jnp.linalg.norm(gradient))


def edges_per_face(mesh):
    """Sorted numbers of edges of the faces: a summary of the topology that ignores labels."""
    face_of_half_edge = np.asarray(mesh.edges)[:, 5]
    return sorted(int((face_of_half_edge == f).sum()) for f in range(mesh.nb_faces))


initial_topology = edges_per_face(mesh)
print("before: energy %.4f, residual %.4f" % energy_and_residual(mesh))
```

## Relaxation

```python
history = optimizer.inner_optimization(mesh)
e, r = energy_and_residual(mesh)
print(f"after {len(history)} iterations: energy {e:.4f}, residual {r:.4f}, residual/energy {r / e:.3f}")
print("cells with 4, 5, 6, 7, 8 edges, before:", [initial_topology.count(k) for k in range(4, 9)])
print("cells with 4, 5, 6, 7, 8 edges, after: ", [edges_per_face(mesh).count(k) for k in range(4, 9)])

fig, ax = plot_mesh(mesh, face_plot=FacePlot.AREA, title="Areas after relaxation")
fig.savefig("periodic_relaxation.png")
```

The relaxation stops on its tolerance and patience rule, not on a small force residual. The test
records that its reference stops at `residual / energy = 0.246`, and checks that a new run stops no
worse than that, with the same histogram of edges per face and the same energy within 0.1 %.
