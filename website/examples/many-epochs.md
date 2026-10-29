# Many epochs

[`do_n_bilevel_optimization`][vertax.bilevelopt.bilevelopt._BilevelOptimizer.do_n_bilevel_optimization]
runs a number of epochs and records them: costs, times, custom metrics, plots and meshes. This
example follows `tests/test_n_bilevel_opt.py`, a tension-inference problem on 20 periodic cells, with
the line-tension energy built by `make_energy_line_tensions` (the test writes the same kind of energy
by hand).

## The problem

```python
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pandas

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import make_energy_line_tensions

n_cells = 20
width = height = math.sqrt(n_cells)
e2 = make_energy_line_tensions(20.0)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)


def tensions(seed, n_edges):
    return jnp.repeat(1.2 + 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_edges,)), 2)


mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = tensions(1, mesh.nb_edges)
mesh.faces_params = jnp.ones(n_cells)

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.min_dist_T1 = 0.005
optimizer.max_nb_iterations = 1000
optimizer.tolerance = 1e-4
optimizer.patience = 5
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.beta = 0.01
optimizer.inner_optimization(mesh)

target = PbcMesh.copy_mesh(mesh)
target.edges_params = tensions(2, mesh.nb_edges)
optimizer.inner_optimization(target)
optimizer.vertices_target = target.vertices
optimizer.edges_target = target.edges
optimizer.faces_target = target.faces
optimizer.loss_function_outer = cost_v2v
```

## A custom metric

A custom metric is a function of the mesh and the optimizer that returns a float. It is computed at
every reported epoch and becomes a column of the summary and a plot of its own.

```python
def tension_correlation(mesh, optimizer):
    return float(jnp.corrcoef(mesh.edges_params[::2], target.edges_params[::2])[0, 1])


optimizer.add_custom_metric("Tension correlation", tension_correlation)
print(optimizer.self_summary())
```

## Running and reading the records

```python
optimizer.do_n_bilevel_optimization(
    nb_epochs=6,
    mesh=mesh,
    report_every=1,
    also_report_to_stdout=True,
    save_plotmesh_every=3,
    save_mesh_data_every=3,
    save_folder="n_bilevel",
)

folder = Path("n_bilevel")
print(sorted(str(p.relative_to(folder)) for p in folder.rglob("*")))
summary = pandas.read_csv(folder / "summary.csv", index_col=0)
print(summary.round(5).to_string())

saved = PbcMesh.load_mesh(str(folder / "meshes_data" / "mesh_epoch_6.npz"))
print(bool(jnp.allclose(saved.edges_params, mesh.edges_params)))
```

The folder holds `hyper-parameters.txt` (the output of `self_summary`), `summary.csv`, the plots
`outer_cost_over_time.png`, `inner_cost_over_time.png` and one per custom metric, and the meshes in
`meshes_plot/` and `meshes_data/`, at epoch 0 and every `save_*_every` epochs. `pre_inner_optimization=True`
would relax the mesh once before the first epoch.
