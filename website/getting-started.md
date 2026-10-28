# Getting started

## Installation

VertAX is not on PyPI yet, so `pip install vertax` does not work. Install it from source, in a
virtual environment:

```sh
git clone https://github.com/VirtualEmbryo/vertax.git
cd vertax
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

VertAX needs Python 3.11 or later. Its dependencies are listed in `pyproject.toml`, most of them
with an exact version: JAX and jaxlib 0.4.34, Optax, JAXopt, NumPy, SciPy, Matplotlib, scikit-image,
tifffile, pandas, importlib-resources and Cellpose.

The repository also pins a complete environment for [uv](https://docs.astral.sh/uv/): `uv.lock`, with
Python 3.12 named in `.python-version`. With uv, the same steps are:

```sh
uv sync            # the library and the development tools
uv run pytest      # the test suite; long-running tests are deselected
```

For GPU support, install JAX with CUDA as described in the
[JAX documentation](https://github.com/google/jax#installation) before installing VertAX. VertAX
pins `jax==0.4.34` and `jaxlib==0.4.34`, so the CUDA build must be of that version.

To check the installation:

```python
from importlib.metadata import version

import vertax

print("VertAX", version("vertax"))
```

The three quick starts below take from a few seconds to about a minute each on one CPU thread, most
of it spent compiling. They use JAX's default precision, `float32`; see
[Numerical precision](user-guide/precision.md) before doing quantitative work.

## Quick start: relax a periodic tissue

A [`PbcMesh`][vertax.meshes.pbc_mesh.PbcMesh] is a tissue in a periodic box. Here it is built from
the Voronoi diagram of 20 random seeds, in a square box of area 20, so that the mean cell area is 1.

The energy is the shape-factor energy
$E_1 = \sum_\alpha (A_\alpha - 1)^2 + (P_\alpha - p_0)^2$, with the same preferred perimeter
$p_0 = 3.7$ for every cell. VertAX calls the energy with six arguments: the three mesh tables and the
three parameter arrays. The box size is not one of them, so it is closed over.

A bilevel optimizer holds the energy and the settings of the relaxation. Its
`inner_optimization` moves the vertices towards a minimum of the energy, with T1 transitions.

```python
import math

import jax.numpy as jnp
import optax

from vertax import PbcBilevelOptimizer, PbcMesh, plot_mesh
from vertax.energy import energy_shape_factor_homo

n_cells = 20
width = height = math.sqrt(n_cells)
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)

# One parameter per vertex, per half-edge and per face. This energy only reads the face parameters.
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(mesh.nb_faces, 3.7)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)


optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_solver = optax.sgd(learning_rate=0.01)  # the default, written out

print(f"{mesh.nb_faces} cells, {mesh.nb_edges} edges, {mesh.nb_vertices} vertices")
print("energy before:", optimizer.compute_inner_loss(mesh))
history = optimizer.inner_optimization(mesh)
print("energy after: ", optimizer.compute_inner_loss(mesh), f"({len(history)} iterations)")

fig, ax = plot_mesh(mesh, title="Relaxed periodic tissue")
fig.savefig("periodic_tissue.png")
```

`plot_mesh` returns the matplotlib figure and axes, and does not display the figure unless you pass
`show=True`.

## Quick start: relax a bounded cluster

A [`BoundedMesh`][vertax.meshes.bounded_mesh.BoundedMesh] is a finite cluster of cells. The edges on
its outline are circular arcs, and the angle of each arc is a degree of freedom relaxed together with
the vertices.

The energy here is the line-tension energy
$E_2 = \sum_e \Lambda_e \ell_e + \tfrac{K}{2} \sum_\alpha (A_\alpha - A^0_\alpha)^2$, built by
[`make_energy_line_tensions_bounded`][vertax.energy.make_energy_line_tensions_bounded]. For a bounded
mesh it takes one tension per edge (not per half-edge) and one target area per cell. The area
stiffness `K_areas` has no default.

```python
import math

import jax.numpy as jnp
import numpy as np

from vertax import BoundedBilevelOptimizer, BoundedMesh, plot_mesh
from vertax.energy import make_energy_line_tensions_bounded

n_cells = 20
width = height = math.sqrt(n_cells)
cluster = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)

rng = np.random.default_rng(seed=1)
cluster.vertices_params = jnp.zeros(cluster.nb_vertices)
cluster.edges_params = jnp.asarray(1.0 + rng.random(cluster.nb_edges))  # tensions between 1 and 2
cluster.faces_params = jnp.full(cluster.nb_faces, 0.6)  # target area of every cell

bounded_optimizer = BoundedBilevelOptimizer()
bounded_optimizer.loss_function_inner = make_energy_line_tensions_bounded(K_areas=40.0)
bounded_optimizer.min_dist_T1 = 0.025

print(f"{cluster.nb_faces} cells, {cluster.nb_edges} edges, {cluster.nb_vertices} vertices")
print("energy before:", bounded_optimizer.compute_inner_loss(cluster))
bounded_optimizer.inner_optimization(cluster)
print("energy after: ", bounded_optimizer.compute_inner_loss(cluster))
print("smallest cell area:", float(cluster.get_area(jnp.arange(cluster.nb_faces)).min()))

fig, ax = plot_mesh(cluster, title="Relaxed bounded cluster")
fig.savefig("bounded_cluster.png")
```

`nb_angles` equals `nb_edges`: the mesh stores one angle per edge, and only the angles of the
outline edges are used.

## Quick start: a small inverse problem

An inverse problem looks for parameters whose equilibrium matches a target. Here the target is made
by the code itself: the relaxed periodic tissue of the first quick start is copied, given other
preferred perimeters (one per cell, now), and relaxed again. The bilevel optimizer then changes the
preferred perimeters of the first tissue so that its equilibrium vertices move towards the target's.

The outer cost is [`cost_v2v`][vertax.cost.cost_v2v], the mean squared distance between each vertex
and the same vertex of the target. The gradient of the cost with respect to the parameters is
estimated by equilibrium propagation, which needs a nudging size `beta`; set it explicitly.

```python
import math

import jax
import jax.numpy as jnp
import optax

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import energy_shape_factor_hetero

n_cells = 20
width = height = math.sqrt(n_cells)
all_faces = jnp.arange(n_cells)

mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(mesh.nb_faces, 3.7)  # one preferred perimeter per cell


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_hetero(vertTable, heTable, faceTable, width, height, all_faces, face_params)


optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_optimization(mesh)

# The target: the same tissue with other preferred perimeters, relaxed.
target = PbcMesh.copy_mesh(mesh)
target.faces_params = 3.7 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n_cells,))
optimizer.inner_optimization(target)
optimizer.vertices_target = target.vertices
optimizer.edges_target = target.edges
optimizer.faces_target = target.faces

optimizer.loss_function_outer = cost_v2v
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.beta = 0.01
optimizer.outer_solver = optax.adam(learning_rate=1e-3)

print("cost before:", optimizer.compute_outer_loss(mesh))
for epoch in range(1, 6):
    optimizer.bilevel_optimization(mesh)  # one outer step, then a relaxation
    print(f"epoch {epoch}: cost {optimizer.compute_outer_loss(mesh):.3e}")

error = jnp.abs(mesh.faces_params - target.faces_params)
print("mean error on the preferred perimeters:", float(error.mean()))
```

Five epochs only move the parameters a little; an inverse problem usually needs hundreds. The
[tension inference example](examples/tension-inference.md) runs a longer one, and the
[inverse problems](user-guide/inverse-problems.md) page explains the estimators and their settings.

## Where to go next

- The [user guide](user-guide/meshes.md) explains the meshes, the energies, the relaxation and the
  inverse problems.
- The [examples](examples/index.md) are complete scripts, one per page.
- The [API reference](api/index.md) is generated from the docstrings.
