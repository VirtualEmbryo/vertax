# T1 statistics

This example counts the T1 transitions of a relaxation, and those that the energy check rejects. It
follows `tests/test_t1_counts_in_loop_state.py`: a 16-cell periodic tissue with random line
tensions, a T1 threshold of 0.05, and relaxations of a fixed number of iterations, in `float64`.

## The tissue

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import make_energy_line_tensions
from vertax.opt import minimize, recording_t1_statistics

WIDTH = HEIGHT = 4.0
MIN_DIST = 0.05
e2 = make_energy_line_tensions(10.0)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return e2(vertTable, heTable, faceTable, WIDTH, HEIGHT, he_params, face_params)


tissue = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
rng = np.random.default_rng(0)
tensions = jnp.asarray(1.0 + 0.6 * rng.random(tissue.nb_half_edges))
solver = optax.sgd(learning_rate=0.01)


def relax(he_params, check, n=400):
    return minimize(tissue.vertices, tissue.edges, tissue.faces, WIDTH, HEIGHT,
                    jnp.zeros(tissue.nb_vertices), he_params, jnp.ones(tissue.nb_faces), energy, solver, MIN_DIST,
                    iterations_max=n, tolerance=0.0, perform_energy_check=check, count_t1=True)
```

## Counts per iteration

With `count_t1=True`, `minimize` returns a third history: for each iteration, the number of edges
presented to the T1 update and the number of flips kept.

```python
for check in (False, True):
    _, (history, steps, counts) = relax(tensions, check)
    counts = np.asarray(counts)
    active = np.flatnonzero(counts[:, 0])
    print(f"energy check {check}: {counts[:, 0].sum()} presented, {counts[:, 1].sum()} kept, "
          f"at iterations {active.tolist()}")
```

Without the energy check, every presented edge is flipped. With it, a flip is kept only when it does
not raise the energy.

<figure markdown="span">
  ![Two event plots against iteration: the edges presented to the T1 update and the flips kept, without and with the energy check](../figures/example-t1-counts.svg)
  <figcaption markdown="span">The edges presented to the T1 update (grey bars) and the flips kept (dots) at each iteration of the 400-iteration relaxation of this page: 8 of 8 kept without the energy check, 8 of 12 with it.</figcaption>
</figure>

## Under `jax.vmap`

The counts are part of the state of the loop, so they stay exact when the relaxation is vectorized,
here over two sets of tensions:

```python
tension_sets = jnp.stack([tensions, jnp.asarray(1.0 + 0.3 * rng.random(tissue.nb_half_edges))])
batched = jax.vmap(lambda hp: relax(hp, False, n=150)[1][2])(tension_sets)
one_by_one = [relax(hp, False, n=150)[1][2] for hp in tension_sets]
for b, s in zip(batched, one_by_one):
    print("kept:", int(b[:, 1].sum()), int(s[:, 1].sum()), "identical:", bool(jnp.all(b == s)))
```

## In the gradient estimators

[`recording_t1_statistics`][vertax.opt.recording_t1_statistics] collects the counts of every periodic
relaxation run inside it, including the relaxation that automatic differentiation unrolls and the two
solves of equilibrium propagation. Each record has the name of the function that ran it.

```python
def one_epoch(method):
    mesh = PbcMesh.copy_mesh(tissue)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = tensions
    mesh.faces_params = jnp.ones(mesh.nb_faces)
    optimizer = PbcBilevelOptimizer()
    optimizer.loss_function_inner = energy
    optimizer.loss_function_outer = cost_v2v
    optimizer.inner_solver = solver
    optimizer.outer_solver = optax.sgd(learning_rate=1e-3)
    optimizer.bilevel_optimization_method = method
    optimizer.beta = 1e-3
    optimizer.min_dist_T1 = MIN_DIST
    optimizer.max_nb_iterations = 200
    optimizer.tolerance = 0.0
    optimizer.vertices_target = tissue.vertices
    optimizer.edges_target = tissue.edges
    optimizer.faces_target = tissue.faces
    with recording_t1_statistics() as records:
        optimizer.bilevel_optimization(mesh)
    for record in records:
        presented, kept = record["counts"].sum(axis=0)
        print(f"  {record['source']:26s} {len(record['counts']):4d} iterations, {presented} presented, {kept} kept")


for method in (BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION, BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION):
    print(method.name)
    one_epoch(method)
```

The record named `minimize` is the relaxation that `bilevel_optimization` runs after the outer step.
Relaxations traced inside your own `jax.jit`, `jax.vmap` or `jax.grad` cannot be recorded this way;
call `minimize(..., count_t1=True)` there and return the counts.
