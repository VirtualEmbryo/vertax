# T1 transitions

A T1 transition is the elementary change of neighbors in a tissue: an edge between two cells shrinks
to a point, and a new edge appears between the two other cells that met at its ends. VertAX performs
T1 transitions during relaxations, between two iterations of the solver.

## When an edge flips

At every iteration, after the solver step, the periodic T1 update
[`vertax.topo.update_T1`][vertax.topo.update_T1] looks at every edge. Each edge no longer than
`min_dist_T1`, and whose two cells are not triangles, is flipped. The new edge is centered on the
midpoint of the old one, and is given the length `(1 + reopen_margin) * min_dist_T1`, that is
`1.1 * min_dist_T1` with the default `reopen_margin=0.1`.

On an optimizer, two attributes control this:

- `update_T1` (default `True`): perform T1 transitions at all. With `False`, the topology never
  changes.
- `min_dist_T1` (default `0.005`): the length threshold. It is in the units of the mesh; a mesh whose
  cells have area 1 has edges of length about 0.6.

## The energy check

With `perform_energy_check=True`, a flip is kept only if its energy does not exceed that of the
unflipped configuration in which the short edge is stretched, along its own direction, to the same
length as the new edge. Otherwise the stretched, unflipped configuration is kept. Without the check,
every flip is kept.

`inner_optimization` checks the energy by default (`perform_energy_check=True`; its docstring says it
should be true for forward simulations). `bilevel_optimization` and `do_n_bilevel_optimization` do
not: they pass `perform_energy_check=False` to the relaxations they run.

## Other options

These are arguments of `minimize` and of `update_T1`, not attributes of the optimizers:

- `reopen_margin`: the length of the new edge, as above. `minimize` calls the T1 function with
  fixed positional arguments, so pass a different margin with `functools.partial`:
  `minimize(..., update_t1_func=functools.partial(update_T1, reopen_margin=0.2))`.
- `count_t1=True` makes `minimize` also return, for each iteration, the number of edges presented to
  the T1 update (no longer than `min_dist_T1`, neither cell a triangle) and the number of flips kept.
  The counts are carried in the state of the loop, so they are exact under `jax.jit`, `jax.vmap` and
  `jax.grad`. The number kept is exact only when the flips of one iteration share no cell.
- [`vertax.opt.recording_t1_statistics`][vertax.opt.recording_t1_statistics] is a context manager
  that collects these counts from every periodic relaxation run inside it, including the relaxations
  run by the automatic-differentiation and equilibrium-propagation estimators. Each record is a
  dictionary with the counts and the name of the function that ran the relaxation.

```python
import math

import jax.numpy as jnp
import numpy as np
import optax

from vertax import PbcMesh
from vertax.energy import make_energy_line_tensions
from vertax.opt import minimize, recording_t1_statistics

WIDTH = HEIGHT = 4.0
e2 = make_energy_line_tensions(10.0)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return e2(vertTable, heTable, faceTable, WIDTH, HEIGHT, he_params, face_params)


mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
tensions = jnp.asarray(1.0 + 0.6 * np.random.default_rng(0).random(mesh.nb_half_edges))
solver = optax.sgd(learning_rate=0.01)

for check in (False, True):
    _, (history, steps, counts) = minimize(
        mesh.vertices, mesh.edges, mesh.faces, WIDTH, HEIGHT,
        jnp.zeros(mesh.nb_vertices), tensions, jnp.ones(mesh.nb_faces), energy, solver, 0.05,
        iterations_max=400, tolerance=0.0, perform_energy_check=check, count_t1=True)
    presented, kept = np.asarray(counts).sum(axis=0)
    print(f"energy check {check}: {presented} edges presented, {kept} flips kept")

with recording_t1_statistics() as records:
    minimize(mesh.vertices, mesh.edges, mesh.faces, WIDTH, HEIGHT,
             jnp.zeros(mesh.nb_vertices), tensions, jnp.ones(mesh.nb_faces), energy, solver, 0.05,
             iterations_max=400, tolerance=0.0)
print([(r["source"], r["counts"].shape) for r in records])
```

The tensions are random, so the relaxation shrinks some edges below the threshold. This tissue is the
one of the test of `count_t1`; it has fewer than 20 cells, so its construction prints a warning.

## Bounded meshes

For a bounded mesh, [`update_T1_bounded`][vertax.topo.update_T1_bounded] distinguishes edges inside
the cluster from edges that touch its outline. A T1 transition can bring a cell to the outline or
take it inside. The arc angles are kept between `ARC_HALF_ANGLE_MIN = 0.017` and
`ARC_HALF_ANGLE_MAX = pi/2 - 0.001`. `reopen_margin`, `count_t1` and `recording_t1_statistics` are
periodic only.

## Vertex labels and targets

A vertex-to-vertex cost compares each vertex with the target vertex of the same index. When an
optimizer has a vertex target (`vertices_target`) and `update_T1` is on, `inner_optimization` of a
periodic optimizer, after the relaxation, looks at the edges that underwent a T1 transition and swaps
the indices of their two vertices when the swap lowers the squared distance of the pair to the
target, measured through the periodic boundary.

This repair is not implemented for bounded meshes: `inner_optimization` of a
`BoundedBilevelOptimizer` raises `NotImplementedError` when `vertices_target` is set and `update_T1`
is on. Bounded inverse problems in VertAX are design problems without a vertex target.

## Large solver steps

The T1 update looks at edge lengths after each step, so it expects vertices to move by small amounts.
A line-search solver can move a vertex across whole cells in one step. The step cap described in
[Relaxation and solvers](relaxation.md#the-solvers) keeps the step of each vertex below
`5 * min_dist_T1`.
