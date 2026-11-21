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

<figure markdown="span">
  ![Two rows of three panels: four cells whose shared edge shrinks to the threshold and flips, inside the tissue and at the outline of a bounded mesh, where an inner cell reaches the outline](../figures/t1-transition.svg)
  <figcaption markdown="span">A T1 transition inside the tissue (top) and at the outline of a bounded mesh (bottom): the purple edge shrinks to `min_dist_T1` and is replaced by an edge between the two other cells; read from right to left, the bottom row is the transition that takes a cell inside (schematic).</figcaption>
</figure>

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

- `reopen_margin`: the length of the new edge, as above (for bounded meshes, see below).
  `minimize` calls the T1 function with fixed positional arguments, so pass a different margin with
  `functools.partial`:
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
`ARC_HALF_ANGLE_MAX = pi/2 - 0.001`. `count_t1` and `recording_t1_statistics` are periodic only.

### The length of the reopened edge

`update_T1_bounded` has a keyword `reopen_margin`, with the same meaning as for the periodic
`update_T1`: the new edge, and the edge that is re-stretched when the flip is refused, is given the
length `(1 + reopen_margin) * min_dist_T1` about the midpoint of the old edge. Its default, `None`,
keeps the length `min_dist_T1 + 1e-3` that bounded results have always used. An edge only `1e-3`
above the threshold can be shortened below it again by a single relaxation step, and then flips back.
`BoundedBilevelOptimizer` does not expose the margin; pass it to the relaxation function with
`functools.partial`, as in the example below.

### The tension of the new edge

In a bounded mesh an edge keeps its index, and so its parameter, through a T1 transition: the new
edge inherits the line tension of the edge it replaces. When the outer optimization has raised that
tension enough for the old edge to collapse, the new edge collapses too, and the pair can flip back
and forth.

[`T1TensionReset`][vertax.topo.T1TensionReset] changes this rule, and is off by default. With it, the
new edge starts at the mean tension of the other edges of its own kind (interior edges for a new
interior edge, free-interface edges for a new free-interface edge). Physically, a new junction
carries none of the tension that was built up on the junction it replaces. The mean is taken over the
tensions rather than over their parameters: if the energy maps the parameters through a sigmoid
(`make_energy_line_tensions_bounded(..., sigmoid_tensions=True)`), create `T1TensionReset(sigmoid_tensions=True)`.
The reset is a smooth function of the parameters of the other edges, so automatic differentiation of
a relaxation differentiates through it.

Set it on the optimizer as `t1_tension_reset`, and it applies in every relaxation, the inner
optimization and those of the estimators:

- `inner_optimization` stores the post-reset edge parameters in the mesh;
- the outer step applies its update to the post-reset parameters of the estimator's relaxation, so
  that a junction created by a T1 transition keeps its reset tension in the design;
- `optimizer.last_t1_resets` is a [`T1ResetReport`][vertax.opt_bounded.T1ResetReport]: the
  post-reset parameters, the T1 transitions accepted on each edge in each relaxation (`flips`, one
  array per relaxation, two for EP), and, for EP, the number of edges on which its two solves
  disagree (`nudged_mismatch`). It is `None` without the option.

The four `outer_*_bounded` functions of [`vertax.opt_bounded`](../api/opt-bounded.md) take
`tension_reset` too, and then return the `T1ResetReport` as an eighth value; `inner_opt_bounded`
returns a [`T1Resets`][vertax.opt_bounded.T1Resets] as a third value. The reset applies to a
relaxation of the vertices and arcs only.

```python
import functools
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from vertax import BoundedBilevelOptimizer, BoundedMesh, T1TensionReset
from vertax.energy import make_energy_line_tensions_bounded
from vertax.opt_bounded import inner_opt_bounded
from vertax.topo import update_T1_bounded

side = math.sqrt(12)
cluster = BoundedMesh.from_random_seeds(nb_seeds=12, width=side, height=side, random_key=1)
cluster.vertices_params = jnp.zeros(1)
cluster.edges_params = jnp.ones(cluster.nb_edges)
cluster.faces_params = jnp.full(cluster.nb_faces, 0.6)
energy = make_energy_line_tensions_bounded(40.0)

optimizer = BoundedBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_solver = optax.sgd(learning_rate=2e-3)
optimizer.min_dist_T1 = 0.025
optimizer.max_nb_iterations = 6000
optimizer.tolerance = 0.0
optimizer.inner_optimization(cluster)

# Edge 4 gets a tension that breaks the triangle inequality at its vertices: it collapses.
cluster.edges_params = cluster.edges_params.at[4].set(3.0)
optimizer.max_nb_iterations = 300

for reset in (None, T1TensionReset()):
    trial = BoundedMesh.copy_mesh(cluster)
    optimizer.t1_tension_reset = reset
    optimizer.inner_optimization(trial)
    print(f"t1_tension_reset={reset}: tension of edge 4 after the relaxation {float(trial.edges_params[4]):.2f}")
    if reset is not None:
        flips = np.asarray(optimizer.last_t1_resets.flips[0])
        print("T1 transitions per edge:", {int(e): int(flips[e]) for e in np.nonzero(flips)[0]})

# The length of the reopened edge is a keyword of the T1 function: pass it with functools.partial.
margin = functools.partial(update_T1_bounded, reopen_margin=0.2)
(vt, at, ht, ft), history = inner_opt_bounded(
    cluster.vertices, cluster.angles, cluster.edges, cluster.faces,
    cluster.vertices_params, cluster.edges_params, cluster.faces_params,
    energy, optax.sgd(2e-3), 0.025, 300, 0.0, 5, update_T1_func=margin)
print("with reopen_margin=0.2: energy after 300 iterations", float(history[-1]))
```

The cluster has 12 cells and edge 4 is an interior edge. Without the option, the new edge keeps the
tension of 3 of the edge it replaces. With it, one T1 transition is accepted on edge 4 and the new
edge starts at the mean tension of the other interior edges, here 1.

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
