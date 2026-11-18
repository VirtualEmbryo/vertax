# Relaxation, minimize and the solvers

A relaxation moves the vertices of a mesh towards a minimum of the energy, with T1 transitions
between iterations. It is the inner problem of every inverse problem, and a forward simulation on its
own.

## Relaxing through an optimizer

[`inner_optimization`][vertax.bilevelopt.bilevelopt._BilevelOptimizer.inner_optimization] relaxes the
mesh in place and returns the history of the energy:

```text
history = optimizer.inner_optimization(mesh, perform_energy_check=True,
                                       only_on_vertices=None, only_on_edges=None, only_on_faces=None)
```

It replaces `mesh.vertices`, `mesh.edges` and `mesh.faces` (and `mesh.angles` for a bounded mesh); the
tables change when T1 transitions happen. For a bounded mesh, the solver moves the vertices and the
arc angles together.

These attributes of the optimizer control the relaxation:

| Attribute | Default | Meaning |
| --- | --- | --- |
| `loss_function_inner` | `None` | the energy, see [Energies](energies.md) |
| `inner_solver` | `optax.sgd(learning_rate=0.01)` | any Optax `GradientTransformation`, see below |
| `max_nb_iterations` | `1000` | maximum number of iterations |
| `tolerance` | `1e-4` | relative change of the energy below which it is stagnating |
| `patience` | `5` | number of stagnating iterations in a row before stopping |
| `update_T1` | `True` | perform T1 transitions, see [T1 transitions](t1-transitions.md) |
| `min_dist_T1` | `0.005` | edges no longer than this undergo a T1 transition |
| `cap_vertex_step` | `None` | cap on the step of each vertex, see below |

The relaxation compares the energy at each iteration with its value `patience` iterations earlier.
When the relative change has stayed below `tolerance` for `patience` iterations in a row, or after
`max_nb_iterations` iterations, it stops. There is no criterion on the gradient: a relaxation that
stops is not necessarily at equilibrium. Check the force residual yourself when it matters:

```python
import math

import jax
import jax.numpy as jnp

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.energy import energy_shape_factor_homo

n_cells = 20
width = height = math.sqrt(n_cells)
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(mesh.nb_faces, 3.7)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)


def force_residual(mesh):
    """Norm of the gradient of the energy with respect to the vertex positions."""
    g = jax.grad(energy)(mesh.vertices, mesh.edges, mesh.faces,
                         mesh.vertices_params, mesh.edges_params, mesh.faces_params)
    return float(jnp.linalg.norm(g))


optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.max_nb_iterations = 2000
optimizer.tolerance = 1e-6

print("residual before:", force_residual(mesh))
history = optimizer.inner_optimization(mesh)
print(f"{len(history)} iterations, energy {history[0]:.4f} -> {history[-1]:.4f}")
print("residual after:", force_residual(mesh))
```

For a bounded mesh the relaxation function is [`inner_opt_bounded`][vertax.opt_bounded.inner_opt_bounded].
It takes `cap_vertex_step` and `reset_solver_on_t1` (which re-initializes the solver state at the new
vertex positions and arc angles after a T1 transition) as `minimize` does, and `tension_reset`, which
sets the tension of an edge created by a T1 transition (see [T1 transitions](t1-transitions.md#the-tension-of-the-new-edge)).
`BoundedBilevelOptimizer` exposes `cap_vertex_step` and `t1_tension_reset` as attributes.

The `only_on_*` arguments restrict the energy to a subset of vertices, half-edges and faces, given as
lists of indices.

## Build the solver once and reuse it

The relaxation loop is compiled with `jax.jit`. The energy function, the solver, the box size and the
settings above are static arguments of the compiled function: a new value of any of them compiles the
loop again, which takes seconds where the relaxation itself may take a fraction of a second.

An Optax solver is a pair of functions, and a new call to `optax.sgd(...)` creates new functions,
which JAX does not recognize as the same solver. So build the solver once, keep it, and reuse the
same object, and do the same for the energy function. Assigning `optimizer.inner_solver` once and
calling the optimizer many times is right; creating a new solver object at every call recompiles
every time:

```python
import time

import optax

from vertax.opt import minimize

solver = optax.sgd(learning_rate=0.01)  # built once


def relax(solver):
    start = time.perf_counter()
    minimize(mesh.vertices, mesh.edges, mesh.faces, width, height,
             mesh.vertices_params, mesh.edges_params, mesh.faces_params,
             energy, solver, 0.005, iterations_max=200)
    return time.perf_counter() - start


print("same solver object:", [f"{relax(solver):.2f} s" for _ in range(3)])
print("new solver object: ", [f"{relax(optax.sgd(learning_rate=0.01)):.2f} s" for _ in range(3)])
```

The first call with the same solver compiles; the next ones reuse the compiled loop. With a new solver
object, every call compiles.

## `minimize`

The optimizers call [`vertax.opt.minimize`][vertax.opt.minimize] for a periodic mesh, and the
equivalent `vertax.opt_bounded` functions for a bounded one. You can call `minimize` directly, on the
tables rather than on a mesh:

```text
(vertTable, heTable, faceTable, vert_params, he_params, face_params), (history, steps) = minimize(
    vertTable, heTable, faceTable, width, height, vert_params, he_params, face_params,
    L_in, solver, min_dist_T1,
    iterations_max=1000, tolerance=1e-4, patience=5,
    selected_verts=None, selected_hes=None, selected_faces=None,
    optimization_target=OptimizationTarget.VERTICES, update_t1_func=update_T1,
    perform_energy_check=False, cap_vertex_step=False,
    reset_solver_on_t1=False, count_t1=False)
```

`history` has `iterations_max` entries, of which the first `steps` are the energies of the
iterations performed. `minimize` gives access to options that the optimizers do not expose:
`reset_solver_on_t1` (below), `count_t1` (see [T1 transitions](t1-transitions.md)), the T1 function,
and `optimization_target`, which says which array is optimized (`OptimizationTarget.VERTICES` by
default). A line-search solver re-evaluates the energy along the array being optimized, whatever
the target. Before version 0.3.0, it did so along another array for every target other than
`OptimizationTarget.VERTICES`, and with the usual table shapes the call raised a `TypeError`;
fixed-step solvers were not affected. The optimizers only use `OptimizationTarget.VERTICES`.

## The solvers

Any Optax `GradientTransformation` can be the inner solver.

- **Fixed-step solvers**, such as `optax.sgd` and `optax.adam`, use the gradient only.
- **Line-search solvers** choose the length of each step by re-evaluating the energy along the search
  direction: [`vertax.cg.cg_linesearch`][vertax.cg.cg_linesearch] (Polak-Ribière nonlinear conjugate
  gradient with restarts), [`vertax.cg.gd_linesearch`][vertax.cg.gd_linesearch] (steepest descent) and
  `optax.lbfgs()`. All three use Optax's zoom line search, which satisfies the strong Wolfe
  conditions. The relaxation loop passes them the energy, the gradient and a function that
  re-evaluates the energy.

A line-search solver can take a step of order one, which moves a vertex across whole cells in one
iteration and can turn a cell inside out. The attribute `cap_vertex_step` of the optimizers handles
this. When it is `None` (the default), the step of each vertex is capped for line-search solvers,
recognized by the line-search state they carry, and left alone for fixed-step solvers. `True` or
`False` forces the choice. The cap is `5 * min_dist_T1` per vertex and per iteration; the factor 5
is read from the environment variable `VERTAX_MAX_VERTEX_STEP_FACTOR` when VertAX is imported. When
you call `minimize` yourself, pass `cap_vertex_step=True` with a line-search solver: its default
there is `False`.

With the cap, the line search scores the step as the update leaves it: the capped vertex step and, for
a bounded mesh, the arc angles clipped to their limits, so the energy it evaluates is that of the
position the step reaches. For L-BFGS on a periodic mesh, the stored positions follow the vertices
that the periodic wrap brings back into the box.

After a T1 transition, the state of a line-search solver (the conjugate direction, or the curvature
pairs of L-BFGS) describes the old topology. `minimize(..., reset_solver_on_t1=True)` resets the
solver state whenever a T1 transition has changed the topology; its docstring recommends it with
`optax.lbfgs()`. It is off by default.

`cg_linesearch(restart_every=10, max_linesearch_steps=20, linesearch_initial_guess="keep")` and
`gd_linesearch(max_linesearch_steps=20, linesearch_initial_guess="keep")` start each line search from
the last accepted step. With `linesearch_initial_guess="one"`, each line search starts from a unit
step instead.

Line-search solvers cannot be differentiated through, so they cannot be used with automatic
differentiation as the gradient estimator: the optimizer refuses the combination, prints a warning
and keeps its previous setting (see [Inverse problems](inverse-problems.md)).

The line-search solvers are tested in `float64`. This relaxes the same tissue with
`cg_linesearch`, `optax.lbfgs()` and `optax.sgd`:

```python
import math

import jax
import jax.numpy as jnp
import optax
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.cg import cg_linesearch
from vertax.energy import energy_shape_factor_homo
from vertax.opt import minimize

n_cells = 20
width = height = math.sqrt(n_cells)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)


with enable_x64():
    mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
    vp, hp, fp = jnp.zeros(mesh.nb_vertices), jnp.zeros(mesh.nb_half_edges), jnp.full(mesh.nb_faces, 3.7)
    for name, solver, line_search in [("cg_linesearch", cg_linesearch(), True),
                                      ("optax.lbfgs", optax.lbfgs(), True),
                                      ("optax.sgd", optax.sgd(learning_rate=0.01), False)]:
        (v, he, fa, *_), (history, steps) = minimize(
            mesh.vertices, mesh.edges, mesh.faces, width, height, vp, hp, fp, energy, solver, 0.005,
            iterations_max=500, tolerance=0.0, cap_vertex_step=line_search, reset_solver_on_t1=line_search)
        g = jax.grad(energy)(v, he, fa, vp, hp, fp)
        print(f"{name:14s} {int(steps)} iterations, energy {float(history[int(steps) - 1]):.6f}, "
              f"residual {float(jnp.linalg.norm(g)):.2e}")
```

With `tolerance=0.0` the loop never stagnates and runs all `iterations_max` iterations.
