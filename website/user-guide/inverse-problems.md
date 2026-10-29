# Inverse problems

An inverse problem looks for parameters $\theta$, such as tensions, target areas or preferred
perimeters, whose equilibrium $X^\ast_\theta$ minimizes a cost:

$$
\min_\theta \; \mathcal{C}\left(X^\ast_\theta\right) + R(\theta)
\quad \text{with} \quad
X^\ast_\theta \in \arg\min_X \mathcal{E}(X, \theta).
$$

The inner problem is a relaxation of the energy $\mathcal{E}$ (`loss_function_inner`); the outer
problem changes the parameters to lower the cost $\mathcal{C}$ (`loss_function_outer`). The cost is
called with the mesh tables and the targets, not with the parameters. A term $R(\theta)$ on the
parameters alone is given separately, as `params_regularizer`.

## The bilevel loop

One call of [`bilevel_optimization(mesh)`][vertax.bilevelopt.bilevelopt._BilevelOptimizer.bilevel_optimization]
is one epoch:

1. [`outer_optimization`][vertax.bilevelopt.bilevelopt._BilevelOptimizer.outer_optimization]
   estimates the gradient of the cost with respect to `vertices_params`, `edges_params` and
   `faces_params`, starting the relaxations it needs from the current state of the mesh, and applies
   one step of `outer_solver` to the three parameter arrays;
2. it then applies the mean-value constraints of `params_normalization`, if any (below);
3. `inner_optimization` relaxes the mesh with the new parameters, without the T1 energy check.

The attributes used by the outer problem are:

| Attribute | Default | Meaning |
| --- | --- | --- |
| `loss_function_outer` | `None` | the cost, see [Costs](costs.md) |
| `bilevel_optimization_method` | `EQUILIBRIUM_PROPAGATION` | the gradient estimator, below |
| `outer_solver` | `optax.adam(learning_rate=0.0001, nesterov=True)` | the Optax solver of the parameters |
| `vertices_target`, `edges_target`, `faces_target` | empty arrays | target tables, passed to the cost |
| `image_target` | empty array | a target image, passed to the cost |
| `angles_target` | empty array | bounded only: target angles, passed to the cost |
| `beta` | unset (resolves to `0.01`) | nudging size of equilibrium propagation |
| `ep_variant` | `EPVariant.CENTERED` | form of the equilibrium-propagation estimate |
| `params_normalization` | `None` | mean-value constraints on parameter arrays |
| `mean_tension_gauge` | `None` | alias for `params_normalization={"edges_params": value}` |
| `params_regularizer` | `None` | `R(vert_params, he_params, face_params)`, added to the cost |
| `persist_outer_state` | `False` | keep the outer solver state between epochs (EP, periodic only) |

The outer solver is initialized again at every outer step, for every estimator, unless
`persist_outer_state` is set. With Adam, a step from a fresh state moves each parameter by about the
learning rate, in the direction opposite to the sign of its gradient component, whatever the size of
that component.

[`do_n_bilevel_optimization`][vertax.bilevelopt.bilevelopt._BilevelOptimizer.do_n_bilevel_optimization]`(nb_epochs, mesh, ...)`
runs several epochs and can report and save along the way: with `report_every > 0` it writes the
costs and the time of every reported epoch to `summary.csv`, and plots of the costs, in
`save_folder`; it always writes the settings to `hyper-parameters.txt` there. `add_custom_metric(name,
function)` adds a column computed by `function(mesh, optimizer)`. See the
[many epochs example](../examples/many-epochs.md).

!!! note "Long runs on Linux"
    On Linux with glibc, a long JAX run can grow by several GB, although nothing leaks: the
    allocator keeps memory that XLA has freed in per-thread arenas. Starting the run with
    `MALLOC_ARENA_MAX=2` limits this. In our 36-cell design runs on a 40-thread machine, memory at
    epoch 400 went from 3.9 GB to 1.5 GB, with identical results (every table and snapshot compared
    on a 30-epoch run).

    ```bash
    MALLOC_ARENA_MAX=2 python my_run.py
    ```

## The four gradient estimators

The gradient of the cost with respect to the parameters goes through the equilibrium $X^\ast_\theta$.
`bilevel_optimization_method` chooses how it is estimated, among the members of
[`BilevelOptimizationMethod`][vertax.method_enum.BilevelOptimizationMethod].

### Automatic differentiation (AD)

`AUTOMATIC_DIFFERENTIATION` relaxes the mesh from its current state and differentiates the cost of
the result with `jax.grad`, in reverse mode, through all the iterations of the relaxation, T1
transitions included. The gradient is that of the truncated relaxation, not of the exact
equilibrium. Memory and time grow with the number of iterations.

Line-search solvers cannot be differentiated through. The optimizer refuses to combine AD with a
line-search inner solver: it prints a warning and keeps its previous setting, whether you change the
method or the solver.

### Implicit differentiation (Sen-ID)

`IMPLICIT_DIFFERENTIATION` relaxes the mesh, then uses the optimality condition
$\nabla_X \mathcal{E}(X^\ast, \theta) = 0$. For each parameter $\theta_j$, it solves
$H s_j = -\partial_{\theta_j} \nabla_X \mathcal{E}$, where $H$ is the Hessian of the energy with
respect to the vertex positions, and the gradient is $s_j \cdot \nabla_X \mathcal{C}$. The linear
systems are solved by SciPy's MINRES, with Hessian-vector products from JAX; the Hessian is never
formed. There is one solve for every entry of the three parameter arrays, including entries the
energy does not use, so the cost grows with the number of parameters.

### Adjoint state (Adj-ID)

`ADJOINT_STATE` uses the same optimality condition, solved in the other direction: one solve of
$H \lambda = \nabla_X \mathcal{C}$, then the gradient is
$-\lambda \cdot \partial_\theta \nabla_X \mathcal{E}$. There is one linear solve per outer step,
whatever the number of parameters.

Both implicit estimators assume that the relaxation has reached an equilibrium, and that the Hessian
is well enough conditioned for MINRES. The convergence flag returned by MINRES is not checked.
Use `float64` with them (see [Numerical precision](precision.md)).

### Equilibrium propagation (EP)

`EQUILIBRIUM_PROPAGATION`, the default, needs no derivative of the relaxation. It relaxes the mesh
twice from its current state, with the energy nudged by the cost, $\mathcal{E} \pm \beta \mathcal{C}$,
and takes the difference of the parameter gradients of the energy at the two equilibria:

$$
\nabla_\theta \mathcal{C} \approx
\frac{\partial_\theta \mathcal{E}\left(X^{+\beta}\right) - \partial_\theta \mathcal{E}\left(X^{-\beta}\right)}{2\beta}.
$$

This is the centered form, `EPVariant.CENTERED`, the default. Its error is of order $\beta^2$.
`EPVariant.ONE_SIDED` solves at $0$ and $+\beta$ and divides by $\beta$; its error is of order $\beta$,
and it is there for comparison. `beta` is the half-span: the two solves are at $-\beta$ and $+\beta$.

If an EP run never sets `beta`, the optimizer prints a warning on the first outer step and uses
`BETA_DEFAULT = 0.01`. Set it, and record it with your results.

With `persist_outer_state = True`, a `PbcBilevelOptimizer` using EP keeps the state of the outer
solver from one epoch to the next. With any other method, or with a `BoundedBilevelOptimizer`,
setting it raises a `ValueError` on the first outer step; a non-default `ep_variant` with another
method prints a warning.

### Choosing

| | AD | Sen-ID | Adj-ID | EP |
| --- | --- | --- | --- | --- |
| Differentiates the relaxation | yes | no | no | no |
| Linear solves with the Hessian | none | one per parameter | one | none |
| Relaxations per outer step | one, unrolled | one | one | two |
| Line-search inner solvers | no | yes | yes | yes |
| Approximation | truncated relaxation | equilibrium reached | equilibrium reached | finite $\beta$ |

The [gradient estimators example](../examples/gradient-estimators.md) computes the four estimates on
the same problem.

## Parameter constraints

Some directions of parameter space are hard to observe. For line tensions, adding the same constant to
every tension changes the equilibrium very little, so an inverse problem determines the mean tension
poorly. `params_normalization` fixes the mean of chosen parameter arrays after every outer step, by
the projection $p \leftarrow p - (\operatorname{mean}(p) - \text{target})$. For example,
`{"edges_params": 1.0}` fixes the mean tension at 1, and `{"faces_params": 4.1}` the mean preferred
perimeter at 4.1; the example at the end of this page sets it.

The keys can be `vertices_params`, `edges_params` and `faces_params`. `mean_tension_gauge = value` is
the older spelling of `{"edges_params": value}`; setting both raises a `ValueError`. The constraint is
off by default, and it applies to every estimator and to both mesh types.

Fixing the mean is an assumption that comes from outside the data, and the docstring of
`params_normalization` explains why it matters. On real data, where the true mean is unknown, compare
recovered and reference parameters after subtracting the mean of each.

`params_regularizer`, a function `(vert_params, he_params, face_params) -> scalar`, adds a penalty on
the parameters: its gradient is added to the estimated gradient before the outer step. Its docstring
recommends `params_normalization` rather than a penalty to fix the mean, because a penalty is scaled
differently from a gradient that a truncated relaxation has filtered. Both optimizers pass it to
every estimator.

## Bounded meshes

A `BoundedBilevelOptimizer` has the same attributes and the same estimators, plus `angles_target`.
It honours `params_regularizer` and `ep_variant` as the periodic optimizer does, and refuses
`persist_outer_state` (above).
The bounded inverse problems of the repository are design problems: the cost measures a shape
([`cost_ratio`][vertax.cost.cost_ratio]) or a pattern of fates
([`cost_checkerboard`][vertax.cost.cost_checkerboard]) rather than a distance to a target mesh. Do not
set `vertices_target` with T1 transitions on: the vertex relabelling that follows T1 transitions is
not implemented for bounded meshes and raises an error (see [T1 transitions](t1-transitions.md)).

After an outer step, `BoundedBilevelOptimizer` replaces `mesh.vertices`, `mesh.angles`,
`mesh.edges` and `mesh.faces` by the equilibrium computed by the estimator (the free equilibrium, for
EP; the adjoint state leaves them as they were) and then relaxes the mesh as usual. The functions
behind the four estimators, [`outer_opt_bounded`][vertax.opt_bounded.outer_opt_bounded],
[`outer_eq_prop_bounded`][vertax.opt_bounded.outer_eq_prop_bounded],
[`outer_implicit_bounded`][vertax.opt_bounded.outer_implicit_bounded] and
[`outer_adjoint_state_bounded`][vertax.opt_bounded.outer_adjoint_state_bounded], return seven
arrays: the vertices, angles, edges and faces, then the vertex, edge and face parameters.

Before version 0.3.0, these functions returned six arrays, without the angles, and the mesh kept
the angles it had before the outer step. A bounded run in which a relaxation follows an outer step
(`bilevel_optimization`, `do_n_bilevel_optimization`) gives other results since 0.3.0 with AD, EP
and Sen-ID; see the [release notes](../release-notes.md).

## A first run

This is the inverse problem of the [Getting started](../getting-started.md) page, with the two
settings that matter most written out: the estimator and `beta`, and a mean-value constraint on the
preferred perimeters.

```python
import math

import jax
import jax.numpy as jnp
import optax

from vertax import BilevelOptimizationMethod, EPVariant, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import energy_shape_factor_hetero

n_cells = 20
width = height = math.sqrt(n_cells)
all_faces = jnp.arange(n_cells)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_hetero(vertTable, heTable, faceTable, width, height, all_faces, face_params)


mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(n_cells, 3.7)

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_optimization(mesh)

target = PbcMesh.copy_mesh(mesh)
target.faces_params = 3.7 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n_cells,))
optimizer.inner_optimization(target)
optimizer.vertices_target = target.vertices
optimizer.edges_target = target.edges
optimizer.faces_target = target.faces

optimizer.loss_function_outer = cost_v2v
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.ep_variant = EPVariant.CENTERED
optimizer.beta = 0.01
optimizer.outer_solver = optax.adam(learning_rate=1e-3)
optimizer.params_normalization = {"faces_params": float(target.faces_params.mean())}

for epoch in range(3):
    optimizer.bilevel_optimization(mesh)
    print(f"epoch {epoch + 1}: cost {optimizer.compute_outer_loss(mesh):.3e}, "
          f"mean p0 {float(mesh.faces_params.mean()):.4f}")

print(optimizer.self_summary())
```

The mean of the preferred perimeters is fixed here to that of the target, which is known because the
target was made by the code; on real data that value is an assumption.

For tension inference on periodic meshes, the repository also has a practical guide,
`docs/tension_inference_guide.md`, with measured results on the settings that matter.
