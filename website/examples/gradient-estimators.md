# Gradient estimators

One outer step with each of the four estimators, from the same state, on the same problem. With the
outer solver `optax.sgd(learning_rate=1.0)`, an outer step changes the parameters by exactly minus
the estimated gradient, so the gradient can be read from the change. The problem is small: the 16-cell
tissue of `tests/test_minimize_lbfgs_t1_reset.py`, the shape-factor energy with one preferred
perimeter per cell, and a target made by the script. The optimizer settings follow
`tests/test_cap_vertex_step.py`, which runs all four estimators. Everything is in `float64`.

## An equilibrium and a target

The implicit estimators assume that the relaxation has reached an equilibrium. The tissue and the
target are therefore relaxed with L-BFGS and T1 transitions to a force residual near machine
precision, with the settings of the L-BFGS test.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import energy_shape_factor_hetero
from vertax.opt import minimize

WIDTH = HEIGHT = 4.0
mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
all_faces = jnp.arange(mesh.nb_faces)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_hetero(vertTable, heTable, faceTable, WIDTH, HEIGHT, all_faces, face_params)


lbfgs = optax.lbfgs()


def equilibrate(m):
    (v, he, f, *_), _ = minimize(m.vertices, m.edges, m.faces, WIDTH, HEIGHT,
                                 m.vertices_params, m.edges_params, m.faces_params, energy, lbfgs, 0.05,
                                 iterations_max=2000, tolerance=0.0, perform_energy_check=True,
                                 cap_vertex_step=True, reset_solver_on_t1=True)
    m.vertices, m.edges, m.faces = v, he, f
    g = jax.grad(energy)(v, he, f, m.vertices_params, m.edges_params, m.faces_params)
    return float(jnp.linalg.norm(g))


# Only the face parameters are used. The other two arrays have one entry: implicit differentiation
# solves one linear system per entry of every parameter array.
mesh.vertices_params = jnp.zeros(1)
mesh.edges_params = jnp.zeros(1)
mesh.faces_params = 3.7 + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_faces,))
print("residual of the tissue:", equilibrate(mesh))

target = PbcMesh.copy_mesh(mesh)
target.faces_params = 3.7 + 0.05 * jax.random.normal(jax.random.PRNGKey(2), (mesh.nb_faces,))
print("residual of the target:", equilibrate(target))
print("same topology:", bool(jnp.all(mesh.edges == target.edges)))
```

## One outer step per estimator

T1 transitions are off during the outer steps, so that every estimator differentiates the same
topology. Automatic differentiation unrolls the 2000 iterations of SGD of its relaxation; equilibrium
propagation runs two relaxations of 2000 iterations with $\beta = 10^{-3}$.

```python
import time

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.loss_function_outer = cost_v2v
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.outer_solver = optax.sgd(learning_rate=1.0)  # the step is minus the gradient
optimizer.update_T1 = False
optimizer.min_dist_T1 = 0.05
optimizer.max_nb_iterations = 2000
optimizer.tolerance = 0.0
optimizer.beta = 1e-3
optimizer.vertices_target = target.vertices
optimizer.edges_target = target.edges
optimizer.faces_target = target.faces


def estimates(start):
    gradients = {}
    for method in BilevelOptimizationMethod:
        trial = PbcMesh.copy_mesh(start)
        optimizer.bilevel_optimization_method = method
        t0 = time.perf_counter()
        optimizer.outer_optimization(trial)
        gradients[method.name] = start.faces_params - trial.faces_params
        print(f"  {method.name:26s} |gradient| {float(jnp.linalg.norm(gradients[method.name])):.6e}"
              f"  ({time.perf_counter() - t0:.0f} s)")
    reference = gradients["ADJOINT_STATE"]
    for name, g in gradients.items():
        difference = float(jnp.linalg.norm(g - reference) / jnp.linalg.norm(reference))
        print(f"  {name:26s} relative difference from the adjoint state: {difference:.1e}")


estimates(mesh)
```

From an equilibrium, the four estimates agree closely: automatic differentiation and equilibrium
propagation approximate, through a long relaxation and a small $\beta$, the gradient that the two
implicit estimators compute at the equilibrium. In the run that checked this page, the norm of the
gradient was $2.77 \times 10^{-3}$ for automatic differentiation and equilibrium propagation and
$2.79 \times 10^{-3}$ for the two implicit estimators; relative to the adjoint-state estimate, the
difference was $0.9\,\%$ for automatic differentiation and equilibrium propagation and $0.1\,\%$ for the
sensitivity estimate.

## When the state is not at equilibrium

The same comparison from the unrelaxed Voronoi tissue, with relaxations of 200 iterations only:

```python
unrelaxed = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
unrelaxed.vertices_params = mesh.vertices_params
unrelaxed.edges_params = mesh.edges_params
unrelaxed.faces_params = 3.7 + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_faces,))
optimizer.max_nb_iterations = 200
estimates(unrelaxed)
```

Far from equilibrium the four estimates disagree by orders of magnitude. In the run that checked
this page (one CPU thread, `float64`), the norms were about $1.4 \times 10^{9}$ for automatic
differentiation, 27 for equilibrium propagation, and 0.014 for the two implicit estimators, which
agree with each other. None of these is the gradient at an equilibrium: automatic differentiation and
equilibrium propagation work on relaxations that are far from converged, and the implicit estimators
apply the optimality condition at a state where it does not hold. Relax to equilibrium, and check the
force residual, before trusting an estimate.
