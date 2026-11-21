# Rheology

[`vertax.rheology`](../api/rheology.md) measures the linear elastic response of a periodic tissue
without deforming the periodic box. [`ShearRheologyOptimizer`](../api/shear-rheology-optimizer.md)
uses that measurement in an inverse design: it changes the parameters so that the tissue reaches a
target compliance. Both require `float64`.

## The drive

The tissue is driven by a body force along one axis, modulated as a cosine along the other,
$f_x(y) = a \cos(q y)$ with $q = 2\pi n / H$. The total force is zero, so no wall is needed. Since
VertAX minimizes an energy, the force enters as a potential added to the energy,

$$
\mathcal{E}_{\text{drive}} = -a \sum_i \hat c_i \, \Delta x_i ,
\qquad
\hat c_i = \cos(q y^0_i) - \tfrac{1}{N} \textstyle\sum_j \cos(q y^0_j),
$$

where the weights $\hat c_i$ are computed once, on the relaxed undriven state $y^0$, and $\Delta x_i$
is the displacement of vertex $i$ from that state, measured through the periodic boundary. The
response is the amplitude $U = \sum_i \hat c_i \Delta x_i / \sum_i \hat c_i^2$ of the displacement
along the drive.

<figure markdown="span">
  ![Left, a periodic tissue with horizontal arrows on the vertices whose length follows a cosine of the height; right, the horizontal displacement of each vertex against its height, with a cosine curve](../figures/rheology-shear-drive.svg)
  <figcaption markdown="span">The drive on the 20-cell tissue of the code below ($p_0 = 3.6$): the force $a\,\hat c_i$ on each vertex (left; arrows to scale with one another), and the displacements $\Delta x_i$ at $a = +a_0 = 0.02$ against the height $y^0_i$, with the mode at the measured amplitude $U = 0.038$ (right).</figcaption>
</figure>

- [`make_shear_mode`][vertax.rheology.make_shear_mode]`(vertices, width, height, n=1, mod_axis=1, pol_axis=0)`
  builds the mode, a [`ShearMode`][vertax.rheology.ShearMode], from the relaxed undriven vertices.
  `mod_axis` is the axis of the cosine (1 for $\cos(qy)$), `pol_axis` the direction of the force and
  of the measured displacement (0 for $x$). It raises a `RuntimeError` when 64-bit mode is off.
- [`make_drive_energy`][vertax.rheology.make_drive_energy]`(base_energy, mode, a)` adds the drive at
  amplitude `a` to a six-argument periodic energy. The energy returned raises a `ValueError` when it
  is called with another number of vertices than the mode has, which happens when the relaxation is
  restricted to a subset of vertices.
- [`mode_amplitude`][vertax.rheology.mode_amplitude]`(vertices, mode)` is $U$.
- [`compliance`][vertax.rheology.compliance]`(u_plus, u_minus, a0)` is
  $J = (U_+ - U_-) / (2 a_0)$, from relaxations at $+a_0$ and $-a_0$.
- [`modulus_from_compliance`][vertax.rheology.modulus_from_compliance]`(j, mode)` is the apparent
  modulus $G = \rho_v / (J q^2)$, with $\rho_v$ the number of vertices per unit area, and
  [`target_compliance`][vertax.rheology.target_compliance]`(g_target, mode)` its inverse.

With `mod_axis=1` and `pol_axis=0` (the defaults), the measured constant is the shear modulus; the
docstring of `ShearMode` lists the other combinations. With T1 transitions on, a compliance measured
across a T1 transition is not a linear elastic constant; `topology_signature` and `t1_event_count`
count the transitions between two states.

This measures the compliance of a small tissue. The reference state is relaxed with the optimizer;
the two driven states with `minimize`, from the reference:

```python
import math

import jax
import jax.numpy as jnp
import optax
from jax.experimental import enable_x64

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.energy import energy_shape_factor_homo
from vertax.opt import minimize
from vertax.rheology import (
    compliance,
    make_drive_energy,
    make_shear_mode,
    mode_amplitude,
    modulus_from_compliance,
    t1_event_count,
    topology_signature,
)

side = math.sqrt(20.0)
A0 = 0.02  # drive amplitude


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, side, side, face_params)


with enable_x64():
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=231)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    mesh.faces_params = jnp.full(mesh.nb_faces, 3.6)

    optimizer = PbcBilevelOptimizer()
    optimizer.loss_function_inner = energy
    optimizer.max_nb_iterations = 3000
    optimizer.tolerance = 1e-8
    optimizer.inner_optimization(mesh)

    mode = make_shear_mode(mesh.vertices, side, side, n=1)
    solver = optax.sgd(learning_rate=0.01)
    amplitudes, tables = {}, {}
    for sign in (+1, -1):
        driven = make_drive_energy(energy, mode, sign * A0)
        (v, he, fa, *_), _ = minimize(
            mesh.vertices, mesh.edges, mesh.faces, side, side,
            mesh.vertices_params, mesh.edges_params, mesh.faces_params,
            driven, solver, optimizer.min_dist_T1, iterations_max=3000, tolerance=0.0)
        amplitudes[sign] = float(mode_amplitude(v, mode))
        tables[sign] = he

    j = compliance(amplitudes[+1], amplitudes[-1], A0)
    print("amplitudes:", amplitudes)
    print(f"compliance J = {float(j):.4f}, modulus G = {float(modulus_from_compliance(j, mode)):.4f}")
    print("T1 events, +a0 and -a0:", [t1_event_count(topology_signature(mesh.edges), topology_signature(tables[s]))
                                      for s in (+1, -1)])
```

The number of iterations needed to reach equilibrium has to be measured, not assumed. The docstring
of `ShearRheologyOptimizer` reports that, on a 72-cell tissue, a driven solve started from a
converged free state still needed about 5000 iterations of conjugate gradient to reach a force
residual of $10^{-8}$. Check the residual, as in [Relaxation and solvers](relaxation.md).

## Designing a compliance

[`ShearRheologyOptimizer`][vertax.bilevelopt.rheobop.ShearRheologyOptimizer] changes one parameter
array (`theta_target`, the face parameters by default) so that the compliance reaches `j_target`. Its
`epoch(vertices, edges, faces, params, outer_solver, outer_state)` runs one outer step:

1. the free phase relaxes the tissue at $+a_0$ and at $-a_0$;
2. the nudged phases relax both conditions again with the cost
   [`make_compliance_cost`][vertax.rheology.make_compliance_cost] added with weight $+\beta$, then
   $-\beta$, alternating between the two conditions `n_sweeps` times;
3. the gradient is the centered equilibrium-propagation estimate, summed over the two conditions;
4. one step of `outer_solver` updates the parameters.

It returns the new parameters, the new state of the outer solver, and a dictionary of diagnostics:
the compliance and modulus, the cost, the force residuals, the T1 events, and whether the $+\beta$
and $-\beta$ solves crossed the same T1 transitions (`ep_step_valid`).

Its settings are fields of the dataclass:

- `energy_theta`: the undriven six-argument energy;
- `mode`, `a0`, `j_target`, `beta`: the drive, its amplitude, the target and the nudging size;
- `solver_factory`: a function without arguments that returns the inner solver;
- `width`, `height`, `min_dist_T1`;
- `free_iters` (20000) and `nudged_iters` (5000): the number of iterations of each relaxation. The
  relaxations run with `tolerance=0.0`, so they always run all these iterations;
- `theta_target`, `update_t1` (`True`), `cap_vertex_step` (`True`), `n_sweeps` (2, at most 4).

Each relaxation builds a new drive energy, so each one is compiled anew.

$\beta$ has to be small compared with $a_0^2$: the cost divides by $(2 a_0)^2$, and a larger $\beta$
turns the nudge into a constraint. [`beta_upper_bound`][vertax.bilevelopt.rheobop.ShearRheologyOptimizer.beta_upper_bound]
computes the bound, and `beta_diagnostics` the quantities that show whether the nudge was a small
perturbation. The module docstring of [`vertax.bilevelopt.rheobop`](../api/shear-rheology-optimizer.md)
explains these choices.

See the [shear rheometry example](../examples/shear-rheometry.md), which runs one epoch with small
iteration budgets.

## Diagnostics

`vertax.rheology` also contains diagnostics that `vertax` does not export: virial stresses
(`virial_stress`, `virial_stress_e1`, `virial_stress_e2`, `stress_profile_xy`), Hessian spectra
(`hessian_spectrum`, `hessian_lambda_max`, `top_mode_participation`, `hessian_compliance`),
`grad_x_norm`, `drive_drift`, and the topology helpers `topology_signature`, `t1_event_count` and
`connectivity_hash`. Import them from `vertax.rheology`.
