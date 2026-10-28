# Shear rheometry

The drive of [`vertax.rheology`](../user-guide/rheology.md) adds to the energy a potential that pushes
the vertices with a cosine force. This example builds it on a 20-cell periodic tissue, as the README
and `tests/test_rheology.py` do, and checks the properties that the tests check before any relaxation:
the weights sum to zero, the force is exactly the one intended, the drive does not change the Hessian,
and a vertex that crosses the periodic boundary changes nothing. Then it measures a compliance from
two driven relaxations. Everything is in `float64`, which `vertax.rheology` requires.

## The mode and the drive

The tests use the shape-factor energy with $p_0 = 3.6$ and a drive amplitude $a_0 = 0.02$. The mode
is built here on the mesh as constructed, as in the tests; for a measurement it must be built on the
relaxed undriven state, as in the last section.

```python
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

from vertax import PbcMesh
from vertax.energy import energy_shape_factor_homo
from vertax.geo import update_pbc
from vertax.opt import minimize
from vertax.rheology import (
    compliance,
    hessian_lambda_max,
    make_drive_energy,
    make_shear_mode,
    mode_amplitude,
    modulus_from_compliance,
    target_compliance,
)

SIDE = math.sqrt(20.0)
P0 = 3.60
A0 = 0.02

mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=SIDE, height=SIDE, random_key=231)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(mesh.nb_faces, P0)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_homo(vertTable, heTable, faceTable, SIDE, SIDE, face_params)


mode = make_shear_mode(mesh.vertices, SIDE, SIDE, n=1)
drive = make_drive_energy(energy, mode, A0)
print(mode.label, f"q = {mode.q:.4f}, rho_v = {mode.rho_v:.4f}, c_sq_ratio = {mode.c_sq_ratio:.4f}")
print("sum of the weights:", float(jnp.sum(mode.c)))
```

## What the drive does

```python
args = (mesh.edges, mesh.faces, mesh.vertices_params, mesh.edges_params, mesh.faces_params)

# The force of the drive is a0 * c_i along x, and zero along y.
force = jax.grad(energy)(mesh.vertices, *args) - jax.grad(drive)(mesh.vertices, *args)
expected = jnp.zeros_like(force).at[:, 0].set(A0 * mode.c)
print("force error:", float(jnp.max(jnp.abs(force - expected))))

# The drive is linear in the positions, so the largest Hessian eigenvalue does not change.
bare = hessian_lambda_max(energy, mesh.vertices, *args)
driven = hessian_lambda_max(drive, mesh.vertices, *args)
print("largest Hessian eigenvalue, without and with the drive:", bare, driven)

# A vertex pushed across the periodic boundary and wrapped back gives the same amplitude and energy.
i = int(jnp.argmin(mesh.vertices[:, 0]))
pushed = mesh.vertices.at[i, 0].add(-2.0 * float(mesh.vertices[i, 0]) - 1e-3)
wrapped, edges_w, faces_w = update_pbc(pushed, mesh.edges, mesh.faces, SIDE, SIDE)
print("amplitude before and after wrapping:", float(mode_amplitude(pushed, mode)), float(mode_amplitude(wrapped, mode)))
print("energy before and after wrapping:",
      float(drive(pushed, *args)),
      float(drive(wrapped, edges_w, faces_w, *args[2:])))

# The modulus and the target compliance are inverse functions.
print("G from target_compliance(0.5):", float(modulus_from_compliance(target_compliance(0.5, mode), mode)))
```

## A compliance from two driven relaxations

The reference state is relaxed without drive, the mode is built on it, and the tissue is relaxed at
$+a_0$ and at $-a_0$ from the reference. The compliance is $J = (U_+ - U_-) / (2 a_0)$.

```python
solver = optax.sgd(learning_rate=0.01)
(reference, edges_ref, faces_ref, *_), _ = minimize(
    mesh.vertices, mesh.edges, mesh.faces, SIDE, SIDE, *args[2:], energy, solver, 0.005,
    iterations_max=4000, tolerance=0.0)
mode = make_shear_mode(reference, SIDE, SIDE, n=1)

amplitude = {}
for sign in (+1, -1):
    (v, *_), _ = minimize(reference, edges_ref, faces_ref, SIDE, SIDE, *args[2:],
                          make_drive_energy(energy, mode, sign * A0), solver, 0.005,
                          iterations_max=4000, tolerance=0.0)
    amplitude[sign] = float(mode_amplitude(v, mode))

residual = jnp.linalg.norm(jax.grad(energy)(reference, edges_ref, faces_ref, *args[2:]))
j = compliance(amplitude[+1], amplitude[-1], A0)
print(f"residual of the reference: {float(residual):.1e}")
print(f"U+ = {amplitude[+1]:.5f}, U- = {amplitude[-1]:.5f}, J = {float(j):.4f}, "
      f"G = {float(modulus_from_compliance(j, mode)):.4f}")
```

Check the force residual of every state before trusting a compliance: a compliance read from states
that are not at equilibrium is not a compliance. The design of a compliance with
[`ShearRheologyOptimizer`](../user-guide/rheology.md#designing-a-compliance) repeats such relaxations
at every epoch; the tests of the repository construct it, but none runs an epoch.
