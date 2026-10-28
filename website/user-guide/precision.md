# Numerical precision

VertAX follows JAX's default float dtype. That is `float32`, unless JAX's 64-bit mode is enabled, in
which case meshes and relaxations are in `float64`.

## Enabling float64

Enable 64-bit mode before the first JAX array is created, at the start of a script, with
`jax.config.update("jax_enable_x64", True)` (shown at the end of this section), or set the
environment variable `JAX_ENABLE_X64=1` before starting Python. To use `float64` in one part of a
program only, use the context manager `jax.experimental.enable_x64()`; the previous setting is
restored when the block ends.

Meshes built while 64-bit mode is on are `float64`: the constructors of `PbcMesh` (`from_seeds`,
`from_random_seeds`, `from_mask`, `from_image`) and of `BoundedMesh` (`from_seeds`,
`from_random_seeds`) use JAX's default float dtype for the vertices, and for the angles of a bounded
mesh. No cast is needed after construction. The integer tables, `edges` and `faces`, are `int32` in
both modes.

```python
import math

import jax.numpy as jnp
from jax.experimental import enable_x64

from vertax import BoundedMesh, PbcMesh

side = math.sqrt(20)

mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
print("default:", mesh.vertices.dtype, mesh.edges.dtype, mesh.faces.dtype)

with enable_x64():
    mesh64 = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
    cluster64 = BoundedMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=1)
    print("x64:    ", mesh64.vertices.dtype, mesh64.edges.dtype, mesh64.faces.dtype)
    print("bounded:", cluster64.vertices.dtype, cluster64.angles.dtype)
    print("parameters made here:", jnp.zeros(3).dtype)
```

A mesh built before 64-bit mode was enabled stays `float32`. Its vertices can be converted with
`jnp.asarray(mesh.vertices, dtype=jnp.float64)` once 64-bit mode is on. Parameter arrays follow the
same rule: create them, or convert them, while 64-bit mode is on.

Before version 0.2.1, the constructors always built `float32` vertices; code written for 0.2.0 may
still cast them after construction, which is now harmless.

The usual form, at the start of a script:

```python
import math

import jax

jax.config.update("jax_enable_x64", True)

from vertax import PbcMesh

side = math.sqrt(20)
mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
print(mesh.vertices.dtype)  # float64
```

## When to use float64

Use `float64` for anything that relies on small differences or small residuals: Hessians, their
conditioning, equilibrium residuals, and the implicit-differentiation and adjoint-state gradient
estimators, which solve linear systems with the Hessian. Two modules require it:

- [`vertax.rheology`](rheology.md): `make_shear_mode` calls `require_float64`, which raises a
  `RuntimeError` when 64-bit mode is off.
- [`vertax.clamped`](clamped-patches.md): its residuals reach `1e-13`, which `float32` cannot
  represent.

Relaxations with T1 transitions branch on comparisons of edge lengths, so a last-bit difference can
change which transitions happen and lead to a different equilibrium. The same script can therefore
give different meshes in `float32` and in `float64`, or on two machines in `float32`. The regression
tests of VertAX compare energies, residuals and topology rather than vertex coordinates for this
reason.

The line-search solvers `cg_linesearch`, `gd_linesearch` and `optax.lbfgs()` work under 64-bit mode
since version 0.2.0, and their tests run in `float64`.
