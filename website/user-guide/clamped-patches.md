# Clamped patches

A crop of a microscopy image is neither periodic nor a closed cluster: some cells are cut by the edge
of the field of view, and at the vertices on that edge one of the cells was never imaged. The force
balance of such a vertex cannot be written from the data. [`vertax.clamped`](../api/clamped.md)
relaxes such a patch with those vertices held fixed, and reports residuals over the other vertices
only. It is the companion of [`vertax.trace`](images-and-masks.md#tracing-a-label-mask-vertaxtrace),
which gives the free and clamped vertices.

`float64` is required: the residuals reached here are of order $10^{-13}$.

## The patch

[`load_mesh`][vertax.clamped.load_mesh]`(path)` reads a patch from an `.npz` file with the keys
`verts_px`, `faces_padded`, `face_len`, `edges`, `edge_is_cellcell`, `free`, `a_ref_px`,
`face_is_interior`, `face_labels` and `stem`, and returns a [`ClampedPatch`][vertax.clamped.ClampedPatch].
The coordinates are divided by $\sqrt{a_\text{ref}}$, so that a typical interior cell has area 1.
Only the edges between two cells are kept as edges (`EA`, `EB`): an edge along unimaged space has both
ends clamped, so it exerts no force on a free vertex, which
[`clamped_edges_are_inert`][vertax.clamped.clamped_edges_are_inert] checks.

!!! note "TODO (H. Turlier)"
    The docstring of `load_mesh` says the file is "written by the tracing stage", but on main no
    function writes it: `vertax.trace` returns a `TracedMesh` in memory. The repository has one such
    file, `tests/fixtures/clamped_patch_demo_003_pos2.npz`.

## The energy is yours

The module defines no energy. Every function takes the energy as a function `E(x, K)` of the vertex
positions `x`, an array of shape `(Nv, 2)`, and of an area stiffness `K`. Build it from
[`patch_geometry`][vertax.clamped.patch_geometry]`(x, patch, eps=0.0)`, which returns the signed face
areas, the face perimeters and the lengths of the cell-cell edges.

The areas are signed, counterclockwise positive, unlike `vertax.geo.get_area` which returns an
absolute value. An area term built on an absolute value has a spurious minimum at zero area and
cannot tell a cell from its inversion; with signed areas, a negative area means that a face has
inverted.

## Relaxing and measuring

- [`relax`][vertax.clamped.relax]`(E, patch, K, maxiter=30000, gtol=1e-12, E_exact=None)` minimizes
  `E` over the free vertices with SciPy's L-BFGS-B, the clamped vertices staying at their observed
  positions. It returns a dictionary with the final positions `x`, the residual `r_eq`, the residual
  at the observed configuration `r_eq_obs`, the signed `min_area`, the number of inverted faces, and
  more. The topology is fixed: edges shorter than `T1_LEN = 1e-3` are counted in `n_edge_lt_1e3`.
- [`r_eq`][vertax.clamped.r_eq]`(grad, patch)` is the residual per free degree of freedom,
  $\lVert \nabla E \rVert / \sqrt{2 N_\text{free}}$ over the free vertices only.
- [`split_gradient`][vertax.clamped.split_gradient]`(E, patch, x)` returns `(u, v)` with
  $\nabla E = K u + v$ exactly, for an energy that is affine in `K`;
  [`best_K`][vertax.clamped.best_K]`(u, v, patch)` gives the `K` that minimizes the free residual, and
  that residual.
- [`newton_polish`][vertax.clamped.newton_polish]`(E, patch, K, x_start, iters=60, tol=1e-13)` refines
  a configuration with Levenberg-Marquardt steps on the force residual, over the free vertices.

```python
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from vertax.clamped import best_K, load_mesh, newton_polish, patch_geometry, r_eq, relax, split_gradient

with enable_x64():
    patch = load_mesh("tests/fixtures/clamped_patch_demo_003_pos2.npz")
    print(f"{len(patch.x0)} vertices, {patch.n_free} free; {patch.nf} faces; {len(patch.EA)} cell-cell edges")
    gamma = jnp.ones(len(patch.EA))  # one tension per cell-cell edge

    def E(x, K):
        A, P, L = patch_geometry(x, patch)
        return 0.5 * K * jnp.sum((A - 1.0) ** 2) + jnp.sum(gamma * L)

    out = relax(E, patch, K=5.0)
    print("residual at the observed positions:", out["r_eq_obs"])
    print("residual after relaxation:", out["r_eq"], "; smallest signed area:", out["min_area"])

    u, v = split_gradient(E, patch, out["x"])
    print("best K for these tensions, and its residual:", best_K(u, v, patch))

    x, info = newton_polish(E, patch, 5.0, out["x"], return_info=True)
    print("after Levenberg-Marquardt:", info["grad_norm"], "converged:", info["converged"])
```

The patch read here is the test fixture of the repository; the path is relative to the root of the
clone. The energy is the one of the README: unit tensions on the cell-cell edges and
an area term with stiffness `K`.
