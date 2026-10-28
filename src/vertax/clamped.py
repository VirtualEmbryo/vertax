"""Mechanics on a clamped patch: a tissue fragment whose boundary is a field-of-view cut.

A crop of a microscopy field is not a closed tissue and it is not periodic. Some cells are cut by
the edge of the image, and at the junctions on that edge one of the incident cells was never
imaged, so their force balance cannot be written from the data at all. This module provides the
mechanics for that situation: which degrees of freedom carry an equation, how to relax only those,
and how to report a residual that means something.

It is the companion to `vertax.trace`, which produces the patch geometry from a segmentation mask.

## What this module does NOT contain: the energy

**No energy is defined here, deliberately.** Every function below takes the energy as a callable
`E(x, K)`. Defining `E1` and `E2` here would put a second implementation of the same physics inside
`src/` alongside `vertax.energy`, which is the drift hazard this package has already paid for once.

The reason `vertax.energy` cannot simply be called instead is worth stating precisely, because it
is a data-structure mismatch rather than a disagreement about physics:

* `vertax.energy` and `vertax.geo` operate on the **half-edge tables** of a `PbcMesh`
  (`vertTable`, `heTable`, `faceTable`, plus `width`/`height`), and every length and area they
  compute goes through the periodic minimum-image convention.
* A clamped patch has **no periodicity and no half-edge table**. It is a set of polygons over a
  shared vertex array, and its boundary is a real boundary rather than a wrap.

Putting a patch into the periodic representation would mean inventing half-edge twins across an
edge that has no other side, and setting a box large enough that the minimum-image rule never
fires — a representation that is numerically inert but structurally false. So the caller supplies
the energy, built from `patch_geometry` below, and this module supplies everything else.

## Signed areas, and why `geo.get_area` is not used

`vertax.geo.get_area` returns `0.5 * |shoelace|`. That absolute value is correct for the periodic
meshes it serves and is not changed here. It is **wrong for a clamped patch**, and the failure is
silent:

* an area term written `(|A| - 1)²` has a **spurious minimum at `A = 0`**, so a cell collapsing to
  a point looks like an improvement rather than a failure;
* `|A|` cannot distinguish a cell from its own inversion, so a mesh that has folded through itself
  scores the same as one that has not.

Collapse and inversion are exactly the failure modes a fixed-topology relaxation on real,
imperfect geometry runs into, so this module computes a **signed** area throughout, via
`signed_face_areas`. The sign convention is counter-clockwise-positive, matching `vertax.trace`,
whose faces are emitted in that winding.

## The free/clamped partition, which is the substantive content

A vertex is **clamped** when one of the materials meeting there was not imaged. Such a vertex
carries an unknown reaction force from the missing environment, so:

* it is **held fixed** during relaxation — `relax` optimises the free coordinates only;
* its gradient component is **excluded from the residual** — `r_eq` normalises over free
  components alone. Including it would report the strength of the clamp rather than the quality of
  the mechanics, and it would improve as the clamp got stiffer.

## Typical use

```python
from vertax.clamped import load_mesh, patch_geometry, relax, r_eq, split_gradient

patch = load_mesh("crop.npz")                  # geometry + the free/clamped partition

def E(x, K):                                   # the caller's energy; see the note above
    A, P, L = patch_geometry(x, patch)
    return 0.5 * K * jnp.sum((A - 1.0) ** 2) + jnp.sum(gamma * L)

out = relax(E, patch, K=5.0)                   # free coordinates only
print(out["r_eq"], out["min_area"])            # residual over free DOF; signed, so <0 means folded
u, v = split_gradient(E, patch, out["x"])      # grad E = K u + v, exactly
```

`float64` is required: the residuals reported here reach `1e-13`, which `float32` cannot represent.
Enable it before importing JAX arrays, e.g. `jax.config.update("jax_enable_x64", True)` at process
scope or `jax.experimental.enable_x64()` as a context manager. The arrays of a loaded patch are
`float64` on disk and become JAX arrays in JAX's default float dtype, so under x64 they stay `float64`
with no cast; without x64 they are rounded to `float32`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from numpy.typing import NDArray

# An edge shorter than this counts as collapsed to a point: a T1 the fixed-topology relaxation
# cannot perform. In units of sqrt(A_ref), so it is a fraction of a typical cell diameter.
T1_LEN = 1e-3

__all__ = [
    "ClampedPatch",
    "T1_LEN",
    "best_K",
    "clamped_edges_are_inert",
    "load_mesh",
    "newton_polish",
    "patch_geometry",
    "r_eq",
    "relax",
    "signed_face_areas",
    "smoothed_norm",
    "split_gradient",
]


@dataclass
class ClampedPatch:
    """Geometry and boundary partition of one clamped patch, in units of `sqrt(A_ref)`.

    Faces are stored as a flat list of polygon corners rather than as cycles, so that areas and
    perimeters are one `segment_sum` each: `PA[i] -> PB[i]` is a polygon edge belonging to face
    `PF[i]`.

    Attributes:
        stem: identifier of the crop this came from.
        x0: `(Nv, 2)` observed vertex positions, normalised by `sqrt(A_ref)`.
        PA, PB, PF: polygon-corner arrays — start vertex, end vertex, owning face.
        nf: number of faces.
        EA, EB: endpoints of the **cell-cell** edges only. Edges bordering unimaged space carry no
            tension and exert no force on any free vertex; see `clamped_edges_are_inert`.
        free_idx: indices of vertices whose force balance can be written from the data.
        n_free: `len(free_idx)`, cached because the residual normalises by it.
        a_ref_px: the reference area in px², i.e. the square of the length unit.
        face_is_interior: faces none of whose edges borders unimaged space.
        face_labels: original segmentation label of each face.
    """

    stem: str
    x0: NDArray
    PA: NDArray
    PB: NDArray
    PF: NDArray
    nf: int
    EA: NDArray
    EB: NDArray
    free_idx: NDArray
    a_ref_px: float
    face_is_interior: NDArray
    face_labels: NDArray
    n_free: int


def load_mesh(path: str | Path) -> ClampedPatch:
    """Load a traced patch from an `.npz` file with the keys listed below.

    No function of VertAX writes this file: it is the output format of the tracing stage of the
    analysis pipeline, outside the library. `tests/fixtures/clamped_patch_demo_003_pos2.npz` is one.

    Coordinates are divided by `sqrt(a_ref_px)` on load, so that a typical interior cell has unit
    area and every length in this module is dimensionless. That is what makes tensions comparable
    between crops imaged at different magnifications.

    Args:
        path: `.npz` carrying `verts_px`, `faces_padded`, `face_len`, `edges`,
            `edge_is_cellcell`, `free`, `a_ref_px`, `face_is_interior`, `face_labels`, `stem`.

    Returns:
        A `ClampedPatch` in normalised coordinates.
    """
    z = np.load(path, allow_pickle=True)
    a_ref = float(z["a_ref_px"])
    x0 = np.asarray(z["verts_px"], dtype=np.float64) / np.sqrt(a_ref)
    fpad, flen = z["faces_padded"], z["face_len"]
    PA, PB, PF = [], [], []
    for k in range(fpad.shape[0]):
        cyc = fpad[k, : flen[k]]
        PA.extend(cyc.tolist())
        PB.extend(np.roll(cyc, -1).tolist())
        PF.extend([k] * len(cyc))
    cc = np.asarray(z["edge_is_cellcell"], dtype=bool)
    e = np.asarray(z["edges"])[cc]
    free = np.asarray(z["free"], dtype=bool)
    return ClampedPatch(
        stem=str(z["stem"]),
        x0=x0,
        PA=np.asarray(PA, np.int64),
        PB=np.asarray(PB, np.int64),
        PF=np.asarray(PF, np.int64),
        nf=int(fpad.shape[0]),
        EA=e[:, 0],
        EB=e[:, 1],
        free_idx=np.nonzero(free)[0],
        a_ref_px=a_ref,
        face_is_interior=np.asarray(z["face_is_interior"], bool),
        face_labels=np.asarray(z["face_labels"]),
        n_free=int(free.sum()),
    )


def smoothed_norm(d, eps: float):  # noqa: ANN001, ANN201
    """Edge length, optionally smoothed as `sqrt(l² + eps²)`.

    `l = ||dx||` has a kink at `l = 0`, and an energy with tension drives some edges there — that is
    a T1 demand, not a numerical accident. The kink does not merely stall the local force balance:
    it breaks the quasi-Newton model globally, so L-BFGS terminates with a residual on **every**
    free vertex and a genuine obstruction becomes indistinguishable from a failed minimisation.

    With `eps > 0` the kink is gone — `d(sqrt(l²+eps²))/dl = l/sqrt(l²+eps²) -> 0` as `l -> 0`, so a
    zero-length edge is a true smooth stationary point — and the solve converges to machine
    precision. The **unsmoothed** residual is then re-evaluated at that configuration, and what
    remains is the irreducible T1 force and nothing else.

    At `eps = 1e-6` the smoothing shifts a real edge (`l > 1e-2`) by under `5e-11` in relative
    terms, orders below any residual worth reporting.

    Args:
        d: `(N, 2)` edge vectors.
        eps: smoothing scale; `0` gives the exact norm.
    """
    sq = jnp.sum(d * d, axis=1)
    return jnp.sqrt(sq + eps * eps) if eps > 0 else jnp.sqrt(sq)


def signed_face_areas(x, patch: ClampedPatch):  # noqa: ANN001, ANN201
    """Shoelace area per face, **signed**, counter-clockwise positive.

    Explicitly signed, and not `vertax.geo.get_area`, which returns `0.5 * |shoelace|`. The absolute
    value is right for the periodic meshes that function serves and is left alone; here it would
    hide the two failure modes that matter on real geometry. An area term built on `|A|` has a
    spurious minimum at `A = 0`, so a collapsing cell reads as an improvement, and it scores an
    inverted cell identically to a healthy one.

    Callers should treat `A <= 0` as an inverted face and a small positive `A` as a collapsing one.
    """
    pa, pb = x[patch.PA], x[patch.PB]
    return _areas_from_corners(pa, pb, patch)


def _areas_from_corners(pa, pb, patch: ClampedPatch):  # noqa: ANN001, ANN201
    """Shoelace from already-gathered corners, so `patch_geometry` gathers exactly once.

    Not merely an optimisation: gathering twice changes the traced graph, and XLA then reassociates
    the perimeter reduction differently. That was measured as a 4.8e-16 relative drift in the E1
    gradient against the pre-promotion implementation -- one ulp, harmless in itself, but it breaks
    bit-exact equivalence, which is the only equivalence claim worth making.
    """
    cross = pa[:, 0] * pb[:, 1] - pa[:, 1] * pb[:, 0]
    return 0.5 * segment_sum(cross, patch.PF, patch.nf)


def patch_geometry(x, patch: ClampedPatch, eps: float = 0.0):  # noqa: ANN001, ANN201
    """Signed face areas, face perimeters and cell-cell edge lengths at configuration `x`.

    The three quantities every energy on a patch is built from, computed together because they
    share the same gather. Areas are signed — see `signed_face_areas`.

    Args:
        x: `(Nv, 2)` vertex positions in normalised units.
        patch: the patch whose connectivity to use.
        eps: edge-length smoothing, passed to `smoothed_norm`. Applies to perimeters and edge
            lengths; areas are polynomial and need no smoothing.

    Returns:
        `(areas, perimeters, edge_lengths)`, with `edge_lengths` over cell-cell edges only.
    """
    pa, pb = x[patch.PA], x[patch.PB]
    A = _areas_from_corners(pa, pb, patch)
    P = segment_sum(smoothed_norm(pb - pa, eps), patch.PF, patch.nf)
    L = smoothed_norm(x[patch.EB] - x[patch.EA], eps)
    return A, P, L


def split_gradient(E, patch: ClampedPatch, x):  # noqa: ANN001, ANN201
    """Decompose `grad E = K u + v` exactly, and return `(u, v)` at `x`.

    Exact, not a fit: an energy of the form `(K/2) Σ(A - A⁰)² + (terms independent of K)` is affine
    in `K`, so two gradient evaluations determine the decomposition completely. `u = d(grad E)/dK`
    and `v = grad E` at `K = 0`.

    This is what makes the `K` sensitivity band cheap. Sweeping `K` over a declared range needs no
    further autodiff — every gradient in the band is `K u + v` — and it is what `best_K` minimises
    over in closed form.
    """
    g = jax.grad(E, argnums=0)
    g1 = np.asarray(g(x, 1.0))
    g0 = np.asarray(g(x, 0.0))
    return g1 - g0, g0


def r_eq(grad: NDArray, patch: ClampedPatch) -> float:
    """Force residual per free degree of freedom: `||grad E||₂ / sqrt(2 N_free)`, free rows only.

    Clamped vertices are excluded, and the exclusion is the point. A clamped vertex carries a
    reaction force from the environment that was never imaged; that force is not a residual, and
    including it would report how hard the clamp is pulling rather than how well the mechanics
    balances. It would also *improve* as the clamp stiffened, which is the wrong direction.

    Normalising by `sqrt(2 N_free)` makes the number comparable between patches of different size.
    """
    gf = grad[patch.free_idx]
    return float(np.linalg.norm(gf) / np.sqrt(2 * patch.n_free))


def best_K(u: NDArray, v: NDArray, patch: ClampedPatch) -> tuple[float, float]:
    """Closed-form `argmin_K ||(K u + v)_free||₂`, and the residual there.

    **A diagnostic, not an inference.** The tension field is held at its declared value, so this is
    the area stiffness that best balances an *assumed* tension field — not a fitted `K`. It answers
    "is there any `K` at which this configuration is close to balanced", which is a useful check on
    a declared band, and nothing more.

    Uses the exact affine decomposition from `split_gradient`, so no search is involved.

    Returns:
        `(K, residual)`. `K` is `nan` when `u` vanishes on the free rows, i.e. when the area term
        exerts no force there and no `K` changes anything.
    """
    uf, vf = u[patch.free_idx].ravel(), v[patch.free_idx].ravel()
    den = float(uf @ uf)
    if den <= 0:
        return float("nan"), float(np.linalg.norm(vf) / np.sqrt(2 * patch.n_free))
    K = -float(uf @ vf) / den
    return K, float(np.linalg.norm(K * uf + vf) / np.sqrt(2 * patch.n_free))


def relax(E, patch: ClampedPatch, K: float, maxiter: int = 30000, gtol: float = 1e-12,
          E_exact=None) -> dict:  # noqa: ANN001
    """Minimise `E` over the FREE vertices only, holding clamped vertices at their observed places.

    The clamped coordinates never enter the optimisation vector, so the reaction forces at the
    boundary are simply not modelled — which is the honest treatment when the material on the other
    side was not imaged.

    Args:
        E: energy `E(x, K)`; may be smoothed via `eps > 0` in `patch_geometry`.
        patch: the patch to relax.
        K: area stiffness.
        maxiter: L-BFGS-B iteration cap.
        gtol: L-BFGS-B gradient tolerance.
        E_exact: unsmoothed energy used to report the residual. When `E` is smoothed, the residual
            must be re-evaluated without smoothing or it reports the smoothed problem's balance.

    Returns:
        A dict of the converged state and its diagnostics. Beyond the obvious keys:

        * `r_eq` — residual under `E_exact`; `r_eq_smoothed` — under `E`; `r_eq_obs` — at the
          **observed** configuration, i.e. before relaxing, which is the force-inference residual.
        * `min_area` is **signed**: a negative value means a face has inverted.
        * `r_eq_excl_T1`, `n_free_at_T1`, `n_free_kept` — see the note in the body on blocked T1s.
    """
    from scipy.optimize import minimize

    x0 = jnp.asarray(patch.x0)
    fi = jnp.asarray(patch.free_idx)
    n = len(patch.free_idx)

    def unpack(xf):  # noqa: ANN001, ANN202
        """Scatter the free coordinates back into the full vertex array."""
        return x0.at[fi].set(xf.reshape(n, 2))

    @jax.jit
    def vg(xf):  # noqa: ANN001, ANN202
        val, g = jax.value_and_grad(lambda z: E(unpack(z), K))(xf)
        return val, g

    z0 = np.asarray(patch.x0[patch.free_idx]).ravel()
    res = minimize(lambda z: [np.asarray(t, np.float64) for t in vg(jnp.asarray(z))],
                   z0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter,
                            "ftol": 1e-18, "gtol": gtol, "maxcor": 30})
    xz = np.asarray(unpack(jnp.asarray(res.x)))
    A, P, L = (np.asarray(t) for t in patch_geometry(jnp.asarray(xz), patch))
    Ex = E if E_exact is None else E_exact
    grad = np.asarray(jax.grad(Ex, argnums=0)(jnp.asarray(xz), K))
    grad_sm = np.asarray(jax.grad(E, argnums=0)(jnp.asarray(xz), K))
    dx = np.linalg.norm(xz - patch.x0, axis=1)

    # A tension edge that has shrunk to zero length is a T1 the fixed-topology relaxation cannot
    # perform. Its force does NOT vanish with its length -- d(gamma*l)/dx has modulus gamma for any
    # l > 0 -- so it leaves an irreducible O(gamma) residual at both endpoints. Report it
    # separately, or a handful of blocked T1s is misread as non-convergence of everything else.
    short = L < T1_LEN
    at_t1 = np.zeros(len(xz), dtype=bool)
    at_t1[patch.EA[short]] = True
    at_t1[patch.EB[short]] = True
    keep = np.array([i for i in patch.free_idx if not at_t1[i]], dtype=np.int64)
    r_excl = (float(np.linalg.norm(grad[keep]) / np.sqrt(2 * len(keep))) if len(keep)
              else float("nan"))
    return {
        "K": K, "success": bool(res.success), "nit": int(res.nit), "message": str(res.message),
        "E": float(res.fun), "r_eq": r_eq(grad, patch), "r_eq_smoothed": r_eq(grad_sm, patch),
        "r_eq_obs": r_eq(np.asarray(jax.grad(E, argnums=0)(x0, K)), patch),
        "min_area": float(A.min()), "min_area_interior": float(A[patch.face_is_interior].min()),
        "n_inverted": int((A <= 0).sum()), "n_area_lt_10pct": int((A < 0.1).sum()),
        "n_cells_collapsed": int((A < 0.05).sum()), "n_cells": int(len(A)),
        "min_edge_len": float(L.min()), "n_edge_lt_1e3": int(short.sum()),
        "n_free_at_T1": int(at_t1[patch.free_idx].sum()), "r_eq_excl_T1": r_excl,
        "n_free_kept": int(len(keep)),
        "max_dx": float(dx.max()), "rms_dx": float(np.sqrt((dx ** 2).mean())),
        "area_std": float(A.std(ddof=1)), "total_area": float(A.sum()),
        "nan": bool(not np.isfinite(res.fun) or not np.isfinite(xz).all()),
        "x": xz,
    }


def newton_polish(E, patch: ClampedPatch, K: float, x_start: NDArray,
                  iters: int = 60, tol: float = 1e-13,
                  return_info: bool = False) -> NDArray:
    """Levenberg-Marquardt on the force residual, over the free coordinates only.

    L-BFGS stalls on this problem at `||grad||/sqrt(2N) ~ 1e-5`, and extending the smoothing
    continuation does not help: the floor is the quasi-Newton model rather than `eps` (ladders of
    4, 6 and 8 smoothing steps all bottom out between `9e-6` and `1e-5`). The free block is small
    — at most `86 x 86` on the patches this was built for — so the exact Hessian is cheap and a
    damped Newton step on `grad E = 0` drives the residual to machine precision.

    Levenberg-Marquardt rather than plain Newton because the Hessian is indefinite wherever the
    configuration sits near a saddle, which is common when cells are close to collapsing.

    A solve that exhausts `iters` without reaching `tol` is otherwise indistinguishable, in the
    returned array alone, from one that converged -- there is no separate signal that the loop
    ran out rather than succeeded. `return_info=True` exposes that signal; it does not change
    the array returned, and when `return_info` is False (the default) behaviour is bit-identical
    to before this argument existed.

    Args:
        return_info: when True, also return a dict of convergence evidence.

    Returns:
        The polished full vertex array; clamped vertices are unchanged. When `return_info` is
        True, returns `(x, info)` with `info` carrying `grad_norm` (achieved `||grad||`),
        `iterations` (Newton steps actually taken), `tol` and `converged`
        (`grad_norm < tol * max(1, n_free)`).
    """
    fi = jnp.asarray(patch.free_idx)
    n = len(patch.free_idx)
    x0 = jnp.asarray(x_start)

    def f(z):  # noqa: ANN001, ANN202
        return E(x0.at[fi].set(z.reshape(n, 2)), K)

    g_fn = jax.jit(jax.grad(f))
    h_fn = jax.jit(jax.hessian(f))
    z = jnp.asarray(np.asarray(x_start[patch.free_idx]).ravel())
    lam = 1e-10
    gn = float(jnp.linalg.norm(g_fn(z)))
    n_iter = 0
    for _ in range(iters):
        if gn < tol * max(1.0, n):
            break
        n_iter += 1
        g = np.asarray(g_fn(z), np.float64)
        H = np.asarray(h_fn(z), np.float64)
        A = H.T @ H + lam * np.eye(2 * n)
        try:
            step = np.linalg.solve(A, -(H.T @ g))
        except np.linalg.LinAlgError:
            lam *= 100
            continue
        accepted = False
        for _ls in range(40):
            zt = z + jnp.asarray(step)
            gt = float(jnp.linalg.norm(g_fn(zt)))
            if np.isfinite(gt) and gt < gn:
                z, gn, lam, accepted = zt, gt, max(lam * 0.3, 1e-14), True
                break
            # Reject and halve: an LM step that does not reduce the residual is outside the
            # trust region, and the damping is raised only once backtracking has failed.
            step = step * 0.5
        if not accepted:
            lam *= 30
            if lam > 1e6:
                break
    x_out = np.asarray(x0.at[fi].set(z.reshape(n, 2)))
    if not return_info:
        return x_out
    info = {"grad_norm": gn, "iterations": n_iter, "tol": tol,
            "converged": bool(gn < tol * max(1.0, n))}
    return x_out, info


def clamped_edges_are_inert(mesh_npz: str | Path) -> dict:
    """Check that edges bordering unimaged space exert no force on any free vertex.

    This module drops the tension term on cell-background edges entirely. That is only legitimate
    if those edges cannot influence the force balance where it is written — and they cannot,
    because both endpoints of such an edge are clamped by construction. The claim is checkable, so
    it should be checked rather than assumed.

    Builds the tension energy twice, over all edges and over cell-cell edges only, and compares the
    two gradients on the free components. They must agree to machine precision.

    Returns:
        `max_abs_diff_free` (expected ~1e-16), `rel` relative to the cell-cell gradient scale, and
        `max_abs_diff_clamped`, which is expected to be LARGE — the clamped rows are exactly where
        the dropped term does act, and it is not a residual there because no equation is written.
    """
    z = np.load(mesh_npz, allow_pickle=True)
    a_ref = float(z["a_ref_px"])
    x = jnp.asarray(np.asarray(z["verts_px"], np.float64) / np.sqrt(a_ref))
    e_all = jnp.asarray(np.asarray(z["edges"]))
    cc = np.asarray(z["edge_is_cellcell"], bool)
    free = np.nonzero(np.asarray(z["free"], bool))[0]

    def tens(edges):  # noqa: ANN001, ANN202
        """Unit-tension energy over the given edge set: sum of edge lengths."""
        return lambda xx: jnp.sum(jnp.linalg.norm(xx[edges[:, 1]] - xx[edges[:, 0]], axis=1))

    g_all = np.asarray(jax.grad(tens(e_all))(x))
    g_cc = np.asarray(jax.grad(tens(jnp.asarray(np.asarray(z["edges"])[cc])))(x))
    d = np.abs(g_all[free] - g_cc[free]).max()
    return {"max_abs_diff_free": float(d),
            "rel": float(d / max(np.abs(g_cc[free]).max(), 1e-300)),
            "max_abs_diff_clamped": float(np.abs(g_all - g_cc).max())}
