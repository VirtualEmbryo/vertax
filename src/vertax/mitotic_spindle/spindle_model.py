"""
Physical model of the mitotic spindle and equilibrium solver.

Each centrosome emits microtubules uniformly over all angles (θ ∈ [0, 2π]).
A microtubule in direction ê(θ) exerts a force proportional to its length
L(θ) and the local membrane tension Λ(θ):

    dF(θ) = Λ(θ) · L(θ) · ê(θ) · dθ

Spindle equilibrium (c, φ) is the condition:

    F₁ + F₂ = 0   (force balance, 2 scalar equations)
    M₁ + M₂ = 0   (moment balance, 1 scalar equation)

solved by Newton-Raphson with exact Jacobian (JAX automatic differentiation).

Contains:
  - _ray_segment_length_jax : microtubule length to a segment
  - _centrosome_force_jax   : force and moment of one centrosome (Gauss quadrature)
  - _make_residual           : builds the residual vector R(cx, cy, φ)
  - _newton                  : Newton-Raphson solver
  - find_spindle             : public API — finds (c, u, c1, c2)
"""

import numpy as np
import jax
import jax.numpy as jnp

from vertax import BoundedMesh

from mesh_utils import get_face_polygon, get_cell_segments, polygon_centroid, _inertia_axis


@jax.jit
def _ray_segment_length_jax(origin: jax.Array, direction: jax.Array,
                             v0: jax.Array, v1: jax.Array) -> jax.Array:
    """Distance from origin to the intersection of the ray with segment [v0, v1].

    Solve the parametric system:
        origin + t · direction = v0 + s · (v1 - v0)

    Setting e = v1 - v0 and w = v0 - origin, Cramer's rule gives:
        denom = direction × e   (2-D cross product, scalar)
        t     = (w × e) / denom
        s     = (w × direction) / denom

    The intersection is valid if and only if:
        - |denom| > ε  (ray not parallel to segment)
        - t > 0        (intersection in the positive ray direction)
        - s ∈ [0, 1]   (intersection inside the segment)

    Branchless implementation (no if/else) for compatibility with @jax.jit
    and jax.vmap: invalid cases return jnp.inf.
    """
    e     = v1 - v0
    denom = direction[0] * e[1] - direction[1] * e[0]   # direction × e
    w     = v0 - origin
    t     = (w[0] * e[1] - w[1] * e[0]) / denom         # w × e / denom
    s     = (w[0] * direction[1] - w[1] * direction[0]) / denom  # w × dir / denom
    valid = (jnp.abs(denom) > 1e-12) & (t > 1e-9) & (s >= -1e-9) & (s <= 1.0 + 1e-9)
    return jnp.where(valid, t, jnp.inf)


@jax.jit
def _centrosome_force_jax(centrosome: jax.Array,
                          v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
                          gauss_xi: jax.Array, gauss_wi: jax.Array,
                          ) -> tuple[jax.Array, jax.Array]:
    """Compute the total force and moment exerted by one centrosome.

    The centrosome emits microtubules in all directions θ ∈ [0, 2π].
    The contribution of each membrane segment [v0ⱼ, v1ⱼ] is integrated over
    the angular arc [α0ⱼ, α1ⱼ] subtended by that segment as seen from the centrosome:

        α0ⱼ = atan2(v0ⱼ - centrosome)
        α1ⱼ = atan2(v1ⱼ - centrosome)
        Δαⱼ = α1ⱼ - α0ⱼ  (corrected to stay in ]-π, π])

    Elementary force from the microtubule in direction θ hitting segment j:
        dFⱼ(θ) = Λⱼ · L(θ) · ê(θ) · dθ

    where L(θ) = min_k dist(centrosome, segment_k in direction θ) is the
    microtubule length (distance to the first membrane hit).

    The integral over each segment is approximated by Gauss-Legendre quadrature
    with n points on [-1, 1] → [α0ⱼ, α1ⱼ] via the change of variable:
        θ = α0ⱼ + (1 + ξ)/2 · Δαⱼ,  dθ = Δαⱼ/2 · dξ

    Total force summed over all segments and Gauss points:
        F = Σⱼ Σₖ wₖ · (Δαⱼ/2) · Λⱼ · L(θⱼₖ) · ê(θⱼₖ)

    Moment about the origin:
        M = Σⱼ Σₖ wₖ · (Δαⱼ/2) · Λⱼ · L(θⱼₖ) · (r(θⱼₖ) × ê(θⱼₖ))

    where r(θ) = centrosome + L(θ)·ê(θ) is the contact point on the membrane,
    and r × ê = rx·êy − ry·êx is the 2-D cross product (scalar moment).
    """
    # Angles of each segment's endpoints as seen from the centrosome
    a0s = jnp.arctan2(v0s[:, 1] - centrosome[1], v0s[:, 0] - centrosome[0])
    a1s = jnp.arctan2(v1s[:, 1] - centrosome[1], v1s[:, 0] - centrosome[0])
    das = a1s - a0s
    # Discontinuity correction: keep Δα in ]-π, π] to avoid 2π jumps
    # when a segment crosses the atan2 branch cut
    das = jnp.where(das >  jnp.pi, das - 2 * jnp.pi, das)
    das = jnp.where(das < -jnp.pi, das + 2 * jnp.pi, das)

    # Pre-compute all Gauss angles: thetas[j, k] = θⱼₖ
    # Gauss-Legendre change of variable [-1,1] → [α0ⱼ, α1ⱼ]:
    #   θ = α0ⱼ + (1 + ξₖ)/2 · Δαⱼ
    thetas = a0s[:, None] + 0.5 * (1.0 + gauss_xi[None, :]) * das[:, None]

    def force_one_segment(seg_idx):
        """Force and moment from the angular arc of segment seg_idx."""
        tension  = tensions[seg_idx]
        half_da  = 0.5 * jnp.abs(das[seg_idx])  # Jacobian of the change of variable

        def force_one_gauss(g_idx):
            """Contribution of a single Gauss point."""
            theta     = thetas[seg_idx, g_idx]
            direction = jnp.array([jnp.cos(theta), jnp.sin(theta)])
            # L(θ) = distance to the nearest segment in direction θ
            # Cast the ray against all segments and take the minimum
            lengths = jax.vmap(
                lambda v0, v1: _ray_segment_length_jax(centrosome, direction, v0, v1)
            )(v0s, v1s)
            L = jnp.min(lengths)
            contact = centrosome + L * direction   # contact point on the membrane
            # Gauss-point contribution to force and moment:
            #   dF = Λ · L · ê · w · (Δα/2)
            #   dM = (r × dF) = (contact × dF), 2-D scalar cross product
            dF  = tension * L * direction * gauss_wi[g_idx] * half_da
            dM  = contact[0] * dF[1] - contact[1] * dF[0]
            valid = jnp.isfinite(L)   # ignore directions with no intersection
            return jnp.where(valid, dF, jnp.zeros(2)), jnp.where(valid, dM, 0.0)

        dFs, dMs = jax.vmap(force_one_gauss)(jnp.arange(len(gauss_xi)))
        return dFs.sum(axis=0), dMs.sum()

    Fs, Ms = jax.vmap(force_one_segment)(jnp.arange(len(tensions)))
    return Fs.sum(axis=0), Ms.sum()


def _make_residual(v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
                   d: float, n_gauss: int = 12):
    """Build the spindle equilibrium residual function (JIT-compiled).

    The residual is the 3-component vector we set to zero to find equilibrium:

        R(cx, cy, φ) = [F₁ₓ + F₂ₓ,
                        F₁ᵧ + F₂ᵧ,
                        M₁  + M₂ ]

    where (F₁, M₁) are the force and moment from C₁ = c + d·u,
    and (F₂, M₂) those from C₂ = c − d·u.

    Gauss-Legendre nodes and weights are computed once via
    np.polynomial.legendre.leggauss(n_gauss) and captured in the closure.
    """
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    gauss_xi = jnp.array(xi, dtype=jnp.float64)
    gauss_wi = jnp.array(wi, dtype=jnp.float64)

    @jax.jit
    def residual(params: jax.Array) -> jax.Array:
        cx, cy, phi = params[0], params[1], params[2]
        c  = jnp.array([cx, cy])
        u  = jnp.array([jnp.cos(phi), jnp.sin(phi)])
        F1, M1 = _centrosome_force_jax(c + d * u, v0s, v1s, tensions, gauss_xi, gauss_wi)
        F2, M2 = _centrosome_force_jax(c - d * u, v0s, v1s, tensions, gauss_xi, gauss_wi)
        return jnp.array([F1[0] + F2[0], F1[1] + F2[1], M1 + M2])

    return residual


def _newton(residual, x0: np.ndarray, max_iter: int = 30,
            tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """Solve residual(x) = 0 by Newton-Raphson.

    At each iteration, the Jacobian J = ∂R/∂x is computed by automatic
    differentiation (jax.jacobian), then the update is applied:

        x ← x − J⁻¹ · R(x)

    The Jacobian is a 3×3 matrix (3 equations, 3 unknowns: cx, cy, φ).
    The linear system J·Δx = R is solved by jnp.linalg.solve (LU factorisation).
    Convergence is reached when ‖R(x)‖₂ < tol.

    Returns the solution vector and the final residual norm.
    """
    jac_fn = jax.jit(jax.jacobian(residual))
    x = jnp.array(x0, dtype=jnp.float64)
    for _ in range(max_iter):
        r = residual(x)
        if float(jnp.linalg.norm(r)) < tol:
            break
        J = jac_fn(x)
        x = x - jnp.linalg.solve(J, r)
    return np.array(x), float(jnp.linalg.norm(residual(x)))


def find_spindle(face_id: int, mesh: BoundedMesh,
                 d_fraction: float = 0.3,
                 n_gauss: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the equilibrium position of the mitotic spindle in cell face_id.

    Initialisation:
      - Centre c₀  : polygon centroid (Green's formula).
      - Axis u₀    : principal inertia axis (direction of maximum elongation).
      - Half-length: d = d_fraction · mean_radius, where mean_radius is the
        mean distance from the centroid to the boundary vertices.

    Resolution:
      Newton-Raphson on the residual (F₁+F₂, M₁+M₂) until convergence.
      Falls back to (c₀, u₀) if Newton diverges or places the centre outside
      the cell polygon.

    Returns:
      c  : spindle centre (2,)
      u  : spindle axis, unit vector (2,)
      c1 : centrosome 1 position = c + d·u (2,)
      c2 : centrosome 2 position = c − d·u (2,)
    """
    poly        = get_face_polygon(face_id, mesh)

    if np.any(np.isnan(poly)) or len(poly) < 3:
        raise ValueError(f"Cell {face_id} has a degenerate polygon (NaN or < 3 vertices).")

    v0s, v1s, tensions = get_cell_segments(face_id, mesh)

    c0   = polygon_centroid(poly)
    if np.any(np.isnan(c0)):
        raise ValueError(f"Cell {face_id} has zero area (degenerate polygon).")
    u0   = _inertia_axis(poly, c0)
    phi0 = np.arctan2(u0[1], u0[0])
    # d = fraction du rayon moyen de la cellule, pour rester à l'intérieur
    d    = float(d_fraction * np.mean(np.linalg.norm(poly - c0, axis=1)))

    from matplotlib.path import Path
    poly_path = Path(poly)

    residual = _make_residual(v0s, v1s, tensions, d, n_gauss)
    params, res_norm = _newton(residual, np.array([c0[0], c0[1], phi0]))

    fallback = False
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        fallback = True
    else:
        cx, cy, phi = params
        c = np.array([cx, cy])
        if not poly_path.contains_point(c):
            fallback = True

    if fallback:
        # Newton diverged or placed the center outside the cell: fall back to
        # the geometric centroid with the inertia axis as the division direction.
        c   = c0
        phi = phi0
        print(f"Cell {face_id} — spindle (fallback to geometric centroid)")
    else:
        print(f"Cell {face_id} — spindle")

    cx, cy = float(c[0]), float(c[1])
    u  = np.array([np.cos(phi), np.sin(phi)])
    c1, c2 = c + d * u, c - d * u

    print(f"  Center    : ({cx:.4f}, {cy:.4f})")
    print(f"  Axis u    : ({u[0]:.4f}, {u[1]:.4f})")
    if not fallback:
        print(f"  Residual  : {res_norm:.2e}")
    return c, u, c1, c2
