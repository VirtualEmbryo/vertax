"""
Physical model of the mitotic spindle and equilibrium solver.

Each centrosome emits microtubules in all directions θ ∈ [0, 2π] with an
angular density Λ(θ) that combines local membrane tension and cell polarity:

    Λ(θ) = tension_j · (1 + p · cos²(θ − φ))

where φ is the cell polarity axis and p ∈ [0, 1] the polarity strength.
For p = 0 the model reduces to the original uniform-density case.

The microtubule force is:

    dF(θ) = Λ(θ) · L(θ) · ê(θ) · dθ

--- Method 1 : Newton-Raphson (method='newton') ---

Spindle equilibrium (c, φ) is the condition:

    F₁ + F₂ = 0   (force balance, 2 scalar equations)
    M₁ + M₂ = 0   (moment balance, 1 scalar equation)

solved by Newton-Raphson with exact Jacobian (JAX automatic differentiation).

--- Method 2 : Probability density (method='pdf' or 'pdf_sample') ---

Following Thery et al., the orientational potential is:

    V̄(α) = −∫₀^α T̄(α') dα'

where T̄(α) is the normalised net torque at spindle orientation α (centrosome
centre fixed at the polygon centroid).

The probability density over orientations α ∈ [0, π] is:

    p(α) = p₀ exp(−V̄(α) / C)

with C a noise parameter (larger C → flatter distribution).

- 'pdf'        : uses the mode of p(α) as the division axis.
- 'pdf_sample' : draws the division axis at random from p(α).

Contains:
  - _ray_segment_length_jax   : microtubule length to a segment
  - _centrosome_force_jax     : force and moment of one centrosome (Gauss quadrature)
  - _torque_at_angle          : net moment M₁ + M₂ at a fixed spindle orientation α
  - spindle_orientation_pdf   : compute p(α) curve over α ∈ [0, π]
  - _make_residual             : builds the residual vector R(cx, cy, φ)
  - _newton                    : Newton-Raphson solver
  - find_spindle               : public API — finds (c, u, c1, c2)
                                 method ∈ {'newton', 'pdf', 'pdf_sample'}
"""

import functools

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


@functools.partial(jax.jit, static_argnames=())
def _centrosome_force_jax(centrosome: jax.Array,
                          v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
                          gauss_xi: jax.Array, gauss_wi: jax.Array,
                          polarity_phi: jax.Array,
                          polarity_strength: jax.Array,
                          beta: jax.Array,
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
            #   dF = Λ · Lᵝ · ê · w · (Δα/2)
            #   dM = (r × dF) = (contact × dF), 2-D scalar cross product
            # The exponent β tunes how strongly microtubule length biases the pull:
            # β=1 is linear, β>1 makes long microtubules dominate (more eccentric).
            polarity_bias = 1.0 + polarity_strength * jnp.cos(theta - polarity_phi) ** 2
            dF  = tension * polarity_bias * (L ** beta) * direction * gauss_wi[g_idx] * half_da
            dM  = contact[0] * dF[1] - contact[1] * dF[0]
            valid = jnp.isfinite(L)   # ignore directions with no intersection
            return jnp.where(valid, dF, jnp.zeros(2)), jnp.where(valid, dM, 0.0)

        dFs, dMs = jax.vmap(force_one_gauss)(jnp.arange(len(gauss_xi)))
        return dFs.sum(axis=0), dMs.sum()

    Fs, Ms = jax.vmap(force_one_segment)(jnp.arange(len(tensions)))
    return Fs.sum(axis=0), Ms.sum()


def _torque_at_angle(alpha: float,
                     centroid: np.ndarray,
                     v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
                     d: float,
                     gauss_xi: jax.Array, gauss_wi: jax.Array,
                     polarity_phi: float = 0.0,
                     polarity_strength: float = 0.0,
                     beta: float = 3.0) -> float:
    """Net torque M₁ + M₂ for a spindle centred at centroid with orientation α.

    The spindle axis is u = (cos α, sin α).  The two centrosomes are placed at
    c ± d·u.  Only the moment balance is evaluated (torque about the origin).
    """
    c  = jnp.array(centroid, dtype=jnp.float64)
    u  = jnp.array([jnp.cos(alpha), jnp.sin(alpha)], dtype=jnp.float64)
    d_jax = jnp.float64(d)
    pp = jnp.float64(polarity_phi)
    ps = jnp.float64(polarity_strength)
    bb = jnp.float64(beta)
    _, M1 = _centrosome_force_jax(c + d_jax * u, v0s, v1s, tensions, gauss_xi, gauss_wi, pp, ps, bb)
    _, M2 = _centrosome_force_jax(c - d_jax * u, v0s, v1s, tensions, gauss_xi, gauss_wi, pp, ps, bb)
    return float(M1 + M2)


def spindle_orientation_pdf(
        centroid: np.ndarray,
        v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
        d: float,
        C: float = 0.1,
        n_alpha: int = 180,
        n_gauss: int = 12,
        polarity_phi: float = 0.0,
        polarity_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Boltzmann-like probability density p(α) over α ∈ [0, π].

    Following Thery et al., the orientational potential is:

        V̄(α) = −∫₀^α T̄(α') dα'

    where T̄(α') = T(α') / (r_N · F₀) is the normalised net torque.
    Since r_N and F₀ are scale factors shared by all angles, normalisation of
    the torque cancels in the exponential and we can use the raw torque T(α)
    directly:

        V(α) = −∫₀^α T(α') dα'
        p(α) ∝ exp(−V(α) / C)

    The integral is approximated by the trapezoidal rule on a uniform grid of
    n_alpha points in [0, π].

    Parameters
    ----------
    centroid  : spindle centre (2,)
    v0s, v1s  : segment endpoints (N, 2)
    tensions  : membrane tensions (N,)
    d         : centrosome half-separation
    C         : noise parameter (default 0.1); larger C → flatter distribution
    n_alpha   : number of orientation samples in [0, π]
    n_gauss   : Gauss-Legendre quadrature order for _centrosome_force_jax

    Returns
    -------
    alphas : (n_alpha,) angles in [0, π]
    p      : (n_alpha,) normalised probability density
    """
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    gauss_xi = jnp.array(xi, dtype=jnp.float64)
    gauss_wi = jnp.array(wi, dtype=jnp.float64)

    alphas = np.linspace(0.0, np.pi, n_alpha)
    torques = np.array([
        _torque_at_angle(a, centroid, v0s, v1s, tensions, d, gauss_xi, gauss_wi,
                         polarity_phi, polarity_strength)
        for a in alphas
    ])

    # Potential V(α) = −∫₀^α T(α') dα'  (trapezoidal rule)
    V = -np.cumsum(np.concatenate([[0.0], 0.5 * (torques[:-1] + torques[1:]) * np.diff(alphas)]))

    # Boltzmann distribution; subtract V.min() for numerical stability
    log_p = -(V - V.min()) / C
    p = np.exp(log_p)
    p /= np.trapz(p, alphas)   # normalise so ∫p dα = 1
    return alphas, p


# =====================================================================
#  Théry et al. 2007 model (Nature 447:493, Supplementary Eqs. S1–S15)
# =====================================================================

def _thery_cortex(centroid, v0s, v1s, tensions, n_psi):
    """Sample the cell cortex at uniform cortical angles ψ ∈ [-π, π).

    For each ψ, cast a ray from the cell centre and find the first boundary
    segment hit. Returns, per ψ:
      R    : cortical position R⃗(ψ) relative to the centre   (n_psi, 2)
      Fpsi : cortical force magnitude F(ψ); here the tension σ of the segment
             reached at ψ (the tissue cue, Eq. S7 with F ∝ cortical cue)  (n_psi,)
    Points with no hit get R=0 and F=0 so they drop out of the torque integral.
    """
    c = np.asarray(centroid, dtype=float)
    v0 = np.asarray(v0s, dtype=float); v1 = np.asarray(v1s, dtype=float)
    tens = np.asarray(tensions, dtype=float)
    psis = np.linspace(-np.pi, np.pi, n_psi, endpoint=False)
    R = np.zeros((n_psi, 2)); Fpsi = np.zeros(n_psi)
    cj = jnp.asarray(c)
    for i, psi in enumerate(psis):
        dirn = jnp.array([np.cos(psi), np.sin(psi)])
        Ls = np.array(jax.vmap(
            lambda a, b: _ray_segment_length_jax(cj, dirn, a, b)
        )(jnp.asarray(v0), jnp.asarray(v1)))
        j = int(np.argmin(Ls)); L = Ls[j]
        if np.isfinite(L):
            R[i] = L * np.array([np.cos(psi), np.sin(psi)])
            Fpsi[i] = tens[j]                       # F(ψ) = C·σ(ψ), C folded into d
    return psis, R, Fpsi


def thery_orientation_pdf(
        centroid: np.ndarray,
        v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
        a: float,
        d: float = 1.0,
        n_phi: int = 180,
        n_psi: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Spindle-orientation density P(φ) of Théry et al. 2007 (Eqs. S1–S15).

    Implements the exact statistical machinery of the supplement, with the
    cortical force magnitude F(ψ) taken as the junctional tension of the vertex
    model (a tissue-appropriate instance of their cortical cue, Eq. S7):

      - cortical point R⃗(ψ) on the real cell boundary                      (S1)
      - astral microtubule direction m⃗(ψ,φ): unit vector from the spindle
        pole towards R⃗(ψ)                                                  (S2)
      - angular MT density ρ_MT(θ)=(N_MT/2π)·dα/dθ with θ=ψ−φ, α the
        emission angle at the pole; each pole emits only towards the half of
        the cortex it faces                                                (S4)
      - cortical force per angular element f⃗(ψ,φ)=F(ψ) ρ_MT(θ) m⃗(ψ,φ)      (S3)
      - net torque τ_z(φ)=∫(R_x f_y − R_y f_x) dψ summed over both poles    (S5)
      - potential W(φ)=−∫ τ_z dφ′                                          (S6)
      - stationary distribution P(φ)=N exp(−W(φ)/d)                        (S12/S14)

    Parameters
    ----------
    a       : spindle half-separation (poles at centre ± a·u(φ)); ratio a/R in S13
    d       : dimensionless noise parameter (Eq. S15); larger d → flatter P(φ)
    n_phi   : grid size for the spindle orientation φ ∈ [0, π]
    n_psi   : grid size for the cortical angle ψ used in the torque integral

    Returns (phis, P) with phis ∈ [0, π] and P normalised to unit integral.
    """
    c = np.asarray(centroid, dtype=float)
    psis, R, Fpsi = _thery_cortex(c, v0s, v1s, tensions, n_psi)
    dpsi = 2 * np.pi / n_psi
    valid = Fpsi > 0.0   # cortical samples with an actual membrane hit

    def torque(phi):
        """Net torque τ_z(φ) summed over the two poles (Eqs. S3–S5)."""
        u = np.array([np.cos(phi), np.sin(phi)])
        tau = 0.0
        for sgn in (+1.0, -1.0):
            pole = sgn * a * u                       # pole position rel. to centre
            mvec = R - pole                          # pole → cortex point
            nrm = np.linalg.norm(mvec, axis=1)
            mhat = mvec / np.where(nrm[:, None] > 1e-12, nrm[:, None], 1.0)

            # Emission angle at the pole, α(ψ) = atan2(R⃗(ψ) − pole); ρ_MT ∝ |dα/dψ|
            # (S4). Unwrap so the numerical derivative is well-defined on the ring.
            alpha = np.unwrap(np.arctan2(mvec[:, 1], mvec[:, 0]))
            dalpha = np.gradient(alpha, psis)        # dα/dψ
            rho_mt = np.abs(dalpha)                  # (N_MT/2π factor folds into d)

            # Each pole emits only towards the cortex half it faces (S4, finite
            # angular span): keep points whose microtubule direction m⃗ points
            # outward, i.e. aligns with the outward radial unit vector R̂(ψ).
            Rn = np.linalg.norm(R, axis=1)
            Rhat = R / np.where(Rn[:, None] > 1e-12, Rn[:, None], 1.0)
            facing = np.sum(mhat * Rhat, axis=1) >= 0.0
            mask = valid & facing

            f = (Fpsi * rho_mt)[:, None] * mhat      # f⃗ = F(ψ)·ρ_MT(θ)·m⃗  (S3)
            tau_psi = R[:, 0] * f[:, 1] - R[:, 1] * f[:, 0]   # R × f  (S5)
            tau += np.sum(np.where(mask, tau_psi, 0.0)) * dpsi
        return tau

    phis = np.linspace(0.0, np.pi, n_phi)
    taus = np.array([torque(p) for p in phis])
    # W(φ) = −∫ τ dφ′  (cumulative trapezoid)
    W = -np.cumsum(np.concatenate([[0.0], 0.5 * (taus[:-1] + taus[1:]) * np.diff(phis)]))
    logP = -(W - W.min()) / d
    P = np.exp(logP)
    P /= np.trapz(P, phis)
    return phis, P


def _make_residual(v0s: jax.Array, v1s: jax.Array, tensions: jax.Array,
                   d: float, n_gauss: int = 12,
                   polarity_phi: float = 0.0, polarity_strength: float = 0.0,
                   beta: float = 3.0):
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
    pp = jnp.float64(polarity_phi)
    ps = jnp.float64(polarity_strength)
    bb = jnp.float64(beta)

    @jax.jit
    def residual(params: jax.Array) -> jax.Array:
        cx, cy, phi = params[0], params[1], params[2]
        c  = jnp.array([cx, cy])
        u  = jnp.array([jnp.cos(phi), jnp.sin(phi)])
        F1, M1 = _centrosome_force_jax(c + d * u, v0s, v1s, tensions, gauss_xi, gauss_wi, pp, ps, bb)
        F2, M2 = _centrosome_force_jax(c - d * u, v0s, v1s, tensions, gauss_xi, gauss_wi, pp, ps, bb)
        return jnp.array([F1[0] + F2[0], F1[1] + F2[1], M1 + M2])

    return residual


def _newton(residual, x0: np.ndarray, max_iter: int = 30,
            tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """Solve residual(x) = 0 with MINPACK's hybrid (modified Powell) method.

    scipy.optimize.root(method='hybr') is a trust-region dogleg solver that uses
    the analytic Jacobian J = ∂R/∂x (computed by JAX autodiff) but, unlike plain
    Newton-Raphson, never takes an unbounded step — so it does not blow up to
    NaN/Inf on ill-conditioned cells. Falls back gracefully if it fails to
    converge (caller checks the residual norm).

    Returns the solution vector and the final residual norm.
    """
    from scipy.optimize import root

    res_np = lambda x: np.asarray(residual(jnp.asarray(x, dtype=jnp.float64)), dtype=float)
    jac_fn = jax.jit(jax.jacobian(residual))
    jac_np = lambda x: np.asarray(jac_fn(jnp.asarray(x, dtype=jnp.float64)), dtype=float)

    sol = root(res_np, np.asarray(x0, dtype=float), jac=jac_np,
               method='hybr', tol=tol, options={'maxfev': max_iter * 10})
    x = sol.x
    return x, float(np.linalg.norm(res_np(x)))


def _newton_history(residual, x0: np.ndarray, poly: np.ndarray,
                    max_iter: int = 100, tol: float = 1e-9) -> list[np.ndarray]:
    """Nonlinear conjugate gradient (Polak-Ribière) minimising f = ½‖R(x)‖².

    ∇f = Jᵀ R.  Steps that move the centre outside `poly` are clipped to the
    boundary along the search direction, guaranteeing the centre stays inside.

    The first entry is x0, the last is the converged solution. At least 2 entries.
    """
    from matplotlib.path import Path
    poly_path = Path(poly)

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda x: 0.5 * jnp.dot(residual(x), residual(x))
    ))

    x = np.array(x0, dtype=float)
    history = [x.copy()]

    f, g = loss_and_grad(jnp.array(x))
    f, g = float(f), np.array(g)
    d = -g  # initial search direction

    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

        # Backtracking line search along d, clipped to stay inside poly.
        alpha = 1.0
        for _ in range(30):
            x_new = x + alpha * d
            if poly_path.contains_point(x_new[:2]):
                f_new, g_new = loss_and_grad(jnp.array(x_new))
                f_new, g_new = float(f_new), np.array(g_new)
                if f_new < f:
                    break
            alpha *= 0.5
        else:
            break  # line search failed, converged as far as possible

        x, f, g_old = x_new, f_new, g
        g = g_new
        history.append(x.copy())

        # Polak-Ribière beta, reset to steepest descent if negative.
        beta = max(0.0, float(np.dot(g, g - g_old) / (np.dot(g_old, g_old) + 1e-300)))
        d = -g + beta * d

    if len(history) == 1:
        history.append(history[0])
    return history


def find_spindle(face_id: int, mesh: BoundedMesh,
                 d_fraction: float = 0.3,
                 n_gauss: int = 12,
                 method: str = 'newton',
                 pdf_C: float = 0.1,
                 pdf_n_alpha: int = 180,
                 rng: np.random.Generator | None = None,
                 polarity_phi: float = 0.0,
                 polarity_strength: float = 0.0,
                 beta: float = 3.0,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the equilibrium position of the mitotic spindle in cell face_id.

    Parameters
    ----------
    face_id           : face index in the mesh
    mesh              : BoundedMesh
    d_fraction        : centrosome half-separation as a fraction of the mean cell radius
    n_gauss           : Gauss-Legendre quadrature order
    method            : one of
                          'newton'     — Newton-Raphson on F₁+F₂=0, M₁+M₂=0 (default)
                          'pdf'        — mode of p(α) = p₀ exp(−V̄(α)/C)
                          'pdf_sample' — random draw from p(α) = p₀ exp(−V̄(α)/C)
    pdf_C             : noise parameter C for the PDF methods (default 0.1)
    pdf_n_alpha       : number of orientation samples for PDF methods (default 180)
    rng               : numpy random generator for 'pdf_sample' (uses default if None)
    polarity_phi      : cell polarity axis angle in radians (default 0 = no effect)
    polarity_strength : polarity bias strength p ∈ [0, 1]; 0 = geometry-only,
                        1 = polarity doubles the microtubule density along φ
                        without suppressing it in other directions (default 0)

    The angular density of microtubules is modulated as:
        Λ(θ) = tension_j · (1 + polarity_strength · cos²(θ − polarity_phi))

    The spindle centre is always fixed at the polygon centroid.
    In 'newton' mode, the centre can shift from the centroid during Newton
    iterations (the full 3-DOF problem is solved); in 'pdf' and 'pdf_sample'
    modes the centre is kept at the centroid.

    Falls back to (centroid, inertia axis) if the chosen method fails.

    Returns
    -------
    c  : spindle centre (2,)
    u  : spindle axis, unit vector (2,)
    c1 : centrosome 1 position = c + d·u (2,)
    c2 : centrosome 2 position = c − d·u (2,)
    """
    poly = get_face_polygon(face_id, mesh)

    if np.any(np.isnan(poly)) or len(poly) < 3:
        raise ValueError(f"Cell {face_id} has a degenerate polygon (NaN or < 3 vertices).")

    v0s, v1s, tensions = get_cell_segments(face_id, mesh)

    c0   = polygon_centroid(poly)
    if np.any(np.isnan(c0)):
        raise ValueError(f"Cell {face_id} has zero area (degenerate polygon).")
    u0   = _inertia_axis(poly, c0)
    phi0 = np.arctan2(u0[1], u0[0])
    d    = float(d_fraction * np.mean(np.linalg.norm(poly - c0, axis=1)))

    from matplotlib.path import Path
    poly_path = Path(poly)

    if method == 'newton':
        residual = _make_residual(v0s, v1s, tensions, d, n_gauss,
                                  polarity_phi, polarity_strength, beta)
        params, res_norm = _newton(residual, np.array([c0[0], c0[1], phi0]))

        fallback = False
        fallback_reason = ""
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            fallback = True
            fallback_reason = "Newton diverged (NaN/Inf)"
        elif res_norm > 1e-6:
            # MINPACK stopped at a non-equilibrium point (|R| still large); the
            # returned centre is meaningless and may sit in a corner. Fall back.
            fallback = True
            fallback_reason = f"not converged (|R|={res_norm:.1e})"
        else:
            cx, cy, phi = params
            c = np.array([cx, cy])
            if not poly_path.contains_point(c):
                fallback = True
                fallback_reason = "spindle centre outside the cell"

        if fallback:
            c   = c0
            phi = phi0
            print(f"Cell {face_id} — spindle (fallback to geometric centroid: {fallback_reason})")
        else:
            print(f"Cell {face_id} — spindle (newton)")

        if not fallback:
            print(f"  Residual  : {res_norm:.2e}")

    elif method in ('pdf', 'pdf_sample'):
        try:
            # Théry et al. 2007 orientation density (Eqs. S1–S15), cortical force
            # F(ψ) taken as junctional tension. pdf_C plays the role of the
            # dimensionless noise d; the half-separation a is the spindle d.
            alphas, p = thery_orientation_pdf(
                c0, v0s, v1s, tensions, a=d,
                d=pdf_C, n_phi=pdf_n_alpha, n_psi=2 * pdf_n_alpha,
            )
            if method == 'pdf':
                phi = float(alphas[np.argmax(p)])
                print(f"Cell {face_id} — spindle (Théry mode, α={np.degrees(phi):.1f}°)")
            else:
                rng = rng or np.random.default_rng()
                p_norm = p / p.sum()
                phi = float(rng.choice(alphas, p=p_norm))
                print(f"Cell {face_id} — spindle (Théry sample, α={np.degrees(phi):.1f}°)")
        except Exception as e:
            phi = phi0
            print(f"Cell {face_id} — spindle (Théry fallback: {e})")
        c = c0
        fallback = False   # centre stays at centroid, no additional check needed

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'newton', 'pdf', or 'pdf_sample'.")

    cx, cy = float(c[0]), float(c[1])
    u  = np.array([np.cos(phi), np.sin(phi)])
    c1, c2 = c + d * u, c - d * u

    print(f"  Center    : ({cx:.4f}, {cy:.4f})")
    print(f"  Axis u    : ({u[0]:.4f}, {u[1]:.4f})")
    return c, u, c1, c2


def eval_residual(face_id: int, mesh: BoundedMesh,
                  c: np.ndarray, alpha: float,
                  d_fraction: float = 0.3, n_gauss: int = 12,
                  polarity_phi: float = 0.0, polarity_strength: float = 0.0,
                  ) -> np.ndarray:
    """Evaluate R(c, α) = [F₁ₓ+F₂ₓ, F₁ᵧ+F₂ᵧ, M₁+M₂] for the given spindle state.

    Returns a (3,) array [Fx, Fy, M].
    """
    v0s, v1s, tensions = get_cell_segments(face_id, mesh)
    poly = get_face_polygon(face_id, mesh)
    c0 = polygon_centroid(poly)
    d = float(d_fraction * np.mean(np.linalg.norm(poly - c0, axis=1)))
    residual = _make_residual(v0s, v1s, tensions, d, n_gauss, polarity_phi, polarity_strength)
    params = jnp.array([c[0], c[1], alpha], dtype=jnp.float64)
    return np.array(residual(params), dtype=float)
