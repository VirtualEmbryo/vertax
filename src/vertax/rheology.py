r"""Bulk-driven shear rheometry for periodic vertex-model meshes.

Measure a tissue's linear elastic response **without deforming the periodic box**, by driving the
bulk with a cosine body force and reading the amplitude of the relaxed displacement mode. This
module carries no optimizer coupling.

The method
----------
Apply a force along one axis, modulated sinusoidally along another,

$$
f_x(y) = a\cos(q y),\qquad q = q_n = 2\pi n/H,
$$

which puts $+a$ at $y=0$ and $-a$ at $y=H/2$: **two counter-driven bands per
box**, hence two shear slabs, one interior and one closed through the periodic boundary. The total
force vanishes by symmetry, so no wall and no reaction boundary is needed. This is the *periodic
perturbation method* of non-equilibrium molecular dynamics (Gosling 1973; Hess 2002; GROMACS
``cos-acceleration``; LAMMPS ``fix accelerate/cos``), not a new construction.

VertAX minimises an energy, so the drive is applied as the Lagrangian potential

$$
\mathcal E_{\rm drive}(\mathbf X;a) = -a\sum_i \hat c_i\,\Delta x_i ,
\qquad \hat c_i = \cos(q y^0_i) - \tfrac1N\textstyle\sum_j\cos(q y^0_j),
$$

with the weights frozen on the material points of the undriven reference state and
$\Delta x_i$ the **minimum-image** displacement from $x^0_i$.

Two details are load-bearing, and both are asserted rather than assumed (see `make_shear_mode`
and `min_image_delta_x`):

* **Mean removal is mandatory.** Without it the drive carries a net force along the polarisation
  axis, the uniform translation mode is unbounded below, and the inner minimisation runs away.
* **Minimum image is mandatory.** ``geo.update_pbc`` re-wraps any vertex leaving the box at *every*
  inner iteration, so a term linear in the raw coordinate jumps by $\pm a\hat c_i W$ whenever a
  vertex crosses the seam.

Because the observable is a linear functional of vertex positions with **no** parameter dependence,
$\partial_{\bm\theta}\mathcal C|_{\mathbf X}=0$ exactly. That is what makes the existing
centered-EP core correct for this problem unmodified. A $\bm\theta$-dependent outer cost would
instead need the implicit $\partial\mathbf X/\partial\bm\theta$ term at the equilibrium;
dropping it silently would corrupt the estimator, which is why this cost is kept purely positional.

Conventions and gotchas
-----------------------
* **Everything here assumes ``float64``.** Call `require_float64` in any driver. Enable x64
  before the first JAX array is created and ``PbcMesh.vertices`` is then built in ``float64`` too
  (meshes follow JAX's default float dtype); no cast after construction is needed.
* **Unsliced tables only.** ``opt._jit_minimize`` calls the inner loss as
  ``L_in(vt[selected_verts], ..., fp[selected_faces])`` while taking its *gradient* on the unsliced
  tables. The drive term indexes vertices positionally, so a partial ``selected_verts`` would
  silently mis-pair weights and vertices. `make_drive_energy` therefore raises on a row-count
  mismatch rather than broadcasting.
* **Report ``G``, optimise ``J``.** $G=\rho_v/(Jq^2)$ is hyperbolic in the stiff limit
  ($U\to0$); the compliance is linear in the vertex positions.
* **T1 transitions are enabled**, not frozen — physically because they are the fluidity mechanism,
  numerically because freezing them drives $\eta\lambda_{\max}$ to the SGD stability bound
  (measured directly). With T1 on, a
  compliance measured across an event is a **secant, possibly plastic** compliance, not a linear
  elastic constant. Use `topology_signature` / `t1_event_count` to stratify; never pool.

The drive is **linear in X and contributes nothing to the Hessian**, so ``a0`` cannot move the
stability bound. The EP nudge term is quadratic and adds a small positive rank-one piece, so
$\eta\lambda_{\max}$ must be checked at the nudged level too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from vertax.geo import get_area, get_length, get_perimeter

__all__ = [
    "ShearMode",
    "compliance",
    "connectivity_hash",
    "drive_drift",
    "face_areas",
    "face_perimeters",
    "grad_x_norm",
    "half_edge_vectors",
    "hessian_compliance",
    "hessian_lambda_max",
    "hessian_spectrum",
    "make_compliance_cost",
    "make_drive_energy",
    "make_shear_mode",
    "min_image_delta_x",
    "mode_amplitude",
    "modulus_from_compliance",
    "require_float64",
    "stress_profile_xy",
    "t1_event_count",
    "target_compliance",
    "top_mode_participation",
    "topology_signature",
    "virial_stress",
    "virial_stress_e1",
    "virial_stress_e2",
]

MAX_EDGES_IN_ANY_FACE = 20
"""Face-walk bound, matching ``energy.MAX_EDGES_IN_ANY_FACE``. The walk latches once it returns to
its starting half-edge (``geo._sum_edges``), so an over-estimate costs time, not correctness."""


def require_float64() -> None:
    """Raise unless JAX x64 is enabled. Standing rule: all Hessian/equilibrium work in ``float64``."""
    if jnp.zeros(1).dtype != jnp.float64:
        msg = (
            "rheology requires float64. Put `jax.config.update('jax_enable_x64', True)` before the "
            "first jax array is created; meshes built afterwards are float64 too."
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# The mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShearMode:
    r"""A single Fourier drive mode: which axis is modulated, which is driven, and the weights.

    ``mod_axis`` selects the modulation direction (0 = $\cos(qx)$, 1 = $\cos(qy)$) and
    ``pol_axis`` the polarisation of the force and of the observed displacement. The four
    combinations give three independent constants of an orthotropic 2D tissue:

    | modulation    | polarisation | character    | constant                      |
    | ------------- | ------------ | ------------ | ----------------------------- |
    | ``cos(qy)``   | ``x``        | transverse   | ``C66 = G``                   |
    | ``cos(qx)``   | ``y``        | transverse   | ``C66`` (consistency check)   |
    | ``cos(qy)``   | ``y``        | longitudinal | ``C22``                       |
    | ``cos(qx)``   | ``x``        | longitudinal | ``C11``                       |

    Attributes:
        c: Mean-removed weights $\hat c_i$, one per vertex, frozen on the reference state.
        r0: Reference vertex positions, shape ``(N, 2)``.
        q: Wavevector $2\pi n/L_{\rm mod}$.
        n: Mode index.
        width: Box dimensions.
        height: Box dimensions.
        mod_axis: See above.
        pol_axis: See above.
        rho_v: Vertex number density $N/(WH)$ — the force-balance calibration constant.
        c_sq: $\sum_i\hat c_i^2$.
        c_sq_ratio: $2\sum\hat c_i^2/N$. **Finite-size diagnostic, not a correction factor.**
            The continuum calibration $G=\rho_v/(Jq^2)$ follows from force balance
            ($f_x(y)=a\rho_v\cos qy$) and is the definition used here. Equating the *discrete*
            drive energy $-aU\sum\hat c_i^2$ to the continuum $-\tfrac12 FUWH$ instead
            gives $F=2a\sum\hat c_i^2/(WH)$, i.e. the same thing multiplied by this ratio. The
            two routes agree exactly only for a uniform vertex distribution, so a ratio far from 1
            measures the density fluctuation the continuum calibration neglects, at
            $O(N^{-1/2})$. Report it; do not silently divide by it.
    """

    c: Array
    r0: Array
    q: float
    n: int
    width: float
    height: float
    mod_axis: int
    pol_axis: int
    rho_v: float
    c_sq: float
    c_sq_ratio: float

    @property
    def pol_period(self) -> float:
        """Box length along the polarisation axis — the period for the minimum-image wrap."""
        return self.width if self.pol_axis == 0 else self.height

    @property
    def mod_period(self) -> float:
        """Box length along the modulation axis."""
        return self.width if self.mod_axis == 0 else self.height

    @property
    def label(self) -> str:
        """Short human-readable tag, e.g. ``n1_cosy_polx``."""
        return f"n{self.n}_cos{'xy'[self.mod_axis]}_pol{'xy'[self.pol_axis]}"


def make_shear_mode(
    vertices: Array,
    width: float,
    height: float,
    n: int = 1,
    mod_axis: int = 1,
    pol_axis: int = 0,
    *,
    mean_tol: float = 1e-12,
) -> ShearMode:
    r"""Build a drive mode from the **undriven reference configuration**.

    The weights are attached to material points, so ``vertices`` must be the *relaxed* $a=0$
    state, and the mode must be re-referenced whenever that state moves appreciably (the drift
    diagnostic is `drive_drift`).

    Args:
        vertices: Reference positions, shape ``(N, 2)`` — the relaxed ``a = 0`` state.
        width: Box width.
        height: Box height.
        n: Mode index; ``q = 2*pi*n / L_mod``.
        mod_axis: 0 for ``cos(qx)``, 1 for ``cos(qy)``.
        pol_axis: 0 to drive/observe ``x``, 1 to drive/observe ``y``.
        mean_tol: Tolerance on $|\sum_i\hat c_i|$.

    Raises:
        ValueError: If the mean-removal assertion fails, or the axes are not in ``{0, 1}``, or
            ``n < 1``.
    """
    require_float64()
    if mod_axis not in (0, 1) or pol_axis not in (0, 1):
        msg = f"mod_axis and pol_axis must be 0 or 1; got {mod_axis}, {pol_axis}"
        raise ValueError(msg)
    if n < 1:
        msg = f"mode index n must be >= 1 (n = 0 is a rigid translation); got {n}"
        raise ValueError(msg)
    r0 = jnp.asarray(vertices[:, :2], dtype=jnp.float64)
    n_vert = r0.shape[0]
    mod_period = width if mod_axis == 0 else height
    q = 2.0 * np.pi * n / mod_period
    raw = jnp.cos(q * r0[:, mod_axis])
    c = raw - jnp.mean(raw)
    resid = float(jnp.abs(jnp.sum(c)))
    if resid > mean_tol:
        msg = f"|sum c_i| = {resid:.3e} exceeds {mean_tol:.1e}: the drive carries a net force."
        raise ValueError(msg)
    c_sq = float(jnp.sum(c**2))
    return ShearMode(
        c=c,
        r0=r0,
        q=float(q),
        n=int(n),
        width=float(width),
        height=float(height),
        mod_axis=int(mod_axis),
        pol_axis=int(pol_axis),
        rho_v=n_vert / (width * height),
        c_sq=c_sq,
        c_sq_ratio=2.0 * c_sq / n_vert,
    )


def min_image_delta_x(x: Array, x0: Array, period: float) -> Array:
    r"""Minimum-image displacement ``x - x0`` wrapped into the half-open $[-L/2, +L/2)$.

    Spelled ``d - L*floor(d/L + 1/2)``. Three properties matter, and V0 checks all three:

    * **Its derivative is exactly 1 everywhere** in JAX, because ``jnp.floor`` carries a zero JVP.
      No ``stop_gradient`` is needed, and none is used: the image index ``floor(d/L + 1/2)`` is an
      integer label of the periodic image, piecewise constant, with derivative zero almost
      everywhere.
    * **It is exactly invariant under a lattice translation**, at every point including the
      tie. The obvious spelling ``d - L*round(d/L)`` is *not*: ``round`` is half-to-even, so it maps
      both $\pm L/2$ to themselves, the interval comes out closed, and the two representatives
      at exactly $d=L/2$ differ by one lattice vector, $|\Delta|=L$. The physics is
      untouched -- either representative is a valid minimum image -- but an exact invariance is
      worth more than a caveat, which is why the half-open interval is used here.
    * **It is not a nine-image ``jnp.min`` reduction**, deliberately: that form has a kink whose AD
      path is the one ``geo.min_image_sq_dist`` documents as having produced NaN gradients on real
      meshes.
    """
    d = x - x0
    return d - period * jnp.floor(d / period + 0.5)


# ---------------------------------------------------------------------------
# Drive energy and observable
# ---------------------------------------------------------------------------


def make_drive_energy(
    base_energy: Callable[[Array, Array, Array, Array, Array, Array], Array],
    mode: ShearMode,
    a: float,
) -> Callable[[Array, Array, Array, Array, Array, Array], Array]:
    r"""Wrap an inner loss with the cosine drive potential at amplitude ``a``.

    Args:
        base_energy: A callable with the **exact** six-argument inner-loss signature
            ``(vertTable, heTable, faceTable, vert_params, he_params, face_params)``, as
            ``opt.InnerLossFunction`` requires. ``width``/``height`` are ``static_argnums``
            throughout the library and are therefore closed over by the caller, not passed here.
        mode: The drive mode; its weights and reference state are captured by closure.
        a: Drive amplitude. ``a = 0`` returns the undriven energy up to a term that is identically
            zero, so it is a valid way to build the free condition.

    Returns:
        A callable with the same six-argument signature, adding
        $-a\sum_i\hat c_i\Delta r_i^{\rm pol}$.

    The returned energy raises on a vertex-row-count mismatch. That converts the one silent failure
    mode of this construction — a partial ``selected_verts``, which slices ``vertTable`` and so
    mis-pairs weights with vertices — into a loud one.
    """
    c = mode.c
    r0_pol = mode.r0[:, mode.pol_axis]
    pol = mode.pol_axis
    period = mode.pol_period
    n_expected = int(c.shape[0])
    amp = float(a)

    def energy_with_drive(
        vertTable: Array,  # noqa: N803
        heTable: Array,  # noqa: N803
        faceTable: Array,  # noqa: N803
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        """Inner loss = vertex-model energy + cosine drive potential."""
        if vertTable.shape[0] != n_expected:
            msg = (
                f"drive energy built for {n_expected} vertices, called with {vertTable.shape[0]}. "
                "The weights are attached to vertex rows, so a sliced vertTable (a partial "
                "selected_verts) would mis-pair them. Pass the full vertex selection."
            )
            raise ValueError(msg)
        base = base_energy(vertTable, heTable, faceTable, vert_params, he_params, face_params)
        delta = min_image_delta_x(vertTable[:, pol], r0_pol, period)
        return base - amp * jnp.sum(c * delta)

    return energy_with_drive


def mode_amplitude(vertices: Array, mode: ShearMode) -> Array:
    r"""Amplitude of the driven Fourier component of the displacement field,
    $U=\sum_i\hat c_i\Delta r_i/\sum_i\hat c_i^2$ — a **linear** functional of the positions.
    """
    delta = min_image_delta_x(vertices[:, mode.pol_axis], mode.r0[:, mode.pol_axis], mode.pol_period)
    return jnp.sum(mode.c * delta) / mode.c_sq


def compliance(u_plus: Array | float, u_minus: Array | float, a0: float) -> Array:
    r"""Mode compliance $J=(U_+-U_-)/(2a_0)$ from the antisymmetric pair.

    The pair removes the $a=0$ offset that any finite disordered mesh carries, and with it the
    leading $O(a_0^2)$ nonlinearity. Note $\Delta r^+_i-\Delta r^-_i=r^+_i-r^-_i$
    exactly, so ``J`` does not depend on the reference positions at all — only the weights do.
    """
    return (u_plus - u_minus) / (2.0 * a0)


def modulus_from_compliance(j: Array | float, mode: ShearMode) -> Array:
    r"""Apparent modulus at the mode's wavevector, $G_n=\rho_v/(Jq_n^2)$.

    This is the force-balance calibration. See
    `ShearMode.c_sq_ratio` for the discrete-vs-continuum diagnostic that must be *reported*
    beside it rather than folded into it. ``G`` diverges as $J\to0$, which is exactly why the
    outer cost is written on ``J``.
    """
    return mode.rho_v / (jnp.asarray(j) * mode.q**2)


def target_compliance(g_target: float, mode: ShearMode) -> float:
    r"""Compliance target $J_{\rm tgt}=\rho_v/(G_{\rm tgt}q^2)$ for a modulus target."""
    if g_target <= 0:
        msg = (
            f"G_target must be > 0; got {g_target}. A zero-modulus target is the degenerate "
            "fluid-side case: J_tgt is infinite, the map G -> p0 is many-to-one, and the design "
            "problem must be reported as a boundary, not optimised against."
        )
        raise ValueError(msg)
    return mode.rho_v / (g_target * mode.q**2)


def make_compliance_cost(
    mode: ShearMode,
    j_target: float,
    a0: float,
    partner_amplitude: float,
    sign: Literal[1, -1],
) -> Callable[..., Array]:
    r"""Outer loss $\mathcal C=(\hat J/J_{\rm tgt}-1)^2$ for **one** condition of the pair.

    The partner condition's amplitude is frozen and captured by closure, which is what makes the
    block-coordinate alternation expressible against the library's one-mesh outer loss.
    ``sign = +1`` for the $+a_0$ condition, ``-1`` for $-a_0$.

    Returns a callable with the exact twelve-argument ``opt.OuterLossFunction`` signature. It uses
    only ``vertTable``: the cost has **no** parameter dependence, by construction, which is the
    property that keeps the implemented centered-EP estimator correct.
    """
    if sign not in (1, -1):
        msg = f"sign must be +1 or -1; got {sign}"
        raise ValueError(msg)
    two_a0 = 2.0 * float(a0)
    partner = float(partner_amplitude)
    jt = float(j_target)

    def cost(
        vertTable: Array,  # noqa: N803
        heTable: Array | None = None,  # noqa: N803, ARG001
        faceTable: Array | None = None,  # noqa: N803, ARG001
        width: float | None = None,  # noqa: ARG001
        height: float | None = None,  # noqa: ARG001
        vertTable_target: Array | None = None,  # noqa: N803, ARG001
        heTable_target: Array | None = None,  # noqa: N803, ARG001
        faceTable_target: Array | None = None,  # noqa: N803, ARG001
        selected_verts: Array | None = None,  # noqa: ARG001
        selected_hes: Array | None = None,  # noqa: ARG001
        selected_faces: Array | None = None,  # noqa: ARG001
        image_target: Array | None = None,  # noqa: ARG001
    ) -> Array:
        """Squared relative compliance error, with the partner amplitude held frozen."""
        u = mode_amplitude(vertTable, mode)
        j_hat = (u - partner) / two_a0 if sign == 1 else (partner - u) / two_a0
        return (j_hat / jt - 1.0) ** 2

    return cost


# ---------------------------------------------------------------------------
# Geometry helpers (periodic, offset-aware)
# ---------------------------------------------------------------------------


def half_edge_vectors(vertices: Array, edges: Array, width: float, height: float) -> Array:
    r"""Per-half-edge vector $\mathbf d_{he}=(\mathbf x_{\rm tgt}+\mathbf n L)-\mathbf x_{\rm src}$.

    Columns 3/4 of ``heTable`` are source/target vertex, 6/7 the $\pm1$ box offsets — the same
    convention ``geo.get_length`` uses, and read from the same columns so the two cannot drift.
    """
    src = edges[:, 3].astype(jnp.int32)
    tgt = edges[:, 4].astype(jnp.int32)
    off = jnp.stack([edges[:, 6] * width, edges[:, 7] * height], axis=-1)
    return vertices[tgt, :2] + off - vertices[src, :2]


def face_areas(vertices: Array, edges: Array, faces: Array, width: float, height: float) -> Array:
    """Per-face area, via ``geo.get_area`` so the shoelace/offset convention is the library's."""
    return jax.vmap(lambda f: get_area(f, vertices, edges, faces, width, height, MAX_EDGES_IN_ANY_FACE))(
        jnp.arange(len(faces))
    )


def face_perimeters(vertices: Array, edges: Array, faces: Array, width: float, height: float) -> Array:
    """Per-face perimeter, via ``geo.get_perimeter``."""
    return jax.vmap(lambda f: get_perimeter(f, vertices, edges, faces, width, height, MAX_EDGES_IN_ANY_FACE))(
        jnp.arange(len(faces))
    )


def half_edge_lengths(vertices: Array, edges: Array, width: float, height: float) -> Array:
    """Per-half-edge length, via ``geo.get_length``."""
    return jax.vmap(lambda h: get_length(h, vertices, edges, None, width, height))(jnp.arange(len(edges)))


# ---------------------------------------------------------------------------
# Virial (Born/Batchelor) stress — validation only
# ---------------------------------------------------------------------------


def virial_stress(
    vertices: Array,
    edges: Array,
    faces: Array,
    width: float,
    height: float,
    he_tensions: Array,
    iso_coeff: Array | float,
) -> Array:
    r"""Assemble $A\,\sigma_{ij}=\sum_{he}T_{he}\,d_i d_j/\ell + \delta_{ij}c_{\rm iso}$.

    **Why the naive definition vanishes.** $\bm\sigma=A^{-1}\mathbf X^\top\nabla_1\mathcal E$
    is correct for an affine strain of the *stored coordinates* alone — and it is identically zero at
    any relaxed configuration, because that is what $\nabla_1\mathcal E=0$ means. Under PBC an
    affine strain must act on the **box vectors** too, and the missing term
    $\mathbf h^\top\partial\mathcal E/\partial\mathbf h$ is the entire signal. The Born/virial
    form used here is that term, written per edge: it is the derivative with respect to the edge
    vectors, which do respond to a box strain.

    ``he_tensions`` is $\partial\mathcal E/\partial\ell_{he}$ per **half-edge**, so the factor
    conventions of the energy (the leading 1/2 that cancels the half-edge double count in ``E2``; the
    two-faces-per-edge sum in ``E1``) are inherited from the energy rather than re-derived here.
    ``iso_coeff`` is the $\delta_{ij}$ coefficient from the area terms.

    Returns the ``(2, 2)`` stress. The normalising area is the **box** area $WH$; compare it
    against ``face_areas(...).sum()`` as a mesh-consistency check.
    """
    d = half_edge_vectors(vertices, edges, width, height)
    ell = jnp.sqrt(jnp.sum(d**2, axis=-1))
    safe = jnp.where(ell > 0, ell, 1.0)
    outer = jnp.einsum("ei,ej->eij", d, d)
    tension_part = jnp.sum((he_tensions / safe)[:, None, None] * outer, axis=0)
    return (tension_part + jnp.eye(2) * iso_coeff) / (width * height)


def virial_stress_e2(
    vertices: Array,
    edges: Array,
    faces: Array,
    width: float,
    height: float,
    he_params: Array,
    face_params: Array,
    k_areas: float,
) -> Array:
    r"""Virial stress for the periodic ``E2`` energy of ``energy.make_energy_line_tensions``.

    $\mathcal E_2=\tfrac12\sum_{he}\Lambda_{he}\ell_{he}+\tfrac K2\sum_\alpha(A_\alpha-A^0_\alpha)^2$
    gives $\partial\mathcal E/\partial\ell_{he}=\tfrac12\Lambda_{he}$ and, from a uniform
    dilation $A\to(1+\varepsilon)^2A$, the isotropic coefficient
    $K\sum_\alpha(A_\alpha-A^0_\alpha)A_\alpha$. The latter is
    $-\sum_\alpha P_\alpha A_\alpha$ with the cell pressure
    $P_\alpha=-K(A_\alpha-A^0_\alpha)$: the Batchelor tissue stress, ported to periodic tables.
    """
    areas = face_areas(vertices, edges, faces, width, height)
    iso = k_areas * jnp.sum((areas - face_params) * areas)
    return virial_stress(vertices, edges, faces, width, height, 0.5 * he_params, iso)


def virial_stress_e1(
    vertices: Array,
    edges: Array,
    faces: Array,
    width: float,
    height: float,
    face_params: Array,
) -> Array:
    r"""Virial stress for the periodic ``E1`` energy (``energy.energy_shape_factor_*``).

    $\mathcal E_1=\sum_\alpha[(A_\alpha-1)^2+(P_\alpha-p^0_\alpha)^2]$ gives an effective
    half-edge tension $2(P_{\alpha(he)}-p^0_{\alpha(he)})$, where $\alpha(he)$ is the
    face owning the half-edge (column 5). Summing over half-edges therefore reproduces the
    per-edge $T_e^{\rm eff}=\sum_{\alpha\ni e}2(P_\alpha-p^0_\alpha)$
    automatically, since an edge's two half-edges belong to its two adjacent faces. The isotropic
    coefficient is $2\sum_\alpha(A_\alpha-1)A_\alpha$.

    ``face_params`` may be a scalar (the homogeneous case) or one value per face.
    """
    areas = face_areas(vertices, edges, faces, width, height)
    perims = face_perimeters(vertices, edges, faces, width, height)
    p0 = jnp.broadcast_to(jnp.asarray(face_params), (len(faces),))
    face_of_he = edges[:, 5].astype(jnp.int32)
    he_tensions = 2.0 * (perims - p0)[face_of_he]
    iso = 2.0 * jnp.sum((areas - 1.0) * areas)
    return virial_stress(vertices, edges, faces, width, height, he_tensions, iso)


def stress_profile_xy(
    vertices: Array,
    edges: Array,
    width: float,
    height: float,
    he_tensions: Array,
    n_bins: int = 12,
    n_quad: int = 16,
) -> tuple[Array, Array]:
    r"""Profile of $\sigma_{xy}$ across the modulation axis, for V1.

    Force balance requires $\partial_y\sigma_{xy}=-f_x(y)$, hence
    $\sigma_{xy}(y)=-(a\rho_v/q)\sin(qy)$ — parameter-free, so it validates the drive, the
    stress and the calibration constant $\rho_v$ at once.

    Binning is a **sub-segment quadrature of the Irving–Kirkwood line integral**: each half-edge is
    split into ``n_quad`` equal pieces and each piece's midpoint deposits $1/n_{\rm quad}$ of
    that half-edge's $T d_x d_y/\ell$ into the bin containing it (wrapped modulo the box). This
    is what makes the discrete profile satisfy the continuum balance: depositing the whole
    contribution at the edge *midpoint* instead smears the profile at
    $O(\ell^2/\Delta y^2)$, which at these bin widths is not negligible. Convergence in
    ``n_quad`` is cheap to check and should be reported.

    Returns:
        ``(bin_centres, sigma_xy)``, each of length ``n_bins``; ``sigma_xy`` is normalised by the
        bin volume $W\,\Delta y$.
    """
    src = edges[:, 3].astype(jnp.int32)
    d = half_edge_vectors(vertices, edges, width, height)
    ell = jnp.sqrt(jnp.sum(d**2, axis=-1))
    safe = jnp.where(ell > 0, ell, 1.0)
    weight = he_tensions * d[:, 0] * d[:, 1] / safe / n_quad

    # Shapes are load-bearing: (n_half_edges, n_quad). `d[:, 1]` must be indexed on the EDGE axis,
    # not broadcast against `frac` on the quadrature axis -- the latter raises only when
    # n_quad != n_half_edges, so it would have passed silently on a mesh with 16 half-edges.
    frac = (jnp.arange(n_quad) + 0.5) / n_quad
    y_start = vertices[src, 1]
    y_pts = (y_start[:, None] + frac[None, :] * d[:, 1][:, None]) % height
    idx = jnp.clip((y_pts / height * n_bins).astype(jnp.int32), 0, n_bins - 1)

    binned = jnp.zeros(n_bins).at[idx.ravel()].add(jnp.broadcast_to(weight[:, None], idx.shape).ravel())
    bin_volume = width * height / n_bins
    centres = (jnp.arange(n_bins) + 0.5) * height / n_bins
    return centres, binned / bin_volume


# ---------------------------------------------------------------------------
# Hessian: exact linear response (V2) and the stability precondition (V7)
# ---------------------------------------------------------------------------


def hessian_spectrum(
    energy: Callable[[Array, Array, Array, Array, Array, Array], Array],
    vertices: Array,
    edges: Array,
    faces: Array,
    vert_params: Array | None,
    he_params: Array | None,
    face_params: Array | None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Eigendecomposition of the inner Hessian $\partial^2\mathcal E/\partial\mathbf X^2$.

    Dense, symmetrised, ``float64``, and deliberately general: a box/energy-specialised
    $\lambda_{\max}$ primitive already exists elsewhere for the square-box ``E1`` case, so
    this one is written to be reused rather than triplicating the method. At 72 cells the matrix
    is about ``288 x 288``, so a dense ``eigvalsh`` is cheaper than a power iteration and returns
    the whole spectrum, which V2 needs anyway.

    Returns:
        ``(eigenvalues ascending, eigenvectors)`` as ``numpy`` arrays over the flattened ``(N, 2)``
        degrees of freedom, C-ordered: dof ``2i + k`` is vertex ``i``, component ``k``.
    """
    require_float64()

    def e_of_x(x: Array) -> Array:
        return energy(x, edges, faces, vert_params, he_params, face_params)

    h = jax.hessian(e_of_x)(vertices)
    n_dof = vertices.shape[0] * vertices.shape[1]
    hm = np.asarray(h, dtype=np.float64).reshape(n_dof, n_dof)
    w, v = np.linalg.eigh(0.5 * (hm + hm.T))
    return w, v


def hessian_lambda_max(
    energy: Callable[[Array, Array, Array, Array, Array, Array], Array],
    vertices: Array,
    edges: Array,
    faces: Array,
    vert_params: Array | None = None,
    he_params: Array | None = None,
    face_params: Array | None = None,
) -> float:
    r"""Largest Hessian eigenvalue — the SGD stability precondition is $\eta\lambda_{\max}<2$.

    **Check it, never assume it.** Unrolled AD carries $|1-\eta\lambda|^k$ over ``k`` inner
    steps; at $\eta\lambda=2.13$ and 1000 steps that is $10^{51}$ per epoch, and the
    failure is silent in the direction that matters — the iterate and the recorded metrics stay
    finite and plausible while the sensitivity diverges. The rheology plan refuses to launch above
    $\eta\lambda_{\max}=1$.
    """
    w, _ = hessian_spectrum(energy, vertices, edges, faces, vert_params, he_params, face_params)
    return float(w[-1])


def top_mode_participation(
    energy: Callable[[Array, Array, Array, Array, Array, Array], Array],
    vertices: Array,
    edges: Array,
    faces: Array,
    vert_params: Array | None = None,
    he_params: Array | None = None,
    face_params: Array | None = None,
) -> dict[str, float]:
    r"""How localised is the stiffest Hessian mode?

    Returns the **participation ratio** $1/\sum_i w_i^2$ of the per-vertex weights
    $w_i=v_{2i}^2+v_{2i+1}^2$ of the top eigenvector, i.e. an effective number of
    participating vertices, together with $\lambda_{\max}$ and the largest single-vertex
    weight.

    This discriminates the two reasons $\lambda_{\max}$ can be large, which call for opposite
    responses. A participation ratio of order the system size means the mesh is *uniformly* stiff and
    the fix is the step size $\eta$. A ratio of order one means a single junction is nearly
    collapsed and dominates the spectrum — reducing $\eta$ then merely hides it, and the
    honest response is to report the configuration. Note the ``E1`` NaN investigation established
    that $\lambda_{\max}$ does **not** track the minimum edge length on its meshes, so the
    two diagnostics are not interchangeable and both are reported.
    """
    w, v = hessian_spectrum(energy, vertices, edges, faces, vert_params, he_params, face_params)
    top = v[:, -1]
    weights = top.reshape(-1, 2)
    per_vertex = np.sum(weights**2, axis=1)
    per_vertex = per_vertex / per_vertex.sum()
    return {
        "lambda_max": float(w[-1]),
        "participation": float(1.0 / np.sum(per_vertex**2)),
        "max_vertex_weight": float(per_vertex.max()),
        "argmax_vertex": int(np.argmax(per_vertex)),
    }


def hessian_compliance(
    energy: Callable[[Array, Array, Array, Array, Array, Array], Array],
    vertices: Array,
    edges: Array,
    faces: Array,
    mode: ShearMode,
    vert_params: Array | None = None,
    he_params: Array | None = None,
    face_params: Array | None = None,
    *,
    rel_cut: float = 1e-8,
) -> dict[str, float]:
    r"""Exact linear-response compliance $J=(\hat{\mathbf c}^\top\mathbf H^{-1}\hat{\mathbf c})/(\hat{\mathbf c}^\top\hat{\mathbf c})$, for V2.

    ``H`` is singular: the two uniform translations are exact zero modes under PBC. The drive vector
    is mean-removed along the polarisation axis and identically zero on the other, so it is
    orthogonal to both by construction — the inverse is taken on the complement, and the leaked
    overlap is **returned** rather than assumed away.

    This is a linear-response identity and applies only in the T1-free regime; once events fire the
    nonlinear relaxation is not measuring the same object.

    Returns a dict with ``J_exact``, ``lambda_max``, ``lambda_min_kept``, ``n_dropped``,
    ``dropped_overlap`` (fraction of $\|\hat c\|^2$ in the discarded subspace — should be
    ~1e-16), and ``cond_kept``.
    """
    w, v = hessian_spectrum(energy, vertices, edges, faces, vert_params, he_params, face_params)
    n_vert = vertices.shape[0]
    chat = np.zeros(n_vert * 2, dtype=np.float64)
    chat[np.arange(n_vert) * 2 + mode.pol_axis] = np.asarray(mode.c, dtype=np.float64)
    proj = v.T @ chat
    lam_max = float(w[-1])
    keep = w > rel_cut * lam_max
    dropped = float(np.sum(proj[~keep] ** 2) / np.sum(proj**2))
    j_exact = float(np.sum(proj[keep] ** 2 / w[keep]) / np.sum(chat**2))
    return {
        "J_exact": j_exact,
        "lambda_max": lam_max,
        "lambda_min_kept": float(w[keep].min()) if keep.any() else float("nan"),
        "n_dropped": int((~keep).sum()),
        "dropped_overlap": dropped,
        "cond_kept": float(lam_max / w[keep].min()) if keep.any() else float("inf"),
    }


def grad_x_norm(
    energy: Callable[[Array, Array, Array, Array, Array, Array], Array],
    vertices: Array,
    edges: Array,
    faces: Array,
    vert_params: Array | None = None,
    he_params: Array | None = None,
    face_params: Array | None = None,
) -> float:
    r"""$\|\nabla_{\mathbf X}\mathcal E\|_2$ — report it for **every** solve before
    interpreting any compliance. A modulus attached to an unconverged equilibrium is not a modulus.
    """
    g = jax.grad(lambda x: energy(x, edges, faces, vert_params, he_params, face_params))(vertices)
    return float(jnp.linalg.norm(g))


# ---------------------------------------------------------------------------
# Topology bookkeeping (V6) — a stratification key, not a prohibition
# ---------------------------------------------------------------------------


def topology_signature(edges: Array) -> tuple[tuple[int, int], ...]:
    r"""Relabelling-invariant signature of the mesh topology: the sorted multiset of unordered
    face pairs, one per half-edge.

    A T1 transition swaps which two faces share an edge, so it changes this multiset; a mere
    re-indexing of half-edges does not. **That distinction is the point.** A counter keyed on raw
    half-edge rows would conflate T1 events with index relabelling, which is why this signature is
    keyed on face pairs instead. Face *labels* are assumed stable across a T1, which is true of
    this implementation but is an assumption, not a theorem.
    """
    e = np.asarray(edges)
    twin = e[:, 2].astype(int)
    face = e[:, 5].astype(int)
    pairs = [tuple(sorted((int(face[i]), int(face[twin[i]])))) for i in range(len(e))]
    return tuple(sorted(pairs))


def t1_event_count(sig_before: tuple, sig_after: tuple) -> int:
    """Number of edges whose face-pair changed between two signatures, halved (each event moves the
    pair on both half-edges of the edge). A **lower bound** on the event count: two T1s that undo
    one another within the same solve are invisible to any before/after comparison."""
    from collections import Counter

    diff = Counter(sig_before) - Counter(sig_after)
    return int(sum(diff.values()) // 2)


def connectivity_hash(edges: Array) -> str:
    """Stable hash of the full half-edge table, for the EP validity check.

    EP's paired cancellation requires the $+\\beta$ and $-\\beta$ solves to cross the
    **same** events. A hash mismatch invalidates that outer step, and it is recorded as invalid —
    **not repaired**. Unlike `topology_signature` this is sensitive to relabelling, which is
    what makes it the right tool for comparing two solves from a common starting state and the wrong
    tool for counting events.
    """
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(np.asarray(edges, dtype=np.int64)).tobytes()).hexdigest()[:16]


def drive_drift(vertices: Array, mode: ShearMode) -> float:
    r"""Drift diagnostic $\max_i|\cos(qr_i)-\overline{\cos(qr)}-\hat c_i|$.

    The weights are attached to material points, so after substantial rearrangement they no longer
    coincide with the Eulerian field. Re-reference the mode from the $a=0$ relaxed state at
    each outer epoch, and treat the reference as a constant of the estimator *within* the epoch.
    """
    raw = jnp.cos(mode.q * vertices[:, mode.mod_axis])
    return float(jnp.max(jnp.abs(raw - jnp.mean(raw) - mode.c)))
