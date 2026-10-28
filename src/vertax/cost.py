"""Cost functions collection, used for outer optimization.

A cost function can be any user-defined function but it has to respect a strict signature.

For a `PbcMesh` and `PbcBilevelOptimizer`, the cost function must have the following signature:
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
and return an Array (or float).

The names can vary and you can give default parameters. But the number and type of parameters is important.
You don't have to use every parameters but they all have to be here.
An unused parameters can of course also have the type None.

Same for `BoundedMesh` and `BoundedBilevelOptimizer`, but with a slightly different signature:
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vertTable_target: Array,
    angTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
and return an Array (or float).

Hopefully the variable names are self-explanatory.

You can create a function with this signature exactly that uses also locally-accessible external variable if you want.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap
from jax.lax import fori_loop
from jax.numpy import arange, array, diff, einsum, exp, expand_dims, int32, meshgrid, pi, sqrt, stack
from jax.numpy import sinc as npsinc
from jax.numpy.fft import ifft2

from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from vertax.energy import MAX_EDGES_IN_ANY_FACE
from vertax.geo import get_area, min_image_sq_dist

TARGET_RATIO = 2.0

# import matplotlib.pyplot as plt


# Image size, larger than the cube [-1,1]² --> FFT produces artefacts otherwise
s = array([-3, +3], dtype=float)

# Number of pixels along each axis
ns = array([256, 256], dtype=int32)

# The point-spread width every recorded `cost_mesh2image` result was produced at.
#
# It is a NAMED CONSTANT, not module state: `sigma_psf` is a parameter of the cost, supplied at
# the call, because it may itself become trainable alongside the mechanical parameters. The
# previous form -- a module-level covariance matrix and a precomputed `fg`, mutated from outside
# -- could not be differentiated with respect to, and was worse than that: `fg` was read as a
# global inside a `@jit` function, so it was captured as a CONSTANT AT TRACE TIME and any change
# made after the first call was silently ignored.
#
# The legacy covariance was `diag(5e-5, 5e-5)`. With an isotropic covariance `Sigma = sigma^2 I`,
# `exp(-xi^T Sigma xi / 2) = exp(-0.5 * sigma^2 * |xi|^2)`, which is the form
# `render_interface_image` already uses, so the two paths now build the same object.
LEGACY_PSF_SIGMA = 5e-5**0.5   # a Python double: jnp.sqrt would round through float32 here


# Fourier grid defined by the Nyquist-Shannon sampling theorem
def _xigrid() -> Array:
    """Computes the Fourier grid.

      The grid is associated the regular grid defined in the box
    [-s₀/2,s₀/2]x[-s₁/2,s₁/2] and defined by the Nyquist-Shannon sampling theorem.
      * s is the box size (centered at 0).
      * ns is an integer vector defining the size of the grid.
    Returns the Fourier grid.
    """
    return stack(
        meshgrid(
            pi / s[0] * (2 * arange(ns[0], dtype=float) - ns[0] + 1),
            pi / s[1] * (2 * arange(ns[1], dtype=float) - ns[1] + 1),
            indexing="ij",
        )
    )


# FFT phase
def _fft_phase() -> Array:
    """Defines the phase term for the FFT computation.

      * ns is an integer vector defining the size of the grid.
    Returns the phase term.
    """
    return stack(
        meshgrid(
            exp(1j * pi * (1 - ns[0]) / ns[0] * arange(ns[0], dtype=float)),
            exp(1j * pi * (1 - ns[1]) / ns[1] * arange(ns[1], dtype=float)),
            indexing="ij",
        )
    ).prod(axis=0)


def _psf_fourier(sigma_psf: Array, xi: Array) -> Array:
    """Fourier transform of an isotropic Gaussian PSF of standard deviation `sigma_psf`.

    Formed at call time from an explicit argument, so it is traced rather than captured and
    `jax.grad` with respect to `sigma_psf` flows through it.

    * `sigma_psf` is the PSF standard deviation, in the units of the box the grid spans.
    * `xi` is the Fourier grid, size (2, p0, p1).

    Returns:
        `exp(-0.5 * sigma_psf**2 * |xi|**2)`, size (p0, p1). The zero-mean phase factor of the
        general form is identically one and is not carried.
    """
    return exp(-0.5 * sigma_psf**2 * (xi**2).sum(axis=0))


# Pre-compute the grid constants. These are geometry -- box size and sampling -- not
# parameters, so they are legitimately module level; the PSF is not, and is passed in.
xi, phase = _xigrid(), _fft_phase()
xiexpa = expand_dims(xi, tuple(arange(1, 2)))
fac = ns.prod() / s.prod() * exp(1j * pi / 2 * (((ns - 1) ** 2 / ns).sum()))

###################################
# Change numpy definition of sinc #
###################################


@jit
def _sinc(x: Array) -> Array:
    """Sinus cardinal."""
    return npsinc(x / pi)


###############################################
# Fourier Transform Î of a line segment [a,b] #
#  Î(ξ) = |b-a| exp(-iξ(a+b)/2) sinc((b-a)/2) #
###############################################


# FT of all line segments (sum) with precomputed xi
@jit
def _sum_line_segment_fourier_transform(x: Array) -> Array:
    """Computes the Fourier transform of a set of line segments.

    * x is the array of line segments, the size is ((2,2,m)).
        x[:,:.i] is the line segment defined by the points
        (x[0,0,i],x[0,1,i]) and (x[1,0,i],x[1,1,i])
    * Needs a precomputed ξ, array of points where to compute
        the Fourier transform. The size is ((2,n)).
        The first dimension is for the Fourier vector coordinates.

    Returns:
    * The Fourier transform of the set of line segments at points ξ.
        It has the size (n,n).
    """
    # midx,difx=x.sum(axis=0)/2,diff(x,axis=0)[0]
    # return (sqrt((difx**2).sum(axis=0,keepdims=True)).T*exp(-1j*(midx.T@xi))*sinc((difx.T@xi)/2)).sum(axis=0)
    ####
    midx, difx = x.sum(axis=0) / 2, diff(x, axis=0)[0]
    midx, difx = expand_dims(midx, (-1, -2)), expand_dims(difx, (-1, -2))
    nonodifx = sqrt((difx**2).sum(axis=0))  # 1            no weight
    # nonodifx=1/sqrt((difx**2).sum(axis=0))  # (1/x)**2     weight
    # nonodifx*=exp(-nonodifx)                # exp(-x)      weight
    # nonodifx=(difx**2).sum(axis=0)           # x            weight
    # nonodifx=((difx**2).sum(axis=0))**2       # x**2         weight
    # nonodifx*=exp(nonodifx)                 # exp(-x)      weight
    return (nonodifx * exp(-1j * (midx * xiexpa).sum(axis=0)) * _sinc((xiexpa * difx).sum(axis=0) / 2)).sum(axis=0)
    ####


@jit
def _gaussian_blur_line_segments(x: Array, sigma_psf: Array) -> Array:
    """Blurs line segments with an isotropic Gaussian PSF.

    The line segments are defined in the box [-s₀/2,s₀/2]x[-s₁/2,s₁/2] at regularly spaced points.
    * x is the array of line segments: size ((2,2,m₀,m₁,m₂,...))
    * sigma_psf is the PSF standard deviation. It is an ARGUMENT, not module state, so it is
      traced: no global mutation is needed, and `jax.grad` can differentiate
      the result with respect to it.

    Returns:
    * The blurring of the line segments with the Gaussian distribution.
    """
    return fac * ifft2(_psf_fourier(sigma_psf, xi) * _sum_line_segment_fourier_transform(x) * phase) * phase


##################
# COST FUNCTIONS #
##################


@partial(jit, static_argnums=(3, 4))
def cost_v2v(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost vertex to vertex. Compare the positions of given vertices to target vertices (`PbcMesh`)."""
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    def squared_distance(v: Array, vertTable: Array, vertTable_target: Array, width: float, height: float) -> Array:
        """Nearest-periodic-image squared distance for vertex ``v`` -- the C_v2v summand.

        Delegates to :func:`vertax.geo.min_image_sq_dist`, which documents why this must not be
        respelled as a min over nine scalars, nor as ``(min sqrt(u))**2``. The
        same helper is used by the periodic T1 swap-repair criterion so the two cannot drift.
        """
        return min_image_sq_dist(vertTable[v], vertTable_target[v], width, height)

    def mapped_fn(v: Array) -> Array:
        return squared_distance(v, vertTable, vertTable_target, width, height)

    distances = vmap(mapped_fn)(selected_verts)

    return (1.0 / (2 * len(distances))) * jnp.sum(distances)


@jit
def cost_mesh2image(
    vertTable: Array,
    heTable: Array,
    _faceTable: Array,
    width: float,
    height: float,
    _vertTable_target: Array,
    _heTable_target: Array,
    _faceTable_target: Array,
    _selected_verts: Array,
    selected_hes: Array,
    _selected_faces: Array,
    image_target: Array,
    sigma_psf: Array,
) -> Array:
    """Cost mesh to image. Compare the given vertices positions to a target image (`PbcMesh`).

    `sigma_psf` is REQUIRED and is a parameter of the cost, not a module setting. It is traced,
    so `jax.grad(..., argnums=12)` differentiates the cost with respect to the PSF width and a
    trainable PSF is possible without touching this function.

    This is the direct entry point, and it is the one to use when `sigma_psf` is itself being
    optimized. For the fixed-width 12-argument outer-cost protocol used by the optimizers, use
    :func:`make_cost_mesh2image`, which closes over the width instead.

    Every result recorded before the width became an argument was produced at
    :data:`LEGACY_PSF_SIGMA`; pass that value to reproduce one.
    """
    wh = jnp.asarray([width, height])
    starting = (vertTable[heTable[selected_hes, 3], :2]) * 2 / wh  # (M, 2)
    # ending = (vertTable[heTable[selected_hes, 4], :2]) * 2 / L_box  # (M, 2)
    ending = (
        (
            vertTable[heTable[selected_hes, 4], :2]
            + jnp.stack(
                [
                    heTable[selected_hes, 6],
                    heTable[selected_hes, 7],
                ],
                axis=-1,
            )
            * wh
        )
        * 2
        / wh
    )

    he_edges = stack((starting, ending), axis=1)  # (N, 2, 2)
    x = he_edges.transpose(1, 2, 0) - 1  # (2, 2, N)

    # Blur
    image = _gaussian_blur_line_segments(x, sigma_psf).real

    # Normalization
    image = image / image.sum()
    image_target = image_target / image_target.sum()

    l2_norm = jnp.linalg.norm(jnp.sqrt(jnp.sum(((image - image_target) ** 2) * (1 - image_target), axis=-1)).flatten())

    return l2_norm



def make_cost_mesh2image(sigma_psf: float):
    """Bind a PSF width to `cost_mesh2image`, giving the 12-argument outer-cost protocol form.

    Parallel to :func:`make_cost_dmic`. The closure is a convenience for the case where the PSF
    width is FIXED, which is what the bilevel optimizers' 12-argument protocol assumes. It is
    deliberately not the traceable entry point for a TRAINABLE width -- closing over `sigma_psf`
    makes it a constant of the returned function. To differentiate with respect to the PSF width,
    call :func:`cost_mesh2image` directly and pass the width as its thirteenth argument.

    Args:
        sigma_psf: PSF standard deviation, in the units of the box the Fourier grid spans.

    Returns:
        A 12-argument callable matching the outer-cost protocol.
    """

    @jit
    def cost(
        vertTable: Array,
        heTable: Array,
        faceTable: Array,
        width: float,
        height: float,
        vertTable_target: Array,
        heTable_target: Array,
        faceTable_target: Array,
        selected_verts: Array,
        selected_hes: Array,
        selected_faces: Array,
        image_target: Array,
    ) -> Array:
        return cost_mesh2image(
            vertTable, heTable, faceTable, width, height,
            vertTable_target, heTable_target, faceTable_target,
            selected_verts, selected_hes, selected_faces, image_target, sigma_psf,
        )

    return cost


@partial(jit, static_argnums=(3, 4))
def cost_areas(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost areas : compare the areas of the given mesh versus the areas of the target mesh (`PbcMesh`).

    Mean over the selected faces (all faces if `selected_faces` is None) of the squared difference
    between the area of a face and the area of the same face in the target mesh, which must have
    the same faces.
    """
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    def mapped_fn(f: Array) -> Array:
        return (
            get_area(f, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
            - get_area(f, vertTable_target, heTable_target, faceTable_target, width, height, MAX_EDGES_IN_ANY_FACE)
        ) ** 2

    difference = vmap(mapped_fn)(selected_faces)
    return (1.0 / len(difference)) * jnp.sum(difference)


@jit
def cost_ratio(
    vertTable: Array,
    _angTable: Array | None = None,
    _heTable: Array | None = None,
    _faceTable: Array | None = None,
    _vertTable_target: Array | None = None,
    _angTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost that nudges a `BoundedMesh` to elongate along one axis while narrowing along the orthogonal axis."""
    # Compute pairwise squared distances by broadcasting
    # diff shape: (n, n, d)
    diff = vertTable[:, None, :] - vertTable[None, :, :]
    sq_dists = jnp.sum(diff**2, axis=-1)  # (n, n)
    distances = jnp.sqrt(sq_dists + 1e-12)  # small eps for numerical stability

    # For each row, get index of farthest point
    max_distances = jnp.max(distances, axis=1)
    max_idx_1 = jnp.argmax(max_distances)  # index of row with largest max distance
    long_axis = max_distances[max_idx_1]  # longest distance (already excludes diagonal)
    max_idx_2 = jnp.argmax(distances[max_idx_1])  # the farthest point from max_idx_1

    max_point_1 = vertTable[max_idx_1]
    max_point_2 = vertTable[max_idx_2]
    long_axis_vector = max_point_2 - max_point_1
    long_axis_vector = long_axis_vector / (jnp.linalg.norm(long_axis_vector) + 1e-12)

    # Short axis (perpendicular)
    short_axis_vector = jnp.array([long_axis_vector[1], -long_axis_vector[0]])

    # Project points onto short axis: signed distances
    # If vertTable is (n,2) this works. If higher dim, modify accordingly.
    signed_distances_to_long_axis = jnp.dot(vertTable - max_point_1, short_axis_vector)

    short_axis = jnp.max(signed_distances_to_long_axis) - jnp.min(signed_distances_to_long_axis)

    # Numerically stabilize division
    ratio = long_axis / (short_axis + 1e-12)
    return (ratio - TARGET_RATIO) ** 2


@jit
def cost_checkerboard(
    vertTable: Array,
    _angTable: Array,
    heTable: Array,
    faceTable: Array,
    _vertTable_target: Array | None = None,
    _angTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Cost that nudges a `BoundedMesh` with different fated cells to avoid having neighboring cells with same fate.

    This leads to a checkerboard pattern when there is 2 fates.
    """

    def body_fun(i: int, current_len: Array) -> Array:
        idx = 2 * i
        he_twin = heTable[idx, 2]
        he_face = heTable[idx, 7]
        he_fate = faceTable[he_face, 1]
        he_face_twin = heTable[he_twin, 7]
        he_fate_twin = faceTable[he_face_twin, 1]
        v_source = heTable[idx, 3] - 2
        v_target = heTable[idx, 4] - 2
        pos_source = vertTable[v_source]
        pos_target = vertTable[v_target]
        return current_len + jnp.where(
            jnp.logical_and(he_fate == he_fate_twin, heTable[idx, 3] > 0), jnp.sum((pos_target - pos_source) ** 2), 0.0
        )

    n_edges = heTable.shape[0] // 2
    return fori_loop(0, n_edges, body_fun, 0.0)





def topological_distance(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> int:
    r"""Discrete topological distance ``d_topo`` between two meshes: differing cell contacts.

    Builds the dual (cell) adjacency matrix of each mesh, ``A[a, b] = 1`` iff cells ``a`` and ``b``
    share a junction, puts the two cell labellings into correspondence with the Hungarian
    algorithm (cell labels are arbitrary), and counts the cell--cell contacts on which the two
    meshes then disagree, each unordered pair once:

    .. math::
        d_{topo} = \sum_{a<b} \bigl| A^{(1)}_{ab} - A^{(2)}_{\sigma(a)\sigma(b)} \bigr| .

    An elementary T1 transition destroys exactly one contact and creates exactly one other, so it
    changes **two** upper-triangle entries. Therefore ``d_topo`` is always **even**, and
    ``d_topo / 2`` is the number of T1 events separating the two topologies. (Verified against the
    Figure-3 synthetic dataset, whose levels 0, 2, 4, 6, 8, 10 this function reproduces exactly.)

    **This is a measurement, not an objective.** It returns a Python ``int`` and provides no
    gradient, so it cannot be assigned to ``loss_function_outer``. Use it to quantify topological
    distance and to time phase switches in a staged schedule. Separately, any purely topological or
    edge-level cost is a strict information loss for edge-tension inference relative to a
    vertex-level cost such as :func:`cost_v2v` -- the cell-adjacency Laplacian has a large null
    space -- so topology should not be used to infer tensions in the first place.
    """

    def build_dual_adj_np(heTable: Array, n_faces: int) -> NDArray:
        face = heTable[:, 5]
        twin = heTable[:, 2]

        face_twin = face[twin]
        mask = (face != face_twin).astype(float)

        A = np.zeros((n_faces, n_faces))

        A[face, face_twin] += mask
        A[face_twin, face] += mask

        A = np.clip(A, 0.0, 1.0)
        return A

    n1 = len(faceTable)
    n2 = len(faceTable_target)

    if n1 != n2:
        msg = "This version requires same number of faces"
        raise ValueError(msg)

    n = n1

    A1 = build_dual_adj_np(heTable, n)
    A2 = build_dual_adj_np(heTable_target, n)

    # --- Step 1: node matching via Hungarian ---
    # cost between nodes = L1 difference of adjacency rows
    C = np.sum(np.abs(A1[:, None, :] - A2[None, :, :]), axis=-1)

    row_ind, col_ind = linear_sum_assignment(C)

    # permutation: face i in A1 ↔ face col_ind[i] in A2
    P = np.zeros((n, n))
    P[row_ind, col_ind] = 1.0

    # --- Step 2: permute A2 ---
    A2_perm = P @ A2 @ P.T

    # --- Step 3: count edge mismatches ---
    diff = np.abs(A1 - A2_perm)

    # count each edge once
    diff_upper = np.triu(diff, k=1)

    loss = int(np.sum(diff_upper))

    return loss




# =========================================================================================
# Differentiable microscopy: the interface-density image model
# =========================================================================================
#
# This follows the published formulation (Ichbiah et al., ICCV 2025). Three properties of that
# formulation matter for how this module is built:
#
#   * edges are NOT drawn as Gaussian tubes of finite width and then blurred: the fluorophore
#     density is a Dirac distribution supported directly on the interface, so the PSF alone sets
#     the conditioning;
#   * the cost is a SQUARED L2 norm, so its gradient vanishes at a perfect fit -- an unsquared norm
#     leaves a kink at the optimum that no fixed step size can reach;
#   * the full image is compared, not a cropped sub-region: the render is periodic to machine
#     precision, so there are no wrap artefacts to avoid by discarding data.
#
# THE MODEL. The fluorophore density is a Dirac distribution supported ON the interface, normalised
# by total interface measure, and the image is that density convolved with the PSF. **There is
# exactly one length scale, `sigma_psf`.** For straight segments `a_e -> b_e` of length `l_e`:
#
#     u(x)   = (1/|G|) sum_e w_e integral_e delta(x - gamma_e(s)) ds ,    |G| = sum_e w_e l_e
#     uhat(xi) = (1/|G|) sum_e w_e l_e exp(-i xi.(a_e+b_e)/2) sinc(xi.(b_e-a_e)/2)
#     Ihat(xi) = uhat(xi) * exp(-sigma_psf^2 |xi|^2 / 2)
#
# The segment transform is exact, not a quadrature: parameterising `gamma(t) = a + t(b-a)` gives
# `integral_0^1 exp(-i xi.gamma) l dt = l exp(-i xi.(a+b)/2) sinc(xi.(b-a)/2)`. It is verified
# against numerical quadrature to machine precision in `tests/test_dmic.py`.
#
# EDGE WEIGHTS. `w_e = 2` for a cell-cell interface and `w_e = 1` for a free boundary edge, because
# a cell-cell interface is two membranes. Iterating `heTable` visits every internal edge twice,
# which IS that weighting -- and on a confluent periodic tissue, where every edge is cell-cell, the
# doubling is a constant that `1/|G|` cancels exactly. The weights only bite on a bounded mesh.
# Unlike the old real-space renderer, the two half-edges of an edge now render IDENTICALLY: the
# transform is even in `(b-a)` and a lattice shift of the midpoint contributes `exp(-i 2 pi k) = 1`.


def _dmic_fourier_grid(resolution: int, width: float, height: float) -> tuple[Array, Array]:
    """Angular-frequency grid `xi = 2 pi k / L` matching numpy's FFT layout.

    Returned as `(xi_x, xi_y)` with `xi_x` varying along axis 1 and `xi_y` along axis 0, so that
    `ifft2` produces an image indexed `[row = y, col = x]` -- the same convention as the rest of
    this module.
    """
    k = jnp.fft.fftfreq(resolution) * resolution
    kx, ky = jnp.meshgrid(k, k, indexing="xy")
    return 2.0 * jnp.pi * kx / width, 2.0 * jnp.pi * ky / height


def render_interface_image(
    vertices: Array,
    heTable: Array,
    width: float,
    height: float,
    resolution: int,
    sigma_psf: float,
    selected_hes: Array | None = None,
    edge_weights: Array | None = None,
    normalise: bool = True,
    pixel_footprint: bool = False,
    footprint_alias_orders: int | None = None,
) -> Array:
    """Render a periodic mesh as an interface-density image convolved with a Gaussian PSF.

    Args:
        vertices: `(n_v, >=2)` vertex table; only the first two columns are used.
        heTable: half-edge table; columns 3 and 4 are the endpoints and 6, 7 the periodic offsets.
        width, height: box dimensions, in the same physical units as `vertices`.
        resolution: pixels per axis. The image is `(resolution, resolution)`.
        sigma_psf: PSF standard deviation **in physical units, not pixels** -- so the physical
            problem does not change when the resolution does, which was a defect of the previous
            implementation.
        selected_hes: optional subset of half-edge indices to render. `None` renders all of them.
        edge_weights: optional `w_e` per selected half-edge. `None` weights every one equally,
            which is correct for a confluent periodic tissue.
        normalise: default `True`, bit-identical to every prior call. When `False`, the division
            by total interface length (`width * height * sum(w * length)`) below is skipped, and
            the returned image carries the RAW, un-normalised interface density. The division
            forces the render to unit integral regardless of `p0`'s effect on total perimeter --
            defensible for a real image with unknown detector gain, not defensible when comparing
            two renders from the same renderer at a known, shared scale. See `make_cost_dmic`'s
            matching flag.
        pixel_footprint: default `False`, bit-identical to every prior call, and each pixel is then
            the blurred density POINT-SAMPLED at its centre. When `True`, each pixel is the MEAN of
            the blurred density over its square footprint, which is what a camera records. The box
            function of width `Delta` transforms to `sinc(xi Delta / 2)` (unnormalised sinc,
            `sin(u)/u`), so the average is one more factor on the spectrum,
            `sinc(xi_x Delta_x / 2) * sinc(xi_y Delta_y / 2)` with `Delta_x = width / resolution`
            and `Delta_y = height / resolution`. It is exactly 1 at `xi = 0`, so the image still
            integrates to 1. The point sample and the footprint average differ appreciably only
            when `sigma_psf` is comparable to a pixel; the two coincide in the limit of a wide PSF.
            Sampling on the pixel grid folds every frequency `xi + 2 pi m / Delta` (integer `m`)
            onto `xi`, and below about two pixels the blurred density still has content there, so
            the footprint render sums the spectrum over those alias orders before the inverse
            transform. With the fold, the render is the exact pixel average up to the omitted
            orders, which the PSF suppresses below `1e-6`.
        footprint_alias_orders: the number `M` of alias orders folded in on each side, per axis,
            so that `(2M + 1)^2` spectra are summed. `None` (default) takes the smallest `M` at
            which the Gaussian PSF at the first omitted alias, `(2M + 1) pi / Delta`, is below
            `1e-6`: `M = 0` from about 1.7 px up, `M = 1` down to about 0.56 px, `M = 2` down to
            about 0.33 px. It needs a concrete `sigma_psf`; under a trace over `sigma_psf`, pass it
            explicitly. `0` evaluates the footprint factor on the image's own frequency grid only.
            Ignored when `pixel_footprint` is False.

    Returns:
        When `normalise=True` (default): a real `(resolution, resolution)` image whose integral
        over the box is exactly 1, so that two renders are directly comparable without further
        normalisation. When `normalise=False`: the raw interface-density image, on the scale set
        by `sum(w * length)`, comparable to another `normalise=False` render only if both come
        from the same renderer at the same resolution (never to a `normalise=True` one).

    Note:
        Cost is `O(n_edges * resolution^2)` in both time and peak memory, because the sum over
        edges is evaluated on the full frequency grid before reduction. There is no factorisation
        that avoids the `(k, e)` intermediate; a non-uniform FFT would, and is the route to
        resolutions above ~1024. The footprint render multiplies the time by the number of alias
        orders summed, `(2M + 1)^2`.
    """
    v = vertices[:, :2]
    hes = jnp.arange(heTable.shape[0]) if selected_hes is None else jnp.asarray(selected_hes)
    hes = hes.astype(jnp.int32)
    a = v[heTable[hes, 3].astype(jnp.int32)]
    offset = jnp.stack([heTable[hes, 6], heTable[hes, 7]], axis=-1) * jnp.asarray([width, height])
    b = v[heTable[hes, 4].astype(jnp.int32)] + offset

    length = jnp.sqrt(jnp.sum((b - a) ** 2, axis=-1) + 1e-300)
    w = jnp.ones_like(length) if edge_weights is None else jnp.asarray(edge_weights)

    xi_x, xi_y = _dmic_fourier_grid(resolution, width, height)
    mid, diff = 0.5 * (a + b), b - a

    def segment_spectrum(kx: Array, ky: Array) -> Array:
        phase = kx[..., None] * mid[:, 0] + ky[..., None] * mid[:, 1]
        half = 0.5 * (kx[..., None] * diff[:, 0] + ky[..., None] * diff[:, 1])
        uhat = jnp.sum(w * length * jnp.exp(-1j * phase) * _sinc(half), axis=-1)
        if normalise:
            uhat = uhat / (width * height * jnp.sum(w * length))
        return uhat

    def gaussian(kx: Array, ky: Array) -> Array:
        return jnp.exp(-0.5 * sigma_psf**2 * (kx**2 + ky**2))

    if not pixel_footprint:
        return jnp.real(jnp.fft.ifft2(segment_spectrum(xi_x, xi_y) * gaussian(xi_x, xi_y))) \
            * resolution**2

    dx, dy = width / resolution, height / resolution
    orders = (_footprint_alias_orders(sigma_psf, max(dx, dy)) if footprint_alias_orders is None
              else int(footprint_alias_orders))
    folded = jnp.zeros_like(xi_x, dtype=jnp.result_type(xi_x.dtype, jnp.complex64))
    for mx in range(-orders, orders + 1):
        for my in range(-orders, orders + 1):
            kx, ky = xi_x + 2.0 * jnp.pi * mx / dx, xi_y + 2.0 * jnp.pi * my / dy
            box = _sinc(0.5 * kx * dx) * _sinc(0.5 * ky * dy)
            folded = folded + segment_spectrum(kx, ky) * (gaussian(kx, ky) * box)
    return jnp.real(jnp.fft.ifft2(folded)) * resolution**2


# Gaussian PSF at the first alias order the pixel-footprint render leaves out.
_FOOTPRINT_ALIAS_ENVELOPE = 1e-6


def _footprint_alias_orders(sigma_psf: float, pixel_size: float) -> int:
    """Smallest alias order `M` whose first omitted neighbour the Gaussian PSF has suppressed.

    The orders `|m| <= M` are summed; the nearest omitted frequency is `(2M + 1) pi / Delta`, where
    the PSF is `exp(-(2M + 1)^2 pi^2 sigma^2 / (2 Delta^2))`. `Delta` is the larger pixel side.
    """
    try:
        sigma = float(sigma_psf)
    except TypeError as err:  # a traced width cannot set a Python loop bound
        msg = ("pixel_footprint=True chooses its alias orders from a concrete sigma_psf; "
               "pass footprint_alias_orders explicitly when sigma_psf is traced")
        raise TypeError(msg) from err
    if sigma <= 0.0:
        msg = "pixel_footprint=True needs sigma_psf > 0, or footprint_alias_orders set explicitly"
        raise ValueError(msg)
    needed = math.sqrt(-2.0 * math.log(_FOOTPRINT_ALIAS_ENVELOPE)) * pixel_size / (math.pi * sigma)
    return max(0, math.ceil((needed - 1.0) / 2.0))


def erode_mask_for_psf(mask: Array, sigma_psf: float, width: float, resolution: int) -> Array:
    """Shrink a visibility mask by `3 * sigma_psf`, for bounded or cropped fields.

    A pixel within `3 sigma` of the edge of the imaged region receives intensity from tissue the
    model does not represent, so scoring it asks the model to explain data it cannot see. This is
    the image-space analogue of not writing force balance at a clamped vertex.

    Periodic synthetic images need no mask at all: the render is wrap-consistent, so the whole
    frame is comparable and using it quadruples the data relative to a central-quarter crop.
    """
    radius = int(jnp.ceil(3.0 * sigma_psf * resolution / width))
    if radius <= 0:
        return mask
    m = mask
    for axis in (0, 1):
        for shift in range(-radius, radius + 1):
            m = m * jnp.roll(mask, shift, axis=axis)
    return m


def make_cost_dmic(
    sigma_psf: float,
    mask: Array | None = None,
    edge_weights: Array | None = None,
    resolution: int | None = None,
    intensity_weighted: bool = False,
    weight_floor: float = 0.0,
    normalise: bool = True,
    pixel_footprint: bool = False,
):
    """Build a `cost_dmic` bound to a PSF width, for the 12-argument outer-cost protocol.

    `sigma_psf` is a **parameter of the cost** and is captured here explicitly. It is deliberately
    not a module-level global: a global is captured by `jit` at trace time, so rebinding it through
    a setter after the first call silently does nothing -- a trap present in a sibling
    implementation of this cost, and exactly the parameter one wants to sweep.

    Args:
        sigma_psf: PSF standard deviation in physical units.
        mask: optional `(resolution, resolution)` comparison mask `M`. `None` compares the whole
            frame, which is correct for a periodic synthetic image.
        edge_weights: optional per-half-edge `w_e`.
        resolution: pixels per axis. `None` takes it from the target image.
        intensity_weighted: weight each pixel's squared discrepancy by the observed intensity at
            that pixel, giving `0.5 * sum_p w_p (Ihat_p - I_p)^2` with `w_p` the **normalised**
            target image. Off by default, so existing behaviour is bit-identical.

            The unweighted form spends most of its budget on background: at small `sigma_psf` the
            interfaces occupy a vanishing fraction of the frame, yet every pixel counts equally.
            Weighting by the image concentrates the loss where signal actually is, which is the
            form used in the differentiable-microscopy literature.

            **The weight is a fixed field, not a function of the mesh.** It is built from the
            target image only, so no gradient flows through it and the minimiser at a perfect fit
            is unchanged; only the metric in which discrepancies are measured changes.

            **Note the asymmetry.** Because `w` vanishes in background, a model interface that has
            drifted into background is penalised only through the target's own intensity at that
            location -- so missing edges are penalised more strongly than spurious ones, and a
            badly displaced edge can sit in a region of near-zero gradient. `weight_floor` exists
            to bound that effect.
        weight_floor: added to the normalised weight as a fraction of its maximum, i.e.
            `w <- w + weight_floor * max(w)`. `0.0` is the pure form. A small value such as `0.01`
            keeps a restoring gradient in background regions, at the cost of no longer being the
            literature's exact objective. Ignored when `intensity_weighted` is False.
        normalise: default `True`, bit-identical to every prior recorded result. Passed through to
            `render_interface_image`; when `False`, the stored target is also NOT forced to unit
            mass (`image_target` is compared as-is), so both sides move together and the cost is
            no longer blind to total interface length. This is only meaningful when both images
            share a known, common scale and there is no detector-gain ambiguity to be robust to.
            `image_target` must already be on the
            `normalise=False` scale (i.e. rendered with `render_interface_image(...,
            normalise=False)`) when this is set -- passing a `normalise=True` target here compares
            two images on different scales and is a caller error, not caught here.
        pixel_footprint: default `False`, bit-identical to every prior recorded result. Passed
            through to `render_interface_image`: when `True` the model image is the pixel-footprint
            average rather than the point sample at pixel centres, which is the forward model a
            real camera image needs. A synthetic target must then be rendered with the same
            setting, or the two sides differ at the true geometry.

    Returns:
        A function with the outer-cost signature.
    """

    def _cost(
        vertTable: Array,
        heTable: Array,
        _faceTable: Array,
        width: float,
        height: float,
        _vertTable_target: Array | None = None,
        _heTable_target: Array | None = None,
        _faceTable_target: Array | None = None,
        _selected_verts: Array | None = None,
        selected_hes: Array | None = None,
        _selected_faces: Array | None = None,
        image_target: Array | None = None,
    ) -> Array:
        res = image_target.shape[0] if resolution is None else resolution
        image = render_interface_image(
            vertTable, heTable, width, height, res, sigma_psf, selected_hes, edge_weights,
            normalise=normalise, pixel_footprint=pixel_footprint,
        )
        if normalise:
            # the render integrates to 1 by construction; normalise the target to match
            pixel_area = (width / res) * (height / res)
            target = image_target / (jnp.sum(image_target) * pixel_area)
        else:
            # both sides stay on the raw scale -- see the `normalise` docstring above
            target = image_target
        residual = image - target
        if mask is not None:
            residual = residual * mask
        if intensity_weighted:
            # Pixel-by-pixel weight from the observed image. `target` is already normalised, and
            # `jax.lax.stop_gradient` is not needed because it does not depend on `vertTable`;
            # it is a constant field for a given observation.
            weight = target + weight_floor * jnp.max(target)
            return 0.5 * jnp.sum(weight * residual**2)
        return 0.5 * jnp.sum(residual**2)

    return _cost


def cost_dmic(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    _vertTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    image_target: Array | None = None,
    *,
    sigma_psf: float = 0.05,
    mask: Array | None = None,
    intensity_weighted: bool = False,
    weight_floor: float = 0.0,
    pixel_footprint: bool = False,
) -> Array:
    """Differentiable-microscopy outer cost: `0.5 * ||M * (Ihat(theta) - I_obs)||^2`.

    **Squared**, so the gradient vanishes at a perfect fit and the cost is dimensionally consistent
    with `cost_v2v`. Both images carry unit total intensity, so the cost cannot be reduced by
    changing overall brightness -- which a mesh can do by changing total edge length.

    Prefer `make_cost_dmic(sigma_psf, ...)` when using this as an outer loss: the bilevel optimizer
    calls outer costs with a fixed 12-argument signature and cannot pass keyword arguments, so the
    default `sigma_psf` here would silently apply.

    Args:
        sigma_psf: PSF standard deviation in **physical** units. The default is a placeholder; it
            should be set from the imaging system, and swept rather than assumed.
        mask: optional comparison mask. `None` compares the whole frame, correct for a periodic
            image; use `erode_mask_for_psf` for a bounded or cropped one.
        pixel_footprint: average each pixel over its footprint instead of point-sampling it; see
            `render_interface_image`.
    """
    return make_cost_dmic(sigma_psf, mask, intensity_weighted=intensity_weighted,
                          weight_floor=weight_floor, pixel_footprint=pixel_footprint)(
        vertTable, heTable, faceTable, width, height,
        _vertTable_target, _heTable_target, _faceTable_target,
        _selected_verts, selected_hes, _selected_faces, image_target,
    )
