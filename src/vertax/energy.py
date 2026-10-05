"""Energy related functions, used for the inner optimization.

They can be user defined functions but they have to respect a strict signature.

For a `PbcMesh` and `PbcBilevelOptimizer`, the energy function must have the following signature:
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
and return an Array (or float).

The names can vary and you can give default parameters. But the number and type of parameters is important.
You don't have to use every parameters but they all have to be here.
An unused parameters can of course also have the type None.

Same for `BoundedMesh` and `BoundedBilevelOptimizer`, but with a slightly different signature:
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
and return an Array (or float).

Hopefully the variable names are self-explanatory.

You can create a function with this signature exactly that uses also locally-accessible external variable if you want.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, jit, vmap

from vertax.geo import get_area, get_area_bounded, get_edge_length, get_length, get_perimeter, get_surface_length

MAX_EDGES_IN_ANY_FACE = 20


def _cell_energy(
    face: Array, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> Array:
    """E1 energy for `PbcMesh` for a given face. Elastic term on cell areas and shape factors."""
    area = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)


def energy_shape_factor_homo(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    face_params: Array,
) -> Array:
    """E1 energy where the shape factor is uniform (give only one face_params, it will be broadcasted)."""

    def mapped_fn(face: Array, param: Array) -> Array:
        return _cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable), *face_params.shape[1:]))
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
    return jnp.sum(cell_energies)


def energy_shape_factor_hetero(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    selected_faces: Array,
    face_params: Array,
) -> Array:
    """E1 energy where the shape factor depends on the cell."""

    def mapped_fn(face: Array, param: Array) -> Array:
        return _cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    cell_energies = vmap(mapped_fn)(selected_faces, face_params[selected_faces])
    # cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params)
    return jnp.sum(cell_energies)


def make_energy_shape_factor(lambda_area: float | Array = 1.0) -> Callable:
    r"""Factory for the periodic E1 energy with an **explicit** area stiffness.

    Returns a callable with the signature of :func:`energy_shape_factor_hetero`, implementing

    .. math::
        E_1 = \sum_\alpha \lambda_A\,(A_\alpha - 1)^2 + (P_\alpha - p^0_\alpha)^2 ,

    with ``face_params`` holding the per-cell preferred perimeter :math:`p^0_\alpha`. At
    ``lambda_area = 1`` it is :func:`energy_shape_factor_hetero` exactly.

    ``lambda_area`` is closed over, not stored at module level, and it is never converted to a
    Python float, so it may be a traced array. That is what makes it trainable: an energy that
    reads ``lambda_area = exp(kappa)`` from an outer-parameter slot can call this factory inside
    itself, and ``jax.grad`` with respect to ``kappa`` flows through the area term. The
    exponential keeps the stiffness positive without a constraint.
    """

    def energy_shape_factor(
        vertTable: Array,
        heTable: Array,
        faceTable: Array,
        width: float,
        height: float,
        selected_faces: Array,
        face_params: Array,
    ) -> Array:
        """E1 energy with per-cell preferred perimeters and area stiffness `lambda_area`."""

        def mapped_fn(face: Array, param: Array) -> Array:
            area = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
            perimeter = get_perimeter(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
            return (lambda_area * (area - 1) ** 2) + ((perimeter - param) ** 2)

        cell_energies = vmap(mapped_fn)(selected_faces, face_params[selected_faces])
        return jnp.sum(cell_energies)

    return energy_shape_factor


def _area_part(
    face: Array, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> Array:
    """Part of an energy function."""
    a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    return (a - face_param) ** 2


def _hedge_part(
    he: Array, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> Array:
    """Part of an energy function."""
    # `get_length` takes 6 arguments and needs NO face-iteration bound: a half-edge has exactly
    # two endpoints, so there is nothing to iterate over. `get_area` does take a 7th
    # `max_iter=MAX_EDGES_IN_ANY_FACE` argument because it walks a face's half-edge cycle.
    # Passing MAX_EDGES_IN_ANY_FACE here raises TypeError. Guarded by
    # tests/test_api_contracts_guard.py.
    length = get_length(he, vertTable, heTable, faceTable, width, height)
    return he_param * length


def make_energy_line_tensions(K_areas: float) -> Callable:
    r"""Factory for the periodic E2 energy with an **explicit** area stiffness.

    Returns a callable with the inner-loss signature expected by the bilevel optimizers,
    implementing exactly

    .. math::
        E_2 = \tfrac12\sum_{he}\Lambda_{he}\,\ell_{he}
            + \tfrac{K}{2}\sum_\alpha (A_\alpha - A^0_\alpha)^2 ,

    where the leading 1/2 on the tension term cancels the half-edge double count, so
    ``he_params`` stores the physical per-edge tension directly.

    ``K_areas`` is a required argument: it must be passed explicitly at every call site so
    that the value used is always visible in the driver/notebook rather than buried in the
    library.
    """

    def energy_line_tensions(
        vertTable: Array,
        heTable: Array,
        faceTable: Array,
        width: float,
        height: float,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        """E2 energy for PBC meshes: elastic penalty on cell areas + line tensions x edge lengths."""

        def mapped_areas_part(face: Array, face_param: Array) -> Array:
            return _area_part(face, face_param, vertTable, heTable, faceTable, width, height)

        def mapped_hedges_part(he: Array, he_param: Array) -> Array:
            return _hedge_part(he, he_param, vertTable, heTable, faceTable, width, height)

        areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), face_params)
        hedges_part = vmap(mapped_hedges_part)(jnp.arange(len(heTable)), he_params)
        return (0.5 * jnp.sum(hedges_part)) + (0.5 * K_areas) * jnp.sum(areas_part)

    return energy_line_tensions


# ==========
# Bounded
# ==========
@jit
def _cell_area_energy(
    face: Array, target_area: Array, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array
) -> Array:
    """Part of an energy function. `target_area` is the per-cell target area A^0_alpha."""
    area = get_area_bounded(face, vertTable, angTable, heTable, faceTable)
    return (area - target_area) ** 2


@jit
def _surface_edge_energy(edge: Array, tension: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    """Part of an energy function."""
    length = get_surface_length(edge, vertTable, angTable, heTable)
    return length * tension


@jit
def _inner_edge_energy(edge: Array, tension: Array, vertTable: Array, heTable: Array) -> Array:
    """Part of an energy function."""
    length = get_edge_length(edge, vertTable, heTable)
    return length * tension


def make_energy_line_tensions_bounded(K_areas: float, *, sigmoid_tensions: bool = False) -> Callable:
    r"""Factory for the bounded E2 energy, in the **same** convention as the periodic one.

    .. math::
        E_2 = \sum_{\text{edges}}\Lambda_e\,\ell_e
            + \tfrac{K}{2}\sum_\alpha (A_\alpha - A^0_\alpha)^2 .

    In this convention:

    * ``K_areas`` is an explicit argument, so the area-stiffness value used is always visible
      at the call site, and matches ``K_areas`` used by every periodic driver.
    * Target areas are taken **per cell** from ``face_params``, so heterogeneous target areas
      are supported.
    * ``sigmoid_tensions`` is **off by default**: ``he_params`` are the raw physical tensions,
      as in the periodic case. When enabled, ``he_params`` is instead passed through
      ``sigmoid(he_params) + 1``, confining tensions to ``(1, 2)``; this is an opt-in option
      because it usefully enforces positivity for inverse-design runs, but it makes
      ``he_params`` mean a different quantity than in the periodic case, so it must be
      requested explicitly and reported.
    """

    @jit
    def energy_line_tensions_bounded(
        vertTable: Array,
        angTable: Array,
        heTable: Array,
        faceTable: Array,
        _selected_verts: Array | None,
        _selected_hes: Array | None,
        _selected_faces: Array | None,
        _vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        """E2 energy for bounded meshes: elastic penalty on cell areas + line tensions x edge lengths."""
        num_faces = faceTable.shape[0]
        faces = jnp.arange(num_faces)
        num_edges = angTable.size
        num_half_edges = num_edges * 2
        unique_edges = jnp.arange(num_edges) * 2
        edges = jnp.arange(num_half_edges)
        vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), vertTable])
        angTable = jnp.repeat(angTable, 2)
        if sigmoid_tensions:
            he_params = jax.nn.sigmoid(he_params) + 1
        target_areas = jnp.broadcast_to(jnp.asarray(face_params).reshape(-1)[:1], (num_faces,)) \
            if jnp.asarray(face_params).size == 1 else jnp.asarray(face_params).reshape(-1)[:num_faces]

        def mapped_fn_area(face: Array, target_area: Array) -> Array:
            return _cell_area_energy(face, target_area, vertTable, angTable, heTable, faceTable)

        cell_area_energies = jnp.sum(vmap(mapped_fn_area)(faces, target_areas))

        def mapped_fn_inner(edge: Array, tension: Array) -> Array:
            return _inner_edge_energy(edge, tension, vertTable, heTable)

        inner_edge_energies = jnp.sum(vmap(mapped_fn_inner)(unique_edges, he_params))

        def mapped_fn_surface(edge: Array, tension: Array) -> Array:
            return _surface_edge_energy(edge, tension, vertTable, angTable, heTable)

        surface_edge_energies = jnp.sum(vmap(mapped_fn_surface)(edges, jnp.repeat(he_params, 2)))
        return (0.5 * K_areas) * cell_area_energies + inner_edge_energies + surface_edge_energies

    return energy_line_tensions_bounded
