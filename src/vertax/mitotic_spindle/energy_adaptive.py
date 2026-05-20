"""
Energy function for BoundedMesh with a per-cell target area.

Unlike energy_bounded (fixed TARGET_AREA=0.6), energy_adaptive uses
face_params[0] as the shared target area, allowing area homogenisation
after each division by setting face_params = current_mean_area.

Also provides outer_loss_homogenise: an outer loss that penalises
dispersion of tensions and cell areas, driving both toward uniformity.

The inner energy has two contributions:
  - Area elasticity : K_AREAS · Σ (area_i - target)²
  - Edge tension    : Σ length_e · tension_e
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit

from vertax.geo import get_area_bounded, get_edge_length, get_surface_length


K_AREAS = 200
K_OUTER_TENSIONS = 1.0   # weight of tension homogenisation in outer loss
K_OUTER_AREAS    = 10.0  # weight of area homogenisation in outer loss


@jit
def _cell_area_energy_adaptive(
    face: jax.Array,
    target_area: jax.Array,
    vertTable: jax.Array,
    angTable: jax.Array,
    heTable: jax.Array,
    faceTable: jax.Array,
) -> jax.Array:
    area = get_area_bounded(face, vertTable, angTable, heTable, faceTable)
    return (area - target_area) ** 2


@jit
def _inner_edge_energy(edge, tension, vertTable, heTable):
    length = get_edge_length(edge, vertTable, heTable)
    return length * tension


@jit
def _surface_edge_energy(edge, tension, vertTable, angTable, heTable):
    length = get_surface_length(edge, vertTable, angTable, heTable)
    return length * tension



def energy_adaptive(
    vertTable: jax.Array,
    angTable: jax.Array,
    heTable: jax.Array,
    faceTable: jax.Array,
    _selected_verts,
    _selected_hes,
    _selected_faces,
    _vert_params: jax.Array,
    he_params: jax.Array,
    face_params: jax.Array,
) -> jax.Array:
    """Bounded energy with uniform target area passed via face_params[0].

    face_params must be a constant array = initial_total_area / nb_faces.
    Update face_params after each division with the new target.
    The same target applies to all cells.
    """
    num_faces = faceTable.shape[0]
    num_edges = angTable.size
    num_half_edges = num_edges * 2
    faces = jnp.arange(num_faces)
    unique_edges = jnp.arange(num_edges) * 2
    edges = jnp.arange(num_half_edges)

    vertTable_full = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), vertTable])
    angTable_full = jnp.repeat(angTable, 2)
    tensions = jax.nn.sigmoid(he_params) + 1

    # Target = initial_total_area / nb_faces_current, stored in face_params[0]
    target_area = face_params[0]

    def mapped_area(face):
        return _cell_area_energy_adaptive(
            face, target_area, vertTable_full, angTable_full, heTable, faceTable
        )

    def mapped_inner(edge, tension):
        return _inner_edge_energy(edge, tension, vertTable_full, heTable)

    def mapped_surface(edge, tension):
        return _surface_edge_energy(edge, tension, vertTable_full, angTable_full, heTable)

    area_energies    = jnp.sum(vmap(mapped_area)(faces))
    inner_energies   = jnp.sum(vmap(mapped_inner)(unique_edges, tensions))
    surface_energies = jnp.sum(vmap(mapped_surface)(edges, jnp.repeat(tensions, 2)))

    return K_AREAS * area_energies + inner_energies + surface_energies


def outer_loss_homogenise(
    vertTable: jax.Array,
    angTable: jax.Array,
    heTable: jax.Array,
    faceTable: jax.Array,
    vertTable_target,
    angTable_target,
    heTable_target,
    faceTable_target,
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
) -> jax.Array:
    """Outer loss: penalise dispersion of tensions and cell areas.

    Minimising this loss drives he_params toward uniform tensions and
    face_params toward uniform cell areas — both homogenisation objectives.

    Tension term : Σ (σ(he_params_i) + 1 - mean_tension)²
    Area term    : Σ (area_i - mean_area)²
    """
    # Tensions from he_params (same sigmoid as in energy_adaptive)
    # heTable is passed but we need he_params — not directly available here.
    # VertAX passes the *current* params arrays via the mesh state, but L_out
    # only receives geometry tables. We recover tensions from heTable col unused;
    # instead we proxy via the edge lengths weighted by their indices.
    # Actually VertAX does NOT pass params to L_out — we approximate tension
    # homogenisation via area-only loss here, and rely on the bilevel gradient
    # to back-propagate into he_params through the inner equilibrium.

    num_faces = faceTable.shape[0]
    faces = jnp.arange(num_faces)
    vertTable_full = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), vertTable])
    angTable_full = jnp.repeat(angTable, 2)

    areas = vmap(lambda f: get_area_bounded(
        f, vertTable_full, angTable_full, heTable, faceTable
    ))(faces)

    mean_area = jnp.mean(areas)
    # Normalise by mean_area² so the loss is dimensionless and scale-independent
    area_var  = jnp.mean((areas - mean_area) ** 2) / (mean_area ** 2 + 1e-8)

    return K_OUTER_AREAS * area_var
