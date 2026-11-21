# Energies

The energy, or inner loss, defines mechanical equilibrium: a relaxation moves the vertices towards
one of its minima. You give it to a bilevel optimizer as `loss_function_inner`. VertAX provides a
few energies in [`vertax.energy`](../api/energy.md), and any differentiable JAX function with the
right signature can be used instead.

## The signatures

For a periodic mesh, the optimizer calls the energy with six arguments, the three tables of the mesh
and its three parameter arrays:

```text
energy(vertTable, heTable, faceTable, vert_params, he_params, face_params) -> scalar
```

The box size is not passed. The built-in periodic energies need it, so you wrap them in a function
that closes over `width` and `height`.

For a bounded mesh, the energy takes ten arguments: the tables, including the arc angles, the
selections of vertices, half-edges and faces (`None` when the whole mesh is used), and the parameters:

```text
energy(vertTable, angTable, heTable, faceTable,
       selected_verts, selected_hes, selected_faces,
       vert_params, he_params, face_params) -> scalar
```

The names do not matter, but the number and the order of the arguments do. An energy may ignore some
of them.

Give each parameter array one entry per vertex, per half-edge (or per edge, for the bounded
line-tension energy) and per face. The periodic relaxation passes the energy the rows selected with
`only_on_vertices`, `only_on_edges` and `only_on_faces`, and it selects the parameter arrays with the
same indices.

## Shape-factor energy (periodic)

$$
E_1 = \sum_\alpha \lambda_A (A_\alpha - 1)^2 + (P_\alpha - p^0_\alpha)^2
$$

$A_\alpha$ and $P_\alpha$ are the area and the perimeter of cell $\alpha$, and $p^0_\alpha$ its
preferred perimeter, read from `face_params`. The target area is 1, so a box of area $N$ holds $N$
cells of mean area 1.

- [`energy_shape_factor_homo`][vertax.energy.energy_shape_factor_homo]`(vertTable, heTable, faceTable, width, height, face_params)`
  has $\lambda_A = 1$ and broadcasts `face_params` to all cells, so a single value gives every cell the
  same preferred perimeter.
- [`energy_shape_factor_hetero`][vertax.energy.energy_shape_factor_hetero]`(vertTable, heTable, faceTable, width, height, selected_faces, face_params)`
  has $\lambda_A = 1$ and one preferred perimeter per cell; it sums over the cells in
  `selected_faces`.
- [`make_energy_shape_factor`][vertax.energy.make_energy_shape_factor]`(lambda_area=1.0)` returns an
  energy with the signature of `energy_shape_factor_hetero` and the area stiffness `lambda_area`. The
  stiffness is closed over without conversion to a Python float, so it can be a traced array and be
  optimized; at `lambda_area = 1` the result is `energy_shape_factor_hetero`.

<figure markdown="span">
  ![Two relaxed periodic tissues of 64 cells coloured by shape index: at p0 = 3.5 the shape indices range from 3.7 to 4.4, at p0 = 3.9 they are all close to 3.9](../figures/energy-shape-index.svg)
  <figcaption markdown="span">Relaxed 64-cell tissues of $E_1$ with every cell at the same $p_0$, coloured by the shape index $P/\sqrt{A}$: at $p_0 = 3.5$ the shape indices stay well above $p_0$ (3.70 to 4.43, $E_1 = 12.2$), at $p_0 = 3.9$ they gather around it (3.88 to 3.96, $E_1 = 0.008$); L-BFGS relaxations in `float64` to a force norm below $10^{-13}$.</figcaption>
</figure>

```python
import math

import jax.numpy as jnp

from vertax import PbcMesh
from vertax.energy import energy_shape_factor_hetero, energy_shape_factor_homo, make_energy_shape_factor

n_cells = 20
width = height = math.sqrt(n_cells)
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
p0 = jnp.full(mesh.nb_faces, 3.7)
all_faces = jnp.arange(mesh.nb_faces)

e_homo = energy_shape_factor_homo(mesh.vertices, mesh.edges, mesh.faces, width, height, jnp.asarray([3.7]))
e_hetero = energy_shape_factor_hetero(mesh.vertices, mesh.edges, mesh.faces, width, height, all_faces, p0)
e_stiff = make_energy_shape_factor(lambda_area=5.0)(mesh.vertices, mesh.edges, mesh.faces, width, height, all_faces, p0)
print(float(e_homo), float(e_hetero), float(e_stiff))


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    """The form a periodic optimizer calls: the box size is closed over."""
    return energy_shape_factor_hetero(vertTable, heTable, faceTable, width, height, all_faces, face_params)
```

## Line-tension energy (periodic)

[`make_energy_line_tensions`][vertax.energy.make_energy_line_tensions]`(K_areas)` returns the energy

$$
E_2 = \tfrac12 \sum_{h} \Lambda_h \ell_h + \tfrac{K}{2} \sum_\alpha (A_\alpha - A^0_\alpha)^2 ,
$$

where the first sum runs over half-edges, $\Lambda_h$ is read from `he_params` and $\ell_h$ is the
length, and $A^0_\alpha$ is the target area of cell $\alpha$, read from `face_params`. Each edge is
counted twice, once per half-edge, and the factor $\tfrac12$ compensates, so `he_params` holds the
physical tension of the edge. Give both half-edges of an edge the same tension, for instance with
`jnp.repeat(tensions, 2)`: the two half-edges of edge `k` are rows `2k` and `2k + 1` of `edges` for
meshes built by `from_seeds`.

`K_areas` has no default and must be passed.

The function returned has seven arguments,
`(vertTable, heTable, faceTable, width, height, he_params, face_params)`, so it cannot be given to an
optimizer as it is. Wrap it:

```python
import math

import jax
import jax.numpy as jnp

from vertax import PbcMesh
from vertax.energy import make_energy_line_tensions

n_cells = 20
width = height = math.sqrt(n_cells)
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)

K_AREAS = 10.0
e2 = make_energy_line_tensions(K_AREAS)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)


tensions = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_edges,))
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.repeat(tensions, 2)  # one value per half-edge, equal on twins
mesh.faces_params = jnp.ones(mesh.nb_faces)  # target areas

print(float(energy(mesh.vertices, mesh.edges, mesh.faces, mesh.vertices_params, mesh.edges_params, mesh.faces_params)))
```

## Line-tension energy (bounded)

[`make_energy_line_tensions_bounded`][vertax.energy.make_energy_line_tensions_bounded]`(K_areas, *, sigmoid_tensions=False)`
returns, in the ten-argument bounded signature, the energy

$$
E_2 = \sum_e \Lambda_e \ell_e + \tfrac{K}{2} \sum_\alpha (A_\alpha - A^0_\alpha)^2 .
$$

The sum runs over edges, not half-edges: `he_params` has one tension per edge (`nb_edges` values).
The length of an outline edge is the length of its arc. `face_params` holds the target areas, one per
cell, or a single value for all cells. The selection arguments are ignored.

With `sigmoid_tensions=True`, the tensions are `sigmoid(he_params) + 1`, between 1 and 2, instead of
`he_params` itself. This keeps them positive in inverse design, but `he_params` then means something
else than in the periodic energy; it is off by default.

```python
import math

import jax.numpy as jnp

from vertax import BoundedMesh
from vertax.energy import make_energy_line_tensions_bounded

n_cells = 20
width = height = math.sqrt(n_cells)
cluster = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)

energy = make_energy_line_tensions_bounded(K_areas=40.0)
tensions = jnp.ones(cluster.nb_edges)
target_areas = jnp.full(cluster.nb_faces, 0.6)
value = energy(cluster.vertices, cluster.angles, cluster.edges, cluster.faces, None, None, None,
               jnp.zeros(cluster.nb_vertices), tensions, target_areas)
print(float(value))
```

## Writing your own energy

An energy is a JAX function, so it can use the geometric functions of [`vertax.geo`](../api/geo.md).
For a periodic mesh, `get_area(face, vertTable, heTable, faceTable, width, height, max_iter)` and
`get_perimeter(...)` walk around a face for at most `max_iter` half-edges (VertAX uses 20), and
`get_length(he, vertTable, heTable, faceTable, width, height)` measures one half-edge. Map them over
faces and half-edges with `jax.vmap`.

This is the periodic line-tension energy written by hand; it gives the same value as the factory:

```python
import math

import jax
import jax.numpy as jnp
from jax import vmap

from vertax import PbcMesh
from vertax.energy import make_energy_line_tensions
from vertax.geo import get_area, get_length

n_cells = 20
width = height = math.sqrt(n_cells)
K_AREAS = 10.0
MAX_EDGES_IN_ANY_FACE = 20
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)


def my_energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    def area_term(face, target_area):
        area = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        return (area - target_area) ** 2

    def tension_term(he, tension):
        return tension * get_length(he, vertTable, heTable, faceTable, width, height)

    areas = vmap(area_term)(jnp.arange(len(faceTable)), face_params)
    lengths = vmap(tension_term)(jnp.arange(len(heTable)), he_params)
    return 0.5 * jnp.sum(lengths) + 0.5 * K_AREAS * jnp.sum(areas)


he_params = jnp.repeat(1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_edges,)), 2)
face_params = jnp.ones(mesh.nb_faces)
mine = my_energy(mesh.vertices, mesh.edges, mesh.faces, None, he_params, face_params)
factory = make_energy_line_tensions(K_AREAS)(mesh.vertices, mesh.edges, mesh.faces, width, height, he_params, face_params)
print(float(mine), float(factory))
assert jnp.allclose(mine, factory)
```

Define the energy once and reuse the same function object: the relaxation is compiled for a given
energy function, and a new function recompiles it (see [Relaxation and solvers](relaxation.md)).

The clamped patches of [`vertax.clamped`](clamped-patches.md) use energies of another form,
`E(x, K)`, described on their own page.
