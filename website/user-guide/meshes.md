# The two mesh types

A mesh in VertAX is a vertex model of a tissue: faces are cells, edges are the junctions between two
cells, and vertices are the points where three or more cells meet. VertAX has two kinds of meshes.

| | [`PbcMesh`][vertax.meshes.pbc_mesh.PbcMesh] | [`BoundedMesh`][vertax.meshes.bounded_mesh.BoundedMesh] |
| --- | --- | --- |
| Use | a piece of bulk tissue, with no boundary | a finite cluster of cells |
| Boundary | periodic, in a box of size `width` x `height` | an outline of circular arcs |
| Degrees of freedom | vertex positions | vertex positions and one angle per outline arc |
| Built from | a Voronoi diagram of seeds (or a mask, see below) | a Voronoi diagram of seeds |
| Optimizer | `PbcBilevelOptimizer` | `BoundedBilevelOptimizer` |

![A periodic mesh and a bounded mesh](../figures/periodic_bounded.png){ width="220" }

## What the two have in common

Both store the mesh as a half-edge structure: every edge is made of two half-edges, one in each of
the two faces it separates. The tables are JAX arrays:

- `vertices`: the positions, an array of shape `(nb_vertices, 2)`. Its dtype is JAX's default float
  dtype when the mesh is built: `float32`, or `float64` under `jax_enable_x64` (see
  [Numerical precision](precision.md)).
- `edges`: one row of 8 integers per half-edge, with the previous, next and twin half-edges, the
  vertices, and the face. The column layout differs between the two mesh types; it is given in the
  docstrings of [`PbcMesh`][vertax.meshes.pbc_mesh.PbcMesh] and
  [`BoundedMesh`][vertax.meshes.bounded_mesh.BoundedMesh].
- `faces`: for each face, one half-edge of that face (and, for a bounded mesh, a fate marker; see
  below).
- `width` and `height`: the size of the box.

<figure markdown="span">
  ![Hexagonal cells with a half-edge h of face f, its twin in face g, the next and previous half-edges, its source and target vertices, and a half-edge crossing the right side of a periodic box](../figures/mesh-half-edge-structure.svg)
  <figcaption markdown="span">The half-edge tables: a half-edge `h` of face *f* and its twin in the neighbouring face *g*, the next and previous half-edges around *f*, the source and target vertices of `h`, and, for a periodic mesh, a half-edge whose target is stored on the other side of the box and read one box width further, as its offset (+1, 0) in columns 6 and 7 says (schematic).</figcaption>
</figure>

The properties `nb_vertices`, `nb_half_edges`, `nb_edges` (half the number of half-edges) and
`nb_faces` give the sizes.

A mesh also carries three parameter arrays, `vertices_params`, `edges_params` and `faces_params`.
VertAX gives them no meaning of its own: the energy and the cost decide what they are, for example a
preferred perimeter per face or a tension per half-edge. Set them before the first relaxation; an
inverse problem optimizes them.

Meshes have no public constructor. Calling `PbcMesh()` raises a `TypeError`; use the class methods
below.

## Periodic meshes

[`PbcMesh.from_random_seeds`][vertax.meshes.pbc_mesh.PbcMesh.from_random_seeds] draws `nb_seeds`
uniform seeds in the box with `jax.random.PRNGKey(random_key)`, and
[`PbcMesh.from_seeds`][vertax.meshes.pbc_mesh.PbcMesh.from_seeds] takes the seeds as an array. The
mesh is the Voronoi diagram of the seeds and of their eight periodic copies, so there is one cell per
seed.

```python
import math

import jax.numpy as jnp

from vertax import PbcMesh

n_cells = 20
width = height = math.sqrt(n_cells)  # mean cell area 1
mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)

print(mesh.vertices.shape, mesh.vertices.dtype)  # (40, 2) float32
print(mesh.edges.shape, mesh.faces.shape)        # (120, 8) (20,)
print(mesh.nb_faces, mesh.nb_edges, mesh.nb_vertices)

faces = jnp.arange(mesh.nb_faces)
print("total area:", float(mesh.get_area(faces).sum()), "=", width * height)
print("mean perimeter:", float(mesh.get_perimeter(faces).mean()))
```

The construction prints a warning below 20 seeds. At small cell numbers some seed sets give a
degenerate Voronoi cell, and the construction then raises an error that says so; the message
suggests more seeds, a box closer to square, or another `random_key`.

`get_length`, `get_perimeter` and `get_area` take an array of half-edge or face indices and return
one value per index. For a periodic mesh, `get_area` returns the absolute value of the shoelace
area. Edges that cross the box boundary are measured through the boundary: columns 6 and 7 of
`edges` say by how many box widths and heights the target vertex of a half-edge is shifted.

[`PbcMesh.copy_mesh`][vertax.meshes.pbc_mesh.PbcMesh.copy_mesh] makes an independent copy, with its
parameters. [`PbcMesh.from_mask`][vertax.meshes.pbc_mesh.PbcMesh.from_mask] and `from_image` build a
periodic mesh from a segmentation; they are not reliable on real masks, see
[Meshes from images and masks](images-and-masks.md).

## Bounded meshes

[`BoundedMesh.from_random_seeds`][vertax.meshes.bounded_mesh.BoundedMesh.from_random_seeds] draws
seeds with `numpy.random.default_rng(random_key)` and keeps the bounded cells of their Voronoi
diagram that lie inside the box. Cells connected to only one other cell are removed, and the
construction adds or redraws seeds until exactly `nb_seeds` connected cells remain. The final mesh
may therefore not use the seeds you gave to
[`BoundedMesh.from_seeds`][vertax.meshes.bounded_mesh.BoundedMesh.from_seeds].

```python
import math

import jax.numpy as jnp

from vertax import BoundedMesh

n_cells = 20
width = height = math.sqrt(n_cells)
cluster = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)

print(cluster.vertices.shape, cluster.edges.shape)  # (38, 2) (114, 8)
print(cluster.faces.shape)                          # (20, 2): a half-edge and a fate per cell
print(cluster.angles.shape, cluster.nb_angles == cluster.nb_edges)
print("fates:", jnp.unique(cluster.faces[:, 1]))

faces = jnp.arange(cluster.nb_faces)
print("areas:", cluster.get_area(faces))
```

What is specific to bounded meshes:

- `angles` has one entry per edge. For an edge of the outline, it is the half-angle of the circular
  arc: 0 is a straight chord and $\pi/2$ a half circle. The angles of inner edges are not used. The
  initial angles of the outline edges are drawn at random between 0.017 and $\pi/2 - 0.001$, and the
  relaxation keeps every angle in that range.
- `faces` has two columns: a half-edge of the face, and a fate marker, an integer from 0 to
  `nb_fates - 1` (`nb_fates=2` by default), assigned at random so that the fates have the same
  number of cells, give or take one. The fate is used by
  [`cost_checkerboard`][vertax.cost.cost_checkerboard] and by the plot option `FacePlot.FATES`.
- Every face is listed counterclockwise, and the arc of an outline edge bulges away from its cell.
  [`get_area`][vertax.meshes.bounded_mesh.BoundedMesh.get_area] relies on this: it returns the signed
  area of the polygon of the cell's vertices plus the areas between each arc and its chord. A folded
  cell has a non-positive area.
- `from_seeds`, `load_mesh` and `load_mesh_txt` check the orientation. A mesh whose faces are
  clockwise raises a `ValueError`; a mesh with a face whose vertex polygon has turned over (a
  crescent, or a fold) gives a `UserWarning`.

## Saving and plotting

See [Plotting and saving](plotting-and-saving.md).
