# Costs

The cost, or outer loss, says how far an equilibrium is from what you want. You give it to a bilevel
optimizer as `loss_function_outer`. VertAX provides costs in [`vertax.cost`](../api/cost.md), and any
differentiable JAX function with the right signature can be used instead.

## The signatures

For a periodic mesh, the optimizer calls the cost with twelve arguments:

```text
cost(vertTable, heTable, faceTable, width, height,
     vertTable_target, heTable_target, faceTable_target,
     selected_verts, selected_hes, selected_faces,
     image_target) -> scalar
```

The targets are the optimizer's `vertices_target`, `edges_target`, `faces_target` and
`image_target`; the selections are `None` unless `only_on_vertices`, `only_on_edges` or
`only_on_faces` were given. For a bounded mesh the twelve arguments are:

```text
cost(vertTable, angTable, heTable, faceTable,
     vertTable_target, angTable_target, heTable_target, faceTable_target,
     selected_verts, selected_hes, selected_faces,
     image_target) -> scalar
```

The cost receives no parameters. A term on the parameters goes in `params_regularizer` (see
[Inverse problems](inverse-problems.md#parameter-constraints)).

## Vertex to vertex

[`cost_v2v`][vertax.cost.cost_v2v] (periodic) is the mean of the squared distances between each
selected vertex and the target vertex with the same index, divided by two:
$\frac{1}{2N}\sum_i \lVert x_i - x_i^{\text{target}} \rVert^2$. Each distance is measured to the
nearest periodic image of the target vertex. The cost needs the two meshes to have the same vertices
in the same order; after T1 transitions, a periodic optimizer with a vertex target relabels vertices
to keep that correspondence (see [T1 transitions](t1-transitions.md#vertex-labels-and-targets)).

```python
import math

import jax
import jax.numpy as jnp

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v, topological_distance
from vertax.energy import energy_shape_factor_hetero

n_cells = 20
width = height = math.sqrt(n_cells)
all_faces = jnp.arange(n_cells)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return energy_shape_factor_hetero(vertTable, heTable, faceTable, width, height, all_faces, face_params)


mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
mesh.faces_params = jnp.full(n_cells, 3.7)
optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_optimization(mesh)

target = PbcMesh.copy_mesh(mesh)
target.faces_params = 3.7 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n_cells,))
optimizer.inner_optimization(target)

print("cost_v2v:", float(cost_v2v(mesh.vertices, mesh.edges, mesh.faces, width, height,
                                  target.vertices, target.edges, target.faces)))
print("d_topo:", topological_distance(mesh.vertices, mesh.edges, mesh.faces, width, height,
                                      target.vertices, target.edges, target.faces))
```

## Topological distance

[`topological_distance`][vertax.cost.topological_distance] (periodic) counts the cell contacts on
which two meshes with the same number of cells disagree, after matching their cells with the Hungarian
algorithm. A T1 transition removes one contact and creates another, so the distance is even, and half
of it is the number of T1 transitions between the two topologies. It returns a Python `int` and has
no gradient: it is a measurement, not a cost for `loss_function_outer`.

## Images

Two costs compare the mesh with an image of its junctions, so they need no correspondence of vertices.
Both are periodic.

[`make_cost_dmic`][vertax.cost.make_cost_dmic]`(sigma_psf, ...)` returns a twelve-argument cost
$\tfrac12 \lVert M \odot (\hat I - I) \rVert^2$. $\hat I$ is
[`render_interface_image`][vertax.cost.render_interface_image]: the edges of the mesh drawn as lines
of uniform density, blurred by a Gaussian of standard deviation `sigma_psf` (in the units of the
mesh, not in pixels), computed in Fourier space. $I$ is `image_target`, and $M$ an optional mask. By
default both images are normalized to unit integral, so the cost does not depend on the overall
brightness. [`cost_dmic`][vertax.cost.cost_dmic] is the same cost with `sigma_psf` as a keyword
argument whose default is 0.05; the optimizer cannot pass keywords, so use `make_cost_dmic` to set the
width.

```python
import math

import jax.numpy as jnp

from vertax import PbcMesh
from vertax.cost import make_cost_dmic, render_interface_image

side = math.sqrt(20)
mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
image = render_interface_image(mesh.vertices, mesh.edges, side, side, 64, 0.3)
print(image.shape, float(image.sum() * (side / 64) ** 2))  # integral 1

cost = make_cost_dmic(sigma_psf=0.3)
print(float(cost(mesh.vertices, mesh.edges, mesh.faces, side, side,
                 None, None, None, None, None, None, image)))  # 0 at a perfect fit
```

[`make_cost_mesh2image`][vertax.cost.make_cost_mesh2image]`(sigma_psf)` gives the older image cost
`cost_mesh2image` in the twelve-argument form. `cost_mesh2image` takes the PSF width as a thirteenth
argument, and `LEGACY_PSF_SIGMA` is the width with which earlier results were produced.

See the [image cost example](../examples/image-cost.md).

## Bounded design costs

- [`cost_ratio`][vertax.cost.cost_ratio] measures the elongation of a bounded cluster: the longest
  distance between two vertices, divided by the width of the cluster across that axis, compared with
  `TARGET_RATIO = 2.0`, as $(\text{ratio} - 2)^2$.
- [`cost_checkerboard`][vertax.cost.cost_checkerboard] adds the squared length of every inner edge
  between two cells of the same fate, so that minimizing it separates cells of the same fate.

## `cost_areas`

!!! note "TODO (H. Turlier)"
    `cost_areas` raises `TypeError: get_area() missing 1 required positional argument: 'max_iter'`
    on main: it calls `get_area` with six arguments. Its arguments are also not in the order of the
    twelve-argument signature (it takes the selections before the targets). It is left out of this
    guide until it is fixed.

The cost below compares the cell areas with those of the target, in the twelve-argument form:

```python
from jax import vmap

from vertax.geo import get_area

MAX_EDGES_IN_ANY_FACE = 20


def cost_cell_areas(vertTable, heTable, faceTable, width, height,
                    vertTable_target, heTable_target, faceTable_target,
                    selected_verts, selected_hes, selected_faces, image_target):
    faces = jnp.arange(len(faceTable)) if selected_faces is None else selected_faces

    def squared_difference(face):
        area = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        area_target = get_area(face, vertTable_target, heTable_target, faceTable_target, width, height,
                               MAX_EDGES_IN_ANY_FACE)
        return (area - area_target) ** 2

    return jnp.mean(vmap(squared_difference)(faces))


print(float(cost_cell_areas(mesh.vertices, mesh.edges, mesh.faces, side, side,
                            mesh.vertices, mesh.edges, mesh.faces, None, None, None, None)))
```

## Writing your own cost

As for energies, a cost is a JAX function, and every estimator differentiates it with respect to the
vertex positions: AD and the implicit estimators at the equilibrium, EP in its nudged relaxations,
which minimize $\mathcal{E} \pm \beta \mathcal{C}$. Define the cost once and reuse the same function:
the relaxations of equilibrium propagation are compiled for a given cost.
