# Costs

The cost, or outer loss, says how far an equilibrium is from what you want. You give it to a bilevel
optimizer as `loss_function_outer`. VertAX provides the costs below in [`vertax.cost`](../api/cost.md),
and any differentiable JAX function with the right signature can be used instead.

| Cost | Mesh | Compares | Section |
| --- | --- | --- | --- |
| $\mathcal C_{\text{v2v}}$, `cost_v2v` | periodic | vertex positions with a target mesh | [Vertex to vertex](#vertex-to-vertex) |
| $\mathcal C_W$, `make_cost_hessian_weighted` | periodic | the same, weighted by the Hessian of the energy at the target | [Hessian-weighted](#hessian-weighted-vertex-cost-and-the-staged-schedule) |
| `cost_areas` | periodic | cell areas with a target mesh | [Cell areas](#cell-areas) |
| `make_cost_aspect_ratio_along_x`, `make_cost_aspect_ratio`, `cost_ratio`, `cost_checkerboard` | bounded | the shape of a cluster, with no target mesh | [Bounded design costs](#bounded-design-costs) |

The topological distance $d_{\mathrm{T1}}$ is a measurement, not a cost: see
[Topological distance](#topological-distance).

## The signatures

For a periodic mesh, the optimizer calls the cost with twelve arguments:

```text
cost(vertTable, heTable, faceTable, width, height,
     vertTable_target, heTable_target, faceTable_target,
     selected_verts, selected_hes, selected_faces,
     image_target) -> scalar
```

The targets are the optimizer's `vertices_target`, `edges_target` and `faces_target`; the selections
are `None` unless `only_on_vertices`, `only_on_edges` or `only_on_faces` were given. The last
argument is not used by any cost of the library: a cost you write must accept it, and the optimizer
passes its (empty by default) `image_target`. For a bounded mesh the twelve arguments are:

```text
cost(vertTable, angTable, heTable, faceTable,
     vertTable_target, angTable_target, heTable_target, faceTable_target,
     selected_verts, selected_hes, selected_faces,
     image_target) -> scalar
```

The cost receives no parameters. A term on the parameters goes in `params_regularizer` (see
[Inverse problems](inverse-problems.md#parameter-constraints)).

## Vertex to vertex

[`cost_v2v`][vertax.cost.cost_v2v] (periodic), written $\mathcal C_{\text{v2v}}$ in the paper, is the
mean of the squared distances between each selected vertex and the target vertex with the same
index, divided by two:
$\mathcal C_{\text{v2v}} = \frac{1}{2N}\sum_i \lVert \mathbf x_i - \mathbf x_i^{\text{target}} \rVert^2$,
where $N$ is the number of selected vertices. Each distance is measured to the nearest periodic image
of the target vertex. The cost needs the two meshes to have the same vertices
in the same order; after T1 transitions, a periodic optimizer with a vertex target relabels vertices
to keep that correspondence (see [T1 transitions](t1-transitions.md#vertex-labels-and-targets)).

```python
import math

import jax

jax.config.update("jax_enable_x64", True)  # the Hessian of C_W below needs float64

import jax.numpy as jnp

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
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
```

<figure markdown="span">
  ![A relaxed periodic tissue drawn over its dashed target, with an arrow from each vertex to the target vertex of the same index](../figures/cost-vertex-to-vertex.svg)
  <figcaption markdown="span">The relaxed tissue of the code above, its target (dashed) and the displacements $\mathbf x_i^{\text{target}} - \mathbf x_i$, drawn five times longer than they are, whose mean square, halved, is $\mathcal C_{\text{v2v}} = 1.1 \times 10^{-3}$.</figcaption>
</figure>

To first order, $\mathcal C_{\text{v2v}}$ is $\tfrac12\lVert\mathbf H^{+}\mathbf M\,\delta\boldsymbol\theta\rVert^2$
for a parameter error $\delta\boldsymbol\theta$, where $\mathbf H$ is the Hessian of the energy and
$\mathbf M$ the derivative of the forces with respect to the parameters. The soft modes of the tissue
(small eigenvalues of $\mathbf H$) therefore amplify a geometric error, and a small mismatch of the
vertices can come with a large error on the parameters. The next cost compensates for this.

## Hessian-weighted vertex cost and the staged schedule

[`make_cost_hessian_weighted`][vertax.cost.make_cost_hessian_weighted] builds the cost $\mathcal C_W$
of the paper for a periodic mesh. It weights the same vertex mismatch by the mechanical Hessian
$\widehat{\mathbf H}$ of the energy at the target:

$$
\mathcal C_W = \tfrac12 \bigl\lVert \widehat{\mathbf H}\,(\mathbf X - \mathbf X_{\text{target}}) \bigr\rVert^2 ,
$$

where $\mathbf X$ holds the vertex positions (each displacement is taken to the nearest periodic
image, as in `cost_v2v`). $\widehat{\mathbf H}$ is [`hessian_at_target`][vertax.cost.hessian_at_target]:
the Hessian of the energy with respect to the vertex positions, at the target geometry, with every
edge parameter set to one, symmetrized and divided by its spectral norm (its largest eigenvalue in
magnitude is 1). It needs the geometry of the target and none of the tensions you are inferring.
Near the target, $\mathcal C_W$ is about half the squared force residual, which the equilibrium does not
amplify. Compute the Hessian in `float64` (`jax_enable_x64`), because it depends on the small
eigenvalues.

The Hessian belongs to the topology of the target and changes discontinuously at a T1 transition, so
$\mathcal C_W$ is meant for a mesh that has the target topology and keeps it (`update_T1 = False`). To
infer parameters from a mesh whose topology differs from the target's, the paper uses a **staged
schedule**: minimize $\mathcal C_{\text{v2v}}$ with T1 transitions on until the topology of the mesh
first matches the target's, then switch off T1 transitions and minimize $\mathcal C_W$.
[`switch_cost_at_target_topology`][vertax.cost.switch_cost_at_target_topology] makes that switch.
Call it after every epoch: at the first epoch where $d_{\mathrm{T1}} = 0$ (see
[Topological distance](#topological-distance)), it sets `update_T1 = False` and the new cost on the
optimizer, and it returns `True`; before that, it changes nothing and returns `False`. If you cap the
first stage at a number of epochs, make the switch yourself when the cap is reached.

The example makes two targets from one tissue with the same new tensions, one relaxed without T1
transitions and one with them (it has $d_{\mathrm{T1}} = 1$), and calls the switch for each.

```python
import optax

from vertax import BilevelOptimizationMethod
from vertax.cost import make_cost_hessian_weighted, switch_cost_at_target_topology, topological_distance
from vertax.energy import make_energy_line_tensions

K_AREAS = 10.0
e2 = make_energy_line_tensions(K_AREAS)


def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)


def line_tensions(seed, n_edges, std=0.3):
    return jnp.repeat(1.0 + std * jax.random.normal(jax.random.PRNGKey(seed), (n_edges,)), 2)


start = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
start.vertices_params = jnp.zeros(start.nb_vertices)
start.edges_params = jnp.ones(start.nb_half_edges)
start.faces_params = jnp.ones(start.nb_faces)

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_solver = optax.sgd(learning_rate=0.01)
optimizer.min_dist_T1 = 0.005
optimizer.max_nb_iterations = 1000
optimizer.tolerance = 1e-4
optimizer.patience = 5
optimizer.inner_optimization(start)

targets = {}
for name, with_t1 in (("same topology", False), ("one T1 away", True)):
    optimizer.update_T1 = with_t1
    targets[name] = PbcMesh.copy_mesh(start)
    targets[name].edges_params = line_tensions(7, start.nb_edges)
    optimizer.inner_optimization(targets[name])

for name, target in targets.items():
    d_topo = topological_distance(start.vertices, start.edges, start.faces, width, height,
                                  target.vertices, target.edges, target.faces)
    print(f"{name}: d_T1 = {d_topo // 2}")

target = targets["same topology"]
cost_w = make_cost_hessian_weighted(target, energy)
print("Hessian", cost_w.hessian.shape, "largest eigenvalue", float(jnp.abs(jnp.linalg.eigvalsh(cost_w.hessian)).max()))

arguments = (start.vertices, start.edges, start.faces, width, height,
             target.vertices, target.edges, target.faces, None, None, None, None)
print("C_v2v:", float(cost_v2v(*arguments)), "  C_W:", float(cost_w(*arguments)))

for name, target in targets.items():
    optimizer.vertices_target, optimizer.edges_target, optimizer.faces_target = (
        target.vertices, target.edges, target.faces)
    optimizer.update_T1 = True
    optimizer.loss_function_outer = cost_v2v
    cost_w = make_cost_hessian_weighted(target, energy)
    second_stage = switch_cost_at_target_topology(optimizer, start, cost_w)
    print(f"{name}: second stage {second_stage}, T1 transitions {optimizer.update_T1}, "
          f"cost is C_W {optimizer.loss_function_outer is cost_w}")
```

After the switch, run the epochs as usual, here one epoch of equilibrium propagation on the tissue
that has the target topology:

```python
target = targets["same topology"]
optimizer.vertices_target, optimizer.edges_target, optimizer.faces_target = (
    target.vertices, target.edges, target.faces)
optimizer.loss_function_outer = cost_v2v
switch_cost_at_target_topology(optimizer, start, make_cost_hessian_weighted(target, energy))

optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.beta = 0.01
optimizer.outer_solver = optax.adam(learning_rate=2e-3, nesterov=True)
optimizer.mean_tension_gauge = 1.0
print("outer cost before:", float(optimizer.compute_outer_loss(start)))
optimizer.bilevel_optimization(start)
print("outer cost after one epoch:", float(optimizer.compute_outer_loss(start)))
```

`optimizer.mean_tension_gauge = 1.0` fixes the mean line tension, which the equilibrium barely
determines (see [Inverse problems](inverse-problems.md#parameter-constraints)). A real run needs
hundreds of epochs. `make_cost_hessian_weighted` accepts `selected_verts`: the sum then runs over the
rows of $\widehat{\mathbf H}(\mathbf X-\mathbf X_{\text{target}})$ at the selected vertices.

## Cell areas

[`cost_areas`][vertax.cost.cost_areas] (periodic) is the mean, over the selected faces, of the
squared difference between the area of a face and that of the same face in the target mesh, which
must have the same faces. With no selection (`None`), every face is used. It takes the
twelve-argument form, so it can be the `loss_function_outer` of a `PbcBilevelOptimizer`.

```python
import math

import jax

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_areas

side = math.sqrt(20)
target = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
moved = PbcMesh.copy_mesh(target)
moved.vertices = target.vertices + 0.05 * jax.random.normal(jax.random.PRNGKey(0), target.vertices.shape)
print(float(cost_areas(moved.vertices, moved.edges, moved.faces, side, side,
                       target.vertices, target.edges, target.faces, None, None, None, None)))

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_outer = cost_areas
optimizer.vertices_target = target.vertices
optimizer.edges_target = target.edges
optimizer.faces_target = target.faces
print(optimizer.compute_outer_loss(moved), optimizer.compute_outer_loss(moved, only_on_faces=[0, 3, 7]))
print(optimizer.compute_outer_loss(target))  # 0 at the target
```

## Bounded design costs

These costs have no target mesh: they measure the shape of a bounded cluster, and `vertices_target`
stays empty. They are used to design the line tensions that give a cluster a shape. The aspect ratios
are computed from the vertices and the arc angles.

- [`make_cost_aspect_ratio_along_x`][vertax.cost.make_cost_aspect_ratio_along_x]`(target_ratio)`
  returns the cost $(r_x - r_{\text{target}})^2$, where
  [`aspect_ratio_along_x`][vertax.cost.aspect_ratio_along_x] is $r_x = \sqrt{S_{xx}/S_{yy}}$ and
  $S_{xx}$, $S_{yy}$ are the central second moments of the region the cluster covers, circular caps of
  the free interface included ([`get_cluster_moments_bounded`][vertax.geo.get_cluster_moments_bounded]).
  The axis is fixed to $x$: for an ellipse of semi-axes $a$ along $x$ and $b$ along $y$,
  $r_x = a/b$, and an ellipse turned away from $x$ has a smaller $r_x$. Use it when the direction of
  elongation is part of the design.
- [`make_cost_aspect_ratio`][vertax.cost.make_cost_aspect_ratio]`(target_ratio)` returns
  $(r - r_{\text{target}})^2$ with [`aspect_ratio_principal`][vertax.cost.aspect_ratio_principal],
  $r = \sqrt{\lambda_1/\lambda_2}$, where $\lambda_1 \ge \lambda_2$ are the eigenvalues of the second-moment
  matrix. It does not depend on the orientation of the cluster, so the direction of elongation is free
  and can drift during the design.
- [`cost_ratio`][vertax.cost.cost_ratio] is the older measure of elongation: the longest distance
  between two vertices, divided by the width of the cluster across that direction, compared with
  `TARGET_RATIO = 2.0`, as $(\text{ratio} - 2)^2$. It uses the vertex positions only, and it changes
  abruptly when the farthest pair of vertices changes, which the two costs above do not.
- [`cost_checkerboard`][vertax.cost.cost_checkerboard] is the sum of the squared lengths of the edges
  between two cells of the same fate (the second column of `faces`), so that minimizing it separates
  neighbors of the same fate.

```python
from vertax import BoundedMesh
from vertax.cost import (aspect_ratio_along_x, aspect_ratio_principal, cost_ratio,
                         make_cost_aspect_ratio, make_cost_aspect_ratio_along_x)

cluster = BoundedMesh.from_random_seeds(nb_seeds=20, width=math.sqrt(20), height=math.sqrt(20), random_key=1)
print("r_x:", float(aspect_ratio_along_x(cluster.vertices, cluster.angles, cluster.edges)))
print("r:  ", float(aspect_ratio_principal(cluster.vertices, cluster.angles, cluster.edges)))

for name, cost in (("along x", make_cost_aspect_ratio_along_x(2.0)), ("principal axes", make_cost_aspect_ratio(2.0))):
    print(f"cost, {name}:", float(cost(cluster.vertices, cluster.angles, cluster.edges)))
print("cost_ratio:", float(cost_ratio(cluster.vertices)))
```

The first three arguments of the aspect-ratio costs are enough to call them directly, as above; the
other nine have defaults. [Bounded elongation](../examples/bounded-elongation.md) runs a design with
these costs.

## Topological distance

[`topological_distance`][vertax.cost.topological_distance] (periodic) is $d_{\text{topo}}$, the number
of cell contacts, pairs of neighboring cells, that are present in one of two meshes and not in the
other. The two meshes must have the same number of cells; their cells are put in correspondence with
the Hungarian algorithm, which minimizes the differences between the neighborhoods, because the cell
labels are arbitrary. A T1 transition removes one contact and creates another, so $d_{\text{topo}}$ is
even, and the paper reports

$$
d_{\mathrm{T1}} = d_{\text{topo}} / 2 .
$$

<figure markdown="span">
  ![Two periodic tissues side by side; four shaded cells in each, with one purple edge between two of them in the first tissue and between the two others in the second](../figures/cost-topological-distance.svg)
  <figcaption markdown="span">The starting tissue and the target one T1 transition away from the example of the staged schedule: two cells lose their contact (purple edge, left) and the two others gain one (right), so $d_{\text{topo}} = 2$ and $d_{\mathrm{T1}} = 1$.</figcaption>
</figure>

$d_{\mathrm{T1}}$ is a lower bound on the number of T1 transitions between two tissues. It is reached
for every pair of tissues used in the paper, but it need not be in general. The function returns
$d_{\text{topo}}$, a Python `int`, with no gradient: it is a measurement, not a cost for
`loss_function_outer`. Use it to measure how far a mesh is from a target topology, or to decide when
to switch stage (see above). A cost made of contacts alone would also discard the vertex positions,
which carry the information on the tensions.

## Writing your own cost

As for energies, a cost is a JAX function, and every estimator differentiates it with respect to the
vertex positions: AD and the implicit estimators at the equilibrium, EP in its nudged relaxations,
which minimize $\mathcal{E} \pm \beta \mathcal{C}$. Define the cost once and reuse the same function:
the relaxations of equilibrium propagation are compiled for a given cost.
