# Plotting and saving

## Plotting a mesh

[`plot_mesh`][vertax.meshes.plot.plot_mesh] draws a periodic or a bounded mesh with matplotlib and
returns the figure and the axes. It does not display the figure unless `show=True`, and does not close
it, so you can save it, annotate it or close it yourself. With `save=True` it also saves it to
`save_path`.

Three enumerations choose what is drawn:

- `face_plot`, a [`FacePlot`][vertax.meshes.plot.FacePlot]: `MULTICOLOR` (the default), `FACE_PARAMETER`,
  `AREA`, `PERIMETER`, `WHITE`, or `FATES` (the fate markers of a bounded mesh; a periodic mesh is
  then drawn white);
- `edge_plot`, an [`EdgePlot`][vertax.meshes.plot.EdgePlot]: `BLACK` (the default), `EDGE_PARAMETER`,
  `LENGTH`, or `INVISIBLE`;
- `vertex_plot`, a [`VertexPlot`][vertax.meshes.plot.VertexPlot]: `INVISIBLE` (the default), `BLACK`,
  or `VERTEX_PARAMETER`.

Color-coded quantities get a color bar. The `*_parameters_name` arguments label the color bars of
the parameters, the `*_cmap_name` arguments choose the color maps, and the `forced_*_scale`
arguments fix their ranges, for example to compare several plots.

```python
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vertax import EdgePlot, FacePlot, PbcMesh, VertexPlot, plot_mesh

side = math.sqrt(20)
mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
mesh.edges_params = jnp.repeat(1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_edges,)), 2)
mesh.faces_params = jnp.ones(mesh.nb_faces)

fig, ax = plot_mesh(mesh, title="Cells in random colors")
fig.savefig("multicolor.png")
plt.close(fig)

fig, ax = plot_mesh(mesh, face_plot=FacePlot.AREA, edge_plot=EdgePlot.EDGE_PARAMETER,
                    edge_parameters_name="tension", vertex_plot=VertexPlot.BLACK,
                    title="Areas and tensions")
fig.savefig("areas_and_tensions.png")
plt.close(fig)
```

`get_plot_mesh` takes the same choices without the saving and color-bar options, and returns the
figure and the axes.

## Saving a mesh

[`save_mesh`][vertax.meshes.pbc_mesh.PbcMesh.save_mesh]`(path)` writes the tables, the box size and
the three parameter arrays to a compressed NumPy `.npz` file (and, for a periodic mesh,
`MAX_EDGES_IN_ANY_FACE`; for a bounded mesh, the angles). `load_mesh(path)` reads it back:

```python
mesh.save_mesh("saved/mesh.npz")
same = PbcMesh.load_mesh("saved/mesh.npz")
print(bool(jnp.all(same.vertices == mesh.vertices)), bool(jnp.all(same.edges_params == mesh.edges_params)))
```

`save_mesh_txt(directory)` writes one text file per table and per parameter array, plus a file of
constants (`width`, `height`, and for a periodic mesh `MAX_EDGES_IN_ANY_FACE`), with names that can
be changed by keyword arguments; `load_mesh_txt(directory)` reads them back. The parameter arrays are
read as floats.

```python
mesh.save_mesh_txt("saved/text")
from_text = PbcMesh.load_mesh_txt("saved/text")
print(bool(jnp.allclose(from_text.edges_params, mesh.edges_params)))
```

!!! note "TODO (H. Turlier)"
    The docstrings of `save_mesh_txt` and `load_mesh_txt` say that they handle "only the vertices,
    edges and faces, not other parameters"; on main they also write and read the parameter arrays
    and the constants.

`BoundedMesh.load_mesh` and `BoundedMesh.load_mesh_txt` check the orientation of the faces, as
described in [The two mesh types](meshes.md#bounded-meshes).

## Saving during an inverse problem

[`do_n_bilevel_optimization`][vertax.bilevelopt.bilevelopt._BilevelOptimizer.do_n_bilevel_optimization]
can save a plot of the mesh every `save_plotmesh_every` epochs (in `meshes_plot/`) and the mesh itself
every `save_mesh_data_every` epochs (in `meshes_data/`), in `save_folder`. See the
[many epochs example](../examples/many-epochs.md).
