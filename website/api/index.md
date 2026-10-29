# API reference

These pages are generated from the docstrings of the code, one page per module. The names that the
package `vertax` exports are listed at the end of `vertax/__init__.py` (`__all__`), and can be
imported from `vertax` directly, for example `from vertax import PbcMesh, cost_v2v`.

| Module | Contents |
| --- | --- |
| [`vertax.meshes.mesh`](mesh.md) | `Mesh`, the base class of the meshes |
| [`vertax.meshes.pbc_mesh`](pbc-mesh.md) | `PbcMesh`, periodic meshes |
| [`vertax.meshes.bounded_mesh`](bounded-mesh.md) | `BoundedMesh`, bounded meshes with arcs |
| [`vertax.meshes.plot`](plot.md) | `plot_mesh`, `get_plot_mesh`, `FacePlot`, `EdgePlot`, `VertexPlot` |
| [`vertax.bilevelopt.bilevelopt`](bilevel-optimizer.md) | `_BilevelOptimizer`, the settings and methods shared by the optimizers |
| [`vertax.bilevelopt.pbcbop`](pbc-bilevel-optimizer.md) | `PbcBilevelOptimizer` |
| [`vertax.bilevelopt.boundedbop`](bounded-bilevel-optimizer.md) | `BoundedBilevelOptimizer` |
| [`vertax.bilevelopt.rheobop`](shear-rheology-optimizer.md) | `ShearRheologyOptimizer`, `ShearState` |
| [`vertax.method_enum`](method-enum.md) | `BilevelOptimizationMethod`, `EPVariant` |
| [`vertax.energy`](energy.md) | energies: shape factor, line tensions |
| [`vertax.cost`](cost.md) | costs: vertex to vertex, images, bounded design, topological distance |
| [`vertax.opt`](opt.md) | relaxation (`minimize`) and the four estimators, periodic meshes |
| [`vertax.opt_bounded`](opt-bounded.md) | relaxation and the four estimators, bounded meshes |
| [`vertax.cg`](cg.md) | `cg_linesearch`, `gd_linesearch` |
| [`vertax.topo`](topo.md) | T1 transitions |
| [`vertax.geo`](geo.md) | lengths, perimeters and areas |
| [`vertax.rheology`](rheology.md) | shear rheometry and its diagnostics |
| [`vertax.trace`](trace.md) | tracing a label mask into a mesh |
| [`vertax.clamped`](clamped.md) | mechanics of a clamped patch |
| [`vertax.mask_analysis`](mask-analysis.md) | segmentation and mask helpers used by `PbcMesh.from_mask` |

## The package

::: vertax
    options:
      members: false
