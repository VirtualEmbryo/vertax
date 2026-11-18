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
| [`vertax.energy`](energy.md) | energies: shape factor, line tensions (periodic and bounded) |
| [`vertax.cost`](cost.md) | costs and measurements: `cost_v2v`, the Hessian-weighted cost $\mathcal C_W$ (`hessian_at_target`, `make_cost_hessian_weighted`) and its staged schedule (`switch_cost_at_target_topology`), `cost_areas`, the bounded design costs (`aspect_ratio_along_x`, `aspect_ratio_principal`, `make_cost_aspect_ratio_along_x`, `make_cost_aspect_ratio`, `cost_ratio`, `cost_checkerboard`), `topological_distance` |
| [`vertax.opt`](opt.md) | relaxation (`minimize`) and the four estimators, periodic meshes |
| [`vertax.opt_bounded`](opt-bounded.md) | relaxation and the four estimators, bounded meshes, with the reports of the T1 tension reset (`T1Resets`, `T1ResetReport`) |
| [`vertax.cg`](cg.md) | `cg_linesearch`, `gd_linesearch` |
| [`vertax.topo`](topo.md) | T1 transitions: `update_T1`, `update_T1_bounded` (with `reopen_margin`), `T1TensionReset` |
| [`vertax.geo`](geo.md) | lengths, perimeters, areas, and the second moments of a bounded cluster (`get_cluster_moments_bounded`) |
| [`vertax.rheology`](rheology.md) | shear rheometry and its diagnostics |
| [`vertax.trace`](trace.md) | tracing a label mask into a mesh |
| [`vertax.clamped`](clamped.md) | mechanics of a clamped patch |
| [`vertax.mask_analysis`](mask-analysis.md) | segmentation and mask helpers used by `PbcMesh.from_mask` |

## The package

::: vertax
    options:
      members: false
