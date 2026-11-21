# Figures

The images of the README and of the documentation website (`website/figures` is a link to this folder).

The figures drawn from a computation are made by `scripts/make_doc_figures.py`, one function per figure, with the settings of the page that shows it and fixed seeds. From the root of the clone, `uv run python scripts/make_doc_figures.py` draws them all on the CPU in a few minutes; `uv run python scripts/make_doc_figures.py costs` draws those whose name contains `costs`. The schematics are hand-written SVG files, edited by hand. `concept.png`, `periodic_bounded.png` and `vertax_text.png` are drawings made outside this repository.

All figures share one set of colours, defined at the top of the script and used in the SVG files: edges `#333333`, cells `#EEEBE3` (and `#D3CBB8` for a second kind of cell), vertices and their displacements `#C98500`, half-edges `#2B8CC4`, the edge of a T1 transition `#B8559A`, a reference geometry (target, observed or initial state) dashed `#9A9A9A`, cell quantities in `viridis` and line tensions in `plasma`. The four gradient estimators keep one hue and one marker: AD `#3953a4` circle, Sen-ID `#8753FF` triangle, Adj-ID `#ed2024` square, EP `#0c8140` diamond. Each figure has a white background; on the dark scheme of the site, `website/stylesheets/extra.css` puts it on a white plate.

| Figure | Page | Source |
| --- | --- | --- |
| `concept.png` | Home, and the README | drawing |
| `periodic_bounded.png` | The two mesh types, and the README | drawing |
| `mesh-half-edge-structure.svg` | The two mesh types | hand-written SVG |
| `energy-shape-index.svg` | Energies | `energy_shape_index` |
| `relaxation-solvers.svg` | Relaxation and solvers | `relaxation_solvers` |
| `t1-transition.svg` | T1 transitions | hand-written SVG |
| `bilevel-loop-estimators.svg` | Inverse problems | hand-written SVG |
| `cost-vertex-to-vertex.svg` | Costs | `costs_vertex_to_vertex` |
| `cost-topological-distance.svg` | Costs | `costs_topological_distance` |
| `rheology-shear-drive.svg` | Rheology | `rheology_drive` |
| `image-and-mask.png` | Meshes from images and masks | `images_and_masks` (from `examples/image.tif` and `examples/mask.tif`) |
| `traced-mask.svg` | Meshes from images and masks | `images_and_masks` |
| `clamped-patch-relaxation.svg` | Clamped patches | `clamped_patch` |
| `plot-mesh-multicolor.svg` | Plotting and saving | `plotting_plot_mesh` (drawn by `plot_mesh`) |
| `plot-mesh-areas-tensions.svg` | Plotting and saving | `plotting_plot_mesh` (drawn by `plot_mesh`) |
| `example-periodic-relaxation.svg` | Examples: periodic relaxation | `example_periodic_relaxation` |
| `example-bounded-relaxation.svg` | Examples: bounded relaxation | `example_bounded_relaxation` |
| `example-tension-inference.svg` | Examples: tension inference | `example_tension_inference` |
| `example-bounded-elongation.svg` | Examples: bounded elongation | `example_bounded_elongation` |
| `example-gradient-estimators.svg` | Examples: gradient estimators | `example_gradient_estimators` |
| `example-many-epochs.svg` | Examples: many epochs | `example_many_epochs` |
| `example-vertex-step-cap.svg` | Examples: line-search solvers | `example_vertex_step_cap` |
| `example-t1-counts.svg` | Examples: T1 statistics | `example_t1_counts` |
| `example-shear-compliance.svg` | Examples: shear rheometry | `example_shear_compliance` |
| `example-trace-rim-tolerance.svg` | Examples: traced patch | `example_trace_rim_tolerance` |
| `vertax_text.png` | README of the repository | drawing |

The numbers quoted in the captions of the pages are printed by the script when it draws the figure. Relaxations with T1 transitions branch on comparisons of edge lengths, so on another machine or another version of JAX a figure can differ in its details; regenerate the figures and check the captions after such a change.
