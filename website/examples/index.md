# Examples

Each page is one complete script, built from the notebooks in `examples/` and from the tests in
`tests/`. The scripts make their own meshes from random seeds, except the last one, which reads the
segmentation mask and the patch that the tests use; its paths are relative to the root of the clone.
Sizes are small, so that each page runs in a few minutes at most on one CPU thread.

| Page | What it shows | Built from |
| --- | --- | --- |
| [Periodic relaxation](periodic-relaxation.md) | relaxing 100 cells, and checking the result without comparing coordinates | `tests/test_forward_modeling.py` |
| [Bounded relaxation](bounded-relaxation.md) | relaxing a 20-cell cluster with curved outline | `tests/test_forward_modeling_bounded.py` |
| [Tension inference](tension-inference.md) | recovering line tensions from equilibrium vertex positions, with EP | `examples/inverse_modelling_example.ipynb`, `tests/test_inverse_modeling.py` |
| [Bounded elongation](bounded-elongation.md) | designing tensions so that a cluster elongates | `examples/inverse_modelling_example_bounded.ipynb`, `tests/test_inverse_modeling_bounded.py` |
| [Gradient estimators](gradient-estimators.md) | one outer step with AD, Sen-ID, Adj-ID and EP from the same state | `tests/test_cap_vertex_step.py`, `tests/test_attribute_dispatch_guard.py` |
| [Many epochs](many-epochs.md) | `do_n_bilevel_optimization` with reports, saved files and a custom metric | `tests/test_n_bilevel_opt.py` |
| [Line-search solvers](line-search-solvers.md) | `gd_linesearch`, the vertex step cap, and L-BFGS with T1 transitions | `tests/test_cap_vertex_step.py`, `tests/test_minimize_lbfgs_t1_reset.py` |
| [T1 statistics](t1-statistics.md) | counting T1 transitions in relaxations and estimators | `tests/test_t1_counts_in_loop_state.py` |
| [Shear rheometry](shear-rheometry.md) | the drive mode and its properties | `tests/test_rheology.py`, README |
| [Traced patch](traced-patch.md) | tracing a mask and relaxing a clamped patch | `tests/test_trace.py`, `tests/test_clamped.py`, README |

The notebooks themselves are in `examples/` in the repository. `examples/pbc_from_image.ipynb` builds a
periodic mesh from a mask and an image; its first cell says that it does not work on main, see
[Meshes from images and masks](../user-guide/images-and-masks.md).
