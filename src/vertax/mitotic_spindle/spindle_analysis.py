"""
Main pipeline for mitotic cell division simulation.

Orchestrates the modules:
  - mesh_utils    : boundary extraction and polygon geometry
  - spindle_model : force model and spindle equilibrium solver
  - cell_division : half-edge table topology surgery

Mechanical optimisation and T1 transitions are called directly via the
VertAX APIs (BoundedBilevelOptimizer, energy_bounded).
Visualisation is handled by tutorial.ipynb.

Entry point:
  python spindle_analysis.py [n_divisions]
"""

import math
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import numpy as np
import jax
import jax.numpy as jnp
import optax

from vertax import BoundedMesh, BoundedBilevelOptimizer
from vertax.topo import update_T1_bounded

from energy_adaptive import energy_adaptive, outer_loss_homogenise
from spindle_model import find_spindle
from cell_division import divide_cell



def _pick_face(mesh: BoundedMesh, d_fraction: float, n_gauss: int,
               spindle_method: str = 'newton',
               spindle_pdf_C: float = 0.1,
               spindle_rng: np.random.Generator | None = None) -> int:
    """Randomly select a cell and verify that division is feasible.

    Tries cells in random order and returns the first one for which
    find_spindle + divide_cell succeed without error.
    Cells smaller than half the mean area are skipped to avoid creating
    degenerate daughter cells.
    Any unexpected error is re-raised immediately.
    """
    face_ids = jnp.arange(mesh.nb_faces)
    areas = np.array(mesh.get_area(face_ids))
    mean_area = float(np.mean(areas))
    min_area = mean_area / 2.0

    rng = np.random.default_rng()
    for fid in rng.permutation(mesh.nb_faces).tolist():
        if areas[fid] < min_area:
            print(f"  Cell {fid} skipped: area {areas[fid]:.4f} < min {min_area:.4f}")
            continue
        try:
            c_test, u_test, _, _ = find_spindle(
                fid, mesh, d_fraction, n_gauss,
                method=spindle_method, pdf_C=spindle_pdf_C, rng=spindle_rng,
            )
            divide_cell(mesh, fid, c_test, u_test)
            return fid
        except ValueError as e:
            print(f"  Cell {fid} skipped: {e}")
            continue
        except Exception as e:
            print(f"  Cell {fid} unexpected error ({type(e).__name__}): {e}")
            raise
    raise RuntimeError("No divisible cell found in the current mesh.")


def run_n_divisions(mesh: BoundedMesh, n_divisions: int,
                    d_fraction: float = 0.3, n_gauss: int = 12,
                    max_opt_iter: int = 5000,
                    n_outer_epochs: int = 3,
                    spindle_method: str = 'newton',
                    spindle_pdf_C: float = 0.1,
                    spindle_rng: np.random.Generator | None = None,
                    ) -> tuple[BoundedMesh, list[BoundedMesh]]:
    """Perform n_divisions successive cell divisions.

    For each division:
      1. Randomly select a divisible cell.
      2. Compute the mitotic spindle (find_spindle).
      3. Topological division (divide_cell).
      4. Mechanical relaxation with T1 transitions (BoundedBilevelOptimizer).

    Returns (final_mesh, snapshots) where snapshots is a list of (mesh_post, mesh_opt)
    pairs — one per division — with mesh_post the topology right after division and
    mesh_opt the mesh after mechanical relaxation + outer optimisation.
    Visualisation is handled separately (see tutorial.ipynb).
    """
    # Initial total area = sum of all cell areas at start (kept as reference)
    total_area = float(jnp.sum(mesh.get_area(jnp.arange(mesh.nb_faces))))

    current = mesh
    snapshots: list[tuple[BoundedMesh, BoundedMesh]] = []
    for i in range(n_divisions):
        print(f"\n{'='*50}")
        print(f"Division {i + 1}/{n_divisions}")

        fid = _pick_face(current, d_fraction, n_gauss,
                         spindle_method=spindle_method,
                         spindle_pdf_C=spindle_pdf_C,
                         spindle_rng=spindle_rng)
        print(f"Selected cell: {fid}")

        c, u, _, _ = find_spindle(fid, current, d_fraction, n_gauss,
                                  method=spindle_method,
                                  pdf_C=spindle_pdf_C,
                                  rng=spindle_rng)
        mesh_post, _ = divide_cell(current, fid, c, u)

        mesh_opt = BoundedMesh.copy_mesh(mesh_post)

        face_ids = jnp.arange(mesh_opt.nb_faces)
        areas = mesh_opt.get_area(face_ids)
        print(f"  Areas — min: {float(jnp.min(areas)):.4f}, max: {float(jnp.max(areas)):.4f}, NaN: {int(jnp.sum(jnp.isnan(areas)))}")
        if jnp.any(jnp.isnan(areas)) or jnp.any(areas <= 0):
            print("Optimization skipped: invalid areas.")
        else:
            # Target = initial_total_area / nb_faces_current
            target_area = total_area / mesh_opt.nb_faces
            print(f"Target area: {target_area:.4f} (initial total {total_area:.4f} / {mesh_opt.nb_faces} cells)")
            mesh_opt.faces_params = jnp.full(mesh_opt.nb_faces, target_area)

            mesh_backup = BoundedMesh.copy_mesh(mesh_opt)

            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=5e-4,
                warmup_steps=200, decay_steps=max_opt_iter, end_value=1e-6)
            for solver, label in [(optax.adam(learning_rate=schedule), "Adam warmup-cosine"),
                                  (optax.adam(learning_rate=5e-5), "Adam 5e-5"),
                                  (optax.adam(learning_rate=1e-5), "Adam 1e-5")]:
                candidate = BoundedMesh.copy_mesh(mesh_backup)
                opt1 = BoundedBilevelOptimizer()
                opt1.loss_function_inner = energy_adaptive
                opt1.inner_solver        = solver
                opt1.update_T1           = True
                opt1._update_T1_func     = update_T1_bounded
                opt1.min_dist_T1         = 0.05
                opt1.max_nb_iterations   = max_opt_iter
                opt1.tolerance           = 1e-6
                opt1.patience            = 50
                loss_hist = list(opt1.inner_optimization(candidate))
                final_loss = float(loss_hist[-1])
                verts = np.array(candidate.vertices)
                if not math.isnan(final_loss) and not np.any(np.isnan(verts)):
                    mesh_opt = candidate
                    mid = float(loss_hist[len(loss_hist)//2])
                    print(f"Optimization ({label}): {len(loss_hist)} iter, loss {float(loss_hist[0]):.4f} → {mid:.4f} → {final_loss:.4f}")
                    break
                print(f"Divergence ({label}) — retrying.")
            else:
                print("Optimization failed — keeping post-division mesh.")

        # Outer optimisation: homogenise tensions (he_params) and area targets (face_params)
        # Only run if inner converged reasonably (loss < 10 × nb_faces as heuristic)
        inner_converged = not math.isnan(final_loss) and final_loss < 10.0 * mesh_opt.nb_faces
        if n_outer_epochs > 0 and inner_converged:
            mesh_pre_outer = BoundedMesh.copy_mesh(mesh_opt)
            opt_outer = BoundedBilevelOptimizer()
            opt_outer.loss_function_inner = energy_adaptive
            opt_outer.loss_function_outer = outer_loss_homogenise
            opt_outer.inner_solver        = optax.adam(learning_rate=optax.cosine_decay_schedule(1e-3, 200))
            opt_outer.outer_solver        = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=1e-4),
            )
            opt_outer.update_T1           = True
            opt_outer._update_T1_func     = update_T1_bounded
            opt_outer.min_dist_T1         = 0.05
            opt_outer.max_nb_iterations   = 200
            opt_outer.tolerance           = 1e-6
            opt_outer.patience            = 20
            for epoch in range(n_outer_epochs):
                opt_outer.bilevel_optimization(mesh_opt)
            if np.any(np.isnan(np.array(mesh_opt.vertices))):
                print("Outer optimisation diverged (NaN vertices) — reverting to pre-outer mesh.")
                mesh_opt = mesh_pre_outer
            else:
                print(f"Outer optimisation: {n_outer_epochs} epochs done")

        current = mesh_opt
        snapshots.append((BoundedMesh.copy_mesh(mesh_post), BoundedMesh.copy_mesh(current)))
    return current, snapshots


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Simulation of mitotic cell divisions on a 2-D tissue.")
    parser.add_argument('n_divisions', type=int, nargs='?', default=3,
                        help='Number of cell divisions to simulate (default: 3)')
    args = parser.parse_args()

    # Initial tissue: n_cells Voronoi cells in a square of side sqrt(n_cells)
    n_cells = 20
    width   = math.sqrt(n_cells)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width,
                                          height=width, random_key=2)

    # Edge tensions drawn from a logistic distribution (values between 1 and 2)
    key = jax.random.PRNGKey(3)
    mesh.edges_params = 1 + jax.nn.sigmoid(
        jax.random.uniform(key, (mesh.nb_edges,)) * 20 - 10
    )
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.faces_params    = jnp.full(mesh.nb_faces, 0.6)

    run_n_divisions(mesh, n_divisions=args.n_divisions)
