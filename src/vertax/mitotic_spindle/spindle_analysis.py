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

from energy_adaptive import energy_adaptive
from spindle_model import find_spindle
from cell_division import divide_cell
from mesh_utils import get_cell_segments


def _compute_polarity(mesh: BoundedMesh) -> tuple[np.ndarray, np.ndarray]:
    """Compute each cell's polarity axis from the mechanical tension field.

    The polarity axis φ_f is the principal axis of the tension-weighted
    edge-direction tensor:

        T_f = Σ_{e ∈ f}  tension_e · (ê_e ⊗ ê_e)

    where ê_e = (v1 - v0) / |v1 - v0| is the unit direction of edge e and
    tension_e = sigmoid(he_params_e) + 1.

    The principal axis is the eigenvector of T_f associated with its largest
    eigenvalue — the direction along which the cell is most strongly pulled.

    Returns
    -------
    polarity_phi      : (nb_faces,) principal axis angle in radians for each cell
    polarity_strength : constant array of zeros — caller sets the desired strength
    """
    polarity_phi = np.zeros(mesh.nb_faces)
    for fid in range(mesh.nb_faces):
        try:
            v0s, v1s, tensions = get_cell_segments(fid, mesh)
        except Exception:
            continue
        v0s_np = np.array(v0s)
        v1s_np = np.array(v1s)
        tensions_np = np.array(tensions)
        T = np.zeros((2, 2))
        for v0, v1, t in zip(v0s_np, v1s_np, tensions_np):
            e = v1 - v0
            norm = np.linalg.norm(e)
            if norm < 1e-12:
                continue
            e_hat = e / norm
            T += t * np.outer(e_hat, e_hat)
        # Principal axis = eigenvector for largest eigenvalue
        eigvals, eigvecs = np.linalg.eigh(T)
        polarity_phi[fid] = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
    return polarity_phi


def _edge_score_biased_axis(face_id: int, mesh: BoundedMesh, u_newton: np.ndarray,
                            score_rng: np.random.Generator,
                            draw_rng: np.random.Generator,
                            kappa: float = 4.0, beta: float = 2.0,
                            n_alpha: int = 361) -> np.ndarray:
    """Bias the base Newton spindle axis with per-edge polarity scores.

    Starting from the Newton orientation φ0 = angle(u_newton), build a density
    over axis orientations α ∈ [-π/2, π/2]:

        p(α) ∝ exp(κ·cos 2(α-φ0)) · exp(β·Σ_e w_e ℓ_e cos²(α-θ_e))

    where each EDGE carries a single random score w_e (one per edge, shared by all
    sub-segments of a curved arc), θ_e is its direction and ℓ_e its length. κ anchors
    on the mechanically-balanced Newton axis, β is the polarity-bias strength; their
    ratio sets the geometry-vs-polarity trade-off. φ is then drawn stochastically
    from p(α). Returns the biased unit axis u = (cos φ, sin φ).
    """
    E = np.array(mesh.edges); V = np.array(mesh.vertices); ANG = np.array(mesh.angles)
    from vertax.meshes.plot import _get_arc_with_n_points

    edge_score: dict[int, float] = {}
    seg_v0, seg_v1, seg_w = [], [], []
    start = int(mesh.faces[face_id][0]); he = start; first = True
    while first or he != start:
        first = False
        eid = he // 2
        if eid not in edge_score:
            edge_score[eid] = float(score_rng.uniform(0.0, 1.0))  # one score per edge
        wv = edge_score[eid]
        src = int(E[he, 3] + E[he, 5] - 2); tgt = int(E[he, 4] + E[he, 6] - 3)
        if src >= 0 and tgt >= 0:
            p0, p1 = V[src], V[tgt]
            ang = float(ANG[eid])
            if int(E[he, 3]) == 0 and ang > 0.01:  # surface arc -> sub-segments, same score
                pts = [p0] + _get_arc_with_n_points(ang, p0, p1) + [p1]
                for a, b in zip(pts[:-1], pts[1:]):
                    seg_v0.append(a); seg_v1.append(b); seg_w.append(wv)
            else:
                seg_v0.append(p0); seg_v1.append(p1); seg_w.append(wv)
        he = int(E[he, 1])

    seg_v0 = np.array(seg_v0); seg_v1 = np.array(seg_v1); w = np.array(seg_w)
    dirs = seg_v1 - seg_v0
    theta = np.arctan2(dirs[:, 1], dirs[:, 0])
    length = np.hypot(dirs[:, 0], dirs[:, 1])

    phi0 = float(np.arctan2(u_newton[1], u_newton[0]))
    phi0 = (phi0 + np.pi / 2) % np.pi - np.pi / 2

    alphas = np.linspace(-np.pi / 2, np.pi / 2, n_alpha)
    anchor = np.exp(kappa * np.cos(2 * (alphas - phi0)))
    bias = np.exp(beta * np.array([np.sum(w * length * np.cos(a - theta) ** 2) for a in alphas]))
    p = anchor * bias
    p = p / p.sum()
    phi = float(draw_rng.choice(alphas, p=p))
    return np.array([np.cos(phi), np.sin(phi)])


def _pick_face(mesh: BoundedMesh, d_fraction: float, n_gauss: int,
               spindle_method: str = 'newton',
               spindle_pdf_C: float = 0.1,
               spindle_rng: np.random.Generator | None = None,
               cell_rng: np.random.Generator | None = None,
               polarity_phi: np.ndarray | None = None,
               polarity_strength: float = 0.0) -> int:
    """Randomly select a cell and verify that division is feasible.

    Tries cells in random order and returns the first one for which
    find_spindle + divide_cell succeed without error.
    Cells smaller than half the mean area are skipped to avoid creating
    degenerate daughter cells.
    Any unexpected error is re-raised immediately.

    cell_rng controls which cell is picked (permutation seed).
    spindle_rng controls the axis draw for pdf_sample method.
    """
    face_ids = jnp.arange(mesh.nb_faces)
    areas = np.array(mesh.get_area(face_ids))
    mean_area = float(np.mean(areas))
    min_area = mean_area / 2.0

    rng = cell_rng if cell_rng is not None else np.random.default_rng()
    for fid in rng.permutation(mesh.nb_faces).tolist():
        if areas[fid] < min_area:
            print(f"  Cell {fid} skipped: area {areas[fid]:.4f} < min {min_area:.4f}")
            continue
        try:
            phi = float(polarity_phi[fid]) if polarity_phi is not None else 0.0
            c_test, u_test, _, _ = find_spindle(
                fid, mesh, d_fraction, n_gauss,
                method=spindle_method, pdf_C=spindle_pdf_C, rng=spindle_rng,
                polarity_phi=phi, polarity_strength=polarity_strength,
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
                    max_opt_iter: int = 2000,
                    spindle_method: str = 'newton',
                    spindle_pdf_C: float = 0.1,
                    spindle_rng: np.random.Generator | None = None,
                    cell_rng: np.random.Generator | None = None,
                    polarity_strength: float = 0.0,
                    edge_bias: bool = False,
                    edge_bias_kappa: float = 4.0,
                    edge_bias_beta: float = 2.0,
                    edge_bias_rng: np.random.Generator | None = None,
                    ) -> tuple[BoundedMesh, list[BoundedMesh]]:
    """Perform n_divisions successive cell divisions.

    For each division:
      1. Compute cell polarity axes from the current tension field.
      2. Randomly select a divisible cell.
      3. Compute the mitotic spindle (find_spindle), biased by polarity.
      4. Topological division (divide_cell).
      5. Mechanical relaxation with T1 transitions (BoundedBilevelOptimizer).

    polarity_strength : p ∈ [0, 1] — how strongly the tension-derived polarity
        axis biases the spindle orientation. 0 = geometry only (default),
        1 = polarity doubles microtubule density along the principal tension axis.

    Returns (final_mesh, snapshots) where snapshots is a list of (mesh_post, mesh_opt)
    pairs — one per division — with mesh_post the topology right after division and
    mesh_opt the mesh after mechanical relaxation + outer optimisation.
    Visualisation is handled separately (see tutorial.ipynb).
    """
    current = mesh
    snapshots: list[tuple[BoundedMesh, BoundedMesh]] = []
    for i in range(n_divisions):
        print(f"\n{'='*50}")
        print(f"Division {i + 1}/{n_divisions}")

        # Polarity axes from the current mechanical state (zero cost if p=0)
        polarity_phi = _compute_polarity(current) if polarity_strength > 0 else None

        fid = _pick_face(current, d_fraction, n_gauss,
                         spindle_method=spindle_method,
                         spindle_pdf_C=spindle_pdf_C,
                         spindle_rng=spindle_rng,
                         cell_rng=cell_rng,
                         polarity_phi=polarity_phi,
                         polarity_strength=polarity_strength)
        print(f"Selected cell: {fid}")

        phi = float(polarity_phi[fid]) if polarity_phi is not None else 0.0
        if polarity_strength > 0:
            print(f"  Polarity  : φ={np.degrees(phi):.1f}°  p={polarity_strength:.2f}")

        c, u, _, _ = find_spindle(fid, current, d_fraction, n_gauss,
                                  method=spindle_method,
                                  pdf_C=spindle_pdf_C,
                                  rng=spindle_rng,
                                  polarity_phi=phi,
                                  polarity_strength=polarity_strength)

        # Optionally bias the Newton axis with per-edge polarity scores.
        if edge_bias:
            _sr = edge_bias_rng if edge_bias_rng is not None else np.random.default_rng()
            u = _edge_score_biased_axis(fid, current, np.asarray(u), _sr, _sr,
                                        kappa=edge_bias_kappa, beta=edge_bias_beta)
            print(f"  Edge-bias : φ={np.degrees(np.arctan2(u[1], u[0])):.1f}°  "
                  f"(κ={edge_bias_kappa}, β={edge_bias_beta})")

        mesh_post, _ = divide_cell(current, fid, c, u)

        mesh_opt = BoundedMesh.copy_mesh(mesh_post)

        face_ids = jnp.arange(mesh_opt.nb_faces)
        areas = mesh_opt.get_area(face_ids)
        print(f"  Areas — min: {float(jnp.min(areas)):.4f}, max: {float(jnp.max(areas)):.4f}, NaN: {int(jnp.sum(jnp.isnan(areas)))}")
        if jnp.any(jnp.isnan(areas)) or jnp.any(areas <= 0):
            print("Optimization skipped: invalid areas.")
        else:
            # Fixed per-cell target area (no total-area target): every cell relaxes
            # towards 0.6, so the tissue's total area grows as cells divide.
            target_area = 0.6
            print(f"Per-cell target area: {target_area:.4f} ({mesh_opt.nb_faces} cells "
                  f"→ total target {target_area * mesh_opt.nb_faces:.4f})")
            mesh_opt.faces_params = jnp.full(mesh_opt.nb_faces, target_area)

            # Same relaxation as the notebook's single-division step: SGD, lr=1e-3.
            opt1 = BoundedBilevelOptimizer()
            opt1.loss_function_inner = energy_adaptive
            opt1.inner_solver        = optax.sgd(learning_rate=1e-3)
            opt1.update_T1           = True
            opt1._update_T1_func     = update_T1_bounded
            opt1.min_dist_T1         = 0.01
            opt1.max_nb_iterations   = max_opt_iter
            opt1.tolerance           = 1e-6
            opt1.patience            = 50
            loss_hist = list(opt1.inner_optimization(mesh_opt))
            print(f"Optimization (SGD 1e-3): {len(loss_hist)} iter, "
                  f"loss {float(loss_hist[0]):.4f} → {float(loss_hist[-1]):.4f}")

        # After division, recalculate polarity from the relaxed mesh for the
        # next iteration — no explicit inheritance needed since _compute_polarity
        # always reads from the current tension field.

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
