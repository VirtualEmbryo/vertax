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
from spindle_model import find_spindle, _make_residual
from cell_division import divide_cell, divide_cells
from mesh_utils import get_cell_segments, get_face_polygon, polygon_centroid, _inertia_axis


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
               polarity_strength: float = 0.0) -> tuple[int, np.ndarray, np.ndarray]:
    """Randomly select a cell and compute its mitotic spindle.

    Picks one cell at random (cell_rng) and returns (fid, c, u): the cell id and
    its spindle centre/axis from find_spindle. Any cell may be divided — there is
    no feasibility pre-check or area filter, and no throw-away divide_cell trial
    (geometric degeneracies, if any, are handled downstream by the caller).

    The spindle is computed here and returned so the caller does NOT recompute it.

    cell_rng controls which cell is picked.
    spindle_rng controls the axis draw for pdf_sample method.
    """
    rng = cell_rng if cell_rng is not None else np.random.default_rng()
    fid = int(rng.permutation(mesh.nb_faces)[0])
    phi = float(polarity_phi[fid]) if polarity_phi is not None else 0.0
    c, u, _, _ = find_spindle(
        fid, mesh, d_fraction, n_gauss,
        method=spindle_method, pdf_C=spindle_pdf_C, rng=spindle_rng,
        polarity_phi=phi, polarity_strength=polarity_strength,
    )
    return fid, c, u


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
                    inner_solver: optax.GradientTransformation | None = None,
                    tolerance: float = 1e-6,
                    patience: int = 50,
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

    inner_solver : optax solver used for the mechanical relaxation after each
        division. None (default) reproduces the notebook's SGD(lr=1e-3); pass any
        optax.GradientTransformation (e.g. optax.adam, nonlinear_cg, optax.lbfgs)
        to benchmark alternative solvers.

    tolerance, patience : early-stopping criterion for each inner relaxation. The
        relaxation stops once the relative energy change stays below `tolerance`
        for `patience` consecutive iterations (or `max_opt_iter` is reached).

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

        # _pick_face picks the cell AND computes its spindle (c, u); we reuse them
        # directly instead of recomputing find_spindle here.
        fid, c, u = _pick_face(current, d_fraction, n_gauss,
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

            # Same relaxation as the notebook's single-division step. Default is
            # SGD, lr=1e-3; an explicit inner_solver overrides it (for benchmarks).
            solver = inner_solver if inner_solver is not None else optax.sgd(learning_rate=1e-3)
            opt1 = BoundedBilevelOptimizer()
            opt1.loss_function_inner = energy_adaptive
            opt1.inner_solver        = solver
            opt1.update_T1           = True
            opt1._update_T1_func     = update_T1_bounded
            opt1.min_dist_T1         = 0.01
            opt1.max_nb_iterations   = max_opt_iter
            opt1.tolerance           = tolerance
            opt1.patience            = patience
            loss_hist = list(opt1.inner_optimization(mesh_opt))
            print(f"Optimization: {len(loss_hist)} iter, "
                  f"loss {float(loss_hist[0]):.4f} → {float(loss_hist[-1]):.4f}")

        # After division, recalculate polarity from the relaxed mesh for the
        # next iteration — no explicit inheritance needed since _compute_polarity
        # always reads from the current tension field.

        current = mesh_opt
        snapshots.append((BoundedMesh.copy_mesh(mesh_post), BoundedMesh.copy_mesh(current)))
    return current, snapshots


def run_n_divisions_parallel(mesh: BoundedMesh, n_waves: int, n_parallel: int,
                             d_fraction: float = 0.3, n_gauss: int = 12,
                             max_opt_iter: int = 2000,
                             spindle_method: str = 'newton',
                             spindle_pdf_C: float = 0.1,
                             spindle_rng: np.random.Generator | None = None,
                             cell_rng: np.random.Generator | None = None,
                             inner_solver: optax.GradientTransformation | None = None,
                             tolerance: float = 1e-6,
                             patience: int = 50,
                             ) -> tuple[BoundedMesh, list[BoundedMesh]]:
    """Simultaneous (parallel) cell divisions: each wave divides `n_parallel`
    distinct cells in a SINGLE cell_division call, then relaxes the tissue ONCE.

    This trades biological realism (cells are not relaxed between the cuts of a
    wave — a synchronous mitosis model) for speed: `n_waves` relaxations instead
    of `n_waves * n_parallel`, which is the dominant cost. Contrast with
    run_n_divisions, which relaxes after every single division.

    For each wave:
      1. Pick `n_parallel` distinct cells at random (without replacement).
      2. Compute each cell's spindle axis on the current mesh (find_spindle).
      3. Divide them all at once (divide_cells -> one cell_division call).
      4. Relax the whole tissue once (BoundedBilevelOptimizer.inner_optimization).

    Returns (final_mesh, snapshots) with one (mesh_post, mesh_opt) pair per wave.
    """
    current = mesh
    snapshots: list[tuple[BoundedMesh, BoundedMesh]] = []
    rng = cell_rng if cell_rng is not None else np.random.default_rng()

    for w in range(n_waves):
        print(f"\n{'='*50}")
        print(f"Wave {w + 1}/{n_waves} — dividing {n_parallel} cells simultaneously")

        k = min(n_parallel, current.nb_faces)
        fids = rng.permutation(current.nb_faces)[:k].tolist()

        us = []
        for fid in fids:
            _, u, _, _ = find_spindle(fid, current, d_fraction, n_gauss,
                                      method=spindle_method, pdf_C=spindle_pdf_C,
                                      rng=spindle_rng)
            us.append(np.asarray(u))
        print(f"Selected cells: {fids}")

        mesh_post, _ = divide_cells(current, fids, us)

        mesh_opt = BoundedMesh.copy_mesh(mesh_post)
        areas = mesh_opt.get_area(jnp.arange(mesh_opt.nb_faces))
        print(f"  Areas — min: {float(jnp.min(areas)):.4f}, max: {float(jnp.max(areas)):.4f}, "
              f"NaN: {int(jnp.sum(jnp.isnan(areas)))}")
        if jnp.any(jnp.isnan(areas)) or jnp.any(areas <= 0):
            print("Optimization skipped: invalid areas.")
        else:
            target_area = 0.6
            mesh_opt.faces_params = jnp.full(mesh_opt.nb_faces, target_area)
            solver = inner_solver if inner_solver is not None else optax.sgd(learning_rate=1e-3)
            opt1 = BoundedBilevelOptimizer()
            opt1.loss_function_inner = energy_adaptive
            opt1.inner_solver        = solver
            opt1.update_T1           = True
            opt1._update_T1_func     = update_T1_bounded
            opt1.min_dist_T1         = 0.01
            opt1.max_nb_iterations   = max_opt_iter
            opt1.tolerance           = tolerance
            opt1.patience            = patience
            loss_hist = list(opt1.inner_optimization(mesh_opt))
            print(f"Optimization: {len(loss_hist)} iter, "
                  f"loss {float(loss_hist[0]):.4f} → {float(loss_hist[-1]):.4f}")

        current = mesh_opt
        snapshots.append((BoundedMesh.copy_mesh(mesh_post), BoundedMesh.copy_mesh(current)))
    return current, snapshots


def run_n_divisions_animated(mesh: BoundedMesh, n_divisions: int,
                             relax_frames: int = 10,
                             d_fraction: float = 0.3, n_gauss: int = 12,
                             max_opt_iter: int = 2000,
                             spindle_method: str = 'newton',
                             spindle_pdf_C: float = 0.1,
                             spindle_rng: np.random.Generator | None = None,
                             cell_rng: np.random.Generator | None = None,
                             inner_solver: optax.GradientTransformation | None = None,
                             tolerance: float = 1e-6,
                             patience: int = 50,
                             ) -> tuple[BoundedMesh, list[dict]]:
    """Like run_n_divisions, but captures intermediate frames for an animation.

    For each division it records one 'division' frame (the mesh right after the
    cut) followed by `relax_frames` 'relax' frames sampled along the relaxation.
    The relaxation is run once with inner_opt_bounded_snapshots, which records the
    mesh at EVERY iteration; we then sub-sample `relax_frames` of them on a
    near-log schedule (dense at the start, where the motion is, sparse later).
    This captures the TRUE continuous trajectory of any solver — including
    fast-converging line-search solvers (CG, L-BFGS) — because the optimiser
    state is never reset.

    Returns (final_mesh, frames) where frames is a flat list of dicts:
        {'mesh': BoundedMesh, 'kind': 'division'|'relax',
         'division': i, 'sub': k, 'energy': float}
    ready to feed a FuncAnimation (see tutorial.ipynb).
    """
    from vertax.opt_bounded import inner_opt_bounded_snapshots

    current = mesh
    frames: list[dict] = []
    rng = cell_rng if cell_rng is not None else np.random.default_rng()
    solver = inner_solver if inner_solver is not None else optax.sgd(learning_rate=1e-3)

    def _energy(m: BoundedMesh) -> float:
        return float(energy_adaptive(
            jnp.asarray(m.vertices), jnp.asarray(m.angles), jnp.asarray(m.edges),
            jnp.asarray(m.faces), None, None, None,
            jnp.asarray(m.vertices_params), jnp.asarray(m.edges_params),
            jnp.asarray(m.faces_params)))

    def _log_indices(n_snap: int, n_pick: int) -> list[int]:
        # Near-log sample of [0, n_snap-1]: dense early, sparse late; always
        # includes the last (converged) snapshot. Deduplicated, sorted.
        if n_snap <= n_pick:
            return list(range(n_snap))
        last = n_snap - 1
        idx = {int(round((last) ** (t / (n_pick - 1)))) for t in range(n_pick)}
        idx.add(last)
        return sorted(idx)

    for i in range(n_divisions):
        print(f"\n{'='*50}\nDivision {i + 1}/{n_divisions}")

        fid = int(rng.permutation(current.nb_faces)[0])
        _, u, _, _ = find_spindle(fid, current, d_fraction, n_gauss,
                                  method=spindle_method, pdf_C=spindle_pdf_C, rng=spindle_rng)
        # divide_cell ignores c (recomputes the centroid internally), so pass a dummy.
        mesh_post, _ = divide_cell(current, fid, np.zeros(2), u)
        mesh_post.faces_params = jnp.full(mesh_post.nb_faces, 0.6)

        frames.append({'mesh': BoundedMesh.copy_mesh(mesh_post), 'kind': 'division',
                       'division': i, 'sub': 0, 'energy': _energy(mesh_post)})

        # One continuous relaxation, snapshotted at every iteration.
        snaps, _losses = inner_opt_bounded_snapshots(
            mesh_post.vertices, mesh_post.angles, mesh_post.edges, mesh_post.faces,
            mesh_post.vertices_params, mesh_post.edges_params, mesh_post.faces_params,
            L_in=energy_adaptive, solver=solver, min_dist_T1=0.01,
            iterations_max=max_opt_iter, tolerance=tolerance, patience=patience,
            update_T1_func=update_T1_bounded,
        )

        # Build a relaxed mesh from a snapshot (params are constant during relax).
        def _mesh_from_snap(snap):
            vt, at, ht, ft = snap
            m = BoundedMesh.copy_mesh(mesh_post)
            m.vertices, m.angles, m.edges, m.faces = vt, at, ht, ft
            return m

        for k, s_idx in enumerate(_log_indices(len(snaps), relax_frames), start=1):
            m_k = _mesh_from_snap(snaps[s_idx])
            frames.append({'mesh': m_k, 'kind': 'relax',
                           'division': i, 'sub': k, 'energy': _energy(m_k)})

        current = _mesh_from_snap(snaps[-1])
    return current, frames


def run_n_divisions_parallel_animated(mesh: BoundedMesh, n_waves: int, n_parallel: int,
                                      relax_frames: int = 10,
                                      spindle_frames: int = 20,
                                      start_offset: float = 0.3,
                                      method: str = 'newton',
                                      polarity_strength: float = 1.0,
                                      edge_bias_kappa: float = 4.0,
                                      edge_bias_beta: float = 2.0,
                                      edge_bias_rng: np.random.Generator | None = None,
                                      d_fraction: float = 0.3, n_gauss: int = 12,
                                      force_beta: float = 3.0,
                                      max_opt_iter: int = 2000,
                                      spindle_pdf_C: float = 0.1,
                                      spindle_rng: np.random.Generator | None = None,
                                      cell_rng: np.random.Generator | None = None,
                                      inner_solver: optax.GradientTransformation | None = None,
                                      tolerance: float = 1e-6,
                                      patience: int = 50,
                                      ) -> tuple[BoundedMesh, list[dict]]:
    """Animated parallel divisions with a pre-division spindle phase.

    Each wave:
      1. For every cell to divide, compute the EXACT spindle (centre, axis) of the
         chosen `method` (newton / thery / polarity / edge_score), then show it
         easing deterministically (smoothstep) from a perturbed start (centre
         offset by `start_offset`·radius, random axis) to that exact spindle —
         `spindle_frames` 'spindle' frames with the centrosomes, centre and
         cleavage line settling into place. No stochastic dynamics.
      2. Cut every cell at its exact (centre, axis) via divide_cells(centers=…),
         so e.g. Newton cuts ⟂ to the cell's long axis (Hertwig) — one 'division'
         frame.
      3. Relax the tissue once, captured as `relax_frames` 'relax' frames.

    Returns (final_mesh, frames). Frame dicts:
      {'mesh', 'kind': 'spindle'|'division'|'relax', 'division': wave, 'sub': k,
       'energy': float, and for 'spindle' frames also 'spindles': list of
       (centre(2,), axis_angle, d) for the cells being set up}.
    """
    from vertax.opt_bounded import inner_opt_bounded_snapshots

    current = mesh
    frames: list[dict] = []
    rng = cell_rng if cell_rng is not None else np.random.default_rng()
    lrng = spindle_rng if spindle_rng is not None else np.random.default_rng()
    solver = inner_solver if inner_solver is not None else optax.sgd(learning_rate=1e-3)

    def _energy(m: BoundedMesh) -> float:
        return float(energy_adaptive(
            jnp.asarray(m.vertices), jnp.asarray(m.angles), jnp.asarray(m.edges),
            jnp.asarray(m.faces), None, None, None,
            jnp.asarray(m.vertices_params), jnp.asarray(m.edges_params),
            jnp.asarray(m.faces_params)))

    def _log_indices(n_snap: int, n_pick: int) -> list[int]:
        if n_snap <= n_pick:
            return list(range(n_snap))
        last = n_snap - 1
        idx = {int(round(last ** (t / (n_pick - 1)))) for t in range(n_pick)}
        idx.add(last)
        return sorted(idx)

    for w in range(n_waves):
        print(f"\n{'='*50}\nWave {w + 1}/{n_waves} — {n_parallel} simultaneous divisions")

        # Per-face polarity axes (only needed for the polarity method).
        polarity_phi = _compute_polarity(current) if method == 'polarity' else None
        ebrng = edge_bias_rng if edge_bias_rng is not None else np.random.default_rng()

        k_cells = min(n_parallel, current.nb_faces)
        fids = rng.permutation(current.nb_faces)[:k_cells].tolist()

        # --- Per cell: find equilibrium with find_spindle, animate linearly -------
        # find_spindle is robust (Newton from centroid+inertia axis, with fallback).
        # The animation interpolates linearly from (centroid, inertia axis) to the
        # converged solution — purely visual, residuals decrease monotonically.
        # trajectories[fid] = (cs (T+1,2), alphas (T+1,), ds (T+1,)).
        trajectories = {}
        for fid in fids:
            poly = get_face_polygon(fid, current)
            c0 = polygon_centroid(poly)
            radius = float(np.mean(np.linalg.norm(poly - c0, axis=1)))
            d_max = d_fraction * radius
            u0 = _inertia_axis(poly, c0)
            a0 = float(np.arctan2(u0[1], u0[0]))
            p_phi = float(polarity_phi[fid]) if method == 'polarity' else 0.0
            p_str = polarity_strength if method == 'polarity' else 0.0
            c_star, u_star, _, _ = find_spindle(
                fid, current, d_fraction, n_gauss,
                polarity_phi=p_phi, polarity_strength=p_str,
                beta=force_beta,
            )
            a_star = float(np.arctan2(u_star[1], u_star[0]))
            # Build the residual once per cell (re-used for every animation frame,
            # avoiding a JAX recompilation on each frame).
            v0s, v1s, tensions = get_cell_segments(fid, current)
            res_fn = _make_residual(v0s, v1s, tensions, d_max, n_gauss, p_phi, p_str, force_beta)
            # Shortest rotation on the axis (period π).
            da = (a_star - a0 + np.pi / 2) % np.pi - np.pi / 2
            # Linear interpolation from (c0, a0) to (c_star, a_star).
            T = spindle_frames
            ts = np.linspace(0.0, 1.0, T + 1)
            cs  = (1 - ts)[:, None] * c0[None, :] + ts[:, None] * c_star[None, :]
            als = a0 + ts * da
            # Per-frame d: clip so both centrosomes stay inside the cell polygon.
            ds = np.empty(T + 1)
            n_poly = len(poly)
            for t in range(T + 1):
                c_t = cs[t]
                u_t = np.array([np.cos(als[t]), np.sin(als[t])])
                max_d = d_max
                for i in range(n_poly):
                    p0, p1 = poly[i], poly[(i + 1) % n_poly]
                    e = p1 - p0
                    for sign in (1.0, -1.0):
                        denom = sign * u_t[0] * e[1] - sign * u_t[1] * e[0]
                        if abs(denom) < 1e-12:
                            continue
                        diff = p0 - c_t
                        s = (diff[0] * e[1] - diff[1] * e[0]) / denom
                        seg_t = (diff[0] * (-sign * u_t[1]) - diff[1] * (-sign * u_t[0])) / denom
                        if s > 1e-9 and -1e-9 <= seg_t <= 1.0 + 1e-9:
                            max_d = min(max_d, s * 0.9)
                ds[t] = max(max_d, 0.0)
            trajectories[fid] = (cs, als, ds, res_fn)

        # Spindle frames: the tissue is unchanged (still `current`); only the
        # per-cell spindle overlays move. Each frame holds all cells' state at step t.
        for t in range(spindle_frames + 1):
            spindles = []
            residuals = {}  # fid -> (Fx, Fy, M)
            for fid in fids:
                cs, als, ds, res_fn = trajectories[fid]
                spindles.append((cs[t], float(als[t]), float(ds[t])))
                params = jnp.array([cs[t][0], cs[t][1], float(als[t])], dtype=jnp.float64)
                residuals[fid] = np.array(res_fn(params), dtype=float)
            frames.append({'mesh': BoundedMesh.copy_mesh(current), 'kind': 'spindle',
                           'division': w, 'sub': t, 'energy': _energy(current),
                           'spindles': spindles, 'residuals': residuals})

        # --- Cut every cell at its settled (centre, axis) -----------------------
        us, centers = [], []
        for fid in fids:
            cs, als, ds, _res_fn = trajectories[fid]
            a = float(als[-1])
            us.append(np.array([np.cos(a), np.sin(a)]))
            centers.append(np.asarray(cs[-1]))

        mesh_post, _ = divide_cells(current, fids, us, centers=centers)
        mesh_post.faces_params = jnp.full(mesh_post.nb_faces, 0.6)

        frames.append({'mesh': BoundedMesh.copy_mesh(mesh_post), 'kind': 'division',
                       'division': w, 'sub': 0, 'energy': _energy(mesh_post)})

        snaps, _losses = inner_opt_bounded_snapshots(
            mesh_post.vertices, mesh_post.angles, mesh_post.edges, mesh_post.faces,
            mesh_post.vertices_params, mesh_post.edges_params, mesh_post.faces_params,
            L_in=energy_adaptive, solver=solver, min_dist_T1=0.01,
            iterations_max=max_opt_iter, tolerance=tolerance, patience=patience,
            update_T1_func=update_T1_bounded,
        )

        def _mesh_from_snap(snap):
            vt, at, ht, ft = snap
            m = BoundedMesh.copy_mesh(mesh_post)
            m.vertices, m.angles, m.edges, m.faces = vt, at, ht, ft
            return m

        for k, s_idx in enumerate(_log_indices(len(snaps), relax_frames), start=1):
            m_k = _mesh_from_snap(snaps[s_idx])
            frames.append({'mesh': m_k, 'kind': 'relax',
                           'division': w, 'sub': k, 'energy': _energy(m_k)})

        current = _mesh_from_snap(snaps[-1])
    return current, frames


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
