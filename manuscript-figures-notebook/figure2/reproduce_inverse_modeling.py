#!/usr/bin/env python3
"""
Reproduce "Fig. 2 a-e Benchmarking bilevel optimization strategies for mechanical
parameter inference in VertAX".

What is measured
----------------
For each method ∈ {AD, EP, AS, ID} and each of 5 random initial conditions
(n_cells = 20):

  1. Build a PBC Voronoi mesh with random shape factors and relax it
     (inner optimization).
  2. Build a target mesh with different shape factors and relax it; register
     its geometry as the outer-cost target.
  3. For epochs 0..350:
       - one bilevel step (outer parameter update + inner re-equilibration);
       - record vertex-to-vertex cost C(X, X*);
       - record Pearson correlation corr(p, p*) of predicted vs target
         shape factors;
       - record relative L1 error ‖p − p*‖₁ / ‖p*‖₁.

Paper plots aggregate runs {1, 2, 3, 4, 5}:
  - mean ± std histories of cost / Pearson / relative error vs epoch
  - last-epoch boxplots of Pearson and relative error across methods
  - scatter of predicted vs target shape factors at epoch 0 (AD) and
    epoch 350 (all methods, run 2)

Usage
-----
1) Make VertAX importable (from this folder).

2) Paper settings (long: 5 runs × 4 methods × 351 epochs):

   python reproduce_inverse_modeling_10runs.py

   Outputs:
     - data:  outputs_10runs/{AD,EP,AS,ID}_10runs/
     - plots: outputs_10runs/paper_plots/

3) Useful flags:

   --methods AD EP AS ID     methods to run (default: all four)
   --n-runs 5                number of random seeds (paper: 5)
   --epochs 350              outer epochs (paper: 350)
   --n-cells 20              number of cells (paper: 20)
   --comparison-runs 1 2 3 4 5
                             runs used in comparison plots (1-based)
   --save-tables             also dump full mesh tables every epoch (disk-heavy)
   --plots-only              rebuild plots from existing outputs_10runs/
   --in-process              skip subprocess isolation (debug only; memory leaks)

4) Quick smoke test:

   python reproduce_inverse_modeling_10runs.py --n-runs 1 --epochs 2 --methods AD

5) Replot without re-running simulations:

   python reproduce_inverse_modeling_10runs.py --plots-only

Notes
-----
  - Comparison curves use runs {1, 2, 3, 4, 5} (1-based).
  - Scatter plots use run 2 at epochs 0 and 350.
  - Full 5 × 4 × 351 bilevel epochs is computationally heavy.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force CPU for reproducibility.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import energy_shape_factor_hetero

# ---------------------------------------------------------------------------
# Paths / output layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_ROOT = SCRIPT_DIR / "outputs_10runs"
PLOTS_DIR = OUTPUT_ROOT / "paper_plots"

# Method short codes used in folder names / legends 
METHOD_SPECS = {
    "AD": {
        "label": "AD Model",
        "enum": BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
        "color": "blue",
    },
    "EP": {
        "label": "EP Model",
        "enum": BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION,
        "color": "green",
    },
    "AS": {
        "label": "AS Model",
        "enum": BilevelOptimizationMethod.ADJOINT_STATE,
        "color": "red",
    },
    "ID": {
        "label": "ID Model",
        "enum": BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION,
        "color": "purple",
    },
}

# Runs used for multi-method comparison plots in the paper (1-based indices).
COMPARISON_RUN_INDICES = [1, 2, 3, 4, 5]
# Run used for the shape-factor scatter panels (1-based).
SCATTER_RUN_INDEX = 1


# ---------------------------------------------------------------------------
# Physics / optimization helpers (new VertAX API)
# ---------------------------------------------------------------------------
def make_shape_factor_energy(width: float, height: float):
    """Return an E1 energy:
        sum_α  (A_α − 1)² + (P_α − p_α)²
    with heterogeneous target shape factors p_α stored in face_params.
    """

    def energy(vertTable, heTable, faceTable, _vert_params, _he_params, face_params):
        selected_faces = jnp.arange(len(faceTable))
        return energy_shape_factor_hetero(
            vertTable,
            heTable,
            faceTable,
            width,
            height,
            selected_faces,
            face_params,
        )

    return energy


def make_optimizer(
    method_code: str,
    *,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    max_nb_iterations: int = 1000,
    tolerance: float = 1e-3,
    patience: int = 10,
    min_dist_T1: float = 0.005,
    beta_ep: float = 5.179475e-01,
) -> PbcBilevelOptimizer:
    """Build a `PbcBilevelOptimizer` with the paper hyperparameters.
    """
    if method_code not in METHOD_SPECS:
        raise ValueError(f"Unknown method '{method_code}'. Choose from {list(METHOD_SPECS)}.")

    opt = PbcBilevelOptimizer()
    opt.update_T1 = True
    opt.min_dist_T1 = min_dist_T1
    opt.max_nb_iterations = max_nb_iterations
    opt.tolerance = tolerance
    opt.patience = patience
    opt.inner_solver = optax.sgd(learning_rate=inner_lr)
    opt.outer_solver = optax.adam(learning_rate=outer_lr, nesterov=True)
    opt.bilevel_optimization_method = METHOD_SPECS[method_code]["enum"]
    # Only used by Equilibrium Propagation.
    opt.beta = beta_ep
    return opt


def compute_cost(mesh: PbcMesh, opt: PbcBilevelOptimizer) -> float:
    """Vertex-to-vertex cost against the registered target geometry."""
    return float(
        cost_v2v(
            mesh.vertices,
            mesh.edges,
            mesh.faces,
            mesh.width,
            mesh.height,
            opt.vertices_target,
            opt.edges_target,
            opt.faces_target,
        )
    )


def pearson_corr(predicted: jnp.ndarray, target: jnp.ndarray) -> float:
    """Pearson correlation between predicted and ground-truth face parameters."""
    return float(jnp.corrcoef(predicted, target)[0, 1])


def relative_l1_error(predicted: jnp.ndarray, target: jnp.ndarray) -> float:
    """Relative L1 error ‖p − p*‖₁ / ‖p*‖₁"""
    return float(jnp.linalg.norm(predicted - target, ord=1) / jnp.linalg.norm(target, ord=1))


# ---------------------------------------------------------------------------
# Single run (one method, one random seed)
# ---------------------------------------------------------------------------
def run_one_experiment(
    method_code: str,
    run_idx_zero_based: int,
    epochs: int,
    n_cells: int,
    mean_shape_factors: float,
    std_shape_factors: float,
    save_every_epoch_tables: bool,
) -> dict:
    """Run one inverse-modeling trajectory and save per-epoch metrics/tables.
    """
    run_1based = run_idx_zero_based + 1
    method_dir = OUTPUT_ROOT / f"{method_code}_10runs"
    run_dir = method_dir / f"inverse_{method_code}_run_{run_1based}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of this script next to the run for auditability.
    try:
        shutil.copy(Path(__file__).resolve(), run_dir / Path(__file__).name)
    except OSError as exc:
        print(f"  [warn] Could not copy script into run folder: {exc}")

    width = height = math.sqrt(n_cells)
    energy = make_shape_factor_energy(width, height)
    opt = make_optimizer(method_code)
    opt.loss_function_inner = energy
    opt.loss_function_outer = cost_v2v

    # --- Initial mesh (Voronoi from random seeds) ---
    mesh = PbcMesh.from_random_seeds(
        nb_seeds=n_cells,
        width=width,
        height=height,
        random_key=run_idx_zero_based + 123,
    )
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    key_params = jax.random.PRNGKey(run_idx_zero_based + 123 + 1)
    mesh.faces_params = mean_shape_factors + std_shape_factors * jax.random.normal(
        key_params, shape=(mesh.nb_faces,)
    )

    # Relax initial condition to mechanical equilibrium.
    opt.inner_optimization(mesh)

    # --- Target mesh: same topology, different shape factors ---
    target = PbcMesh.copy_mesh(mesh)
    key_target = jax.random.PRNGKey(run_idx_zero_based + 123 + 2)
    target.vertices_params = jnp.zeros(target.nb_vertices)
    target.edges_params = jnp.zeros(target.nb_half_edges)
    target.faces_params = mean_shape_factors + std_shape_factors * jax.random.normal(
        key_target, shape=(target.nb_faces,)
    )
    # Ground-truth parameters we try to recover (fixed during learning).
    face_params_target = target.faces_params.copy()

    opt.inner_optimization(target)

    # Register target geometry on the optimizer (required by cost_v2v / outer methods).
    opt.vertices_target = target.vertices.copy()
    opt.edges_target = target.edges.copy()
    opt.faces_target = target.faces.copy()
    del target

    cost_history: list[float] = []
    pearson_history: list[float] = []
    rel_error_history: list[float] = []

    for epoch in range(epochs + 1):
        print(f"  [{method_code}] run {run_1based}  epoch {epoch}/{epochs}")

        # One bilevel step = outer parameter update + inner re-equilibration.
        opt.bilevel_optimization(mesh, perform_energy_check=False)

        cost_history.append(compute_cost(mesh, opt))
        pearson_history.append(pearson_corr(mesh.faces_params, face_params_target))
        rel_error_history.append(relative_l1_error(mesh.faces_params, face_params_target))

        # Face parameters are required for scatter plots; tables optional (disk-heavy).
        np.savetxt(run_dir / f"face_params_epoch{epoch}.txt", np.array(mesh.faces_params))
        np.savetxt(run_dir / f"face_params_target_epoch{epoch}.txt", np.array(face_params_target))

        if save_every_epoch_tables:
            np.savetxt(run_dir / f"vertTable_epoch{epoch}.txt", np.array(mesh.vertices))
            np.savetxt(run_dir / f"heTable_epoch{epoch}.txt", np.array(mesh.edges))
            np.savetxt(run_dir / f"faceTable_epoch{epoch}.txt", np.array(mesh.faces))
            np.savetxt(run_dir / f"vertTable_target_epoch{epoch}.txt", np.array(opt.vertices_target))
            np.savetxt(run_dir / f"heTable_target_epoch{epoch}.txt", np.array(opt.edges_target))
            np.savetxt(run_dir / f"faceTable_target_epoch{epoch}.txt", np.array(opt.faces_target))

        if epoch % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(range(epoch + 1), cost_history, color="blue")
            ax1.set_ylabel("Cost")
            ax2.plot(range(epoch + 1), pearson_history, color="green")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Pearson Corr")
            plt.tight_layout()
            plt.savefig(run_dir / "cost_and_pearson_corr.pdf")
            plt.close()

    np.savetxt(run_dir / "cost_history.txt", cost_history)
    np.savetxt(run_dir / "pearson_history.txt", pearson_history)
    np.savetxt(run_dir / "rel_error_history.txt", rel_error_history)

    return {
        "cost": np.asarray(cost_history),
        "pearson": np.asarray(pearson_history),
        "rel_error": np.asarray(rel_error_history),
    }


def load_run_result(method_code: str, run_1based: int) -> dict:
    """Load per-run histories written by `run_one_experiment`."""
    run_dir = OUTPUT_ROOT / f"{method_code}_10runs" / f"inverse_{method_code}_run_{run_1based}"
    cost_path = run_dir / "cost_history.txt"
    pearson_path = run_dir / "pearson_history.txt"
    rel_error_path = run_dir / "rel_error_history.txt"
    for path in (cost_path, pearson_path, rel_error_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing run output: {path}")
    return {
        "cost": np.loadtxt(cost_path),
        "pearson": np.loadtxt(pearson_path),
        "rel_error": np.loadtxt(rel_error_path),
    }


def save_method_aggregates(method_code: str, all_runs: list[dict]) -> None:
    """Save mean±std curves and last-epoch boxplot data for one method."""
    method_dir = OUTPUT_ROOT / f"{method_code}_10runs"
    method_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("cost", "pearson", "rel_error"):
        stacked = np.stack([r[metric] for r in all_runs], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        np.savetxt(method_dir / f"{metric}_total.txt", mean)
        np.savetxt(method_dir / f"{metric}_total_std.txt", std)
        np.savetxt(method_dir / f"{metric}_last_epoch.txt", stacked[:, -1])

        # Per-method overview plots.
        color = {"cost": "blue", "pearson": "green", "rel_error": "red"}[metric]
        ylabel = {
            "cost": "Cost",
            "pearson": "Pearson Corr",
            "rel_error": "Relative Error",
        }[metric]
        plt.figure(figsize=(8, 4))
        x = np.arange(len(mean))
        plt.plot(x, mean, color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Total over Epochs ({method_code})")
        plt.tight_layout()
        stem = {
            "cost": "cost_total",
            "pearson": "pearson_corr_total",
            "rel_error": "rel_error_total",
        }[metric]
        plt.savefig(method_dir / f"{stem}.pdf")
        plt.savefig(method_dir / f"{stem}.eps")
        plt.close()


# ---------------------------------------------------------------------------
# Subprocess orchestration
# ---------------------------------------------------------------------------
def launch_isolated_run(
    method_code: str,
    run_idx_zero_based: int,
    epochs: int,
    n_cells: int,
    mean_shape_factors: float,
    std_shape_factors: float,
    save_tables: bool,
) -> None:
    """Spawn a fresh Python process for one (method, run)."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--methods",
        method_code,
        "--run-id",
        str(run_idx_zero_based),
        "--n-runs",
        "1",
        "--epochs",
        str(epochs),
        "--n-cells",
        str(n_cells),
        "--mean-shape-factors",
        str(mean_shape_factors),
        "--std-shape-factors",
        str(std_shape_factors),
    ]
    if save_tables:
        cmd.append("--save-tables")
    print(f"\n>>> subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_method_isolated(
    method_code: str,
    n_runs: int,
    epochs: int,
    n_cells: int,
    mean_shape_factors: float,
    std_shape_factors: float,
    save_tables: bool,
    in_process: bool = False,
) -> list[dict]:
    """Run all seeds for one method (subprocess per run by default), then aggregate."""
    print(f"\n===== Method {method_code} ({METHOD_SPECS[method_code]['enum']}) =====")
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        if in_process:
            run_one_experiment(
                method_code=method_code,
                run_idx_zero_based=run,
                epochs=epochs,
                n_cells=n_cells,
                mean_shape_factors=mean_shape_factors,
                std_shape_factors=std_shape_factors,
                save_every_epoch_tables=save_tables,
            )
        else:
            launch_isolated_run(
                method_code=method_code,
                run_idx_zero_based=run,
                epochs=epochs,
                n_cells=n_cells,
                mean_shape_factors=mean_shape_factors,
                std_shape_factors=std_shape_factors,
                save_tables=save_tables,
            )

    all_runs = [load_run_result(method_code, run + 1) for run in range(n_runs)]
    save_method_aggregates(method_code, all_runs)
    return all_runs


# ---------------------------------------------------------------------------
# Paper comparison / scatter plots
# ---------------------------------------------------------------------------
def _load_selected_histories(method_code: str, metric: str, run_indices_1based: list[int]) -> np.ndarray | None:
    """Load `{metric}_history.txt` for selected runs of one method."""
    histories = []
    for run_num in run_indices_1based:
        path = (
            OUTPUT_ROOT
            / f"{method_code}_10runs"
            / f"inverse_{method_code}_run_{run_num}"
            / f"{metric}_history.txt"
        )
        if not path.exists():
            print(f"  [warn] Missing {path}")
            continue
        histories.append(np.loadtxt(path))
    if not histories:
        return None
    return np.stack(histories, axis=0)


def plot_history_comparison(
    methods: list[str],
    metric: str,
    ylabel: str,
    title: str,
    save_stem: str,
    run_indices_1based: list[int],
) -> None:
    """Mean ± std overlapping history plot across methods (paper panel)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    max_epochs = 0
    for method_code in methods:
        data = _load_selected_histories(method_code, metric, run_indices_1based)
        if data is None:
            continue
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        color = METHOD_SPECS[method_code]["color"]
        label = f"{METHOD_SPECS[method_code]['label']} (N={data.shape[0]})"
        epochs = np.arange(len(mean))
        max_epochs = max(max_epochs, len(mean))
        plt.plot(epochs, mean, color=color, label=label)
        plt.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{title} (Runs: {run_indices_1based})")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    if max_epochs > 0:
        plt.xlim(0, max_epochs - 1)
    plt.tight_layout()
    run_suffix = "_".join(map(str, run_indices_1based))
    base = PLOTS_DIR / f"{save_stem}_Comparison_Runs{run_suffix}"
    plt.savefig(f"{base}.pdf")
    plt.savefig(f"{base}.eps")
    plt.close()
    print(f"  Wrote {base}.pdf")


def plot_boxplot_comparison(
    methods: list[str],
    metric: str,
    ylabel: str,
    title: str,
    save_stem: str,
    run_indices_1based: list[int],
) -> None:
    """Side-by-side last-epoch boxplots across methods (paper panel)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    box_data = []
    labels = []
    colors = []
    for method_code in methods:
        data = _load_selected_histories(method_code, metric, run_indices_1based)
        if data is None:
            continue
        box_data.append(data[:, -1])
        labels.append(f"{METHOD_SPECS[method_code]['label']}\n(N={data.shape[0]})")
        colors.append(METHOD_SPECS[method_code]["color"])

    if not box_data:
        print(f"  [warn] No data for boxplot {metric}")
        return

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        medianprops={"color": "black"},
        flierprops={"marker": "o", "markerfacecolor": "grey", "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.title(f"{title} at Last Epoch (Runs: {run_indices_1based})")
    plt.ylabel(f"{ylabel} Value")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    run_suffix = "_".join(map(str, run_indices_1based))
    base = PLOTS_DIR / f"{save_stem}_Boxplot_Runs{run_suffix}"
    plt.savefig(f"{base}.pdf")
    plt.savefig(f"{base}.eps")
    plt.close()
    print(f"  Wrote {base}.pdf")


def plot_scatter_comparison(epoch: int, methods: list[str], run_1based: int, save_name: str) -> None:
    """Predicted vs target shape-factor scatter (paper panel).

    Epoch 0: AD only (initial mismatch).
    Epoch 350: all methods overlaid.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    active = ["AD"] if epoch == 0 else methods
    all_pts: list[float] = []

    for method_code in active:
        run_dir = OUTPUT_ROOT / f"{method_code}_10runs" / f"inverse_{method_code}_run_{run_1based}"
        init_path = run_dir / f"face_params_epoch{epoch}.txt"
        tgt_path = run_dir / f"face_params_target_epoch{epoch}.txt"
        if not (init_path.exists() and tgt_path.exists()):
            print(f"  [warn] Missing scatter files for {method_code} epoch {epoch}")
            continue
        predicted = np.loadtxt(init_path)
        target = np.loadtxt(tgt_path)
        ax.scatter(
            target,
            predicted,
            color=METHOD_SPECS[method_code]["color"],
            alpha=0.6,
            edgecolors="none",
            label=method_code,
        )
        all_pts.extend(predicted.tolist())
        all_pts.extend(target.tolist())

    if not all_pts:
        plt.close()
        return

    arr = np.asarray(all_pts)
    margin = (arr.max() - arr.min()) * 0.1
    lims = [arr.min() - margin, arr.max() + margin]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Target shape factor $p_{\alpha}^{GT}$", fontsize=14)
    ax.set_ylabel(r"Initial/Predicted shape factor", fontsize=14)
    ax.set_title(f"Comparison at Epoch {epoch}", fontsize=16)
    if epoch != 0:
        ax.legend(loc="upper left")
    plt.tight_layout()
    out = PLOTS_DIR / save_name
    plt.savefig(out, dpi=300)
    # Also write .eps for journal submission convenience.
    plt.savefig(out.with_suffix(".eps"), dpi=300)
    plt.close()
    print(f"  Wrote {out}")


def make_all_paper_plots(methods: list[str], epochs: int, run_indices_1based: list[int]) -> None:
    """Generate all comparison / scatter PDFs expected by the paper."""
    print("\n=== Generating paper comparison plots ===")
    plot_history_comparison(
        methods,
        metric="cost",
        ylabel=r"Cost (Mean $\pm$ Std Dev)",
        title="Cost History Comparison Across Models",
        save_stem="cost_total",
        run_indices_1based=run_indices_1based,
    )
    plot_history_comparison(
        methods,
        metric="pearson",
        ylabel=r"Pearson Correlation (Mean $\pm$ Std Dev)",
        title="Pearson Correlation History Comparison Across Models",
        save_stem="pearson_corr_total",
        run_indices_1based=run_indices_1based,
    )
    plot_history_comparison(
        methods,
        metric="rel_error",
        ylabel=r"Relative Error (Mean $\pm$ Std Dev)",
        title="Relative Error History Comparison Across Models",
        save_stem="rel_error_total",
        run_indices_1based=run_indices_1based,
    )
    plot_boxplot_comparison(
        methods,
        metric="pearson",
        ylabel="Pearson Correlation",
        title="Pearson Correlation Comparison",
        save_stem="pearson_corr_last_epoch",
        run_indices_1based=run_indices_1based,
    )
    plot_boxplot_comparison(
        methods,
        metric="rel_error",
        ylabel="Relative Error",
        title="Relative Error Comparison",
        save_stem="rel_error_last_epoch",
        run_indices_1based=run_indices_1based,
    )

    # Scatter: epoch 0 (AD) and final epoch (overlap). Filename keeps 350 when
    # using the paper setting so it matches the original asset name.
    # Prefer paper run index 2; fall back to the first available comparison run.
    scatter_run = SCATTER_RUN_INDEX if SCATTER_RUN_INDEX in run_indices_1based else run_indices_1based[0]
    plot_scatter_comparison(0, methods, scatter_run, "comparison_epoch_0.pdf")
    final_name = (
        "comparison_epoch_350_overlap.pdf"
        if epochs == 350
        else f"comparison_epoch_{epochs}_overlap.pdf"
    )
    plot_scatter_comparison(epochs, methods, scatter_run, final_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Fig. 2 a-e Benchmarking bilevel optimization strategies for mechanical parameter inference in VertAX."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_SPECS.keys()),
        choices=list(METHOD_SPECS.keys()),
        help="Bilevel methods to run (default: AD EP AS ID).",
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Number of random seeds / runs (paper: 5).")
    parser.add_argument("--epochs", type=int, default=350, help="Outer epochs (paper: 350).")
    parser.add_argument("--n-cells", type=int, default=20, help="Number of cells (paper: 20).")
    parser.add_argument("--mean-shape-factors", type=float, default=4.1)
    parser.add_argument("--std-shape-factors", type=float, default=0.1)
    parser.add_argument(
        "--save-tables",
        action="store_true",
        help="Also dump full mesh tables every epoch (very disk-heavy; off by default).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip simulations; only rebuild paper plots from existing outputs_10runs/.",
    )
    parser.add_argument(
        "--comparison-runs",
        nargs="+",
        type=int,
        default=COMPARISON_RUN_INDICES,
        help="1-based run indices used for multi-method comparison plots.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal: run a single isolated (method, run-id) and exit.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="0-based run index (required with --worker).",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run all trials in one process (debug only; XLA memory will accumulate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("VertAX devices:", jax.devices())
    print(f"Output root: {OUTPUT_ROOT}")

    # --- Worker: one isolated run in this process, then exit ---
    if args.worker:
        if args.run_id is None:
            raise SystemExit("--worker requires --run-id")
        if len(args.methods) != 1:
            raise SystemExit("--worker requires exactly one --methods code")
        run_one_experiment(
            method_code=args.methods[0],
            run_idx_zero_based=int(args.run_id),
            epochs=args.epochs,
            n_cells=args.n_cells,
            mean_shape_factors=args.mean_shape_factors,
            std_shape_factors=args.std_shape_factors,
            save_every_epoch_tables=args.save_tables,
        )
        return

    if not args.plots_only:
        if args.in_process:
            print("[warn] --in-process: XLA/JAX memory will accumulate across runs/methods.")
        for method_code in args.methods:
            run_method_isolated(
                method_code=method_code,
                n_runs=args.n_runs,
                epochs=args.epochs,
                n_cells=args.n_cells,
                mean_shape_factors=args.mean_shape_factors,
                std_shape_factors=args.std_shape_factors,
                save_tables=args.save_tables,
                in_process=args.in_process,
            )

    # Filter comparison-run indices to those that actually exist / were requested.
    available_runs = list(range(1, args.n_runs + 1)) if not args.plots_only else list(range(1, 6))
    comparison_runs = [r for r in args.comparison_runs if r in available_runs or args.plots_only]
    if not comparison_runs:
        comparison_runs = available_runs

    make_all_paper_plots(args.methods, args.epochs, comparison_runs)
    print("\nDone. Paper plots are in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
