#!/usr/bin/env python3
"""
Reproduce "Fig. 2 f-g Benchmarking bilevel optimization strategies for mechanical parameter inference
in VertAX".

What is measured
----------------
For each method ∈ {AD, EP, AS, ID}, each n_cells ∈ {20,30,40,50,60},
and each of 5 random initial conditions:

  1. Build a PBC Voronoi mesh and relax it (inner optimization).
  2. Build a target mesh with different shape factors and relax it.
  3. For epochs 0..50:
       - time ONLY the outer parameter update (not the subsequent inner solve);
       - record process Unique Set Size (USS) memory after the outer step;
       - then re-equilibrate with an inner optimization.

Scaling metrics exclude epoch 0 (JIT / first-call overhead):
  - Peak memory  = max(USS[1:])
  - Mean runtime = mean(runtime_per_epoch[1:]) averaged across runs

Usage
-----
1) Make VertAX importable (from this folder).

2) Paper settings (long: 5 runs × 4 methods × 5 sizes × 51 epochs):

   python reproduce_memory_runtime_scaling.py

   Outputs:
     - data:  outputs_scaling/{AD,EP,AS,ID}_memory_runtime_increasing_cells/
     - plots: outputs_scaling/paper_plots/
              (scaling_memory.pdf, scaling_runtime.pdf)

3) Useful flags:

   --methods AD EP AS ID     methods to run (default: all four)
   --n-runs 5                runs per (method, n_cells) (paper: 5)
   --epochs 50               outer epochs (paper: 50)
   --n-cells 20 30 40 50 60  system sizes (paper: those five)
   --plots-only              rebuild plots from existing pickles
   --in-process              skip subprocess isolation (debug only; USS leaks)

4) Quick smoke test:

   python reproduce_memory_runtime_scaling.py --n-runs 1 --epochs 2 --n-cells 20 30 --methods AD

5) Replot without re-running simulations:

   python reproduce_memory_runtime_scaling.py --plots-only
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force CPU for reproducibility.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import optax
import psutil

from vertax import BilevelOptimizationMethod, PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.energy import energy_shape_factor_hetero

# ---------------------------------------------------------------------------
# Paths / method metadata
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_ROOT = SCRIPT_DIR / "outputs_scaling"
PLOTS_DIR = OUTPUT_ROOT / "paper_plots"
DETAILED_DATA_FILE = "scaling_detailed_raw_data.pkl"


def configure_output_root(path: str | Path) -> None:
    """Set OUTPUT_ROOT / PLOTS_DIR (relative paths are under SCRIPT_DIR)."""
    global OUTPUT_ROOT, PLOTS_DIR
    root = Path(path)
    if not root.is_absolute():
        root = SCRIPT_DIR / root
    OUTPUT_ROOT = root.resolve()
    PLOTS_DIR = OUTPUT_ROOT / "paper_plots"

METHOD_SPECS = {
    "AD": {
        "label": "Automatic Differentiation",
        "enum": BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
        "color": "tab:blue",
        "fmt": "-o",
    },
    "EP": {
        "label": "Equilibrium Propagation",
        "enum": BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION,
        "color": "tab:green",
        "fmt": "--v",
    },
    "AS": {
        "label": "Adjoint State Method",
        "enum": BilevelOptimizationMethod.ADJOINT_STATE,
        "color": "tab:red",
        "fmt": "-.s",
    },
    "ID": {
        "label": "Implicit Differentiation",
        "enum": BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION,
        "color": "tab:purple",
        "fmt": ":^",
    },
}


# ---------------------------------------------------------------------------
# Utilities (memory / energy / optimizer)
# ---------------------------------------------------------------------------
def get_uss_mb() -> float:
    """Unique Set Size of the current process, in megabytes."""
    try:
        return psutil.Process(os.getpid()).memory_full_info().uss / (1024 * 1024)
    except Exception:
        # Fallback if USS is unavailable on this platform.
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def block_until_ready_on_pytree(pytree):
    """Force completion of all JAX arrays in a pytree (accurate wall-clock)."""

    def _block(x):
        return x.block_until_ready() if hasattr(x, "block_until_ready") else x

    return jax.tree_util.tree_map(_block, pytree)


def make_shape_factor_energy(width: float, height: float):
    """E1 energy."""

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


def make_optimizer(method_code: str) -> PbcBilevelOptimizer:
    """Paper hyperparameters for the scaling experiment."""
    opt = PbcBilevelOptimizer()
    opt.update_T1 = True
    opt.min_dist_T1 = 0.005
    opt.max_nb_iterations = 1000
    opt.tolerance = 1e-3
    opt.patience = 10
    opt.inner_solver = optax.sgd(learning_rate=0.01)
    opt.outer_solver = optax.adam(learning_rate=0.001, nesterov=True)
    opt.bilevel_optimization_method = METHOD_SPECS[method_code]["enum"]
    # EP uses this beta in the paper.
    opt.beta = 5.179475e-01
    return opt


def method_dir(method_code: str) -> Path:
    return OUTPUT_ROOT / f"{method_code}_memory_runtime_increasing_cells"


def run_pickle_path(method_code: str, n_cells: int, run_id: int) -> Path:
    return method_dir(method_code) / f"n_cells_{n_cells}" / f"run_{run_id}_raw.pkl"


# ---------------------------------------------------------------------------
# Core experiment: one isolated (method, n_cells, run)
# ---------------------------------------------------------------------------
def run_single_experiment(
    method_code: str,
    n_cells: int,
    run_id: int,
    epochs: int,
    mean_shape_factors: float,
    std_shape_factors: float,
    seed_offset: int = 0,
) -> dict:
    """Run one bilevel scaling trial and return its raw record."""
    width = height = math.sqrt(n_cells)
    energy = make_shape_factor_energy(width, height)

    # Seeds: (run_id + seed_offset)*333, +1, +2  (offset shifts away from prior campaigns)
    seed_index = run_id + seed_offset
    print(
        f"\n--- [{method_code}] run {run_id + 1} (n_cells={n_cells}) "
        f"seed_index={seed_index} [pid={os.getpid()}] ---"
    )

    key_geom = seed_index * 333
    key_params = jax.random.PRNGKey(seed_index * 333 + 1)
    key_target = jax.random.PRNGKey(seed_index * 333 + 2)

    opt = make_optimizer(method_code)
    opt.loss_function_inner = energy
    opt.loss_function_outer = cost_v2v

    # Initial mesh + parameters
    mesh = PbcMesh.from_random_seeds(
        nb_seeds=n_cells, width=width, height=height, random_key=key_geom
    )
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    mesh.faces_params = mean_shape_factors + std_shape_factors * jax.random.normal(
        key_params, shape=(mesh.nb_faces,)
    )
    opt.inner_optimization(mesh)

    # Target mesh
    target = PbcMesh.copy_mesh(mesh)
    target.vertices_params = jnp.zeros(target.nb_vertices)
    target.edges_params = jnp.zeros(target.nb_half_edges)
    target.faces_params = mean_shape_factors + std_shape_factors * jax.random.normal(
        key_target, shape=(target.nb_faces,)
    )
    opt.inner_optimization(target)
    opt.vertices_target = target.vertices.copy()
    opt.edges_target = target.edges.copy()
    opt.faces_target = target.faces.copy()
    del target

    runtime_per_epoch: list[float] = []
    uss_history: list[float] = []
    cumulative = 0.0

    for epoch in range(epochs + 1):
        # --- Time ONLY the outer step ---
        t0 = time.time()
        opt.outer_optimization(mesh)
        block_until_ready_on_pytree(
            (mesh.vertices_params, mesh.edges_params, mesh.faces_params)
        )
        duration = time.time() - t0
        runtime_per_epoch.append(duration)
        cumulative += duration

        uss = get_uss_mb()
        uss_history.append(uss)

        # Re-equilibrate (not included in the timed outer duration).
        opt.inner_optimization(mesh, perform_energy_check=False)

        print(
            f"  epoch {epoch}: runtime={duration:.4f}s  "
            f"cum={cumulative:.2f}s  USS={uss:.2f} MB"
        )

    uss_arr = np.asarray(uss_history, dtype=float)
    runt_arr = np.asarray(runtime_per_epoch, dtype=float)
    peak_scaling = float(np.max(uss_arr[1:])) if len(uss_arr) > 1 else float(np.max(uss_arr))

    return {
        "n_cells": n_cells,
        "run_id": run_id,
        "seed_offset": seed_offset,
        "seed_index": seed_index,
        "peak_mem_mb_all_epochs": float(np.max(uss_arr)),
        "peak_mem_mb_scaling": peak_scaling,
        "uss_history_mb": uss_arr,
        "runtime_per_epoch_s": runt_arr,
    }


def save_run_record(method_code: str, record: dict) -> Path:
    """Persist one run pickle under n_cells_*/run_*_raw.pkl."""
    out = run_pickle_path(method_code, int(record["n_cells"]), int(record["run_id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(record, f)
    print(f"  Saved run record → {out}")
    return out


def load_run_records(method_code: str, n_cells_list: list[int], n_runs: int) -> list[dict]:
    """Load per-run pickles in stable (n_cells, run_id) order."""
    detailed: list[dict] = []
    for n_cells in n_cells_list:
        for run_id in range(n_runs):
            path = run_pickle_path(method_code, n_cells, run_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing isolated-run pickle: {path}")
            with open(path, "rb") as f:
                detailed.append(pickle.load(f))
    return detailed


def aggregate_method_outputs(
    method_code: str,
    n_cells_list: list[int],
    n_runs: int,
    epochs: int,
) -> list[dict]:
    """Build per-method summary plots + DETAILED_DATA_FILE from run pickles."""
    mdir = method_dir(method_code)
    mdir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(SCRIPT_PATH, mdir / SCRIPT_PATH.name)
    except OSError as exc:
        print(f"  [warn] Could not copy script: {exc}")

    detailed = load_run_records(method_code, n_cells_list, n_runs)

    global_n_cells: list[int] = []
    global_mean_peak_mem: list[float] = []
    global_std_peak_mem: list[float] = []
    global_mean_runtime: list[float] = []
    global_std_runtime: list[float] = []

    for n_cells in n_cells_list:
        cells_dir = mdir / f"n_cells_{n_cells}"
        cells_dir.mkdir(parents=True, exist_ok=True)
        runs = [r for r in detailed if int(r["n_cells"]) == n_cells]
        runs = sorted(runs, key=lambda r: int(r["run_id"]))
        if len(runs) != n_runs:
            raise RuntimeError(
                f"[{method_code}] n_cells={n_cells}: expected {n_runs} runs, found {len(runs)}"
            )

        all_runtime = [np.asarray(r["runtime_per_epoch_s"], dtype=float) for r in runs]
        all_uss = [np.asarray(r["uss_history_mb"], dtype=float) for r in runs]
        peak_mems = [float(r["peak_mem_mb_scaling"]) for r in runs]

        runtime_epoch_array = np.stack(all_runtime, axis=0)
        uss_array = np.stack(all_uss, axis=0)
        mean_runt = runtime_epoch_array.mean(axis=0)
        std_runt = runtime_epoch_array.std(axis=0)
        mean_uss = uss_array.mean(axis=0)
        std_uss = uss_array.std(axis=0)
        mean_cum = np.cumsum(mean_runt)
        epoch_numbers = np.arange(epochs + 1)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        alpha = 0.25
        axes[0].plot(mean_cum, mean_uss, color="red", marker="s", markersize=2)
        axes[0].fill_between(mean_cum, mean_uss - std_uss, mean_uss + std_uss, color="red", alpha=alpha)
        axes[0].set_ylabel("Mean Memory (MB)")
        axes[0].set_xlabel("Mean Cumulative Runtime (s)")
        axes[0].set_title(
            f"Memory & Runtime Metrics for $n_{{cells}}={n_cells}$ ({n_runs} Runs) [{method_code}]"
        )
        axes[0].grid(True, linestyle="--", alpha=0.6)

        axes[1].plot(epoch_numbers, mean_runt, color="green", marker="^", markersize=2)
        axes[1].fill_between(
            epoch_numbers, mean_runt - std_runt, mean_runt + std_runt, color="green", alpha=alpha
        )
        axes[1].set_xlabel("Epoch Number")
        axes[1].set_ylabel("Mean Runtime per Epoch (s)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(cells_dir / "memo_runt_plots_averaged.pdf")
        plt.close()

        global_n_cells.append(n_cells)
        global_mean_peak_mem.append(float(np.mean(peak_mems)))
        global_std_peak_mem.append(float(np.std(peak_mems)))
        runtime_for_scaling = runtime_epoch_array[:, 1:]
        mean_runtime_across_runs = runtime_for_scaling.mean(axis=1)
        global_mean_runtime.append(float(np.mean(mean_runtime_across_runs)))
        global_std_runtime.append(float(np.std(mean_runtime_across_runs)))

    g_n = np.asarray(global_n_cells)
    g_mem_m = np.asarray(global_mean_peak_mem)
    g_mem_s = np.asarray(global_std_peak_mem)
    g_runt_m = np.asarray(global_mean_runtime)
    g_runt_s = np.asarray(global_std_runtime)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].errorbar(
        g_n,
        g_mem_m,
        yerr=g_mem_s,
        fmt="-o",
        color="darkorange",
        capsize=5,
        ecolor="red",
        label=f"Mean Peak Memory ± STD (Excl. Epoch 0, {n_runs} runs)",
    )
    axes[0].set_title("Scaling Analysis with System Size (Excl. Epoch 0 JIT)")
    axes[0].set_xlabel(r"Number of Cells ($n_{cells}$)")
    axes[0].set_ylabel("Mean Peak Memory (MB)")
    axes[0].grid(True, which="both", ls="--", alpha=0.6)
    axes[0].legend(loc="upper left")

    axes[1].errorbar(
        g_n,
        g_runt_m,
        yerr=g_runt_s,
        fmt="-s",
        color="darkgreen",
        capsize=5,
        ecolor="green",
        label=f"Mean Runtime per Epoch ± STD (Excl. Epoch 0, {n_runs} runs)",
    )
    axes[1].set_xlabel(r"Number of Cells ($n_{cells}$)")
    axes[1].set_ylabel("Mean Runtime per Epoch (s)")
    axes[1].grid(True, which="both", ls="--", alpha=0.6)
    axes[1].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(mdir / "final_scaling_memory_and_runtime_analysis_excl_epoch_0.pdf")
    plt.close()

    summary = np.column_stack((g_n, g_mem_m, g_mem_s, g_runt_m, g_runt_s))
    np.savetxt(
        mdir / "scaling_summary_stats_excl_epoch_0.txt",
        summary,
        delimiter=",",
        header=(
            "n_cells,Mean_Peak_Mem_MB_Excl_Epoch_0,Std_Peak_Mem_MB_Excl_Epoch_0,"
            "Mean_Runtime_Per_Epoch_S_Excl_Epoch_0,Std_Runtime_Per_Epoch_S_Excl_Epoch_0"
        ),
        comments="",
    )

    with open(mdir / DETAILED_DATA_FILE, "wb") as f:
        pickle.dump(detailed, f)
    print(f"[{method_code}] Saved detailed data → {mdir / DETAILED_DATA_FILE}")

    return detailed


# ---------------------------------------------------------------------------
# Subprocess orchestration
# ---------------------------------------------------------------------------
def launch_isolated_run(
    method_code: str,
    n_cells: int,
    run_id: int,
    epochs: int,
    mean_shape_factors: float,
    std_shape_factors: float,
    seed_offset: int = 0,
) -> None:
    """Spawn a fresh Python process for one (method, n_cells, run_id)."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--methods",
        method_code,
        "--n-cells",
        str(n_cells),
        "--run-id",
        str(run_id),
        "--n-runs",
        "1",
        "--epochs",
        str(epochs),
        "--mean-shape-factors",
        str(mean_shape_factors),
        "--std-shape-factors",
        str(std_shape_factors),
        "--seed-offset",
        str(seed_offset),
        "--output-root",
        str(OUTPUT_ROOT),
    ]
    print(f"\n>>> subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_method_scaling_isolated(
    method_code: str,
    n_cells_list: list[int],
    n_runs: int,
    epochs: int,
    mean_shape_factors: float,
    std_shape_factors: float,
    in_process: bool = False,
    seed_offset: int = 0,
) -> list[dict]:
    """Run all (n_cells, run) trials for one method, then aggregate."""
    mdir = method_dir(method_code)
    mdir.mkdir(parents=True, exist_ok=True)

    for n_cells in n_cells_list:
        print(f"\n===== [{method_code}] n_cells = {n_cells} =====")
        for run_id in range(n_runs):
            if in_process:
                record = run_single_experiment(
                    method_code=method_code,
                    n_cells=n_cells,
                    run_id=run_id,
                    epochs=epochs,
                    mean_shape_factors=mean_shape_factors,
                    std_shape_factors=std_shape_factors,
                    seed_offset=seed_offset,
                )
                save_run_record(method_code, record)
            else:
                launch_isolated_run(
                    method_code=method_code,
                    n_cells=n_cells,
                    run_id=run_id,
                    epochs=epochs,
                    mean_shape_factors=mean_shape_factors,
                    std_shape_factors=std_shape_factors,
                    seed_offset=seed_offset,
                )

    return aggregate_method_outputs(method_code, n_cells_list, n_runs, epochs)


# ---------------------------------------------------------------------------
# Paper scaling plots
# ---------------------------------------------------------------------------
def load_and_group(method_code: str):
    """Load pickle and group records by n_cells."""
    path = method_dir(method_code) / DETAILED_DATA_FILE
    if not path.exists():
        print(f"  [warn] Missing {path}")
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    grouped = defaultdict(list)
    for run in data:
        grouped[run["n_cells"]].append(run)
    return grouped, sorted(grouped.keys())


def compute_scaling_metrics(grouped, n_cells_list):
    """Mean/std peak memory and mean runtime vs n_cells (excl. epoch 0)."""
    metrics = {
        "n_cells": [],
        "mem_mean": [],
        "mem_std": [],
        "runt_mean": [],
        "runt_std": [],
        "num_runs": [],
    }
    for n_cells in n_cells_list:
        runs = grouped[n_cells]
        uss = np.stack([r["uss_history_mb"] for r in runs], axis=0)
        runt = np.stack([r["runtime_per_epoch_s"] for r in runs], axis=0)
        peaks = np.asarray([r["peak_mem_mb_scaling"] for r in runs], dtype=float)
        runt_scaling = runt[:, 1:]
        mean_runt_across_runs = runt_scaling.mean(axis=1)

        metrics["n_cells"].append(n_cells)
        metrics["mem_mean"].append(float(peaks.mean()))
        metrics["mem_std"].append(float(peaks.std()))
        metrics["runt_mean"].append(float(mean_runt_across_runs.mean()))
        metrics["runt_std"].append(float(mean_runt_across_runs.std()))
        metrics["num_runs"].append(len(runs))

    for key in ("n_cells", "mem_mean", "mem_std", "runt_mean", "runt_std"):
        metrics[key] = np.asarray(metrics[key])
    return metrics


def plot_paper_scaling(methods: list[str]) -> None:
    """Write scaling_memory.pdf and scaling_runtime.pdf (paper panels)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for method_code in methods:
        grouped, n_cells_list = load_and_group(method_code)
        if grouped is None:
            continue
        all_metrics[method_code] = compute_scaling_metrics(grouped, n_cells_list)

    if not all_metrics:
        print("  [error] No scaling data found; cannot plot.")
        return

    min_runs = min(int(np.min(m["num_runs"])) for m in all_metrics.values())

    # --- Peak memory ---
    fig_mem, ax_mem = plt.subplots(figsize=(8, 6))
    fig_mem.suptitle(f"Scaling Comparison: Peak Memory ({min_runs} runs)", fontsize=16)
    for method_code, s in all_metrics.items():
        spec = METHOD_SPECS[method_code]
        ax_mem.errorbar(
            s["n_cells"],
            s["mem_mean"],
            yerr=s["mem_std"],
            fmt=spec["fmt"],
            color=spec["color"],
            capsize=5,
            label=spec["label"],
        )
    ax_mem.set_yscale("log")
    ax_mem.set_xlabel(r"Number of Cells ($n_{cells}$)")
    ax_mem.set_ylabel("Mean Peak Memory (MB)")
    ax_mem.grid(True, which="both", ls="--", alpha=0.6)
    ax_mem.legend(loc="upper left")
    ax_mem.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    mem_path = PLOTS_DIR / "scaling_memory.pdf"
    plt.savefig(mem_path)
    plt.savefig(mem_path.with_suffix(".eps"))
    plt.close()
    print(f"  Wrote {mem_path}")

    # --- Runtime ---
    fig_runt, ax_runt = plt.subplots(figsize=(8, 6))
    fig_runt.suptitle(f"Scaling Comparison: Runtime ({min_runs} runs)", fontsize=16)
    for method_code, s in all_metrics.items():
        spec = METHOD_SPECS[method_code]
        ax_runt.errorbar(
            s["n_cells"],
            s["runt_mean"],
            yerr=s["runt_std"],
            fmt=spec["fmt"],
            color=spec["color"],
            capsize=5,
            label=spec["label"],
        )
    ax_runt.set_yscale("log")
    ax_runt.set_xlabel(r"Number of Cells ($n_{cells}$)")
    ax_runt.set_ylabel("Mean Runtime per Epoch (s)")
    ax_runt.grid(True, which="both", ls="--", alpha=0.6)
    ax_runt.legend(loc="upper left")
    ax_runt.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    runt_path = PLOTS_DIR / "scaling_runtime.pdf"
    plt.savefig(runt_path)
    plt.savefig(runt_path.with_suffix(".eps"))
    plt.close()
    print(f"  Wrote {runt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Fig. 2 f-g Benchmarking bilevel optimization strategies "
            "for mechanical parameter inference in VertAX."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_SPECS.keys()),
        choices=list(METHOD_SPECS.keys()),
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per (method, n_cells) (paper: 5).")
    parser.add_argument("--epochs", type=int, default=50, help="Outer epochs (paper: 50).")
    parser.add_argument(
        "--n-cells",
        nargs="+",
        type=int,
        default=[20, 30, 40, 50, 60],
        help="System sizes (paper: 20 30 40 50 60).",
    )
    parser.add_argument("--mean-shape-factors", type=float, default=4.1)
    parser.add_argument("--std-shape-factors", type=float, default=0.1)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip simulations; rebuild scaling_memory.pdf / scaling_runtime.pdf from pickles.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal: run a single isolated (method, n_cells, run-id) and exit.",
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
        help="Run all trials in one process (debug only; USS will accumulate).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output directory (default: outputs_scaling next to this script).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Added to run_id before building RNG seeds (seed_index = run_id + offset). "
            "Use a positive offset to draw different ICs than a prior campaign."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is not None:
        configure_output_root(args.output_root)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("VertAX devices:", jax.devices())
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Seed offset: {args.seed_offset}")

    # --- Worker: one isolated run in this process, then exit ---
    if args.worker:
        if args.run_id is None:
            raise SystemExit("--worker requires --run-id")
        if len(args.methods) != 1:
            raise SystemExit("--worker requires exactly one --methods code")
        if len(args.n_cells) != 1:
            raise SystemExit("--worker requires exactly one --n-cells value")

        method_code = args.methods[0]
        n_cells = int(args.n_cells[0])
        record = run_single_experiment(
            method_code=method_code,
            n_cells=n_cells,
            run_id=int(args.run_id),
            epochs=args.epochs,
            mean_shape_factors=args.mean_shape_factors,
            std_shape_factors=args.std_shape_factors,
            seed_offset=args.seed_offset,
        )
        save_run_record(method_code, record)
        return

    if not args.plots_only:
        if args.in_process:
            print("[warn] --in-process: USS will accumulate across runs/methods.")
        for method_code in args.methods:
            print(f"\n########## METHOD {method_code} ##########")
            run_method_scaling_isolated(
                method_code=method_code,
                n_cells_list=list(args.n_cells),
                n_runs=args.n_runs,
                epochs=args.epochs,
                mean_shape_factors=args.mean_shape_factors,
                std_shape_factors=args.std_shape_factors,
                in_process=args.in_process,
                seed_offset=args.seed_offset,
            )

    print("\n=== Generating paper scaling plots ===")
    plot_paper_scaling(args.methods)
    print("\nDone. Paper plots are in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
