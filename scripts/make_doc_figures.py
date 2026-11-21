"""Draw the figures of the documentation website from the library.

Every figure drawn from a computation is made by one function of this file, with the settings of the
page that shows it and fixed seeds. The schematics are hand-written SVG files in `figures/`, listed in
`figures/README.md` with the page that shows each figure and its source.

The figures share one set of conventions, defined at the top of this file: the same colour for the
same object in every figure (edges, cells, vertices, half-edges, the contact that changes in a T1
transition, the reference geometry), one colour map for a quantity of the cells and another for the
line tensions, and the hue and marker of each gradient estimator. Text is drawn as paths with the
DejaVu Sans font shipped with Matplotlib, so the files look the same on every machine. Each figure has
a white background, so that it reads on the light and on the dark scheme of the site.

Usage, from the root of the clone, on the CPU:

    uv run python scripts/make_doc_figures.py              # every figure, a few minutes
    uv run python scripts/make_doc_figures.py costs rheo   # the figures whose name contains a word
    uv run python scripts/make_doc_figures.py --list       # the names

The figures are written to `figures/`; files written by the library on the way (the folder of
`do_n_bilevel_optimization`) go to `build/doc-figures/`, which git ignores.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import math
import re
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas
import tifffile
from jax.experimental import enable_x64
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from vertax import (
    BilevelOptimizationMethod,
    BoundedBilevelOptimizer,
    BoundedMesh,
    EdgePlot,
    FacePlot,
    PbcBilevelOptimizer,
    PbcMesh,
    VertexPlot,
    plot_mesh,
)
from vertax.cg import cg_linesearch, gd_linesearch
from vertax.clamped import load_mesh, patch_geometry, relax
from vertax.cost import cost_ratio, cost_v2v, topological_distance
from vertax.energy import (
    energy_shape_factor_hetero,
    energy_shape_factor_homo,
    make_energy_line_tensions,
    make_energy_line_tensions_bounded,
)
from vertax.opt import minimize
from vertax.rheology import make_drive_energy, make_shear_mode, mode_amplitude
from vertax.trace import regularize, trace_mask, trace_mask_closed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
WORK = ROOT / "build" / "doc-figures"

# One colour per object, in every figure. The accents are taken from the Okabe-Ito set, darkened so
# that thin lines keep a contrast of about 3:1 on white; each also differs by a marker or a line style.
INK = "#333333"  # edges, frames and text
CELL = "#EEEBE3"  # cells, when no quantity is shown
CELL_ALT = "#D3CBB8"  # cells of the second fate of a bounded mesh
VERTEX = "#C98500"  # vertices, and their displacements
CLAMPED = "#333333"  # clamped vertices (squares)
HALF_EDGE = "#2B8CC4"  # a half-edge and its twin
CONTACT = "#B8559A"  # the edge of a T1 transition: the contact that changes
REFERENCE = "#9A9A9A"  # a reference geometry (target, observed or initial state), dashed
CELL_CMAP = mpl.colormaps["viridis"]  # a quantity of the cells
TENSION_CMAP = ListedColormap(mpl.colormaps["plasma"](np.linspace(0.0, 0.85, 256)), name="tension")
# The four gradient estimators: the hue and the marker never change between figures.
ESTIMATORS = {
    "AUTOMATIC_DIFFERENTIATION": ("AD", "#3953a4", "o"),
    "IMPLICIT_DIFFERENTIATION": ("Sen-ID", "#8753FF", "^"),
    "ADJOINT_STATE": ("Adj-ID", "#ed2024", "s"),
    "EQUILIBRIUM_PROPAGATION": ("EP", "#0c8140", "D"),
}
# Solvers are told apart by grey level and line style, so that no hue takes a second meaning.
SOLVERS = {
    "optax.sgd": ("#333333", "-"),
    "cg_linesearch": ("#666666", "--"),
    "optax.lbfgs": ("#999999", ":"),
}

WIDTH_IN = 6.6  # the width of a figure, in inches
DISPLAY_SCALE = 1.2  # the SVG files declare a size 1.2 times that of the drawing (see `save`)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.5,
        "svg.fonttype": "path",
        "svg.hashsalt": "vertax-documentation",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# --------------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------------


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to `figures/`, without the date, so that a new run gives the same file."""
    path = OUT / name
    metadata = {"Date": None} if name.endswith(".svg") else {"Software": None}
    fig.savefig(path, metadata=metadata, bbox_inches="tight", pad_inches=0.03, dpi=110)
    plt.close(fig)
    if name.endswith(".svg"):
        # Shown 1.2 times larger than drawn: 9-point text then reads as about 14 pixels, the size of the text of the
        # page, and a figure drawn 6.6 inches wide fills the width of the page.
        text = path.read_text()
        header = re.search(r'<svg [^>]*width="([0-9.]+)pt" height="([0-9.]+)pt"', text)
        w, h = (float(v) * DISPLAY_SCALE for v in header.groups())
        start, end = header.span()
        text = text[:start] + re.sub(r'width="[0-9.]+pt" height="[0-9.]+pt"', f'width="{w:.1f}pt" height="{h:.1f}pt"', text[start:end]) + text[end:]
        path.write_text(text)
    print(f"  {name}: {path.stat().st_size / 1024:.0f} kB")


def style_plot(ax: plt.Axes) -> None:
    """A plot: closed frame and a light dashed grid behind the data."""
    ax.grid(True, ls="--", alpha=0.6, lw=0.5)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    """A short label above the top-left corner of a panel."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def mesh_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.axis("off")


SHIFTS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def periodic_copies(points: np.ndarray, W: float, H: float) -> list[np.ndarray]:
    """The periodic copies of a polygon or a segment that reach into the box, the only ones drawn."""
    copies = []
    for sx, sy in SHIFTS:
        copy = points + np.array([sx * W, sy * H])
        low, high = copy.min(axis=0), copy.max(axis=0)
        if high[0] > 0 and low[0] < W and high[1] > 0 and low[1] < H:
            copies.append(copy)
    return copies


def pbc_tables(mesh: PbcMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    return (
        np.asarray(mesh.vertices, dtype=float),
        np.asarray(mesh.edges),
        np.asarray(mesh.faces),
        float(mesh.width),
        float(mesh.height),
    )


def pbc_polygons(mesh: PbcMesh) -> list[np.ndarray]:
    """The corners of every face, unwrapped across the periodic boundary with columns 6 and 7."""
    V, E, F, W, H = pbc_tables(mesh)
    polygons = []
    for start in F:
        he, offset, corners = int(start), np.zeros(2), []
        while True:
            corners.append(V[E[he, 3]] + offset)
            offset = offset + E[he, 6:8] * np.array([W, H])
            he = int(E[he, 1])
            if he == int(start):
                break
        polygons.append(np.array(corners))
    return polygons


def pbc_segment(mesh: PbcMesh, he: int) -> np.ndarray:
    """A half-edge from its source to the periodic image of its target."""
    V, E, _, W, H = pbc_tables(mesh)
    return np.array([V[E[he, 3]], V[E[he, 4]] + E[he, 6:8] * np.array([W, H])])


def draw_pbc(
    ax: plt.Axes,
    mesh: PbcMesh,
    face_colors: list | None = None,
    fill: bool = True,
    edge_color: str = INK,
    edge_values: np.ndarray | None = None,
    edge_norm: Normalize | None = None,
    lw: float = 0.8,
    ls: str = "-",
    vertices: bool = False,
    zorder: float = 1,
    frame: bool = True,
) -> None:
    """A periodic mesh in its box: the faces and edges that cross the boundary come back on the other side."""
    V, E, _, W, H = pbc_tables(mesh)
    clip = Rectangle((0, 0), W, H, transform=ax.transData)
    if fill:
        polygons = pbc_polygons(mesh)
        colors = face_colors if face_colors is not None else [CELL] * len(polygons)
        shifted, shifted_colors = [], []
        for polygon, color in zip(polygons, colors):
            for copy in periodic_copies(polygon, W, H):
                shifted.append(copy)
                shifted_colors.append(color)
        faces = PolyCollection(shifted, facecolors=shifted_colors, edgecolors="none", zorder=zorder)
        ax.add_collection(faces)
        faces.set_clip_path(clip)
    segments, colors = [], []
    for he in range(len(E)):
        if he > int(E[he, 2]):  # each edge once
            continue
        segment = pbc_segment(mesh, he)
        color = edge_color if edge_values is None else TENSION_CMAP(edge_norm(edge_values[he]))
        for copy in periodic_copies(segment, W, H):
            segments.append(copy)
            colors.append(color)
    edges = LineCollection(segments, colors=colors, linewidths=lw, linestyles=ls, zorder=zorder + 1, capstyle="round")
    ax.add_collection(edges)
    edges.set_clip_path(clip)
    if vertices:
        ax.scatter(V[:, 0], V[:, 1], s=7, color=VERTEX, zorder=zorder + 2, linewidths=0)
    if frame:
        ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=INK, lw=0.8, zorder=zorder + 3))
    ax.set_xlim(-0.02 * W, 1.02 * W)
    ax.set_ylim(-0.02 * H, 1.02 * H)
    mesh_axes(ax)


def pbc_edge_between(mesh: PbcMesh, face_a: int, face_b: int) -> int:
    """A half-edge of face `face_a` whose twin is in face `face_b`."""
    E = np.asarray(mesh.edges)
    for he in range(len(E)):
        if E[he, 5] == face_a and E[int(E[he, 2]), 5] == face_b:
            return he
    raise ValueError(f"faces {face_a} and {face_b} are not neighbours")


def draw_pbc_half_edge(ax: plt.Axes, mesh: PbcMesh, he: int, color: str, lw: float = 2.2, zorder: float = 6) -> None:
    W, H = float(mesh.width), float(mesh.height)
    segment = pbc_segment(mesh, he)
    clip = Rectangle((0, 0), W, H, transform=ax.transData)
    lines = LineCollection(
        periodic_copies(segment, W, H),
        colors=color, linewidths=lw, zorder=zorder, capstyle="round",
    )
    ax.add_collection(lines)
    lines.set_clip_path(clip)


def contacts(mesh: PbcMesh) -> set[tuple[int, int]]:
    """The pairs of neighbouring cells, read from the face of each half-edge and of its twin."""
    E = np.asarray(mesh.edges)
    return {tuple(sorted((int(E[he, 5]), int(E[int(E[he, 2]), 5])))) for he in range(len(E))}


def arc_points(half_angle: float, a: np.ndarray, b: np.ndarray, n: int = 24) -> list[np.ndarray]:
    """Points of the circular arc of half-angle `half_angle` from `a` to `b`, bulging to the right of a -> b."""
    chord = b - a
    half = np.linalg.norm(chord) / 2
    radius = half / np.sin(half_angle)
    unit = chord / np.linalg.norm(chord)
    center = (a + b) / 2 + np.array([-unit[1], unit[0]]) * np.sqrt(radius**2 - half**2)
    t_a = np.arctan2(*(a - center)[::-1])
    t_b = np.arctan2(*(b - center)[::-1])
    if abs(t_b - t_a) > np.pi:
        if t_a < t_b:
            t_a += 2 * np.pi
        else:
            t_b += 2 * np.pi
    return [center + radius * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(t_a, t_b, n)[1:-1]]


def bounded_half_edge(mesh: BoundedMesh, he: int) -> np.ndarray:
    """The points of a half-edge of a bounded mesh: a chord, or an arc for an edge of the outline."""
    V, E, angles = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.edges), np.asarray(mesh.angles)
    a = V[E[he, 3] + E[he, 5] - 2]
    b = V[E[he, 4] + E[he, 6] - 3]
    if E[he, 3] == 0 and angles[he // 2] > 1e-3:
        return np.array([a, *arc_points(float(angles[he // 2]), a, b), b])
    return np.array([a, b])


def bounded_polygons(mesh: BoundedMesh) -> list[np.ndarray]:
    E, F = np.asarray(mesh.edges), np.asarray(mesh.faces)
    polygons = []
    for start in F[:, 0]:
        he, points = int(start), []
        while True:
            points.extend(bounded_half_edge(mesh, he)[:-1])
            he = int(E[he, 1])
            if he == int(start):
                break
        polygons.append(np.array(points))
    return polygons


def draw_bounded(
    ax: plt.Axes,
    mesh: BoundedMesh,
    face_colors: list | None = None,
    fill: bool = True,
    edge_color: str = INK,
    edge_values: np.ndarray | None = None,
    edge_norm: Normalize | None = None,
    lw: float = 0.8,
    ls: str = "-",
    vertices: bool = False,
    zorder: float = 1,
) -> None:
    """A bounded mesh; `edge_values` has one value per edge (half-edges 2k and 2k + 1)."""
    E = np.asarray(mesh.edges)
    if fill:
        if face_colors is None:
            fates = np.asarray(mesh.faces)[:, 1]
            face_colors = [CELL if fate == 0 else CELL_ALT for fate in fates]
        ax.add_collection(PolyCollection(bounded_polygons(mesh), facecolors=face_colors, edgecolors="none", zorder=zorder))
    segments, colors = [], []
    for he in range(len(E)):
        segments.append(bounded_half_edge(mesh, he))
        colors.append(edge_color if edge_values is None else TENSION_CMAP(edge_norm(edge_values[he // 2])))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=lw, linestyles=ls, zorder=zorder + 1, capstyle="round"))
    if vertices:
        V = np.asarray(mesh.vertices)
        ax.scatter(V[:, 0], V[:, 1], s=7, color=VERTEX, zorder=zorder + 2, linewidths=0)
    ax.autoscale_view()
    mesh_axes(ax)


def colorbar(fig: plt.Figure, axes, cmap, norm: Normalize, label: str, **kwargs) -> None:
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, pad=0.02, **kwargs)
    bar.set_label(label)
    bar.outline.set_linewidth(0.6)
    bar.ax.tick_params(width=0.6)


def unwrap(d: np.ndarray, W: float, H: float) -> np.ndarray:
    """A displacement taken to its nearest periodic image."""
    return d - np.array([W, H]) * np.round(d / np.array([W, H]))


def relax_lbfgs(mesh: PbcMesh, energy, min_dist: float = 0.005, iterations: int = 3000) -> float:
    """Relax a periodic mesh with L-BFGS and T1 transitions; return the force residual."""
    (v, he, f, *_), _ = minimize(
        mesh.vertices, mesh.edges, mesh.faces, mesh.width, mesh.height,
        mesh.vertices_params, mesh.edges_params, mesh.faces_params, energy, optax.lbfgs(), min_dist,
        iterations_max=iterations, tolerance=0.0, perform_energy_check=True,
        cap_vertex_step=True, reset_solver_on_t1=True,
    )
    mesh.vertices, mesh.edges, mesh.faces = v, he, f
    g = jax.grad(energy)(v, he, f, mesh.vertices_params, mesh.edges_params, mesh.faces_params)
    return float(jnp.linalg.norm(g))


# --------------------------------------------------------------------------------------------------
# User guide
# --------------------------------------------------------------------------------------------------


def energy_shape_index() -> None:
    """Energies: relaxed tissues coloured by the shape index P / sqrt(A), at p0 = 3.5 and p0 = 3.9."""
    n_cells, side = 64, 8.0
    results = {}
    with enable_x64():
        for p0 in (3.5, 3.9):
            mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=3)
            mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
            mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
            mesh.faces_params = jnp.full(n_cells, p0)

            def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
                return energy_shape_factor_homo(vertTable, heTable, faceTable, side, side, face_params)

            residual = relax_lbfgs(mesh, energy)
            faces = jnp.arange(n_cells)
            q = np.asarray(mesh.get_perimeter(faces) / jnp.sqrt(mesh.get_area(faces)))
            e = float(energy(mesh.vertices, mesh.edges, mesh.faces, None, None, mesh.faces_params))
            results[p0] = (mesh, q)
            print(f"  p0 = {p0}: energy {e:.3e}, residual {residual:.1e}, q in [{q.min():.3f}, {q.max():.3f}], mean {q.mean():.3f}")
    norm = Normalize(3.5, 4.1)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0))
    for ax, (p0, (mesh, q)) in zip(axes, results.items()):
        draw_pbc(ax, mesh, face_colors=[CELL_CMAP(norm(v)) for v in q])
        panel_label(ax, f"$p_0 = {p0}$")
    colorbar(fig, axes, CELL_CMAP, norm, r"shape index $P/\sqrt{A}$", extend="max")
    save(fig, "energy-shape-index.svg")


def relaxation_solvers() -> None:
    """Relaxation: the tissue before and after, and energy and force norm against iteration for three solvers."""
    n_cells = 20
    width = height = math.sqrt(n_cells)
    checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    with enable_x64():

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)

        mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
        vp, hp, fp = jnp.zeros(mesh.nb_vertices), jnp.zeros(mesh.nb_half_edges), jnp.full(mesh.nb_faces, 3.7)
        before = PbcMesh.copy_mesh(mesh)
        curves, after = {}, None
        g0 = float(jnp.linalg.norm(jax.grad(energy)(mesh.vertices, mesh.edges, mesh.faces, vp, hp, fp)))
        e0 = float(energy(mesh.vertices, mesh.edges, mesh.faces, vp, hp, fp))
        for name, solver, line_search in [
            ("optax.sgd", optax.sgd(learning_rate=0.01), False),
            ("cg_linesearch", cg_linesearch(), True),
            ("optax.lbfgs", optax.lbfgs(), True),
        ]:
            residuals = []
            for k in checkpoints:
                (v, he, fa, *_), (history, steps) = minimize(
                    mesh.vertices, mesh.edges, mesh.faces, width, height, vp, hp, fp, energy, solver, 0.005,
                    iterations_max=k, tolerance=0.0, cap_vertex_step=line_search, reset_solver_on_t1=line_search,
                )
                residuals.append(float(jnp.linalg.norm(jax.grad(energy)(v, he, fa, vp, hp, fp))))
            energies = np.concatenate([[e0], np.asarray(history)[: int(steps)]])
            curves[name] = (energies, [g0, *residuals])
            print(f"  {name}: energy {energies[-1]:.6f}, residual {residuals[-1]:.2e}")
            if name == "optax.lbfgs":
                after = PbcMesh.copy_mesh(mesh)
                after.vertices, after.edges, after.faces = v, he, fa
    fig = plt.figure(figsize=(WIDTH_IN, 5.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8], hspace=0.22, wspace=0.32)
    for i, (m, label) in enumerate([(before, "before: Voronoi tissue"), (after, "after 500 iterations of L-BFGS")]):
        ax = fig.add_subplot(grid[0, i])
        draw_pbc(ax, m, vertices=True)
        panel_label(ax, label)
    ax_e, ax_g = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for name, (energies, residuals) in curves.items():
        color, ls = SOLVERS[name]
        ax_e.plot(np.arange(len(energies)) + 1, energies, color=color, ls=ls, lw=1.2, label=name)
        ax_g.plot(np.array([0, *checkpoints]) + 1, residuals, color=color, ls=ls, lw=1.2, marker="o", ms=2.5, label=name)
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("iteration + 1")
    ax_e.set_ylabel(r"energy $E_1$")
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    ax_g.set_xlabel("iteration + 1")
    ax_g.set_ylabel(r"force norm $\|\nabla_X E_1\|$")
    ax_g.legend(loc="lower left")
    for ax in (ax_e, ax_g):
        style_plot(ax)
    save(fig, "relaxation-solvers.svg")


def costs_vertex_to_vertex() -> None:
    """Costs: a relaxed tissue, its target, and the vertex displacements that C_v2v sums."""
    n_cells = 20
    width = height = math.sqrt(n_cells)
    all_faces = jnp.arange(n_cells)
    with enable_x64():

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return energy_shape_factor_hetero(vertTable, heTable, faceTable, width, height, all_faces, face_params)

        mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
        mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
        mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
        mesh.faces_params = jnp.full(n_cells, 3.7)
        optimizer = PbcBilevelOptimizer()
        optimizer.loss_function_inner = energy
        optimizer.inner_optimization(mesh)
        target = PbcMesh.copy_mesh(mesh)
        target.faces_params = 3.7 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n_cells,))
        optimizer.inner_optimization(target)
        cost = float(cost_v2v(mesh.vertices, mesh.edges, mesh.faces, width, height, target.vertices, target.edges, target.faces))
    V, T = np.asarray(mesh.vertices), np.asarray(target.vertices)
    d = unwrap(T - V, width, height)
    print(f"  cost_v2v {cost:.3e}; displacement: mean {np.linalg.norm(d, axis=1).mean():.3f}, max {np.linalg.norm(d, axis=1).max():.3f}")
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    draw_pbc(ax, mesh, lw=0.8)
    draw_pbc(ax, target, fill=False, edge_color=REFERENCE, ls=(0, (3, 2)), lw=0.8, zorder=3, frame=False)
    magnify = 5  # the displacements are a few hundredths of a cell size: drawn five times longer
    ax.quiver(V[:, 0], V[:, 1], magnify * d[:, 0], magnify * d[:, 1], angles="xy", scale_units="xy", scale=1, color=VERTEX,
              width=0.006, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=8)
    ax.scatter(V[:, 0], V[:, 1], s=9, color=VERTEX, zorder=9, linewidths=0)
    handles = [
        Line2D([], [], color=INK, lw=0.8, label="relaxed tissue"),
        Line2D([], [], color=REFERENCE, lw=0.8, ls=(0, (3, 2)), label="target"),
        Line2D([], [], color=VERTEX, marker="o", ms=3, lw=1.2, label=r"$\mathbf{x}_i^{\mathrm{target}} - \mathbf{x}_i$, drawn $\times 5$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "cost-vertex-to-vertex.svg")


def costs_topological_distance() -> None:
    """Costs: two tissues one T1 transition apart, with the contact that changes."""
    n_cells = 20
    width = height = math.sqrt(n_cells)
    with enable_x64():
        e2 = make_energy_line_tensions(10.0)

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)

        start = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
        start.vertices_params = jnp.zeros(start.nb_vertices)
        start.edges_params = jnp.ones(start.nb_half_edges)
        start.faces_params = jnp.ones(start.nb_faces)
        optimizer = PbcBilevelOptimizer()
        optimizer.loss_function_inner = energy
        optimizer.inner_solver = optax.sgd(learning_rate=0.01)
        optimizer.min_dist_T1 = 0.005
        optimizer.max_nb_iterations = 1000
        optimizer.tolerance = 1e-4
        optimizer.patience = 5
        optimizer.inner_optimization(start)
        target = PbcMesh.copy_mesh(start)
        target.edges_params = jnp.repeat(1.0 + 0.3 * jax.random.normal(jax.random.PRNGKey(7), (start.nb_edges,)), 2)
        optimizer.update_T1 = True
        optimizer.inner_optimization(target)
        d_topo = topological_distance(start.vertices, start.edges, start.faces, width, height,
                                      target.vertices, target.edges, target.faces)
    lost = sorted(contacts(start) - contacts(target))
    gained = sorted(contacts(target) - contacts(start))
    print(f"  d_T1 = {d_topo // 2}; contact lost {lost}, gained {gained}")
    assert d_topo == 2 and len(lost) == 1 and len(gained) == 1
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN * 0.8, 2.55))
    for ax, m, pair, label in [(axes[0], start, lost[0], "starting tissue"), (axes[1], target, gained[0], r"target, $d_{\mathrm{T1}} = 1$")]:
        colors = [CELL] * n_cells
        for f in (*lost[0], *gained[0]):
            colors[f] = CELL_ALT
        draw_pbc(ax, m, face_colors=colors)
        draw_pbc_half_edge(ax, m, pbc_edge_between(m, *pair), CONTACT, lw=3.0)
        panel_label(ax, label)
    handles = [
        Line2D([], [], color=CONTACT, lw=3.0, label="edge of the contact that changes"),
        Rectangle((0, 0), 1, 1, fc=CELL_ALT, ec="none", label="the four cells of the T1 transition"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.06))
    save(fig, "cost-topological-distance.svg")


def rheology_drive() -> None:
    """Rheology: the cosine body force on the vertices, and the displacement profile it drives."""
    side = math.sqrt(20.0)
    a0 = 0.02
    with enable_x64():

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return energy_shape_factor_homo(vertTable, heTable, faceTable, side, side, face_params)

        mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=231)
        mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
        mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
        mesh.faces_params = jnp.full(mesh.nb_faces, 3.6)
        optimizer = PbcBilevelOptimizer()
        optimizer.loss_function_inner = energy
        optimizer.max_nb_iterations = 3000
        optimizer.tolerance = 1e-8
        optimizer.inner_optimization(mesh)
        mode = make_shear_mode(mesh.vertices, side, side, n=1)
        (v, *_), _ = minimize(
            mesh.vertices, mesh.edges, mesh.faces, side, side,
            mesh.vertices_params, mesh.edges_params, mesh.faces_params,
            make_drive_energy(energy, mode, a0), optax.sgd(learning_rate=0.01), optimizer.min_dist_T1,
            iterations_max=3000, tolerance=0.0,
        )
        u = float(mode_amplitude(v, mode))
        c = np.asarray(mode.c)
        q = float(mode.q)
    V = np.asarray(mesh.vertices)
    dx = unwrap(np.asarray(v) - V, side, side)[:, 0]
    print(f"  U = {u:.5f} for a0 = {a0}")
    fig, (ax_m, ax_p) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.25})
    draw_pbc(ax_m, mesh)
    arrow = 0.35  # the length, in units of the box, of an arrow of force a0
    ax_m.quiver(V[:, 0], V[:, 1], c * arrow, 0 * c, angles="xy", scale_units="xy", scale=1, color=INK,
                width=0.007, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=8)
    ax_m.scatter(V[:, 0], V[:, 1], s=7, color=VERTEX, zorder=9, linewidths=0)
    panel_label(ax_m, r"force $a\,\hat c_i$ along $x$")
    y = np.linspace(0, side, 200)
    c_mean = float(np.mean(np.cos(q * V[:, 1])))
    ax_p.plot(dx, V[:, 1], "o", color=VERTEX, ms=3.5, label=r"$\Delta x_i$ at $a = +a_0$")
    ax_p.plot(u * (np.cos(q * y) - c_mean), y, color=INK, lw=1.0, label=r"$U\,(\cos q y - \overline{\cos q y^0})$")
    ax_p.set_xlabel(r"displacement $\Delta x$")
    ax_p.set_ylabel(r"height $y^0$")
    ax_p.set_ylim(0, side)
    ax_p.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, handlelength=1.5, columnspacing=1.0)
    style_plot(ax_p)
    save(fig, "rheology-shear-drive.svg")


def images_and_masks() -> None:
    """Images and masks: the image and the mask of `examples/`, and the mesh traced from the mask."""
    image = tifffile.imread(ROOT / "examples" / "image.tif").astype(float)
    mask = tifffile.imread(ROOT / "examples" / "mask.tif")
    step = 2  # shown at half resolution: the page shows each panel at about 350 pixels
    low, high = np.percentile(image, [1, 99.5])
    boundary = np.zeros(mask.shape, bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[1:, :] |= boundary[:-1, :].copy()
    boundary[:, 1:] |= boundary[:, :-1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.25))
    axes[0].imshow(image[::step, ::step], cmap="gray", vmin=low, vmax=high, interpolation="antialiased")
    cells = np.ones((*boundary[::step, ::step].shape, 3)) * mpl.colors.to_rgb(CELL)
    cells[boundary[::step, ::step]] = mpl.colors.to_rgb(INK)
    axes[1].imshow(cells, interpolation="antialiased")
    for ax, label in zip(axes, ["image.tif", "mask.tif (label boundaries)"]):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        panel_label(ax, label)
    save(fig, "image-and-mask.png")

    clean, report = regularize(mask)
    traced = trace_mask_closed(clean, rim_tol=2.0)
    print(f"  traced: {traced.n_verts} vertices ({int(traced.free.sum())} free), {traced.n_faces} faces; moved {report['frac_pixels_moved']:.2e}")
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    draw_traced(ax, traced, mask.shape)
    handles = [
        Line2D([], [], ls="", marker="o", color=VERTEX, ms=4, label="free vertex"),
        Line2D([], [], ls="", marker="s", color=CLAMPED, ms=3.5, label="clamped vertex"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    save(fig, "traced-mask.svg")


def draw_traced(ax: plt.Axes, traced, shape: tuple[int, int], clamped: bool = True) -> None:
    """A traced mesh in pixel coordinates, the image orientation kept (y downwards)."""
    verts = np.asarray(traced.verts, dtype=float)
    polygons = [verts[np.asarray(face)] for face in traced.faces]
    colors = [CELL if interior else CELL_ALT for interior in traced.face_is_interior]
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="none"))
    edges = np.asarray(traced.edges)
    ax.add_collection(LineCollection(verts[edges], colors=INK, linewidths=0.7))
    free = np.asarray(traced.free)
    ax.scatter(verts[free, 0], verts[free, 1], s=7, color=VERTEX, linewidths=0, zorder=4)
    if clamped:
        ax.scatter(verts[~free, 0], verts[~free, 1], s=7, color=CLAMPED, marker="s", linewidths=0, zorder=4)
    ax.add_patch(Rectangle((0, 0), shape[1], shape[0], fill=False, ec=REFERENCE, lw=0.6, ls=(0, (3, 2))))
    ax.set_xlim(-10, shape[1] + 10)
    ax.set_ylim(shape[0] + 10, -10)
    mesh_axes(ax)


def clamped_patch() -> None:
    """Clamped patches: the observed patch and its relaxation, with the clamped rim held."""
    with enable_x64():
        patch = load_mesh(str(ROOT / "tests" / "fixtures" / "clamped_patch_demo_003_pos2.npz"))
        gamma = jnp.ones(len(patch.EA))

        def E(x, K):
            A, P, L = patch_geometry(x, patch)
            return 0.5 * K * jnp.sum((A - 1.0) ** 2) + jnp.sum(gamma * L)

        out = relax(E, patch, K=5.0)
    print(f"  residual observed {out['r_eq_obs']:.2e}, relaxed {out['r_eq']:.2e}")
    x0, x = np.asarray(patch.x0), np.asarray(out["x"])
    free = np.zeros(len(x0), bool)
    free[np.asarray(patch.free_idx)] = True
    PA, PB = np.asarray(patch.PA), np.asarray(patch.PB)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.add_collection(LineCollection(np.stack([x0[PA], x0[PB]], axis=1), colors=REFERENCE, linewidths=0.8, linestyles=(0, (3, 2))))
    ax.add_collection(LineCollection(np.stack([x[PA], x[PB]], axis=1), colors=INK, linewidths=0.8))
    ax.scatter(x[free, 0], x[free, 1], s=9, color=VERTEX, linewidths=0, zorder=4)
    ax.scatter(x[~free, 0], x[~free, 1], s=9, color=CLAMPED, marker="s", linewidths=0, zorder=4)
    ax.autoscale_view()
    ax.invert_yaxis()
    mesh_axes(ax)
    handles = [
        Line2D([], [], color=REFERENCE, lw=0.8, ls=(0, (3, 2)), label="observed"),
        Line2D([], [], color=INK, lw=0.8, label="relaxed"),
        Line2D([], [], ls="", marker="o", color=VERTEX, ms=4, label="free vertex"),
        Line2D([], [], ls="", marker="s", color=CLAMPED, ms=3.5, label="clamped vertex"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "clamped-patch-relaxation.svg")


def plotting_plot_mesh() -> None:
    """Plotting: the two figures that the code of the page draws with `plot_mesh`, without their titles."""
    side = math.sqrt(20)
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.repeat(1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_edges,)), 2)
    mesh.faces_params = jnp.ones(mesh.nb_faces)
    fig, _ = plot_mesh(mesh)
    fig.set_size_inches(3.3, 3.3)
    save(fig, "plot-mesh-multicolor.svg")
    fig, _ = plot_mesh(mesh, face_plot=FacePlot.AREA, edge_plot=EdgePlot.EDGE_PARAMETER,
                       edge_parameters_name="tension", vertex_plot=VertexPlot.BLACK)
    fig.set_size_inches(4.2, 3.3)
    save(fig, "plot-mesh-areas-tensions.svg")


# --------------------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------------------


def example_periodic_relaxation() -> None:
    """Periodic relaxation: 100 cells before and after, coloured by area (the settings of the page)."""
    n_cells = 100
    width = height = math.sqrt(n_cells)
    mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
    mesh.faces_params = jnp.full(mesh.nb_faces, 3.7)

    def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
        return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)

    optimizer = PbcBilevelOptimizer()
    optimizer.loss_function_inner = energy
    optimizer.inner_solver = optax.sgd(learning_rate=0.01)
    optimizer.min_dist_T1 = 0.2
    optimizer.max_nb_iterations = 1000
    optimizer.tolerance = 1e-4
    optimizer.patience = 5
    before = PbcMesh.copy_mesh(mesh)
    history = optimizer.inner_optimization(mesh)
    print(f"  {len(history)} iterations")
    faces = jnp.arange(n_cells)
    areas = [np.asarray(m.get_area(faces)) for m in (before, mesh)]
    norm = Normalize(0.3, 1.7)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0))
    for ax, m, a, label in zip(axes, (before, mesh), areas, ("Voronoi tissue", f"after {len(history)} iterations")):
        draw_pbc(ax, m, face_colors=[CELL_CMAP(norm(v)) for v in a], lw=0.6)
        panel_label(ax, label)
    colorbar(fig, axes, CELL_CMAP, norm, r"cell area $A$", extend="both")
    save(fig, "example-periodic-relaxation.svg")


def bounded_example_mesh() -> BoundedMesh:
    """The cluster of the bounded examples: 20 cells, random_key 1, tension parameters from seed 1."""
    n_cells = 20
    side = math.sqrt(n_cells)
    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=1)
    rng = np.random.default_rng(seed=1)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = jnp.asarray(rng.random(mesh.nb_edges) * 20 - 10)
    mesh.faces_params = jnp.full(mesh.nb_faces, 0.6)
    return mesh


def bounded_example_optimizer() -> BoundedBilevelOptimizer:
    optimizer = BoundedBilevelOptimizer()
    optimizer.loss_function_inner = make_energy_line_tensions_bounded(40.0, sigmoid_tensions=True)
    optimizer.inner_solver = optax.sgd(learning_rate=0.01)
    optimizer.min_dist_T1 = 0.025
    optimizer.max_nb_iterations = 1000
    optimizer.tolerance = 1e-6
    optimizer.patience = 5
    return optimizer


def sigmoid_tensions(mesh: BoundedMesh) -> np.ndarray:
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(mesh.edges_params, dtype=float)))


def two_clusters(meshes: list[BoundedMesh], labels: list[str], name: str) -> None:
    norm = Normalize(1.0, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0))
    for ax, m, label in zip(axes, meshes, labels):
        draw_bounded(ax, m, edge_values=sigmoid_tensions(m), edge_norm=norm, lw=1.6)
        panel_label(ax, label)
    limits = np.array([[*ax.get_xlim(), *ax.get_ylim()] for ax in axes])
    for ax in axes:  # the same scale in both panels
        ax.set_xlim(limits[:, 0].min(), limits[:, 1].max())
        ax.set_ylim(limits[:, 2].min(), limits[:, 3].max())
    colorbar(fig, axes, TENSION_CMAP, norm, r"line tension $\Lambda$")
    save(fig, name)


def example_bounded_relaxation() -> None:
    """Bounded relaxation: the cluster before and after, edges coloured by tension, cells by fate."""
    mesh = bounded_example_mesh()
    before = BoundedMesh.copy_mesh(mesh)
    history = bounded_example_optimizer().inner_optimization(mesh)
    print(f"  {len(history)} iterations")
    two_clusters([before, mesh], ["Voronoi cluster", f"after {len(history)} iterations"], "example-bounded-relaxation.svg")


def example_bounded_elongation() -> None:
    """Bounded elongation: the relaxed cluster, and the same cluster after 5 epochs of design with cost_ratio."""
    mesh = bounded_example_mesh()
    optimizer = bounded_example_optimizer()
    optimizer.inner_optimization(mesh)
    before = BoundedMesh.copy_mesh(mesh)
    optimizer.loss_function_outer = cost_ratio
    optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    optimizer.beta = 0.01
    optimizer.outer_solver = optax.adam(learning_rate=0.01, nesterov=True)
    costs = [float(cost_ratio(mesh.vertices))]
    for _ in range(5):
        optimizer.bilevel_optimization(mesh)
        costs.append(float(cost_ratio(mesh.vertices)))
    print("  cost_ratio by epoch:", np.round(costs, 4).tolist())
    two_clusters([before, mesh], [f"relaxed, cost {costs[0]:.3f}", f"after 5 epochs, cost {costs[-1]:.3f}"], "example-bounded-elongation.svg")


def example_tension_inference() -> None:
    """Tension inference: the cost and the recovered tensions over 20 epochs of EP (the settings of the page)."""
    n_cells = 20
    width = height = math.sqrt(n_cells)
    with enable_x64():
        e2 = make_energy_line_tensions(10.0)

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)

        def random_tensions(seed, n_edges, mean=1.2, std=0.1):
            return jnp.repeat(mean + std * jax.random.normal(jax.random.PRNGKey(seed), (n_edges,)), 2)

        mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
        mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
        mesh.edges_params = random_tensions(1, mesh.nb_edges)
        mesh.faces_params = jnp.ones(mesh.nb_faces)
        optimizer = PbcBilevelOptimizer()
        optimizer.loss_function_inner = energy
        optimizer.inner_solver = optax.sgd(learning_rate=0.01)
        optimizer.update_T1 = True
        optimizer.min_dist_T1 = 0.005
        optimizer.max_nb_iterations = 1000
        optimizer.tolerance = 1e-4
        optimizer.patience = 5
        optimizer.inner_optimization(mesh)
        target = PbcMesh.copy_mesh(mesh)
        target.edges_params = random_tensions(2, mesh.nb_edges)
        optimizer.inner_optimization(target)
        optimizer.vertices_target = target.vertices
        optimizer.edges_target = target.edges
        optimizer.faces_target = target.faces
        optimizer.loss_function_outer = cost_v2v
        optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
        optimizer.beta = 0.01
        optimizer.outer_solver = optax.adam(learning_rate=2e-3, nesterov=True)
        optimizer.mean_tension_gauge = float(target.edges_params.mean())
        truth = np.asarray(target.edges_params[::2])
        costs, estimates = [float(optimizer.compute_outer_loss(mesh))], {0: np.asarray(mesh.edges_params[::2])}
        for epoch in range(1, 21):
            optimizer.bilevel_optimization(mesh)
            costs.append(float(optimizer.compute_outer_loss(mesh)))
        estimates[20] = np.asarray(mesh.edges_params[::2])

    def error_over_spread(estimate):
        return np.linalg.norm((estimate - estimate.mean()) - (truth - truth.mean())) / np.linalg.norm(truth - truth.mean())

    for epoch, estimate in estimates.items():
        print(f"  epoch {epoch}: cost {costs[epoch]:.3e}, Pearson {np.corrcoef(estimate, truth)[0, 1]:+.3f}, error/spread {error_over_spread(estimate):.3f}")
    _, color, marker = ESTIMATORS["EQUILIBRIUM_PROPAGATION"]
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.8), gridspec_kw={"wspace": 0.35})
    ax_c.plot(np.arange(len(costs)), costs, color=color, marker=marker, ms=3, lw=1.0)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("epoch")
    ax_c.set_ylabel(r"cost $\mathcal{C}_{\mathrm{v2v}}$")
    low, high = truth.min() - 0.03, truth.max() + 0.03
    ax_s.plot([low, high], [low, high], color=REFERENCE, ls="--", lw=0.8)
    ax_s.plot(truth, estimates[0], ls="", marker=marker, mfc="none", mec=color, ms=4, mew=0.8, label="epoch 0")
    ax_s.plot(truth, estimates[20], ls="", marker=marker, color=color, ms=4, label="epoch 20")
    ax_s.set_xlabel(r"target tension $\Lambda^{\mathrm{target}}$")
    ax_s.set_ylabel(r"tension $\Lambda$")
    ax_s.legend(loc="upper left")
    for ax in (ax_c, ax_s):
        style_plot(ax)
    save(fig, "example-tension-inference.svg")


def example_gradient_estimators() -> None:
    """Gradient estimators: the four estimates from an equilibrium, and their norms from an unrelaxed state."""
    WIDTH = HEIGHT = 4.0
    with enable_x64():
        mesh = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
        all_faces = jnp.arange(mesh.nb_faces)

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return energy_shape_factor_hetero(vertTable, heTable, faceTable, WIDTH, HEIGHT, all_faces, face_params)

        mesh.vertices_params = jnp.zeros(1)
        mesh.edges_params = jnp.zeros(1)
        mesh.faces_params = 3.7 + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_faces,))
        relax_lbfgs(mesh, energy, min_dist=0.05, iterations=2000)
        target = PbcMesh.copy_mesh(mesh)
        target.faces_params = 3.7 + 0.05 * jax.random.normal(jax.random.PRNGKey(2), (mesh.nb_faces,))
        relax_lbfgs(target, energy, min_dist=0.05, iterations=2000)
        optimizer = PbcBilevelOptimizer()
        optimizer.loss_function_inner = energy
        optimizer.loss_function_outer = cost_v2v
        optimizer.inner_solver = optax.sgd(learning_rate=0.01)
        optimizer.outer_solver = optax.sgd(learning_rate=1.0)
        optimizer.update_T1 = False
        optimizer.min_dist_T1 = 0.05
        optimizer.max_nb_iterations = 2000
        optimizer.tolerance = 0.0
        optimizer.beta = 1e-3
        optimizer.vertices_target = target.vertices
        optimizer.edges_target = target.edges
        optimizer.faces_target = target.faces

        def estimates(start):
            gradients = {}
            for method in BilevelOptimizationMethod:
                trial = PbcMesh.copy_mesh(start)
                optimizer.bilevel_optimization_method = method
                optimizer.outer_optimization(trial)
                gradients[method.name] = np.asarray(start.faces_params - trial.faces_params)
            return gradients

        at_equilibrium = estimates(mesh)
        unrelaxed = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
        unrelaxed.vertices_params = mesh.vertices_params
        unrelaxed.edges_params = mesh.edges_params
        unrelaxed.faces_params = 3.7 + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (mesh.nb_faces,))
        optimizer.max_nb_iterations = 200
        far = estimates(unrelaxed)
    reference = at_equilibrium["ADJOINT_STATE"]
    for name, g in at_equilibrium.items():
        print(f"  {name}: norm {np.linalg.norm(g):.3e}, from Adj-ID {np.linalg.norm(g - reference) / np.linalg.norm(reference):.1e}; unrelaxed norm {np.linalg.norm(far[name]):.2e}")
    fig, (ax_g, ax_n) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.4})
    index = np.arange(len(reference))
    offsets = {"AUTOMATIC_DIFFERENTIATION": -0.24, "IMPLICIT_DIFFERENTIATION": -0.08, "ADJOINT_STATE": 0.08, "EQUILIBRIUM_PROPAGATION": 0.24}
    for name in ESTIMATORS:
        g = at_equilibrium[name]
        label, color, marker = ESTIMATORS[name]
        ax_g.plot(index + offsets[name], g * 1e3, ls="", marker=marker, color=color, ms=3.5, label=label)
    ax_g.axhline(0, color=REFERENCE, lw=0.6)
    ax_g.set_xlabel(r"cell $\alpha$")
    ax_g.set_ylabel(r"$\partial \mathcal{C} / \partial p^0_\alpha$  ($\times 10^{-3}$)")
    ax_g.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, handletextpad=0.2, columnspacing=0.8)
    for row, name in enumerate(ESTIMATORS):
        label, color, marker = ESTIMATORS[name]
        ax_n.plot([np.linalg.norm(at_equilibrium[name])], [row + 0.15], ls="", marker=marker, color=color, ms=5)
        ax_n.plot([np.linalg.norm(far[name])], [row - 0.15], ls="", marker=marker, mfc="none", mec=color, mew=1.0, ms=5)
    ax_n.set_yticks(range(len(ESTIMATORS)), [ESTIMATORS[n][0] for n in ESTIMATORS])
    ax_n.set_ylim(-0.6, len(ESTIMATORS) - 0.4)
    ax_n.invert_yaxis()
    ax_n.set_xscale("log")
    ax_n.set_xlabel("norm of the estimated gradient")
    handles = [
        Line2D([], [], ls="", marker="o", color=INK, ms=4.5, label="from the equilibrium"),
        Line2D([], [], ls="", marker="o", mfc="none", mec=INK, ms=4.5, label="from the unrelaxed tissue"),
    ]
    ax_n.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=1, fontsize=7.5)
    for ax in (ax_g, ax_n):
        style_plot(ax)
    save(fig, "example-gradient-estimators.svg")


def example_many_epochs() -> None:
    """Many epochs: the records of `do_n_bilevel_optimization`, read back from summary.csv."""
    n_cells = 20
    width = height = math.sqrt(n_cells)
    e2 = make_energy_line_tensions(20.0)

    def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
        return e2(vertTable, heTable, faceTable, width, height, he_params, face_params)

    def tensions(seed, n_edges):
        return jnp.repeat(1.2 + 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_edges,)), 2)

    mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
    mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
    mesh.edges_params = tensions(1, mesh.nb_edges)
    mesh.faces_params = jnp.ones(n_cells)
    optimizer = PbcBilevelOptimizer()
    optimizer.loss_function_inner = energy
    optimizer.min_dist_T1 = 0.005
    optimizer.max_nb_iterations = 1000
    optimizer.tolerance = 1e-4
    optimizer.patience = 5
    optimizer.inner_solver = optax.sgd(learning_rate=0.01)
    optimizer.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)
    optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    optimizer.beta = 0.01
    optimizer.inner_optimization(mesh)
    target = PbcMesh.copy_mesh(mesh)
    target.edges_params = tensions(2, mesh.nb_edges)
    optimizer.inner_optimization(target)
    optimizer.vertices_target = target.vertices
    optimizer.edges_target = target.edges
    optimizer.faces_target = target.faces
    optimizer.loss_function_outer = cost_v2v

    def tension_correlation(mesh, optimizer):
        return float(jnp.corrcoef(mesh.edges_params[::2], target.edges_params[::2])[0, 1])

    optimizer.add_custom_metric("Tension correlation", tension_correlation)
    folder = WORK / "many-epochs"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    optimizer.do_n_bilevel_optimization(nb_epochs=6, mesh=mesh, report_every=1, save_folder=str(folder))
    summary = pandas.read_csv(folder / "summary.csv", index_col=0)
    print(summary[["Epoch", "Outer cost", "Tension correlation"]].round(6).to_string(index=False))
    _, color, marker = ESTIMATORS["EQUILIBRIUM_PROPAGATION"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.5), gridspec_kw={"wspace": 0.35})
    for ax, column, label in [(axes[0], "Outer cost", r"outer cost $\mathcal{C}_{\mathrm{v2v}}$"), (axes[1], "Tension correlation", "tension correlation")]:
        ax.plot(summary["Epoch"], summary[column], color=color, marker=marker, ms=3, lw=1.0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        style_plot(ax)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    save(fig, "example-many-epochs.svg")


def example_vertex_step_cap() -> None:
    """Line-search solvers: one step of gd_linesearch on a bounded cluster, without and with the step cap."""
    MIN_DIST_T1 = 0.01
    energy = make_energy_line_tensions_bounded(10.0)
    solver = gd_linesearch()
    seed, n_cells = 5, 20
    side = math.sqrt(n_cells)
    results = {}
    for cap in (False, None):
        mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=side, height=side, random_key=seed)
        rng = np.random.default_rng(seed)
        mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
        mesh.edges_params = jnp.asarray(1.0 + 0.5 * rng.random(mesh.nb_edges))
        mesh.faces_params = jnp.full(mesh.nb_faces, 1.5)
        before = BoundedMesh.copy_mesh(mesh)
        optimizer = BoundedBilevelOptimizer()
        optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
        optimizer.inner_solver = solver
        optimizer.loss_function_inner = energy
        optimizer.min_dist_T1 = MIN_DIST_T1
        optimizer.max_nb_iterations = 1
        optimizer.tolerance = 0.0
        optimizer.cap_vertex_step = cap
        optimizer.inner_optimization(mesh, perform_energy_check=False)
        step = np.abs(np.asarray(mesh.vertices) - np.asarray(before.vertices)).max() / MIN_DIST_T1
        folded = np.flatnonzero(np.asarray(mesh.get_area(jnp.arange(mesh.nb_faces))) <= 0)
        results[cap] = (before, mesh, step, folded)
        print(f"  cap_vertex_step={cap}: largest step {step:.1f} x min_dist_T1, folded cells {folded.tolist()}")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.1), gridspec_kw={"wspace": 0.15})
    for ax, cap in zip(axes, (False, None)):
        before, after, step, folded = results[cap]
        colors = [CELL] * after.nb_faces
        draw_bounded(ax, after, face_colors=colors, lw=0.8, zorder=2)
        draw_bounded(ax, before, fill=False, edge_color=REFERENCE, ls=(0, (3, 2)), lw=0.8, zorder=1)
        for f in folded:  # a folded cell is small: it is circled
            center = bounded_polygons(after)[f].mean(axis=0)
            ax.scatter(*center, s=260, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=9)
        b, a = np.asarray(before.vertices), np.asarray(after.vertices)
        ax.quiver(b[:, 0], b[:, 1], a[:, 0] - b[:, 0], a[:, 1] - b[:, 1], angles="xy", scale_units="xy", scale=1,
                  color=VERTEX, width=0.005, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=8)
        name = "uncapped (False)" if cap is False else "capped (None, the default)"
        panel_label(ax, f"{name}\nlargest step {step:.0f} " + r"$\times$ min_dist_T1")
    limits = np.array([[*ax.get_xlim(), *ax.get_ylim()] for ax in axes])
    for ax in axes:
        ax.set_xlim(limits[:, 0].min(), limits[:, 1].max())
        ax.set_ylim(limits[:, 2].min(), limits[:, 3].max())
    handles = [
        Line2D([], [], color=REFERENCE, lw=0.8, ls=(0, (3, 2)), label="before the step"),
        Line2D([], [], color=INK, lw=0.8, label="after the step"),
        Line2D([], [], color=VERTEX, lw=1.2, label="vertex step"),
        Line2D([], [], ls="", marker="o", mfc="none", mec=INK, ms=9, label="folded cell"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.08))
    save(fig, "example-vertex-step-cap.svg")


def example_t1_counts() -> None:
    """T1 statistics: edges presented to the T1 update and flips kept, per iteration, without and with the energy check."""
    WIDTH = HEIGHT = 4.0
    MIN_DIST = 0.05
    with enable_x64():
        e2 = make_energy_line_tensions(10.0)

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return e2(vertTable, heTable, faceTable, WIDTH, HEIGHT, he_params, face_params)

        tissue = PbcMesh.from_random_seeds(nb_seeds=16, width=WIDTH, height=HEIGHT, random_key=2)
        rng = np.random.default_rng(0)
        tensions = jnp.asarray(1.0 + 0.6 * rng.random(tissue.nb_half_edges))
        solver = optax.sgd(learning_rate=0.01)
        counts = {}
        for check in (False, True):
            _, (history, steps, c) = minimize(
                tissue.vertices, tissue.edges, tissue.faces, WIDTH, HEIGHT,
                jnp.zeros(tissue.nb_vertices), tensions, jnp.ones(tissue.nb_faces), energy, solver, MIN_DIST,
                iterations_max=400, tolerance=0.0, perform_energy_check=check, count_t1=True,
            )
            counts[check] = np.asarray(c)
            print(f"  energy check {check}: {counts[check][:, 0].sum()} presented, {counts[check][:, 1].sum()} kept")
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_IN, 2.9), sharex=True, gridspec_kw={"hspace": 0.35})
    top = max(int(c.max()) for c in counts.values()) + 1
    for ax, check in zip(axes, (False, True)):
        c = counts[check]
        it = np.flatnonzero(c[:, 0])
        ax.vlines(it + 1, 0, c[it, 0], color=REFERENCE, lw=3.0, label="edges presented")
        kept = np.flatnonzero(c[:, 1])
        ax.plot(kept + 1, c[kept, 1], ls="", marker="o", color=CONTACT, ms=4, label="flips kept")
        ax.set_ylim(0, top)
        ax.set_xscale("log")
        ax.set_xlim(0.8, len(c) * 1.2)
        ax.set_ylabel("count")
        panel_label(ax, f"perform_energy_check={check}")
        style_plot(ax)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_xlabel("iteration + 1")
    save(fig, "example-t1-counts.svg")


def example_shear_compliance() -> None:
    """Shear rheometry: the displacements along the drive at +a0 and -a0, against the weights c_i."""
    SIDE = math.sqrt(20.0)
    A0 = 0.02
    with enable_x64():
        mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=SIDE, height=SIDE, random_key=231)
        mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
        mesh.edges_params = jnp.zeros(mesh.nb_half_edges)
        mesh.faces_params = jnp.full(mesh.nb_faces, 3.60)

        def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
            return energy_shape_factor_homo(vertTable, heTable, faceTable, SIDE, SIDE, face_params)

        params = (mesh.vertices_params, mesh.edges_params, mesh.faces_params)
        solver = optax.sgd(learning_rate=0.01)
        (reference, edges_ref, faces_ref, *_), _ = minimize(
            mesh.vertices, mesh.edges, mesh.faces, SIDE, SIDE, *params, energy, solver, 0.005,
            iterations_max=4000, tolerance=0.0)
        mode = make_shear_mode(reference, SIDE, SIDE, n=1)
        displacement, amplitude = {}, {}
        for sign in (+1, -1):
            (v, *_), _ = minimize(reference, edges_ref, faces_ref, SIDE, SIDE, *params,
                                  make_drive_energy(energy, mode, sign * A0), solver, 0.005,
                                  iterations_max=4000, tolerance=0.0)
            amplitude[sign] = float(mode_amplitude(v, mode))
            displacement[sign] = unwrap(np.asarray(v) - np.asarray(reference), SIDE, SIDE)[:, 0]
        c = np.asarray(mode.c)
    j = (amplitude[+1] - amplitude[-1]) / (2 * A0)
    print(f"  U+ = {amplitude[+1]:.5f}, U- = {amplitude[-1]:.5f}, J = {j:.4f}")
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    line = np.linspace(c.min(), c.max(), 2)
    for sign, fill in ((+1, VERTEX), (-1, "none")):
        label = rf"$a = {'+' if sign > 0 else '-'}a_0$, slope $U_{'+' if sign > 0 else '-'}$"
        ax.plot(c, displacement[sign], ls="", marker="o", mfc=fill, mec=VERTEX, mew=0.8, ms=4, label=label)
        ax.plot(line, amplitude[sign] * line, color=INK, lw=0.8, ls="-" if sign > 0 else "--")
    ax.set_xlabel(r"weight $\hat c_i$")
    ax.set_ylabel(r"displacement $\Delta x_i$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
    style_plot(ax)
    save(fig, "example-shear-compliance.svg")


def example_trace_rim_tolerance() -> None:
    """Traced patch: the fixture mask traced with rim_tol 1, 2 and 8 pixels."""
    mask = tifffile.imread(ROOT / "tests" / "trace_fixture_demo_003_pos2_mask.tif")
    clean, _ = regularize(mask)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.4))
    for ax, tol in zip(axes, (1.0, 2.0, 8.0)):
        traced = trace_mask(clean, rim_tol=tol)
        draw_traced(ax, traced, mask.shape)
        panel_label(ax, f"rim_tol = {tol:g}: {traced.n_verts} vertices")
        print(f"  rim_tol {tol}: {traced.n_verts} vertices, {int(traced.free.sum())} free")
    handles = [
        Line2D([], [], ls="", marker="o", color=VERTEX, ms=4, label="free vertex"),
        Line2D([], [], ls="", marker="s", color=CLAMPED, ms=3.5, label="clamped vertex"),
        Rectangle((0, 0), 1, 1, fc=CELL, ec="none", label="interior cell"),
        Rectangle((0, 0), 1, 1, fc=CELL_ALT, ec="none", label="cell touching the background"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.1))
    save(fig, "example-trace-rim-tolerance.svg")


FIGURES = {
    "energy-shape-index": energy_shape_index,
    "relaxation-solvers": relaxation_solvers,
    "cost-vertex-to-vertex": costs_vertex_to_vertex,
    "cost-topological-distance": costs_topological_distance,
    "rheology-shear-drive": rheology_drive,
    "image-mask-trace": images_and_masks,
    "clamped-patch-relaxation": clamped_patch,
    "plot-mesh": plotting_plot_mesh,
    "example-periodic-relaxation": example_periodic_relaxation,
    "example-bounded-relaxation": example_bounded_relaxation,
    "example-tension-inference": example_tension_inference,
    "example-bounded-elongation": example_bounded_elongation,
    "example-gradient-estimators": example_gradient_estimators,
    "example-many-epochs": example_many_epochs,
    "example-vertex-step-cap": example_vertex_step_cap,
    "example-t1-counts": example_t1_counts,
    "example-shear-compliance": example_shear_compliance,
    "example-trace-rim-tolerance": example_trace_rim_tolerance,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="draw only the figures whose name contains one of these words")
    parser.add_argument("--list", action="store_true", help="list the figures and stop")
    args = parser.parse_args()
    if args.list:
        print("\n".join(FIGURES))
        return
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    selected = [n for n in FIGURES if not args.names or any(word in n for word in args.names)]
    start = time.perf_counter()
    for name in selected:
        t0 = time.perf_counter()
        print(f"{name}")
        FIGURES[name]()
        print(f"  {time.perf_counter() - t0:.0f} s")
    print(f"{len(selected)} figure function(s) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
