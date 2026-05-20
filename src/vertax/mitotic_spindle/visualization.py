"""
Visualisation utilities for the tissue, spindle, and cell divisions.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vertax import BoundedMesh, plot_mesh, get_plot_mesh, EdgePlot, FacePlot, VertexPlot

from mesh_utils import get_face_polygon, polygon_centroid
from spindle_model import _ray_segment_length_jax

MIN_AREA = 0.01  # faces below this threshold are ignored in figures
ANG_MAX  = np.pi / 2 - 0.01  # max arc angle to avoid visual artefacts at boundaries


def _clip_angles(mesh: BoundedMesh) -> BoundedMesh:
    """Return a copy of the mesh with edge angles clipped for rendering."""
    m = BoundedMesh.copy_mesh(mesh)
    m.angles = jnp.clip(m.angles, 0.017, ANG_MAX)
    return m



def _add_cell_labels(ax, mesh):
    """Add each cell's index at its centroid; degenerate faces are silently skipped."""
    areas = np.array(mesh.get_area(jnp.arange(mesh.nb_faces)))
    for fid in range(mesh.nb_faces):
        if areas[fid] < MIN_AREA:
            continue
        try:
            poly = get_face_polygon(fid, mesh)
            c = polygon_centroid(poly)
            ax.text(c[0], c[1], str(fid), ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold', zorder=20)
        except Exception:
            continue

PLOT_KWARGS = dict(
    edge_plot=EdgePlot.EDGE_PARAMETER,
    edge_parameters_name="tension",
    face_plot=FacePlot.AREA,
    vertex_plot=VertexPlot.INVISIBLE,
)


def _overlay_spindle(ax, mesh, face_id, c, u, c1, c2, v0s, v1s, n_rays=32):
    poly = get_face_polygon(face_id, mesh)

    for centro in [c1, c2]:
        centro_j = jnp.array(centro, dtype=jnp.float64)
        for theta in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
            direction = jnp.array([np.cos(theta), np.sin(theta)], dtype=jnp.float64)
            lengths = jax.vmap(
                lambda v0, v1: _ray_segment_length_jax(centro_j, direction, v0, v1)
            )(v0s, v1s)
            L = float(jnp.min(lengths))
            if np.isfinite(L):
                contact = centro + L * np.array([np.cos(theta), np.sin(theta)])
                ax.plot([centro[0], contact[0]], [centro[1], contact[1]],
                        color='#aaaaaa', lw=0.6, alpha=0.6, zorder=6)

    projs = (poly - c) @ u
    half_span = float(np.max(np.abs(projs)))
    ax.annotate('', xy=c + half_span * u, xytext=c - half_span * u,
                arrowprops=dict(arrowstyle='<->', color='#1a7abf', lw=2.0), zorder=8)

    ax.scatter(*c, color='red', s=120, marker='+', linewidths=2.5, zorder=9)
    for ci, label in [(c1, 'C1'), (c2, 'C2')]:
        ax.scatter(*ci, color='#e63946', s=220, marker='o',
                   edgecolors='black', linewidths=1, zorder=10)
        ax.annotate(label, ci, fontsize=9, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', zorder=10)

    perp = np.array([-u[1], u[0]])
    projs_d = (poly - c) @ perp
    half_d = float(np.max(np.abs(projs_d))) * 1.1
    ax.plot([c[0] - half_d * perp[0], c[0] + half_d * perp[0]],
            [c[1] - half_d * perp[1], c[1] + half_d * perp[1]],
            color='#2dc653', lw=2.0, linestyle='--', zorder=8)

    ax.fill(poly[:, 0], poly[:, 1], color='#f4a261', alpha=0.25, zorder=3)


def _total_area(mesh):
    return float(jnp.sum(mesh.get_area(jnp.arange(mesh.nb_faces))))


def _plot_spindle_fig(mesh_pre: BoundedMesh, face_id: int,
                      c: np.ndarray, u: np.ndarray,
                      c1: np.ndarray, c2: np.ndarray,
                      v0s: jax.Array, v1s: jax.Array,
                      face_colors=None) -> None:
    ta = _total_area(mesh_pre)
    mesh_pre = _clip_angles(mesh_pre)
    fig, ax = get_plot_mesh(mesh_pre,
                            title=f"Cell {face_id} — spindle axis and division line — total area: {ta:.3f}",
                            **PLOT_KWARGS)
    _add_cell_labels(ax, mesh_pre)
    _overlay_spindle(ax, mesh_pre, face_id, c, u, c1, c2, v0s, v1s)
    plt.draw()
    plt.pause(0.001)


def _plot_post_div_fig(mesh_pre: BoundedMesh, mesh_post: BoundedMesh,
                       face_id: int, c: np.ndarray, u: np.ndarray,
                       division_idx: int, face_colors=None) -> None:
    ta = _total_area(mesh_post)
    mesh_post = _clip_angles(mesh_post)
    fig, ax = get_plot_mesh(mesh_post,
                            title=f"Division {division_idx} — just after (cell {face_id}) — total area: {ta:.3f}",
                            **{**PLOT_KWARGS, 'vertex_plot': VertexPlot.BLACK},
                            vertices_size=30)
    _add_cell_labels(ax, mesh_post)

    perp = np.array([-u[1], u[0]])
    poly_pre = get_face_polygon(face_id, mesh_pre)
    projs_d = (poly_pre - c) @ perp
    half_d = float(np.max(np.abs(projs_d))) * 1.1
    ax.plot([c[0] - half_d * perp[0], c[0] + half_d * perp[0]],
            [c[1] - half_d * perp[1], c[1] + half_d * perp[1]],
            color='#2dc653', lw=2.0, linestyle='--', zorder=7)
    plt.draw()
    plt.pause(0.001)


def _plot_opt_fig(mesh_opt: BoundedMesh, division_idx: int,
                  face_colors=None) -> None:
    verts = np.array(mesh_opt.vertices)
    if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
        print("[_plot_opt_fig] invalid vertices (NaN/Inf) — figure skipped.")
        return
    if np.any(verts < -1) or np.any(verts > mesh_opt.width + 1):
        print("[_plot_opt_fig] vertices out of bounds — figure skipped.")
        return

    ta = _total_area(mesh_opt)
    mesh_opt = _clip_angles(mesh_opt)
    fig, ax = get_plot_mesh(mesh_opt,
                            title=f"Division {division_idx} — after relaxation — total area: {ta:.3f}",
                            **PLOT_KWARGS)
    _add_cell_labels(ax, mesh_opt)
    plt.draw()
    plt.pause(0.001)


def _make_fig_with_colorbars(mesh: BoundedMesh, title: str = "",
                              area_vmax: float = 0.4):
    fig, ax = get_plot_mesh(mesh, title=title, **PLOT_KWARGS)
    return fig, ax
