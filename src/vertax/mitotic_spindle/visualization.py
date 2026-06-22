"""
Visualisation utilities for the tissue, spindle, and cell divisions.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from vertax import BoundedMesh, plot_mesh, get_plot_mesh, EdgePlot, FacePlot, VertexPlot
from vertax.meshes.plot import _get_face_vertices_bounded, _draw_edge_bounded

from mesh_utils import get_face_polygon, polygon_centroid, get_cell_segments
from spindle_model import _ray_segment_length_jax

MIN_AREA = 0.01  # faces below this threshold are ignored in figures
ANG_MAX  = np.pi / 2 - 0.01  # max arc angle to avoid visual artefacts at boundaries


def _clip_angles(mesh: BoundedMesh) -> BoundedMesh:
    """Return a copy of the mesh with edge angles clipped for rendering."""
    m = BoundedMesh.copy_mesh(mesh)
    m.angles = jnp.clip(m.angles, 0.017, ANG_MAX)
    return m



def _add_cell_labels(ax, mesh, offset_face=None):
    """Add each cell's index at its centroid; degenerate faces are silently skipped.

    `offset_face`, if given, nudges that one cell's label up and to the side so it
    does not sit on top of the centroid marker drawn there (used in the zoom panel)."""
    areas = np.array(mesh.get_area(jnp.arange(mesh.nb_faces)))
    for fid in range(mesh.nb_faces):
        if areas[fid] < MIN_AREA:
            continue
        poly = get_face_polygon(fid, mesh)
        c = polygon_centroid(poly)
        if fid == offset_face:
            ax.annotate(str(fid), c, fontsize=8, color='white', fontweight='bold',
                        ha='center', va='center', zorder=20,
                        xytext=(26, 46), textcoords='offset points')
        else:
            ax.text(c[0], c[1], str(fid), ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold', zorder=20)

PLOT_KWARGS = dict(
    edge_plot=EdgePlot.EDGE_PARAMETER,
    edge_parameters_name="tension",
    face_plot=FacePlot.AREA,
    vertex_plot=VertexPlot.INVISIBLE,
)

# Common colour scales shared by every mesh plot in the notebook so colours are
# directly comparable across all figures.
#   - tension = sigmoid(he_params) + 1 is always in [1, 2] (see energy_adaptive),
#     so its scale is fixed.
#   - area scale spans the raw initial mesh (≈ 0.20–1.29); set by the notebook via
#     set_area_scale() so it matches the "Initial mesh and relaxation" plot.
TENSION_SCALE = (1.0, 2.0)
AREA_SCALE = (0.20, 1.30)


def set_area_scale(lo: float, hi: float) -> None:
    """Set the common 'Area of cell' colour scale used by all mesh plots."""
    global AREA_SCALE
    AREA_SCALE = (float(lo), float(hi))


def _overlay_spindle(ax, mesh, face_id, c, u, c1, c2, v0s, v1s, n_rays=32,
                     show_centroid=False, show_phi=False, legend=False):
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

    # Spindle axis: draw it exactly between where the axis line crosses the cell
    # boundary, so it never sticks out past the membrane. Cast a ray from the
    # spindle centre c along +u and -u and take the first boundary hit each way.
    c_j = jnp.asarray(c, dtype=jnp.float64)
    end_pos, end_neg = c + 0.0 * u, c - 0.0 * u
    for sgn, store in ((+1.0, 'pos'), (-1.0, 'neg')):
        d_ray = jnp.asarray(sgn * u, dtype=jnp.float64)
        lengths = jax.vmap(
            lambda v0, v1: _ray_segment_length_jax(c_j, d_ray, v0, v1)
        )(v0s, v1s)
        L = float(jnp.min(lengths))
        if not np.isfinite(L):                       # fallback: polygon projection
            L = float(np.max(np.abs((poly - c) @ u)))
        if store == 'pos':
            end_pos = c + L * u
        else:
            end_neg = c - L * u
    # Plain line (no arrow heads, which produced a small transverse tick near C2).
    ax.plot([end_neg[0], end_pos[0]], [end_neg[1], end_pos[1]],
            color='#1a7abf', lw=2.0, zorder=8,
            label='mitotic spindle' if legend else None)

    ax.scatter(*c, color='red', s=120, marker='+', linewidths=2.5, zorder=9,
               label='spindle centre' if legend else None)
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
            color='#2dc653', lw=2.0, linestyle='--', zorder=8,
            label='division line' if legend else None)

    ax.fill(poly[:, 0], poly[:, 1], color='#f4a261', alpha=0.25, zorder=3)

    # Geometric centroid of the cell (may differ from the spindle centre c,
    # which can drift from the centroid in the Newton method).
    if show_centroid:
        centroid = polygon_centroid(poly)
        ax.scatter(*centroid, color='#7b2cbf', s=110, marker='x', linewidths=2.5,
                   zorder=11, label='centroid' if legend else None)
        ax.annotate('centroid', centroid, fontsize=8, color='#7b2cbf',
                    ha='center', va='top', xytext=(0, -9),
                    textcoords='offset points', zorder=11)

    # Spindle-axis angle φ: show the value only (no +x reference line, no arc).
    # The spindle is an undirected axis, so φ and φ±180° are equivalent; fold φ
    # into [-90°, 90°]. Place the label off to the side, on a white background.
    if show_phi:
        phi = float(np.arctan2(u[1], u[0]))
        phi = (phi + np.pi / 2) % np.pi - np.pi / 2
        r = 0.35 * float(np.max(np.abs((poly - c) @ u)))   # offset scaled to the cell
        mid = 0.5 * phi
        label_pos = c + 2.2 * r * np.array([np.cos(mid), np.sin(mid)])
        ax.annotate(rf'$\varphi$ = {np.degrees(phi):.1f}°',
                    label_pos, fontsize=11, color='#1a7abf', fontweight='bold',
                    ha='center', va='center', zorder=12,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#1a7abf',
                              alpha=0.85, lw=0.8))


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


def _draw_mesh_on_ax(mesh, ax, face_scale, edge_scale, equal_aspect=True):
    """Draw a BoundedMesh (faces coloured by area, edges by tension) onto a given
    axis WITHOUT any colorbar.

    equal_aspect=True (default) forces `set_aspect('equal')` so the mesh isn't
    stretched — needed for subplot grids whose cells aren't square. Pass False
    when the axes rect is already square in inches (e.g. the manually-placed
    side-by-side panels), where equal-aspect would shrink the box.

    The high-level `_plot_faces_bounded` / `_plot_edges_bounded` helpers append a
    colorbar to the figure on every call; creating then deleting those perturbs
    constrained_layout (axes collapse to zero). Here we draw the faces and edges
    directly (a single shared colorbar pair is added once for the whole figure)."""
    faces_cmap = matplotlib.colormaps["cividis"]
    edges_cmap = matplotlib.colormaps["coolwarm"]
    f_min, f_max = face_scale
    e_min, e_max = edge_scale

    # Faces, coloured by cell area.
    areas = mesh.get_area(jnp.arange(mesh.nb_faces))
    for face in range(mesh.nb_faces):
        norm = 1.0 if f_max == f_min else (float(areas[face]) - f_min) / (f_max - f_min)
        poly = _get_face_vertices_bounded(mesh, face)
        ax.fill(poly[:, 0], poly[:, 1], color=faces_cmap(norm))

    # Edges, coloured by physical tension = sigmoid(he_params) + 1 ∈ [1, 2]
    # (same transform as energy_adaptive; he_params raw can be negative — do NOT
    # plot it directly or the colorbar shows misleading negative "tensions").
    values = jax.nn.sigmoid(mesh.edges_params) + 1.0
    for he in range(len(mesh.edges)):
        norm = 1.0 if e_max == e_min else (float(values[he]) - e_min) / (e_max - e_min)
        _draw_edge_bounded(mesh, ax, he, edges_cmap(norm), 2)

    ax.set_xlim(-0.5, mesh.width + 0.5)
    ax.set_ylim(-0.5, mesh.height + 0.5)
    if equal_aspect:
        ax.set_aspect("equal")


def _plot_spindle_pair(mesh_pre: BoundedMesh, face_id: int,
                       c: np.ndarray, u: np.ndarray,
                       c1: np.ndarray, c2: np.ndarray,
                       v0s: jax.Array, v1s: jax.Array,
                       zoom_margin: float = 0.4) -> None:
    """Two side-by-side panels for the same spindle:

    1. (left)  the classic full-tissue view with the spindle overlaid on the
       dividing cell;
    2. (right) a zoom on the dividing cell showing the two centrosomes (C1, C2),
       the geometric centroid, the spindle axis (mitotic spindle), and the
       angle φ.

    Both panels are drawn directly into one ``subplots(1, 2)`` figure (no offscreen
    image capture), sharing a single tension colorbar and a single area colorbar.
    """
    ta = _total_area(mesh_pre)
    mesh_clip = _clip_angles(mesh_pre)
    poly = get_face_polygon(face_id, mesh_pre)

    # Common notebook-wide colour scales (tension fixed to [1,2]; area = initial mesh).
    face_scale = AREA_SCALE
    edge_scale = TENSION_SCALE

    # Two equal-aspect panels + two shared colorbars. set_box_aspect(1) (below)
    # makes constrained_layout give each panel a SQUARE box that fills the cell
    # height, so there is no blank band above the plots and the data (a square
    # window) fits the box exactly without distortion.
    # Manual placement (no layout engine): two square panels filling the full
    # figure height, and two colorbars of the SAME height beside them. Equal
    # heights matter because Jupyter renders inline with bbox_inches='tight',
    # which would otherwise keep the taller artist's extent and leave a blank
    # band above the shorter one. figsize is 2 squares + colorbars wide.
    # Panel rects are made exactly SQUARE in inches (w_frac·figW = h_frac·figH)
    # and the mesh data window is square too, so the mesh fills the rect without
    # any aspect constraint — no shrunk box, hence no blank band (which `tight`
    # would otherwise keep, since the colorbars span the full height).
    fig_w, fig_h = 13.0, 5.0
    h_frac = 0.86
    w_frac = h_frac * fig_h / fig_w          # → square panel rect
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_full = fig.add_axes((0.030, 0.04, w_frac, h_frac))
    ax_zoom = fig.add_axes((0.030 + w_frac + 0.05, 0.04, w_frac, h_frac))
    # tension on the left, area of cell on the right.
    cax_tension = fig.add_axes((0.815, 0.04, 0.016, h_frac))
    cax_area = fig.add_axes((0.910, 0.04, 0.016, h_frac))

    # ── Panel 1 — classic full view ──────────────────────────────────────────
    _draw_mesh_on_ax(mesh_clip, ax_full, face_scale, edge_scale, equal_aspect=False)
    _add_cell_labels(ax_full, mesh_clip)
    _overlay_spindle(ax_full, mesh_clip, face_id, c, u, c1, c2, v0s, v1s)
    ax_full.set_title(f"Cell {face_id} — spindle axis and division line"
                      f" — total area: {ta:.3f}")

    # ── Panel 2 — zoom on the dividing cell ───────────────────────────────────
    _draw_mesh_on_ax(mesh_clip, ax_zoom, face_scale, edge_scale, equal_aspect=False)
    # Offset the dividing cell's index so it doesn't cover the centroid marker.
    _add_cell_labels(ax_zoom, mesh_clip, offset_face=face_id)
    _overlay_spindle(ax_zoom, mesh_clip, face_id, c, u, c1, c2, v0s, v1s,
                     show_centroid=True, show_phi=True, legend=True)
    # Square crop centred on the cell (aspect is ≈1, so a square data window keeps
    # the cell undistorted).
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * (1.0 + 2.0 * zoom_margin) * max(xmax - xmin, ymax - ymin)
    ax_zoom.set_xlim(cx - half, cx + half)
    ax_zoom.set_ylim(cy - half, cy + half)
    ax_zoom.set_title(f"Cell {face_id} — zoom: centrosomes, centroid, spindle, angle φ")
    ax_zoom.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # ── Single shared colorbars for the whole figure ──────────────────────────
    sm_tension = ScalarMappable(Normalize(*edge_scale),
                                matplotlib.colormaps["coolwarm"])
    sm_area = ScalarMappable(Normalize(*face_scale),
                             matplotlib.colormaps["cividis"])
    cb_tension = fig.colorbar(sm_tension, cax=cax_tension)
    cb_area = fig.colorbar(sm_area, cax=cax_area)
    # Labels rotated the other way (90° instead of the default 270°).
    cb_tension.set_label("tension", rotation=90, labelpad=10)
    cb_area.set_label("Area of cell", rotation=90, labelpad=10)

    plt.draw()
    plt.pause(0.001)


def _plot_div_relax_pair(mesh_post: BoundedMesh, mesh_opt: BoundedMesh,
                         division_idx: int = 1,
                         mesh_ref: BoundedMesh | None = None) -> None:
    """Two side-by-side panels: the mesh just after division (left) and after
    relaxation / inner optimization (right).

    Faces coloured by area, edges by tension, with a shared tension + area
    colorbar placed automatically by constrained_layout (no manual axes coords).
    Uses the notebook-wide common scales (AREA_SCALE, TENSION_SCALE) so colours
    match every other figure. `mesh_ref` is accepted for API compatibility.
    """
    m_post = _clip_angles(mesh_post)
    m_opt = _clip_angles(mesh_opt)
    face_scale, edge_scale = AREA_SCALE, TENSION_SCALE

    fig, (ax_div, ax_opt) = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")

    ta_post = _total_area(mesh_post)
    _draw_mesh_on_ax(m_post, ax_div, face_scale, edge_scale)
    _add_cell_labels(ax_div, m_post)
    ax_div.set_title(f"Division {division_idx} — just after "
                     f"({m_post.nb_faces} cells) — total area: {ta_post:.3f}")

    ta_opt = _total_area(mesh_opt)
    _draw_mesh_on_ax(m_opt, ax_opt, face_scale, edge_scale)
    _add_cell_labels(ax_opt, m_opt)
    ax_opt.set_title(f"Division {division_idx} — after relaxation "
                     f"({m_opt.nb_faces} cells) — total area: {ta_opt:.3f}")

    sm_tension = ScalarMappable(Normalize(*edge_scale), matplotlib.colormaps["coolwarm"])
    sm_area = ScalarMappable(Normalize(*face_scale), matplotlib.colormaps["cividis"])
    fig.colorbar(sm_tension, ax=(ax_div, ax_opt), label="tension")
    fig.colorbar(sm_area, ax=(ax_div, ax_opt), label="Area of cell")

    plt.show()


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
