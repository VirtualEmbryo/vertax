"""
Simulation and visualisation of a T1 transition in an epithelial tissue.

A T1 transition is a neighbour exchange: two cells lose the contact they shared
(short edge) and two other cells (their common neighbours) gain a new one.

Geometry:
  - Before : A (left) and B (right) share the short vertical edge.
             C (top) and D (bottom) touch only at a point.
  - During : the edge contracts → T1 event → new perpendicular edge.
  - After  : C (top) and D (bottom) share the new short horizontal edge.
             A and B are no longer direct neighbours.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

# ── Fixed vertices of the hexagonal border ──────────────────────────────────
VTL  = np.array([-2.0,  1.2])   # top-left
VTOP = np.array([ 0.0,  2.5])   # top-centre
VTR  = np.array([ 2.0,  1.2])   # top-right
VBR  = np.array([ 2.0, -1.2])   # bottom-right
VBOT = np.array([ 0.0, -2.5])   # bottom-centre
VBL  = np.array([-2.0, -1.2])   # bottom-left

HALF_EDGE = 0.5   # demi-longueur de l'arête courte initiale

COLORS = {
    'A': '#4477AA',   # blue   – left
    'B': '#EE6677',   # red    – right
    'C': '#228833',   # green  – top
    'D': '#CCBB44',   # yellow – bottom
}


def get_state(t: float):
    """
    Return cell polygons and short-edge endpoints at parameter t ∈ [0, 1].

    Phase 1 (t ≤ 0.5): vertical A-B edge contracts.
    Phase 2 (t > 0.5): horizontal C-D edge grows in its place.
    """
    t = float(np.clip(t, 0.0, 1.0))

    if t <= 0.5:
        s = 1.0 - 2.0 * t                       # 1 → 0 (contraction)
        V1 = np.array([ 0.0,  HALF_EDGE * s])   # top vertex of the edge
        V2 = np.array([ 0.0, -HALF_EDGE * s])   # bottom vertex

        cells = {
            'A': np.array([VTL, V1,   V2,  VBL]),           # left quadrilateral
            'B': np.array([VTR, VBR,  V2,  V1 ]),           # right quadrilateral
            'C': np.array([VTL, VTOP, VTR, V1 ]),           # top quadrilateral
            'D': np.array([VBL, V2,   VBR, VBOT]),          # bottom quadrilateral
        }
        short_edge = np.array([V1, V2])

    else:
        s = 2.0 * t - 1.0                       # 0 → 1 (extension)
        Vl = np.array([-HALF_EDGE * s,  0.0])   # left vertex of new edge
        Vr = np.array([ HALF_EDGE * s,  0.0])   # right vertex

        cells = {
            'A': np.array([VTL,  Vl,   VBL]),                  # left triangle
            'B': np.array([VTR,  VBR,  Vr ]),                  # right triangle
            'C': np.array([VTL,  VTOP, VTR, Vr, Vl]),          # top pentagon
            'D': np.array([VBL,  Vl,   Vr,  VBR, VBOT]),       # bottom pentagon
        }
        short_edge = np.array([Vl, Vr])

    return cells, short_edge


# ── Time sequence: hold → transition → hold ─────────────────────────────────
HOLD    = 30    # frozen frames at start and end
TRANSIT = 80    # frames for the transition
ts = [0.0] * HOLD + list(np.linspace(0.0, 1.0, TRANSIT)) + [1.0] * HOLD

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 7))
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-3.2, 3.2)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.set_facecolor('#f5f5f5')

ax.set_title('T1 transition — neighbour exchange\nin an epithelial tissue',
             fontsize=13, fontweight='bold', pad=12, linespacing=1.5)

phase_text = ax.text(0, -3.05, '', ha='center', va='center',
                     fontsize=10, style='italic', color='#444444')

# ── Patches cellulaires et étiquettes ───────────────────────────────────────
patches_dict = {}
labels_dict  = {}

for name, color in COLORS.items():
    patch = Polygon([[0, 0]], closed=True, facecolor=color,
                    edgecolor='white', linewidth=2.5, alpha=0.82, zorder=2)
    ax.add_patch(patch)
    patches_dict[name] = patch

    lbl = ax.text(0, 0, name, ha='center', va='center',
                  fontsize=18, fontweight='bold', color='white', zorder=4)
    lbl.set_path_effects([pe.withStroke(linewidth=3, foreground='#1a1a1a')])
    labels_dict[name] = lbl

# Short edge highlight
edge_line, = ax.plot([], [], color='#111111', lw=5, zorder=3,
                     solid_capstyle='round')

# Compact legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=c, ec='white', lw=1.5, alpha=0.85)
    for c in COLORS.values()
]
ax.legend(legend_elements, [f'Cell {n}' for n in COLORS],
          loc='upper right', fontsize=9, framealpha=0.7,
          handlelength=1.2, handleheight=1.0)


# ── Animation function ───────────────────────────────────────────────────────
def update(frame):
    t = ts[frame]
    cells, short_edge = get_state(t)

    for name in ['A', 'B', 'C', 'D']:
        patches_dict[name].set_xy(cells[name])
        cx, cy = cells[name].mean(axis=0)
        labels_dict[name].set_position((cx, cy))

    edge_line.set_data(short_edge[:, 0], short_edge[:, 1])

    if t < 0.45:
        msg = "Before T1  ·  A and B share a short edge"
    elif t < 0.55:
        msg = "T1 transition  ·  topological exchange"
    else:
        msg = "After T1  ·  C and D share a new edge"
    phase_text.set_text(msg)

    return list(patches_dict.values()) + list(labels_dict.values()) + [edge_line, phase_text]


anim = FuncAnimation(fig, update, frames=len(ts),
                     interval=40, blit=True, repeat=True)

plt.tight_layout()
plt.show()
