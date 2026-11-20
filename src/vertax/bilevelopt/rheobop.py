r"""Bilevel optimizer for bulk-driven shear rheometry: design $\bm\theta$ against a compliance.

Implements the centered Equilibrium-Propagation estimator over the **two driven conditions**
$\pm a_0$, with an alternating (block-coordinate) nudged phase.

Why this lives at the optimizer layer
-------------------------------------
``PbcMesh`` and ``BoundedMesh`` are distinct classes because they carry distinct *data structures*.
The sheared problem carries the **same** structure as ``PbcMesh`` — same tables, same box, same
offsets, same T1 rules. What is new is that there are **two** meshes and a **protocol** relating them.
A ``ShearedPbcMesh`` would duplicate ``PbcMesh`` in full while leaving the genuinely new thing
homeless.

The estimator
-------------
With $\mathbf Z=(\mathbf X_+,\mathbf X_-)$,
$\mathcal E_{\rm tot}=\mathcal E_{+a_0}(\mathbf X_+)+\mathcal E_{-a_0}(\mathbf X_-)$,
$\widehat J=(U(\mathbf X_+)-U(\mathbf X_-))/(2a_0)$ and
$\widehat{\mathcal C}=(\widehat J/J_{\rm tgt}-1)^2$:

$$
\widehat{\nabla_{\bm\theta}\mathcal C}=\frac{1}{2\beta}\Big[
  \big(\partial_{\bm\theta}\mathcal E(\mathbf X_+^{+\beta})
      +\partial_{\bm\theta}\mathcal E(\mathbf X_-^{+\beta})\big)
 -\big(\partial_{\bm\theta}\mathcal E(\mathbf X_+^{-\beta})
      +\partial_{\bm\theta}\mathcal E(\mathbf X_-^{-\beta})\big)\Big]
$$

Four properties of the estimator that the implementation relies on:

1. **The outer cost carries no** $\bm\theta$. $U$ is a linear functional of the vertex
   positions alone, so $\partial_{\bm\theta}\widehat{\mathcal C}|_{\mathbf X}=0$ *exactly* and
   this collapses directly to the two bare parameter gradients. This is not
   a convenience: ``loss_function_outer`` is called without the parameters, and
   ``_compute_ep_grads`` differentiates $\mathcal E+\beta\mathcal C$, so a
   $\bm\theta$-dependent cost would lose its explicit term **silently**.
2. **The** $\beta$ **convention is fixed**: nudges at $\pm\beta$, prefactor
   $1/(2\beta)$, not $\beta/2$ with prefactor $1/\beta$.
3. **Every nudged solve is warm-started from its own free equilibrium, and the** $+\beta$
   **and** $-\beta$ **solves start from the same point.** That pairing is what makes EP's
   residual error track the inner budget rather than $\beta$ — measured to be
   $\beta$-independent across five decades.
4. **This is not two calls to** ``outer_eq_prop``. That routine performs its own free-and-nudged pair
   internally with a fixed partner and cannot express the alternation; the estimator above is
   evaluated directly from the four recorded states.

The solver is **injected**, not chosen here
-------------------------------------------
The library does not fix the configuration of the inner solver, which for this problem should be a
line-search conjugate gradient in double precision. ``solver_factory`` is a caller-supplied
zero-argument callable returning a fresh ``optax.GradientTransformation``. ``minimize`` is called
**directly** rather than through ``inner_opt`` because ``inner_opt`` does not forward
``cap_vertex_step``, and a line-search solver with T1 enabled needs it: it chooses its own step and
can otherwise move a vertex across whole cells in one iteration, breaking the incremental T1
detector.

Choosing $\beta$: it scales as $a_0^2$, and this is not optional
---------------------------------------------------------------------------
The cost is written on the compliance
$\widehat J=(U_+-U_-)/(2a_0)$, so its curvature in $U$ carries a factor
$1/(2a_0)^2$. At $a_0=7\times10^{-4}$ that is $5.1\times10^{5}$, while
the drive mode's own stiffness is $\sum\hat c^2/J\approx4.3$. The nudge therefore **out-stiffens
the tissue by four orders of magnitude at the conventional** $\beta=10^{-2}$: the nudged solve
drives $\widehat{\mathcal C}$ to zero and the state moves by exactly the distance to the target,
**independently of** $\beta$. Measured: $\delta U/U=-9.27\%$ at $\beta=3\times10^{-3}$
and $-9.52\%$ at $\beta=1$, a 333-fold $\beta$ range. That is a hard constraint, not
a nudge, and differencing two saturated states gives noise which $1/(2\beta)$ then amplifies —
the estimator came out $3\times10^{4}$ times the finite-difference reference and scaled as
$1/\beta$.

**Linear response requires** $\beta\,\dfrac{2}{(2a_0J_{\rm tgt})^2}\ll\dfrac{\sum\hat c^2}{J}$,
i.e. $\beta=O(a_0^2)$ — here $\beta\lesssim8\times10^{-5}$. Use
`ShearRheologyOptimizer.beta_upper_bound` to compute it, and **verify linearity by halving**
$\beta$ and checking that the response halves. A pilot $\beta=0.01$, as used in
line-tension inference, where the cost is not divided by a small amplitude, does not transfer.

Diagnostics are not optional
----------------------------
Every phase returns ``‖∇ₓE‖`` for every solve, T1 event counts, the connectivity hash of each nudged
state, ``β·Ĉ`` against the inner energy scale, and the relative state change. A compliance attached to
an unconverged equilibrium is not a compliance, and EP's paired cancellation is only valid when the
$\pm\beta$ solves cross the *same* T1 events — a connectivity mismatch invalidates that outer
step and is **recorded, not repaired**.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from vertax.opt import OptimizationTarget, minimize
from vertax.rheology import (
    ShearMode,
    compliance,
    connectivity_hash,
    grad_x_norm,
    make_compliance_cost,
    make_drive_energy,
    mode_amplitude,
    modulus_from_compliance,
    t1_event_count,
    topology_signature,
)
from vertax.topo import do_not_update_T1, update_T1

__all__ = ["ShearRheologyOptimizer", "ShearState"]


@dataclass
class ShearState:
    """One relaxed configuration plus the diagnostics that make it interpretable."""

    vertices: Array
    edges: Array
    faces: Array
    grad_norm: float
    amplitude: float
    """The mode amplitude $U$ of this state."""
    t1_events: int
    conn_hash: str
    seconds: float
    energy: float


@dataclass
class ShearRheologyOptimizer:
    r"""Design $\bm\theta$ so a periodic tissue attains a prescribed mode compliance.

    Args:
        energy_theta: the **undriven** vertex-model energy, in the six-argument inner-loss signature
            ``(vertTable, heTable, faceTable, vert_params, he_params, face_params)``. ``width`` and
            ``height`` are ``static_argnums`` throughout the library and must be closed over by the
            caller.
        mode: the drive mode. Re-reference it from the $a=0$ relaxed state at each outer epoch
            and treat it as a constant of the estimator *within* the epoch.
        a0: drive amplitude, chosen before a design run, small enough for the response to be linear.
        j_target: target compliance. **Derive it as** ``rheology.target_compliance(G_tgt, mode)``
            with ``G_tgt`` taken *on the measured forward curve*, so that
            $\widehat{\mathcal C}$ starts $O(1)$. Setting it to a bare constant makes
            $\beta\widehat{\mathcal C}$ comparable to the whole energy and the "nudge" is then a
            large perturbation
        beta: nudge magnitude. Code convention: solves at $\pm\beta$, prefactor
            $1/(2\beta)$.
        solver_factory: zero-argument callable returning a fresh optax solver for the inner solve.
        width: box width.
        height: box height.
        min_dist_T1: T1 threshold.
        free_iters: inner iteration budget of the free solves. See `nudged_iters`.
        nudged_iters: inner iteration budget of the nudged solves. **Measure these, do not assume
            them.** A warm-started driven solve does *not* converge quickly merely because it travels a short
            distance: CG's iteration count is set by the spectrum, and the drive excites the softest
            modes. Measured on the 72-cell production box, from a free state at
            $\|\nabla_xE\|\approx2\times10^{-13}$, the driven solve still needs ~5000
            iterations for $10^{-8}$ and ~20000 for $10^{-13}$.
        theta_target: which parameter array is designed.
        update_t1: T1 during the inner solves. **On** for production — it is the fluidity mechanism.
        cap_vertex_step: per-step vertex displacement cap. Keep **True** with a line-search solver
            and T1 on.
        n_sweeps: alternating half-sweep count in the nudged phase. Do **not** raise past 4: if the
            alternation is not contracting by then, block-coordinate is inadequate and a joint solver
            is needed — report rather than paper over it.
    """

    energy_theta: Callable[..., Array]
    mode: ShearMode
    a0: float
    j_target: float
    beta: float
    solver_factory: Callable[[], optax.GradientTransformation]
    width: float
    height: float
    min_dist_T1: float = 0.005
    free_iters: int = 20000
    nudged_iters: int = 5000
    theta_target: OptimizationTarget = OptimizationTarget.FACE_PARAMETERS
    update_t1: bool = True
    cap_vertex_step: bool = True
    n_sweeps: int = 2
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the conventions that cannot be recovered from a wrong run."""
        if self.beta <= 0:
            msg = f"beta must be > 0; got {self.beta}"
            raise ValueError(msg)
        if self.n_sweeps < 1 or self.n_sweeps > 4:
            msg = (f"n_sweeps must be in 1..4; got {self.n_sweeps}. Raising it past 4 hides a "
                   "non-contracting alternation instead of reporting it.")
            raise ValueError(msg)
        if self.theta_target not in (OptimizationTarget.FACE_PARAMETERS,
                                     OptimizationTarget.EDGE_PARAMETERS,
                                     OptimizationTarget.VERTEX_PARAMETERS):
            msg = f"theta_target must be a parameter array, got {self.theta_target}"
            raise ValueError(msg)

    # ------------------------------------------------------------------ inner
    def _relax(self, inner_loss, v, he, fa, params, iters):  # noqa: ANN001, ANN202
        """One inner relaxation of the vertices, through the production `minimize`."""
        import time

        vp, hp, fp = params
        t0 = time.perf_counter()
        (vv, hh, ff, _a, _b, _c), _hist = minimize(
            jnp.asarray(v, jnp.float64), he, fa, self.width, self.height,
            jnp.asarray(vp, jnp.float64), jnp.asarray(hp, jnp.float64),
            jnp.asarray(fp, jnp.float64),
            inner_loss, self.solver_factory(), self.min_dist_T1,
            iterations_max=iters, tolerance=0.0, patience=5,
            optimization_target=OptimizationTarget.VERTICES,
            update_t1_func=(update_T1 if self.update_t1 else do_not_update_T1),
            cap_vertex_step=self.cap_vertex_step,
        )
        return vv, hh, ff, time.perf_counter() - t0

    def _state(self, inner_loss, v, he, fa, params, ref_sig, secs) -> ShearState:  # noqa: ANN001
        """Package a relaxed configuration with its diagnostics."""
        vp, hp, fp = params
        return ShearState(
            vertices=v, edges=he, faces=fa,
            grad_norm=grad_x_norm(inner_loss, v, he, fa, vp, hp, fp),
            amplitude=float(mode_amplitude(v, self.mode)),
            t1_events=t1_event_count(ref_sig, topology_signature(he)),
            conn_hash=connectivity_hash(he),
            seconds=secs,
            energy=float(inner_loss(v, he, fa, vp, hp, fp)),
        )

    # ------------------------------------------------------------ free phase
    def free_phase(self, vertices, edges, faces, params):  # noqa: ANN001, ANN201
        r"""Relax the two conditions **independently** at $\beta=0$.

        Returns ``{+1: ShearState, -1: ShearState}``. These are the warm-start points for every
        nudged solve, and the pair from which $\widehat J$ is read.
        """
        ref_sig = topology_signature(edges)
        out = {}
        for sgn in (+1, -1):
            inner = make_drive_energy(self.energy_theta, self.mode, sgn * self.a0)
            v, he, fa, secs = self._relax(inner, vertices, edges, faces, params, self.free_iters)
            out[sgn] = self._state(inner, v, he, fa, params, ref_sig, secs)
        return out

    # ---------------------------------------------------------- nudged phase
    def nudged_phase(self, free, params, b):  # noqa: ANN001, ANN201
        r"""Alternating (block-coordinate) nudged solves at nudge level ``b``.

        Each half-sweep minimises one condition with the **partner's amplitude frozen**, which is what
        makes the joint nudged equilibrium expressible against a one-mesh inner solver. Both the
        $+\beta$ and $-\beta$ calls enter from the *same* free states, so their residuals
        are correlated and largely cancel in the difference.
        """
        cur = {sgn: (free[sgn].vertices, free[sgn].edges, free[sgn].faces) for sgn in (+1, -1)}
        amp = {sgn: free[sgn].amplitude for sgn in (+1, -1)}
        sweep_moves = []
        for _sweep in range(self.n_sweeps):
            moved = 0.0
            for sgn in (+1, -1):
                cost = make_compliance_cost(self.mode, self.j_target, self.a0,
                                            partner_amplitude=amp[-sgn], sign=sgn)
                driven = make_drive_energy(self.energy_theta, self.mode, sgn * self.a0)
                w, h_, f_ = self.width, self.height, None

                def nudged(vt, ht, ft, vp, hp, fp, _d=driven, _c=cost, _b=b):  # noqa: ANN001, ANN202
                    return _d(vt, ht, ft, vp, hp, fp) + _b * _c(
                        vt, ht, ft, w, h_, None, None, None, None, None, None, None)

                v_prev = cur[sgn][0]
                v, he, fa, secs = self._relax(nudged, *cur[sgn], params, self.nudged_iters)
                cur[sgn] = (v, he, fa)
                amp[sgn] = float(mode_amplitude(v, self.mode))
                moved = max(moved, float(jnp.linalg.norm(v - v_prev) / jnp.linalg.norm(v_prev)))
            sweep_moves.append(moved)

        out = {}
        for sgn in (+1, -1):
            cost = make_compliance_cost(self.mode, self.j_target, self.a0,
                                        partner_amplitude=amp[-sgn], sign=sgn)
            driven = make_drive_energy(self.energy_theta, self.mode, sgn * self.a0)
            w, h_ = self.width, self.height

            def nudged(vt, ht, ft, vp, hp, fp, _d=driven, _c=cost, _b=b):  # noqa: ANN001, ANN202
                return _d(vt, ht, ft, vp, hp, fp) + _b * _c(
                    vt, ht, ft, w, h_, None, None, None, None, None, None, None)

            ref_sig = topology_signature(free[sgn].edges)
            out[sgn] = self._state(nudged, *cur[sgn], params, ref_sig, 0.0)
        return out, sweep_moves

    # -------------------------------------------------------------- estimator
    def _theta_grad(self, state: ShearState, params, sgn: int) -> Array:  # noqa: ANN001
        r"""$\partial_{\bm\theta}\mathcal E$ at a recorded state.

        The drive carries no $\bm\theta$, so this is the bare vertex-model parameter gradient —
        but it is evaluated on the *driven* energy so that a future $\bm\theta$-dependent drive
        could not silently drop a term.
        """
        driven = make_drive_energy(self.energy_theta, self.mode, sgn * self.a0)
        argnum = {OptimizationTarget.VERTEX_PARAMETERS: 3,
                  OptimizationTarget.EDGE_PARAMETERS: 4,
                  OptimizationTarget.FACE_PARAMETERS: 5}[self.theta_target]
        vp, hp, fp = params

        def e_of_theta(theta):  # noqa: ANN001, ANN202
            args = [state.vertices, state.edges, state.faces, vp, hp, fp]
            args[argnum] = theta
            return driven(*args)

        return jax.grad(e_of_theta)(params[argnum - 3])

    def compliance_gradient(self, nudged_plus, nudged_minus, params):  # noqa: ANN001, ANN201
        r"""The centered EP estimate, **summed over both conditions**."""
        plus = sum(self._theta_grad(nudged_plus[s], params, s) for s in (+1, -1))
        minus = sum(self._theta_grad(nudged_minus[s], params, s) for s in (+1, -1))
        return (plus - minus) / (2.0 * self.beta)

    # ------------------------------------------------------------ diagnostics
    def beta_upper_bound(self, j_measured: float, ratio: float = 0.1) -> float:
        r"""Largest $\beta$ at which the nudge is still a perturbation, not a constraint.

        The nudge adds a stiffness $\beta\cdot2/(2a_0J_{\rm tgt})^2$ along the mode; the tissue's
        own mode stiffness is $\sum\hat c^2/J$. Linear response needs the first well below the
        second, so

        $$
        \beta \le \text{ratio}\cdot\frac{\sum\hat c^2}{J}\cdot\frac{(2a_0J_{\rm tgt})^2}{2}
        $$

        with ``ratio`` the tolerated stiffness fraction (0.1 by default). **This is a necessary
        condition, not a sufficient one** — confirm linearity empirically by halving $\beta$ and
        checking that the state response halves.
        """
        k_mode = self.mode.c_sq / j_measured
        return ratio * k_mode * (2.0 * self.a0 * self.j_target) ** 2 / 2.0

    def compliance_and_cost(self, states) -> tuple[float, float]:  # noqa: ANN001
        r"""$(\widehat J, \widehat{\mathcal C})$ from a pair of states."""
        j = float(compliance(states[+1].amplitude, states[-1].amplitude, self.a0))
        return j, float((j / self.j_target - 1.0) ** 2)

    def beta_diagnostics(self, free, nudged_plus) -> dict[str, float]:  # noqa: ANN001
        r"""The two numbers to log at every outer step, which check that the nudge is in the linear regime.

        ``rel_state_change`` must sit in $[10^{-4},10^{-2}]$ and ``beta_cost_over_energy``
        below $10^{-2}$. Outside those the "nudge" is either invisible or is a large
        perturbation, and in both cases the estimator is not measuring what it claims to.
        """
        _j, c = self.compliance_and_cost(free)
        e_scale = max(abs(free[+1].energy), 1e-300)
        rel = max(
            float(jnp.linalg.norm(nudged_plus[s].vertices - free[s].vertices)
                  / jnp.linalg.norm(free[s].vertices))
            for s in (+1, -1)
        )
        return {
            "cost_free": c,
            "beta_times_cost": self.beta * c,
            "beta_cost_over_energy": self.beta * c / e_scale,
            "rel_state_change": rel,
            "in_state_window": float(1e-4 <= rel <= 1e-2),
            "beta_scale_ok": float(self.beta * c / e_scale < 1e-2),
        }

    # ------------------------------------------------------------------ epoch
    def epoch(self, vertices, edges, faces, params, outer_solver, outer_state):  # noqa: ANN001, ANN201
        r"""One outer step: free phase, both nudged phases, the centered EP estimate, one optax update.

        Returns ``(new_params, new_outer_state, diagnostics)``. The mode is **not** re-referenced
        here — the caller does that between epochs, so that the reference is a constant of the
        estimator within the epoch.
        """
        free = self.free_phase(vertices, edges, faces, params)
        j, cost = self.compliance_and_cost(free)
        n_plus, moves_p = self.nudged_phase(free, params, +self.beta)
        n_minus, moves_m = self.nudged_phase(free, params, -self.beta)
        grad = self.compliance_gradient(n_plus, n_minus, params)
        bd = self.beta_diagnostics(free, n_plus)

        # EP validity: the +-beta solves must cross the SAME T1 events. This must be judged on the
        # relabelling-invariant face-pair signature, NOT on `conn_hash`: that hash covers the whole
        # half-edge table including the periodic offset columns, which `update_pbc` rewrites whenever
        # a vertex crosses the seam. Judging validity on it produces mismatches even on runs with T1
        # disabled entirely, where no topology change is possible.
        conn_ok = all(
            topology_signature(n_plus[s].edges) == topology_signature(n_minus[s].edges)
            for s in (+1, -1)
        )

        argnum = {OptimizationTarget.VERTEX_PARAMETERS: 0,
                  OptimizationTarget.EDGE_PARAMETERS: 1,
                  OptimizationTarget.FACE_PARAMETERS: 2}[self.theta_target]
        theta = params[argnum]
        updates, new_outer_state = outer_solver.update(grad, outer_state, theta)
        new_theta = optax.apply_updates(theta, updates)
        new_params = list(params)
        new_params[argnum] = new_theta

        diag = {
            "J": j, "G": float(modulus_from_compliance(j, self.mode)),
            "J_target": self.j_target,
            "G_target": float(modulus_from_compliance(self.j_target, self.mode)),
            "cost": cost,
            "grad_norm_theta": float(jnp.linalg.norm(grad)),
            "gradE_free_plus": free[+1].grad_norm, "gradE_free_minus": free[-1].grad_norm,
            "gradE_nudged_max": max(n_plus[s].grad_norm for s in (+1, -1)),
            "t1_free_plus": free[+1].t1_events, "t1_free_minus": free[-1].t1_events,
            "t1_nudged_plus": n_plus[+1].t1_events + n_plus[-1].t1_events,
            "t1_nudged_minus": n_minus[+1].t1_events + n_minus[-1].t1_events,
            "connectivity_match_pm": int(conn_ok),
            "ep_step_valid": int(conn_ok),
            "sweep_moves_plus": moves_p, "sweep_moves_minus": moves_m,
            "alternation_contracting": int(len(moves_p) < 2 or moves_p[-1] <= moves_p[0]),
            "theta_mean": float(jnp.mean(new_theta)),
            "secs": sum(free[s].seconds for s in (+1, -1)),
            **bd,
        }
        self.history.append(diag)
        return tuple(new_params), new_outer_state, diag
