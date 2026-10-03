# Practical guide to tension inference with VertAX

What to set, what to report, and which conclusions from a bilevel run are trustworthy.

Everything below was established by measurement on periodic 20-cell meshes at `K_areas = 10` during
the 2026-08 analysis campaign. It is written for someone using the library, not for someone
reproducing the paper; the quantitative provenance is in the manuscript's Figure 3 and its
Supplementary Information. Where a number is quoted, it is the measured value on that benchmark set,
not a general law.

---

## 1. Setting up an inference run

```python
import optax
import jax.numpy as jnp
from vertax import PbcMesh, PbcBilevelOptimizer, cost_v2v, BilevelOptimizationMethod
from vertax.energy import make_energy_line_tensions

K_AREAS = 10.0                      # always explicit -- see section 6
N_CELLS = 20
WIDTH = HEIGHT = N_CELLS ** 0.5

mesh   = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=WIDTH, height=HEIGHT, random_key=0)
target = PbcMesh.from_random_seeds(nb_seeds=N_CELLS, width=WIDTH, height=HEIGHT, random_key=1)

# ---- the inner energy -------------------------------------------------------------------
# NOTE: `make_energy_line_tensions(K)` returns a function of
#   (vertTable, heTable, faceTable, width, height, he_params, face_params)   -- 7 arguments,
# but the optimizer calls `loss_function_inner` with
#   (vertices, edges, faces, vertices_params, edges_params, faces_params)    -- 6 arguments.
# So the factory's output CANNOT be assigned to `loss_function_inner` directly: it raises
# `TypeError: ... missing 1 required positional argument: 'face_params'`.
# Wrap it, closing over the box size. This is what the tests and examples do.
_E2 = make_energy_line_tensions(K_AREAS)

def energy_inner(vertTable, heTable, faceTable, vert_params, he_params, face_params):
    return _E2(vertTable, heTable, faceTable, WIDTH, HEIGHT, he_params, face_params)

# ---- the optimizer ----------------------------------------------------------------------
bop = PbcBilevelOptimizer()
bop.loss_function_inner = energy_inner
bop.loss_function_outer = cost_v2v
bop.vertices_target     = target.vertices

bop.inner_solver = optax.sgd(learning_rate=0.01)          # inner mechanical relaxation
bop.outer_solver = optax.adam(learning_rate=1e-4, nesterov=True)
bop.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION

bop.max_nb_iterations = 1000       # the inner-solve iteration cap, "k" below -- see section 3
bop.tolerance = 1e-5
bop.patience = 5
bop.mean_tension_gauge = 1.0       # NOT the default; see section 2 -- omitting it is a mistake

bop.do_n_bilevel_optimization(nb_epochs=500, mesh=mesh)
```

Two defaults worth knowing:

* **`mean_tension_gauge` (the mean-tension normalization constraint) defaults to `None`, i.e. off.** Leaving it off makes tension error
  meaningless (section 2).
* **`ep_variant` defaults to `EPVariant.CENTERED`**, which is correct. `ONE_SIDED` exists for
  comparison; one-sided EP gradients bias *both* accuracy metrics, not only the error norm.

---

## 2. Fix the mean tension, and know that it is an assumption

**The overall tension level is nearly unobservable** from geometry alone: the uniform direction is the
least observable one by a wide margin (it is near-null, not exactly null). In a periodic mesh the
mechanical Hessian additionally has an exact double-zero eigenvalue from the two global translational
symmetries. Without pinning the overall level, the recovered tension vector wanders along the
undetermined direction and the error norm is dominated by that drift — it inflated `err/spread` by
roughly 25x on the runs measured here.

VertAX fixes it with a **mean-tension normalization constraint**, a post-step additive projection, applied after every outer step for every
gradient estimator:

```
Lambda  <-  Lambda - (mean(Lambda) - 1)
```

Set `bop.mean_tension_gauge = 1.0` to enable it.

**The additive convention is deliberate, and it differs from the force-inference literature**, which
normalises *multiplicatively* (`Lambda -> a * Lambda`). Multiplicative rescaling is equivalent to
`K -> K/a`, so it trades against the area stiffness instead of being near-invariant. If you compare
against results from that literature, state which convention you are using — a reader will otherwise
assume the other one.

**This is an assumption supplied from outside the data, not bookkeeping.** Fixing `mean(Lambda) = 1`
injects the true value of one direction out of 60 — and it is the *worst-conditioned* direction, the
one that would need about 100x more data to determine than the best-conditioned one. It is
legitimate and standard, but it must be stated.

**On real data there is no ground-truth mean.** Fixing the mean to 1 is then not available, so any
error metric must be **centered**: subtract the mean from both the recovered and the reference
tensions before comparing. Reporting an uncentered error against real images silently credits the
method with an assumption it was not given.

---

## 3. The inner-solve iteration cap `k` buys efficiency, not accuracy

Truncating the inner mechanical solve acts as a **spectral (Landweber) regulariser**: it caps the
amplification of the soft modes of the equilibrium map. The practical consequence is worth stating
plainly, because the natural guess is wrong:

> **Truncation buys roughly 150x less inner work at matching accuracy. It does not buy accuracy.**

How much of that filter a run inherits depends on **which gradient estimator** is used, and the
ordering *is* the mechanism. Measured as the ratio of `err/spread` between the most- and
least-truncated ends of the `k` ladder:

| estimator | swing (max/min) | why |
|---|---|---|
| AD (unrolled) | **50.6x** | differentiates the truncated solver, inherits the filter in full |
| Adjoint-ID | **8.7x** | inherits it only through the state at which the adjoint is evaluated |
| EP (paired difference) | **5.4x** | differences two solves from the same point; the filter largely cancels |

EP's insensitivity is **relative, not immunity** — 5.4x is still an order-of-magnitude-scale effect.
EP is also the only one of the three that is **non-monotone in `k`**, with a minimum at `k = 20`:
it cannot inherit the filter's benefit but still pays truncation's first-order cost.

**Do not stack truncation with a Hessian-weighted cost** expecting the gains to add. On the primary
metric they interfere, and they hurt both estimators measured — the two corrections target the same
ill-conditioning.

---

## 4. Report `err/spread`, and check equilibrium first

**Primary metric.** Use the spread-normalised error

```
err/spread  =  || Lambda_hat - Lambda_target ||_2  /  || Lambda_target - mean(Lambda_target) ||_2
```

**The conventional mean-normalised relative error flatters by roughly twelvefold** and must not be
compared against percentages quoted for other problems. On one representative run the same result
reads `err/spread = 0.0213` but "relative error 0.178%"; for a weaker run, `0.6125` versus "5.30%".
Report Pearson correlation as a secondary metric — it saturates near 1.0 well before the error norm
becomes small, so it cannot substitute for `err/spread`.

**Check `||grad_x E||` before interpreting any recovery number.** A tension estimate attached to an
unconverged inner equilibrium is not a result. This is the single most common way to produce a
confident wrong number with this library.

---

## 5. There is a hard accuracy floor

For vertex noise of scale `sigma_eta`, the Cramér–Rao bound gives

```
std(Lambda_hat_i)  >=  sigma_eta / sigma_i(J_e)
```

where `sigma_i(J_e)` are the singular values of the map from tensions to equilibrium vertex
positions. Consequences that save wasted effort:

* **No reweighting of a purely geometric objective beats this bound.** For i.i.d. vertex noise,
  plain `cost_v2v` is already statistically efficient. Hessian weighting belongs in the optimization
  **step**, not in the loss.
* Accuracy improves only by lowering `sigma_eta`, raising `sigma_i(J_e)`, or accepting bias via a
  prior.
* A least-squares solve against the force-balance equations at the *known* target geometry recovers
  tensions essentially exactly (Pearson `1.000000`, `err/spread` about `6.2e-5`). **The information
  is present in the geometry.** Any shortfall three to four orders of magnitude above that floor
  comes from the estimation pipeline — a truncated inner solve and the outer optimiser — not from an
  ill-posed inverse problem.

**Why tensions are harder than shape factors.** Cell-level shape-factor inference enjoys about 4x
data redundancy against about 1.33x for edge-level tensions, is parameterised per cell rather than
per edge, and has a dimensionless stiffness ratio `K*sqrt(A_0)/mean(Lambda)` of order 1 rather than
10. Expect tension inference to be the harder problem by a wide margin.

---

## 6. Two things that must be explicit

**Always pass `K_areas` explicitly and assert it.** The factories require it as a positional
argument precisely so the value is visible at the call site rather than buried in the library. A
guard test (`tests/test_api_contracts_guard.py`) enforces that there is no hidden default. Note that
the bare name `energy_line_tensions` **no longer exists** — it is produced by the factory
`make_energy_line_tensions(K_areas)`. Any code or documentation calling `energy_line_tensions`
directly is stale.

**Do not use edge-to-edge or topological costs as tension estimators.** Any such cost is a strict
information loss relative to a vertex-level cost — the cell-adjacency Laplacian has a large null
space (18 dimensions on these meshes), and the estimated loss of Fisher information exceeds 96%.
`topological_distance` and relatives are useful as *metrics* and as basin-selection heuristics for guiding T1
transitions when only topological information is available. They are not estimators of tension.

---

## 7. Topology

When the initial and target meshes differ topologically, what degrades with topological distance is
the **probability of resolving the topology within the epoch budget** — not the accuracy achieved
once it is resolved, which is roughly flat in distance.

* **Crossing rate is indistinguishable across the three estimators** (21, 22 and 22 of 25 on the
  benchmark set) — a spread within binomial noise at that sample size. Do not choose an estimator to
  improve topological crossing.
* **Budget is usually the binding constraint.** Of runs that failed to converge after reaching the
  correct topology, about 92% are attributable to insufficient epochs. There is a usable design
  threshold of roughly **340 epochs of refinement after the correct topology is first reached**:
  below it, no run converged; above it, 15 of 17 did.
* Never pool statistics across the frozen-topology `d_topo = 0` condition and topology-active runs.
  They are different experimental conditions even though they can share a distance label.

---

## Related documents

* `docs/technical/CODE_CORRECTIONS_2026-08-09.md` (in the manuscript project, **not** in this
  repository) — the energy-convention corrections, why `K_areas` became explicit, and the removal of
  the former topology-cost family.
* The measured numbers behind this guide and the current limitations are kept with the manuscript
  analysis, which is not part of this repository snapshot.
