# Practical guide to tension inference with VertAX

What to set, what to report, and what the paper and its Supplementary Information show about line-tension inference on periodic tissues.

The numbers below are those of the benchmark of Figure 3 of the paper: synthetic periodic tissues of 20 cells with 60 line tensions $\gamma$, area stiffness $K = 10$, five meshes, medians. They are measured values on that benchmark, not general laws. Section names refer to the Supplementary Information.

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
bop.loss_function_outer = cost_v2v                         # plain vertex matching; see section 5
bop.vertices_target     = target.vertices

bop.inner_solver = optax.sgd(learning_rate=0.01)          # inner mechanical relaxation
bop.outer_solver = optax.adam(learning_rate=1e-4, nesterov=True)
bop.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION

bop.max_nb_iterations = 1000       # the inner-solve iteration cap, "k" below -- see section 3
bop.tolerance = 1e-5
bop.patience = 5
bop.mean_tension_gauge = 1.0       # off by default; the paper's runs set it, see section 2

bop.do_n_bilevel_optimization(nb_epochs=500, mesh=mesh)
```

Two defaults worth knowing:

* **`mean_tension_gauge` (the mean-tension normalization constraint) defaults to `None`, i.e. off.** The paper's line-tension runs impose it (section 2). Shape-factor inference and bounded design runs leave it off.
* **`ep_variant` defaults to `EPVariant.CENTERED`**, the centered equilibrium-propagation estimator (nudges at $\pm\beta$) used in the paper. `ONE_SIDED` exists for comparison.

---

## 2. Fix the mean tension, and know that it is an assumption

**The overall tension level is nearly unobservable from geometry alone.** The softest singular direction of the force-balance matrix $\mathbf M$ has $\sigma\approx0.019$, two orders of magnitude below the largest and 25 times below the next one, and overlaps the uniform vector $\mathbf 1/\sqrt n$ with an inner product of 0.985 (section "Normalization and regularization of the inferred tensions"). Determining it as precisely as the best-conditioned direction would take of the order of $10^4$ times more data. In a periodic mesh the Hessian also has two exact zero eigenvalues, the rigid translations.

VertAX fixes the overall level by the **mean-tension normalization**, a post-step additive projection applied after every outer step for every gradient estimator:

```
gamma  <-  gamma - (mean(gamma) - 1)
```

Set `bop.mean_tension_gauge = 1.0` to enable it. It adds no term to the gradient, so it cannot be mis-scaled against a filtered hypergradient.

**It is a constraint, not a symmetry.** A uniform shift $c\mathbf 1$ of the tensions changes the force by $c\,\mathbf M\mathbf 1$, which is of the same order as the tension force itself ($\lVert\mathbf M\mathbf 1\rVert = 1.40$–$1.72$ against $\lVert\mathbf M\boldsymbol\gamma\rVert = 1.04$–$1.55$ on the benchmark meshes). The exact invariance of the problem is the joint rescaling $(K,\boldsymbol\gamma)\to a(K,\boldsymbol\gamma)$. Rescaling the tensions alone at fixed $K$ is equivalent to $K\to K/a$ and is not an invariance. In the benchmark the ratio of the two force scales is fixed by the data, so the multiplicative scale is pinned.

**The additive convention differs from classical force inference**, which normalizes the tensions by a common factor and lets the pressures carry a free offset. If you compare with results from that literature, state which convention you use.

**It is an assumption supplied from outside the data.** Fixing $\bar\gamma = 1$ injects the value of the least observable direction, known here because the benchmark was built with that mean. Fixing the mean is preferable to fixing a single reference edge, which would tie the whole solution to the noise of one measurement.

**The recovery error does not depend on the normalization if both tensions are centered.** The paper's recovery error $\varepsilon$ (section 4) subtracts the mean of the true tensions in its denominator, and the comparisons of Figure 4 center both the estimated and the reference tensions, so that $\varepsilon$ does not depend on how each method fixes the mean.

---

## 3. The inner-solve iteration cap $k$

Truncating the inner mechanical relaxation at $k$ steps acts as a **spectral (Landweber) regularization** of the differentiated map: a mode of stiffness $\lambda$ is amplified by $[1-(1-\eta\lambda)^k]/\lambda$ instead of $1/\lambda$, which caps the amplification of the soft modes at $t = k\eta$ (section "Truncation of the inner solve as spectral regularization"). The closed form is for gradient descent as the inner solver.

How much of that filter a run inherits depends on **the gradient estimator**. Ratio of the recovery error $\varepsilon$ at $k = 1000$ to its value at $k = 6$ (plain vertex-matching cost, five meshes, medians at epoch 350):

| estimator | $\varepsilon$ at $k=1000$ / $\varepsilon$ at $k=6$ | why |
|---|---|---|
| AD (unrolled) | **108** | differentiates the truncated solver and inherits the filter in full |
| Adj-ID | **13** | inherits it only through the state at which the adjoint system is evaluated |
| EP (paired difference) | **5.2** | both nudged solves are truncated from the same point, so the filter largely cancels |

EP is less sensitive to $k$ than the other estimators, but not insensitive: its error still changes 5.2-fold along the ladder. It is also the only estimator whose error is not monotone in $k$, with its minimum at $k = 20$.

Truncation also reduces the inner work: an epoch at $k = 6$ does 167 times less inner work than at $k = 1000$. For AD it brings the Pearson correlation from 0.7245 to 1.0000; for EP only to $\varepsilon = 0.10$–$0.13$ (Pearson 0.99).

**With the Hessian-weighted cost the best $k$ is not the same** (section 5): for AD and EP, $\varepsilon$ is 2.5 and 2.3 times larger at $k = 6$ than at $k = 1000$ (0.0487 against 0.0195, and 0.0493 against 0.0214), whereas Adj-ID is more accurate at $k = 6$ (0.0090 against 0.0185). Truncation and the Hessian-weighted cost both address the same ill-conditioning, so their effects do not add.

---

## 4. Report the recovery error $\varepsilon$ and the Pearson correlation

The recovery error is

```
epsilon  =  || gamma_hat - gamma_true ||_2  /  || gamma_true - mean(gamma_true) ||_2
```

It is 0 for exact recovery and 1 for an estimate no better than the mean true tension. Report the Pearson correlation between inferred and true tensions beside it: it ranks the tensions, whereas $\varepsilon$ also measures their magnitude. The two are not interchangeable: in the truncation ladder, EP at $k = 6$ has a Pearson correlation of 0.9918 at $\varepsilon = 0.1287$.

Report the residual force $\lVert\nabla_{\mathbf X}\mathcal E\rVert$ of the relaxed state with the recovery error. A short inner solve stops before equilibrium (end-of-epoch residual $7.8\times10^{-3}$ at $k = 6$ and $2.6\times10^{-4}$ at $k = 1000$, AD), so the recovery error alone does not tell how close to equilibrium the run was.

---

## 5. Choice of the outer cost, and what bounds the accuracy

**The Hessian-weighted cost $\mathcal C_W$ is an outer cost, a loss, like the plain vertex-matching cost $\mathcal C_{\text{v2v}}$.** With $\mathbf e = \mathbf X^* - \mathbf X_{\text{target}}$,

$$
\mathcal C_{\text{v2v}} = \tfrac12\lVert\mathbf e\rVert^2 \simeq \tfrac12\lVert\mathbf H^{+}\mathbf M\,\delta\boldsymbol\gamma\rVert^2,
\qquad
\mathcal C_W = \tfrac12\lVert\widehat{\mathbf H}\,\mathbf e\rVert^2 \simeq \tfrac12\lVert\mathbf M\,\delta\boldsymbol\gamma\rVert^2 ,
$$

where $\mathbf H$ is the Hessian of the energy, $\mathbf M$ the force-balance matrix and $\delta\boldsymbol\gamma$ the tension error. To first order $\mathcal C_W$ is the residual of the force-balance problem at the target geometry. $\widehat{\mathbf H}$ is the Hessian at the target with every tension set to one, scaled to unit norm, so it needs no knowledge of the tensions being inferred. In VertAX it is built by `make_cost_hessian_weighted` and passed as `loss_function_outer`.

On the benchmark, $\operatorname{cond}(\mathbf M) = 4.08$ against $\operatorname{cond}(\mathbf H^{+}\mathbf M) = 51.4$ (mean-tension direction excluded), a factor 12.6 that comes from the area stiffness, not from force balance. For shape-factor inference ($\mathcal E_1$) the factor is 1.27 and $\mathcal C_W$ is not needed.

**$\mathcal C_W$ belongs to a fixed topology.** $\widehat{\mathbf H}$ is assembled from a fixed edge set and grows as $1/\ell$ on an edge that shrinks towards a T1 transition, so $\mathcal C_W$ is unstable while the topology still has to change. The staged schedule uses $\mathcal C_{\text{v2v}}$ until the topology first matches that of the target, then disables T1 transitions and switches to $\mathcal C_W$ (`switch_cost_at_target_topology`).

**The accuracy reference.** For vertices observed with isotropic Gaussian error of variance $\sigma_X^2$, an unbiased estimator of the tensions has covariance at least $\sigma_X^2(J^\top J)^{-1}$, with $J$ the sensitivity of the equilibrium to the tensions in the mean-zero subspace (section "Cramér–Rao bound for a geometric observation model"). The paper calls it a reference and not a floor, because it bounds unbiased estimators only: estimators with a prior on the tensions, which are biased, fall below it at the higher noise levels. Inferring $K$ jointly inflates the reference by a factor 1.09–1.22 (median 1.16).

**Why tensions are harder than shape factors.** Shape factors are cell-level parameters and tensions are edge-level, so the geometry constrains the former with about 4 times the redundancy against about 1.33 times. The dimensionless stiffness ratio $\kappa \sim K(A^0)^{3/2}/\bar\gamma$ is of order 1 for $\mathcal E_1$ and about 10 for $\mathcal E_2$.

---

## 6. Two things that must be explicit

**Always pass `K_areas` explicitly and assert it.** The factories require it as a positional argument so that the value is visible at the call site rather than buried in the library. A guard test (`tests/test_api_contracts_guard.py`) enforces that there is no hidden default. The bare name `energy_line_tensions` no longer exists: it is produced by the factory `make_energy_line_tensions(K_areas)`, and code that calls it directly is stale.

**The topological distance is a measurement, not an objective.** $d_{\text{topo}}$ (`topological_distance`) counts the cell contacts that differ between two tissues, and $d_{\mathrm{T1}} = d_{\text{topo}}/2$ is a lower bound on the number of T1 transitions that separate them. It is an integer and is never optimized: it labels the inference problems and decides when the staged schedule switches cost.

---

## 7. Topology

When the initial and target tissues differ in topology, the staged schedule first has to reach the target topology.

* **The share of runs that reach the target topology decreases with $d_{\mathrm{T1}}$, the accuracy of the runs that recover does not.** By epoch 900, 87, 93, 87, 73 and 73 % of the runs reach the target topology at $d_{\mathrm{T1}} = 1$ to 5. Among the runs that recover, the median $\varepsilon$ is 0.013–0.018 at every $d_{\mathrm{T1}}$ (Spearman correlation 0.02 over 94 runs).
* **The estimators do not differ significantly in reaching the target topology by epoch 500** (AD 42, Adj-ID 36, EP 41 of 50 runs; paired exact tests, $p \ge 0.15$). By epoch 900 only the difference between Adj-ID and EP is significant ($p = 0.012$ uncorrected, 0.036 corrected for three comparisons).
* **The topology must be frozen after the switch.** With T1 transitions left free after the switch to $\mathcal C_W$, the schedule recovers 4 of 25 runs at $d_{\mathrm{T1}} > 0$; with the topology frozen, 14 of 25 (EP, 5 instances per $d_{\mathrm{T1}}$, epoch 500). Vertex matching alone never recovers the tensions on these runs (median $\varepsilon$ 0.66–1.09).
* **Failures after the switch exist.** With T1 transitions disabled, a shrinking edge cannot be relieved and may collapse; the relaxation then stops converging. Of the 150 staged runs of Figure 3, 30 reach the target topology without recovering the tensions.
* Single runs are not reproducible bit for bit across processors, so rates over instances are more reliable than the outcome of one run. Do not pool statistics across the frozen-topology $d_{\mathrm{T1}} = 0$ condition and topology-active runs: they are different conditions.

---

## Further reading

The derivations and the measurements behind this guide are in the Supplementary Information of the paper (sections on the conditioning of parameter inference, the truncation of the inner solve and the normalization of the inferred tensions) and in the documentation website.
