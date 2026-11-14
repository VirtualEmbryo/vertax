"""Write ``hessian_weighted_cost_reference.npz``: the Hessian-weighted cost as computed by the Figure 3 code.

The Figure 3 inference runs of the VertAX paper defined the Hessian-weighted cost C_W in their own
script, ``figure3_campaign.py`` (functions ``target_hessian`` and ``hessian_weighted_cost``), before it
entered the library as `vertax.cost.make_cost_hessian_weighted`. This script evaluates those two
functions on the two mesh pairs of this folder and stores the results, so that
``tests/test_cost_hessian_weighted.py`` can check that the library gives the same values without
depending on the paper's code.

Run it with the paper's script and the library it was run with on the path:

    python tests/fixtures/make_hessian_weighted_cost_reference.py <path to figure3_campaign.py>

Stored, for each instance ``i`` (1008: target topology; 1000: one T1 transition away):

- ``H_uniform_<i>``: the Hessian of E2 at the target, every tension one, divided by its spectral norm;
- ``cost_<i>`` and ``grad_<i>``: C_W at the initial mesh and its gradient in the vertex positions;
- for 1008 only, ``H_target_1008``, ``cost_target_tensions_1008`` and ``grad_target_tensions_1008``:
  the same with the true tensions in the Hessian.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import vertax
from vertax import PbcMesh

HERE = Path(__file__).resolve().parent
INSTANCES = (1008, 1000)


def main(campaign_path: str) -> None:
    """Evaluate the Figure 3 functions of ``campaign_path`` on the mesh pairs and write the reference file."""
    spec = importlib.util.spec_from_file_location("figure3_campaign", campaign_path)
    fig3 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fig3  # its dataclasses look their module up here
    spec.loader.exec_module(fig3)
    out = {}
    for i in INSTANCES:
        mesh = PbcMesh.load_mesh(str(HERE / f"mesh_{i}.npz"))
        target = PbcMesh.load_mesh(str(HERE / f"mesh_target_{i}.npz"))
        variants = ("uniform", "target") if i == 1008 else ("uniform",)
        for tensions in variants:
            h = fig3.target_hessian(target, tensions)
            cost = fig3.hessian_weighted_cost(h)

            def at(v: jnp.ndarray, cost=cost, mesh: PbcMesh = mesh, target: PbcMesh = target) -> jnp.ndarray:  # noqa: ANN001
                return cost(v, mesh.edges, mesh.faces, mesh.width, mesh.height, target.vertices, target.edges,
                            target.faces)

            value, grad = jax.value_and_grad(at)(jnp.asarray(mesh.vertices, dtype=jnp.float64))
            suffix = f"_{i}" if tensions == "uniform" else f"_target_tensions_{i}"
            out[("H_uniform" if tensions == "uniform" else "H_target") + f"_{i}"] = np.asarray(h)
            out["cost" + suffix] = np.asarray(value)
            out["grad" + suffix] = np.asarray(grad)
    np.savez_compressed(HERE / "hessian_weighted_cost_reference.npz", **out)
    print(f"written with the library at {Path(vertax.__file__).resolve().parent}")


if __name__ == "__main__":
    main(sys.argv[1])
