# VertAX

VertAX is a Python library for vertex models of epithelial tissues, written with
[JAX](https://github.com/google/jax). A tissue is a two-dimensional polygonal mesh: cells are faces,
junctions between two cells are edges, and points where three or more cells meet are vertices. The
mechanical equilibrium of the tissue is a minimum of an energy that you choose.

Because the whole computation is written in JAX, VertAX can differentiate through it. It is meant
for forward simulation and for inverse problems: finding parameters, such as line tensions, target
areas or shape factors, whose equilibrium matches observed data or reaches a design target. VertAX
treats an inverse problem as a bilevel optimization:

$$
\theta^\ast = \arg\min_\theta \, \mathcal{C}\left(X^\ast_\theta, \theta\right)
\quad \text{subject to} \quad
X^\ast_\theta \in \arg\min_X \, \mathcal{E}(X, \theta),
$$

where $X$ is the configuration (the vertex positions), $\theta$ the parameters, $\mathcal{E}$ the
energy and $\mathcal{C}$ the cost to minimize.

![Bilevel optimization loop in VertAX](figures/concept.png)

VertAX has two kinds of meshes: periodic meshes, for a piece of bulk tissue without boundaries, and
bounded meshes, for a finite cluster of cells whose outline is made of circular arcs. It provides
energies, costs and four ways to estimate the gradient of the cost with respect to the parameters
(AD, Sen-ID, Adj-ID and EP), and it performs T1 transitions during the relaxations.

## Contents

- [Getting started](getting-started.md): installation, and three quick starts.
- [User guide](user-guide/meshes.md): meshes, energies, relaxation and solvers, T1 transitions,
  numerical precision, inverse problems, costs, rheology, meshes from images and masks.
- [Examples](examples/index.md): complete scripts, one per page, that run on generated data.
- [API reference](api/index.md): one page per module, generated from the docstrings.
- [Changelog](changelog.md) and [release notes](release-notes.md).

## Citing VertAX

If you use VertAX in your research, please cite the preprint
[arXiv:2604.06896](https://arxiv.org/abs/2604.06896):

```bibtex
@misc{pasqui2026vertaxdifferentiablevertexmodel,
      title={VertAX: a differentiable vertex model for learning epithelial tissue mechanics},
      author={Alessandro Pasqui and Jim Martin Catacora Ocana and Anshuman Sinha and Matthieu Perez and Fabrice Delbary and Giorgio Gosti and Mattia Miotto and Domenico Caudo and Maxence Ernoult and Hervé Turlier},
      year={2026},
      eprint={2604.06896},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.06896},
}
```

To cite the software itself, use its Zenodo record:
[doi:10.5281/zenodo.24414586](https://doi.org/10.5281/zenodo.24414586) for all versions, or
[doi:10.5281/zenodo.24414587](https://doi.org/10.5281/zenodo.24414587) for version 0.4.0.

## License

VertAX is free software: you can redistribute it and modify it under the terms of the GNU General
Public License, version 3 or (at your option) any later version (GPL-3.0-or-later). It is
distributed without any warranty. Copyright 2025-2026 the VertAX authors, who are listed in the
`AUTHORS` file of the repository.
