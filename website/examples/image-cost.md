# Image cost

The image cost compares the junctions of a mesh with an image, so it needs no correspondence between
vertices. This example renders a generated periodic tissue and checks the properties that
`tests/test_dmic.py` checks on a stored mesh: the image integrates to 1, a translation of the mesh by
whole pixels rolls the image, the cost and its gradient vanish at a perfect fit, and the gradient
agrees with finite differences. It runs in `float64`, at 64 x 64 pixels and with a PSF width of 0.3,
the values of the test.

## Rendering

[`render_interface_image`][vertax.cost.render_interface_image]`(vertices, heTable, width, height, resolution, sigma_psf)`
draws every half-edge as a segment of uniform density, normalized by the total length, blurred by a
Gaussian of standard deviation `sigma_psf` in the units of the mesh. The image is computed exactly in
Fourier space and is periodic, like the mesh.

```python
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import enable_x64

from vertax import PbcMesh
from vertax.cost import make_cost_dmic, render_interface_image

RES = 64
SIGMA = 0.30

with enable_x64():
    side = math.sqrt(20)
    mesh = PbcMesh.from_random_seeds(nb_seeds=20, width=side, height=side, random_key=0)
    v, he, fa = mesh.vertices, mesh.edges, mesh.faces

    image = render_interface_image(v, he, side, side, RES, SIGMA)
    print("integral:", float(jnp.sum(image)) * (side / RES) ** 2)

    shift = RES // 8
    moved = render_interface_image(v.at[:, 0].add(side * shift / RES), he, side, side, RES, SIGMA)
    print("translation rolls the image:", bool(np.allclose(moved, np.roll(image, shift, axis=1), atol=1e-12)))

fig, ax = plt.subplots()
ax.imshow(np.asarray(image), origin="lower", extent=(0, side, 0, side))
ax.set_title("render_interface_image, sigma_psf = 0.3")
fig.savefig("interface_image.png")
plt.close(fig)
```

The image is indexed `[row, column] = [y, x]`, hence `origin="lower"` to draw it the right way up.

## The cost and its gradient

[`make_cost_dmic`][vertax.cost.make_cost_dmic]`(sigma_psf)` returns the cost
$\tfrac12 \lVert \hat I - I \rVert^2$ in the twelve-argument form of an outer cost, with the image
target last.

```python
with enable_x64():
    cost = make_cost_dmic(SIGMA)

    def f(vertices, target):
        return cost(vertices, he, fa, side, side, None, None, None, None, None, None, target)

    print("cost at a perfect fit:", float(f(v, image)))
    print("largest gradient component there:", float(jnp.max(jnp.abs(jax.grad(f)(v, image)))))

    displaced = v.at[:, :2].add(0.02)
    gradient = np.asarray(jax.grad(f)(displaced, image))
    eps = 1e-6
    for i, j in [(0, 0), (5, 1), (17, 0)]:
        central = (f(displaced.at[i, j].add(eps), image) - f(displaced.at[i, j].add(-eps), image)) / (2 * eps)
        print(f"vertex {i}, coordinate {j}: autodiff {gradient[i, j]:+.6e}, central difference {float(central):+.6e}")
```

## In an inverse problem

To use the image as the target of a bilevel optimizer, assign the cost and the image:

```python
from vertax import PbcBilevelOptimizer

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_outer = make_cost_dmic(SIGMA)
optimizer.image_target = image
print(optimizer.self_summary())
```

The optimizer passes `image_target` as the last argument of the cost. `cost_dmic` itself takes the PSF
width as a keyword argument that the optimizer cannot pass, which is why `make_cost_dmic` is used.
Other options of `make_cost_dmic` (a mask, edge weights, intensity weighting, pixel-footprint
averaging, no normalization) are described in its docstring.
