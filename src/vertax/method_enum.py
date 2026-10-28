"""Enumerations declaration, separated here to avoid dependency cycles."""

from enum import Enum


class BilevelOptimizationMethod(Enum):
    """Which optimization method to use in the bi-level optimization."""

    AUTOMATIC_DIFFERENTIATION = "ad"
    """Unrolls the inner optimization steps and differentiates through them in reverse mode
    (`jax.grad`) ; memory and cost scale with the number of unrolled inner iterations, and
    line-search inner solvers are not supported."""
    EQUILIBRIUM_PROPAGATION = "ep"
    """Estimates the gradient from perturbed free vs nudged equilibria ; no backdrop required.
    Most efficient but depends on the perturbation size β."""
    IMPLICIT_DIFFERENTIATION = "id"
    """Differentiates the optimality condition ∇ₓE=0 via Implicit Function Theorem ; JVP variant ;
    requires Hessian solve and sensitive to ill-conditioning."""
    ADJOINT_STATE = "as"
    """Differentiates the optimality condition ∇ₓE=0 via Implicit Function Theorem ; VJP variant ;
    requires Hessian solve or sensitive to ill-conditioning."""


class EPVariant(Enum):
    """Which finite-difference form of the Equilibrium-Propagation estimator to use.

    ``beta`` is the *half-span*: perturbations are placed at ``+/-beta`` (CENTERED) so the
    total span is ``2*beta``, with **no** conversion factor applied anywhere.
    """

    CENTERED = "centered"
    """Solve at ``-beta`` and ``+beta``; gradient = (grad_nudged - grad_free) / (2*beta).
    The leading O(beta) bias cancels by symmetry, leaving O(beta**2) error. Default, and the
    variant used for all VertAX results."""
    ONE_SIDED = "one_sided"
    """Solve at ``0`` (true free equilibrium) and ``+beta``; gradient =
    (grad_nudged - grad_free) / beta. First-order accurate, O(beta) bias. Provided for
    ablation/comparison; not recommended for production inference."""
