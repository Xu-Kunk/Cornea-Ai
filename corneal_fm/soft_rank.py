"""Vendored differentiable soft-rank / soft-Spearman (torchsort replacement).

torchsort 0.1.10 is installed in the venv, but its linux-aarch64 wheel is
unreliable on this machine: ``torchsort.soft_rank`` / ``torchsort.soft_sort``
return degenerate all-constant output (observed nondeterministically across
processes — the compiled ``isotonic_cpu`` extension loads yet misbehaves
against torch 2.13). Rather than depend on a broken binary, this module
vendors a pure-PyTorch logistic-kernel soft rank (exact autograd, O(n^2)):

    soft_rank(x)_i = sum_j sigmoid((x_i - x_j) / tau)

which converges to the competition rank as tau -> 0 and is smooth for
tau > 0 (ties contribute 0.5, i.e. average-rank-like behaviour). The soft
Spearman rho is the Pearson correlation of the two soft-rank vectors,
centred and normalised — a differentiable proxy of
``scipy.stats.spearmanr`` that matches it within ~0.01-0.05 on typical data
(validated in tests/test_method_arms.py).

Used by:

* ``corneal_fm/soft_spearman.py`` — the SS arm loss (differentiable rho
  between the axis projection and the ordinal label),
* ``corneal_fm/axis_ref.py`` — the 50-step projected gradient ascent that
  refines the frozen differentiation axis.

All functions are torch-only and dtype-agnostic (fp32/fp64). Cost is O(n^2)
per rho — negligible for loss batches (n <= ~64) and small for the one-off
axis-optimisation subsample (n <= 2048).

No file outside this module is modified; this is an additive replacement for
the broken ``torchsort`` dependency (see the ARM-2 task report).
"""

from __future__ import annotations

import torch


def soft_rank(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Differentiable soft rank via the logistic kernel.

    ``x`` is a 1-D tensor of shape ``(n,)``; returns a 1-D tensor of shape
    ``(n,)`` where entry ``i`` is ``sum_j sigmoid((x_i - x_j) / tau)`` — a
    smooth approximation of the competition rank (0..n). Gradient flows
    through ``x`` (never through ``tau``).
    """
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (n, n)
    return torch.sigmoid(diff / tau).sum(dim=1)


def soft_spearman(
    x: torch.Tensor, y: torch.Tensor, tau_scale: float = 0.1
) -> tuple[torch.Tensor, bool]:
    """Differentiable Spearman rho (centred, normalised soft ranks).

    Parameters
    ----------
    x : (n,) tensor — the differentiable argument (e.g. axis projections).
    y : (n,) tensor — the reference (e.g. ordinal labels; gradient not needed).
    tau_scale : float — soft-rank temperature as a fraction of each input's
        std (``tau = max(1e-4, tau_scale * std)``), so the approximation is
        scale-invariant. Smaller -> closer to exact ranks, spikier gradients.

    Returns
    -------
    (rho, valid) : rho is a scalar tensor in approx [-1, 1] (a
        grad-carrying ``x.sum() * 0.0`` when the rho is undefined, so the
        caller can safely add ``(1 - rho)`` to a loss without breaking
        ``Trainer.backward()``); ``valid`` is False when fewer than 2 samples
        are given or ``y`` has zero variance (no ordinal spread -> no rank
        information to preserve).
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    if n < 2:
        return x.sum() * 0.0, False
    y_std = y.std()
    if not bool(torch.isfinite(y_std)) or float(y_std.detach()) <= 0.0:
        return x.sum() * 0.0, False
    x_std = x.std()
    tau_x = max(1e-4, tau_scale * float(x_std.detach()))
    tau_y = max(1e-4, tau_scale * float(y_std.detach()))
    xr = soft_rank(x, tau_x)
    yr = soft_rank(y, tau_y)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denom = xc.norm() * yc.norm()
    rho = (xc * yc).sum() / (denom + 1e-8)
    return rho, True
