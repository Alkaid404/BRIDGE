"""Fixed-budget alternating selection."""

import torch

from selection.alternating import AlternatingSelector, select_fixed


def compute_utility(
    gamma_hat: torch.Tensor,
    beta_hat: torch.Tensor,
    S_p: torch.Tensor,
    S_r: torch.Tensor,
) -> float:
    """BRIDGE 代理效用（``AlternatingSelector.compute_utility`` 的便捷别名）。"""
    return AlternatingSelector.compute_utility(gamma_hat, beta_hat, S_p, S_r)


__all__ = [
    "AlternatingSelector",
    "select_fixed",
    "compute_utility",
]
