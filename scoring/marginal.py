"边际评分：token 选择的条件边际贡献。"

from typing import Optional

import torch

from scoring.proxy import InteractionProxy
from utils import get_logger

logger = get_logger("bridge.scoring")


class MarginalScorer:
    "计算 prompt / response 的条件边际评分。"

    def __init__(self, interaction_proxy: InteractionProxy):
        self._proxy = interaction_proxy
        logger.debug("MarginalScorer initialized")

    def compute_prompt_scores(self, beta_hat: torch.Tensor, S_r: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``s(i | S_r) = Σ_{j∈S_r} β̂_{k,i,j}`` → ``[batch, prompt_len]``。"""
        if beta_hat.dim() != 3:
            raise ValueError(f"beta_hat must be 3-D, got shape {beta_hat.shape}")
        if S_r is None:
            prompt_scores = beta_hat.sum(dim=1)
        else:
            S_r_mask = S_r.unsqueeze(-1).to(device=beta_hat.device, dtype=beta_hat.dtype)
            prompt_scores = (beta_hat * S_r_mask).sum(dim=1)
        if torch.isnan(prompt_scores).any():
            nan_count = torch.isnan(prompt_scores).sum().item()
            logger.warning(f"Replacing {nan_count} NaN prompt scores with 0.0")
            prompt_scores = torch.where(torch.isnan(prompt_scores), torch.zeros_like(prompt_scores), prompt_scores)
        return prompt_scores

    def compute_response_scores(
        self, gamma_hat: torch.Tensor, beta_hat: torch.Tensor,
        S_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``s(j | S_p) = γ̂_{k,j} + Σ_{i∈S_p} β̂_{k,i,j}`` → ``[batch, response_len]``。"""
        if gamma_hat.dim() != 2:
            raise ValueError(f"gamma_hat must be 2-D, got shape {gamma_hat.shape}")
        if beta_hat.dim() != 3:
            raise ValueError(f"beta_hat must be 3-D, got shape {beta_hat.shape}")

        device = beta_hat.device
        if gamma_hat.device != device:
            gamma_hat = gamma_hat.to(device)
        if S_p is None:
            prompt_gain = beta_hat.sum(dim=-1)
        else:
            S_p_mask = S_p.unsqueeze(1).to(device=device, dtype=beta_hat.dtype)
            prompt_gain = (beta_hat * S_p_mask).sum(dim=-1)
        response_scores = gamma_hat + prompt_gain
        if torch.isnan(response_scores).any():
            nan_count = torch.isnan(response_scores).sum().item()
            logger.warning(f"Replacing {nan_count} NaN response scores with 0.0")
            response_scores = torch.where(torch.isnan(response_scores), torch.zeros_like(response_scores), response_scores)
        return response_scores

    def compute_utility(
        self, gamma_hat: torch.Tensor, beta_hat: torch.Tensor,
        S_p: Optional[torch.Tensor] = None, S_r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        "代理效用 ``U(S_p, S_r) = Σ_{j∈S_r} s(j | S_p)`` → ``[batch]``。"
        response_scores = self.compute_response_scores(gamma_hat, beta_hat, S_p)
        if S_r is None:
            return response_scores.sum(dim=-1)
        return (response_scores * S_r.to(device=response_scores.device, dtype=response_scores.dtype)).sum(dim=-1)
