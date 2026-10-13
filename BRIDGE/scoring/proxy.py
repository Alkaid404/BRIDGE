"交互代理 β̂ 与历史项 γ̂（论文评分的核心量）。"

from typing import Optional, Tuple

import torch

from utils import get_logger

logger = get_logger("bridge.scoring")


class InteractionProxy:
    "构造交互代理 β̂ 与历史项 γ̂。"

    def __init__(self, attention_type: str = "last_layer"):
        self._attention_type = attention_type
        logger.debug(f"InteractionProxy: attention_type={attention_type}")

    def compute_interaction_proxy(self, A_kji: torch.Tensor, a_kj: torch.Tensor) -> torch.Tensor:
        "``beta_hat_{k,i,j} = A_{k,j,i} · a_{k,j}``。"
        if A_kji.dim() != 3:
            raise ValueError(f"A_kji must be 3-D, got shape {A_kji.shape}")
        if a_kj.dim() != 2:
            raise ValueError(f"a_kj must be 2-D, got shape {a_kj.shape}")
        if A_kji.shape[0] != a_kj.shape[0] or A_kji.shape[1] != a_kj.shape[1]:
            raise ValueError(f"Batch/response length mismatch: {A_kji.shape} vs {a_kj.shape}")


        logger.debug(
            f"InteractionProxy device: A_kji={A_kji.device}, a_kj={a_kj.device}"
        )
        device = A_kji.device
        if a_kj.device != device:
            a_kj = a_kj.to(device)


        beta_hat = A_kji * a_kj.unsqueeze(-1)
        if torch.isnan(beta_hat).any():
            nan_count = torch.isnan(beta_hat).sum().item()
            logger.warning(f"Replacing {nan_count} NaN values in beta_hat with 0.0")
            beta_hat = torch.where(torch.isnan(beta_hat), torch.zeros_like(beta_hat), beta_hat)
        return beta_hat

    def compute_history_term(self, a_kj: torch.Tensor, c_kj: torch.Tensor) -> torch.Tensor:
        """``gamma_hat_{k,j} = a_{k,j} · c_{k,j}``。"""
        if a_kj.shape != c_kj.shape:
            raise ValueError(f"Shape mismatch: a_kj {a_kj.shape} vs c_kj {c_kj.shape}")

        device = a_kj.device
        if c_kj.device != device:
            c_kj = c_kj.to(device)
        gamma_hat = a_kj * c_kj
        if torch.isnan(gamma_hat).any():
            nan_count = torch.isnan(gamma_hat).sum().item()
            logger.warning(f"Replacing {nan_count} NaN values in gamma_hat with 0.0")
            gamma_hat = torch.where(torch.isnan(gamma_hat), torch.zeros_like(gamma_hat), gamma_hat)
        return gamma_hat

    def decompose_response_score(
        self, a_kj: torch.Tensor, c_kj: torch.Tensor, A_kji: torch.Tensor,
        S_p: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        "将响应评分分解为历史项与 prompt 中介增益。"
        gamma_hat = self.compute_history_term(a_kj, c_kj)
        beta_hat = self.compute_interaction_proxy(A_kji, a_kj)
        if S_p is None:
            prompt_mediated_gain = beta_hat.sum(dim=-1)
        else:
            S_p_mask = S_p.unsqueeze(1).to(device=beta_hat.device, dtype=beta_hat.dtype)
            prompt_mediated_gain = (beta_hat * S_p_mask).sum(dim=-1)
        return gamma_hat, prompt_mediated_gain

    def compute_full_response_scores(
        self, a_kj: torch.Tensor, c_kj: torch.Tensor, A_kji: torch.Tensor,
        S_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``s(j | S_p) = γ̂ + Σ β̂``（``decompose`` 的便捷合并）。"""
        gamma_hat, prompt_gain = self.decompose_response_score(a_kj, c_kj, A_kji, S_p)
        return gamma_hat + prompt_gain

    @property
    def attention_type(self) -> str:
        return self._attention_type
