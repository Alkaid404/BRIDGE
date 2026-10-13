"交替条件选择（论文核心选择算法）。"

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from utils import get_logger

logger = get_logger("bridge.selection")


def eligible_budget(ratio: float, eligible_length: int, rounding: str = "round") -> int:
    """Integer cap over eligible positions; fixed BOS/EOS never consume it."""
    if rounding not in ("round", "ceil"):
        raise ValueError("rounding must be 'round' or 'ceil'")
    if eligible_length < 0 or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("Expected a nonnegative length and ratio in [0,1]")
    convert = math.ceil if rounding == "ceil" else round
    return max(0, min(eligible_length, int(convert(ratio * eligible_length))))


class AlternatingSelector:
    "交替条件选择。"

    def __init__(self, T_alt: int = 2, monotonic_check: bool = True):
        if T_alt < 0:
            raise ValueError(f"T_alt must be >= 0, got {T_alt}")
        self._T_alt = T_alt
        self._monotonic_check = monotonic_check
        logger.debug(f"AlternatingSelector: T_alt={T_alt}, monotonic_check={monotonic_check}")

    def select(self, gamma_hat: torch.Tensor, beta_hat: torch.Tensor,
               bp: int, br: int) -> Tuple[torch.Tensor, torch.Tensor]:
        "运行交替选择，返回 ``(S_p, S_r)``。"
        S_p, S_r, _ = self.alternating_select(gamma_hat, beta_hat, bp, br)
        return S_p, S_r

    def alternating_select(
        self, gamma_hat: torch.Tensor, beta_hat: torch.Tensor,
        bp: int, br: int, S_r_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """运行交替选择并跟踪历史。"""
        response_len = gamma_hat.shape[-1]
        prompt_len = beta_hat.shape[-1]
        bp = min(bp, prompt_len)
        br = min(br, response_len)
        if bp <= 0 or br <= 0:
            raise ValueError(f"Budgets must be positive (bp={bp}, br={br})")


        S_r = S_r_init if S_r_init is not None else self._select_initial_response(gamma_hat, br)
        S_p = self._compute_prompt_given_response(beta_hat, S_r, bp)

        utility_history: List[float] = []
        if self._monotonic_check:
            utility_history.append(self.compute_utility(gamma_hat, beta_hat, S_p, S_r))


        for t in range(1, self._T_alt + 1):
            S_p = self._compute_prompt_given_response(beta_hat, S_r, bp)
            S_r_new = self._compute_response_given_prompt(gamma_hat, beta_hat, S_p, br)
            if self._monotonic_check:
                new_util = self.compute_utility(gamma_hat, beta_hat, S_p, S_r_new)
                prev_util = utility_history[-1]
                if not (new_util - prev_util) > 1e-6:
                    logger.warning(f"Round {t}/{self._T_alt}: utility did not improve ({prev_util:.6f} -> {new_util:.6f})")
                utility_history.append(new_util)
            S_r = S_r_new

        history = {"T_alt": self._T_alt, "utility_history": utility_history}
        return S_p, S_r, history


    @staticmethod
    def _select_initial_response(gamma_hat: torch.Tensor, br: int) -> torch.Tensor:
        """``S_r^(0) = top-br(γ̂)``。"""
        _, indices = torch.topk(gamma_hat, br, dim=-1, sorted=True)
        return indices

    @staticmethod
    def _compute_prompt_given_response(beta_hat: torch.Tensor, S_r: torch.Tensor, bp: int) -> torch.Tensor:
        """``s(i | S_r) = Σ_{j∈S_r} β̂_{k,i,j}`` → top-bp prompt 索引。"""
        prompt_len = beta_hat.shape[2]
        S_r_exp = S_r.unsqueeze(-1).expand(-1, -1, prompt_len)
        beta_selected = torch.gather(beta_hat, 1, S_r_exp)
        prompt_scores = beta_selected.sum(dim=1)
        _, S_p = torch.topk(prompt_scores, bp, dim=-1, sorted=True)
        return S_p

    @staticmethod
    def _compute_response_given_prompt(gamma_hat: torch.Tensor, beta_hat: torch.Tensor,
                                       S_p: torch.Tensor, br: int) -> torch.Tensor:
        """``s(j | S_p) = γ̂_{k,j} + Σ_{i∈S_p} β̂_{k,i,j}`` → top-br response 索引。"""
        response_len = beta_hat.shape[1]
        S_p_exp = S_p.unsqueeze(1).expand(-1, response_len, -1)
        beta_selected = torch.gather(beta_hat, 2, S_p_exp)
        prompt_gain = beta_selected.sum(dim=2)
        response_scores = gamma_hat + prompt_gain
        _, S_r = torch.topk(response_scores, br, dim=-1, sorted=True)
        return S_r

    @staticmethod
    def compute_utility(gamma_hat: torch.Tensor, beta_hat: torch.Tensor,
                        S_p: torch.Tensor, S_r: torch.Tensor) -> float:
        "BRIDGE 代理效用 ``U = Σ_{j∈S_r} γ̂_{k,j} + Σ_{i∈S_p} Σ_{j∈S_r} β̂_{k,i,j}``。"
        gamma_selected = torch.gather(gamma_hat, 1, S_r)
        gamma_utility = gamma_selected.sum(dim=1)
        br = S_r.shape[1]
        prompt_len = beta_hat.shape[2]
        S_r_expanded = S_r.unsqueeze(-1).expand(-1, -1, prompt_len)
        beta_r = torch.gather(beta_hat, 1, S_r_expanded)
        S_p_expanded = S_p.unsqueeze(1).expand(-1, br, -1)
        beta_rp = torch.gather(beta_r, 2, S_p_expanded)
        beta_utility = beta_rp.sum(dim=[1, 2])
        return (gamma_utility + beta_utility).mean().item()

    @property
    def T_alt(self) -> int:
        return self._T_alt

    @property
    def monotonic_check(self) -> bool:
        return self._monotonic_check

    def __repr__(self) -> str:
        return f"AlternatingSelector(T_alt={self._T_alt}, monotonic_check={self._monotonic_check})"


def select_fixed(
    samples: List[Any],
    prompt_keep_ratio: float,
    response_keep_ratio: float,
    T_alt: int = 4,
    prompt_free_indices: Optional[List[Any]] = None,
    response_free_indices: Optional[List[Any]] = None,
    budget_rounding: str = "round",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    "逐样本按比例预算运行交替选择。"
    if prompt_free_indices is not None and len(prompt_free_indices) != len(samples):
        raise ValueError("prompt_free_indices must match samples length")
    if response_free_indices is not None and len(response_free_indices) != len(samples):
        raise ValueError("response_free_indices must match samples length")
    alternator = AlternatingSelector(T_alt=T_alt, monotonic_check=False)
    S_p_list: List[torch.Tensor] = []
    S_r_list: List[torch.Tensor] = []
    _NEG = -1.0e30
    for idx, rec in enumerate(tqdm(samples, desc="Token selection", unit="sample")):
        P, R = rec.prompt_length, rec.response_length
        free_p = set(int(v) for v in (prompt_free_indices[idx] if prompt_free_indices is not None else []))
        free_r = set(int(v) for v in (response_free_indices[idx] if response_free_indices is not None else []))
        for v in free_p:
            if not 0 <= v < P:
                raise ValueError(f"prompt free index {v} out of range [0,{P})")
        for v in free_r:
            if not 0 <= v < R:
                raise ValueError(f"response free index {v} out of range [0,{R})")
        sem_P = P - len(free_p)
        sem_R = R - len(free_r)
        bp = eligible_budget(prompt_keep_ratio, sem_P, budget_rounding)
        br = eligible_budget(response_keep_ratio, sem_R, budget_rounding)
        gamma = rec.gamma_hat.unsqueeze(0).clone()
        beta = rec.beta_hat.unsqueeze(0).clone()
        if free_r:
            gamma[0, list(free_r)] = _NEG
        if free_p:
            beta[0, :, list(free_p)] = _NEG
        if bp > 0 and br > 0:
            S_p, S_r = alternator.select(gamma, beta, bp, br)
            S_p = S_p.squeeze(0)
            S_r = S_r.squeeze(0)
        elif bp > 0:
            prompt_scores = beta[0].sum(dim=0)
            S_p = torch.topk(prompt_scores, bp).indices
            S_r = torch.empty(0, dtype=torch.long)
        elif br > 0:
            S_r = torch.topk(gamma[0], br).indices
            S_p = torch.empty(0, dtype=torch.long)
        else:
            S_p = torch.empty(0, dtype=torch.long)
            S_r = torch.empty(0, dtype=torch.long)
        if free_p:
            S_p = torch.cat([S_p, torch.tensor(sorted(free_p), dtype=torch.long)])
        if free_r:
            S_r = torch.cat([S_r, torch.tensor(sorted(free_r), dtype=torch.long)])
        S_p_list.append(S_p)
        S_r_list.append(S_r)
    mean_bp = sum(s.shape[0] for s in S_p_list) / max(len(S_p_list), 1)
    mean_br = sum(s.shape[0] for s in S_r_list) / max(len(S_r_list), 1)
    logger.info(
        f"select_fixed: r_p={prompt_keep_ratio} (mean_bp={mean_bp:.1f}), "
        f"r_r={response_keep_ratio} (mean_br={mean_br:.1f}), T_alt={T_alt}, "
        f"free_tokens={'on' if prompt_free_indices is not None or response_free_indices is not None else 'off'}"
    )
    return S_p_list, S_r_list
