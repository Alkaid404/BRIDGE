"""Aggregate existing BRIDGE token utilities onto closure spans in O(L)."""

import math
from typing import Any, Dict, List, Optional, Sequence

import torch

from structure.types import ClosureSpan, SpanScore


def conditional_response_utility(record: Any, prompt_selection: torch.Tensor) -> torch.Tensor:
    """Return the exact response marginal used by the final alternating step."""
    gamma = record.gamma_hat.detach().cpu().float()
    prompt = torch.as_tensor(prompt_selection, dtype=torch.long).cpu()
    prompt = prompt[(prompt >= 0) & (prompt < int(record.prompt_length))]
    if prompt.numel() == 0:
        return gamma.clone()
    beta = record.beta_hat.detach().cpu().float()
    return gamma + beta.index_select(1, prompt).sum(dim=1)


def aggregate_span_scores(
    spans: Sequence[ClosureSpan],
    token_scores: torch.Tensor,
    reduction: str = "length_normalized_signed_sum",
    core_tokens_only: bool = True,
    length_alpha: float = 0.5,
    negative_penalty: float = 1.0,
    top_mean_fraction: float = 0.25,
    negative_clip: Optional[float] = None,
    span_type_bonuses: Optional[Dict[str, float]] = None,
) -> List[SpanScore]:
    """Aggregate BRIDGE scores with one pass over the span partition."""
    values = token_scores.detach().cpu().float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("token_scores contains NaN/Inf")
    if negative_clip is not None:
        negative_clip = float(negative_clip)
        if not math.isfinite(negative_clip) or negative_clip < 0:
            raise ValueError(
                "negative_clip must be finite, non-negative, or None"
            )
        clipped_values = values.clamp_min(-negative_clip)
    else:
        clipped_values = values
    results: List[SpanScore] = []
    epsilon = 1e-12
    for span in spans:
        selected = span.core_token_indices if core_tokens_only and span.core_token_indices else span.token_indices
        raw_local = (
            values[selected] if selected
            else torch.empty(0, dtype=torch.float32)
        )
        local = (
            clipped_values[selected] if selected
            else torch.empty(0, dtype=torch.float32)
        )
        positive = float(local.clamp_min(0).sum().item()) if local.numel() else 0.0
        negative = float((-local.clamp_max(0)).sum().item()) if local.numel() else 0.0
        signed = positive - float(negative_penalty) * negative
        normalized = signed / ((len(selected) + epsilon) ** float(length_alpha))
        if reduction == "positive_sum":
            utility = positive
        elif reduction == "signed_sum":
            utility = signed
        elif reduction == "top_mean":
            count = max(1, int(math.ceil(len(selected) * float(top_mean_fraction))))
            utility = float(torch.topk(local, min(count, local.numel())).values.mean().item()) if local.numel() else 0.0
        elif reduction == "length_normalized_signed_sum":
            utility = normalized
        else:
            raise ValueError(f"unknown span score reduction: {reduction}")
        span_types = span.metadata.get("span_types", [span.span_type])
        bonus = sum(
            float((span_type_bonuses or {}).get(str(span_type), 0.0))
            for span_type in sorted(set(span_types))
        )
        utility += bonus
        if local.numel():
            position = int(torch.argmax(local).item())
            highest_index = int(selected[position])
            highest_score = float(local[position].item())
        else:
            highest_index = None
            highest_score = 0.0
        cost = span.cost
        results.append(SpanScore(
            span_id=span.span_id,
            positive_utility=positive,
            negative_utility=negative,
            signed_utility=signed,
            normalized_utility=normalized,
            selection_utility=utility,
            cost=cost,
            utility_per_token=utility / max(cost, 1),
            highest_token_index=highest_index,
            highest_token_score=highest_score,
            token_scores=[float(values[index].item()) for index in span.token_indices],
            clipped_token_scores=[
                float(clipped_values[index].item())
                for index in span.token_indices
            ],
            clipped_token_count=int(
                sum(
                    float(clipped_values[index].item())
                    != float(values[index].item())
                    for index in span.token_indices
                )
            ),
            structure_bonus=bonus,
        ))
    return results
