"""Configuration and public-profile validation for Closure supervision."""

import math
from copy import deepcopy
from typing import Any, Dict


SUPPORTED_PROFILES = (
    "closure_only",
)

REMOVED_PROFILES = frozenset({
    "token",
    "bridge",
    "legacy_bridge",
    "closure_bridge",
    "bridge_reward",
    "graph_path",
    "path_search",
    "graph_completion",
    "closure_completion",
    "legacy_completion",
    "reserved_completion",
    "slack_completion",
    "candidate_conditioned_completion",
    "reasoning_completion",
    "guarded_swap",
    "legacy_swap",
    "completion_reserved_03",
    "completion_reserved_05",
    "completion_reserved_10",
    "completion_slack_strong",
    "completion_conditional_reserved_05",
})

REMOVED_CONFIG_KEYS = frozenset({
    "completion",
    "reserved_completion",
    "slack_completion",
    "path_completion",
    "structure_graph",
    "legacy_swap",
    "candidate_conditioned_reservation",
})

STRUCTURE_DEFAULTS: Dict[str, Any] = {
    "min_span_tokens": 4,
    "max_span_tokens": 96,
    "merge_short_spans": True,
    "fallback_to_token": True,
    "core_tokens_only": True,
    "exclude_delimiter_tokens": True,
    "score_reduction": "length_normalized_signed_sum",
    "length_alpha": 0.5,
    "negative_penalty": 1.0,
    "negative_clip": None,
    "top_mean_fraction": 0.25,
    "selection_strategy": "greedy_density",
    "require_exact_budget": False,
    "budget_fill_strategy": "dependency_then_score",
    "reasoning_sentence_bonus": 0.0,
    "quantity_unit_bonus": 0.0,
    "knapsack_tie_break": "position",
    "max_knapsack_capacity": 2048,
}

def validate_profile(profile: str) -> str:
    normalized = str(profile).strip().lower()
    if normalized in REMOVED_PROFILES:
        raise ValueError(
            f"Profile '{normalized}' has been removed. Supported profiles: "
            f"{', '.join(SUPPORTED_PROFILES)}."
        )
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unknown profile '{normalized}'. Supported profiles: "
            f"{', '.join(SUPPORTED_PROFILES)}."
        )
    return normalized


def normalize_structure_config(value: Any) -> Dict[str, Any]:
    if value is None:
        result = deepcopy(STRUCTURE_DEFAULTS)
    elif not isinstance(value, dict):
        raise TypeError(
            f"structure must be a mapping, got {type(value).__name__}"
        )
    else:
        unknown = sorted(set(value) - set(STRUCTURE_DEFAULTS))
        if unknown:
            raise ValueError(f"Unknown structure options: {unknown}")
        result = deepcopy(STRUCTURE_DEFAULTS)
        result.update(value)
    if result["score_reduction"] not in {
        "positive_sum",
        "signed_sum",
        "top_mean",
        "length_normalized_signed_sum",
    }:
        raise ValueError("Unsupported structure.score_reduction")
    if result["selection_strategy"] not in {
        "greedy_utility",
        "greedy_density",
        "budgeted_hybrid",
        "knapsack",
    }:
        raise ValueError("Unsupported structure.selection_strategy")
    if result["budget_fill_strategy"] != "dependency_then_score":
        raise ValueError(
            "structure.budget_fill_strategy must be "
            "'dependency_then_score'"
        )
    if result["knapsack_tie_break"] != "position":
        raise ValueError(
            "structure.knapsack_tie_break must be 'position'"
        )
    if not isinstance(result["require_exact_budget"], bool):
        raise TypeError(
            "structure.require_exact_budget must be a boolean"
        )
    if int(result["min_span_tokens"]) < 1:
        raise ValueError("structure.min_span_tokens must be >= 1")
    if int(result["max_span_tokens"]) < int(result["min_span_tokens"]):
        raise ValueError(
            "structure.max_span_tokens must be >= min_span_tokens"
        )
    for numeric_key in (
        "length_alpha",
        "negative_penalty",
        "reasoning_sentence_bonus",
        "quantity_unit_bonus",
    ):
        try:
            numeric_value = float(result[numeric_key])
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"structure.{numeric_key} must be a finite "
                "non-negative float"
            ) from error
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"structure.{numeric_key} must be a finite "
                "non-negative float"
            )
        result[numeric_key] = numeric_value
    negative_clip = result["negative_clip"]
    if negative_clip is not None:
        try:
            negative_clip = float(negative_clip)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "structure.negative_clip must be null or a non-negative float"
            ) from error
        if not math.isfinite(negative_clip) or negative_clip < 0:
            raise ValueError(
                "structure.negative_clip must be null or a finite "
                "non-negative float"
            )
        result["negative_clip"] = negative_clip
    if (
        isinstance(result["max_knapsack_capacity"], bool)
        or not isinstance(result["max_knapsack_capacity"], int)
    ):
        raise TypeError(
            "structure.max_knapsack_capacity must be an integer"
        )
    if int(result["max_knapsack_capacity"]) < 1:
        raise ValueError(
            "structure.max_knapsack_capacity must be >= 1"
        )
    if not 0 < float(result["top_mean_fraction"]) <= 1:
        raise ValueError(
            "structure.top_mean_fraction must be in (0, 1]"
        )
    return result
