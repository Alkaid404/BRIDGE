"""CPU data structures for Closure response selection."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ClosureSpan:
    """A non-overlapping response span; ``token_end`` is exclusive."""

    span_id: int
    token_start: int
    token_end: int
    token_indices: List[int]
    core_token_indices: List[int]
    char_start: Optional[int]
    char_end: Optional[int]
    span_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> int:
        return len(self.token_indices)


@dataclass
class SpanScore:
    """Unchanged BRIDGE utility aggregated over one Closure span."""

    span_id: int
    positive_utility: float
    negative_utility: float
    signed_utility: float
    normalized_utility: float
    selection_utility: float
    cost: int
    utility_per_token: float
    highest_token_index: Optional[int]
    highest_token_score: float
    token_scores: List[float]
    clipped_token_scores: List[float] = field(default_factory=list)
    clipped_token_count: int = 0
    structure_bonus: float = 0.0


@dataclass
class StructuredSelectionResult:
    """Static Closure selection and diagnostics, all resident on CPU."""

    selected_span_ids: List[int]
    selected_response_token_indices: List[int]
    selected_response_mask: List[bool]
    span_boundaries: List[Tuple[int, int]]
    span_scores: List[Dict[str, Any]]
    span_types: List[str]
    selection_budget: int
    actual_selected_tokens: int
    budget_utilization: float
    fragmentation_rate: float
    utility_retention: float
    fallback_reason: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
