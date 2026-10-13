"""Static Closure projection from BRIDGE response tokens to complete spans."""

import re
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from structure.aggregation import (
    aggregate_span_scores,
    conditional_response_utility,
)
from structure.spans import (
    AlignmentError,
    build_closure_spans,
    build_typed_spans,
    is_core_token,
    token_indices_for_char_range,
    tokenize_with_verified_offsets,
)
from structure.types import ClosureSpan, SpanScore, StructuredSelectionResult


def valid_indices(values: Any, upper_bound: int) -> List[int]:
    tensor = torch.as_tensor(values, dtype=torch.long).view(-1).cpu()
    return sorted({
        int(value) for value in tensor.tolist()
        if 0 <= int(value) < upper_bound
    })


def fragmentation_rate(
    spans: Sequence[ClosureSpan], selected_tokens: Set[int],
) -> float:
    touched = 0
    fragmented = 0
    for span in spans:
        selected = selected_tokens.intersection(span.token_indices)
        if selected:
            touched += 1
            fragmented += int(len(selected) != span.cost)
    return fragmented / touched if touched else 0.0


def positive_utility(
    token_scores: torch.Tensor, indices: Iterable[int],
) -> float:
    chosen = list(indices)
    if not chosen:
        return 0.0
    return float(token_scores[chosen].clamp_min(0).sum().item())


def ranked_span_ids(
    scores: Sequence[SpanScore], strategy: str,
) -> List[int]:
    if strategy == "greedy_utility":
        key = lambda item: (
            item.selection_utility,
            item.utility_per_token,
            -item.span_id,
        )
    elif strategy == "greedy_density":
        key = lambda item: (
            item.utility_per_token,
            item.selection_utility,
            -item.span_id,
        )
    elif strategy == "budgeted_hybrid":
        positive = [max(item.selection_utility, 0.0) for item in scores]
        index = max(0, int(0.75 * max(len(positive) - 1, 0)))
        high = sorted(positive)[index] if positive else 0.0
        key = lambda item: (
            item.utility_per_token
            + (item.selection_utility >= high)
            * abs(item.selection_utility) / max(item.cost, 1),
            item.selection_utility,
            -item.span_id,
        )
    else:
        raise ValueError(f"unknown Closure selection strategy: {strategy}")
    return [
        item.span_id for item in sorted(scores, key=key, reverse=True)
    ]


def select_spans(
    spans: Sequence[ClosureSpan],
    scores: Sequence[SpanScore],
    budget: int,
    strategy: str,
    preselected: Optional[Sequence[int]] = None,
    max_knapsack_capacity: int = 2048,
) -> List[int]:
    """Greedy 0/1 complete-span packing, matching the existing Closure path."""
    if strategy == "knapsack":
        selected, _ = knapsack_select_spans(
            spans,
            scores,
            budget,
            preselected=preselected,
            max_capacity=max_knapsack_capacity,
        )
        return selected
    by_id = {span.span_id: span for span in spans}
    selected = list(dict.fromkeys(preselected or []))
    selected_set = set(selected)
    used = sum(by_id[span_id].cost for span_id in selected)
    if used > budget:
        raise ValueError(f"preselected spans exceed budget: {used} > {budget}")
    for span_id in ranked_span_ids(scores, strategy):
        if span_id in selected_set:
            continue
        cost = by_id[span_id].cost
        if used + cost <= budget:
            selected.append(span_id)
            selected_set.add(span_id)
            used += cost
    return sorted(selected)


def _state_is_better(
    candidate: Tuple[float, int, float, Tuple[int, ...], Tuple[Tuple[int, int, int], ...]],
    current: Optional[
        Tuple[float, int, float, Tuple[int, ...], Tuple[Tuple[int, int, int], ...]]
    ],
) -> bool:
    if current is None:
        return True
    epsilon = 1e-12
    for candidate_value, current_value in zip(
        candidate[:3], current[:3],
    ):
        if candidate_value > current_value + epsilon:
            return True
        if candidate_value < current_value - epsilon:
            return False
    return candidate[4] < current[4]


def knapsack_select_spans(
    spans: Sequence[ClosureSpan],
    scores: Sequence[SpanScore],
    budget: int,
    preselected: Optional[Sequence[int]] = None,
    max_capacity: int = 2048,
) -> Tuple[List[int], Optional[str]]:
    "Deterministic O(num_items * budget) 0/1 packing."
    by_id = {span.span_id: span for span in spans}
    score_by_id = {score.span_id: score for score in scores}
    fixed = sorted(set(int(value) for value in (preselected or [])))
    used = sum(by_id[span_id].cost for span_id in fixed)
    if used > budget:
        raise ValueError(f"preselected spans exceed budget: {used} > {budget}")
    capacity = budget - used
    if capacity > int(max_capacity):
        selected = select_spans(
            spans,
            scores,
            budget,
            "greedy_density",
            preselected=fixed,
        )
        return selected, (
            f"knapsack_capacity_exceeded:{capacity}>{int(max_capacity)}"
        )

    candidates = sorted(
        (
            span for span in spans
            if span.span_id not in set(fixed)
            and span.cost <= capacity
        ),
        key=lambda span: (
            span.token_start,
            span.token_end,
            span.span_id,
        ),
    )
    empty_state = (0.0, 0, 0.0, tuple(), tuple())
    states: Dict[
        int,
        Tuple[
            float, int, float, Tuple[int, ...],
            Tuple[Tuple[int, int, int], ...],
        ],
    ] = {0: empty_state}
    for span in candidates:
        score = score_by_id[span.span_id]
        priority = int(span.metadata.get("structure_priority", 0))
        signature = (
            int(span.token_start),
            int(span.token_end),
            int(span.span_id),
        )
        for current_cost, state in sorted(
            list(states.items()), reverse=True,
        ):
            new_cost = current_cost + span.cost
            if new_cost > capacity:
                continue
            candidate_state = (
                state[0] + float(score.selection_utility),
                state[1] + priority,
                state[2] + float(score.positive_utility),
                state[3] + (span.span_id,),
                state[4] + (signature,),
            )
            if _state_is_better(
                candidate_state, states.get(new_cost),
            ):
                states[new_cost] = candidate_state
    best_cost = 0
    best_state = empty_state
    for state_cost, state in sorted(states.items()):
        if _state_is_better(state, best_state):
            best_cost = state_cost
            best_state = state
        elif state == best_state and state_cost > best_cost:
            best_cost = state_cost
    del best_cost
    return sorted(fixed + list(best_state[3])), None


def span_token_set(
    spans: Sequence[ClosureSpan], span_ids: Sequence[int],
) -> Set[int]:
    by_id = {span.span_id: span for span in spans}
    return {
        token
        for span_id in span_ids
        for token in by_id[int(span_id)].token_indices
    }


def _identity_result(
    baseline: List[int],
    budget: int,
    response_length: int,
    reason: str,
    timings: Dict[str, float],
    spans: Sequence[ClosureSpan] = (),
    scores: Sequence[SpanScore] = (),
    diagnostics: Optional[Dict[str, Any]] = None,
) -> StructuredSelectionResult:
    baseline_set = set(baseline)
    return StructuredSelectionResult(
        selected_span_ids=[],
        selected_response_token_indices=baseline,
        selected_response_mask=[
            index in baseline_set for index in range(response_length)
        ],
        span_boundaries=[
            (span.token_start, span.token_end) for span in spans
        ],
        span_scores=[asdict(score) for score in scores],
        span_types=[span.span_type for span in spans],
        selection_budget=budget,
        actual_selected_tokens=budget,
        budget_utilization=1.0 if budget else 0.0,
        fragmentation_rate=fragmentation_rate(spans, baseline_set),
        utility_retention=1.0,
        fallback_reason=reason,
        timings=timings,
        diagnostics={
            **(diagnostics or {}),
            "identity_token_fallback": True,
        },
    )


def _closure_result(
    spans: Sequence[ClosureSpan],
    span_scores: Sequence[SpanScore],
    selected_span_ids: Sequence[int],
    token_scores: torch.Tensor,
    baseline: Sequence[int],
    budget: int,
    response_length: int,
    fallback_reason: Optional[str],
    timings: Dict[str, float],
    construction: Dict[str, Any],
    selected_token_indices: Optional[Sequence[int]] = None,
) -> StructuredSelectionResult:
    selected_indices = (
        sorted(set(int(value) for value in selected_token_indices))
        if selected_token_indices is not None
        else sorted(span_token_set(spans, selected_span_ids))
    )
    if len(selected_indices) > budget:
        raise RuntimeError(
            f"Closure selection leaked budget: {len(selected_indices)} > {budget}"
        )
    mask = [False] * response_length
    for index in selected_indices:
        mask[index] = True
    baseline_positive = positive_utility(token_scores, baseline)
    selected_positive = positive_utility(token_scores, selected_indices)
    if baseline_positive > 0:
        utility_retention = selected_positive / baseline_positive
    else:
        baseline_signed = (
            float(token_scores[list(baseline)].sum().item())
            if baseline else 0.0
        )
        selected_signed = (
            float(token_scores[selected_indices].sum().item())
            if selected_indices else 0.0
        )
        utility_retention = (
            1.0 if baseline_signed == selected_signed else 0.0
        )
    by_id = {span.span_id: span for span in spans}
    diagnostics = {
        **construction,
        "baseline_selected_tokens": len(baseline),
        "baseline_positive_utility": baseline_positive,
        "structured_positive_utility": selected_positive,
        "average_span_length": (
            sum(span.cost for span in spans) / len(spans)
            if spans else 0.0
        ),
        "average_selected_span_length": (
            sum(by_id[span_id].cost for span_id in selected_span_ids)
            / len(selected_span_ids)
            if selected_span_ids else 0.0
        ),
        "selected_span_count": len(selected_span_ids),
        "selected_fragmentation_rate": fragmentation_rate(
            spans, set(selected_indices)
        ),
        "span_token_indices": [
            list(span.token_indices) for span in spans
        ],
        "span_core_token_indices": [
            list(span.core_token_indices) for span in spans
        ],
        "span_char_ranges": [
            [span.char_start, span.char_end] for span in spans
        ],
        "span_metadata": [dict(span.metadata) for span in spans],
    }
    return StructuredSelectionResult(
        selected_span_ids=list(selected_span_ids),
        selected_response_token_indices=selected_indices,
        selected_response_mask=mask,
        span_boundaries=[
            (span.token_start, span.token_end) for span in spans
        ],
        span_scores=[asdict(score) for score in span_scores],
        span_types=[span.span_type for span in spans],
        selection_budget=budget,
        actual_selected_tokens=len(selected_indices),
        budget_utilization=(
            len(selected_indices) / budget if budget else 0.0
        ),
        fragmentation_rate=fragmentation_rate(spans, set(baseline)),
        utility_retention=utility_retention,
        fallback_reason=fallback_reason,
        timings=timings,
        diagnostics=diagnostics,
    )


def _required_span_ids(
    spans: Sequence[ClosureSpan],
    response_text: str,
    protection: Dict[str, Any],
) -> Tuple[List[int], Dict[str, int]]:
    "v3 数学结构保护：按优先级返回 required span id 列表（boxed > final_answer >"
    if not protection:
        return [], {}
    final_answer_re = re.compile(
        r"\\boxed\{|####|final answer|the answer is|therefore|thus", re.IGNORECASE,
    )
    boxed_re = re.compile(r"\\boxed\{")
    latex_re = re.compile(r"\\frac|\\sqrt|\$[^$]+\$|\\begin\{|\\sum|\\int|\\pi|\\alpha|\\beta")
    required: List[int] = []
    stats = {"boxed": 0, "final_answer": 0, "equation": 0, "balanced_latex": 0,
             "sentence": 0, "symbol_definition": 0}

    def span_text(span: ClosureSpan) -> str:
        if span.char_start is None or span.char_end is None:
            return ""
        return response_text[span.char_start:span.char_end]

    def add(span_id: int, kind: str):
        if span_id not in required:
            required.append(span_id)
            stats[kind] += 1


    if protection.get("protect_boxed_answer"):
        for span in spans:
            if boxed_re.search(span_text(span)):
                add(span.span_id, "boxed")

    if protection.get("protect_final_answer"):
        last = None
        for span in spans:
            if final_answer_re.search(span_text(span)):
                last = span.span_id
        if last is not None:
            add(last, "final_answer")

    if protection.get("protect_equation_spans"):
        for span in spans:
            if span.span_type in ("latex_environment", "formula"):
                add(span.span_id, "equation")

    if protection.get("protect_balanced_latex"):
        for span in spans:
            if latex_re.search(span_text(span)):
                add(span.span_id, "balanced_latex")

    if protection.get("protect_reasoning_sentence"):
        for span in spans:
            if span.span_type == "sentence":
                add(span.span_id, "sentence")
    if protection.get("protect_symbol_definition"):
        for span in spans:
            if re.search(r"\b(let|denote|define|suppose)\b", span_text(span), re.IGNORECASE):
                add(span.span_id, "symbol_definition")
    return required, stats


_STRUCTURE_PRIORITY = {
    "boxed_answer": 100,
    "final_answer": 90,
    "equation": 60,
    "balanced_latex": 55,
    "quantity_unit_relation": 50,
    "reasoning_sentence": 40,
    "symbol_definition": 30,
}


def _span_from_indices(
    indices: Sequence[int],
    aligned: Any,
    span_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[ClosureSpan]:
    unique = sorted(set(int(index) for index in indices))
    if not unique:
        return None
    char_offsets = [
        aligned.offsets[index]
        for index in unique
        if aligned.offsets[index][1] > aligned.offsets[index][0]
    ]
    if not char_offsets:
        return None
    core = [
        index for index in unique
        if is_core_token(
            aligned.token_texts[index],
            aligned.offsets[index],
            True,
        )
    ]
    return ClosureSpan(
        span_id=-1,
        token_start=min(unique),
        token_end=max(unique) + 1,
        token_indices=unique,
        core_token_indices=core or unique,
        char_start=min(start for start, _ in char_offsets),
        char_end=max(end for _, end in char_offsets),
        span_type=span_type,
        confidence=1.0,
        metadata=dict(metadata or {}),
    )


def _boxed_char_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    marker = re.compile(r"\\boxed\s*\{")
    for match in marker.finditer(text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{" and not (
                cursor > 0 and text[cursor - 1] == "\\"
            ):
                depth += 1
            elif text[cursor] == "}" and not (
                cursor > 0 and text[cursor - 1] == "\\"
            ):
                depth -= 1
            cursor += 1
        ranges.append((match.start(), cursor))
    return ranges


def _latex_char_ranges(text: str) -> List[Tuple[int, int]]:
    """Return compact verified LaTeX regions without pairing currency signs."""
    ranges: List[Tuple[int, int]] = []
    for pattern in (
        re.compile(r"\$\$[^$]*\$\$", re.DOTALL),
        re.compile(r"\\\([^\n]*?\\\)"),
        re.compile(r"\\\[[\s\S]*?\\\]"),
        re.compile(
            r"\\begin\{([^{}]+)\}[\s\S]*?\\end\{\1\}"
        ),
    ):
        ranges.extend(
            (match.start(), match.end())
            for match in pattern.finditer(text)
        )
    for match in re.finditer(r"\$([^$\n]{1,160})\$", text):
        content = match.group(1)
        word_count = len(re.findall(r"[A-Za-z]{2,}", content))
        if (
            "\\" in content
            or re.search(r"[=+\-*/^<>]", content)
            or word_count <= 2
        ):
            ranges.append((match.start(), match.end()))
    for command in re.finditer(
        r"\\(?:frac|sqrt|boxed)\s*\{", text,
    ):
        depth = 1
        cursor = command.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{" and text[cursor - 1] != "\\":
                depth += 1
            elif text[cursor] == "}" and text[cursor - 1] != "\\":
                depth -= 1
            cursor += 1
        ranges.append((command.start(), cursor))
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(set(ranges)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _build_protection_spans(
    text: str,
    aligned: Any,
    base_spans: Sequence[ClosureSpan],
    protection: Dict[str, Any],
) -> List[ClosureSpan]:
    result: List[ClosureSpan] = []

    def add_indices(
        indices: Sequence[int],
        span_type: str,
        mandatory: bool = False,
    ) -> None:
        span = _span_from_indices(
            indices,
            aligned,
            span_type,
            {
                "mandatory": mandatory,
                "protected": True,
                "structure_priority": _STRUCTURE_PRIORITY[span_type],
            },
        )
        if span is not None:
            result.append(span)

    if protection.get("protect_boxed_answer"):
        for start, end in _boxed_char_ranges(text):
            add_indices(
                token_indices_for_char_range(aligned, start, end),
                "boxed_answer",
                mandatory=True,
            )

    if protection.get("protect_final_answer"):
        final_pattern = re.compile(
            r"####|final\s+answer|the\s+answer\s+is|answer\s*:",
            re.IGNORECASE,
        )
        matches = list(final_pattern.finditer(text))
        if matches:
            final_match = matches[-1]
            boundary = re.search(
                r"[\n.!?。！？]", text[final_match.end():],
            )
            char_end = (
                final_match.end() + boundary.end()
                if boundary is not None else len(text)
            )
            add_indices(
                token_indices_for_char_range(
                    aligned, final_match.start(), char_end,
                ),
                "final_answer",
                mandatory=True,
            )

    if protection.get("protect_equation_spans"):
        for start, end in _latex_char_ranges(text):
            fragment = text[start:end]
            if re.search(r"[=<>+\-*/^]|\\(?:frac|sqrt|sum|int)", fragment):
                add_indices(
                    token_indices_for_char_range(
                        aligned, start, end,
                    ),
                    "equation",
                )
        atom = (
            r"(?:-?\d+(?:\.\d+)?[A-Za-z]?|"
            r"[A-Za-z](?:_[A-Za-z0-9]+)?(?:\^\d+)?|"
            r"\([^.;,\n]{1,24}\))"
        )
        expression = (
            rf"{atom}(?:\s*[+\-*/^]\s*{atom}){{0,8}}"
        )
        plain_equation = re.compile(
            rf"(?<!\w){expression}\s*(?:=|<=|>=|<|>)\s*"
            rf"{expression}(?!\w)"
        )
        for match in plain_equation.finditer(text):
            add_indices(
                token_indices_for_char_range(
                    aligned, match.start(), match.end(),
                ),
                "equation",
            )

    if protection.get("protect_balanced_latex"):
        for start, end in _latex_char_ranges(text):
            add_indices(
                token_indices_for_char_range(
                    aligned, start, end,
                ),
                "balanced_latex",
            )

    if protection.get("protect_symbol_definition"):
        symbol_pattern = re.compile(
            r"\b(?:let|denote|define|suppose)\s+[A-Za-z]\b",
            re.IGNORECASE,
        )
        for span in base_spans:
            if span.char_start is None or span.char_end is None:
                continue
            if symbol_pattern.search(text[span.char_start:span.char_end]):
                add_indices(
                    [
                        index for index in span.token_indices
                        if aligned.offsets[index][1]
                        > aligned.offsets[index][0]
                    ],
                    "symbol_definition",
                )
    return result


def _connected_structural_components(
    structural_spans: Sequence[ClosureSpan],
) -> List[List[int]]:
    parents = list(range(len(structural_spans)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root < right_root:
                parents[right_root] = left_root
            else:
                parents[left_root] = right_root

    owners: Dict[int, int] = {}
    for span_index, span in enumerate(structural_spans):
        for token_index in sorted(span.token_indices):
            previous = owners.get(token_index)
            if previous is None:
                owners[token_index] = span_index
            else:
                union(previous, span_index)
    grouped: Dict[int, List[int]] = {}
    for span_index in range(len(structural_spans)):
        grouped.setdefault(find(span_index), []).append(span_index)
    return [
        grouped[root]
        for root in sorted(
            grouped,
            key=lambda root: (
                min(
                    structural_spans[index].token_start
                    for index in grouped[root]
                ),
                root,
            ),
        )
    ]


def build_atomic_items(
    text: str,
    aligned: Any,
    base_spans: Sequence[ClosureSpan],
    typed_spans: Sequence[ClosureSpan],
    protection: Dict[str, Any],
) -> Tuple[List[ClosureSpan], List[ClosureSpan], Dict[str, int]]:
    """Merge overlapping protected spans, then partition all residual tokens."""
    protection_spans = _build_protection_spans(
        text, aligned, base_spans, protection,
    )
    structural_spans = list(typed_spans) + protection_spans


    typed_tokens = {
        token for span in typed_spans for token in span.token_indices
    }
    partition_spans = list(typed_spans)
    for span in protection_spans:
        residual = [
            token for token in span.token_indices
            if token not in typed_tokens
        ]
        residual_span = _span_from_indices(
            residual,
            aligned,
            span.span_type,
            dict(span.metadata),
        )
        if residual_span is not None:
            partition_spans.append(residual_span)

    components = _connected_structural_components(partition_spans)
    raw_items: List[ClosureSpan] = []
    structural_tokens: Set[int] = set()
    dependency_merge_count = 0
    for component in components:
        sources = [partition_spans[index] for index in component]
        tokens = sorted({
            token
            for source in sources
            for token in source.token_indices
        })
        structural_tokens.update(tokens)
        span_types = sorted({source.span_type for source in sources})
        dependency_merge_count += int(len(sources) > 1)
        item = _span_from_indices(
            tokens,
            aligned,
            span_types[0],
            {
                "span_types": span_types,
                "protected": True,
                "mandatory": any(
                    bool(source.metadata.get("mandatory"))
                    for source in sources
                ),
                "structure_priority": max(
                    _STRUCTURE_PRIORITY.get(span_type, 0)
                    for span_type in span_types
                ),
                "source_span_count": len(sources),
            },
        )
        if item is not None:
            raw_items.append(item)

    for base_span in base_spans:
        residual = [
            index for index in base_span.token_indices
            if aligned.offsets[index][1] > aligned.offsets[index][0]
            and index not in structural_tokens
        ]
        runs: List[List[int]] = []
        for token_index in residual:
            if not runs or token_index != runs[-1][-1] + 1:
                runs.append([token_index])
            else:
                runs[-1].append(token_index)
        for run in runs:
            item = _span_from_indices(
                run,
                aligned,
                base_span.span_type,
                {
                    "span_types": [base_span.span_type],
                    "protected": False,
                    "mandatory": False,
                    "structure_priority": 0,
                },
            )
            if item is not None:
                raw_items.append(item)

    items: List[ClosureSpan] = []
    for span_id, item in enumerate(sorted(
        raw_items,
        key=lambda span: (
            span.token_start,
            span.token_end,
            tuple(span.metadata.get("span_types", [span.span_type])),
        ),
    )):
        item.span_id = span_id
        items.append(item)
    semantic_tokens = {
        index for index, (start, end) in enumerate(aligned.offsets)
        if end > start
    }
    covered = {
        token for item in items for token in item.token_indices
    }
    if covered != semantic_tokens:
        raise RuntimeError(
            "atomic item partition mismatch: "
            f"missing={len(semantic_tokens - covered)}, "
            f"extra={len(covered - semantic_tokens)}"
        )
    return items, structural_spans, {
        "dependency_merge_count": dependency_merge_count,
        "atomic_item_count": len(items),
    }


def _select_hard_mandatory(
    items: Sequence[ClosureSpan],
    budget: int,
    aligned: Any,
) -> Tuple[List[int], Set[int], Set[int], int]:
    mandatory = sorted(
        (item for item in items if item.metadata.get("mandatory")),
        key=lambda item: (
            -int(item.metadata.get("structure_priority", 0)),
            -item.token_start,
            item.token_end,
            item.span_id,
        ),
    )
    selected_ids: List[int] = []
    forced_tokens: Set[int] = set()
    blocked_ids = {item.span_id for item in mandatory}
    conflicts = 0
    for item in mandatory:
        missing = [
            index for index in item.token_indices
            if index not in forced_tokens
        ]
        remaining = budget - len(forced_tokens) - sum(
            next(
                candidate.cost for candidate in items
                if candidate.span_id == span_id
            )
            for span_id in selected_ids
        )
        if len(missing) <= remaining:
            selected_ids.append(item.span_id)
            continue
        conflicts += 1
        if remaining <= 0:
            continue
        ranked_tokens = sorted(
            missing,
            key=lambda index: (
                bool(re.search(
                    r"[0-9+\-*/=<>×÷]",
                    aligned.token_texts[index],
                )),
                index,
            ),
            reverse=True,
        )
        forced_tokens.update(ranked_tokens[:remaining])
    return selected_ids, forced_tokens, blocked_ids, conflicts


def _fill_exact_budget(
    items: Sequence[ClosureSpan],
    scores: Sequence[SpanScore],
    selected_ids: Sequence[int],
    initial_tokens: Set[int],
    budget: int,
    clipped_scores: torch.Tensor,
) -> Tuple[List[int], Set[int], Dict[str, Any]]:
    by_id = {item.span_id: item for item in items}
    score_by_id = {score.span_id: score for score in scores}
    selected_id_set = set(int(value) for value in selected_ids)
    selected_tokens = set(initial_tokens)
    for span_id in selected_id_set:
        selected_tokens.update(by_id[span_id].token_indices)
    before = len(selected_tokens)
    reasons: Dict[str, int] = {}
    added_full_spans = 0

    def is_hard_atomic(item: ClosureSpan) -> bool:
        return bool(set(item.metadata.get("span_types", [])).intersection({
            "reasoning_sentence",
            "quantity_unit_relation",
        }))

    def add_reason(reason: str, count: int) -> None:
        if count:
            reasons[reason] = reasons.get(reason, 0) + int(count)


    represented = {
        token
        for span_id in selected_id_set
        for token in by_id[span_id].token_indices
    }
    forced_tokens = selected_tokens - represented
    fixed_ids = {
        span_id for span_id in selected_id_set
        if by_id[span_id].metadata.get("mandatory")
    }
    fixed_tokens = set(forced_tokens)
    for span_id in fixed_ids:
        fixed_tokens.update(by_id[span_id].token_indices)
    optional_protected = sorted(
        (
            item for item in items
            if is_hard_atomic(item)
            and not item.metadata.get("mandatory")
            and not forced_tokens.intersection(item.token_indices)
        ),
        key=lambda item: (
            item.token_start,
            item.token_end,
            item.span_id,
        ),
    )
    filler_items = [
        item for item in items
        if not is_hard_atomic(item)
    ]
    filler_tokens = {
        token
        for item in filler_items
        for token in item.token_indices
        if token not in fixed_tokens
    }
    remaining_capacity = budget - len(fixed_tokens)
    if remaining_capacity >= 0 and optional_protected:
        states: Dict[
            int,
            Tuple[
                float, int, float, Tuple[int, ...],
                Tuple[Tuple[int, int, int], ...],
            ],
        ] = {0: (0.0, 0, 0.0, tuple(), tuple())}
        for item in optional_protected:
            item_score = score_by_id[item.span_id]
            signature = (
                item.token_start, item.token_end, item.span_id,
            )
            for used, state in sorted(
                list(states.items()), reverse=True,
            ):
                new_used = used + item.cost
                if new_used > remaining_capacity:
                    continue
                candidate_state = (
                    state[0] + float(item_score.selection_utility),
                    state[1] + int(item.metadata.get(
                        "structure_priority", 0,
                    )),
                    state[2] + float(item_score.positive_utility),
                    state[3] + (item.span_id,),
                    state[4] + (signature,),
                )
                if _state_is_better(
                    candidate_state, states.get(new_used),
                ):
                    states[new_used] = candidate_state

        filler_scores = sorted(
            (
                (float(clipped_scores[token].item()), token)
                for token in filler_tokens
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
        filler_prefix = [0.0]
        for value, _ in filler_scores:
            filler_prefix.append(filler_prefix[-1] + value)
        best_state = None
        for protected_cost, state in sorted(states.items()):
            filler_count = remaining_capacity - protected_cost
            if not 0 <= filler_count <= len(filler_scores):
                continue
            evaluated = (
                state[0] + filler_prefix[filler_count],
                state[1],
                state[2],
                state[3],
                state[4],
            )
            if _state_is_better(evaluated, best_state):
                best_state = evaluated
        if best_state is not None:
            chosen_protected = set(best_state[3])
            selected_id_set = set(fixed_ids) | chosen_protected
            selected_tokens = set(fixed_tokens)
            for span_id in selected_id_set:
                selected_tokens.update(by_id[span_id].token_indices)
            filler_budget = budget - len(selected_tokens)


            full_candidates = sorted(
                filler_items,
                key=lambda item: (
                    -(item.span_id in set(selected_ids)),
                    -float(score_by_id[item.span_id].utility_per_token),
                    -float(score_by_id[item.span_id].selection_utility),
                    item.token_start,
                    item.token_end,
                    item.span_id,
                ),
            )
            for item in full_candidates:
                if item.cost <= filler_budget:
                    selected_tokens.update(item.token_indices)
                    selected_id_set.add(item.span_id)
                    filler_budget -= item.cost
                    added_full_spans += 1
                    add_reason("whole_span", item.cost)
                    if filler_budget == 0:
                        break
            if filler_budget:
                available_filler = sorted(
                    filler_tokens - selected_tokens,
                    key=lambda token: (
                        -float(clipped_scores[token].item()),
                        token,
                    ),
                )
                for token in available_filler[:filler_budget]:
                    selected_tokens.add(token)
                    add_reason("score_token", 1)
            if len(selected_tokens) == budget:
                added_protected = (
                    chosen_protected - set(selected_ids)
                )
                add_reason(
                    "atomic_component_repack",
                    sum(by_id[span_id].cost for span_id in added_protected),
                )
                return sorted(selected_id_set), selected_tokens, {
                    "target_budget": budget,
                    "selected_before_fill": before,
                    "selected_after_fill": len(selected_tokens),
                    "backfilled_token_count": budget - before,
                    "backfilled_span_count": (
                        added_full_spans + len(added_protected)
                    ),
                    "exact_budget_match": True,
                    "backfill_reason": reasons,
                    "forced_atomicity_relaxation_count": 0,
                }

    def add_full_item(item: ClosureSpan, reason: str) -> bool:
        nonlocal added_full_spans
        missing = set(item.token_indices) - selected_tokens
        if not missing or len(selected_tokens) + len(missing) > budget:
            return False
        selected_tokens.update(missing)
        selected_id_set.add(item.span_id)
        added_full_spans += 1
        add_reason(reason, len(missing))
        return True


    touched = [
        item for item in items
        if item.span_id not in selected_id_set
        and is_hard_atomic(item)
        and selected_tokens.intersection(item.token_indices)
    ]
    for item in sorted(
        touched,
        key=lambda candidate: (
            -int(candidate.metadata.get("structure_priority", 0)),
            candidate.token_start,
            candidate.token_end,
            candidate.span_id,
        ),
    ):
        types = set(item.metadata.get("span_types", []))
        reason = (
            "quantity_dependency"
            if "quantity_unit_relation" in types
            else "reasoning_adjacency"
        )
        add_full_item(item, reason)


    candidates = sorted(
        (
            item for item in items
            if item.span_id not in selected_id_set
            and not selected_tokens.intersection(item.token_indices)
        ),
        key=lambda item: (
            -float(score_by_id[item.span_id].selection_utility),
            -int(item.metadata.get("structure_priority", 0)),
            item.token_start,
            item.token_end,
            item.span_id,
        ),
    )
    progress = True
    while progress and len(selected_tokens) < budget:
        progress = False
        for item in candidates:
            if item.span_id in selected_id_set:
                continue
            if add_full_item(item, "whole_span"):
                progress = True
                if len(selected_tokens) == budget:
                    break

    protected_tokens = {
        token
        for item in items
        if is_hard_atomic(item)
        and item.span_id not in selected_id_set
        for token in item.token_indices
    }
    available = sorted({
        token
        for item in items
        for token in item.token_indices
        if token not in selected_tokens
        and token not in protected_tokens
    })
    adjacent = sorted(
        (
            token for token in available
            if token - 1 in selected_tokens or token + 1 in selected_tokens
        ),
        key=lambda token: (
            -float(clipped_scores[token].item()),
            token,
        ),
    )
    for token in adjacent:
        if len(selected_tokens) >= budget:
            break
        selected_tokens.add(token)
        add_reason("adjacent_token", 1)
    remaining_tokens = sorted(
        (token for token in available if token not in selected_tokens),
        key=lambda token: (
            -float(clipped_scores[token].item()),
            token,
        ),
    )
    for token in remaining_tokens:
        if len(selected_tokens) >= budget:
            break
        selected_tokens.add(token)
        add_reason("score_token", 1)


    if len(selected_tokens) < budget:
        selected_protected_tokens = {
            token
            for item in items
            if item.span_id in selected_id_set
            and is_hard_atomic(item)
            for token in item.token_indices
        }
        swap_candidates = sorted(
            (
                item for item in items
                if item.span_id not in selected_id_set
                and is_hard_atomic(item)
                and not selected_tokens.intersection(item.token_indices)
                and item.cost <= budget
            ),
            key=lambda item: (
                -int(item.metadata.get("structure_priority", 0)),
                -float(score_by_id[item.span_id].selection_utility),
                item.token_start,
                item.token_end,
                item.span_id,
            ),
        )
        for item in swap_candidates:
            deficit = budget - len(selected_tokens)
            remove_count = max(0, item.cost - deficit)
            removable = sorted(
                selected_tokens - selected_protected_tokens,
                key=lambda token: (
                    float(clipped_scores[token].item()),
                    -token,
                ),
            )
            if len(removable) < remove_count:
                continue
            removed = set(removable[:remove_count])
            selected_tokens.difference_update(removed)
            for selected_id in list(selected_id_set):
                selected_item = by_id[selected_id]
                if (
                    not selected_item.metadata.get("protected")
                    and removed.intersection(selected_item.token_indices)
                ):
                    selected_id_set.remove(selected_id)
            selected_tokens.update(item.token_indices)
            selected_id_set.add(item.span_id)
            added_full_spans += 1
            types = set(item.metadata.get("span_types", []))
            reason = (
                "quantity_dependency_swap"
                if "quantity_unit_relation" in types
                else "reasoning_atomic_swap"
            )
            add_reason(reason, item.cost)
            if len(selected_tokens) == budget:
                break

    forced_relaxation = 0
    if len(selected_tokens) < budget:


        token_is_hard_atomic: Dict[int, bool] = {}
        for item in items:
            types = set(item.metadata.get("span_types", []))
            is_hard_atomic = bool(types.intersection({
                "reasoning_sentence",
                "quantity_unit_relation",
            }))
            for token in item.token_indices:
                token_is_hard_atomic[token] = is_hard_atomic
        remaining = sorted(
            {
                token for item in items
                for token in item.token_indices
                if token not in selected_tokens
            },
            key=lambda token: (
                token_is_hard_atomic.get(token, False),
                -float(clipped_scores[token].item()),
                token,
            ),
        )
        for token in remaining:
            if len(selected_tokens) >= budget:
                break
            selected_tokens.add(token)
            forced_relaxation += 1
            add_reason("forced_atomicity_relaxation", 1)
    return sorted(selected_id_set), selected_tokens, {
        "target_budget": budget,
        "selected_before_fill": before,
        "selected_after_fill": len(selected_tokens),
        "backfilled_token_count": len(selected_tokens) - before,
        "backfilled_span_count": added_full_spans,
        "exact_budget_match": len(selected_tokens) == budget,
        "backfill_reason": reasons,
        "forced_atomicity_relaxation_count": forced_relaxation,
    }


def _coverage_counts(
    tokens: Set[int],
    selected: Set[int],
) -> Tuple[int, int]:
    return len(tokens.intersection(selected)), len(tokens)


def _enhanced_diagnostics(
    text: str,
    aligned: Any,
    typed_spans: Sequence[ClosureSpan],
    structural_spans: Sequence[ClosureSpan],
    selected_tokens: Set[int],
    budget: int,
    selected_before_fill: int,
    fill_stats: Dict[str, Any],
    clipped_token_count: int,
    dependency_merge_count: int,
    hard_budget_conflict_count: int,
) -> Dict[str, Any]:
    reasoning_spans = [
        span for span in typed_spans
        if span.span_type == "reasoning_sentence"
    ]
    quantity_spans = [
        span for span in typed_spans
        if span.span_type == "quantity_unit_relation"
    ]

    def selected_span_count(spans: Sequence[ClosureSpan]) -> int:
        return sum(
            set(span.token_indices).issubset(selected_tokens)
            for span in spans
        )

    def partial_span_count(spans: Sequence[ClosureSpan]) -> int:
        return sum(
            bool(set(span.token_indices).intersection(selected_tokens))
            and not set(span.token_indices).issubset(selected_tokens)
            for span in spans
        )

    reasoning_tokens = {
        token for span in reasoning_spans for token in span.token_indices
    }
    quantity_tokens = {
        token for span in quantity_spans for token in span.token_indices
    }
    final_tokens = {
        token
        for span in structural_spans
        if span.span_type in {"boxed_answer", "final_answer"}
        for token in span.token_indices
    }
    semantic_tokens = {
        index for index, (start, end) in enumerate(aligned.offsets)
        if end > start
    }
    digit_tokens = {
        index for index in semantic_tokens
        if any(char.isdigit() for char in aligned.token_texts[index])
    }
    operator_tokens = {
        index for index in semantic_tokens
        if re.search(r"[+\-*/=<>×÷%]", aligned.token_texts[index])
    }
    reasoning_selected, reasoning_total = _coverage_counts(
        reasoning_tokens, selected_tokens,
    )
    quantity_selected, quantity_total = _coverage_counts(
        quantity_tokens, selected_tokens,
    )
    final_selected, final_total = _coverage_counts(
        final_tokens, selected_tokens,
    )
    digit_selected, digit_total = _coverage_counts(
        digit_tokens, selected_tokens,
    )
    operator_selected, operator_total = _coverage_counts(
        operator_tokens, selected_tokens,
    )
    reasoning_partial = partial_span_count(reasoning_spans)
    quantity_partial = partial_span_count(quantity_spans)
    return {
        **fill_stats,
        "semantic_response_token_count": len(semantic_tokens),
        "selected_semantic_response_token_count": len(selected_tokens),
        "budget_underflow_count": int(len(selected_tokens) < budget),
        "budget_overflow_count": int(len(selected_tokens) > budget),
        "hard_budget_conflict_count": int(hard_budget_conflict_count),
        "clipped_token_count": int(clipped_token_count),
        "reasoning_sentence_count": len(reasoning_spans),
        "selected_reasoning_sentence_count": selected_span_count(
            reasoning_spans
        ),
        "reasoning_sentence_selected_tokens": reasoning_selected,
        "reasoning_sentence_total_tokens": reasoning_total,
        "partially_selected_reasoning_sentence_count": reasoning_partial,
        "reasoning_atomicity_violation_count": reasoning_partial,
        "quantity_unit_relation_count": len(quantity_spans),
        "selected_quantity_unit_relation_count": selected_span_count(
            quantity_spans
        ),
        "quantity_unit_selected_tokens": quantity_selected,
        "quantity_unit_total_tokens": quantity_total,
        "partial_quantity_relation_count": quantity_partial,
        "final_answer_selected_tokens": final_selected,
        "final_answer_total_tokens": final_total,
        "digit_selected_tokens": digit_selected,
        "digit_total_tokens": digit_total,
        "operator_selected_tokens": operator_selected,
        "operator_total_tokens": operator_total,
        "dependency_merge_count": int(dependency_merge_count),
        "selected_before_fill": int(selected_before_fill),
    }


def project_response_selection(
    record: Any,
    prompt_selection: Any,
    baseline_response_selection: Any,
    sample: Dict[str, Any],
    tokenizer: Any,
    max_response_length: int,
    structure_config: Dict[str, Any],
    expected_response_ids: Optional[Sequence[int]] = None,
    structure_protection: Optional[Dict[str, Any]] = None,
    special_token_mode: str = "legacy",
) -> StructuredSelectionResult:
    """Build the existing static Closure mask with an unchanged hard budget."""
    total_started = time.perf_counter()
    protection = dict(structure_protection or {})
    enhanced_enabled = bool(
        structure_config.get("selection_strategy") == "knapsack"
        or structure_config.get("require_exact_budget", False)
        or structure_config.get("negative_clip") is not None
        or float(structure_config.get("reasoning_sentence_bonus", 0.0))
        > 0.0
        or float(structure_config.get("quantity_unit_bonus", 0.0))
        > 0.0
        or protection.get("protect_reasoning_sentence", False)
        or protection.get("protect_quantity_unit", False)
    )
    response_length = int(record.response_length)
    baseline = valid_indices(
        baseline_response_selection, response_length,
    )
    budget = len(baseline)
    token_scores = conditional_response_utility(
        record, torch.as_tensor(prompt_selection),
    )
    if token_scores.numel() != response_length:
        raise ValueError(
            "conditional response score length does not match score record"
        )
    if not bool(torch.isfinite(token_scores).all()):
        raise FloatingPointError(
            "conditional response utility contains NaN/Inf"
        )

    span_started = time.perf_counter()
    try:
        aligned = tokenize_with_verified_offsets(
            str(sample.get("response", "")),
            tokenizer,
            max_response_length,
            expected_ids=expected_response_ids,
            expected_length=response_length,
            special_token_mode=special_token_mode,
        )
        spans, construction = build_closure_spans(
            str(sample.get("response", "")),
            aligned,
            min_span_tokens=int(structure_config["min_span_tokens"]),
            max_span_tokens=int(structure_config["max_span_tokens"]),
            merge_short_spans=bool(
                structure_config["merge_short_spans"]
            ),
            exclude_delimiter_tokens=bool(
                structure_config["exclude_delimiter_tokens"]
            ),
        )
        construction["token_offsets"] = [
            list(item) for item in aligned.offsets
        ]
        construction["token_texts"] = list(aligned.token_texts)
    except (AlignmentError, RuntimeError, ValueError) as error:
        if not structure_config["fallback_to_token"]:
            raise
        span_elapsed = time.perf_counter() - span_started
        return _identity_result(
            baseline,
            budget,
            response_length,
            f"alignment_or_span_failure: {error}",
            {
                "span_construction": span_elapsed,
                "total_structural_overhead": (
                    time.perf_counter() - total_started
                ),
            },
        )
    span_elapsed = time.perf_counter() - span_started

    selection_spans = spans
    typed_spans: List[ClosureSpan] = []
    structural_spans: List[ClosureSpan] = []
    atomic_stats: Dict[str, int] = {
        "dependency_merge_count": 0,
        "atomic_item_count": len(spans),
    }
    if enhanced_enabled:
        typed_spans, typed_stats = build_typed_spans(
            str(sample.get("response", "")),
            aligned,
            include_reasoning_sentence=bool(
                protection.get("protect_reasoning_sentence")
                or float(structure_config.get(
                    "reasoning_sentence_bonus", 0.0,
                )) > 0.0
            ),
            include_quantity_unit=bool(
                protection.get("protect_quantity_unit")
                or float(structure_config.get(
                    "quantity_unit_bonus", 0.0,
                )) > 0.0
            ),
            exclude_delimiter_tokens=bool(
                structure_config["exclude_delimiter_tokens"]
            ),
        )
        selection_spans, structural_spans, atomic_stats = (
            build_atomic_items(
                str(sample.get("response", "")),
                aligned,
                spans,
                typed_spans,
                protection,
            )
        )
        construction.update(typed_stats)
        construction.update(atomic_stats)

    aggregate_started = time.perf_counter()
    span_scores = aggregate_span_scores(
        selection_spans,
        token_scores,
        reduction=structure_config["score_reduction"],
        core_tokens_only=bool(structure_config["core_tokens_only"]),
        length_alpha=float(structure_config["length_alpha"]),
        negative_penalty=float(structure_config["negative_penalty"]),
        top_mean_fraction=float(
            structure_config["top_mean_fraction"]
        ),
        negative_clip=structure_config.get("negative_clip"),
        span_type_bonuses={
            "reasoning_sentence": float(structure_config.get(
                "reasoning_sentence_bonus", 0.0,
            )),
            "quantity_unit_relation": float(structure_config.get(
                "quantity_unit_bonus", 0.0,
            )),
        } if enhanced_enabled else None,
    )
    aggregate_elapsed = time.perf_counter() - aggregate_started

    if (
        construction.get("fallback_reason")
        and structure_config["fallback_to_token"]
        and not enhanced_enabled
    ):
        return _identity_result(
            baseline,
            budget,
            response_length,
            str(construction["fallback_reason"]),
            {
                "span_construction": span_elapsed,
                "score_aggregation": aggregate_elapsed,
                "total_structural_overhead": (
                    time.perf_counter() - total_started
                ),
            },
            spans,
            span_scores,
            construction,
        )

    selection_started = time.perf_counter()
    if enhanced_enabled:
        semantic_indices = {
            index for index, (start, end) in enumerate(aligned.offsets)
            if end > start
        }
        budget = min(budget, len(semantic_indices))
        preselected, forced_tokens, blocked_ids, hard_conflicts = (
            _select_hard_mandatory(
                selection_spans, budget, aligned,
            )
        )
        eligible_ids = {
            item.span_id for item in selection_spans
            if item.span_id not in blocked_ids
            or item.span_id in set(preselected)
        }
        eligible_spans = [
            item for item in selection_spans
            if item.span_id in eligible_ids
        ]
        eligible_scores = [
            score for score in span_scores
            if score.span_id in eligible_ids
        ]
        item_budget = max(0, budget - len(forced_tokens))
        knapsack_fallback = None
        if structure_config["selection_strategy"] == "knapsack":
            selected_span_ids, knapsack_fallback = (
                knapsack_select_spans(
                    eligible_spans,
                    eligible_scores,
                    item_budget,
                    preselected=preselected or None,
                    max_capacity=int(structure_config.get(
                        "max_knapsack_capacity", 2048,
                    )),
                )
            )
        else:
            selected_span_ids = select_spans(
                eligible_spans,
                eligible_scores,
                item_budget,
                structure_config["selection_strategy"],
                preselected=preselected or None,
            )
        selected_tokens = set(forced_tokens)
        selected_tokens.update(
            span_token_set(selection_spans, selected_span_ids)
        )
        selected_before_fill = len(selected_tokens)
        clipped_scores = token_scores.detach().cpu().float()
        negative_clip = structure_config.get("negative_clip")
        if negative_clip is not None:
            clipped_scores = clipped_scores.clamp_min(
                -float(negative_clip)
            )
        if structure_config.get("require_exact_budget", False):
            selected_span_ids, selected_tokens, fill_stats = (
                _fill_exact_budget(
                    selection_spans,
                    span_scores,
                    selected_span_ids,
                    selected_tokens,
                    budget,
                    clipped_scores,
                )
            )
        else:
            fill_stats = {
                "target_budget": budget,
                "selected_before_fill": selected_before_fill,
                "selected_after_fill": len(selected_tokens),
                "backfilled_token_count": 0,
                "backfilled_span_count": 0,
                "exact_budget_match": len(selected_tokens) == budget,
                "backfill_reason": {},
                "forced_atomicity_relaxation_count": 0,
            }
        hard_conflicts += int(
            fill_stats.get("forced_atomicity_relaxation_count", 0) > 0
        )
        enhanced_stats = _enhanced_diagnostics(
            str(sample.get("response", "")),
            aligned,
            typed_spans,
            structural_spans,
            selected_tokens,
            budget,
            selected_before_fill,
            fill_stats,
            clipped_token_count=sum(
                score.clipped_token_count for score in span_scores
            ),
            dependency_merge_count=int(
                atomic_stats.get("dependency_merge_count", 0)
            ),
            hard_budget_conflict_count=hard_conflicts,
        )
        fallback_reason = (
            knapsack_fallback
            or construction.get("fallback_reason")
        )
        return _closure_result(
            selection_spans,
            span_scores,
            selected_span_ids,
            clipped_scores,
            baseline,
            budget,
            response_length,
            fallback_reason,
            {
                "span_construction": span_elapsed,
                "score_aggregation": aggregate_elapsed,
                "structured_selection": (
                    time.perf_counter() - selection_started
                ),
                "total_structural_overhead": (
                    time.perf_counter() - total_started
                ),
            },
            {
                **construction,
                **enhanced_stats,
                "structure_protection": {
                    "hard_mandatory_preselected": len(preselected),
                    "hard_budget_conflict_count": hard_conflicts,
                },
            },
            selected_token_indices=sorted(selected_tokens),
        )


    preselected: List[int] = []
    protection_stats: Dict[str, Any] = {}
    if structure_protection:
        required_ids, req_stats = _required_span_ids(
            spans, str(sample.get("response", "")), structure_protection,
        )
        by_id = {span.span_id: span for span in spans}
        used = 0
        dropped = []
        for span_id in required_ids:
            cost = by_id[span_id].cost
            if used + cost <= budget:
                preselected.append(span_id)
                used += cost
            else:
                dropped.append(span_id)
        protection_stats = {
            "required_by_kind": req_stats,
            "required_preselected": len(preselected),
            "required_dropped": dropped,
            "required_dropped_count": len(dropped),
        }
    selected_span_ids = select_spans(
        spans,
        span_scores,
        budget,
        structure_config["selection_strategy"],
        preselected=preselected or None,
    )
    if not selected_span_ids and budget and structure_config["fallback_to_token"]:
        return _identity_result(
            baseline,
            budget,
            response_length,
            "no_complete_span_fits_budget",
            {
                "span_construction": span_elapsed,
                "score_aggregation": aggregate_elapsed,
                "structured_selection": (
                    time.perf_counter() - selection_started
                ),
                "total_structural_overhead": (
                    time.perf_counter() - total_started
                ),
            },
            spans,
            span_scores,
            construction,
        )
    return _closure_result(
        spans,
        span_scores,
        selected_span_ids,
        token_scores,
        baseline,
        budget,
        response_length,
        construction.get("fallback_reason"),
        {
            "span_construction": span_elapsed,
            "score_aggregation": aggregate_elapsed,
            "structured_selection": (
                time.perf_counter() - selection_started
            ),
            "total_structural_overhead": (
                time.perf_counter() - total_started
            ),
        },
        {**construction, **({"structure_protection": protection_stats} if protection_stats else {})},
    )
