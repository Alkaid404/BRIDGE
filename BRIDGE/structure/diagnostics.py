"""Streaming diagnostics for static Closure and epoch masks."""

import json
import resource
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List


def peak_cpu_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    accumulator = OnlineSummary()
    for row in results:
        accumulator.add(row)
    return accumulator.result()


class OnlineSummary:
    """O(1)-memory accumulator for Closure-only diagnostics."""

    def __init__(self) -> None:
        self.count = 0
        self.sums: DefaultDict[str, float] = defaultdict(float)
        self.timing_sums: DefaultDict[str, float] = defaultdict(float)
        self.backfill_reasons: DefaultDict[str, int] = defaultdict(int)

    def add(self, row: Dict[str, Any]) -> None:
        self.count += 1
        diagnostics = row.get("diagnostics", {})
        for field in (
            "fragmentation_rate",
            "utility_retention",
            "budget_utilization",
            "actual_selected_tokens",
            "selection_budget",
        ):
            self.sums[field] += float(row.get(field, 0.0))
        self.sums["fallback_count"] += float(
            bool(row.get("fallback_reason"))
        )
        self.sums["alignment_failure_count"] += float(
            "alignment" in str(row.get("fallback_reason") or "").lower()
        )
        self.sums["average_span_length"] += float(
            diagnostics.get("average_span_length", 0.0)
        )
        self.sums["average_selected_span_length"] += float(
            diagnostics.get("average_selected_span_length", 0.0)
        )
        self.sums["selected_span_count"] += float(
            len(row.get("selected_span_ids", []))
        )
        self.sums["selected_fragmentation_rate"] += float(
            diagnostics.get("selected_fragmentation_rate", 0.0)
        )
        self.sums["empty_mask_count"] += float(
            int(row.get("actual_selected_tokens", 0)) == 0
            and int(row.get("selection_budget", 0)) > 0
        )
        self.sums["budget_overflow_count"] += float(
            int(row.get("actual_selected_tokens", 0))
            > int(row.get("selection_budget", 0))
        )
        self.sums["budget_underflow_count"] += float(
            int(row.get("actual_selected_tokens", 0))
            < int(row.get("selection_budget", 0))
        )
        self.sums["exact_budget_match_count"] += float(
            bool(diagnostics.get(
                "exact_budget_match",
                int(row.get("actual_selected_tokens", 0))
                == int(row.get("selection_budget", 0)),
            ))
        )
        for field in (
            "hard_budget_conflict_count",
            "clipped_token_count",
            "reasoning_sentence_count",
            "selected_reasoning_sentence_count",
            "partially_selected_reasoning_sentence_count",
            "reasoning_atomicity_violation_count",
            "quantity_unit_relation_count",
            "selected_quantity_unit_relation_count",
            "partial_quantity_relation_count",
            "dependency_merge_count",
            "backfilled_token_count",
            "backfilled_span_count",
            "semantic_response_token_count",
            "selected_semantic_response_token_count",
            "reasoning_sentence_selected_tokens",
            "reasoning_sentence_total_tokens",
            "quantity_unit_selected_tokens",
            "quantity_unit_total_tokens",
            "final_answer_selected_tokens",
            "final_answer_total_tokens",
            "digit_selected_tokens",
            "digit_total_tokens",
            "operator_selected_tokens",
            "operator_total_tokens",
        ):
            self.sums[field] += float(diagnostics.get(field, 0.0))
        for reason, count in diagnostics.get(
            "backfill_reason", {}
        ).items():
            self.backfill_reasons[str(reason)] += int(count)
        for name, value in row.get("timings", {}).items():
            self.timing_sums[name] += float(value)

    def result(self) -> Dict[str, Any]:
        if not self.count:
            return {
                "sample_count": 0,
                "peak_cpu_memory_mb": peak_cpu_memory_mb(),
            }
        summary = {
            name: value / self.count
            for name, value in self.sums.items()
        }
        summary.update({
            "sample_count": self.count,
            "total_samples": self.count,
            "fallback_count": int(round(self.sums["fallback_count"])),
            "fallback_rate": self.sums["fallback_count"] / self.count,
            "alignment_failure_count": int(round(
                self.sums["alignment_failure_count"]
            )),
            "alignment_failure_ratio": (
                self.sums["alignment_failure_count"] / self.count
            ),
            "empty_mask_count": int(round(
                self.sums["empty_mask_count"]
            )),
            "budget_overflow_count": int(round(
                self.sums["budget_overflow_count"]
            )),
            "budget_underflow_count": int(round(
                self.sums["budget_underflow_count"]
            )),
            "exact_budget_match_rate": (
                self.sums["exact_budget_match_count"] / self.count
            ),
            "final_budget_utilization": (
                self.sums["actual_selected_tokens"]
                / self.sums["selection_budget"]
                if self.sums["selection_budget"] else 1.0
            ),
            "average_timings": {
                name: value / self.count
                for name, value in self.timing_sums.items()
            },
            "peak_cpu_memory_mb": peak_cpu_memory_mb(),
        })
        for field in (
            "hard_budget_conflict_count",
            "clipped_token_count",
            "reasoning_sentence_count",
            "selected_reasoning_sentence_count",
            "partially_selected_reasoning_sentence_count",
            "reasoning_atomicity_violation_count",
            "quantity_unit_relation_count",
            "selected_quantity_unit_relation_count",
            "partial_quantity_relation_count",
            "dependency_merge_count",
            "backfilled_token_count",
            "backfilled_span_count",
        ):
            summary[field] = int(round(self.sums[field]))
        summary["effective_response_keep_ratio"] = (
            self.sums["selected_semantic_response_token_count"]
            / self.sums["semantic_response_token_count"]
            if self.sums["semantic_response_token_count"]
            else summary["final_budget_utilization"]
        )

        def coverage(selected_field: str, total_field: str) -> float:
            total = self.sums[total_field]
            return self.sums[selected_field] / total if total else 1.0

        summary["reasoning_sentence_token_coverage"] = coverage(
            "reasoning_sentence_selected_tokens",
            "reasoning_sentence_total_tokens",
        )
        summary["quantity_unit_token_coverage"] = coverage(
            "quantity_unit_selected_tokens",
            "quantity_unit_total_tokens",
        )
        summary["final_answer_token_coverage"] = coverage(
            "final_answer_selected_tokens",
            "final_answer_total_tokens",
        )
        summary["digit_token_coverage"] = coverage(
            "digit_selected_tokens", "digit_total_tokens",
        )
        summary["operator_token_coverage"] = coverage(
            "operator_selected_tokens", "operator_total_tokens",
        )
        summary["backfill_reason"] = dict(sorted(
            self.backfill_reasons.items()
        ))
        return summary
