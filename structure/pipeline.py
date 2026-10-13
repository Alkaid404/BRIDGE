"""Dataset-level static Closure projection for Stage 4."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from structure.aggregation import conditional_response_utility
from structure.diagnostics import OnlineSummary, peak_cpu_memory_mb
from structure.selection import project_response_selection
from utils import get_logger

logger = get_logger("bridge.structure")


def expected_response_ids(
    dataset: Any, index: int, response_length: int,
) -> List[int]:
    item = dataset[index]
    prompt_length = int(item["prompt_length"])
    ids = item["input_ids"][
        prompt_length:prompt_length + response_length
    ]
    return [int(value) for value in ids.tolist()]


def apply_structured_selection(
    records: Sequence[Any],
    train_samples: Sequence[Dict[str, Any]],
    tokenizer: Any,
    train_dataset: Any,
    prompt_selection: List[torch.Tensor],
    response_selection: List[torch.Tensor],
    max_response_length: int,
    structure_config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    write_sample_diagnostics: bool = True,
    structure_protection: Optional[Dict[str, Any]] = None,
    special_token_mode: str = "legacy",
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Build the static Closure mask; prompt selections remain unchanged."""
    if not (
        len(records)
        == len(train_samples)
        == len(prompt_selection)
        == len(response_selection)
        == len(train_dataset)
    ):
        raise ValueError("Closure selection inputs differ in sample count")

    started = time.perf_counter()
    cpu_peak_before = peak_cpu_memory_mb()
    projected: List[torch.Tensor] = []
    accumulator = OnlineSummary()
    diagnostics_path = None
    diagnostics_handle = None
    temporary_path = None
    if output_dir is not None and write_sample_diagnostics:
        diagnostics_path = (
            Path(output_dir) / "structure_diagnostics" / "samples.jsonl"
        )
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = diagnostics_path.with_suffix(".jsonl.tmp")
        diagnostics_handle = temporary_path.open("w", encoding="utf-8")

    try:
        for index, record in enumerate(tqdm(
            records, desc="Closure projection", unit="sample",
        )):
            expected_ids = expected_response_ids(
                train_dataset, index, int(record.response_length),
            )
            result = project_response_selection(
                record=record,
                prompt_selection=prompt_selection[index],
                baseline_response_selection=response_selection[index],
                sample=train_samples[index],
                tokenizer=tokenizer,
                max_response_length=max_response_length,
                structure_config=structure_config,
                expected_response_ids=expected_ids,
                structure_protection=structure_protection,
                special_token_mode=special_token_mode,
            )
            selected = torch.tensor(
                result.selected_response_token_indices,
                dtype=torch.long,
            )
            if selected.numel() > torch.as_tensor(
                response_selection[index]
            ).numel():
                raise RuntimeError(
                    f"sample {index} exceeds original response token budget"
                )
            projected.append(selected)
            payload = result.to_dict()
            accumulator.add(payload)
            if diagnostics_handle is not None:
                token_scores = conditional_response_utility(
                    record, prompt_selection[index],
                )
                diagnostics_handle.write(json.dumps({
                    "sample_idx": index,
                    "sample_id": train_samples[index].get("id"),
                    "task_type": train_samples[index].get("task_type"),
                    "response_text": train_samples[index].get(
                        "response", ""
                    ),
                    "token_scores": [
                        float(value) for value in token_scores.tolist()
                    ],
                    "baseline_response_token_indices": [
                        int(value)
                        for value in torch.as_tensor(
                            response_selection[index]
                        ).view(-1).tolist()
                        if int(value) >= 0
                    ],
                    **payload,
                }, ensure_ascii=False) + "\n")
    finally:
        if diagnostics_handle is not None:
            diagnostics_handle.close()

    if temporary_path is not None and diagnostics_path is not None:
        temporary_path.replace(diagnostics_path)
    summary = accumulator.result()
    summary["structural_peak_cpu_delta_mb"] = max(
        0.0,
        float(summary.get("peak_cpu_memory_mb", 0.0))
        - cpu_peak_before,
    )
    summary["preprocessing_wall_time"] = time.perf_counter() - started
    summary["profile"] = "closure_only"
    summary["diagnostics_file"] = (
        str(diagnostics_path) if diagnostics_path is not None else None
    )
    if output_dir is not None:
        diagnostics_dir = Path(output_dir) / "structure_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    logger.info(
        "Closure projection: samples=%d, fragmentation=%.4f, "
        "utility_retention=%.4f, budget_utilization=%.4f, fallback=%.4f",
        summary["sample_count"],
        summary["fragmentation_rate"],
        summary["utility_retention"],
        summary["budget_utilization"],
        summary["fallback_rate"],
    )
    return projected, {"summary": summary}
