"Unified retention-prior construction."

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from models.utils import (
    BridgeModel,
    collect_vjp_parameters,
    compute_retention_direction,
)


SUPPORTED_RETENTION_PRIORS = ("auxiliary_ce", "l2_anchor")


def _parameter_metadata(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> Dict[str, Any]:
    rows = []
    aliases: Dict[Tuple[str, int, int], list[str]] = {}
    total = 0
    for name, parameter in named_parameters:
        row = {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        rows.append(row)
        total += row["numel"]
        try:
            storage = parameter.untyped_storage()
            key = (str(parameter.device), int(storage.data_ptr()), int(parameter.storage_offset()))
            aliases.setdefault(key, []).append(name)
        except (AttributeError, RuntimeError):
            pass
    tied_groups = [names for names in aliases.values() if len(names) > 1]
    return {
        "parameters": rows,
        "parameter_count": len(rows),
        "numel": total,
        "trainable_count": sum(row["requires_grad"] for row in rows),
        "frozen_count": sum(not row["requires_grad"] for row in rows),
        "tied_parameter_groups": tied_groups,
        "parameter_policy": (
            "Exactly collect_vjp_parameters(model, num_last_layers=4), including every "
            "parameter in those layers; no name/type filtering. Frozen status is audited "
            "but does not change membership, matching target-direction construction."
        ),
    }


def _assert_same_parameter_space(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    reference_label: str,
    candidate_label: str,
) -> None:
    ref_rows = reference["parameters"]
    cand_rows = candidate["parameters"]
    if len(ref_rows) != len(cand_rows):
        raise RuntimeError(
            f"parameter count mismatch: {reference_label}={len(ref_rows)} "
            f"vs {candidate_label}={len(cand_rows)}"
        )
    for index, (left, right) in enumerate(zip(ref_rows, cand_rows)):
        for key in ("name", "shape", "numel"):
            if left[key] != right[key]:
                raise RuntimeError(
                    f"parameter-space mismatch at index {index}, field {key}: "
                    f"{reference_label}={left[key]!r}, {candidate_label}={right[key]!r}"
                )


def build_l2_anchor_direction(
    pretrained_checkpoint: str,
    warmup_checkpoint: str,
    *,
    target_direction: Optional[torch.Tensor] = None,
    num_last_layers: int = 4,
    layer_indices: Optional[List[int]] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    "Return ``theta_bar - theta_0`` in the exact target/VJP parameter space."
    if layer_indices is None and num_last_layers != 4:
        raise ValueError(
            "formal L2-anchor experiments must use the same last-4-layer space "
            f"as target scoring, got {num_last_layers}"
        )

    warmup_model = BridgeModel(
        warmup_checkpoint,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
    )
    warmup_named = collect_vjp_parameters(warmup_model, num_last_layers, layer_indices)
    warmup_meta = _parameter_metadata(warmup_named)
    if target_direction is not None and int(target_direction.numel()) != warmup_meta["numel"]:
        raise RuntimeError(
            f"target direction has {target_direction.numel()} values but the audited "
            f"last-{num_last_layers} space has {warmup_meta['numel']}"
        )
    anchor = torch.cat(
        [parameter.detach().reshape(-1).float().cpu() for _, parameter in warmup_named]
    )
    del warmup_named, warmup_model
    gc.collect()

    pretrained_model = BridgeModel(
        pretrained_checkpoint,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"},
    )
    pretrained_named = collect_vjp_parameters(pretrained_model, num_last_layers, layer_indices)
    pretrained_meta = _parameter_metadata(pretrained_named)
    _assert_same_parameter_space(
        warmup_meta, pretrained_meta, "warmup", "pretrained"
    )
    offset = 0
    for (_, parameter), row in zip(pretrained_named, pretrained_meta["parameters"]):
        numel = int(row["numel"])
        anchor[offset : offset + numel].sub_(
            parameter.detach().reshape(-1).float().cpu()
        )
        offset += numel
    if offset != anchor.numel():
        raise RuntimeError(f"L2 subtraction consumed {offset}, expected {anchor.numel()}")
    del pretrained_named, pretrained_model
    gc.collect()

    if not bool(torch.isfinite(anchor).all()):
        raise FloatingPointError("L2 anchor direction contains NaN/Inf")
    if float(anchor.norm().item()) == 0.0:
        raise RuntimeError(
            "L2 anchor is all-zero; pretrained and warmup checkpoints may be identical"
        )
    metadata = {
        "prior": "l2_anchor",
        "formula": "grad_theta (1/2 ||theta-theta_0||_2^2) at theta_bar = theta_bar-theta_0",
        "sign": "raw_positive_gradient",
        "pretrained_checkpoint": str(Path(pretrained_checkpoint)),
        "warmup_checkpoint": str(Path(warmup_checkpoint)),
        "num_last_layers": num_last_layers,
        "layer_indices": None if layer_indices is None else [int(i) for i in layer_indices],
        "difference_dtype": "torch.float32",
        "warmup_parameter_space": warmup_meta,
        "pretrained_parameter_space": pretrained_meta,
    }
    return anchor, metadata


def build_retention_direction(
    prior: str,
    *,
    model: Optional[BridgeModel] = None,
    dataloader: Any = None,
    max_batches: Optional[int] = None,
    pretrained_checkpoint: Optional[str] = None,
    warmup_checkpoint: Optional[str] = None,
    target_direction: Optional[torch.Tensor] = None,
    num_last_layers: int = 4,
    layer_indices: Optional[List[int]] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Dispatch a retention prior without changing downstream mixing/scoring."""
    if prior == "auxiliary_ce":
        if model is None or dataloader is None:
            raise ValueError("auxiliary_ce requires model and dataloader")
        value = compute_retention_direction(model, dataloader, max_batches=max_batches,
                                            layer_indices=layer_indices)
        return value, {
            "prior": "auxiliary_ce",
            "formula": "mean auxiliary-data cross-entropy gradient at theta_bar",
            "sign": "raw_positive_gradient",
        }
    if prior == "l2_anchor":
        if not pretrained_checkpoint or not warmup_checkpoint:
            raise ValueError(
                "l2_anchor requires pretrained_checkpoint and warmup_checkpoint"
            )
        return build_l2_anchor_direction(
            pretrained_checkpoint,
            warmup_checkpoint,
            target_direction=target_direction,
            num_last_layers=num_last_layers,
            layer_indices=layer_indices,
            torch_dtype=torch_dtype,
        )
    raise ValueError(
        f"unknown retention prior {prior!r}; supported={SUPPORTED_RETENTION_PRIORS}"
    )


def streaming_direction_stats(
    left: torch.Tensor,
    right: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = 1_000_000,
) -> Dict[str, Any]:
    """Compute norms/dot/cosine in FP64 without full FP64 tensor copies."""
    left = left.detach().reshape(-1).cpu()
    if right is not None:
        right = right.detach().reshape(-1).cpu()
        if left.shape != right.shape:
            raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    sum_left = 0.0
    sum_left_sq = 0.0
    sum_right_sq = 0.0
    dot = 0.0
    nonzero = 0
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, left.numel(), chunk_size):
        l64 = left[start : start + chunk_size].double()
        sum_left += float(l64.sum().item())
        sum_left_sq += float(torch.dot(l64, l64).item())
        nonzero += int(torch.count_nonzero(l64).item())
        if l64.numel():
            minimum = min(minimum, float(l64.min().item()))
            maximum = max(maximum, float(l64.max().item()))
        if right is not None:
            r64 = right[start : start + chunk_size].double()
            sum_right_sq += float(torch.dot(r64, r64).item())
            dot += float(torch.dot(l64, r64).item())
    n = int(left.numel())
    norm = sum_left_sq ** 0.5
    mean = sum_left / max(n, 1)
    variance = max(0.0, sum_left_sq / max(n, 1) - mean * mean)
    result: Dict[str, Any] = {
        "numel": n,
        "dtype": str(left.dtype),
        "norm_fp64": norm,
        "mean_fp64": mean,
        "std_fp64": variance ** 0.5,
        "min": minimum,
        "max": maximum,
        "nonzero": nonzero,
        "finite": bool(torch.isfinite(left).all()),
    }
    if right is not None:
        right_norm = sum_right_sq ** 0.5
        result.update({
            "dot_fp64": dot,
            "right_norm_fp64": right_norm,
            "cosine_fp64": dot / (norm * right_norm) if norm and right_norm else 0.0,
        })
    return result


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(path: str) -> Dict[str, Any]:
    root = Path(path)
    files = sorted(
        candidate for candidate in root.iterdir()
        if candidate.is_file()
        and (
            candidate.name.endswith(".safetensors")
            or candidate.name in {"config.json", "model.safetensors.index.json"}
        )
    )
    if not files:
        raise FileNotFoundError(f"no checkpoint files found in {root}")
    entries = [
        {
            "name": candidate.name,
            "size": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        for candidate in files
    ]
    aggregate = hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"path": str(root), "files": entries, "manifest_sha256": aggregate}
