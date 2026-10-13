"""Build the fixed L2-mixed validation direction using existing checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from configs import load_config
from release_privacy import source_provenance
from directions.retention import (
    build_retention_direction,
    checkpoint_manifest,
    streaming_direction_stats,
)
from models.utils import combine_target_retention_directions
from score_cache import compute_validation_hash


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_snapshot() -> Dict[str, Any]:
    """Compatibility key: content provenance, never private Git metadata."""
    return {"branch": None, "commit": None, "status_porcelain": [],
            "source_provenance": source_provenance()}


def stage_direction(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if cfg.retention_prior != "l2_anchor":
        raise ValueError(f"expected retention_prior=l2_anchor, got {cfg.retention_prior}")
    if cfg.retention_grad_mode != "unit_mix":
        raise ValueError("formal L2-anchor direction requires unit_mix")
    if abs(cfg.retention_lambda - 0.2) > 1e-12:
        raise ValueError(
            f"only the formal lambda=0.2 is authorized, got {cfg.retention_lambda}"
        )
    root = Path(args.output_root)
    direction_dir = root / "direction"
    direction_dir.mkdir(parents=True, exist_ok=True)
    target_path = Path(args.target_direction)
    target = torch.load(target_path, map_location="cpu", weights_only=True).float()
    if target.ndim != 1 or not bool(torch.isfinite(target).all()):
        raise RuntimeError("target direction must be a finite one-dimensional tensor")

    started = time.time()


    d_layer_indices = None
    if cfg.direction_layer_strategy != "last_n_physical":
        from transformers import AutoConfig
        from models.arch_adapter import resolve_layer_indices_from_config
        _hf_cfg = AutoConfig.from_pretrained(args.warmup_checkpoint)
        d_layer_indices = resolve_layer_indices_from_config(
            _hf_cfg, cfg.direction_layer_strategy, cfg.direction_layer_indices,
            cfg.direction_num_last_layers,
        )
    anchor, metadata = build_retention_direction(
        "l2_anchor",
        pretrained_checkpoint=args.pretrained_checkpoint,
        warmup_checkpoint=args.warmup_checkpoint,
        target_direction=target,
        num_last_layers=cfg.direction_num_last_layers,
        layer_indices=d_layer_indices,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )
    raw_path = direction_dir / "l2_anchor_raw.pt"
    torch.save(anchor, raw_path)
    combined, mix_stats = combine_target_retention_directions(
        target, anchor, cfg.retention_lambda, cfg.retention_grad_mode
    )
    combined_path = direction_dir / "v_l2_ret_lambda_0p2.pt"
    torch.save(combined, combined_path)

    target_stats = streaming_direction_stats(target)
    anchor_stats = streaming_direction_stats(anchor, target)
    combined_stats = streaming_direction_stats(combined, target)
    if torch.equal(anchor, target):
        raise RuntimeError("L2 anchor unexpectedly equals target direction")
    if torch.equal(combined, target):
        raise RuntimeError("combined L2 direction unexpectedly equals target direction")
    if anchor_stats["norm_fp64"] <= 0.0:
        raise RuntimeError("L2 anchor norm is zero")
    if combined_stats["norm_fp64"] <= 0.0:
        raise RuntimeError("combined direction norm is zero")

    stats = {
        "status": "PASS",
        "direction_type": "retention_l2_anchor",
        "retention_prior": "l2_anchor",
        "retention_lambda": cfg.retention_lambda,
        "retention_grad_mode": cfg.retention_grad_mode,
        "direction_layer_strategy": cfg.direction_layer_strategy,
        "direction_layer_indices": d_layer_indices,
        "direction_num_last_layers": cfg.direction_num_last_layers,
        "formula": (
            "v=(1-lambda)*normalize(g_target)+lambda*normalize(theta_bar-theta_0)"
        ),
        "sign": "theta_bar-theta_0",
        "target_direction": {
            "path": str(target_path),
            "sha256": compute_validation_hash(target),
            **target_stats,
        },
        "l2_anchor": {
            "path": str(raw_path),
            "sha256": compute_validation_hash(anchor),
            **anchor_stats,
        },
        "combined_direction": {
            "path": str(combined_path),
            "sha256": compute_validation_hash(combined),
            **combined_stats,
        },
        "norm_ratio_l2_to_target": (
            anchor_stats["norm_fp64"] / target_stats["norm_fp64"]
        ),
        "mix_stats": mix_stats,
        "parameter_metadata": metadata,
        "checkpoint_manifests": {
            "pretrained": checkpoint_manifest(args.pretrained_checkpoint),
            "warmup": checkpoint_manifest(args.warmup_checkpoint),
        },
        "git": git_snapshot(),
        "elapsed_sec": time.time() - started,
    }
    write_json(stats, direction_dir / "l2_anchor_stats.json")
    report = [
        "# L2-anchor direction report",
        "",
        "- Status: **PASS**",
        f"- Raw anchor: `{raw_path}`",
        f"- Combined direction: `{combined_path}`",
        f"- Parameter dimension: {anchor.numel():,}",
        f"- `||g_target||` (FP64): {target_stats['norm_fp64']:.8g}",
        f"- `||g_L2||` (FP64): {anchor_stats['norm_fp64']:.8g}",
        f"- Norm ratio L2/target: {stats['norm_ratio_l2_to_target']:.8g}",
        f"- `cos(g_L2,g_target)` (FP64): {anchor_stats['cosine_fp64']:.8g}",
        f"- `cos(v_mix,g_target)` (FP64): {combined_stats['cosine_fp64']:.8g}",
        "- Difference computed in FP32; norms, dot products, and cosines use "
        "chunked FP64 accumulation.",
        "- Parameter names, order, shapes, total dimension, tied aliases and "
        "frozen flags were checked before subtraction.",
        "",
    ]
    (direction_dir / "direction_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    write_json({
        "command": " ".join(sys.argv),
        "config": str(Path(args.config)),
        "git": stats["git"],
        "artifacts": {
            "raw": stats["l2_anchor"],
            "combined": stats["combined_direction"],
        },
        "checkpoint_manifests": stats["checkpoint_manifests"],
    }, root / "reproducibility.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    direction = stages.add_parser("l2-direction")
    direction.add_argument("--config", required=True)
    direction.add_argument("--output-root", required=True)
    direction.add_argument("--pretrained-checkpoint", required=True)
    direction.add_argument("--warmup-checkpoint", required=True)
    direction.add_argument("--target-direction", required=True)
    return parser


def main() -> None:
    stage_direction(build_parser().parse_args())


if __name__ == "__main__":
    main()
