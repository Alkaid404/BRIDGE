"v3 pipeline driver: strict data isolation stages."
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from configs import Config, load_config
from release_privacy import source_provenance
from selection.alternating import eligible_budget
from data import BridgeDataset, load_data
from pipeline.common import (
    hash_id_list,
    read_json,
    sha256_file,
    special_positions,
    write_json,
)
from utils import get_logger, set_seed, seed_worker

logger = get_logger("bridge.v3")

V3_ROOT = Path("outputs_metamath_math_clean_v3")


def load_split(split_dir: str) -> dict:
    split_dir = Path(split_dir)
    return {
        "manifest": read_json(split_dir / "split_manifest.json"),
        "warmup_ids": read_json(split_dir / "warmup_ids.json"),
        "validation_ids": read_json(split_dir / "validation_ids.json"),
        "sft_ids": read_json(split_dir / "sft_ids.json"),
        "map": read_json(split_dir / "sample_split_map.json"),
    }


def load_all_samples(cfg: Config):
    """Reconstruct the converted dataset associated with the saved split IDs."""
    from data import load_samples
    return load_samples(cfg.dataset_name, cfg.dataset_path, split=None)


def split_samples(samples, split, which: str):
    wanted = set(split[f"{which}_ids"])
    out = [s for s in samples if str(s["id"]) in wanted]
    got = {str(s["id"]) for s in out}
    if got != wanted:
        raise RuntimeError(
            f"{which}: manifest/data mismatch: missing={len(wanted - got)} extra={len(got - wanted)}"
        )
    return out


def filter_samples_by_id_manifest(samples, manifest_path):
    """Select an exact ordered audit subset with strict ID coverage."""
    if not manifest_path:
        return samples
    requested = [str(value) for value in read_json(manifest_path)]
    if not requested:
        raise RuntimeError(f"sample-ID manifest is empty: {manifest_path}")
    if len(requested) != len(set(requested)):
        raise RuntimeError(f"sample-ID manifest contains duplicates: {manifest_path}")
    by_id = {str(sample["id"]): sample for sample in samples}
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise RuntimeError(
            f"sample-ID manifest has {len(missing)} IDs outside the active split, "
            f"e.g. {missing[:5]}"
        )
    return [by_id[sample_id] for sample_id in requested]


def assert_disjoint(name, ids_a, ids_b):
    inter = set(ids_a) & set(ids_b)
    if inter:
        raise RuntimeError(f"ISOLATION VIOLATION: {name}: {len(inter)} shared ids, e.g. {sorted(inter)[:5]}")


def training_device_map():
    """Let Accelerate/FSDP place distributed training ranks; shard only single-process jobs."""
    return None if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"


def build_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def build_dataset(samples, tokenizer, cfg: Config, split: str):
    return BridgeDataset(
        samples, tokenizer, cfg.max_seq_length, cfg.max_seq_length, split=split,
        special_token_mode=cfg.special_token_mode,
    )


def tokenizer_hash(tokenizer, cfg: Config) -> str:
    from structure.mask_artifacts import tokenizer_fingerprint
    return tokenizer_fingerprint(tokenizer, cfg.max_seq_length)


def checkpoint_sha256(ckpt_dir: str) -> str:
    p = Path(ckpt_dir) / "model.safetensors"
    if p.is_file():
        return sha256_file(p)

    files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    if not files:
        return ""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def free_indices(records, mode: str, has_bos: bool = True):
    fp_list, fr_list = [], []
    for rec in records:
        pos = special_positions(int(rec.prompt_length), int(rec.response_length), mode,
                                has_bos=has_bos)
        fp_list.append(pos["prompt_free"])
        fr_list.append(pos["response_free"])
    return fp_list, fr_list


def load_stage_config(args) -> Config:
    overrides = {}
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides=overrides or None)
    if cfg.expected_split_sizes is not None and getattr(args, "split_dir", None):
        from scripts.reproduce import validate_split_contract
        validate_split_contract(cfg, Path(args.split_dir))


    if cfg.retention_enabled and getattr(args, "stage", None) == "direction":
        raise RuntimeError("v3 direction computes target-only direction; disable retention here, then use reproduce.py l2-direction for L2 mixing")
    return cfg


def stage_warmup(args):
    from models.utils import BridgeModel, log_full_ft_setup
    from training import TrainingMode
    from pipeline.common import _train_stage

    cfg = load_stage_config(args)
    split = load_split(args.split_dir)
    samples = split_samples(load_all_samples(cfg), split, "warmup")
    assert_disjoint("warmup∩validation", [s["id"] for s in samples], split["validation_ids"])
    assert_disjoint("warmup∩sft", [s["id"] for s in samples], split["sft_ids"])
    if args.max_samples:
        samples = samples[: args.max_samples]
    actual_ids = [str(s["id"]) for s in samples]
    logger.info(f"Stage 1 warmup: {len(samples)} samples (D_warmup only)")

    tokenizer = build_tokenizer(cfg)
    dataset = build_dataset(samples, tokenizer, cfg, "warmup")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = BridgeModel(
        cfg.model_name_or_path, torch_dtype=dtype, device_map=training_device_map()
    )
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    log_full_ft_setup(model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_out = _train_stage(
        model, tokenizer, cfg, dataset, None,
        TrainingMode.MASK_ONLY, method="sft",
        output_dir=str(out_dir),
        epochs=args.epochs or cfg.warmup_epochs,
        lr=cfg.warmup_learning_rate,
        batch_size=cfg.warmup_batch_size,
        grad_accum=cfg.warmup_gradient_accumulation_steps,
        save_strategy="no",
        save_final_model=not args.skip_final_save,
    )
    elapsed = time.time() - t0
    record = {
        "stage": "warmup",
        "actual_training_ids": actual_ids,
        "actual_training_ids_hash": hash_id_list(actual_ids),
        "warmup_ids_manifest_hash": split["manifest"]["hashes"]["warmup_ids"],
        "assert_ids_equal_manifest": (
            hash_id_list(actual_ids) == hash_id_list([str(i) for i in split["warmup_ids"][: len(actual_ids)]])
            if args.max_samples else hash_id_list(actual_ids) == split["manifest"]["hashes"]["warmup_ids"]
        ),
        "epochs": args.epochs or cfg.warmup_epochs,
        "optimizer": {"name": "adamw_torch", "lr": cfg.warmup_learning_rate},
        "effective_global_batch": cfg.warmup_batch_size * cfg.warmup_gradient_accumulation_steps,
        "checkpoint": train_out["model_path"],
        "checkpoint_sha256": checkpoint_sha256(train_out["model_path"]) if train_out["model_path"] else None,
        "tokenizer_hash": tokenizer_hash(tokenizer, cfg),
        "seed": cfg.seed,
        "special_token_mode": cfg.special_token_mode,
        "elapsed_sec": elapsed,
        "final_loss": train_out["history"][-1].get("train_loss") if train_out["history"] else None,
    }


    if int(os.environ.get("RANK", "0")) == 0:
        write_json(record, out_dir / "warmup_record.json")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.barrier()
    logger.info(f"Stage 1 done in {elapsed/60:.1f} min -> {train_out['model_path']}")


def stage_direction(args):
    from models.utils import BridgeModel, compute_validation_direction
    from data import BridgeDataCollator

    cfg = load_stage_config(args)
    split = load_split(args.split_dir)
    samples = split_samples(load_all_samples(cfg), split, "validation")
    assert_disjoint("validation∩warmup", [s["id"] for s in samples], split["warmup_ids"])
    assert_disjoint("validation∩sft", [s["id"] for s in samples], split["sft_ids"])
    if args.max_samples:
        samples = samples[: args.max_samples]
    actual_ids = [str(s["id"]) for s in samples]
    logger.info(f"Stage 2 direction: {len(samples)} samples (D_validation only, retention OFF)")

    tokenizer = build_tokenizer(cfg)
    dataset = build_dataset(samples, tokenizer, cfg, "validation")
    dataset.tokenize_all()
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = BridgeModel(args.warmup_checkpoint, torch_dtype=dtype, device_map="auto")
    model.eval()
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    loader = DataLoader(dataset, batch_size=getattr(cfg, "direction_batch_size", 2), shuffle=False,
                        collate_fn=BridgeDataCollator(tokenizer), worker_init_fn=seed_worker)

    from models.arch_adapter import resolve_layer_indices
    direction_layer_indices = resolve_layer_indices(
        model, cfg.direction_layer_strategy, cfg.direction_layer_indices,
        cfg.direction_num_last_layers,
    )
    logger.info(
        f"Stage 2 direction layer subset: strategy={cfg.direction_layer_strategy} "
        f"indices={direction_layer_indices}"
    )
    t0 = time.time()
    direction = compute_validation_direction(model, loader, layer_indices=direction_layer_indices)
    elapsed = time.time() - t0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direction.detach().cpu(), str(out_dir / "validation_direction.pt"))
    from score_cache import compute_validation_hash
    record = {
        "stage": "validation_direction",
        "direction_type": "metamath_validation_only",
        "retention_aware": False,
        "lambda_ret": 0.0,
        "retention_dataset": None,
        "retention_gradient_path": None,
        "combine_target_retention_directions": False,
        "actual_validation_ids": actual_ids,
        "actual_validation_ids_hash": hash_id_list(actual_ids),
        "validation_ids_manifest_hash": split["manifest"]["hashes"]["validation_ids"],
        "direction_norm": float(direction.norm()),
        "direction_sha256": compute_validation_hash(direction),
        "direction_layer_strategy": cfg.direction_layer_strategy,
        "direction_layer_indices": direction_layer_indices,
        "direction_num_last_layers": cfg.direction_num_last_layers,
        "warmup_checkpoint": args.warmup_checkpoint,
        "elapsed_sec": elapsed,
    }
    write_json(record, out_dir / "direction_record.json")
    logger.info(f"Stage 2 done in {elapsed/60:.1f} min, norm={float(direction.norm()):.4f}")


def stage_scoring(args):
    from models.utils import BridgeModel
    from scoring.pipeline import run_scoring
    from score_cache import compute_validation_hash, compute_config_hash

    cfg = load_stage_config(args)
    split = load_split(args.split_dir)
    samples = split_samples(load_all_samples(cfg), split, "sft")
    samples = filter_samples_by_id_manifest(samples, args.sample_ids_path)
    assert_disjoint("sft∩warmup", [s["id"] for s in samples], split["warmup_ids"])
    assert_disjoint("sft∩validation", [s["id"] for s in samples], split["validation_ids"])
    if args.max_samples:
        samples = samples[: args.max_samples]
    sft_ids = [str(s["id"]) for s in samples]
    logger.info(f"Stage 3 scoring: {len(samples)} samples (D_sft only)")

    tokenizer = build_tokenizer(cfg)
    dataset = build_dataset(samples, tokenizer, cfg, "train")
    direction = torch.load(args.direction_path, map_location="cpu", weights_only=True).float()
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    d_idx = None
    if cfg.direction_layer_strategy != "last_n_physical" or cfg.direction_num_last_layers != 4:
        from transformers import AutoConfig
        from models.arch_adapter import resolve_layer_indices_from_config
        d_idx = resolve_layer_indices_from_config(
            AutoConfig.from_pretrained(args.warmup_checkpoint),
            cfg.direction_layer_strategy, cfg.direction_layer_indices,
            cfg.direction_num_last_layers,
        )

    t0 = time.time()
    model = None
    if args.num_gpus <= 1:
        dataset.tokenize_all()
        model = BridgeModel(args.warmup_checkpoint, torch_dtype=dtype, device_map="auto")
        model.eval()
    results = run_scoring(
        model, tokenizer, direction, dataset, cfg,
        model_path=cfg.model_name_or_path,
        checkpoint_path=args.warmup_checkpoint,
        num_gpus=args.num_gpus,
        direction_path=args.direction_path,
        worker_output_dir=str(out_dir),
        resume_key=f"v3_{hash_id_list(sft_ids)[:12]}",
    )
    elapsed = time.time() - t0


    records = results["samples"]
    if len(records) != len(sft_ids):
        raise RuntimeError(f"scored {len(records)} != D_sft {len(sft_ids)}")
    for i, rec in enumerate(records):
        rec.metadata["sample_id"] = sft_ids[i]
    score_ids = [rec.metadata["sample_id"] for rec in records]
    if score_ids != sft_ids:
        raise RuntimeError("score_sample_ids != D_sft_ids (order/content mismatch)")
    assert_disjoint("score∩warmup", score_ids, split["warmup_ids"])
    assert_disjoint("score∩validation", score_ids, split["validation_ids"])

    meta = {
        "schema_version": "v3.1",
        "git_commit": None,
        "source_provenance": source_provenance(),
        "model_checkpoint": args.warmup_checkpoint,
        "model_checkpoint_sha256": checkpoint_sha256(args.warmup_checkpoint),
        "tokenizer_hash": tokenizer_hash(tokenizer, cfg),
        "special_token_mode": cfg.special_token_mode,
        "full_dataset_hash": split["manifest"]["hashes"]["full_dataset"],
        "warmup_split_hash": split["manifest"]["hashes"]["warmup_ids"],
        "validation_split_hash": split["manifest"]["hashes"]["validation_ids"],
        "sft_split_hash": split["manifest"]["hashes"]["sft_ids"],
        "warmup_ids_hash": split["manifest"]["hashes"]["warmup_ids"],
        "validation_ids_hash": split["manifest"]["hashes"]["validation_ids"],
        "sft_ids_hash": hash_id_list(sft_ids),
        "validation_direction_hash": compute_validation_hash(direction),


        "attention_capture_version": results.get("capture_version", "legacy_v1"),
        "scoring_attention_layer_strategy": cfg.scoring_attention_layer_strategy,
        "scoring_layer_info": results.get("scoring_layer_info"),
        "bos_handling": results.get("bos_handling", cfg.bos_handling),
        "tokenizer_special_ids": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "direction_type": (
            f"retention_{cfg.retention_prior}"
            if cfg.retention_enabled else "metamath_validation_only"
        ),
        "retention_aware": bool(cfg.retention_enabled),
        "retention_prior": (
            cfg.retention_prior if cfg.retention_enabled else None
        ),
        "retention_lambda": (
            cfg.retention_lambda if cfg.retention_enabled else 0.0
        ),
        "retention_grad_mode": (
            cfg.retention_grad_mode if cfg.retention_enabled else None
        ),
        "scoring_backend": cfg.scoring_backend,
        "scored_layer_range": (
            "last_4" if d_idx is None else f"layers_{d_idx[0]}_{d_idx[-1]}"
        ),
        "direction_layer_strategy": cfg.direction_layer_strategy,
        "direction_layer_indices": d_idx,
        "attention_aggregation": cfg.attention_aggregation,
        "special_token_scoring": cfg.special_token_scoring,
        "special_token_budget": cfg.special_token_budget,
        "special_token_attention_policy": "mask_no_renorm" if cfg.special_token_scoring == "exclude" else "none",
        "max_seq_length": cfg.max_seq_length,
        "dtype": "bfloat16" if cfg.bf16 else "float16",
        "seed": cfg.seed,
        "config_hash": compute_config_hash(cfg),
        "num_samples": len(records),
        "sample_ids": sft_ids,
        "elapsed_sec": elapsed,
    }
    payload = {"samples": records, "metadata": [r.metadata for r in records], "meta": meta}
    out_path = out_dir / "scoring_results.pt"
    torch.save(payload, str(out_path))
    write_json({k: v for k, v in meta.items() if k != "sample_ids"},
               out_dir / "scoring_meta.json")
    logger.info(f"Stage 3 done in {elapsed/60:.1f} min -> {out_path}")


def _uniform_random(records, pr, rr, seed, mode, has_bos: bool = True, budget_rounding="round"):
    fp_list, fr_list = free_indices(records, mode, has_bos=has_bos)
    rng = random.Random(seed)
    S_p_list, S_r_list = [], []
    for rec, fp, fr in zip(records, fp_list, fr_list):
        P, R = int(rec.prompt_length), int(rec.response_length)
        cand_p = [i for i in range(P) if i not in set(fp)]
        cand_r = [i for i in range(R) if i not in set(fr)]
        bp = eligible_budget(pr, len(cand_p), budget_rounding)
        br = eligible_budget(rr, len(cand_r), budget_rounding)
        S_p = sorted(rng.sample(cand_p, bp)) + list(fp)
        S_r = sorted(rng.sample(cand_r, br)) + list(fr)
        S_p_list.append(torch.tensor(sorted(S_p), dtype=torch.long))
        S_r_list.append(torch.tensor(sorted(S_r), dtype=torch.long))
    return S_p_list, S_r_list


def _span_random(records, pr, rr, seed, mode, max_spans=5, has_bos: bool = True, budget_rounding="round"):
    fp_list, fr_list = free_indices(records, mode, has_bos=has_bos)
    rng = random.Random(seed)
    S_p_list, S_r_list = [], []

    def spans(cand, k):
        if k <= 0 or not cand:
            return []
        n_spans = max(1, min(max_spans, k // 8 or 1))
        base = k // n_spans
        lengths = [base] * n_spans
        for i in range(k - base * n_spans):
            lengths[i % n_spans] += 1
        chosen, out = set(), []
        cand_set = set(cand)
        lo, hi = cand[0], cand[-1]
        for L in lengths:
            for _ in range(20):
                start = rng.randint(lo, max(lo, hi - L + 1))
                span = [x for x in range(start, start + L) if x in cand_set and x not in chosen]
                if len(span) >= max(1, L // 2):
                    break
            for s in span:
                if len(out) < k:
                    chosen.add(s)
                    out.append(s)
        if len(out) < k:
            pool = [x for x in cand if x not in chosen]
            out.extend(rng.sample(pool, min(k - len(out), len(pool))))
        return out[:k]

    for rec, fp, fr in zip(records, fp_list, fr_list):
        P, R = int(rec.prompt_length), int(rec.response_length)
        cand_p = [i for i in range(P) if i not in set(fp)]
        cand_r = [i for i in range(R) if i not in set(fr)]
        bp = eligible_budget(pr, len(cand_p), budget_rounding)
        br = eligible_budget(rr, len(cand_r), budget_rounding)
        S_p = spans(cand_p, bp) + list(fp)
        S_r = spans(cand_r, br) + list(fr)
        S_p_list.append(torch.tensor(sorted(S_p), dtype=torch.long))
        S_r_list.append(torch.tensor(sorted(S_r), dtype=torch.long))
    return S_p_list, S_r_list


def stage_select(args):
    from selection import select_fixed
    from structure.pipeline import apply_structured_selection

    cfg = load_stage_config(args)
    split = load_split(args.split_dir)
    samples = split_samples(load_all_samples(cfg), split, "sft")
    samples = filter_samples_by_id_manifest(samples, args.sample_ids_path)
    if args.max_samples:
        samples = samples[: args.max_samples]
    payload = torch.load(args.scores, map_location="cpu", weights_only=False)
    records = payload["samples"][: len(samples)]
    meta = payload.get("meta") or {}

    if meta.get("sft_ids_hash") and meta["sft_ids_hash"] != hash_id_list([str(s["id"]) for s in samples]):
        raise RuntimeError("cached sft_ids_hash != current D_sft hash; refusing to use cache")
    if meta.get("special_token_mode") and meta["special_token_mode"] != cfg.special_token_mode:
        raise RuntimeError("cache special_token_mode mismatch")
    tokenizer = build_tokenizer(cfg)
    dataset = build_dataset(samples, tokenizer, cfg, "train")
    dataset.tokenize_all()
    mode = cfg.special_token_mode
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    has_bos = getattr(tokenizer, "bos_token_id", None) is not None
    fp_list, fr_list = free_indices(records, mode, has_bos=has_bos)
    selections = {}

    t0 = time.time()

    S_p_score, S_r_score = select_fixed(
        records, cfg.prompt_keep_ratio, cfg.response_keep_ratio, T_alt=cfg.T_alt,
        prompt_free_indices=fp_list, response_free_indices=fr_list,
        budget_rounding=cfg.budget_rounding,
    )
    selections["bridge_score75"] = (S_p_score, S_r_score)

    eos_list = [fr[0] if fr else None for fr in fr_list]
    S_r_for_closure = [
        torch.tensor([v for v in s.tolist() if v != eos], dtype=torch.long)
        if eos is not None else s
        for s, eos in zip(S_r_score, eos_list)
    ]
    S_r_closure, structure_details = apply_structured_selection(
        records=records,
        train_samples=samples,
        tokenizer=tokenizer,
        train_dataset=dataset,
        prompt_selection=S_p_score,
        response_selection=S_r_for_closure,
        max_response_length=cfg.max_seq_length,
        structure_config=cfg.structure,
        output_dir=out_dir / "closure_diagnostics",
        structure_protection=cfg.structure_protection,
        special_token_mode=cfg.special_token_mode,
    )
    S_r_closure = [
        torch.tensor(sorted(set(s.tolist()) | {eos}), dtype=torch.long)
        if eos is not None else s
        for s, eos in zip(S_r_closure, eos_list)
    ]
    selections["bridge_closure75"] = (S_p_score, S_r_closure)

    selections["random75"] = _uniform_random(records, cfg.prompt_keep_ratio,
                                             cfg.response_keep_ratio, cfg.seed, mode,
                                             has_bos=has_bos, budget_rounding=cfg.budget_rounding)
    selections["span_random75"] = _span_random(records, cfg.prompt_keep_ratio,
                                               cfg.response_keep_ratio, cfg.seed, mode,
                                               has_bos=has_bos, budget_rounding=cfg.budget_rounding)
    elapsed = time.time() - t0


    from pipeline.common import selection_mask_hash
    saved = {}
    for name, (S_p, S_r) in selections.items():
        if len(S_p) != len(records) or len(S_r) != len(records):
            raise RuntimeError(f"{name}: selection/sample count mismatch")
        for i, rec in enumerate(records):
            P, R = int(rec.prompt_length), int(rec.response_length)
            if any(v < 0 or v >= P for v in S_p[i].tolist()):
                raise RuntimeError(f"{name}[{i}]: S_p out of range")
            if any(v < 0 or v >= R for v in S_r[i].tolist()):
                raise RuntimeError(f"{name}[{i}]: S_r out of range")
        saved[name] = {
            "S_p_list": S_p, "S_r_list": S_r,
            "mask_hash": selection_mask_hash(S_r),
            "prompt_keep_ratio": cfg.prompt_keep_ratio,
            "response_keep_ratio": cfg.response_keep_ratio,
            "budget_rounding": cfg.budget_rounding,
            "sample_ids": [str(sample["id"]) for sample in samples],
            "sft_ids_hash": hash_id_list([str(sample["id"]) for sample in samples]),
        }
        torch.save(saved[name], str(out_dir / f"selection_{name}.pt"))
    if structure_details is not None:
        write_json(structure_details["summary"], out_dir / "closure_summary.json")
    write_json({
        "methods": {k: {"mask_hash": v["mask_hash"]} for k, v in saved.items()},
        "elapsed_sec": elapsed,
        "score_source": str(args.scores),
        "budget_rounding": cfg.budget_rounding,
        "attention_capture_version": meta.get("attention_capture_version", "legacy_v1"),
        "sft_ids_hash": hash_id_list([str(s["id"]) for s in samples]),
    }, out_dir / "selection_record.json")
    logger.info(f"Stage 4 done in {elapsed/60:.1f} min: {list(selections)}")


def validate_training_selection(sel, actual_ids, cfg):
    """Bind static masks to split order and integer-budget convention."""
    count = len(actual_ids)
    S_p, S_r = sel["S_p_list"][:count], sel["S_r_list"][:count]
    if len(S_p) != count or len(S_r) != count:
        raise RuntimeError("selection/training sample count mismatch")
    if sel.get("sample_ids") is not None:
        if list(map(str, sel["sample_ids"][:count])) != actual_ids:
            raise RuntimeError("selection/training sample ID order mismatch")
    elif cfg.require_selection_provenance or cfg.expected_split_sizes is not None:
        raise RuntimeError("Paper-protocol selection requires stable sample IDs; "
                           "do not relabel an unverified historical mask")
    else:
        logger.warning("Historical selection has no sample IDs; verify its split/order provenance")
    if sel.get("budget_rounding", "round") != cfg.budget_rounding:
        raise RuntimeError("selection/training budget rounding mismatch")
    return S_p, S_r


def stage_train(args):
    from models.utils import BridgeModel, log_full_ft_setup
    from training import (
        BridgeCollator, BridgeTrainer, SelectedDataset, TrainingMode, build_training_args,
    )

    cfg = load_stage_config(args)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError(f"--epochs must be positive, got {args.epochs}")
        cfg.num_epochs = args.epochs
    if cfg.special_token_scoring != "exclude" or cfg.special_token_budget != "fixed_free":
        logger.warning("v3 expects special_token_scoring=exclude + fixed_free")
    split = load_split(args.split_dir)
    samples = split_samples(load_all_samples(cfg), split, "sft")
    if args.max_samples:
        samples = samples[: args.max_samples]
    actual_ids = [str(s["id"]) for s in samples]
    assert_disjoint("train∩warmup", actual_ids, split["warmup_ids"])
    assert_disjoint("train∩validation", actual_ids, split["validation_ids"])
    method = args.method
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg)
    dataset = build_dataset(samples, tokenizer, cfg, "train")
    selection_results = None
    mode = TrainingMode.MASK_ONLY
    if method != "full_sft":
        sel = torch.load(args.selection, map_location="cpu", weights_only=False)
        S_p, S_r = validate_training_selection(sel, actual_ids, cfg)
        selection_results = (S_p, S_r)
        mode = TrainingMode.COMPACT_PROMPT_SELECTED_LOSS

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    model = BridgeModel(
        cfg.model_name_or_path, torch_dtype=dtype, device_map=training_device_map()
    )


    trainable_stats = None
    trainable_indices = None
    if cfg.trainable_layer_strategy != "full":
        from models.arch_adapter import resolve_layer_indices
        from models.utils import apply_trainable_layer_restriction
        trainable_indices = resolve_layer_indices(
            model, cfg.trainable_layer_strategy, cfg.trainable_layer_indices,
            cfg.direction_num_last_layers,
        )
        trainable_stats = apply_trainable_layer_restriction(model, trainable_indices)


        if hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()
            logger.info("enable_input_require_grads() applied for frozen-layer training")
        logger.info(
            f"Stage 5 trainable layer subset: strategy={cfg.trainable_layer_strategy} "
            f"indices={trainable_indices} "
            f"trainable={trainable_stats['trainable_params']:,}/{trainable_stats['total_params']:,}"
        )
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    log_full_ft_setup(model)

    train_ds = SelectedDataset(
        dataset, training_mode=mode, selection_results=selection_results,
        prompt_order=cfg.prompt_selection_order,
        explicit_response_mask=(method != "full_sft") and cfg.explicit_response_mask,
        always_supervise_eos=cfg.always_supervise_eos,
    )
    targs = build_training_args(cfg, output_dir=str(out_dir))
    collator = BridgeCollator(pad_token_id=tokenizer.pad_token_id or 0)
    trainer = BridgeTrainer(
        model=model, tokenizer=tokenizer, config=cfg,
        training_mode=mode, method="sft" if method == "full_sft" else "bridge",
        args=targs, train_dataset=train_ds, eval_dataset=None,
        data_collator=collator,
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    final_dir = out_dir / "final_model"
    if not args.skip_final_save:
        trainer.save_model(str(final_dir))
    history = list(trainer.state.log_history)
    record = {
        "method": method,
        "stage": "final_train",
        "init": "theta_old (pretrained base)",
        "actual_training_ids": actual_ids,
        "actual_training_ids_hash": hash_id_list(actual_ids),
        "sft_ids_manifest_hash": split["manifest"]["hashes"]["sft_ids"],
        "selection": str(args.selection) if method != "full_sft" else None,
        "training_mode": mode.value,
        "explicit_response_mask": bool(method != "full_sft" and cfg.explicit_response_mask),
        "strict_loss": bool(cfg.strict_loss),
        "prompt_selection_order": cfg.prompt_selection_order,
        "special_token_mode": cfg.special_token_mode,
        "trainable_layer_strategy": cfg.trainable_layer_strategy,
        "trainable_layer_indices": trainable_indices,
        "trainable_layer_stats": trainable_stats,
        "epochs": cfg.num_epochs,
        "global_batch": cfg.batch_size * cfg.gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", "1")),
        "lr": cfg.learning_rate,
        "seed": cfg.seed,
        "checkpoint": str(final_dir) if not args.skip_final_save else None,
        "checkpoint_sha256": checkpoint_sha256(str(final_dir)) if not args.skip_final_save else None,
        "tokenizer_hash": tokenizer_hash(tokenizer, cfg),
        "final_loss": history[-1].get("train_loss") if history else None,
        "history_tail": history[-5:],
        "elapsed_sec": elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
    }


    if int(os.environ.get("RANK", "0")) == 0:
        write_json(record, out_dir / "train_record.json")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.barrier()
    logger.info(f"Stage 5 [{method}] done in {elapsed/60:.1f} min")


def stage_eval(args):
    from models.utils import BridgeModel
    from evaluation import (
        extract_math_answer, compare_answers, prepare_model_for_evaluation,
    )
    import pandas as pd

    cfg = load_stage_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(cfg)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = BridgeModel(args.checkpoint, torch_dtype=dtype, device_map="auto",
                        inference_only=True)
    prepare_model_for_evaluation(model)
    device = model.input_device
    results = {
        "provenance": {
            "seed": int(cfg.seed),
            "config": str(Path(args.config)),
            "checkpoint": str(Path(args.checkpoint)),
            "temperature": float(cfg.temperature),
            "math_decoding": "greedy",
            "aux_generation_decoding": (
                "greedy" if float(cfg.temperature) == 0.0 else "sampling"
            ),
            "multiple_choice_scoring": "candidate_log_likelihood",
            "deterministic_reproducibility_run": float(cfg.temperature) == 0.0,
        }
    }


    math_rows = []
    for f in sorted(Path(args.math_test_dir).glob("test-*.parquet")):
        math_rows.extend(pd.read_parquet(f).to_dict("records"))
    if args.max_eval_samples:
        math_rows = math_rows[: args.max_eval_samples]
    per_subject = {}
    math_records = []
    n_ok = n_exact = n_extract = n_eos = n_trunc = 0
    gen_lens = []
    B = cfg.eval_batch_size
    t0 = time.time()
    for start in range(0, len(math_rows), B):
        chunk = math_rows[start:start + B]
        prompts = [
            f"Solve the following math competition problem ({row['type']}).\n\n"
            f"Problem: {row['problem'].strip()}\n\nSolution:"
            for row in chunk
        ]
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=2048,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.model.generate(
                **enc, max_new_tokens=cfg.max_gen_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for i, row in enumerate(chunk):
            gen_ids = out[i][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            stopped_eos = bool(len(gen_ids) and int(gen_ids[-1]) == tokenizer.eos_token_id)
            truncated = len(gen_ids) >= cfg.max_gen_tokens and not stopped_eos
            extracted = extract_math_answer(text)
            ref = extract_math_answer(str(row["solution"]))
            ok = bool(extracted) and bool(ref) and compare_answers(extracted, ref, "math")
            exact = extracted.strip() == ref.strip() if extracted and ref else False
            subj = str(row["type"])
            st = per_subject.setdefault(subj, {"n": 0, "ok": 0})
            st["n"] += 1
            st["ok"] += int(ok)
            n_ok += int(ok)
            n_exact += int(exact)
            n_extract += int(bool(extracted))
            n_eos += int(stopped_eos)
            n_trunc += int(truncated)
            gen_lens.append(len(gen_ids))
            math_records.append({
                "idx": start + i, "type": subj, "level": str(row["level"]),
                "prompt": prompts[i], "raw_generation": text,
                "extracted_answer": extracted, "reference_answer": ref,
                "equivalent": ok, "exact_match": exact,
                "failure_type": ("ok" if ok else ("no_extraction" if not extracted else "wrong_answer")),
            })
        if (start // B) % 10 == 0:
            logger.info(f"MATH eval {start}/{len(math_rows)}")
    n = len(math_rows)
    results["math"] = {
        "official_equivalence_accuracy": n_ok / n,
        "exact_match_accuracy": n_exact / n,
        "answer_extraction_success": n_extract / n,
        "natural_eos_rate": n_eos / n,
        "truncation_rate": n_trunc / n,
        "mean_generation_length": sum(gen_lens) / len(gen_lens),
        "per_subject": {k: {"accuracy": v["ok"] / v["n"], "n": v["n"]}
                        for k, v in sorted(per_subject.items())},
        "n": n,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "math_records.jsonl", "w", encoding="utf-8") as fh:
        for r in math_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


    if not args.skip_aux:
        from evaluation import evaluate_alignment, evaluate_reasoning, evaluate_code
        aux_cfg = cfg
        if cfg.alignment_benchmarks:
            try:
                results["alignment"] = evaluate_alignment(
                    model, tokenizer, aux_cfg, response_dir=out_dir / "alignment")
            except Exception as e:
                results["alignment_error"] = repr(e)
        if cfg.reasoning_benchmarks:
            try:
                results["reasoning"] = evaluate_reasoning(
                    model, tokenizer, aux_cfg, response_dir=out_dir / "reasoning")
            except Exception as e:
                results["reasoning_error"] = repr(e)
        if cfg.code_benchmarks:
            try:
                results["code"] = evaluate_code(
                    model, tokenizer, aux_cfg, response_dir=out_dir / "code")
            except Exception as e:
                results["code_error"] = repr(e)

    write_json(results, out_dir / "eval_results.json")
    logger.info(f"Eval done: MATH={results['math']['official_equivalence_accuracy']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["warmup", "direction", "scoring", "select", "train", "eval"])
    p.add_argument("--config", required=True)
    p.add_argument("--split-dir", default=str(V3_ROOT / "data_split"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warmup-checkpoint", default=None)
    p.add_argument("--direction-path", default=None)
    p.add_argument("--scores", default=None)
    p.add_argument("--selection", default=None)
    p.add_argument("--method", default=None,
                   choices=["full_sft", "random75", "span_random75",
                            "bridge_score75", "bridge_closure75"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--math-test-dir", default=str(Path(os.environ.get("BRIDGE_DATA_ROOT", "data")) / "math"))
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--sample-ids-path", default=None,
        help="Exact ordered JSON list of IDs to select from the active split",
    )
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed and record it in evaluation provenance",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip-final-save", action="store_true")
    p.add_argument("--skip-aux", action="store_true")
    args = p.parse_args()

    effective_seed = (
        args.seed
        if args.seed is not None
        else load_config(args.config).seed
    )
    set_seed(effective_seed)
    if args.stage == "warmup":
        stage_warmup(args)
    elif args.stage == "direction":
        stage_direction(args)
    elif args.stage == "scoring":
        stage_scoring(args)
    elif args.stage == "select":
        stage_select(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "eval":
        stage_eval(args)


if __name__ == "__main__":
    main()
