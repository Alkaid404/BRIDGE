"Shared provenance, special-token rules and training helpers."
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch

from configs import Config
from evaluation import prepare_model_for_evaluation
from training import BridgeTrainer, SelectedDataset, TrainingMode, build_training_args
from utils import clear_gpu_cache, get_logger

logger = get_logger("bridge.train")


def selection_mask_hash(masks: List[Any]) -> str:
    """Hash response selections with the mask-gate canonical encoding."""
    digest = hashlib.sha256()
    for sample_idx, values in enumerate(masks):
        tensor = torch.as_tensor(values, dtype=torch.int64).view(-1).cpu()
        digest.update(sample_idx.to_bytes(8, "little", signed=False))
        digest.update(tensor.numel().to_bytes(8, "little", signed=False))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_id_list(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for s in ids:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def special_positions(prompt_length: int, response_length: int,
                      special_token_mode: str = "single_bos_eos",
                      has_bos: bool = True) -> Dict[str, List[int]]:
    "Free/excluded token positions under the v3 layout."
    if special_token_mode != "single_bos_eos":
        return {"prompt_free": [], "response_free": [], "eos_response_index": None}
    return {
        "prompt_free": [0] if has_bos else [],
        "response_free": [response_length - 1],
        "eos_response_index": response_length - 1,
    }


def _train_stage(model, tokenizer, cfg: Config, train_dataset, val_dataset,
                 training_mode: TrainingMode, method: str,
                 selection_results=None, token_weights=None, noisy_masks=None,
                 output_dir: str = "./outputs", epochs=None, lr=None,
                 batch_size=None, grad_accum=None, save_strategy=None,
                 save_final_model: bool = True,
                 resume_from_checkpoint=None) -> dict:
    """通用训练阶段：构建 SelectedDataset + BridgeTrainer 并训练。"""
    train_ds = SelectedDataset(
        train_dataset, training_mode=training_mode, selection_results=selection_results,
        token_weights=token_weights, noisy_masks=noisy_masks,
        prompt_order=getattr(cfg, "prompt_selection_order", "position"),
    )
    eval_ds = SelectedDataset(val_dataset, training_mode=TrainingMode.MASK_ONLY) if val_dataset is not None else None

    stage_cfg = SimpleNamespace(
        num_epochs=epochs or cfg.num_epochs,
        learning_rate=lr or cfg.learning_rate,
        batch_size=batch_size or cfg.batch_size,
        gradient_accumulation_steps=grad_accum or cfg.gradient_accumulation_steps,
        bf16=cfg.bf16, lr_scheduler_type=cfg.lr_scheduler_type, warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay, max_grad_norm=cfg.max_grad_norm,
        report_to=cfg.report_to, seed=cfg.seed, num_workers=0,
        logging_steps=getattr(cfg, "logging_steps", 25),
        save_strategy=save_strategy or cfg.save_strategy,
        fsdp=getattr(cfg, "fsdp", False),
        fsdp_config=getattr(cfg, "fsdp_config", None),
    )
    args = build_training_args(stage_cfg, output_dir=output_dir)
    trainer = BridgeTrainer(
        model=model, tokenizer=tokenizer, config=cfg,
        training_mode=training_mode, method=method,
        args=args, train_dataset=train_ds, eval_dataset=eval_ds,
    )
    memory_before_gib = 0.0
    if torch.cuda.is_available():
        memory_before_gib = torch.cuda.memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint)
            if resume_from_checkpoint is not None else None
        )
    )
    peak_memory_gib = 0.0
    if torch.cuda.is_available():
        peak = torch.tensor(
            torch.cuda.max_memory_allocated() / (1024 ** 3),
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        before = torch.tensor(
            memory_before_gib,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(before, op=torch.distributed.ReduceOp.MAX)
        peak_memory_gib = float(peak.item())
        memory_before_gib = float(before.item())
        logger.info(
            "Training GPU memory: before=%.2f GiB, peak=%.2f GiB",
            memory_before_gib, peak_memory_gib,
        )
    prepare_model_for_evaluation(model)
    final_model_path = str(Path(output_dir) / "final_model") if save_final_model else None
    if save_final_model:
        trainer.save_model(final_model_path)
    history = list(trainer.state.log_history)
    model.zero_grad(set_to_none=True)
    del trainer, train_ds, eval_ds, args
    gc.collect()
    clear_gpu_cache()
    return {
        "model_path": final_model_path,
        "history": history,
        "gpu_memory_before_gib": memory_before_gib,
        "peak_gpu_memory_gib": peak_memory_gib,
    }
