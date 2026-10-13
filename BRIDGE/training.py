"BRIDGE 训练：损失、压缩、HF Trainer 子类、baseline 损失变体。"

import inspect
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from utils import get_logger, debug_tensor_range

logger = get_logger("bridge.training")


_DEFAULT_LOGGING_STEPS = 50
_DEFAULT_SAVE_STRATEGY = "epoch"
_DEFAULT_SAVE_TOTAL_LIMIT = 1


class TrainingMode(Enum):
    "BRIDGE 三种训练策略。"

    MASK_ONLY = "mask_only"
    COMPACT_PROMPT_SELECTED_LOSS = "compact_prompt_selected_loss"
    COMPACT_BOTH = "compact_both"

    @classmethod
    def from_string(cls, mode_str: str) -> "TrainingMode":
        normalised = mode_str.strip().lower().replace("-", "_")
        for mode in cls:
            if mode.value == normalised:
                return mode
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown training mode {mode_str!r}. Valid: {valid}")

    def __str__(self) -> str:
        return self.value


class BridgeLossFunction(nn.Module):
    "BRIDGE 的 mode 感知损失。"

    def __init__(self, mode: TrainingMode = TrainingMode.COMPACT_PROMPT_SELECTED_LOSS,
                 label_smoothing: float = 0.0, loss_reduction: str = "per_batch",
                 strict: bool = False):
        super().__init__()
        if not isinstance(mode, TrainingMode):
            raise TypeError(f"mode must be TrainingMode, got {type(mode).__name__}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0,1), got {label_smoothing}")
        if loss_reduction not in ("per_batch", "per_sample"):
            raise ValueError(
                f"loss_reduction must be 'per_batch' or 'per_sample', got {loss_reduction!r}"
            )
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.loss_reduction = loss_reduction


        self.strict = bool(strict)
        logger.info(
            f"BridgeLossFunction: mode={mode.value}, label_smoothing={label_smoothing}, "
            f"loss_reduction={loss_reduction}, strict={self.strict}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        S_p: Optional[list] = None,
        S_r: Optional[list] = None,
        response_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == TrainingMode.MASK_ONLY:
            return self.mask_only_loss(logits, labels, response_mask)
        if self.mode == TrainingMode.COMPACT_PROMPT_SELECTED_LOSS:
            return self.compact_prompt_selected_loss(logits, labels, S_r)
        if self.mode == TrainingMode.COMPACT_BOTH:
            return self.compact_both_loss(logits, labels, S_p, S_r)
        raise ValueError(f"Unsupported training mode: {self.mode}")

    def compact_prompt_selected_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                                     S_r: Optional[list]) -> torch.Tensor:
        "仅在选中的 response 位置（``S_r``，压缩序列绝对索引）上计算 CE。"
        if S_r is None:
            logger.warning("S_r is None; falling back to full-response loss")
            return self._cross_entropy(logits, labels)
        seq_len = logits.size(1)
        vocab = logits.size(-1)

        for i in range(logits.size(0)):
            ok = debug_tensor_range(f"S_r[{i}]", S_r[i], upper_bound=seq_len, lower_bound=0)
            if not ok:
                raise RuntimeError(
                    f"S_r[{i}] 存在 >= seq_len({seq_len}) 的索引（压缩序列绝对位置越界）。"
                    f" 通常因 response_length 漏算 special token 或 S_r 用错坐标系（相对/全局混用）。"
                )
            debug_tensor_range(f"labels[{i}]", labels[i], upper_bound=vocab, lower_bound=0,
                               ignore_values=(-100,))
        selected_logits: List[torch.Tensor] = []
        selected_labels: List[torch.Tensor] = []
        for i in range(logits.size(0)):
            idx = S_r[i]
            if isinstance(idx, torch.Tensor):
                idx = idx.to(logits.device)
            else:
                idx = torch.as_tensor(idx, device=logits.device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            if (idx <= 0).any():
                raise RuntimeError("S_r absolute positions must be > 0 for causal-LM loss")
            if self.strict:
                sel_lab = labels[i].to(logits.device).index_select(0, idx)
                if (sel_lab == -100).any():
                    raise RuntimeError(
                        f"strict_loss: sample {i} has -100 labels at selected positions "
                        "(labels 与 selected positions 不一致)"
                    )
            selected_logits.append(logits[i].index_select(0, idx - 1))
            selected_labels.append(labels[i].to(logits.device).index_select(0, idx))
        if not selected_logits:
            if self.strict:
                raise RuntimeError(
                    "All samples had empty S_r under strict_loss; refusing to train on "
                    "an empty supervision set (check selection / keep ratios)"
                )
            logger.warning("All samples had empty S_r; returning zero loss")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        if self.loss_reduction == "per_batch":

            flat_logits = torch.cat(selected_logits, dim=0)
            flat_labels = torch.cat(selected_labels, dim=0)
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100,
                                   label_smoothing=self.label_smoothing, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                if self.strict:
                    raise RuntimeError(
                        "per-batch selected loss is NaN/Inf under strict_loss "
                        "(possible cause: all selected labels are -100)"
                    )
                logger.warning("per-batch selected loss is NaN/Inf; returning zero loss")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            return loss

        all_losses: List[torch.Tensor] = []
        for sample_logits, sample_labels in zip(selected_logits, selected_labels):
            loss = F.cross_entropy(sample_logits, sample_labels, ignore_index=-100,
                                   label_smoothing=self.label_smoothing, reduction="mean")
            if not (torch.isnan(loss) or torch.isinf(loss)):
                all_losses.append(loss)
        if not all_losses:
            logger.warning("All per-sample losses were NaN/Inf; returning zero loss")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(all_losses).mean()

    def mask_only_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                       loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """标准 CE；``loss_mask==0`` 位置标签置 -100。"""
        if loss_mask is None:
            return self._cross_entropy(logits, labels)
        masked_labels = labels.clone()
        masked_labels[loss_mask.to(masked_labels.device) == 0] = -100
        return self._cross_entropy(logits, masked_labels)

    def compact_both_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                          S_p: Optional[list], S_r: Optional[list]) -> torch.Tensor:
        """压缩后的 response（即 S_r）上计算 CE。"""
        if S_r is None:
            logger.warning("S_r is None for compact_both; using full labels")
            return self._cross_entropy(logits, labels)
        selected_logits: List[torch.Tensor] = []
        selected_labels: List[torch.Tensor] = []
        for i in range(logits.size(0)):
            r_idx = S_r[i]
            if isinstance(r_idx, torch.Tensor):
                r_idx = r_idx.to(logits.device)
            else:
                r_idx = torch.as_tensor(r_idx, device=logits.device, dtype=torch.long)
            if r_idx.numel() == 0:
                continue
            prompt_len = len(S_p[i]) if S_p is not None and S_p[i] is not None else (logits.size(1) - r_idx.numel())
            response_logits = logits[i, prompt_len - 1:-1, :]
            response_labels = labels[i, prompt_len:].to(logits.device)
            n_resp = min(response_logits.size(0), response_labels.size(0))
            selected_logits.append(response_logits[:n_resp])
            selected_labels.append(response_labels[:n_resp])
        if not selected_logits:
            logger.warning("All samples had empty data in compact_both; returning zero loss")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        if self.loss_reduction == "per_batch":

            flat_logits = torch.cat(selected_logits, dim=0)
            flat_labels = torch.cat(selected_labels, dim=0)
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100,
                                   label_smoothing=self.label_smoothing, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("per-batch compact_both loss is NaN/Inf; returning zero loss")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            return loss
        all_losses: List[torch.Tensor] = []
        for sample_logits, sample_labels in zip(selected_logits, selected_labels):
            loss = F.cross_entropy(sample_logits, sample_labels,
                                   ignore_index=-100, label_smoothing=self.label_smoothing,
                                   reduction="mean")
            if not (torch.isnan(loss) or torch.isinf(loss)):
                all_losses.append(loss)
        if not all_losses:
            logger.warning("All per-sample losses in compact_both were NaN/Inf; returning zero loss")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(all_losses).mean()

    def _cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].to(logits.device).contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100, label_smoothing=self.label_smoothing, reduction="mean",
        )


def _real_positions(mask: torch.Tensor, block_offset: int = 0) -> torch.Tensor:
    "从 block mask 取真实 token 的全局位置（兼容 left/right padding）。"
    return mask.nonzero(as_tuple=True)[0].to(torch.long) + block_offset


def _compact_prompt_sample(
    input_ids: torch.Tensor, prompt_mask: torch.Tensor, response_mask: torch.Tensor,
    prompt_block_len: int, S_p: torch.Tensor, S_r: torch.Tensor,
    sort_prompt: bool = True,
    explicit_response_mask: bool = False,
    eos_response_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    "压缩 prompt 为 S_p，保留完整 response；返回 (compact_ids, labels, remapped_S_r)。"
    device = input_ids.device
    S_p = S_p.to(device)
    S_r = S_r.to(device)
    if sort_prompt:
        S_p, _ = torch.sort(S_p)
    prompt_real = _real_positions(prompt_mask, block_offset=0)
    resp_real = _real_positions(response_mask, block_offset=prompt_block_len)
    prompt_tokens = input_ids[prompt_real[S_p]]
    response_tokens = input_ids[resp_real]
    compact = torch.cat([prompt_tokens, response_tokens])
    if explicit_response_mask:
        keep = sorted({int(v) for v in S_r.tolist()})
        if eos_response_index is not None:
            if not 0 <= int(eos_response_index) < response_tokens.numel():
                raise ValueError(
                    f"eos_response_index {eos_response_index} out of range [0,{response_tokens.numel()})"
                )
            keep = sorted(set(keep) | {int(eos_response_index)})
        labels = torch.full_like(compact, -100)
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        labels[len(S_p) + keep_t] = response_tokens[keep_t]
        remapped_s_r = keep_t + len(S_p)

        assert int((labels != -100).sum()) == len(keep), "explicit label mask count mismatch"
        return compact, labels, remapped_s_r
    labels = compact.clone()
    labels[: len(S_p)] = -100
    remapped_s_r = S_r + len(S_p)
    return compact, labels, remapped_s_r


def _compact_both_sample(
    input_ids: torch.Tensor, prompt_mask: torch.Tensor, response_mask: torch.Tensor,
    prompt_block_len: int, S_p: torch.Tensor, S_r: torch.Tensor,
    sort_indices: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    "同时压缩 prompt(S_p) 与 response(S_r)。位置感知，兼容 left/right padding。"
    device = input_ids.device
    S_p = S_p.to(device)
    S_r = S_r.to(device)
    if sort_indices:
        S_p, _ = torch.sort(S_p)
        S_r, _ = torch.sort(S_r)
    prompt_real = _real_positions(prompt_mask, block_offset=0)
    resp_real = _real_positions(response_mask, block_offset=prompt_block_len)
    prompt_tokens = input_ids[prompt_real[S_p]]
    response_tokens = input_ids[resp_real[S_r]]
    compact = torch.cat([prompt_tokens, response_tokens])
    labels = compact.clone()
    labels[: len(S_p)] = -100
    return compact, labels


class SelectedDataset(Dataset):
    "包装 ``BridgeDataset``，按选择结果与模式产出压缩后的样本。"

    def __init__(
        self,
        base_dataset: Dataset,
        training_mode: TrainingMode,
        selection_results: Optional[Tuple[List, List]] = None,
        token_weights: Optional[List[torch.Tensor]] = None,
        noisy_masks: Optional[List[torch.Tensor]] = None,
        epoch_masks_enabled: bool = False,
        prompt_order: str = "position",
        explicit_response_mask: bool = False,
        always_supervise_eos: bool = False,
    ):
        if prompt_order not in ("position", "score_rank"):
            raise ValueError(
                f"prompt_order must be 'position' or 'score_rank', got {prompt_order!r}"
            )
        self.base = base_dataset
        self.mode = training_mode
        self.all_S_p, self.all_S_r = (None, None) if selection_results is None else selection_results
        self.token_weights = token_weights
        self.noisy_masks = noisy_masks
        if epoch_masks_enabled:
            raise ValueError("Complementary epoch masks are not supported in this release")


        self.prompt_order = prompt_order

        self.explicit_response_mask = bool(explicit_response_mask)
        self.always_supervise_eos = bool(always_supervise_eos)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        input_ids = sample["input_ids"]
        prompt_mask = sample.get("prompt_mask")
        prompt_block_len = int(prompt_mask.shape[0]) if isinstance(prompt_mask, torch.Tensor) else sample["prompt_length"]
        response_length = sample["response_length"]

        if self.mode == TrainingMode.MASK_ONLY:

            response_mask = (sample["labels"] != -100).long()
            out = {
                "input_ids": input_ids,
                "attention_mask": sample["attention_mask"],
                "labels": sample["labels"],
                "response_mask": response_mask,
                "prompt_mask": prompt_mask,
                "prompt_length": sample["prompt_length"],
                "response_length": sample["response_length"],
            }

            resp_start = prompt_block_len
            resp_end = prompt_block_len + response_length
            if self.token_weights is not None:
                w_full = torch.zeros_like(input_ids, dtype=torch.float32)
                w_full[resp_start:resp_end] = self.token_weights[idx][:response_length].float()
                out["token_weights"] = w_full
            if self.noisy_masks is not None:
                nm_full = torch.zeros_like(input_ids, dtype=torch.long)
                nm_full[resp_start:resp_end] = self.noisy_masks[idx][:response_length].long()
                out["noisy_mask"] = nm_full
            return out


        S_p = self.all_S_p[idx]
        S_r = self.all_S_r[idx]
        response_mask = sample.get("response_mask")
        if self.mode == TrainingMode.COMPACT_PROMPT_SELECTED_LOSS:
            eos_idx = (
                int(response_length) - 1
                if self.always_supervise_eos else None
            )
            compact_ids, labels, remapped_s_r = _compact_prompt_sample(
                input_ids, prompt_mask, response_mask, prompt_block_len, S_p, S_r,
                sort_prompt=(self.prompt_order == "position"),
                explicit_response_mask=self.explicit_response_mask,
                eos_response_index=eos_idx,
            )
            out = {
                "input_ids": compact_ids,
                "attention_mask": torch.ones_like(compact_ids),
                "labels": labels,
                "S_r": remapped_s_r,
            }
            return out

        compact_ids, labels = _compact_both_sample(
            input_ids, prompt_mask, response_mask, prompt_block_len, S_p, S_r,
            sort_indices=(self.prompt_order == "position"),
        )
        compact_S_p = torch.arange(len(S_p), dtype=torch.long)
        compact_S_r = torch.arange(len(S_r), dtype=torch.long) + len(S_p)
        return {
            "input_ids": compact_ids,
            "attention_mask": torch.ones_like(compact_ids),
            "labels": labels,
            "S_p": compact_S_p,
            "S_r": compact_S_r,
        }


def _right_pad(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensors)
    out = tensors[0].new_full((len(tensors), max_len), pad_value)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


class BridgeCollator:
    """压缩后变长序列的右填充整理器；``labels`` 用 -100 填充。"""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        input_ids = _right_pad([f["input_ids"] for f in features], self.pad_token_id)
        attention_mask = _right_pad(
            [torch.ones_like(f["input_ids"]) for f in features], 0
        )
        labels = _right_pad([f["labels"] for f in features], -100)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        if "S_r" in features[0]:
            batch["S_r"] = [f["S_r"] for f in features]
        if "S_p" in features[0]:
            batch["S_p"] = [f["S_p"] for f in features]
        if "response_mask" in features[0]:
            rm = _right_pad([f["response_mask"] for f in features], 0)
            batch["response_mask"] = rm
        if "token_weights" in features[0]:
            batch["token_weights"] = _right_pad([f["token_weights"] for f in features], 0)
        if "noisy_mask" in features[0]:
            nm = _right_pad([f["noisy_mask"].long() for f in features], 0)
            batch["noisy_mask"] = nm
        for key in ("sample_idx", "sample_id", "response_offset"):
            if key in features[0]:
                batch[key] = [feature[key] for feature in features]
        return batch


def build_training_args(config, output_dir: str) -> TrainingArguments:
    """从 Config 构建 HF ``TrainingArguments``。"""
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    fsdp_enabled = distributed and bool(getattr(config, "fsdp", False))
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=getattr(config, "num_epochs", 1),
        per_device_train_batch_size=getattr(config, "batch_size", 4),
        per_device_eval_batch_size=getattr(config, "batch_size", 4),
        gradient_accumulation_steps=getattr(config, "gradient_accumulation_steps", 1),
        learning_rate=getattr(config, "learning_rate", 2e-5),
        lr_scheduler_type=getattr(config, "lr_scheduler_type", "cosine"),
        warmup_steps=getattr(config, "warmup_steps", 0),
        weight_decay=getattr(config, "weight_decay", 0.0),
        max_grad_norm=getattr(config, "max_grad_norm", 1.0),
        bf16=getattr(config, "bf16", True),
        logging_steps=getattr(config, "logging_steps", _DEFAULT_LOGGING_STEPS),
        logging_first_step=True,
        save_strategy=getattr(config, "save_strategy", _DEFAULT_SAVE_STRATEGY),
        save_total_limit=_DEFAULT_SAVE_TOTAL_LIMIT,
        report_to=getattr(config, "report_to", "none") or "none",
        seed=getattr(config, "seed", 42),
        remove_unused_columns=False,
        dataloader_num_workers=getattr(config, "num_workers", 0) if hasattr(config, "num_workers") else 0,
        fsdp=True if fsdp_enabled else None,
        fsdp_config=getattr(config, "fsdp_config", None) if fsdp_enabled else None,
    )


class BridgeTrainer(Trainer):
    "注入 BRIDGE 损失的 HF Trainer 子类。"

    def __init__(
        self,
        model,
        tokenizer,
        config,
        training_mode: TrainingMode,
        method: str = "bridge",
        epoch_mask_artifact: Optional[Dict[str, Any]] = None,
        epoch_mask_artifact_path: Optional[str] = None,
        **kwargs,
    ):
        self.bridge_config = config
        self.training_mode = training_mode
        self.method = method
        if epoch_mask_artifact is not None or epoch_mask_artifact_path is not None:
            raise ValueError("Complementary epoch masks are not supported in this release")
        self.loss_fn = BridgeLossFunction(
            mode=training_mode, label_smoothing=0.0,
            loss_reduction=getattr(config, "loss_reduction", "per_batch"),
            strict=getattr(config, "strict_loss", False),
        )
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            kwargs["data_collator"] = BridgeCollator(pad_token_id=pad_id)

        if "processing_class" in inspect.signature(super().__init__).parameters:
            super().__init__(model=model, processing_class=tokenizer, **kwargs)
        else:
            super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        "覆写：用 ``BridgeModel.save`` 保存，绕过 HF Trainer 直接"
        if output_dir is None:
            output_dir = self.args.output_dir
        if getattr(self, "is_fsdp_enabled", False):
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.args.should_save:
                self._save_fsdp_full_state(output_dir, state_dict)
            return
        if not hasattr(self.model, "save"):
            result = super().save_model(output_dir, _internal_call=_internal_call)
            return result
        self.model.save(str(output_dir))

    def _save_fsdp_full_state(self, output_dir: str, state_dict: Dict[str, torch.Tensor]) -> None:
        "Save an FSDP full state through the underlying HF model."
        unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        hf_model = getattr(unwrapped, "model", unwrapped)
        config = getattr(hf_model, "config", None)

        wrapper_prefix = "_model."
        wrapper_keys = [key for key in state_dict if key.startswith(wrapper_prefix)]
        if wrapper_keys:
            if len(wrapper_keys) != len(state_dict):
                non_wrapper = [key for key in state_dict if not key.startswith(wrapper_prefix)]
                raise RuntimeError(
                    "FSDP state dict mixes BridgeModel and HF parameter names; "
                    f"examples without '{wrapper_prefix}': {non_wrapper[:3]}"
                )
            state_dict = {
                key[len(wrapper_prefix):]: value for key, value in state_dict.items()
            }
            logger.info(
                "FSDP save: removed BridgeModel '%s' prefix from %d parameters",
                wrapper_prefix,
                len(wrapper_keys),
            )

        if bool(getattr(config, "tie_word_embeddings", False)):
            input_embeddings = hf_model.get_input_embeddings()
            output_embeddings = hf_model.get_output_embeddings()
            module_names = {id(module): name for name, module in hf_model.named_modules()}
            input_prefix = module_names.get(id(input_embeddings))
            output_prefix = module_names.get(id(output_embeddings))
            if not input_prefix or not output_prefix:
                raise RuntimeError(
                    "Cannot resolve tied input/output embedding module names for FSDP save"
                )
            input_key = f"{input_prefix}.weight"
            output_key = f"{output_prefix}.weight"
            if input_key not in state_dict:
                raise RuntimeError(
                    "FSDP state dict is missing the tied input embedding "
                    f"'{input_key}'"
                )
            if output_key != input_key and state_dict.pop(output_key, None) is not None:
                logger.info(
                    "FSDP save: omitted tied output embedding %s; it will be "
                    "restored from %s when loading",
                    output_key,
                    input_key,
                )

        if not hasattr(hf_model, "save_pretrained"):
            raise TypeError("FSDP model does not expose an underlying HF save_pretrained()")

        os.makedirs(output_dir, exist_ok=True)
        hf_model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=getattr(self.args, "save_safetensors", True),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info(f"FSDP full model saved to '{output_dir}'")

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):

        S_r = inputs.pop("S_r", None)
        S_p = inputs.pop("S_p", None)
        response_mask = inputs.pop("response_mask", None)
        token_weights = inputs.pop("token_weights", None)
        noisy_mask = inputs.pop("noisy_mask", None)
        labels = inputs.get("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]


        if S_r is not None or S_p is not None:
            vocab = getattr(getattr(self.model, "config", None), "vocab_size", None)
            seq_len = int(input_ids.shape[1])
            if vocab is not None:
                if not debug_tensor_range("input_ids", input_ids, upper_bound=vocab, lower_bound=0):
                    raise RuntimeError(f"input_ids 存在 >= vocab_size({vocab}) 的 token id")
            if labels is not None:
                if not debug_tensor_range("labels", labels, upper_bound=vocab, lower_bound=0,
                                          ignore_values=(-100,)):
                    raise RuntimeError(f"labels 存在非法值（非 -100 且 >= vocab_size({vocab})）")

            if S_r is not None:
                for i, s in enumerate(S_r):
                    if not debug_tensor_range(f"S_r[{i}]", s, upper_bound=seq_len, lower_bound=0):
                        raise RuntimeError(f"S_r[{i}] 越界（>= seq_len {seq_len}）")
            if S_p is not None:
                for i, s in enumerate(S_p):
                    if not debug_tensor_range(f"S_p[{i}]", s, upper_bound=seq_len, lower_bound=0):
                        raise RuntimeError(f"S_p[{i}] 越界（>= seq_len {seq_len}）")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        labels = labels.to(logits.device)

        if self.method == "rho1":
            loss = self._rho1_loss(logits, labels, response_mask, token_weights)
        elif self.method == "token_cleaning":
            loss = self._token_cleaning_loss(logits, labels, response_mask, noisy_mask)
        else:
            loss = self.loss_fn(logits=logits, labels=labels, S_p=S_p, S_r=S_r,
                                response_mask=response_mask)

        return (loss, outputs) if return_outputs else loss


    def _rho1_loss(self, logits, labels, response_mask, token_weights) -> torch.Tensor:
        "RHO-1 风格 per-token 重要性加权 CE。"
        if token_weights is None:
            return self.loss_fn.mask_only_loss(logits, labels, response_mask)

        shift_labels = labels[:, 1:].to(logits.device).contiguous()
        ce = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view_as(shift_labels)
        valid = (shift_labels != -100).float()
        w = token_weights[:, 1:].to(logits.device).float() * valid
        denom = w.sum().clamp(min=1e-8)
        return (ce * w).sum() / denom

    def _token_cleaning_loss(self, logits, labels, response_mask, noisy_mask) -> torch.Tensor:
        "TokenCleaning：将高方差噪声 token 从损失中剔除。"
        if noisy_mask is None:
            return self.loss_fn.mask_only_loss(logits, labels, response_mask)
        masked_labels = labels.clone()
        masked_labels[noisy_mask.to(logits.device).bool() == 1] = -100
        return self.loss_fn._cross_entropy(logits, masked_labels)
