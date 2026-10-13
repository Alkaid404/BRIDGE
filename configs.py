"BRIDGE configuration and YAML loading."

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils import get_logger, ensure_dir
from structure.config import (
    REMOVED_CONFIG_KEYS,
    normalize_structure_config,
    validate_profile,
)

logger = get_logger("bridge")


@dataclass
class Config:
    "BRIDGE 全部超参与实验开关。"


    method: str = "bridge"
    profile: str = "closure_only"
    output_dir: str = "./outputs"
    seed: int = 42


    model_name_or_path: str = "meta-llama/Llama-3.2-3B"
    bf16: bool = True
    fsdp: bool = False
    fsdp_config: Optional[Dict[str, Any]] = None


    dataset_name: str = "gsm8k"
    dataset_path: Optional[str] = None
    train_split: Optional[str] = None
    validation_split: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_seq_length: int = 512


    special_token_mode: str = "legacy"


    warmup_ratio: float = 0.10
    warmup_epochs: int = 2
    warmup_learning_rate: float = 2e-5
    warmup_batch_size: int = 4
    warmup_gradient_accumulation_steps: int = 8
    distributed_warmup_training: bool = False

    direction_batch_size: int = 2
    scoring_batch_size: int = 4
    proxy_type: str = "attention"
    attention_aggregation: str = "last_layer"
    attention_num_layers: Optional[int] = None
    attention_layer_indices: Optional[List[int]] = None
    scoring_backend: str = "vjp"
    use_vjp: bool = True
    vjp_chunk_size: int = 64


    scoring_attention_layer_strategy: str = "last_physical"
    scoring_attention_layer_index: Optional[int] = None


    bos_handling: str = "zero_no_renorm"


    direction_layer_strategy: str = "last_n_physical"
    direction_num_last_layers: int = 4
    direction_layer_indices: Optional[List[int]] = None


    trainable_layer_strategy: str = "full"
    trainable_layer_indices: Optional[List[int]] = None


    budget_mode: str = "fixed"
    budget_rounding: str = "round"
    prompt_keep_ratio: float = 0.5
    response_keep_ratio: float = 0.5
    T_alt: int = 4


    num_epochs: int = 5
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    training_mode: str = "compact_prompt_selected_loss"
    save_strategy: str = "epoch"
    logging_steps: int = 25
    distributed_final_training: bool = False
    loss_reduction: str = "per_batch"


    prompt_selection_order: str = "position"

    strict_loss: bool = False


    retention_enabled: bool = False
    retention_lambda: float = 0.2
    retention_dataset_name: Optional[str] = None
    retention_dataset_path: Optional[str] = None
    retention_num_samples: int = 256
    retention_seed: int = 42
    retention_grad_mode: str = "raw"
    retention_prior: str = "auxiliary_ce"
    retention_anchor_checkpoint: Optional[str] = None


    eval_batch_size: int = 16
    max_gen_tokens: int = 512
    temperature: float = 0.0
    num_eval_samples: Optional[int] = None
    num_code_eval_samples: Optional[int] = 256
    num_few_shot: int = 8
    max_code_gen_tokens: int = 512
    multiple_choice_mode: str = "generation"
    likelihood_batch_size: int = 8
    likelihood_max_length: int = 4096
    eval_few_shot: Dict[str, int] = field(default_factory=dict)
    humaneval_timeout: float = 5.0
    humaneval_workers: int = 4
    humaneval_memory_limit_mb: int = 1024
    evaluate_after_training: bool = True
    alignment_benchmarks: List[str] = field(default_factory=lambda: ["arc_easy", "arc_challenge"])
    reasoning_benchmarks: List[str] = field(default_factory=lambda: ["gsm8k"])
    code_benchmarks: List[str] = field(default_factory=list)
    eval_dataset_path: Optional[Dict[str, str]] = None
    eval_split: Optional[Dict[str, str]] = None


    num_gpus: int = 1
    wandb_project: str = "BRIDGE"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    report_to: str = "none"


    reuse_score_cache: bool = True
    save_score_cache: bool = True
    cache_level: str = "score"
    cache_dtype: str = "float32"


    debug_score_cache: bool = False
    debug_score_cache_samples: int = 16


    validation_ratio: Optional[float] = None
    data_split_dir: Optional[str] = None
    expected_split_sizes: Optional[Dict[str, int]] = None
    require_selection_provenance: bool = False

    special_token_scoring: str = "include"

    special_token_budget: str = "counted"
    always_keep_bos: bool = False
    always_supervise_eos: bool = False

    explicit_response_mask: bool = False

    structure_protection: Dict[str, Any] = field(default_factory=dict)

    direction_type: str = "validation"


    structure: Dict[str, Any] = field(default_factory=dict)
    complementary: Dict[str, Any] = field(default_factory=dict)


    def __post_init__(self) -> None:
        if self.expected_split_sizes is not None:
            expected = self.expected_split_sizes
            if (set(expected) != {"warmup", "validation", "sft"}
                    or any(type(n) is not int or n < 1 for n in expected.values())):
                raise ValueError("expected_split_sizes requires positive warmup/validation/sft counts")
        if self.budget_rounding not in ("round", "ceil"):
            raise ValueError("budget_rounding must be 'round' or 'ceil'")
        if self.budget_mode != "fixed":
            raise ValueError("This release supports only budget_mode=fixed")
        if self.attention_num_layers is not None and (
            type(self.attention_num_layers) is not int or self.attention_num_layers < 1
        ):
            raise ValueError("attention_num_layers must be a positive integer or None")
        if self.attention_layer_indices is not None:
            indices = self.attention_layer_indices
            if (not indices or any(type(i) is not int or i < 0 for i in indices)
                    or len(indices) != len(set(indices))):
                raise ValueError("attention_layer_indices must be nonempty unique nonnegative integers")
            if self.attention_aggregation != "multi_layer_avg":
                raise ValueError("attention_layer_indices requires multi_layer_avg")
        self.profile = validate_profile(self.profile)
        self.structure = normalize_structure_config(self.structure)
        if self.complementary:
            raise ValueError("Complementary epoch masks are not supported in this release")
        if self.loss_reduction not in ("per_batch", "per_sample"):
            raise ValueError(
                f"loss_reduction must be 'per_batch' or 'per_sample', got {self.loss_reduction!r}"
            )
        if self.retention_grad_mode not in ("raw", "match_norm", "unit_mix"):
            raise ValueError(
                f"retention_grad_mode must be 'raw', 'match_norm' or 'unit_mix', got {self.retention_grad_mode!r}"
            )
        if self.retention_prior not in ("auxiliary_ce", "l2_anchor"):
            raise ValueError(
                "retention_prior must be 'auxiliary_ce' or 'l2_anchor', "
                f"got {self.retention_prior!r}"
            )
        if not (0.0 < self.prompt_keep_ratio <= 1.0):
            raise ValueError(
                f"prompt_keep_ratio must be in (0,1], got {self.prompt_keep_ratio}"
            )
        if not (0.0 < self.response_keep_ratio <= 1.0):
            raise ValueError(
                f"response_keep_ratio must be in (0,1], got {self.response_keep_ratio}"
            )
        if self.retention_lambda < 0:
            raise ValueError(f"retention_lambda must be >= 0, got {self.retention_lambda}")
        if self.retention_grad_mode == "unit_mix" and self.retention_lambda > 1:
            raise ValueError(
                "retention_lambda must be in [0,1] for unit_mix, "
                f"got {self.retention_lambda}"
            )
        if self.retention_enabled:
            if self.retention_prior == "auxiliary_ce":
                if not self.retention_dataset_name:
                    raise ValueError(
                        "auxiliary_ce retention requires retention_dataset_name"
                    )
                if not self.retention_dataset_path:
                    raise ValueError(
                        "auxiliary_ce retention requires an explicit retention_dataset_path"
                    )
            elif not self.retention_anchor_checkpoint:
                raise ValueError(
                    "l2_anchor retention requires retention_anchor_checkpoint"
                )
        for split_name, split_value in (
            ("train_split", self.train_split),
            ("validation_split", self.validation_split),
        ):
            if split_value is not None and str(split_value).strip().lower() == "test":
                raise ValueError(f"{split_name} must not use the test split")
        if self.special_token_mode not in ("legacy", "single_bos_eos"):
            raise ValueError(
                f"special_token_mode must be 'legacy' or 'single_bos_eos', "
                f"got {self.special_token_mode!r}"
            )
        if self.prompt_selection_order not in ("position", "score_rank"):
            raise ValueError(
                f"prompt_selection_order must be 'position' or 'score_rank', "
                f"got {self.prompt_selection_order!r}"
            )
        if self.debug_score_cache_samples < 1:
            raise ValueError(
                f"debug_score_cache_samples must be >= 1, got {self.debug_score_cache_samples}"
            )
        if self.special_token_scoring not in ("include", "exclude"):
            raise ValueError(
                f"special_token_scoring must be 'include' or 'exclude', "
                f"got {self.special_token_scoring!r}"
            )
        if self.special_token_budget not in ("counted", "fixed_free"):
            raise ValueError(
                f"special_token_budget must be 'counted' or 'fixed_free', "
                f"got {self.special_token_budget!r}"
            )
        if self.validation_ratio is not None and not (0.0 < self.validation_ratio < 1.0):
            raise ValueError(
                f"validation_ratio must be in (0,1), got {self.validation_ratio}"
            )
        if self.scoring_attention_layer_strategy not in (
            "last_physical", "last_global", "explicit_index", "last_full_attention"
        ):
            raise ValueError(
                f"scoring_attention_layer_strategy must be 'last_physical', "
                f"'last_global', 'explicit_index' or 'last_full_attention', "
                f"got {self.scoring_attention_layer_strategy!r}"
            )
        if self.scoring_attention_layer_strategy == "explicit_index" and (
            self.scoring_attention_layer_index is None
        ):
            raise ValueError(
                "scoring_attention_layer_strategy='explicit_index' requires "
                "scoring_attention_layer_index"
            )
        if self.bos_handling not in ("auto", "zero_no_renorm", "zero_and_renorm", "keep"):
            raise ValueError(
                f"bos_handling must be 'auto', 'zero_no_renorm', "
                f"'zero_and_renorm' or 'keep', got {self.bos_handling!r}"
            )
        _layer_subset_strategies = (
            "last_n_physical", "include_last_global", "explicit_indices"
        )
        if self.direction_layer_strategy not in _layer_subset_strategies:
            raise ValueError(
                f"direction_layer_strategy must be one of {_layer_subset_strategies}, "
                f"got {self.direction_layer_strategy!r}"
            )
        if self.direction_layer_strategy == "explicit_indices" and not (
            self.direction_layer_indices
        ):
            raise ValueError(
                "direction_layer_strategy='explicit_indices' requires "
                "direction_layer_indices"
            )
        if self.direction_layer_indices is not None and (
            self.direction_layer_strategy != "explicit_indices"
        ):
            raise ValueError(
                "direction_layer_indices is only valid with "
                "direction_layer_strategy='explicit_indices'"
            )
        if self.trainable_layer_strategy not in ("full",) + _layer_subset_strategies:
            raise ValueError(
                f"trainable_layer_strategy must be 'full' or one of "
                f"{_layer_subset_strategies}, got {self.trainable_layer_strategy!r}"
            )
        if self.trainable_layer_strategy == "explicit_indices" and not (
            self.trainable_layer_indices
        ):
            raise ValueError(
                "trainable_layer_strategy='explicit_indices' requires "
                "trainable_layer_indices"
            )
        if self.trainable_layer_indices is not None and (
            self.trainable_layer_strategy != "explicit_indices"
        ):
            raise ValueError(
                "trainable_layer_indices is only valid with "
                "trainable_layer_strategy='explicit_indices'"
            )
        if self.direction_num_last_layers < 1:
            raise ValueError(
                f"direction_num_last_layers must be >= 1, got {self.direction_num_last_layers}"
            )
        _sp_defaults = {
            "protect_balanced_latex": False,
            "protect_equation_spans": False,
            "protect_final_answer": False,
            "protect_boxed_answer": False,
            "protect_reasoning_sentence": False,
            "protect_symbol_definition": False,
            "protect_quantity_unit": False,
        }
        unknown_sp = sorted(set(self.structure_protection) - set(_sp_defaults))
        if unknown_sp:
            raise ValueError(
                f"unknown structure_protection keys {unknown_sp}; "
                f"supported={sorted(_sp_defaults)}"
            )
        self.structure_protection = {**_sp_defaults, **self.structure_protection}

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def save(self, path: str) -> None:
        ensure_dir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)
        logger.info(f"Config saved to '{path}'")


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CODE_ROOT = Path(__file__).resolve().parent
_PATH_ENV_DEFAULTS = {
    "BRIDGE_DATA_ROOT": str(_CODE_ROOT / "data"),
    "BRIDGE_MODEL_PATH": str(_CODE_ROOT / "checkpoints" / "Llama-3.2-3B"),
}


def _expand_env(value):
    "递归展开字符串中的 ``${VAR}`` 占位符为环境变量值；未设则保留原字面量。"
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(
                m.group(1), _PATH_ENV_DEFAULTS.get(m.group(1), m.group(0))
            ),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings; scalars/lists/null in ``override`` replace base values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_extended_config(reference: str, child: Path) -> Path:
    candidate = Path(reference).expanduser()
    if candidate.is_absolute():
        return candidate
    relative = child.parent / candidate
    if relative.exists():
        return relative
    repository_relative = Path(__file__).resolve().parent / candidate
    if repository_relative.exists():
        return repository_relative
    return relative


def _load_config_mapping(path: "str | Path", stack: tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Load YAML plus optional ``extends`` parents, with cycle-safe deep merging."""
    config_path = Path(path).expanduser().resolve()
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Config inheritance cycle: {chain}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping, got {type(raw).__name__}: {config_path}"
        )
    parent_refs = raw.pop("extends", [])
    if isinstance(parent_refs, str):
        parent_refs = [parent_refs]
    if not isinstance(parent_refs, list) or not all(isinstance(x, str) for x in parent_refs):
        raise ValueError(
            f"'extends' must be a path string or list of path strings: {config_path}"
        )
    merged: Dict[str, Any] = {}
    next_stack = (*stack, config_path)
    for reference in parent_refs:
        parent = _resolve_extended_config(reference, config_path)
        merged = _deep_merge_config(merged, _load_config_mapping(parent, next_stack))
    return _deep_merge_config(merged, raw)


def load_config(path: "str | Path", overrides: Optional[Dict[str, Any]] = None) -> Config:
    "从 YAML 加载 Config，可选 CLI 覆盖。"
    data = _load_config_mapping(path)
    _flatten_retention_block(data)
    data = _expand_env(data)
    removed = sorted(set(data) & REMOVED_CONFIG_KEYS)
    if removed:
        raise ValueError(
            "Removed Completion/Graph configuration keys are not supported: "
            f"{removed}. Supported profile: closure_only."
        )
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    valid = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"Unknown configuration keys: {unknown}; "
            "fix the spelling or add an explicit compatibility mapping"
        )
    return Config(**data)


_RETENTION_BLOCK_KEYS = {
    "enabled": "retention_enabled",
    "lambda": "retention_lambda",
    "dataset_name": "retention_dataset_name",
    "dataset_path": "retention_dataset_path",
    "num_samples": "retention_num_samples",
    "seed": "retention_seed",
    "grad_mode": "retention_grad_mode",
    "prior": "retention_prior",
    "anchor_checkpoint": "retention_anchor_checkpoint",
}


def _flatten_retention_block(data: Dict[str, Any]) -> None:
    """把嵌套 ``retention:`` 块就地平铺为 ``retention_*`` 键（平铺键优先）。"""
    block = data.pop("retention", None)
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError(f"'retention' config block must be a mapping, got {type(block)}")
    unknown = sorted(set(block) - set(_RETENTION_BLOCK_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown retention config keys: {unknown}; "
            f"supported={sorted(_RETENTION_BLOCK_KEYS)}"
        )
    for key, flat in _RETENTION_BLOCK_KEYS.items():
        if key in block and flat not in data:
            data[flat] = block[key]
