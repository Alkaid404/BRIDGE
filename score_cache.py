"Scoring provenance hashes."

import hashlib
import json
from typing import Any, Dict, List

import torch

VJP_NUM_LAST_LAYERS = 4


def compute_validation_hash(validation_direction: torch.Tensor) -> str:
    """``SHA256(validation_direction 的 canonical float32 字节)``。方向一变 hash 即变。"""
    v = validation_direction.detach().cpu().float().contiguous()
    return hashlib.sha256(v.numpy().tobytes()).hexdigest()[:16]


def compute_dataset_hash(samples: List[Dict[str, Any]]) -> str:
    """对实际参与评分的有序样本内容做摘要，防止同路径/同大小数据误命中。"""
    digest = hashlib.sha256()
    for sample in samples:
        payload = {
            "id": sample.get("id"),
            "prompt": sample.get("prompt"),
            "response": sample.get("response"),
        }
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def compute_config_hash(cfg: Any) -> str:
    "``SHA256(影响 score 的配置子集 JSON)``。排除 wandb/output_dir/lr/epoch/batch。"
    fields = {
        "model_name_or_path": getattr(cfg, "model_name_or_path", None),
        "dataset_name": getattr(cfg, "dataset_name", None),
        "dataset_path": getattr(cfg, "dataset_path", None),
        "attention_aggregation": getattr(cfg, "attention_aggregation", None),
        "proxy_type": getattr(cfg, "proxy_type", None),
        "scoring_backend": getattr(cfg, "scoring_backend", "vjp"),
        "use_vjp": getattr(cfg, "use_vjp", None),
        "attention_num_layers": getattr(cfg, "attention_num_layers", None),
        "max_seq_length": getattr(cfg, "max_seq_length", None),
        "max_train_samples": getattr(cfg, "max_train_samples", None),
        "seed": getattr(cfg, "seed", None),
        "bf16": getattr(cfg, "bf16", None),
        "num_last_layers": VJP_NUM_LAST_LAYERS,
    }
    payload = json.dumps(fields, sort_keys=True, default=str, ensure_ascii=False)


    _stm = getattr(cfg, "special_token_mode", "legacy")
    if _stm not in (None, "legacy"):
        payload += json.dumps({"special_token_mode": _stm}, sort_keys=True)


    _extra = {}
    _attention_indices = getattr(cfg, "attention_layer_indices", None)
    if _attention_indices is not None:
        _extra["attention_layer_indices"] = list(_attention_indices)
    _dls = getattr(cfg, "direction_layer_strategy", "last_n_physical")
    if _dls not in (None, "last_n_physical"):
        _extra["direction_layer_strategy"] = _dls
        _extra["direction_layer_indices"] = getattr(cfg, "direction_layer_indices", None)
    _dnl = getattr(cfg, "direction_num_last_layers", 4)
    if _dnl not in (None, 4):
        _extra["direction_num_last_layers"] = _dnl
    _sls = getattr(cfg, "scoring_attention_layer_strategy", "last_physical")
    if _sls not in (None, "last_physical"):
        _extra["scoring_attention_layer_strategy"] = _sls
        _extra["scoring_attention_layer_index"] = getattr(cfg, "scoring_attention_layer_index", None)
    _bos = getattr(cfg, "bos_handling", "zero_no_renorm")
    if _bos not in (None, "zero_no_renorm"):
        _extra["bos_handling"] = _bos
    if _extra:
        payload += json.dumps(_extra, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
