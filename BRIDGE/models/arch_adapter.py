"统一架构 adapter：文本 backbone / 层类型 / full-attention / special tokens。"
from typing import Any, Dict, List, Optional

import torch.nn as nn


_BACKBONE_CANDIDATES = (
    "model.model.layers",
    "model.model.language_model.layers",
    "model.language_model.layers",
    "model.model.text_model.layers",
    "model.text_model.layers",
    "model.layers",
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
)


def _unwrap(model: nn.Module) -> nn.Module:
    """BridgeModel/PeftModel -> 底层 HF 模型。"""
    m = getattr(model, "model", model)
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


def _resolve_candidates(base: nn.Module) -> Optional[nn.ModuleList]:
    for candidate in _BACKBONE_CANDIDATES:
        obj: Any = base
        try:
            for part in candidate.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                return nn.ModuleList(list(obj)) if not isinstance(obj, nn.ModuleList) else obj
        except AttributeError:
            continue
    return None


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    "定位 decoder 层列表（候选路径逐一探测）。"
    found = _resolve_candidates(model)
    if found is not None:
        return found
    base = _unwrap(model)
    if base is not model:
        found = _resolve_candidates(base)
        if found is not None:
            return found
    raise AttributeError(
        "cannot locate decoder layers (tried " + ", ".join(_BACKBONE_CANDIDATES) + ")"
    )


def get_attention_module(layer: nn.Module) -> Optional[nn.Module]:
    """返回该层的标准 pairwise self-attention 模块；linear/recurrent 层返回 None。"""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    if not (hasattr(attn, "q_proj") and hasattr(attn, "k_proj")):
        return None
    return attn


def get_layer_types(model: nn.Module) -> List[str]:
    """逐层类型：'full_attention' | 'linear_attention' | 'sliding_attention' | 'unknown'。"""
    layers = get_decoder_layers(model)
    types = []
    for layer in layers:
        attn = get_attention_module(layer)
        if attn is None:
            types.append("linear_attention")
            continue
        sliding = getattr(attn, "is_sliding", None)
        if sliding is True:
            types.append("sliding_attention")
        else:
            types.append("full_attention")
    return types


def get_last_full_attention_layer(model: nn.Module) -> int:
    """最后一个标准 full-attention 层索引；不存在则显式报错（NO-GO，不回退）。"""
    layers = get_decoder_layers(model)
    for i in range(len(layers) - 1, -1, -1):
        attn = get_attention_module(layers[i])
        if attn is None:
            continue
        if getattr(attn, "is_sliding", None) is True:
            continue
        return i
    raise RuntimeError(
        "no full-attention layer found in this model; BRIDGE interaction proxy "
        "requires at least one standard full-attention layer"
    )


def get_text_config(model: nn.Module) -> Any:
    """返回文本 config（多模态包装时取 text_config）。"""
    base = _unwrap(model)
    cfg = getattr(base, "config", None)
    if cfg is None:
        raise AttributeError("model has no config")
    return getattr(cfg, "text_config", cfg)


def get_tokenizer_special_tokens(tokenizer: Any) -> Dict[str, Any]:
    """tokenizer special-token 事实表；BOS 缺失时 has_bos=False（不得按位置臆造）。"""
    bos = getattr(tokenizer, "bos_token_id", None)
    return {
        "bos_token_id": bos,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "has_bos": bos is not None,
    }


LAYER_SUBSET_STRATEGIES = (
    "last_n_physical",
    "include_last_global",
    "explicit_indices",
)


def resolve_layer_indices_from_config(
    hf_config: Any,
    strategy: str,
    explicit_indices: Optional[List[int]] = None,
    n_last: int = 4,
) -> List[int]:
    "不加载模型权重，仅从 HF config 解析层子集索引。"
    tcfg = getattr(hf_config, "text_config", hf_config)
    n = int(getattr(tcfg, "num_hidden_layers"))
    if strategy == "last_n_physical":
        k = max(1, min(int(n_last), n))
        return list(range(n - k, n))
    if strategy == "explicit_indices":
        if not explicit_indices:
            raise ValueError("explicit_indices strategy requires a non-empty index list")
        idx = sorted({int(i) for i in explicit_indices})
        if len(idx) != len(explicit_indices):
            raise ValueError(f"explicit_indices contains duplicates: {explicit_indices}")
        bad = [i for i in idx if i < 0 or i >= n]
        if bad:
            raise ValueError(f"explicit_indices out of range [0, {n}): {bad}")
        return idx
    if strategy == "include_last_global":
        layer_types = getattr(tcfg, "layer_types", None)
        if not layer_types:
            raise RuntimeError(
                "include_last_global requires config.layer_types to identify the "
                "last global/full-attention layer; refusing to guess"
            )
        global_idx = [
            i for i, t in enumerate(layer_types)
            if str(t) in ("full_attention", "global_attention")
        ]
        if not global_idx:
            raise RuntimeError(
                f"include_last_global: no full/global layer in layer_types "
                f"({list(layer_types)}); refusing to fall back"
            )
        last_global = global_idx[-1]
        k = max(1, min(int(n_last), n))
        start = min(last_global, n - k)
        return list(range(start, n))
    raise ValueError(
        f"unknown layer subset strategy {strategy!r}; "
        f"expected one of {LAYER_SUBSET_STRATEGIES}"
    )


def resolve_layer_indices(
    model: nn.Module,
    strategy: str,
    explicit_indices: Optional[List[int]] = None,
    n_last: int = 4,
) -> List[int]:
    "把层子集策略解析为确定的 decoder 层索引列表（升序、去重）。"
    layers = get_decoder_layers(model)
    n = len(layers)
    if strategy == "last_n_physical":
        k = max(1, min(int(n_last), n))
        return list(range(n - k, n))
    if strategy == "include_last_global":
        last_global = get_last_full_attention_layer(model)
        k = max(1, min(int(n_last), n))
        start = min(last_global, n - k)
        return list(range(start, n))
    if strategy == "explicit_indices":
        if not explicit_indices:
            raise ValueError(
                "explicit_indices strategy requires a non-empty index list"
            )
        idx = sorted({int(i) for i in explicit_indices})
        if len(idx) != len(explicit_indices):
            raise ValueError(
                f"explicit_indices contains duplicates: {explicit_indices}"
            )
        bad = [i for i in idx if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f"explicit_indices out of range [0, {n}): {bad}"
            )
        return idx
    raise ValueError(
        f"unknown layer subset strategy {strategy!r}; "
        f"expected one of {LAYER_SUBSET_STRATEGIES}"
    )
