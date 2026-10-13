"注意力记录：内存高效地提取 BRIDGE 所需的 response-attention。"

import inspect
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from models.utils import _get_layers
from utils import get_logger

logger = get_logger("bridge.scoring")


def real_prompt_response_positions(
    attention_mask_i: torch.Tensor,
    prompt_block_len: Optional[int] = None,
    prompt_length: Optional[int] = None,
    response_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    "从 attention_mask 推断单样本真实 prompt / response 的全局位置。"
    if prompt_block_len is not None:
        prompt_pos = attention_mask_i[:prompt_block_len].nonzero(as_tuple=True)[0]
        response_pos = attention_mask_i[prompt_block_len:].nonzero(as_tuple=True)[0]
        response_pos = response_pos + prompt_block_len
    else:
        if prompt_length is None or response_length is None:
            raise ValueError("Either prompt_block_len or (prompt_length, response_length) must be provided")
        real_pos = attention_mask_i.nonzero(as_tuple=True)[0]
        needed = prompt_length + response_length
        if real_pos.numel() < needed:
            raise ValueError(
                f"attention_mask has {real_pos.numel()} real tokens, "
                f"but prompt({prompt_length}) + response({response_length}) = {needed}"
            )
        prompt_pos = real_pos[:prompt_length]
        response_pos = real_pos[prompt_length:prompt_length + response_length]

    if prompt_pos.numel() == 0:
        raise ValueError("No real prompt tokens found in attention_mask")
    if response_pos.numel() == 0:
        raise ValueError("No real response tokens found in attention_mask")


    seq_len = int(attention_mask_i.shape[0])
    p_max = int(prompt_pos.max().item())
    r_max = int(response_pos.max().item())
    if p_max >= seq_len or r_max >= seq_len:
        raise ValueError(
            f"real position out of range: prompt_max={p_max} response_max={r_max} seq_len={seq_len}"
        )


    if prompt_length is not None and prompt_pos.numel() != prompt_length:
        logger.warning(
            f"prompt real tokens ({prompt_pos.numel()}) != prompt_length arg ({prompt_length}); "
            f"special-token 计数不一致，selection 索引可能越界"
        )
    if response_length is not None and response_pos.numel() != response_length:
        logger.warning(
            f"response real tokens ({response_pos.numel()}) != response_length arg ({response_length}); "
            f"special-token 计数不一致 -> S_r 索引将越界 compact_len"
        )
    return prompt_pos, response_pos


def _unwrap_hf_model(model: nn.Module) -> nn.Module:
    """BridgeModel / PeftModel -> 底层 HF 模型（含 ``.model.layers``）。"""
    m = getattr(model, "model", model)
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Llama-family RoPE 的 rotate_half（与 HF 一致）。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    "应用 RoPE。``cos/sin`` 形状归一化到可对 head 维广播 ``[*, 1, S, d]``。"
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    rotary_dim = cos.shape[-1]
    d = q.shape[-1]
    if 0 < rotary_dim < d:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
        return q, k
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class AttentionRecorder:
    "内存高效地记录 BRIDGE 所需的 response-attention。"

    VALID_ATTENTION_TYPES = {"last_layer", "multi_layer_avg"}
    VALID_LAYER_STRATEGIES = {"last_physical", "last_global", "explicit_index", "last_full_attention"}

    def __init__(self, model: nn.Module, attention_type: str = "last_layer", num_layers: int = 1,
                 layer_strategy: str = "last_physical", layer_index: Optional[int] = None,
                 layer_indices: Optional[List[int]] = None):
        if attention_type not in self.VALID_ATTENTION_TYPES:
            raise ValueError(f"attention_type must be one of {self.VALID_ATTENTION_TYPES}, got {attention_type!r}")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}")
        if layer_strategy not in self.VALID_LAYER_STRATEGIES:
            raise ValueError(
                f"layer_strategy must be one of {self.VALID_LAYER_STRATEGIES}, got {layer_strategy!r}"
            )
        if layer_strategy == "explicit_index" and layer_index is None:
            raise ValueError("layer_strategy='explicit_index' requires layer_index")
        self._model = model
        self._attention_type = attention_type
        self._num_layers = num_layers
        self._layer_strategy = layer_strategy
        self._layer_index = layer_index
        if layer_indices is not None:
            if (not layer_indices or len(set(layer_indices)) != len(layer_indices)
                    or any(type(i) is not int or i < 0 for i in layer_indices)):
                raise ValueError("layer_indices must be unique nonnegative integers")
            if attention_type != "multi_layer_avg":
                raise ValueError("explicit layer_indices requires multi_layer_avg")
        self._explicit_layer_indices = None if layer_indices is None else list(layer_indices)
        self._hooks: List = []
        self._hooked_attns: List[nn.Module] = []
        self._selected_layer_indices: List[int] = []
        self._layer_strategy_note: str = ""
        self._capturing = False
        self._ctx: Optional[Dict[str, Any]] = None

        self._cur_q: Optional[torch.Tensor] = None
        self._cur_k: Optional[torch.Tensor] = None
        self._cur_position_ids: Optional[torch.Tensor] = None
        self._cur_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._use_hook = False
        try:
            self._install_hooks()
            self._use_hook = True
        except (ValueError, RuntimeError):


            raise
        except Exception as exc:
            self._cleanup_hooks()
            logger.warning(f"Attention hook unavailable ({exc!r}); using output_attentions fallback")


        self._capture_version = (
            "gemma_qknorm_v2"
            if any(
                getattr(a, "q_norm", None) is not None
                or getattr(a, "k_norm", None) is not None
                for a in self._hooked_attns
            )
            else "legacy_v1"
        )

    @property
    def capture_version(self) -> str:
        """当前捕获路径版本（``gemma_qknorm_v2`` 或 ``legacy_v1``）。"""
        return self._capture_version

    @property
    def scoring_layer_info(self) -> Dict[str, Any]:
        """供 metadata 记录：实际使用的层索引/策略/sliding 信息。"""
        return {
            "strategy": self._layer_strategy,
            "selected_layer_indices": list(self._selected_layer_indices),
            "note": self._layer_strategy_note,
        }

    def _select_layer_by_strategy(self, layers) -> Tuple[int, str]:
        "按策略选择 last_layer 聚合所用的物理层索引。"
        total = len(layers)
        if self._layer_strategy == "explicit_index":
            idx = int(self._layer_index)
            if not (0 <= idx < total):
                raise ValueError(
                    f"explicit layer_index {idx} out of range [0, {total})"
                )
            attn = getattr(layers[idx], "self_attn", None)
            sliding = getattr(attn, "is_sliding", None)
            return idx, f"explicit_index={idx} (is_sliding={sliding})"
        if self._layer_strategy == "last_full_attention":


            from models.arch_adapter import get_last_full_attention_layer
            idx = get_last_full_attention_layer(self._model)
            attn = getattr(layers[idx], "self_attn", None)
            return idx, f"last_full_attention -> index {idx} (class={type(attn).__name__})"
        if self._layer_strategy == "last_global":
            for i in range(total - 1, -1, -1):
                attn = getattr(layers[i], "self_attn", None)
                sliding = getattr(attn, "is_sliding", None)
                if sliding is None:
                    raise RuntimeError(
                        f"last_global: layer {i} has no is_sliding attribute; "
                        f"cannot verify layer type. Refusing to fall back to "
                        f"last_physical (use last_physical or explicit_index "
                        f"explicitly if that is intended)"
                    )
                if sliding is False:
                    return i, f"last_global -> index {i}"
            raise RuntimeError(
                "last_global: all layers are sliding-attention; no verified "
                "global/full-attention layer exists. Refusing to fall back to "
                "last_physical (use last_physical explicitly if intended)"
            )
        return total - 1, "last_physical"

    def _install_hooks(self) -> None:
        base = _unwrap_hf_model(self._model)
        layers = _get_layers(base)
        total = len(layers)
        if total == 0:
            raise RuntimeError("No transformer layers found")
        if self._explicit_layer_indices is not None:
            if max(self._explicit_layer_indices) >= total:
                raise ValueError(f"attention layer index exceeds model's {total} layers")
            selected = [layers[i] for i in self._explicit_layer_indices]
            self._selected_layer_indices = list(self._explicit_layer_indices)
            self._layer_strategy_note = "multi_layer_avg(explicit-indices)"
            for i, layer in zip(self._selected_layer_indices, selected):
                attn = getattr(layer, "self_attn", None)
                if attn is None or not self._probe_self_attn(attn):
                    raise RuntimeError(f"Layer {i} is not a supported full self-attention layer")
        elif self._attention_type == "last_layer":
            sel_idx, note = self._select_layer_by_strategy(layers)
            selected = [layers[sel_idx]]
            self._selected_layer_indices = [sel_idx]
            self._layer_strategy_note = note
        else:
            n = min(self._num_layers, total)
            selected = list(layers[-n:])
            self._selected_layer_indices = list(range(total - n, total))
            self._layer_strategy_note = "multi_layer_avg(last-N)"
        logger.info(
            f"Scoring attention layer: strategy={self._layer_strategy}, "
            f"indices={self._selected_layer_indices}, note={self._layer_strategy_note}"
        )

        for lyr in selected:
            attn = getattr(lyr, "self_attn", None)
            if attn is None:
                raise AttributeError("decoder layer has no 'self_attn'")
            if not self._probe_self_attn(attn):
                raise AttributeError("self_attn 非 Llama-family（无 q_proj/k_proj）；cannot use hook path")

            self._hooks.append(attn.register_forward_pre_hook(self._attn_pre_hook, with_kwargs=True))
            self._hooks.append(attn.q_proj.register_forward_hook(self._q_proj_hook))
            self._hooks.append(attn.k_proj.register_forward_hook(self._k_proj_hook))
            self._hooks.append(attn.register_forward_hook(self._attn_post_hook))
            self._hooked_attns.append(attn)

    def _probe_self_attn(self, attn: nn.Module) -> bool:
        "探测 self_attn：仅要求 q_proj/k_proj（识别 LlamaAttention 及变体）。"
        cls_name = type(attn).__name__
        if not (hasattr(attn, "q_proj") and hasattr(attn, "k_proj")):
            logger.warning(
                f"self_attn probe ({cls_name}): 缺少 q_proj/k_proj（非 Llama-family）；回退 output_attentions"
            )
            self._log_module_structure(attn)
            return False
        logger.info(f"Using response-query attention hook ({cls_name})")
        return True

    def _log_module_structure(self, attn: nn.Module) -> None:
        """打印 module 结构（class + 属性），便于继续适配新版本 transformers。"""
        cls_name = type(attn).__name__
        attrs = sorted(a for a in dir(attn) if not a.startswith("_"))
        logger.warning(
            f"module structure: class={cls_name}; "
            f"has num_heads={hasattr(attn, 'num_heads')} head_dim={hasattr(attn, 'head_dim')} "
            f"rotary_emb={hasattr(attn, 'rotary_emb')} config={hasattr(attn, 'config')}; attrs={attrs}"
        )

    def _cleanup_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._hooked_attns.clear()

    def close(self) -> None:
        """确定性移除 hooks，并释放捕获中的张量引用。"""
        self._capturing = False
        self._ctx = None
        self._cur_q = None
        self._cur_k = None
        self._cur_position_ids = None
        self._cur_position_embeddings = None
        self._cleanup_hooks()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


    def _attn_pre_hook(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        "self_attn forward 前：捕获 position_ids（供 RoPE 重算用）。"
        if not self._capturing:
            return

        hs = kwargs.get("hidden_states") if kwargs else (args[0] if args else None)
        if hs is not None and hasattr(module, "q_proj"):
            q_wt = module.q_proj.weight
            logger.debug(
                f"[attn_pre_hook] hidden_states.dtype={hs.dtype}, "
                f"q_proj.weight.dtype={q_wt.dtype}, match={hs.dtype == q_wt.dtype}"
            )
            if hs.dtype != q_wt.dtype:
                logger.warning(
                    f"[attn_pre_hook] DTYPE MISMATCH: hidden_states={hs.dtype} "
                    f"vs q_proj.weight={q_wt.dtype} — 模型 forward 将报错"
                )
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
            position_ids = bound.arguments.get("position_ids")
            position_embeddings = bound.arguments.get("position_embeddings")
        except (TypeError, ValueError):
            position_ids = kwargs.get("position_ids")
            position_embeddings = kwargs.get("position_embeddings")
        self._cur_position_ids = position_ids
        self._cur_position_embeddings = position_embeddings

    def _q_proj_hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """捕获 q_proj 输出（pre-RoPE Q，[B, S, H*d]）。不重算。不修改 output。"""
        if self._capturing:

            in_dt = inputs[0].dtype if inputs else None
            out_dt = output.dtype
            wt_dt = module.weight.dtype
            logger.debug(
                f"[q_proj_hook] input.dtype={in_dt}, output.dtype={out_dt}, "
                f"weight.dtype={wt_dt}, input==weight={in_dt == wt_dt}"
            )
            self._cur_q = output

    def _k_proj_hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """捕获 k_proj 输出（pre-RoPE K，[B, S, Hkv*d]）。不重算。不修改 output。"""
        if self._capturing:
            in_dt = inputs[0].dtype if inputs else None
            wt_dt = module.weight.dtype
            logger.debug(
                f"[k_proj_hook] input.dtype={in_dt}, output.dtype={output.dtype}, "
                f"weight.dtype={wt_dt}, input==weight={in_dt == wt_dt}"
            )
            self._cur_k = output

    def _attn_post_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        "self_attn forward 后：用捕获的 q_proj/k_proj 输出（pre-RoPE）算 response attention。"
        if not self._capturing or self._ctx is None:
            return
        q_out, k_out = self._cur_q, self._cur_k
        position_ids = self._cur_position_ids
        position_embeddings = self._cur_position_embeddings
        self._cur_q = self._cur_k = self._cur_position_ids = None
        self._cur_position_embeddings = None
        if q_out is None or k_out is None:
            logger.warning("attention capture: q_proj/k_proj 输出未捕获；该层跳过")
            self._ctx["capture_failed"] = True
            return
        try:


            with torch.autocast(device_type=q_out.device.type, enabled=False):
                q, k = self._reshape_and_rope(
                    module, q_out, k_out, position_ids, position_embeddings,
                )
                self._accumulate_layer(module, q, k, self._ctx)
        except Exception as e:
            logger.warning(f"attention capture (post_hook) failed: {e!r}", exc_info=True)
            self._ctx["capture_failed"] = True

    def _config_of(self, attn: nn.Module) -> Any:
        """取 config：优先 attn.config，否则 self._model.config。"""
        cfg = getattr(attn, "config", None)
        if cfg is None:
            cfg = getattr(self._model, "config", None)
        return cfg

    def _reshape_and_rope(
        self, attn: nn.Module, q_out: torch.Tensor, k_out: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        "把 q_proj/k_proj 输出（pre-RoPE）reshape + RoPE + GQA 展开。"

        q_out = q_out.float()
        k_out = k_out.float()
        B, S, _ = q_out.shape
        cfg = self._config_of(attn)
        H = int(getattr(attn, "num_heads", None) or getattr(cfg, "num_attention_heads", 0))
        Hkv = int(getattr(attn, "num_key_value_heads", None) or getattr(cfg, "num_key_value_heads", H))
        hidden = getattr(cfg, "hidden_size", None)
        d = int(getattr(attn, "head_dim", 0) or (hidden // H if hidden and H else 0))
        if H <= 0 or d <= 0:
            raise AttributeError(f"cannot determine num_heads({H})/head_dim({d}) from attn/config")


        if q_out.shape[-1] == H * d:
            q = q_out.view(B, S, H, d)
        elif q_out.shape[-1] == 2 * H * d:
            q = q_out.view(B, S, H, 2 * d)[..., :d]
        else:
            raise AttributeError(
                f"q_proj output dim {q_out.shape[-1]} matches neither H*d({H*d}) nor 2*H*d({2*H*d})"
            )
        q = q.transpose(1, 2)
        k = k_out.view(B, S, Hkv, d).transpose(1, 2)


        q_norm = getattr(attn, "q_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        k_norm = getattr(attn, "k_norm", None)
        if k_norm is not None:
            k = k_norm(k)
        if position_ids is None:
            am = self._ctx["attention_mask"]
            position_ids = am.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(am == 0, 0)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rope(q, k, cos.float(), sin.float())
        else:
            q, k = self._apply_rope_to_qk(attn, q, k, position_ids)
        if Hkv != H:
            k = k.repeat_interleave(H // Hkv, dim=1)
        return q, k

    def _apply_rope_to_qk(self, attn: nn.Module, q: torch.Tensor, k: torch.Tensor,
                          position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        "应用 RoPE，兼容新版 transformers（``rotary_emb`` 可能不在 ``self_attn`` 上）。"
        rotary_emb = getattr(attn, "rotary_emb", None)
        if rotary_emb is not None:
            cos, sin = rotary_emb(q, position_ids)

            cos = cos.float()
            sin = sin.float()
            return _apply_rope(q, k, cos, sin)

        config = getattr(self._model, "config", None)
        base = float(getattr(config, "rope_theta", 10000.0))
        scaling = getattr(config, "rope_scaling", None)
        if scaling is not None:
            logger.warning(
                "rotary_emb 缺失且 config.rope_scaling 存在；重算 RoPE 可能与 HF 不等价（仅无 scaling 时等价）"
            )
        d = q.shape[-1]
        dev = q.device
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=dev, dtype=torch.float32) / d))
        pos = position_ids.unsqueeze(-1).float()
        freqs = pos * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        return _apply_rope(q, k, cos, sin)

    def _accumulate_layer(self, attn: nn.Module, q: torch.Tensor, k: torch.Tensor,
                          ctx: Dict[str, Any]) -> None:
        "单层：逐样本算 response attention，累加 A_kji / A_kjj_prime 到 CPU 累加器。"
        B, H, S, d = q.shape
        scale = 1.0 / math.sqrt(d)
        attn_mask = ctx["attention_mask"]
        prompt_pos_list = ctx["prompt_pos"]
        response_pos_list = ctx["response_pos"]
        A_kji_acc = ctx["A_kji_acc"]
        A_kjj_prime_acc = ctx["A_kjj_prime_acc"]
        cols = torch.arange(S, device=q.device)


        window = None
        if getattr(attn, "is_sliding", False):
            window = getattr(attn, "sliding_window", None)
            if window is None:
                cfg = self._config_of(attn)
                window = getattr(cfg, "sliding_window", None)

        for i in range(B):
            rp = response_pos_list[i].to(q.device)
            pp = prompt_pos_list[i].to(q.device)


            qi = q[i][:, rp, :]

            scores = torch.matmul(qi.float(), k[i].transpose(-1, -2).float()) * scale

            real = attn_mask[i].to(q.device).bool()
            causal = cols[None, :] <= rp[:, None]
            if window is not None:
                causal = causal & ((rp[:, None] - cols[None, :]) < int(window))
            valid = causal & real[None, :]


            scores = scores.masked_fill(
                ~valid.unsqueeze(0), torch.finfo(scores.dtype).min,
            )
            attn_w = torch.softmax(scores, dim=-1)


            attn_w = torch.nan_to_num(attn_w.float(), nan=0.0, posinf=0.0, neginf=0.0)


            A_kji_acc[i] += attn_w[:, :, pp].mean(dim=0).cpu()
            A_kjj_prime_acc[i] += attn_w[:, :, rp].mean(dim=0).cpu()
        ctx["num_layers_captured"] += 1


    def record_attention_per_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        response_lengths: List[int],
        prompt_block_len: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        "逐样本记录 response-attention（变长序列安全，仅一次批量前向）。"
        batch_size = input_ids.shape[0]
        if len(prompt_lengths) != batch_size or len(response_lengths) != batch_size:
            raise ValueError(
                f"prompt/response lengths must have length {batch_size}, "
                f"got {len(prompt_lengths)}/{len(response_lengths)}"
            )
        for i in range(batch_size):
            if prompt_lengths[i] <= 0:
                raise ValueError(f"prompt_lengths[{i}] must be positive, got {prompt_lengths[i]}")
            if response_lengths[i] <= 0:
                raise ValueError(f"response_lengths[{i}] must be positive, got {response_lengths[i]}")

        if not self._use_hook:
            return self._record_via_output_attentions(
                input_ids, attention_mask, prompt_lengths, response_lengths, prompt_block_len
            )


        prompt_pos_list, response_pos_list = [], []
        for i in range(batch_size):
            pp, rp = real_prompt_response_positions(
                attention_mask[i], prompt_block_len=prompt_block_len,
                prompt_length=prompt_lengths[i], response_length=response_lengths[i],
            )
            prompt_pos_list.append(pp)
            response_pos_list.append(rp)


        A_kji_acc = [torch.zeros(rp.numel(), pp.numel(), dtype=torch.float32)
                     for pp, rp in zip(prompt_pos_list, response_pos_list)]
        A_kjj_prime_acc = [torch.zeros(rp.numel(), rp.numel(), dtype=torch.float32)
                           for rp in response_pos_list]
        self._ctx = {
            "attention_mask": attention_mask,
            "prompt_pos": prompt_pos_list,
            "response_pos": response_pos_list,
            "A_kji_acc": A_kji_acc,
            "A_kjj_prime_acc": A_kjj_prime_acc,
            "num_layers_captured": 0,
            "capture_failed": False,
        }

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
            logger.debug(f"[attention] before forward: allocated={mem_before / 1024**2:.1f} MiB")
        self._capturing = True
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                self._model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_attentions=False, output_hidden_states=False,
                )
        finally:
            self._capturing = False

        if cuda_ok:
            mem_after = torch.cuda.memory_allocated()
            mem_peak = torch.cuda.max_memory_allocated()
            logger.debug(
                f"[attention] after forward: allocated={mem_after / 1024**2:.1f} MiB, "
                f"peak={mem_peak / 1024**2:.1f} MiB (vs output_attentions 物化 [L,B,H,S,S])"
            )

        n = self._ctx["num_layers_captured"]
        failed = self._ctx["capture_failed"]
        self._ctx = None
        if n == 0 or failed:
            logger.warning("Hook path failed mid-capture; falling back to output_attentions=True")
            return self._record_via_output_attentions(
                input_ids, attention_mask, prompt_lengths, response_lengths, prompt_block_len
            )
        return [
            {"A_kji": A_kji_acc[i] / n, "A_kjj_prime": A_kjj_prime_acc[i] / n}
            for i in range(batch_size)
        ]

    def compute_response_prefix_ratio(self, A_kjj_prime: torch.Tensor) -> torch.Tensor:
        "计算响应前缀比率 ``c_{k,j} = sum_{j' < j} A_{k,j,j'}``。"
        response_len = A_kjj_prime.shape[-1]
        device = A_kjj_prime.device
        dtype = A_kjj_prime.dtype
        mask = torch.tril(
            torch.ones(response_len, response_len, device=device, dtype=dtype), diagonal=-1
        )
        A_masked = A_kjj_prime * mask.unsqueeze(0)
        return A_masked.sum(dim=-1)


    def _record_via_output_attentions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        response_lengths: List[int],
        prompt_block_len: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        "旧路径：``output_attentions=True`` 取全 attention 再切。"
        hf_model = _unwrap_hf_model(self._model)
        config = getattr(hf_model, "config", None)
        previous_impl = getattr(config, "_attn_implementation", None) if config is not None else None
        if config is not None:
            config._attn_implementation = "eager"
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_attentions=True, output_hidden_states=False,
                )
        finally:
            if config is not None:
                config._attn_implementation = previous_impl
        attention_weights = outputs["attentions"]
        if not attention_weights:
            raise RuntimeError("attention backend returned no weights in eager fallback")
        selected = self._select_layers(attention_weights)

        records: List[Dict[str, torch.Tensor]] = []
        for i in range(input_ids.shape[0]):
            prompt_pos, response_pos = real_prompt_response_positions(
                attention_mask[i], prompt_block_len=prompt_block_len,
                prompt_length=prompt_lengths[i], response_length=response_lengths[i],
            )
            prompt_pos = prompt_pos.to(selected.device)
            response_pos = response_pos.to(selected.device)
            a_block = selected[:, i, :, response_pos, :][:, :, :, prompt_pos]
            A_kji = torch.nan_to_num(
                a_block.mean(dim=1).mean(dim=0).float(), nan=0.0, posinf=0.0, neginf=0.0
            ).cpu()
            a_block_sq = selected[:, i, :, response_pos, :][:, :, :, response_pos]
            A_kjj_prime = torch.nan_to_num(
                a_block_sq.mean(dim=1).mean(dim=0).float(), nan=0.0, posinf=0.0, neginf=0.0
            ).cpu()
            records.append({"A_kji": A_kji, "A_kjj_prime": A_kjj_prime})
        del outputs, attention_weights, selected
        return records

    def verify_against_output_attentions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: List[int],
        response_lengths: List[int],
        prompt_block_len: Optional[int] = None,
    ) -> Dict[str, float]:
        "自检：对比 hook 路径与 ``output_attentions`` 旧路径的 A_kji / A_kjj_prime。"
        hook = self.record_attention_per_sample(
            input_ids, attention_mask, prompt_lengths, response_lengths, prompt_block_len
        )
        ref = self._record_via_output_attentions(
            input_ids, attention_mask, prompt_lengths, response_lengths, prompt_block_len
        )
        diffs = {"A_kji": 0.0, "A_kjj_prime": 0.0}
        for h, r in zip(hook, ref):
            diffs["A_kji"] = max(diffs["A_kji"], (h["A_kji"] - r["A_kji"]).abs().max().item())
            diffs["A_kjj_prime"] = max(
                diffs["A_kjj_prime"],
                (h["A_kjj_prime"] - r["A_kjj_prime"]).abs().max().item(),
            )
        return diffs

    def _select_layers(self, attention_weights) -> torch.Tensor:
        """根据 attention_type 选择层，返回 [num_selected_layers, batch, heads, seq, seq]。"""
        total_layers = len(attention_weights)
        explicit = getattr(self, "_explicit_layer_indices", None)
        if explicit is not None:
            if max(explicit) >= total_layers or any(attention_weights[i] is None for i in explicit):
                raise RuntimeError("Explicit full-attention layers are unavailable in fallback output")
            return torch.stack([attention_weights[i] for i in explicit], dim=0)
        if self._attention_type == "last_layer":


            indices = getattr(self, "_selected_layer_indices", [])
            strategy = getattr(self, "_layer_strategy", "last_physical")
            if indices:
                if len(indices) != 1:
                    raise RuntimeError("Single-layer fallback requires one resolved layer")
                index = indices[0]
            elif strategy == "explicit_index":
                index = getattr(self, "_layer_index", None)
            elif strategy == "last_physical":
                index = total_layers - 1
            else:
                raise RuntimeError(
                    f"Cannot resolve {strategy} in fallback without a verified layer index"
                )
            if (index is None or not 0 <= index < total_layers
                    or attention_weights[index] is None):
                raise RuntimeError("Selected attention layer is unavailable in fallback output")
            return attention_weights[index].unsqueeze(0)
        if self._num_layers > total_layers:
            raise ValueError(f"Requested {self._num_layers} layers but model only has {total_layers}")
        return torch.stack(attention_weights[-self._num_layers:], dim=0)

    @property
    def attention_type(self) -> str:
        return self._attention_type

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def use_hook(self) -> bool:
        """是否走内存高效 hook 路径（False=回退到 output_attentions）。"""
        return self._use_hook
