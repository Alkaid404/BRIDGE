"模型加载、梯度方向计算与 VJP 参数收集（Full Fine-tuning + Last-N-Layer VJP）。"

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from utils import get_logger, clear_gpu_cache, ensure_dir

logger = get_logger("bridge")


def _restore_declared_tied_weights(
    model: nn.Module, preserve_dispatch_hooks: bool = False
) -> None:
    "Restore input/output embedding sharing declared by the HF config."
    config = getattr(model, "config", None)
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError(
            "Model declares tie_word_embeddings=True but does not expose both "
            "input and output embeddings"
        )
    if input_embeddings.weight is output_embeddings.weight:
        return

    input_device = input_embeddings.weight.device
    output_device = output_embeddings.weight.device
    if input_device != output_device:
        raise RuntimeError(
            "Cannot restore tied embeddings placed on different devices: "
            f"input={input_device}, output={output_device}"
        )

    if preserve_dispatch_hooks:


        logger.warning(
            "Checkpoint contains untied input/output embeddings although "
            "tie_word_embeddings=True; synchronizing values without rebinding "
            "Accelerate dispatch hooks"
        )
        with torch.no_grad():
            output_embeddings.weight.copy_(input_embeddings.weight)
        return

    logger.warning(
        "Checkpoint contains untied input/output embeddings although "
        "tie_word_embeddings=True; restoring the declared weight sharing"
    )
    model.tie_weights()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError("Model.tie_weights() did not restore input/output weight sharing")


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """返回 Transformer 层列表，按常见路径探测，不再维护架构注册表。"""
    for candidate in (
        "model.model.layers",
        "model.model.language_model.layers",
        "model.language_model.layers",
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    ):
        obj: Any = model
        try:
            for part in candidate.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        "Cannot locate transformer layers (tried model.model.layers / "
        "model.model.language_model.layers / model.language_model.layers / "
        "model.layers / transformer.h / "
        "model.decoder.layers / gpt_neox.layers)"
    )


def collect_vjp_parameters(
    model: nn.Module,
    num_last_layers: int = 4,
    layer_indices: Optional[List[int]] = None,
) -> List[Any]:
    "返回指定 Transformer 层的全部 (name, param)。"
    layers = _get_layers(model)
    all_layers = list(layers)
    n = len(all_layers)
    if layer_indices is not None:
        idx = [int(i) for i in layer_indices]
        if not idx:
            raise ValueError("layer_indices must be a non-empty list")
        if len(set(idx)) != len(idx):
            raise ValueError(f"layer_indices contains duplicates: {layer_indices}")
        bad = [i for i in idx if i < 0 or i >= n]
        if bad:
            raise ValueError(f"layer_indices out of range [0, {n}): {bad}")
        selected_layers = [all_layers[i] for i in sorted(idx)]
    else:
        if num_last_layers > n:
            logger.warning(f"num_last_layers={num_last_layers} > n_layers={n}; using all {n} layers")
            num_last_layers = n
        selected_layers = all_layers[n - num_last_layers:]

    mod_name = {id(m): name for name, m in model.named_modules()}
    params: List[Any] = []
    for layer in selected_layers:
        prefix = mod_name.get(id(layer))
        for pname, p in layer.named_parameters(recurse=True):
            full = f"{prefix}.{pname}" if prefix else pname
            params.append((full, p))
    return params


def _selected_layer_indices(
    model: nn.Module,
    num_last_layers: int = 4,
    layer_indices: Optional[List[int]] = None,
) -> List[int]:
    """与 ``collect_vjp_parameters`` 完全一致的层索引解析（日志/断言共用）。"""
    n = len(_get_layers(model))
    if layer_indices is not None:
        idx = sorted({int(i) for i in layer_indices})
        if not idx or idx[0] < 0 or idx[-1] >= n:
            raise ValueError(f"layer_indices out of range [0, {n}): {layer_indices}")
        return idx
    return list(range(max(0, n - num_last_layers), n))


def log_full_ft_setup(
    model: nn.Module,
    validation_direction: Optional[torch.Tensor] = None,
    num_last_layers: int = 4,
    layer_indices: Optional[List[int]] = None,
) -> None:
    """打印 Full Fine-tuning + Last-N-VJP 配置（确认训练与评分使用不同参数）。"""
    sel_indices = _selected_layer_indices(model, num_last_layers, layer_indices)
    vjp_params = collect_vjp_parameters(model, num_last_layers, layer_indices)
    vjp_num = sum(p.numel() for _, p in vjp_params)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: trainable={trainable:,}/{total:,}; "
        f"VJP={vjp_num:,} in layers {sel_indices[0]}-{sel_indices[-1]}"
        f" (indices={sel_indices})"
    )
    if validation_direction is not None:
        logger.debug(f"Validation direction: {validation_direction.numel():,} elements")


def assert_vjp_direction_consistent(
    model: nn.Module,
    validation_direction: torch.Tensor,
    num_last_layers: int = 4,
    layer_indices: Optional[List[int]] = None,
) -> None:
    """启动期一致性检查：validation_direction 维度必须 == 选定层参数总数。"""
    vjp_params = collect_vjp_parameters(model, num_last_layers, layer_indices)
    expected = sum(p.numel() for _, p in vjp_params)
    actual = int(validation_direction.shape[0])
    if actual != expected:
        raise RuntimeError(
            f"validation_direction len ({actual}) != VJP params numel ({expected}). "
            f"方向与 VJP 参数维度不一致：确认 compute_validation_direction 与 VJPComputer "
            f"均使用 collect_vjp_parameters(num_last_layers={num_last_layers}, "
            f"layer_indices={layer_indices})。"
        )


def apply_trainable_layer_restriction(
    model: nn.Module,
    layer_indices: List[int],
) -> Dict[str, Any]:
    "冻结除指定 decoder 层之外的全部参数（Stage 5 层子集训练）。"
    layers = _get_layers(model)
    n = len(layers)
    idx = sorted({int(i) for i in layer_indices})
    if not idx or idx[0] < 0 or idx[-1] >= n:
        raise ValueError(f"layer_indices out of range [0, {n}): {layer_indices}")
    keep_ids = {id(layers[i]) for i in idx}

    trainable_by_layer: Dict[str, int] = {}
    decoder_param_ids = set()
    for i, layer in enumerate(layers):
        n_params = 0
        for _, p in layer.named_parameters(recurse=True):
            decoder_param_ids.add(id(p))
            if id(layer) in keep_ids:
                p.requires_grad = True
                n_params += p.numel()
            else:
                p.requires_grad = False
        trainable_by_layer[str(i)] = n_params
    non_decoder_frozen = 0
    for p in model.parameters():
        if id(p) not in decoder_param_ids:
            if p.requires_grad:
                non_decoder_frozen += p.numel()
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stats = {
        "trainable_layer_indices": idx,
        "trainable_params_per_layer": trainable_by_layer,
        "non_decoder_frozen_params": non_decoder_frozen,
        "trainable_params": trainable,
        "total_params": total,
    }
    logger.info(
        f"Trainable layer restriction: trainable={trainable:,}/{total:,} "
        f"on decoder layers {idx}; non-decoder frozen={non_decoder_frozen:,}"
    )
    return stats


def flatten_gradients(params: Any) -> torch.Tensor:
    "将参数梯度展平为单个一维向量。"
    parts: List[torch.Tensor] = []
    target_device: Optional[torch.device] = None
    for p in params:
        if isinstance(p, nn.Parameter) and p.grad is not None:
            target_device = p.grad.device
            break
        if isinstance(p, torch.Tensor) and not isinstance(p, nn.Parameter):
            target_device = p.device
            break
    if target_device is None:
        for p in params:
            if isinstance(p, nn.Parameter):
                target_device = p.device
                break
    if target_device is None:
        target_device = torch.device("cpu")

    for p in params:
        if isinstance(p, torch.Tensor) and not isinstance(p, nn.Parameter):
            parts.append(p.reshape(-1).to(target_device))
        elif p.grad is None:
            parts.append(torch.zeros(p.numel(), device=target_device, dtype=p.dtype))
        else:
            parts.append(p.grad.reshape(-1).to(target_device))

    if not parts:
        raise ValueError("params must contain at least one element")
    return torch.cat(parts)


class BridgeModel(nn.Module):
    "因果语言模型的统一包装（Full fine-tuning，无 LoRA/adapter）。"

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Union[str, Dict[str, Any], None] = "auto",
        trust_remote_code: bool = True,
        inference_only: bool = False,
    ):
        super().__init__()
        self._model_name_or_path = model_name_or_path
        self._torch_dtype = torch_dtype
        self._device_map = device_map
        self._trust_remote_code = trust_remote_code
        self._inference_only = inference_only

        logger.info(f"Loading model from '{model_name_or_path}'")
        self._config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )


        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
            output_attentions=False,
            output_hidden_states=False,
        )
        _restore_declared_tied_weights(
            self._model, preserve_dispatch_hooks=self._inference_only
        )
        for param in self._model.parameters():
            param.requires_grad = True

        logger.info(f"Model ready, dtype={torch_dtype}")

    def __getattr__(self, name):
        "未在 BridgeModel 上定义的属性委托给底层 HF 模型。"
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            model = super().__getattr__("_model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(model, name)


    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> Any:
        return self._config

    @property
    def input_device(self) -> torch.device:
        """前向输入应放置的设备（多 GPU dispatch 下为根模块/嵌入层所在设备）。"""
        hf_map = getattr(self._model, "hf_device_map", None)
        if hf_map:
            root = hf_map.get("")
            if root is not None:
                return torch.device(root)
            try:
                return torch.device(next(iter(hf_map.values())))
            except (StopIteration, TypeError):
                pass
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """前向传播，返回字典；张量保留在 HF 模型的原生输出设备。"""
        forward_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if logits_to_keep is not None:
            forward_kwargs["logits_to_keep"] = logits_to_keep
        outputs = self._model(**forward_kwargs)
        result: Dict[str, Any] = {"logits": outputs.logits}
        if labels is not None and outputs.loss is not None:
            result["loss"] = outputs.loss
        if output_attentions and outputs.attentions is not None:
            result["attentions"] = outputs.attentions
        if output_hidden_states and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states
        return result

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算标量损失；``loss_mask==0`` 的位置标签置 -100 后由 HF 内部忽略。"""
        if labels is None:
            labels = input_ids.clone()
        if loss_mask is not None:
            labels = labels.clone()
            labels[loss_mask == 0] = -100
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return outputs.loss


    def save(self, path: str) -> None:
        """保存完整模型（Full fine-tuning，无 adapter）。"""
        save_dir = ensure_dir(path)
        self._model.save_pretrained(str(save_dir), safe_serialization=True)
        logger.info(f"Model saved to '{save_dir}'")

    def load(self, path: str) -> None:
        """加载完整 HF 模型 checkpoint（Full fine-tuning，无 adapter）。"""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: '{path}'")
        logger.info(f"[checkpoint] loading full model from '{load_path}'")
        old_model = self._model
        self._model = None
        del old_model
        gc.collect()
        clear_gpu_cache()
        self._model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
            trust_remote_code=self._trust_remote_code,
            attn_implementation="sdpa",
            output_attentions=False,
            output_hidden_states=False,
        )
        _restore_declared_tied_weights(
            self._model, preserve_dispatch_hooks=self._inference_only
        )
        for param in self._model.parameters():
            param.requires_grad = True
        logger.info(f"Full model loaded from '{load_path}'")

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        "标准 ``nn.Module.num_parameters`` 接口（HF Trainer 算 FLOPs 时调用"
        return self._model.num_parameters(exclude_embeddings=exclude_embeddings)


def _zero_grad(parameters: List[nn.Parameter]) -> None:
    for p in parameters:
        if p.grad is not None:
            p.grad.zero_()


def compute_validation_direction(
    model: BridgeModel,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    "计算验证目标方向 ``v = ∇L_target(θ̄)``。"
    params = [p for _, p in collect_vjp_parameters(model, layer_indices=layer_indices)]
    device = model.input_device

    total_grad: Optional[torch.Tensor] = None
    num_samples = 0
    num_batches = 0

    model.eval()
    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    if total is not None and max_batches is not None:
        total = min(total, max_batches)
    progress = tqdm(total=total, desc="Validation direction", unit="batch")
    try:
        with torch.enable_grad():
            for batch in dataloader:
                if max_batches is not None and num_batches >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss = model.compute_loss(input_ids, attention_mask, labels)
                grads = torch.autograd.grad(
                    loss, params, retain_graph=False, create_graph=False, allow_unused=True,
                )
                grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


                batch_grad = torch.cat([g.detach().reshape(-1).cpu().float() for g in grads])
                bs = input_ids.shape[0]
                total_grad = batch_grad * bs if total_grad is None else total_grad + batch_grad * bs
                num_samples += bs
                num_batches += 1
                progress.update(1)

                del loss, grads, batch_grad, input_ids, attention_mask, labels
    finally:
        progress.close()

    if total_grad is None or num_samples == 0:
        raise ValueError("dataloader yielded no batches")

    model.zero_grad(set_to_none=True)
    clear_gpu_cache()

    direction = total_grad / num_samples
    logger.debug(
        f"Validation direction: {num_batches} batches, {num_samples} samples, "
        f"norm={direction.norm().item():.6f}"
    )
    return direction


def compute_retention_direction(
    model: BridgeModel,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    "计算保留方向（结构与 ``compute_validation_direction`` 相同，作用于保留集）。"
    params = [p for _, p in collect_vjp_parameters(model, layer_indices=layer_indices)]
    device = model.input_device

    total_grad: Optional[torch.Tensor] = None
    num_samples = 0
    num_batches = 0

    model.eval()
    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    if total is not None and max_batches is not None:
        total = min(total, max_batches)
    progress = tqdm(total=total, desc="Retention direction", unit="batch")
    try:
        with torch.enable_grad():
            for batch in dataloader:
                if max_batches is not None and num_batches >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss = model.compute_loss(input_ids, attention_mask, labels)
                grads = torch.autograd.grad(
                    loss, params, retain_graph=False, create_graph=False, allow_unused=True,
                )
                grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


                batch_grad = torch.cat([g.detach().reshape(-1).cpu().float() for g in grads])
                bs = input_ids.shape[0]
                total_grad = batch_grad * bs if total_grad is None else total_grad + batch_grad * bs
                num_samples += bs
                num_batches += 1
                progress.update(1)

                del loss, grads, batch_grad, input_ids, attention_mask, labels
    finally:
        progress.close()

    if total_grad is None or num_samples == 0:
        raise ValueError("dataloader yielded no batches")

    model.zero_grad(set_to_none=True)
    clear_gpu_cache()

    direction = total_grad / num_samples
    logger.info(
        f"Retention direction: {num_batches} batches, {num_samples} samples, "
        f"norm={direction.norm().item():.6f}"
    )
    return direction


def combine_target_retention_directions(
    g_target: torch.Tensor,
    g_ret: torch.Tensor,
    lam: float,
    grad_mode: str = "raw",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    "合并目标方向与保留方向，返回 ``(v, stats)``。"
    if g_target.shape != g_ret.shape:
        raise ValueError(
            f"direction shape mismatch: target={tuple(g_target.shape)} "
            f"vs retention={tuple(g_ret.shape)}"
        )
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    g_target = g_target.float()
    g_ret = g_ret.float().to(g_target.device)

    target_norm = g_target.norm().item()
    retention_norm = g_ret.norm().item()
    denom = target_norm * retention_norm
    cosine = float((g_target @ g_ret).item() / denom) if denom > 0 else 0.0

    if grad_mode == "raw":
        scale = 1.0
    elif grad_mode == "match_norm":
        scale = target_norm / retention_norm if retention_norm > 0 else 0.0
    elif grad_mode == "unit_mix":


        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"unit_mix requires lambda in [0,1], got {lam}")
        eps = 1e-12
        t_hat = g_target / (target_norm + eps)
        r_hat = g_ret / (retention_norm + eps)
        v = (1.0 - lam) * t_hat + lam * r_hat
        v64 = v.double()
        th64 = t_hat.double()
        rh64 = r_hat.double()
        v_norm = v64.norm().item()
        stats = {
            "target_grad_norm": g_target.double().norm().item(),
            "retention_grad_norm": g_ret.double().norm().item(),
            "combined_grad_norm": v_norm,
            "cos_target_retention": (
                float((g_target.double() @ g_ret.double()).item()
                      / (g_target.double().norm().item() * g_ret.double().norm().item()))
                if target_norm > 0 and retention_norm > 0 else 0.0
            ),
            "cos_mix_target": float((v64 @ th64).item() / (v_norm * th64.norm().item())) if v_norm > 0 else 0.0,
            "cos_mix_retention": float((v64 @ rh64).item() / (v_norm * rh64.norm().item())) if v_norm > 0 else 0.0,
            "lambda_ret": float(lam),
            "retention_scale": 1.0 / (retention_norm + eps),
            "target_scale": 1.0 / (target_norm + eps),
            "effective_retention_coeff": float(lam),
            "grad_mode": grad_mode,
        }
        return v, stats
    else:
        raise ValueError(f"grad_mode must be 'raw', 'match_norm' or 'unit_mix', got {grad_mode!r}")

    v = g_target + (lam * scale) * g_ret
    stats = {
        "target_grad_norm": target_norm,
        "retention_grad_norm": retention_norm,
        "combined_grad_norm": v.norm().item(),
        "cos_target_retention": cosine,
        "lambda_ret": float(lam),
        "retention_scale": float(scale),
        "effective_retention_coeff": float(lam * scale),
        "grad_mode": grad_mode,
    }
    return v, stats
