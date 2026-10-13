"向量-雅可比积 (VJP)：逐 token 验证方向评分 ``a_{k,j} = v^T · g_{k,j}``。"

import gc
import inspect
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, vjp, vmap

from models.utils import collect_vjp_parameters
from scoring.attention import real_prompt_response_positions
from utils import get_logger

logger = get_logger("bridge.scoring")


def extract_logits(outputs) -> torch.Tensor:
    "从模型 forward 返回中提取 logits，兼容三类返回："
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError(f"Cannot extract logits from {type(outputs).__name__}: {outputs!r}")


class VJPComputer:
    "计算逐 token 的验证方向评分 ``a_{k,j} = v^T · g_{k,j}``。"

    def __init__(self, model: nn.Module, validation_direction: torch.Tensor,
                 layer_indices: Optional[List[int]] = None):
        if validation_direction.dim() != 1:
            raise ValueError(f"validation_direction must be 1-D, got shape {validation_direction.shape}")
        self._model = model
        self._layer_indices = None if layer_indices is None else [int(i) for i in layer_indices]

        self._validation_direction = validation_direction.detach().float()
        self._parameters = self._resolve_parameters()

        self._vjp_names, self._vjp_numels, self._vjp_params_dict = self._build_param_mapping()

        _expected = sum(self._vjp_numels)
        _actual = int(self._validation_direction.shape[0])
        if _actual != _expected:
            raise RuntimeError(
                f"validation_direction len ({_actual}) != VJP params numel ({_expected}). "
                f"方向与 VJP 参数维度不一致：二者必须均来自 collect_vjp_parameters"
                f"（layer_indices={self._layer_indices}）。"
            )


        self._fc_tie_weights = "tie_weights" in inspect.signature(functional_call).parameters
        logger.debug(
            f"VJPComputer: num_params={sum(p.numel() for p in self._parameters):,}, "
            f"val_dir_norm={self._validation_direction.norm().item():.6f}"
        )
        self._check_validation_direction()

    def _check_validation_direction(self) -> None:
        """校验 ``v``：打印 norm/min/max/nan/inf；含 NaN/Inf 直接报错（否则所有 score 都会被污染）。"""
        v = self._validation_direction
        n_nan = int(torch.isnan(v).sum().item())
        n_inf = int(torch.isinf(v).sum().item())
        logger.debug(
            f"validation_direction: norm={float(v.norm().item()):.6f} "
            f"min={float(v.min().item()):.6f} max={float(v.max().item()):.6f} "
            f"nan={n_nan} inf={n_inf} dtype={v.dtype}"
        )
        if n_nan or n_inf:
            raise ValueError(
                f"validation_direction contains {n_nan} NaN / {n_inf} Inf; "
                "无法计算有效 token score。请检查 compute_validation_direction 的梯度与 flatten。"
            )

    def _resolve_parameters(self) -> List[nn.Parameter]:
        """解析求导参数：选定 Transformer 层（collect_vjp_parameters，默认最后 4 层）。"""
        params = [p for _, p in collect_vjp_parameters(self._model, layer_indices=self._layer_indices)]
        if not params:
            raise RuntimeError("collect_vjp_parameters returned no parameters; cannot build VJPComputer")
        logger.debug(f"Using {len(params)} VJP parameter tensors")
        return params

    def _build_param_mapping(self) -> Tuple[List[str], List[int], Dict[str, torch.Tensor]]:
        """建立 (名称, numel) 顺序与 detached 参数字典，顺序与 ``v`` 对齐。"""
        items = collect_vjp_parameters(self._model, layer_indices=self._layer_indices)
        names = [n for n, _ in items]
        numels = [p.numel() for _, p in items]


        base_dtype = next(self._model.parameters()).dtype
        pdict = {n: p.detach().to(base_dtype) for n, p in items}
        logger.debug(f"VJP param_dict: {len(pdict)} params, dtype={base_dtype}")
        return names, numels, pdict


    def compute_validation_scores_per_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: List[int],
        response_lengths: List[int],
        prompt_block_len: Optional[int] = None,
        chunk_size: int = 64,
    ) -> List[torch.Tensor]:
        "逐样本计算 ``a_{k,j}``（变长序列安全，chunked batched VJP）。"
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
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        logger.debug(f"Using chunked batched VJP (chunk_size={chunk_size})")
        try:
            return self._compute_chunked(
                input_ids, attention_mask, labels,
                prompt_lengths, response_lengths, prompt_block_len, chunk_size,
            )
        except Exception as e:

            logger.error(f"Chunked batched VJP failed: {e!r}", exc_info=True)
            raise

    def _compute_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: List[int],
        response_lengths: List[int],
        prompt_block_len: Optional[int],
        chunk_size: int,
    ) -> List[torch.Tensor]:
        """chunked batched VJP 实现。"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        v = self._validation_direction.to(device)


        response_pos_list: List[torch.Tensor] = []
        for i in range(batch_size):
            _, rp = real_prompt_response_positions(
                attention_mask[i], prompt_block_len=prompt_block_len,
                prompt_length=prompt_lengths[i], response_length=response_lengths[i],
            )
            response_pos_list.append(rp)
        offsets = [0]
        for rp in response_pos_list:
            offsets.append(offsets[-1] + rp.numel())
        total_R = offsets[-1]
        if total_R == 0:
            return [torch.zeros(0, device=device, dtype=torch.float32) for _ in range(batch_size)]


        response_pos_list_dev = [rp.to(device) for rp in response_pos_list]
        labels_dev = labels.to(device)

        def loss_fn(pd: Dict[str, torch.Tensor]) -> torch.Tensor:


            fc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask,
                         "output_attentions": False, "output_hidden_states": False}
            if self._fc_tie_weights:
                outputs = functional_call(self._model, pd, (), fc_kwargs, tie_weights=False)
            else:
                outputs = functional_call(self._model, pd, (), fc_kwargs)
            logits = extract_logits(outputs)
            flat_losses = []
            for i, rp in enumerate(response_pos_list_dev):
                if (rp <= 0).any():
                    raise RuntimeError("response positions must be > 0 for causal-LM VJP")
                rp_logits = (rp - 1).to(logits.device)
                rp_labels = rp.to(labels_dev.device)
                resp_logits = logits[i, rp_logits, :]
                resp_labels = labels_dev[i, rp_labels]


                nll = F.cross_entropy(
                    resp_logits.float().reshape(-1, resp_logits.shape[-1]),
                    resp_labels.reshape(-1),
                    reduction="none", ignore_index=-100,
                )
                flat_losses.append(nll)
            return torch.cat(flat_losses)


        logger.debug(f"[MEM] before vjp capture: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        losses, vjp_fn = vjp(loss_fn, self._vjp_params_dict)
        logger.debug(f"[MEM] after vjp capture: {torch.cuda.memory_allocated()/1e9:.1f}GB")


        bad_mask = torch.isnan(losses) | torch.isinf(losses)
        n_bad = int(bad_mask.sum().item())
        if n_bad:
            bad_positions = bad_mask.nonzero(as_tuple=True)[0].tolist()
            raise FloatingPointError(
                f"VJP FP32 token loss contains {n_bad}/{total_R} NaN/Inf values "
                f"at flat positions {bad_positions[:8]}; refusing to alter scores"
            )
        logger.debug("VJP: all FP32 token losses finite")
        good_positions = torch.arange(total_R, device=device)


        scores_flat = torch.zeros(total_R, device=device, dtype=torch.float32)
        n_good = int(good_positions.numel())
        loss_dtype = losses.dtype
        n_grad_retried = 0
        for c_start in range(0, n_good, chunk_size):
            chunk_pos = good_positions[c_start:c_start + chunk_size]
            C = int(chunk_pos.numel())

            basis = torch.zeros(C, total_R, device=device, dtype=loss_dtype)
            basis[torch.arange(C, device=device), chunk_pos] = 1.0

            logger.debug(f"[MEM] before vmap: {torch.cuda.memory_allocated()/1e9:.1f}GB")
            grads = vmap(vjp_fn)(basis)
            logger.debug(f"[MEM] after vmap: {torch.cuda.memory_allocated()/1e9:.1f}GB")
            grads_dict = grads[0] if isinstance(grads, tuple) else grads
            chunk_scores, grad_bad_token = self._project_batched_grads(
                grads_dict, C, v,
            )
            bad_local = grad_bad_token.nonzero(as_tuple=True)[0].tolist()
            del grads, grads_dict
            if bad_local:
                n_grad_retried += len(bad_local)
                logger.warning(
                    f"Batched VJP produced {len(bad_local)} non-finite rows; "
                    "retrying those tokens individually"
                )
                for local_idx in bad_local:
                    flat_pos = int(chunk_pos[local_idx].item())
                    cotangent = torch.zeros(total_R, device=device, dtype=loss_dtype)
                    cotangent[flat_pos] = 1.0
                    retry_grads = vjp_fn(cotangent)
                    retry_dict = retry_grads[0] if isinstance(retry_grads, tuple) else retry_grads
                    retry_batched = {
                        name: grad.unsqueeze(0) for name, grad in retry_dict.items()
                    }
                    retry_score, retry_bad = self._project_batched_grads(
                        retry_batched, 1, v,
                    )
                    if bool(retry_bad[0].item()) or not bool(torch.isfinite(retry_score[0]).item()):
                        raise FloatingPointError(
                            f"VJP gradient remains NaN/Inf at flat token position {flat_pos} "
                            "after individual retry; refusing to alter its score"
                        )
                    chunk_scores[local_idx] = retry_score[0]
                    del cotangent, retry_grads, retry_dict, retry_batched, retry_score, retry_bad
            scores_flat[chunk_pos] = chunk_scores
            del basis, chunk_scores, grad_bad_token
            gc.collect()


        a_kj_list: List[torch.Tensor] = []
        for i in range(batch_size):
            s = scores_flat[offsets[i]:offsets[i + 1]]
            nan = torch.isnan(s) | torch.isinf(s)
            if nan.any():
                raise FloatingPointError(
                    f"VJP projection produced {int(nan.sum().item())} NaN/Inf scores "
                    f"for sample {i}; refusing to alter scores"
                )
            a_kj_list.append(s.detach().cpu().clone())
        logger.debug(
            f"VJP done: total_R={total_R}, retried_grad_rows={n_grad_retried}"
        )
        del losses, vjp_fn, scores_flat, v, bad_mask, good_positions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return a_kj_list

    def _project_batched_grads(
        self, grads_dict: Dict[str, torch.Tensor], C: int, direction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """逐参数即时计算 ``sum_p g_p @ v_p``，避免物化拼接后的 ``[C, P]`` 副本。"""
        scores = torch.zeros(C, device=direction.device, dtype=torch.float32)
        bad = torch.zeros(C, device=direction.device, dtype=torch.bool)
        offset = 0
        for name, numel in zip(self._vjp_names, self._vjp_numels):
            g = grads_dict.get(name)
            v_part = direction[offset:offset + numel].float()
            offset += numel
            if g is None:
                continue


            g_flat = g.detach().reshape(C, -1).to(direction.device).float()
            bad |= ~torch.isfinite(g_flat).all(dim=1)
            scores += torch.nan_to_num(g_flat, nan=0.0, posinf=0.0, neginf=0.0) @ v_part
            del g_flat, v_part
        bad |= ~torch.isfinite(scores)
        return scores, bad

    def _flatten_batched_grads(self, grads_dict: Dict[str, torch.Tensor], C: int,
                               target_device: torch.device) -> torch.Tensor:
        "将 ``vmap(vjp_fn)`` 返回的批量梯度 dict 展平为 ``[C, P]``，与 ``v`` 顺序对齐。"
        parts: List[torch.Tensor] = []
        for k, name in enumerate(self._vjp_names):
            g = grads_dict.get(name)
            if g is None:
                parts.append(torch.zeros(C, self._vjp_numels[k], device=target_device, dtype=torch.float32))
            else:

                g_f32 = torch.nan_to_num(
                    g.reshape(C, -1).to(target_device).float(),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                parts.append(g_f32)
        return torch.cat(parts, dim=1)


    def compute_validation_scores_per_sample_reference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_lengths: List[int],
        response_lengths: List[int],
        prompt_block_len: Optional[int] = None,
    ) -> List[torch.Tensor]:
        "原逐 token 实现：1 次前向 + 逐样本逐位置 ``torch.autograd.grad``。"
        batch_size = input_ids.shape[0]
        device = input_ids.device
        v = self._validation_direction.to(device)


        outputs = self._model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=False, output_hidden_states=False,
        )
        logits = outputs["logits"]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        samples_data: List = []
        total_positions = 0
        for i in range(batch_size):
            _, response_pos = real_prompt_response_positions(
                attention_mask[i], prompt_block_len=prompt_block_len,
                prompt_length=prompt_lengths[i], response_length=response_lengths[i],
            )
            if (response_pos <= 0).any():
                raise RuntimeError("response positions must be > 0 for causal-LM VJP")
            resp_logits = logits[i, response_pos.to(logits.device) - 1, :]
            resp_labels = labels[i, response_pos.to(labels.device)]
            samples_data.append((resp_logits, resp_labels, response_pos.numel()))
            total_positions += response_pos.numel()

        a_kj_list: List[torch.Tensor] = []
        pos_done = 0
        for resp_logits, resp_labels, rl in samples_data:
            a_kj_i = torch.zeros(rl, device=device, dtype=torch.float32)
            for j in range(rl):
                valid = (resp_labels[j] != -100).float()
                pos_loss = loss_fct(
                    resp_logits[j:j + 1].float(), resp_labels[j:j + 1].view(1),
                ).squeeze() * valid
                pos_done += 1
                retain = pos_done < total_positions
                grads = torch.autograd.grad(
                    pos_loss, self._parameters, retain_graph=retain, allow_unused=True,
                )
                grad_flat_parts = []
                for g, p in zip(grads, self._parameters):
                    if g is not None:
                        g_f32 = g.reshape(-1).to(device).float()
                        if not torch.isfinite(g_f32).all():
                            raise FloatingPointError("reference VJP produced NaN/Inf gradient")
                        grad_flat_parts.append(g_f32)
                    else:
                        grad_flat_parts.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
                grad_flat = torch.cat(grad_flat_parts)
                a_kj_i[j] = torch.dot(v, grad_flat)
            a_kj_list.append(a_kj_i)

        del outputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, a in enumerate(a_kj_list):
            nan_mask = torch.isnan(a)
            if nan_mask.any():
                logger.warning(f"Replacing {nan_mask.sum().item()} NaN scores (sample {i}) with 0.0")
                a_kj_list[i] = torch.where(nan_mask, torch.zeros_like(a), a)
        return a_kj_list

    def compute_validation_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_length: int,
        response_length: int,
    ) -> torch.Tensor:
        "批量计算 ``a_{k,j}``（固定 prompt/response 长度，``[batch, response_len]``）。"
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device
        if prompt_length <= 0 or response_length <= 0 or prompt_length + response_length > seq_len:
            raise ValueError(f"Invalid lengths: prompt={prompt_length}, response={response_length}, seq={seq_len}")

        v = self._validation_direction.to(device)
        a_kj = torch.zeros(batch_size, response_length, device=device, dtype=torch.float32)

        outputs = self._model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=False, output_hidden_states=False,
        )
        logits = outputs["logits"]
        response_start = prompt_length
        response_end = prompt_length + response_length
        resp_logits = logits[:, response_start - 1:response_end - 1, :]
        resp_labels = labels[:, response_start:response_end]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        flat_logits = resp_logits.reshape(-1, resp_logits.shape[-1])
        flat_labels = resp_labels.reshape(-1)
        per_pos_loss = loss_fct(flat_logits.float(), flat_labels)
        valid_mask = (flat_labels != -100).float()
        per_pos_loss = per_pos_loss * valid_mask

        for j in range(response_length):
            pos_loss = per_pos_loss[j::response_length]
            valid = valid_mask[j::response_length]
            n_valid = valid.sum()
            if n_valid == 0:
                continue
            mean_loss_j = pos_loss.sum() / n_valid
            grads = torch.autograd.grad(
                mean_loss_j, self._parameters,
                retain_graph=(j < response_length - 1), allow_unused=True,
            )
            grad_flat_parts = []
            for g, p in zip(grads, self._parameters):
                if g is not None:
                    g_f32 = g.reshape(-1).to(device).float()
                    if not torch.isfinite(g_f32).all():
                        raise FloatingPointError("reference VJP produced NaN/Inf gradient")
                    grad_flat_parts.append(g_f32)
                else:
                    grad_flat_parts.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
            grad_flat = torch.cat(grad_flat_parts)
            a_kj[:, j] = torch.dot(v, grad_flat)

        del outputs, logits, resp_logits, per_pos_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        nan_mask = torch.isnan(a_kj)
        if nan_mask.any():
            logger.warning(f"Replacing {nan_mask.sum().item()} NaN validation scores with 0.0")
            a_kj = torch.where(nan_mask, torch.zeros_like(a_kj), a_kj)
        return a_kj

    @property
    def validation_direction(self) -> torch.Tensor:
        return self._validation_direction

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._parameters)
