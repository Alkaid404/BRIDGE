"Jacobian-vector product scoring for ``a_{k,j} = v^T g_{k,j}``."

import gc
import inspect
from contextlib import contextmanager, nullcontext
from copy import copy
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from models.utils import _get_layers
from scoring.attention import real_prompt_response_positions
from scoring.vjp import VJPComputer, extract_logits
from utils import get_logger

logger = get_logger("bridge.scoring")


_JVP_ATTN_NAME = "bridge_jvp_query_chunked"
_JVP_QUERY_CHUNK_SIZE = 16
_JVP_LOSS_CHUNK_SIZE = 64


def _repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Equivalent to Transformers repeat_kv without an architecture-specific import."""
    if num_key_value_groups == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, num_key_value_groups, seq_len, head_dim,
    )
    return hidden_states.reshape(
        batch, kv_heads * num_key_value_groups, seq_len, head_dim,
    )


def _query_chunked_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Evaluate attention and its analytic JVP in bounded query chunks."""
    from torch.autograd.forward_ad import make_dual, unpack_dual

    if dropout:
        raise RuntimeError("JVP query-chunk attention requires model.eval() (dropout=0)")
    is_causal = kwargs.get("is_causal")
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)

    query_primal, query_tangent = unpack_dual(query)
    key_primal, key_tangent = unpack_dual(key)
    value_primal, value_tangent = unpack_dual(value)
    key_primal = _repeat_kv(key_primal, module.num_key_value_groups)
    value_primal = _repeat_kv(value_primal, module.num_key_value_groups)
    if key_tangent is not None:
        key_tangent = _repeat_kv(key_tangent, module.num_key_value_groups)
    if value_tangent is not None:
        value_tangent = _repeat_kv(value_tangent, module.num_key_value_groups)

    has_tangent = any(
        tangent is not None
        for tangent in (query_tangent, key_tangent, value_tangent)
    )
    primal_outputs = []
    tangent_outputs = []
    query_len = query_primal.shape[-2]
    key_len = key_primal.shape[-2]
    implicit_causal = query_len > 1 and attention_mask is None and is_causal
    key_positions = torch.arange(key_len, device=query_primal.device)
    query_offset = max(key_len - query_len, 0)
    for start in range(0, query_len, _JVP_QUERY_CHUNK_SIZE):
        end = min(start + _JVP_QUERY_CHUNK_SIZE, query_len)
        query_chunk = query_primal[:, :, start:end, :]
        scores = torch.matmul(query_chunk, key_primal.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[..., start:end, :]
            valid = None
        elif implicit_causal:
            query_positions = torch.arange(
                start + query_offset, end + query_offset, device=query_primal.device,
            )
            valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(
                ~valid.unsqueeze(0).unsqueeze(0),
                torch.finfo(scores.dtype).min,
            )
        else:
            valid = None

        probabilities_fp32 = F.softmax(scores, dim=-1, dtype=torch.float32)
        probabilities = probabilities_fp32.to(query_primal.dtype)
        primal_outputs.append(torch.matmul(probabilities, value_primal))

        if has_tangent:
            score_tangent = torch.zeros_like(scores)
            if query_tangent is not None:
                score_tangent.add_(
                    torch.matmul(
                        query_tangent[:, :, start:end, :],
                        key_primal.transpose(2, 3),
                    ) * scaling
                )
            if key_tangent is not None:
                score_tangent.add_(
                    torch.matmul(query_chunk, key_tangent.transpose(2, 3)) * scaling
                )
            if valid is not None:
                score_tangent.masked_fill_(~valid.unsqueeze(0).unsqueeze(0), 0)
            score_tangent_fp32 = score_tangent.float()
            probability_tangent_fp32 = probabilities_fp32 * (
                score_tangent_fp32
                - (probabilities_fp32 * score_tangent_fp32).sum(dim=-1, keepdim=True)
            )
            probability_tangent = probability_tangent_fp32.to(query_primal.dtype)
            output_tangent = torch.matmul(probability_tangent, value_primal)
            if value_tangent is not None:
                output_tangent.add_(torch.matmul(probabilities, value_tangent))
            tangent_outputs.append(output_tangent)

    primal_output = torch.cat(primal_outputs, dim=2).transpose(1, 2).contiguous()
    if not has_tangent:
        return primal_output, None
    tangent_output = torch.cat(tangent_outputs, dim=2).transpose(1, 2).contiguous()
    return make_dual(primal_output, tangent_output), None


def _register_query_chunked_attention() -> bool:
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if _JVP_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS.register(
                _JVP_ATTN_NAME, _query_chunked_attention_forward,
            )
        return True
    except (ImportError, AttributeError):
        return False


class JVPComputer(VJPComputer):
    "Compute every response-token score with one parameter-direction JVP."

    def __init__(self, model, validation_direction: torch.Tensor,
                 layer_indices: Optional[List[int]] = None):


        super().__init__(model, validation_direction.detach().cpu().float(),
                         layer_indices=layer_indices)
        self._jvp_tangent_components = self._build_direction_tangent_components()
        self._force_math_sdpa = False
        hf_model = getattr(model, "model", None)
        self._supports_logits_to_keep = (
            hf_model is not None
            and "logits_to_keep" in inspect.signature(hf_model.forward).parameters
        )
        self._jvp_attention_modules = self._resolve_jvp_attention_modules()
        tangent_bytes = sum(
            tensor.numel() * tensor.element_size()
            for component in self._jvp_tangent_components
            for tensor in component.values()
        )
        logger.info(
            "Using JVP scoring backend: %d parameter tensors, %d direction components, "
            "%.2f GiB tangents",
            len(self._vjp_names), len(self._jvp_tangent_components),
            tangent_bytes / 1024**3,
        )

    def _resolve_jvp_attention_modules(self) -> List[torch.nn.Module]:
        """Find every attention module traversed by the JVP model forward."""
        if not _register_query_chunked_attention():
            return []
        modules = []
        for layer in _get_layers(self._model):
            attention = getattr(layer, "self_attn", None)
            config = getattr(attention, "config", None)
            if attention is None or config is None or not hasattr(attention, "num_key_value_groups"):
                return []
            modules.append(attention)
        if modules:
            logger.info(
                "JVP query-chunk attention enabled: %d layers, chunk=%d",
                len(modules), _JVP_QUERY_CHUNK_SIZE,
            )
        return modules

    @contextmanager
    def _query_chunked_attention_context(self):
        """Bound the attention temporary for every layer traversed by JVP."""
        original_configs = []
        try:
            for attention in self._jvp_attention_modules:
                original = attention.config
                local_config = copy(original)
                local_config._attn_implementation = _JVP_ATTN_NAME
                attention.config = local_config
                original_configs.append((attention, original))
            yield
        finally:
            for attention, original in original_configs:
                attention.config = original

    def _build_direction_tangent_components(self) -> List[Dict[str, torch.Tensor]]:
        """Map the flat FP32 validation direction to model-dtype tangents."""
        tangents: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, numel in zip(self._vjp_names, self._vjp_numels):
            primal = self._vjp_params_dict[name]
            part = self._validation_direction[offset:offset + numel]
            tangents[name] = part.reshape(primal.shape).to(
                device=primal.device, dtype=primal.dtype, non_blocking=True,
            ).contiguous()
            offset += numel
        if offset != self._validation_direction.numel():
            raise RuntimeError(
                f"JVP direction mapping consumed {offset} values, "
                f"expected {self._validation_direction.numel()}"
            )
        return [tangents]

    @staticmethod
    def _math_sdpa_context():
        """Return a math-SDPA context for forward-AD operator fallback."""
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except (ImportError, AttributeError):
            return nullcontext()
        return sdpa_kernel(SDPBackend.MATH)

    def _run_jvp(
        self,
        loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        tangents: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        with torch.no_grad():
            return self._run_jvp_impl(loss_fn, tangents)

    def _run_jvp_impl(
        self,
        loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        tangents: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._jvp_attention_modules:


            with self._math_sdpa_context(), self._query_chunked_attention_context():
                return jvp(loss_fn, (self._vjp_params_dict,), (tangents,))
        if self._force_math_sdpa:
            with self._math_sdpa_context():
                return jvp(loss_fn, (self._vjp_params_dict,), (tangents,))
        try:
            return jvp(loss_fn, (self._vjp_params_dict,), (tangents,))
        except RuntimeError as error:
            message = str(error).lower()
            forward_ad_unsupported = (
                "forward ad" in message
                or "forward-mode ad" in message
                or "jvp" in message and "not implemented" in message
            )
            if not forward_ad_unsupported:
                raise
            self._force_math_sdpa = True
            logger.warning("Default SDPA does not support forward AD; using math SDPA for JVP")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with self._math_sdpa_context():
                return jvp(loss_fn, (self._vjp_params_dict,), (tangents,))

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
        """Return per-sample ``J_L v`` scores; ``chunk_size`` is VJP-only."""
        del chunk_size
        batch_size = input_ids.shape[0]
        if len(prompt_lengths) != batch_size or len(response_lengths) != batch_size:
            raise ValueError(
                f"prompt/response lengths must have length {batch_size}, "
                f"got {len(prompt_lengths)}/{len(response_lengths)}"
            )

        device = input_ids.device
        response_pos_list: List[torch.Tensor] = []
        offsets = [0]
        for i in range(batch_size):
            if prompt_lengths[i] <= 0 or response_lengths[i] <= 0:
                raise ValueError(
                    f"sample {i} has invalid lengths "
                    f"prompt={prompt_lengths[i]} response={response_lengths[i]}"
                )
            _, response_pos = real_prompt_response_positions(
                attention_mask[i], prompt_block_len=prompt_block_len,
                prompt_length=prompt_lengths[i], response_length=response_lengths[i],
            )
            response_pos = response_pos.to(device)
            if (response_pos <= 0).any():
                raise RuntimeError("response positions must be > 0 for causal-LM JVP")
            response_pos_list.append(response_pos)
            offsets.append(offsets[-1] + response_pos.numel())

        labels_dev = labels.to(device)
        if self._supports_logits_to_keep:


            logit_positions = torch.unique(
                torch.cat([positions - 1 for positions in response_pos_list]), sorted=True,
            )
            response_logit_indices = [
                torch.searchsorted(logit_positions, positions - 1)
                for positions in response_pos_list
            ]
        else:
            logit_positions = None
            response_logit_indices = [positions - 1 for positions in response_pos_list]

        def loss_fn(pd: Dict[str, torch.Tensor]) -> torch.Tensor:
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_attentions": False,
                "output_hidden_states": False,
            }
            if logit_positions is not None:
                kwargs["logits_to_keep"] = logit_positions
            if self._fc_tie_weights:
                outputs = functional_call(self._model, pd, (), kwargs, tie_weights=False)
            else:
                outputs = functional_call(self._model, pd, (), kwargs)
            logits = extract_logits(outputs)
            losses = []
            for i, response_pos in enumerate(response_pos_list):
                response_logits = logits[i, response_logit_indices[i], :]
                response_labels = labels_dev[i, response_pos]
                token_losses = []
                for start in range(0, response_logits.shape[0], _JVP_LOSS_CHUNK_SIZE):
                    end = min(start + _JVP_LOSS_CHUNK_SIZE, response_logits.shape[0])
                    token_losses.append(F.cross_entropy(
                        response_logits[start:end].float().reshape(
                            -1, response_logits.shape[-1],
                        ),
                        response_labels[start:end].reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ))
                losses.append(torch.cat(token_losses))
            return torch.cat(losses)

        losses = None
        scores = None
        for tangents in self._jvp_tangent_components:
            component_losses, component_scores = self._run_jvp(loss_fn, tangents)
            if losses is None:
                losses = component_losses
                scores = component_scores.float()
            else:
                scores = scores + component_scores.float()
            del component_losses, component_scores
        assert losses is not None and scores is not None

        bad_loss = ~torch.isfinite(losses)
        bad_score = ~torch.isfinite(scores)
        if bad_loss.any() or bad_score.any():
            raise FloatingPointError(
                "JVP produced non-finite values: "
                f"losses={int(bad_loss.sum().item())}, scores={int(bad_score.sum().item())}"
            )

        scores = scores.detach().float()
        result = [
            scores[offsets[i]:offsets[i + 1]].cpu().clone()
            for i in range(batch_size)
        ]
        del losses, scores
        gc.collect()
        return result
