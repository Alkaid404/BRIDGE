"评分流水线：编排 attention → VJP → proxy → marginal，产出逐样本记录。"

import gc
import json
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from data import BridgeDataCollator, BridgeDataset
from models.utils import BridgeModel
from scoring.attention import AttentionRecorder
from scoring.jvp import JVPComputer
from scoring.marginal import MarginalScorer
from scoring.proxy import InteractionProxy
from scoring.vjp import VJPComputer
from utils import get_logger, seed_worker

logger = get_logger("bridge.scoring")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


@dataclass
class SampleRecord:
    "单样本完整评分记录。"

    sample_idx: int
    prompt_length: int
    response_length: int
    A_kji: torch.Tensor
    response_prefix_ratio: torch.Tensor
    a_kj: torch.Tensor
    beta_hat: torch.Tensor
    gamma_hat: torch.Tensor
    prompt_scores: torch.Tensor
    response_scores: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        R, P = self.response_length, self.prompt_length
        checks = {
            "A_kji": (self.A_kji.shape, (R, P)),
            "response_prefix_ratio": (self.response_prefix_ratio.shape, (R,)),
            "a_kj": (self.a_kj.shape, (R,)),
            "beta_hat": (self.beta_hat.shape, (R, P)),
            "gamma_hat": (self.gamma_hat.shape, (R,)),
            "prompt_scores": (self.prompt_scores.shape, (P,)),
            "response_scores": (self.response_scores.shape, (R,)),
        }
        for name, (actual, expected) in checks.items():
            if tuple(actual) != expected:
                raise ValueError(
                    f"SampleRecord field '{name}' has shape {tuple(actual)}, "
                    f"expected {expected} (prompt_length={P}, response_length={R})"
                )


class ScoringPipeline:
    "BRIDGE 端到端评分流水线。"

    def __init__(self, model: Any, validation_direction: torch.Tensor, config: Any,
                 progress_desc: str = "Scoring", progress_position: int = 0,
                 sample_offset: int = 0, has_bos: Optional[bool] = None):
        self._model = model
        self._validation_direction = validation_direction
        self._config = config
        self._progress_desc = progress_desc
        self._progress_position = progress_position
        self._sample_offset = int(sample_offset)
        attention_type = getattr(config, "attention_aggregation", "last_layer")
        if attention_type not in {"last_layer", "multi_layer_avg"}:
            attention_type = "last_layer"
        record_layers = getattr(config, "attention_num_layers", None)
        num_layers = len(record_layers) if isinstance(record_layers, list) else (record_layers or 1)
        layer_strategy = getattr(config, "scoring_attention_layer_strategy", "last_physical")
        layer_index = getattr(config, "scoring_attention_layer_index", None)


        self._has_bos = True if has_bos is None else bool(has_bos)
        self._bos_handling = getattr(config, "bos_handling", "zero_no_renorm")
        if self._bos_handling == "auto":
            self._bos_handling = "zero_no_renorm" if self._has_bos else "keep"

        self._attention_recorder = AttentionRecorder(
            model=model, attention_type=attention_type, num_layers=num_layers,
            layer_strategy=layer_strategy, layer_index=layer_index,
            layer_indices=getattr(config, "attention_layer_indices", None),
        )
        self._capture_version = self._attention_recorder.capture_version
        self._scoring_backend = str(getattr(config, "scoring_backend", "vjp")).lower()


        from models.arch_adapter import resolve_layer_indices
        d_strategy = getattr(config, "direction_layer_strategy", "last_n_physical")
        d_indices = getattr(config, "direction_layer_indices", None)
        d_n_last = getattr(config, "direction_num_last_layers", 4)
        self._direction_layer_indices = resolve_layer_indices(
            model, d_strategy, d_indices, n_last=d_n_last
        )
        logger.info(
            f"ScoringPipeline direction layer subset: strategy={d_strategy} "
            f"indices={self._direction_layer_indices}"
        )
        if self._scoring_backend == "jvp":
            self._vjp_computer = JVPComputer(model=model, validation_direction=validation_direction,
                                             layer_indices=self._direction_layer_indices)
        elif self._scoring_backend == "vjp":
            self._vjp_computer = VJPComputer(model=model, validation_direction=validation_direction,
                                             layer_indices=self._direction_layer_indices)
        else:
            raise ValueError(
                f"Unknown scoring_backend={self._scoring_backend!r}; expected 'vjp' or 'jvp'"
            )
        self._interaction_proxy = InteractionProxy(attention_type=attention_type)
        self._marginal_scorer = MarginalScorer(interaction_proxy=self._interaction_proxy)
        self._results: Optional[Dict[str, Any]] = None

        logger.debug(
            f"ScoringPipeline: backend={self._scoring_backend}, "
            f"attention_type={attention_type}, num_layers={num_layers}"
        )

    def run(self, dataloader: DataLoader) -> Dict[str, Any]:
        """在整个数据集上评分，返回 ``{"samples": List[SampleRecord], "metadata": [...]}``。"""
        all_samples: List[SampleRecord] = []
        all_metadata: List[Dict[str, Any]] = []

        total = len(dataloader) if hasattr(dataloader, "__len__") else None
        batch_iter = (
            tqdm(
                dataloader, total=total, desc=self._progress_desc, unit="batch",
                position=self._progress_position,
            )
            if _HAS_TQDM else dataloader
        )

        self._model.eval()
        for batch_idx, batch in enumerate(batch_iter):
            for attempt in range(2):
                try:
                    records = self.score_batch(batch)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    break
                except torch.OutOfMemoryError as e:
                    if attempt > 0:
                        logger.exception(f"Error scoring batch {batch_idx}: {e}")
                        self._attention_recorder.close()
                        raise RuntimeError(
                            f"Scoring failed at batch {batch_idx} after OOM retry: {e}"
                        ) from e
                    logger.warning(
                        "Scoring batch %d hit CUDA OOM; releasing cached blocks and "
                        "retrying the identical batch once",
                        batch_idx,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:

                    logger.exception(f"Error scoring batch {batch_idx}: {e}")
                    self._attention_recorder.close()
                    raise RuntimeError(f"Scoring failed at batch {batch_idx}: {e}") from e

            for rec in records:
                global_idx = self._sample_offset + len(all_samples)
                rec.sample_idx = global_idx
                rec.metadata["sample_idx"] = global_idx
                rec.metadata["batch_idx"] = batch_idx
                all_samples.append(rec)
                all_metadata.append(rec.metadata)
            if _HAS_TQDM:
                batch_iter.set_postfix(samples=len(all_samples))

        self._attention_recorder.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._results = {"samples": all_samples, "metadata": all_metadata,
                         "capture_version": self._capture_version,
                         "scoring_layer_info": self._attention_recorder.scoring_layer_info,
                         "bos_handling": self._bos_handling,
                         "has_bos": self._has_bos}
        logger.info(f"{self._progress_desc} complete: {len(all_samples)} samples")
        return self._results

    def _resolve_input_device(self) -> torch.device:
        input_device = getattr(self._model, "input_device", None)
        if input_device is not None:
            return input_device
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def score_batch(self, batch: Dict[str, torch.Tensor]) -> List[SampleRecord]:
        """对单个批次评分（逐样本、变长序列安全）。"""
        model_device = self._resolve_input_device()
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch["attention_mask"].to(model_device)
        labels = batch["labels"].to(model_device)
        batch_size = input_ids.shape[0]

        prompt_lengths, response_lengths = self._resolve_per_sample_lengths(batch, labels, batch_size)


        prompt_block_len = None


        attention_records = self._attention_recorder.record_attention_per_sample(
            input_ids=input_ids, attention_mask=attention_mask,
            prompt_lengths=prompt_lengths, response_lengths=response_lengths,
            prompt_block_len=prompt_block_len,
        )

        a_kj_list = self._vjp_computer.compute_validation_scores_per_sample(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            prompt_lengths=prompt_lengths, response_lengths=response_lengths,
            prompt_block_len=prompt_block_len, chunk_size=self._config.vjp_chunk_size,
        )

        records: List[SampleRecord] = []
        for i in range(batch_size):
            A_kji = attention_records[i]["A_kji"]
            A_kjj_prime = attention_records[i]["A_kjj_prime"]

            c_kj = self._attention_recorder.compute_response_prefix_ratio(A_kjj_prime.unsqueeze(0)).squeeze(0)
            a_kj = a_kj_list[i]

            compute_device = a_kj.device
            A_kji = A_kji.to(compute_device)
            c_kj = c_kj.to(compute_device)
            beta_hat = self._interaction_proxy.compute_interaction_proxy(A_kji.unsqueeze(0), a_kj.unsqueeze(0)).squeeze(0)
            gamma_hat = self._interaction_proxy.compute_history_term(a_kj.unsqueeze(0), c_kj.unsqueeze(0)).squeeze(0)


            if (
                getattr(self._config, "special_token_scoring", "include") == "exclude"
                and getattr(self._config, "special_token_mode", "legacy") == "single_bos_eos"
                and beta_hat.dim() == 2 and beta_hat.shape[1] > 0
                and self._bos_handling != "keep"
                and self._has_bos
            ):
                if self._bos_handling == "zero_and_renorm" and beta_hat.shape[1] > 1:
                    a_bos = A_kji[:, 0].clamp(min=0.0, max=1.0)
                    scale = 1.0 / (1.0 - a_bos.unsqueeze(1) + 1e-12)
                    beta_hat[:, 0] = 0.0
                    beta_hat[:, 1:] = beta_hat[:, 1:] * scale
                else:
                    beta_hat[:, 0] = 0.0
            prompt_scores = self._marginal_scorer.compute_prompt_scores(beta_hat.unsqueeze(0)).squeeze(0)
            response_scores = self._marginal_scorer.compute_response_scores(
                gamma_hat.unsqueeze(0), beta_hat.unsqueeze(0)
            ).squeeze(0)
            rl = int(A_kji.shape[0])
            pl = int(A_kji.shape[1])
            metadata = {
                "sample_idx": i,
                "prompt_length": pl,
                "response_length": rl,
                "seq_len": int(attention_mask[i].sum().item()),
            }
            records.append(SampleRecord(
                sample_idx=i, prompt_length=pl, response_length=rl,
                A_kji=A_kji.detach().cpu(), response_prefix_ratio=c_kj.detach().cpu(),
                a_kj=a_kj.detach().cpu(), beta_hat=beta_hat.detach().cpu(),
                gamma_hat=gamma_hat.detach().cpu(),
                prompt_scores=prompt_scores.detach().cpu(),
                response_scores=response_scores.detach().cpu(),
                metadata=metadata,
            ))
        return records

    @staticmethod
    def _resolve_per_sample_lengths(
        batch: Dict[str, Any], labels: torch.Tensor, batch_size: int,
    ) -> tuple:
        "解析逐样本 prompt_length / response_length 列表。"
        pl_field = batch.get("prompt_length", None)
        rl_field = batch.get("response_length", None)

        def _to_int_list(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                val = val.tolist()
            if isinstance(val, list):
                return [int(x) for x in val]
            return [int(val)] * batch_size

        prompt_lengths = _to_int_list(pl_field)
        response_lengths = _to_int_list(rl_field)

        if prompt_lengths is None or response_lengths is None:
            prompt_lengths, response_lengths = [], []
            seq_len = labels.shape[1]
            for i in range(batch_size):
                valid = labels[i] != -100
                if not valid.any():
                    prompt_lengths.append(seq_len)
                    response_lengths.append(0)
                else:
                    pl_i = int(valid.nonzero(as_tuple=True)[0][0].item())
                    prompt_lengths.append(pl_i)
                    response_lengths.append(seq_len - pl_i)

        if len(prompt_lengths) != batch_size or len(response_lengths) != batch_size:
            raise ValueError(
                f"prompt/response length lists must have length {batch_size}, "
                f"got {len(prompt_lengths)}/{len(response_lengths)}"
            )
        return prompt_lengths, response_lengths

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        return self._results

    @property
    def attention_recorder(self) -> AttentionRecorder:
        return self._attention_recorder

    @property
    def vjp_computer(self) -> VJPComputer:
        return self._vjp_computer

    @property
    def interaction_proxy(self) -> InteractionProxy:
        return self._interaction_proxy

    @property
    def marginal_scorer(self) -> MarginalScorer:
        return self._marginal_scorer


def _merge_scoring_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并多 worker 评分结果：顺序拼接 samples / metadata（不做 torch.cat）。"""
    versions = {r.get("capture_version", "legacy_v1") for r in results_list}
    if len(versions) > 1:
        raise RuntimeError(f"Multi-GPU scoring capture_version mismatch: {sorted(versions)}")
    infos = {json.dumps(r.get("scoring_layer_info"), sort_keys=True) for r in results_list}
    if len(infos) > 1:
        raise RuntimeError(f"Multi-GPU scoring layer_info mismatch: {sorted(infos)}")
    bos_modes = {r.get("bos_handling", "zero_no_renorm") for r in results_list}
    if len(bos_modes) > 1:
        raise RuntimeError(f"Multi-GPU bos_handling mismatch: {sorted(bos_modes)}")
    return {
        "samples": [s for r in results_list for s in r["samples"]],
        "metadata": [m for r in results_list for m in r["metadata"]],
        "capture_version": versions.pop() if versions else "legacy_v1",
        "scoring_layer_info": results_list[0].get("scoring_layer_info"),
        "bos_handling": bos_modes.pop() if bos_modes else "zero_no_renorm",
    }


def _run_scoring_process(
    gpu_id: int, model_path: str, checkpoint_path: str,
    dtype: torch.dtype,
    direction_path: str, shard_path: str, shard_start: int,
    max_prompt_length: int, max_response_length: int,
    batch_size: int, config,
    result_path: str, error_path: str,
):
    """Spawn worker: load one raw-data shard locally and write one CPU result file."""
    import torch as _torch
    try:
        device = _torch.device(f"cuda:{gpu_id}")
        _torch.cuda.set_device(device)
        logger.info(f"[GPU {gpu_id}] loading model...")

        from transformers import AutoTokenizer

        local_model = BridgeModel(
            model_name_or_path=checkpoint_path,
            torch_dtype=dtype, device_map=device,
        )
        local_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"


        shard_items = _torch.load(shard_path, map_location="cpu", weights_only=False)


        local_dataset = BridgeDataset(
            shard_items, tokenizer, max_prompt_length, max_response_length, split="scoring",
            special_token_mode=getattr(config, "special_token_mode", "legacy"),
        )
        local_dataset.tokenize_all()
        collator = BridgeDataCollator(tokenizer=tokenizer)
        loader = DataLoader(
            local_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collator, worker_init_fn=seed_worker,
        )
        direction = _torch.load(direction_path, map_location="cpu", weights_only=True).float()

        pipeline = ScoringPipeline(
            local_model, direction, config,
            progress_desc=f"Score GPU {gpu_id}", progress_position=gpu_id,
            sample_offset=shard_start,
            has_bos=(getattr(tokenizer, "bos_token_id", None) is not None
                     if tokenizer is not None else None),
        )
        results = pipeline.run(loader)
        _torch.save(results, result_path)
        Path(result_path + ".done").write_text(
            f"{shard_start}:{len(results['samples'])}", encoding="utf-8",
        )
        logger.info(f"[GPU {gpu_id}] scoring done")
        del results, pipeline, loader, local_dataset, shard_items, local_model, direction
        _torch.cuda.empty_cache()
    except Exception as e:
        logger.exception(f"[GPU {gpu_id}] scoring failed: {e}")
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run_scoring(
    model: BridgeModel,
    tokenizer,
    direction: torch.Tensor,
    dataset,
    config: Any,
    model_path: str,
    checkpoint_path: Optional[str] = None,
    num_gpus: int = 1,
    direction_path: Optional[str] = None,
    worker_output_dir: Optional[str] = None,
    resume_key: Optional[str] = None,
) -> Dict[str, Any]:
    "在 1 或多 GPU 上运行评分流水线。"
    batch_size = getattr(config, "scoring_batch_size", 4)
    collator = BridgeDataCollator(tokenizer=tokenizer)

    if num_gpus <= 1 or checkpoint_path is None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collator, worker_init_fn=seed_worker)
        pipeline = ScoringPipeline(model, direction, config,
                                   has_bos=(getattr(tokenizer, "bos_token_id", None) is not None
                                            if tokenizer is not None else None))
        return pipeline.run(loader)

    available = torch.cuda.device_count()
    if available < num_gpus:
        logger.warning(f"Available GPUs ({available}) < requested ({num_gpus}); using {available}")
        num_gpus = available

    total = len(dataset)
    if total == 0:
        return {"samples": [], "metadata": []}
    num_gpus = min(num_gpus, total)
    if not isinstance(dataset, BridgeDataset):
        raise TypeError("Multi-GPU scoring requires BridgeDataset raw data for FD-safe spawn")
    shard_size = total // num_gpus
    shard_ranges = []
    for i in range(num_gpus):
        start = i * shard_size
        end = start + shard_size if i < num_gpus - 1 else total
        shard_ranges.append((start, end))

    dtype = torch.bfloat16 if getattr(config, 'bf16', True) else torch.float16
    output_parent = Path(worker_output_dir or ".")
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = None
    if resume_key:
        safe_key = "".join(ch for ch in resume_key if ch.isalnum())[:20]
        tmp_path = output_parent / f".scoring_resume_{safe_key}"
        tmp_path.mkdir(parents=True, exist_ok=True)
    else:
        temporary_dir = tempfile.TemporaryDirectory(
            prefix=".scoring_workers_", dir=output_parent,
        )
        tmp_path = Path(temporary_dir.name)

    try:
        worker_direction_path = direction_path
        if worker_direction_path is None:
            worker_direction_path = str(tmp_path / "validation_direction.pt")
            torch.save(direction.detach().cpu(), worker_direction_path)

        shard_paths = []
        completed_gpus = set()
        for gpu_id, (start, end) in enumerate(shard_ranges):
            shard_path = tmp_path / f"gpu_{gpu_id}_raw_samples.pt"
            result_path = tmp_path / f"gpu_{gpu_id}_results.pt"
            done_path = Path(str(result_path) + ".done")
            expected_marker = f"{start}:{end - start}"
            if (
                resume_key
                and result_path.exists()
                and done_path.exists()
                and done_path.read_text(encoding="utf-8") == expected_marker
            ):
                completed_gpus.add(gpu_id)
                logger.info(
                    f"GPU {gpu_id} result shard already complete; reusing {result_path}"
                )
            else:

                torch.save(dataset.data[start:end], shard_path)
            shard_paths.append(str(shard_path))

        context = torch.multiprocessing.get_context("spawn")
        processes = []
        for gpu_id in range(num_gpus):
            if gpu_id in completed_gpus:
                continue
            result_path = str(tmp_path / f"gpu_{gpu_id}_results.pt")
            error_path = str(tmp_path / f"gpu_{gpu_id}_error.txt")
            process = context.Process(
                target=_run_scoring_process,
                args=(
                    gpu_id, model_path, checkpoint_path, dtype,
                    worker_direction_path, shard_paths[gpu_id], shard_ranges[gpu_id][0],
                    dataset.max_prompt_length, dataset.max_response_length,
                    batch_size, config,
                    result_path, error_path,
                ),
            )
            process.start()
            processes.append((gpu_id, process))
        for _, process in processes:
            process.join()

        failures = []
        for gpu_id, process in processes:
            if process.exitcode != 0:
                error_path = tmp_path / f"gpu_{gpu_id}_error.txt"
                detail = error_path.read_text(encoding="utf-8") if error_path.exists() else "no traceback"
                failures.append(f"GPU {gpu_id} exit={process.exitcode}:\n{detail[-4000:]}")
        if failures:
            raise RuntimeError("Multi-GPU scoring worker failure:\n" + "\n".join(failures))

        merged_list = []
        for gpu_id in range(num_gpus):
            result_path = tmp_path / f"gpu_{gpu_id}_results.pt"
            if not result_path.exists():
                raise RuntimeError(f"GPU {gpu_id} scoring produced no result file")
            result = torch.load(result_path, map_location="cpu", weights_only=False)
            got = len(result.get("samples", []))
            start, end = shard_ranges[gpu_id]
            expected = end - start
            if got != expected:
                raise RuntimeError(
                    f"GPU {gpu_id} scoring incomplete: got {got}, expected {expected}"
                )
            merged_list.append(result)
        merged = _merge_scoring_results(merged_list)
        sample_indices = [record.sample_idx for record in merged["samples"]]
        if sample_indices != list(range(total)):
            raise RuntimeError("Multi-GPU scoring result order is not globally contiguous")
        if resume_key:
            merged["_worker_resume_dir"] = str(tmp_path)
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()
    logger.info(f"Multi-GPU scoring done: {num_gpus} GPUs, {total} samples")
    return merged
