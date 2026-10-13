"BRIDGE 数据层：加载、分词、整理。"

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import get_logger

logger = get_logger("bridge.data")


def _make_sample(sample_id: str, prompt: str, response: str,
                 task_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": sample_id, "prompt": prompt, "response": response,
            "task_type": task_type, "metadata": metadata}


_GSM8K_HF = "openai/gsm8k"
_GSM8K_ANSWER_MARKER = re.compile(r"####\s*(.+)$", re.MULTILINE)
_MATH_HF = "hendrycks/competition_math"
_MATH_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_MATH_BOLD = re.compile(r"\*\*([^*]+)\*\*\s*$", re.MULTILINE)
_OPEN_R1_HF = "open-r1/OpenR1-Math-220k"
_ARC_HF = "allenai/ai2_arc"
_MMLU_HF = "cais/mmlu"
_HELLASWAG_HF = "Rowan/hellaswag"
_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def _convert_gsm8k(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = item.get("question", "")
    raw_answer = item.get("answer", "")
    prompt = (
        f"Given the following problem, reason step by step and "
        f"write your final answer after ####.\n\n"
        f"Problem: {question.strip()}\n\nAnswer: Let's think step by step.\n"
    )
    response = "\n".join(s for s in raw_answer.strip().split("\n") if s.strip())
    match = _GSM8K_ANSWER_MARKER.search(raw_answer)
    final = match.group(1).strip() if match else ""
    before = _GSM8K_ANSWER_MARKER.split(raw_answer)[0]
    n_steps = len([ln for ln in before.strip().split("\n") if ln.strip()])
    return _make_sample(f"gsm8k_{index}", prompt, response, "math",
                        {"final_answer": final, "num_reasoning_steps": n_steps})


def _convert_arc(item: Dict[str, Any], index: int, variant: str) -> Dict[str, Any]:
    question = item.get("question", "")
    choices = item.get("choices", {})
    answer_key = item.get("answerKey", "")
    if isinstance(choices, dict):
        texts, labels = choices.get("text", []), choices.get("label", [])
    elif isinstance(choices, list) and choices and isinstance(choices[0], str):
        texts = choices
        labels = [_INDEX_TO_LETTER.get(i, chr(65 + i)) for i in range(len(choices))]
    else:
        texts = [c if isinstance(c, str) else c.get("text", "") for c in choices]
        labels = [c.get("label", _INDEX_TO_LETTER.get(i, chr(65 + i))) if isinstance(c, dict)
                  else _INDEX_TO_LETTER.get(i, chr(65 + i)) for i, c in enumerate(choices)]
    lines = [f"Question: {question}", ""]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    prompt = "\n".join(lines)
    answer = answer_key.strip().upper() if answer_key else ""
    response = f"The correct answer is {answer}." if answer else ""
    return _make_sample(
        f"arc_{variant}_{index}", prompt, response, "alignment",
        {
            "variant": variant,
            "question": question,
            "choices": list(texts),
            "choice_labels": [str(label) for label in labels],
            "answer_key": answer_key,
            "num_choices": len(texts),
        },
    )


def _convert_math(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    problem = item.get("problem", "")
    solution = item.get("solution", "")
    problem_type = item.get("type", item.get("level", "unknown"))
    level = item.get("level", "")
    type_label = f" ({problem_type})" if problem_type else ""
    prompt = (f"Solve the following math competition problem{type_label}.\n\n"
              f"Problem: {problem.strip()}\n\nSolution:")
    response = "\n".join(s for s in solution.strip().split("\n") if s.strip()) if solution else ""
    match = _MATH_BOXED.search(solution)
    if match:
        final = match.group(1).strip()
    else:
        m2 = _MATH_BOLD.search(solution)
        final = m2.group(1).strip() if m2 else ""
    n_steps = len([ln for ln in solution.strip().split("\n") if ln.strip()])
    return _make_sample(f"math_{index}", prompt, response, "math",
                        {"problem_type": problem_type, "level": level,
                         "final_answer": final, "num_solution_steps": n_steps})


def _extract_field(item: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _convert_open_r1(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    problem = _extract_field(item, ["problem", "question", "prompt"])
    reasoning = _extract_field(item, ["reasoning", "solution", "chain_of_thought", "cot"])
    final_answer = _extract_field(item, ["answer", "final_answer", "target"])
    source = item.get("source", item.get("dataset", ""))
    difficulty = item.get("difficulty", item.get("level", ""))
    prompt = (f"Solve the following problem. Show your reasoning step by step.\n\n"
              f"Problem: {problem.strip()}\n\nReasoning:")
    parts: List[str] = []
    if reasoning:
        parts.append("\n".join(s for s in reasoning.strip().split("\n") if s.strip()))
    if final_answer:
        parts.append(f"\n#### {final_answer.strip()}")
    response = "\n".join(parts)
    n_steps = len([ln for ln in reasoning.strip().split("\n") if ln.strip()]) if reasoning else 0
    return _make_sample(f"open_r1_{index}", prompt, response, "reasoning",
                        {"source": source, "difficulty": difficulty,
                         "final_answer": final_answer, "num_reasoning_steps": n_steps})


def _convert_mmlu(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = str(item.get("question", "")).strip()
    choices = list(item.get("choices") or [])
    answer = item.get("answer")
    try:
        answer_index = int(answer)
    except (TypeError, ValueError):
        answer_index = -1
    answer_key = _INDEX_TO_LETTER.get(answer_index, "")
    lines = [f"Question: {question}", ""]
    lines.extend(f"{_INDEX_TO_LETTER.get(i, chr(65 + i))}. {text}"
                 for i, text in enumerate(choices))
    response = f"The correct answer is {answer_key}." if answer_key else ""
    subject = str(item.get("subject", ""))
    return _make_sample(
        f"mmlu_{index}", "\n".join(lines), response, "knowledge",
        {
            "subject": subject,
            "question": question,
            "choices": choices,
            "choice_labels": [
                _INDEX_TO_LETTER.get(i, chr(65 + i)) for i in range(len(choices))
            ],
            "answer_key": answer_key,
            "num_choices": len(choices),
        },
    )


def _convert_hellaswag(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    context = str(item.get("ctx") or " ".join(
        part for part in [str(item.get("ctx_a", "")), str(item.get("ctx_b", ""))] if part
    )).strip()
    endings = list(item.get("endings") or [])
    label = item.get("label")
    try:
        answer_index = int(label)
    except (TypeError, ValueError):
        answer_index = -1
    answer_key = _INDEX_TO_LETTER.get(answer_index, "")
    lines = [f"Complete the following passage:\n{context}", ""]
    lines.extend(f"{_INDEX_TO_LETTER.get(i, chr(65 + i))}. {ending}"
                 for i, ending in enumerate(endings))
    response = f"The correct answer is {answer_key}." if answer_key else ""
    sample_id = item.get("ind", index)
    return _make_sample(
        f"hellaswag_{sample_id}", "\n".join(lines), response, "commonsense",
        {
            "activity_label": item.get("activity_label", ""),
            "context": context,
            "choices": endings,
            "choice_labels": [
                _INDEX_TO_LETTER.get(i, chr(65 + i)) for i in range(len(endings))
            ],
            "answer_key": answer_key,
            "num_choices": len(endings),
        },
    )


def _convert_magicoder(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    instruction = str(item.get("instruction", item.get("prompt", ""))).strip()
    response = str(item.get("response", item.get("output", ""))).strip()
    return _make_sample(
        str(item.get("id", f"magicoder_{index}")), instruction, response, "code",
        {"source": item.get("source", "magicoder")},
    )


def _convert_alpaca(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    response = str(item.get("output", item.get("response", ""))).strip()

    prompt_parts = [
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.",
        "",
        "### Instruction:",
        instruction,
    ]
    if input_text:
        prompt_parts.extend(["", "### Input:", input_text])
    prompt_parts.extend(["", "### Response:"])

    return _make_sample(
        str(item.get("id", f"alpaca_{index}")),
        "\n".join(prompt_parts),
        response,
        "instruction_following",
        {"source": "alpaca", "input": input_text},
    )


def _convert_dolly(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    "Databricks Dolly 15k：instruction/context/response/category → 统一 prompt/response。"
    instruction = str(item.get("instruction", "")).strip()
    context = str(item.get("context", "")).strip()
    response = str(item.get("response", "")).strip()

    prompt_parts = [
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.",
        "",
        "### Instruction:",
        instruction,
    ]
    if context:
        prompt_parts.extend(["", "### Input:", context])
    prompt_parts.extend(["", "### Response:"])

    return _make_sample(
        str(item.get("id", f"dolly_{index}")),
        "\n".join(prompt_parts),
        response,
        "general",
        {"source": "dolly", "category": str(item.get("category", ""))},
    )


def _convert_metamath(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    query = str(item.get("query", "")).strip()
    response = str(item.get("response", "")).strip()
    problem_type = str(item.get("type", "unknown"))
    final_match = _GSM8K_ANSWER_MARKER.search(response)
    final_answer = final_match.group(1).strip() if final_match else ""
    prompt = (
        "Solve the following math problem. Show your reasoning step by step.\n\n"
        f"Problem: {query}\n\nSolution:"
    )
    return _make_sample(
        str(item.get("id", f"metamath_{index}")),
        prompt,
        response,
        "math",
        {"source": "metamath", "problem_type": problem_type,
         "final_answer": final_answer},
    )


def _convert_humaneval(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    canonical_solution = str(item.get("canonical_solution", ""))
    task_id = str(item.get("task_id", f"HumanEval/{index}"))
    return _make_sample(
        task_id,
        prompt,
        canonical_solution,
        "code",
        {
            "task_id": task_id,
            "test": str(item.get("test", "")),
            "entry_point": str(item.get("entry_point", "")),
        },
    )


_CONVERTERS = {
    "gsm8k": ("main", _convert_gsm8k),
    "arc_easy": ("ARC-Easy", lambda it, i: _convert_arc(it, i, "easy")),
    "arc_challenge": ("ARC-Challenge", lambda it, i: _convert_arc(it, i, "challenge")),
    "math": ("default", _convert_math),
    "open_r1": (None, _convert_open_r1),
    "mmlu": ("all", _convert_mmlu),
    "hellaswag": (None, _convert_hellaswag),
    "magicoder": (None, _convert_magicoder),
    "alpaca": (None, _convert_alpaca),
    "dolly": (None, _convert_dolly),
    "metamath": (None, _convert_metamath),
    "humaneval": (None, _convert_humaneval),
}


_DEFAULT_SPLITS = {
    "gsm8k": ("train", None),
    "arc_easy": ("train", "validation"),
    "arc_challenge": ("train", "validation"),
    "math": ("train", None),
    "open_r1": ("train", None),
    "mmlu": ("auxiliary_train", "validation"),
    "hellaswag": ("train", "validation"),
    "magicoder": (None, None),
    "alpaca": ("train", None),
    "dolly": (None, None),
    "metamath": (None, None),
    "humaneval": (None, None),
}


def _is_local(source: Optional[str]) -> bool:
    return source is not None and (Path(source).is_file() or Path(source).is_dir())


def _load_raw(name: str, source: Optional[str], split: Optional[str]) -> List[Dict[str, Any]]:
    """加载原始条目列表。本地路径优先，否则从 HuggingFace hub 加载。"""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install 'datasets': pip install datasets") from e

    if _is_local(source):
        path = Path(source)
        if path.is_dir() and list(path.rglob("*.parquet")):
            parquet_files = sorted(path.rglob("*.parquet"))
            available = sorted({f.name.split("-", 1)[0] for f in parquet_files})
            if split is not None:
                selected = [f for f in parquet_files if f.name.startswith(f"{split}-")]
                if not selected:
                    raise ValueError(
                        f"Local dataset '{path}' has no split '{split}'; available={available}"
                    )
                ds = load_dataset(
                    "parquet", data_files={split: [str(f) for f in selected]}, split=split,
                )
                if hasattr(ds, "keys") and split in ds:
                    ds = ds[split]
                return list(ds)

            data_files = {
                split_name: [str(f) for f in parquet_files
                             if f.name.startswith(f"{split_name}-")]
                for split_name in available
            }
            ds = load_dataset("parquet", data_files=data_files)
            items: List[Dict[str, Any]] = []
            for split_name in available:
                items.extend(list(ds[split_name]))
            return items


        files = [path] if path.is_file() else sorted(list(path.glob("*.jsonl")) or list(path.glob("*.json")))
        items = []
        for f in files:
            if f.suffix.lower() == ".json":
                with open(f, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                if isinstance(payload, list):
                    items.extend(payload)
                elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    items.extend(payload["data"])
                elif isinstance(payload, dict):
                    items.append(payload)
                else:
                    raise ValueError(
                        f"Local JSON dataset '{f}' must contain an object or array"
                    )
                continue
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return items


    if name in ("magicoder", "alpaca", "metamath", "humaneval"):
        raise ValueError(f"{name} requires a local dataset_path")
    hf_name, _ = {
        "gsm8k": (_GSM8K_HF, None),
        "arc_easy": (_ARC_HF, "ARC-Easy"),
        "arc_challenge": (_ARC_HF, "ARC-Challenge"),
        "math": (_MATH_HF, None),
        "open_r1": (_OPEN_R1_HF, None),
        "mmlu": (_MMLU_HF, None),
        "hellaswag": (_HELLASWAG_HF, None),
    }[name]
    config = _CONVERTERS[name][0]
    if split is not None:
        ds = load_dataset(hf_name, config, split=split, trust_remote_code=True) if config \
            else load_dataset(hf_name, split=split, trust_remote_code=True)
        return list(ds)
    ds = load_dataset(hf_name, config, trust_remote_code=True) if config \
        else load_dataset(hf_name, trust_remote_code=True)
    items = []
    for s in ds:
        items.extend(list(ds[s]))
    return items


def _convert(name: str, item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """将原始条目转为标准样本；已含 prompt/response 则透传。"""
    if "prompt" in item and "response" in item:
        item.setdefault("id", f"{name}_{index}")
        item.setdefault("task_type", "unknown")
        item.setdefault("metadata", {})
        return item
    if name == "humaneval":
        return _convert_humaneval(item, index)


    if name == "dolly":
        return _convert_dolly(item, index)
    if name == "magicoder" or ("instruction" in item and "response" in item):
        return _convert_magicoder(item, index)
    return _CONVERTERS[name][1](item, index)


def load_samples(name: str, source: Optional[str] = None,
                 split: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载并转换数据；指定 ``split`` 时只读取该 split，否则合并全部 split。"""
    raw = _load_raw(name, source, split)
    samples = [_convert(name, item, i) for i, item in enumerate(raw)]
    logger.info(f"Loaded {len(samples)} {name} samples")
    return samples


def split_samples(samples: List[Dict[str, Any]], train_ratio: float = 0.8,
                  val_ratio: float = 0.1, seed: int = 42) -> Tuple[List, List]:
    """按比例随机划分 train/val（可复现）。"""
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0 + 1e-9:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to <= 1")
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_train = int(len(samples) * train_ratio)
    n_val = (len(samples) - n_train) if train_ratio + val_ratio >= 1.0 - 1e-9 \
        else int(len(samples) * val_ratio)
    train = [samples[i] for i in idx[:n_train]]
    val = [samples[i] for i in idx[n_train:n_train + n_val]]
    return train, val


def load_data(name: str, source: Optional[str] = None, split: Optional[str] = None,
              seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1,
              train_split: Optional[str] = None,
              validation_split: Optional[str] = None,
              ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    "加载训练/验证数据，绝不把 benchmark test split 混入训练。"
    if name not in _CONVERTERS:
        raise ValueError(f"Unsupported dataset '{name}'; supported={sorted(_CONVERTERS)}")
    if split is not None:
        return split_samples(load_samples(name, source, split), train_ratio, val_ratio, seed)

    default_train, default_val = _DEFAULT_SPLITS[name]
    effective_train = train_split if train_split is not None else default_train
    effective_val = validation_split if validation_split is not None else default_val
    if effective_train is None:
        return split_samples(
            load_samples(name, source, None),
            train_ratio=1.0 - val_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    train_samples = load_samples(name, source, effective_train)
    if effective_val is not None:
        val_samples = load_samples(name, source, effective_val)
        logger.info(
            f"Dataset splits: name={name}, train={effective_train} ({len(train_samples)}), "
            f"validation={effective_val} ({len(val_samples)})"
        )
        return train_samples, val_samples

    heldout_train, heldout_val = split_samples(
        train_samples, train_ratio=1.0 - val_ratio, val_ratio=val_ratio, seed=seed,
    )
    logger.info(
        f"Dataset split: name={name}, source_split={effective_train}, "
        f"train={len(heldout_train)}, validation_holdout={len(heldout_val)}"
    )
    return heldout_train, heldout_val


def sample_retention_subset(
    samples: List[Dict[str, Any]],
    num_samples: Optional[int],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    "从样本列表中无放回随机抽取固定数量子集（``seed`` 可复现，顺序稳定）。"
    if num_samples is None or num_samples >= len(samples):
        return list(samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    idx = sorted(random.Random(seed).sample(range(len(samples)), num_samples))
    return [samples[i] for i in idx]


def load_retention_samples(
    name: str,
    source: Optional[str] = None,
    num_samples: Optional[int] = 256,
    seed: int = 42,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    "加载 RA-BRIDGE 的通用能力保留集并抽取固定数量样本。"
    if name not in _CONVERTERS:
        raise ValueError(f"Unsupported dataset '{name}'; supported={sorted(_CONVERTERS)}")
    effective_split = split if split is not None else _DEFAULT_SPLITS[name][0]
    samples = load_samples(name, source, effective_split)
    subset = sample_retention_subset(samples, num_samples, seed)
    logger.info(
        f"Retention set: name={name}, split={effective_split}, "
        f"sampled={len(subset)}/{len(samples)}, seed={seed}"
    )
    return subset


class BridgeDataset(Dataset):
    "指令微调数据集：prompt / response 独立分词后拼接真实 token。"

    def __init__(
        self,
        data: Union[str, List[Dict[str, Any]]],
        tokenizer: Any,
        max_prompt_length: int = 512,
        max_response_length: int = 512,
        split: str = "train",
        special_token_mode: str = "legacy",
    ):
        super().__init__()
        if special_token_mode not in ("legacy", "single_bos_eos"):
            raise ValueError(
                f"special_token_mode must be 'legacy' or 'single_bos_eos', "
                f"got {special_token_mode!r}"
            )
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.split = split


        self.special_token_mode = special_token_mode

        if isinstance(data, (str, Path)):
            self.data = self._load_jsonl_dir(str(data))
        elif isinstance(data, list):
            self.data = data
        else:
            raise TypeError(f"Expected str/Path/list, got {type(data)}")

        self._cache: Dict[int, Dict[str, Any]] = {}
        logger.info(f"BridgeDataset: {len(self.data)} samples (split={split})")

    @staticmethod
    def _load_jsonl_dir(path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
        samples: List[Dict[str, Any]] = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "prompt" not in s or "response" not in s:
                        continue
                    s.setdefault("id", f"{f.stem}_{line_num}")
                    s.setdefault("task_type", "unknown")
                    s.setdefault("metadata", {})
                    samples.append(s)
        if not samples:
            raise ValueError(f"No valid samples found in {path}")
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            return self._cache[idx]
        result = self._tokenize_sample(self.data[idx])
        self._cache[idx] = result
        return result

    def tokenize_all(self) -> None:
        """预分词全部样本。多线程评分前必须调用，避免 FastTokenizer "Already borrowed"。"""
        for idx in tqdm(range(len(self.data)), desc="Tokenization", unit="sample"):
            if idx not in self._cache:
                self._cache[idx] = self._tokenize_sample(self.data[idx])
        logger.info(f"Pre-tokenized {len(self._cache)} samples")

    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = sample["prompt"]
        response = sample["response"]

        if self.special_token_mode == "single_bos_eos":
            return self._tokenize_sample_single_bos_eos(sample)

        prompt_tokens = self.tokenizer(
            prompt, max_length=self.max_prompt_length, truncation=True,
            padding=False, return_tensors="pt",
        )
        response_tokens = self.tokenizer(
            response, max_length=self.max_response_length, truncation=True,
            padding=False, return_tensors="pt",
        )


        prompt_length = int(prompt_tokens["attention_mask"].sum())
        response_length = int(response_tokens["attention_mask"].sum())

        prompt_ids = prompt_tokens["input_ids"].squeeze(0)
        response_ids = response_tokens["input_ids"].squeeze(0)
        input_ids = torch.cat([prompt_ids, response_ids])
        attention_mask = torch.cat([
            prompt_tokens["attention_mask"].squeeze(0),
            response_tokens["attention_mask"].squeeze(0),
        ])
        prompt_mask = prompt_tokens["attention_mask"].squeeze(0)
        response_mask = response_tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_mask": prompt_mask,
            "response_mask": response_mask,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "sample_id": sample.get("id", ""),
            "task_type": sample.get("task_type", "unknown"),
        }

    def _tokenize_sample_single_bos_eos(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        "``single_bos_eos`` 模式：内容 token 不带 special，开头 1 个 BOS + 末尾 1 个 EOS。"
        prompt = sample["prompt"]
        response = sample["response"]
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        prompt_tokens = self.tokenizer(
            prompt, add_special_tokens=False,
            max_length=max(self.max_prompt_length - 1, 1), truncation=True,
            padding=False, return_tensors="pt",
        )
        response_tokens = self.tokenizer(
            response, add_special_tokens=False,
            max_length=max(self.max_response_length - 1, 1), truncation=True,
            padding=False, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"].squeeze(0)
        response_ids = response_tokens["input_ids"].squeeze(0)
        if bos_id is not None:
            prompt_ids = torch.cat([
                torch.tensor([bos_id], dtype=prompt_ids.dtype), prompt_ids,
            ])
        if eos_id is not None:
            response_ids = torch.cat([
                response_ids, torch.tensor([eos_id], dtype=response_ids.dtype),
            ])

        prompt_length = int(prompt_ids.numel())
        response_length = int(response_ids.numel())
        input_ids = torch.cat([prompt_ids, response_ids])
        attention_mask = torch.ones_like(input_ids)
        prompt_mask = torch.ones_like(prompt_ids)
        response_mask = torch.ones_like(response_ids)

        labels = input_ids.clone()
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_mask": prompt_mask,
            "response_mask": response_mask,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "sample_id": sample.get("id", ""),
            "task_type": sample.get("task_type", "unknown"),
        }

    def select(self, indices: List[int]) -> "BridgeDataset":
        """返回共享分词器与缓存的子集（warmup 用）。"""
        subset = BridgeDataset.__new__(BridgeDataset)
        subset.tokenizer = self.tokenizer
        subset.max_prompt_length = self.max_prompt_length
        subset.max_response_length = self.max_response_length
        subset.split = self.split
        subset.special_token_mode = self.special_token_mode
        valid = [i for i in indices if i < len(self.data)]
        subset.data = [self.data[i] for i in valid]
        subset._cache = {new: self._cache[old] for new, old in enumerate(valid) if old in self._cache}
        return subset


def right_pad_sequences(sequences: List[torch.Tensor], padding_value: int = 0,
                        pad_to_multiple_of: Optional[int] = None) -> torch.Tensor:
    """对一维张量列表右侧填充到统一长度并堆叠为 [N, max_len]。"""
    if not sequences:
        raise ValueError("sequences must be non-empty")
    max_len = max(s.size(0) for s in sequences)
    if pad_to_multiple_of and pad_to_multiple_of > 0:
        r = max_len % pad_to_multiple_of
        if r:
            max_len += pad_to_multiple_of - r
    out = sequences[0].new_full((len(sequences), max_len), padding_value)
    for i, seq in enumerate(sequences):
        out[i, : seq.size(0)].copy_(seq)
    return out


_TENSOR_FIELDS = {"input_ids", "attention_mask", "labels", "prompt_mask", "response_mask"}
_LIST_FIELDS = {"prompt_length", "response_length", "sample_id", "task_type"}


class BridgeDataCollator:
    """动态右填充整理器。``labels`` 用 -100 填充（损失忽略）。标量字段收集为列表。"""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("features must be non-empty")
        all_keys: set = set()
        for f in features:
            all_keys.update(f.keys())
        batch: Dict[str, Any] = {}
        for key in sorted(all_keys & _TENSOR_FIELDS):
            tensors = []
            for f in features:
                v = f.get(key)
                if v is None:
                    tensors.append(torch.tensor([], dtype=torch.long))
                elif isinstance(v, torch.Tensor):
                    tensors.append(v.long().view(-1))
                else:
                    tensors.append(torch.tensor(v, dtype=torch.long).view(-1))
            if key == "labels":
                pad_val = -100
            elif key in {"attention_mask", "prompt_mask", "response_mask"}:
                pad_val = 0
            else:
                pad_val = self.pad_token_id
            batch[key] = right_pad_sequences(tensors, pad_val, self.pad_to_multiple_of)
        for key in sorted(all_keys & _LIST_FIELDS):
            batch[key] = [f.get(key) for f in features]
        for key in sorted(all_keys - _TENSOR_FIELDS - _LIST_FIELDS):
            batch[key] = [f.get(key) for f in features]
        return batch
