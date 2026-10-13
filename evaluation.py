"BRIDGE 评估：答案抽取、选择题/数学推理准确率、cost 指标。"

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_data, load_samples
from utils import get_logger

logger = get_logger("bridge.eval")


_GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_BOLD_ANSWER_RE = re.compile(r"\*\*([^*]+)\*\*", re.MULTILINE)
_ANSWER_IS_RE = re.compile(r"(?:the\s+)?answer\s*(?:is|:|=)\s*\**\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_LAST_NUMBER_RE = re.compile(r"(-?[\d,]*\.?\d+(?:e[+-]?\d+)?)\s*$", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\b([A-E])\b")
_LEADING_CHOICE_RE = re.compile(r"^\s*\(?([A-E])\)?(?:[.\s:]|$)", re.IGNORECASE)
_EXPLICIT_CHOICE_RE = re.compile(
    r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:|=)\s*\(?([A-E])\)?\b",
    re.IGNORECASE,
)
_STRIP_RE = re.compile(r"[,\s]")


def extract_math_answer(text: str) -> str:
    """从模型输出提取数学答案：#### → \\boxed{} → **bold** → "answer is N" → 最后一个数字。"""
    if not text:
        return ""
    m = _GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _BOLD_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = _LAST_NUMBER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    return ""


def extract_choice_answer(text: str) -> str:
    """Extract A-E, prioritizing the option emitted at the start of a reply."""
    if not text:
        return ""
    leading = _LEADING_CHOICE_RE.search(text)
    if leading:
        return leading.group(1).upper()
    explicit = _EXPLICIT_CHOICE_RE.search(text)
    if explicit:
        return explicit.group(1).upper()
    matches = _CHOICE_RE.findall(text.upper())
    return matches[0] if matches else ""


def compare_answers(predicted: str, ground_truth: str, task_type: str) -> bool:
    """比较预测与真实答案。alignment 为字母匹配；math/gsm8k 为标准化数值比较。"""
    if not predicted or not ground_truth:
        return False
    pred, gt = predicted.strip(), ground_truth.strip()
    if task_type.lower() in ("alignment", "arc"):
        return pred.upper() == gt.upper()
    pred_norm = _STRIP_RE.sub("", pred)
    gt_norm = _STRIP_RE.sub("", gt)
    if pred_norm == gt_norm:
        return True
    try:
        return abs(float(pred_norm) - float(gt_norm)) < 1e-6
    except (ValueError, TypeError):
        return pred_norm.lower() == gt_norm.lower()


def _choice_index(answer_key: Any, labels: Sequence[str]) -> int:
    """Resolve letter, numeric label, or zero-based index to a choice index."""
    key = str(answer_key).strip().upper()
    normalized_labels = [str(label).strip().upper() for label in labels]
    if key in normalized_labels:
        return normalized_labels.index(key)
    if len(key) == 1 and "A" <= key <= "Z":
        index = ord(key) - ord("A")
        return index if index < len(labels) else -1
    try:
        value = int(key)
    except (TypeError, ValueError):
        return -1
    if 0 <= value < len(labels):
        return value
    if 1 <= value <= len(labels):
        return value - 1
    return -1


_GEN_PROMPT_MAX_LENGTH = 2048
_GEN_TOP_P = 0.95


def prepare_model_for_evaluation(model):
    "Put a trained model into a generation-safe inference state."
    base_model = model.model if hasattr(model, "model") else model
    was_training = bool(model.training)
    checkpointing_enabled = bool(
        getattr(base_model, "is_gradient_checkpointing", False)
    )
    model.eval()
    if checkpointing_enabled and hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()

    config = getattr(base_model, "config", None)
    cache_was_disabled = config is not None and not bool(getattr(config, "use_cache", True))
    if config is not None:
        config.use_cache = True
    generation_config = getattr(base_model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = True

    if was_training or checkpointing_enabled or cache_was_disabled:
        logger.info(
            "Evaluation state prepared: eval_mode=true, "
            "gradient_checkpointing=false, use_cache=true"
        )
    return model


def generate_answers(model, tokenizer, prompts: List[str], max_new_tokens: int,
                     temperature: float = 0.0, device: Optional[torch.device] = None) -> List[str]:
    """批量生成（左 padding），返回新生成的解码文本。"""
    prepare_model_for_evaluation(model)
    if device is None:
        device = model.input_device if hasattr(model, "input_device") else next(model.parameters()).device
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=_GEN_PROMPT_MAX_LENGTH).to(device)
    do_sample = temperature > 0
    generation_config = getattr(model.model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if isinstance(eos_token_id, (list, tuple)):
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("Generation requires EOS and PAD token IDs")
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=_GEN_TOP_P if do_sample else None,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
            repetition_penalty=1.0,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    vocab_size = len(tokenizer)
    invalid = (generated < 0) | (generated >= vocab_size)
    if bool(invalid.any()):
        invalid_values = generated[invalid][:8].detach().cpu().tolist()
        raise RuntimeError(
            "Generation produced token IDs outside the tokenizer vocabulary: "
            f"dtype={generated.dtype}, valid=[0,{vocab_size}), values={invalid_values}"
        )
    return [
        tokenizer.decode(generated[i], skip_special_tokens=True)
        for i in range(generated.shape[0])
    ]


GSM8K_8SHOT: List[Dict[str, str]] = [
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6"},
    {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"},
    {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.\n#### 8"},
    {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"},
    {"question": "Angelo wants to buy some candies. He has $42. Each candy costs $3. How many candies can he buy?",
     "answer": "Angelo has 42 dollars. Each candy costs 3 dollars. So he can buy 42 / 3 = 14 candies.\n#### 14"},
    {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29 computers.\n#### 29"},
    {"question": "Sam bought a dozen boxes, each with 30 highlighter pens inside. He re-packed five of these boxes into packages of six highlighters each. He sold the rest at a rate of $2 per pen. How much did Sam make?",
     "answer": "Sam bought 12 boxes * 30 highlighters = 360 highlighters in total. He repacked 5 boxes * 30 highlighters / 6 highlighters per pack = 25 packs. He had 360 - 5*30 = 210 highlighters left to sell at $2 each. 210 * 2 = $420.\n#### 420"},
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per egg. How much in dollars does she make every day at the farmers' market?",
     "answer": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4, so she uses 3 + 4 = 7 eggs. She sells the remainder: 16 - 7 = 9 eggs. At $2 per egg, she makes 9 * 2 = $18.\n#### 18"},
]


def _build_few_shot_prompt(prompt: str, examples: List[Dict[str, str]], num_few_shot: int) -> str:
    """在 prompt 前拼接少样本示例。"""
    if not examples or num_few_shot <= 0:
        return prompt
    parts: List[str] = []
    for ex in examples[:num_few_shot]:
        q = ex.get("question", ex.get("prompt", ""))
        a = ex.get("answer", ex.get("response", ""))
        parts.append(
            f"Given the following problem, reason step by step and "
            f"write your final answer after ####.\n\n"
            f"Problem: {q.strip()}\n\nAnswer: Let's think step by step.\n{a.strip()}\n"
        )
    parts.append(prompt)
    return "\n\n".join(parts)


def _load_eval_samples(name: str, source: Optional[str],
                       preferred_split: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载评估数据；HellaSwag 默认使用有标签的 validation split。"""
    logger.info(f"Loading eval '{name}' from {'local: ' + source if source else 'HuggingFace hub'}")
    primary = preferred_split or ("validation" if name == "hellaswag" else "test")
    fallbacks = [primary] + [s for s in ("test", "validation") if s != primary]
    errors = []
    for split in fallbacks:
        try:
            samples = load_samples(name, source, split=split)
            logger.info(
                f"Evaluation split: benchmark={name}, split={split}, samples={len(samples)}"
            )
            return samples
        except Exception as error:
            errors.append(f"{split}: {error}")
    raise ValueError(f"No evaluation split available for {name}: {'; '.join(errors)}")


def _eval_source(config, bench: str) -> Optional[str]:
    """从 ``config.eval_dataset_path`` 取某 benchmark 的本地数据路径；未配置则 None(走 HF hub)。"""
    paths = getattr(config, "eval_dataset_path", None)
    return paths.get(bench) if isinstance(paths, dict) else None


def _eval_split(config, bench: str) -> Optional[str]:
    splits = getattr(config, "eval_split", None)
    return splits.get(bench) if isinstance(splits, dict) else None


def _open_response_file(response_dir: Optional[Path], task_name: str):
    """打开逐样本回复 JSONL；流式写入使中断前的结果仍可审阅。"""
    if response_dir is None:
        return None, None
    response_dir = Path(response_dir)
    response_dir.mkdir(parents=True, exist_ok=True)
    path = response_dir / f"{task_name}.jsonl"
    return path.open("w", encoding="utf-8"), path


def _eval_math(data, model, tokenizer, config, task_name, few_shot=False,
               response_dir: Optional[Path] = None) -> Dict[str, Any]:
    batch_size = getattr(config, "eval_batch_size", 16)
    max_gen = getattr(config, "max_gen_tokens", 512)
    temperature = getattr(config, "temperature", 0.0)
    num_samples = getattr(config, "num_eval_samples", None)
    few_shot_examples = getattr(config, "few_shot_examples", None) or GSM8K_8SHOT
    default_few_shot = int(getattr(config, "num_few_shot", 8))
    num_few_shot = (
        _paper_few_shot_count(config, task_name, default_few_shot) if few_shot else 0
    )

    n = min(num_samples or len(data), len(data))
    data = data[:n]

    num_correct, num_total = 0, 0
    response_file, response_path = _open_response_file(response_dir, task_name)
    progress = tqdm(total=len(data), desc=f"Eval {task_name}", unit="sample")
    try:
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start: batch_start + batch_size]
            raw_prompts = [s["prompt"] for s in batch]
            prompts = raw_prompts
            gts = [s["metadata"].get("final_answer", s["response"]) for s in batch]
            if few_shot:
                prompts = [_build_few_shot_prompt(p, few_shot_examples, num_few_shot) for p in prompts]
            responses = generate_answers(model, tokenizer, prompts, max_gen, temperature)
            for idx_in_batch, (pred, gt, raw_prompt) in enumerate(zip(responses, gts, raw_prompts)):
                num_total += 1
                extracted = extract_math_answer(pred)
                correct = compare_answers(extracted, gt, "math")
                if correct:
                    num_correct += 1
                if response_file is not None:
                    response_file.write(json.dumps({
                        "index": batch_start + idx_in_batch,
                        "benchmark": task_name,
                        "prompt": raw_prompt,
                        "response": pred,
                        "extracted_answer": extracted,
                        "reference_answer": gt,
                        "correct": correct,
                        "few_shot": num_few_shot if few_shot else 0,
                    }, ensure_ascii=False) + "\n")
            if response_file is not None:
                response_file.flush()
            progress.update(len(batch))
            progress.set_postfix(accuracy=f"{num_correct / max(num_total, 1):.4f}")
    finally:
        progress.close()
        if response_file is not None:
            response_file.close()
    acc = num_correct / num_total if num_total else 0.0
    logger.info(f"Eval {task_name}: accuracy={acc:.4f} ({num_correct}/{num_total})")
    result = {"accuracy": acc, "num_correct": num_correct, "num_total": num_total}
    if response_path is not None:
        result["responses_file"] = str(response_path)
    return result


def _eval_multiple_choice(data, model, tokenizer, config, task_name,
                          response_dir: Optional[Path] = None) -> Dict[str, Any]:
    batch_size = getattr(config, "eval_batch_size", 16)
    max_gen = getattr(config, "max_gen_tokens", 256)
    temperature = getattr(config, "temperature", 0.0)
    num_samples = getattr(config, "num_eval_samples", None)

    n = min(num_samples or len(data), len(data))
    data = data[:n]
    labeled = [sample for sample in data if sample.get("metadata", {}).get("answer_key")]
    if len(labeled) != len(data):
        logger.warning(
            f"Eval {task_name}: skipped {len(data) - len(labeled)} unlabeled samples"
        )
        data = labeled
    if not data:
        raise ValueError(f"Evaluation dataset '{task_name}' contains no labeled samples")

    num_correct, num_total = 0, 0
    response_file, response_path = _open_response_file(response_dir, task_name)
    progress = tqdm(total=len(data), desc=f"Eval {task_name}", unit="sample")
    try:
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start: batch_start + batch_size]
            raw_prompts = [s["prompt"] for s in batch]
            prompts = [p + "\n\nAnswer with only the option letter:" for p in raw_prompts]
            gts = [s["metadata"].get("answer_key", s["response"]) for s in batch]

            choice_max_gen = min(max_gen, 16)
            responses = generate_answers(model, tokenizer, prompts, choice_max_gen, temperature)
            for idx_in_batch, (pred, gt, raw_prompt, sample) in enumerate(
                zip(responses, gts, raw_prompts, batch)
            ):
                num_total += 1
                pred_letter = extract_choice_answer(pred)
                labels = sample.get("metadata", {}).get("choice_labels") or ["A", "B", "C", "D"]
                gt_index = _choice_index(gt, labels)
                gt_letter = chr(ord("A") + gt_index) if gt_index >= 0 else ""
                correct = compare_answers(pred_letter, gt_letter, "alignment")
                if correct:
                    num_correct += 1
                if response_file is not None:
                    response_file.write(json.dumps({
                        "index": batch_start + idx_in_batch,
                        "benchmark": task_name,
                        "prompt": raw_prompt,
                        "response": pred,
                        "extracted_answer": pred_letter,
                        "reference_answer": gt_letter,
                        "correct": correct,
                    }, ensure_ascii=False) + "\n")
            if response_file is not None:
                response_file.flush()
            progress.update(len(batch))
            progress.set_postfix(accuracy=f"{num_correct / max(num_total, 1):.4f}")
    finally:
        progress.close()
        if response_file is not None:
            response_file.close()
    acc = num_correct / num_total if num_total else 0.0
    logger.info(f"Eval {task_name}: accuracy={acc:.4f} ({num_correct}/{num_total})")
    result = {"accuracy": acc, "num_correct": num_correct, "num_total": num_total}
    if response_path is not None:
        result["responses_file"] = str(response_path)
    return result


def _paper_few_shot_count(config, benchmark: str, default: int) -> int:
    counts = getattr(config, "eval_few_shot", None)
    if isinstance(counts, dict) and benchmark in counts:
        return int(counts[benchmark])
    return default


def _format_choices(question: str, choices: Sequence[str], answer: Optional[str] = None) -> str:
    lines = [question]
    lines.extend(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    suffix = "Answer:" if answer is None else f"Answer: {answer}"
    lines.append(suffix)
    return "\n".join(lines)


def _mmlu_context(sample: Dict[str, Any], examples: Sequence[Dict[str, Any]]) -> str:
    metadata = sample["metadata"]
    subject = str(metadata.get("subject", "")).replace("_", " ")
    parts = [f"The following are multiple choice questions (with answers) about {subject}.\n"]
    for example in examples:
        ex_meta = example["metadata"]
        parts.append(_format_choices(
            str(ex_meta.get("question", "")),
            ex_meta.get("choices", []),
            str(ex_meta.get("answer_key", "")),
        ))
    parts.append(_format_choices(
        str(metadata.get("question", "")), metadata.get("choices", []), None,
    ))
    return "\n\n".join(parts)


def _candidate_request(
    sample: Dict[str, Any], benchmark: str,
    mmlu_examples: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Tuple[str, List[str], List[str], int]:
    metadata = sample.get("metadata", {})
    choices = [str(choice) for choice in metadata.get("choices", [])]
    labels = [str(label) for label in metadata.get("choice_labels", [])]
    if not choices:
        raise ValueError(f"{benchmark} sample {sample.get('id')} has no candidate choices")
    if not labels:
        labels = [chr(65 + index) for index in range(len(choices))]
    gold_index = _choice_index(metadata.get("answer_key", ""), labels)
    if not 0 <= gold_index < len(choices):
        raise ValueError(
            f"{benchmark} sample {sample.get('id')} has invalid answer key "
            f"{metadata.get('answer_key')!r} for labels={labels}"
        )

    if benchmark in ("arc_easy", "arc_challenge"):
        context = f"Question: {metadata.get('question', '')}\nAnswer:"
        continuations = [" " + choice for choice in choices]
    elif benchmark == "hellaswag":
        context = str(metadata.get("context", ""))
        continuations = [" " + choice.lstrip() for choice in choices]
    elif benchmark == "mmlu":
        subject = str(metadata.get("subject", ""))
        examples = (mmlu_examples or {}).get(subject, [])
        context = _mmlu_context(sample, examples)
        continuations = [" " + chr(65 + index) for index in range(len(choices))]
    else:
        raise ValueError(f"Candidate likelihood is not configured for {benchmark}")
    return context, continuations, labels, gold_index


def _score_continuations(
    model,
    tokenizer,
    requests: Sequence[Tuple[str, str]],
    max_length: int,
) -> List[Tuple[float, float, int]]:
    """Return (sum log-likelihood, mean log-likelihood, token count)."""
    if not requests:
        return []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    encoded: List[Tuple[List[int], List[int]]] = []
    for context, continuation in requests:
        context_ids = tokenizer.encode(context, add_special_tokens=True)
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not continuation_ids:
            raise ValueError(f"Candidate continuation tokenized to zero tokens: {continuation!r}")
        max_context = max_length - len(continuation_ids)
        if max_context < 1:
            raise ValueError(
                f"Candidate uses {len(continuation_ids)} tokens, exceeding max_length={max_length}"
            )
        context_ids = context_ids[-max_context:]
        encoded.append((context_ids + continuation_ids, continuation_ids))

    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    for row, (ids, continuation_ids) in enumerate(encoded):
        length = len(ids)
        continuation_start = length - len(continuation_ids)
        input_ids[row, :length] = torch.tensor(ids)
        attention_mask[row, :length] = 1
        labels[row, continuation_start:length] = torch.tensor(continuation_ids)

    device = model.input_device if hasattr(model, "input_device") else next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)


    first_target = min(len(ids) - len(continuation_ids) for ids, continuation_ids in encoded)
    logits_to_keep = width - (first_target - 1)
    forward_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    with torch.inference_mode():
        try:
            outputs = model.model(logits_to_keep=logits_to_keep, **forward_kwargs)
        except TypeError as error:
            if "logits_to_keep" not in str(error):
                raise
            outputs = model.model(**forward_kwargs)
    logits = outputs.logits
    logits_start = width - logits.shape[1]
    results: List[Tuple[float, float, int]] = []
    for row in range(len(encoded)):
        target_positions = labels[row].ne(-100).nonzero(as_tuple=False).flatten()
        predictor_positions = target_positions - 1
        relative_positions = predictor_positions - logits_start
        if bool((relative_positions < 0).any() or (relative_positions >= logits.shape[1]).any()):
            raise RuntimeError(
                "Continuation logits were truncated before scoring: "
                f"width={width}, logits={logits.shape[1]}, targets={target_positions.tolist()}"
            )
        selected_logits = logits[row, relative_positions].float()
        selected_targets = labels[row, target_positions]
        losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
        token_count = int(losses.numel())
        log_likelihood = -float(losses.sum().item())
        results.append((log_likelihood, log_likelihood / token_count, token_count))
    del outputs, logits, input_ids, attention_mask, labels
    return results


def _eval_candidate_likelihood(
    data,
    model,
    tokenizer,
    config,
    task_name: str,
    source: Optional[str] = None,
    response_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    num_samples = getattr(config, "num_eval_samples", None)
    data = data[:min(num_samples or len(data), len(data))]
    if not data:
        raise ValueError(f"Evaluation dataset '{task_name}' is empty")

    mmlu_examples: Dict[str, List[Dict[str, Any]]] = {}
    few_shot = 0
    if task_name == "mmlu":
        few_shot = _paper_few_shot_count(config, "mmlu", 5)
        if few_shot:
            development = load_samples("mmlu", source, split="dev")
            for example in development:
                subject = str(example.get("metadata", {}).get("subject", ""))
                bucket = mmlu_examples.setdefault(subject, [])
                if len(bucket) < few_shot:
                    bucket.append(example)

    requests = [
        _candidate_request(sample, task_name, mmlu_examples) for sample in data
    ]
    likelihood_batch_size = max(1, int(getattr(config, "likelihood_batch_size", 8)))
    max_length = int(getattr(config, "likelihood_max_length", 4096))
    raw_correct = normalized_correct = 0
    response_file, response_path = _open_response_file(response_dir, task_name)
    progress = tqdm(total=len(data), desc=f"Eval {task_name} likelihood", unit="sample")
    try:
        sample_batch_size = max(1, likelihood_batch_size // 4)
        for batch_start in range(0, len(data), sample_batch_size):
            batch_requests = requests[batch_start:batch_start + sample_batch_size]
            flattened: List[Tuple[str, str]] = []
            spans: List[Tuple[int, int]] = []
            for context, continuations, _, _ in batch_requests:
                start = len(flattened)
                flattened.extend((context, continuation) for continuation in continuations)
                spans.append((start, len(flattened)))
            scores: List[Tuple[float, float, int]] = []
            for chunk_start in range(0, len(flattened), likelihood_batch_size):
                scores.extend(_score_continuations(
                    model,
                    tokenizer,
                    flattened[chunk_start:chunk_start + likelihood_batch_size],
                    max_length,
                ))

            for offset, ((_, continuations, labels, gold_index), (start, end)) in enumerate(
                zip(batch_requests, spans)
            ):
                candidate_scores = scores[start:end]
                raw_index = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i][0])
                normalized_index = max(
                    range(len(candidate_scores)), key=lambda i: candidate_scores[i][1]
                )
                raw_correct += int(raw_index == gold_index)
                normalized_correct += int(normalized_index == gold_index)
                if response_file is not None:
                    response_file.write(json.dumps({
                        "index": batch_start + offset,
                        "sample_id": data[batch_start + offset].get("id"),
                        "benchmark": task_name,
                        "choice_labels": labels,
                        "continuations": continuations,
                        "log_likelihood": [score[0] for score in candidate_scores],
                        "mean_log_likelihood": [score[1] for score in candidate_scores],
                        "candidate_token_counts": [score[2] for score in candidate_scores],
                        "raw_prediction_index": raw_index,
                        "normalized_prediction_index": normalized_index,
                        "reference_index": gold_index,
                        "raw_correct": raw_index == gold_index,
                        "normalized_correct": normalized_index == gold_index,
                        "few_shot": few_shot,
                    }, ensure_ascii=False) + "\n")
            if response_file is not None:
                response_file.flush()
            progress.update(len(batch_requests))
            primary_correct = raw_correct if task_name == "mmlu" else normalized_correct
            progress.set_postfix(accuracy=f"{primary_correct / progress.n:.4f}")
    finally:
        progress.close()
        if response_file is not None:
            response_file.close()

    total = len(data)
    raw_accuracy = raw_correct / total
    normalized_accuracy = normalized_correct / total
    use_normalized = task_name in ("arc_easy", "arc_challenge", "hellaswag")
    result = {
        "accuracy": normalized_accuracy if use_normalized else raw_accuracy,
        "accuracy_raw": raw_accuracy,
        "accuracy_normalized": normalized_accuracy,
        "metric": "acc_norm" if use_normalized else "acc",
        "num_correct": normalized_correct if use_normalized else raw_correct,
        "num_total": total,
        "few_shot": few_shot,
        "scoring": "candidate_log_likelihood",
    }
    if response_path is not None:
        result["responses_file"] = str(response_path)
    logger.info(
        f"Eval {task_name} likelihood: {result['metric']}={result['accuracy']:.4f}, "
        f"acc={raw_accuracy:.4f}, acc_norm={normalized_accuracy:.4f} ({total} samples)"
    )
    return result


def evaluate_arc(model, tokenizer, config, variant: str, source: Optional[str] = None,
                 response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """ARC-Easy / ARC-Challenge 准确率。"""
    name = f"arc_{variant}"
    data = _load_eval_samples(name, source, _eval_split(config, name))
    if getattr(config, "multiple_choice_mode", "generation") == "likelihood":
        return _eval_candidate_likelihood(
            data, model, tokenizer, config, name, source=source,
            response_dir=response_dir,
        )
    return _eval_multiple_choice(data, model, tokenizer, config, name, response_dir=response_dir)


def evaluate_multiple_choice_benchmark(model, tokenizer, config, name: str,
                                       source: Optional[str] = None,
                                       response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """MMLU / HellaSwag 的生成式选择题准确率。"""
    data = _load_eval_samples(name, source, _eval_split(config, name))
    if getattr(config, "multiple_choice_mode", "generation") == "likelihood":
        return _eval_candidate_likelihood(
            data, model, tokenizer, config, name, source=source,
            response_dir=response_dir,
        )
    return _eval_multiple_choice(data, model, tokenizer, config, name, response_dir=response_dir)


def evaluate_gsm8k(model, tokenizer, config, source: Optional[str] = None,
                   few_shot: bool = True, response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """GSM8K 准确率（默认 8-shot）。"""
    data = _load_eval_samples("gsm8k", source, _eval_split(config, "gsm8k"))
    return _eval_math(data, model, tokenizer, config, "gsm8k", few_shot=few_shot,
                      response_dir=response_dir)


def evaluate_math(model, tokenizer, config, source: Optional[str] = None,
                  response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Hendrycks MATH 准确率。"""
    data = _load_eval_samples("math", source, _eval_split(config, "math"))
    return _eval_math(data, model, tokenizer, config, "math", few_shot=False,
                      response_dir=response_dir)


def evaluate_open_r1(model, tokenizer, config, source: Optional[str] = None,
                     response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Open-R1-Math 推理准确率。"""
    data = _load_eval_samples("open_r1", source, _eval_split(config, "open_r1"))
    return _eval_math(data, model, tokenizer, config, "open_r1", few_shot=False,
                      response_dir=response_dir)


def _normalize_code(text: str) -> str:
    """Normalize line endings and trailing whitespace without changing code tokens."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def evaluate_magicoder(model, tokenizer, config, source: Optional[str] = None,
                       response_dir: Optional[Path] = None) -> Dict[str, Any]:
    "Generate on the deterministic Magicoder validation holdout."
    if not source:
        raise ValueError("Magicoder evaluation requires eval_dataset_path.magicoder")
    _, data = load_data("magicoder", source, seed=getattr(config, "seed", 42))
    limit = getattr(config, "num_code_eval_samples", 256)
    n = min(limit if limit is not None else len(data), len(data))
    data = data[:n]
    if not data:
        raise ValueError("Magicoder validation holdout is empty")

    batch_size = getattr(config, "eval_batch_size", 16)
    max_gen = getattr(config, "max_code_gen_tokens", 512)
    temperature = getattr(config, "temperature", 0.0)
    exact, normalized_exact, nonempty = 0, 0, 0
    response_file, response_path = _open_response_file(response_dir, "magicoder")
    progress = tqdm(total=len(data), desc="Eval magicoder", unit="sample")
    try:
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start: batch_start + batch_size]
            prompts = [sample["prompt"] for sample in batch]
            references = [sample["response"] for sample in batch]
            responses = generate_answers(model, tokenizer, prompts, max_gen, temperature)
            for offset, (sample, prediction, reference) in enumerate(
                zip(batch, responses, references)
            ):
                is_exact = prediction.strip() == reference.strip()
                is_normalized_exact = _normalize_code(prediction) == _normalize_code(reference)
                exact += int(is_exact)
                normalized_exact += int(is_normalized_exact)
                nonempty += int(bool(prediction.strip()))
                if response_file is not None:
                    response_file.write(json.dumps({
                        "index": batch_start + offset,
                        "sample_id": sample.get("id"),
                        "benchmark": "magicoder",
                        "prompt": sample["prompt"],
                        "response": prediction,
                        "reference_response": reference,
                        "exact_match": is_exact,
                        "normalized_exact_match": is_normalized_exact,
                    }, ensure_ascii=False) + "\n")
            if response_file is not None:
                response_file.flush()
            completed = batch_start + len(batch)
            progress.update(len(batch))
            progress.set_postfix(nonempty=f"{nonempty / completed:.4f}")
    finally:
        progress.close()
        if response_file is not None:
            response_file.close()

    total = len(data)
    result = {
        "exact_match": exact / total,
        "normalized_exact_match": normalized_exact / total,
        "nonempty_rate": nonempty / total,
        "num_total": total,
        "metric_note": "Magicoder has no executable tests; pass@k is not available.",
    }
    if response_path is not None:
        result["responses_file"] = str(response_path)
    logger.info(
        f"Eval magicoder: exact={result['exact_match']:.4f}, "
        f"normalized_exact={result['normalized_exact_match']:.4f}, "
        f"nonempty={result['nonempty_rate']:.4f} ({total} samples)"
    )
    return result


def _extract_humaneval_completion(raw_response: str, prompt: str, entry_point: str) -> str:
    """Keep the generated function body and remove markdown or trailing top-level text."""
    text = raw_response.replace("\r\n", "\n")
    if text.startswith(prompt):
        text = text[len(prompt):]
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)

    lines = text.splitlines(keepends=True)
    repeated_definition = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
    for index, line in enumerate(lines):
        if repeated_definition.match(line):
            lines = lines[index + 1:]
            break

    kept: List[str] = []
    body_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if body_started:
                kept.append(line)
            continue
        if line.startswith((" ", "\t")):
            body_started = True
            kept.append(line)
            continue
        if body_started:
            break

        kept.append(line)
        break
    return "".join(kept).rstrip() + ("\n" if kept else "")


_HUMANEVAL_GUARD = r'''
import os
import socket
import subprocess

def _disabled(*args, **kwargs):
    raise PermissionError("operation disabled during HumanEval")

for _name in (
    "system", "fork", "forkpty", "kill", "killpg", "remove", "removedirs",
    "rename", "replace", "rmdir", "unlink", "chmod", "chown", "chroot",
):
    if hasattr(os, _name):
        setattr(os, _name, _disabled)
socket.socket = _disabled
socket.create_connection = _disabled
subprocess.Popen = _disabled
subprocess.run = _disabled
subprocess.call = _disabled
subprocess.check_call = _disabled
subprocess.check_output = _disabled
'''


def _humaneval_resource_limits(timeout: float, memory_limit_mb: int):
    def apply_limits() -> None:
        import resource

        memory_bytes = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_seconds = max(1, int(math.ceil(timeout)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return apply_limits


def _run_humaneval_test(
    sample: Dict[str, Any], completion: str, timeout: float, memory_limit_mb: int,
) -> Dict[str, Any]:
    metadata = sample["metadata"]
    source = (
        _HUMANEVAL_GUARD
        + "\n"
        + sample["prompt"]
        + completion
        + "\n"
        + str(metadata["test"])
        + f"\ncheck({metadata['entry_point']})\n"
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bridge_humaneval_") as directory:
        runner = Path(directory) / "candidate.py"
        output = Path(directory) / "output.log"
        runner.write_text(source, encoding="utf-8")
        try:
            with output.open("wb") as output_file:
                completed = subprocess.run(
                    [sys.executable, "-I", str(runner)],
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                    start_new_session=True,
                    preexec_fn=_humaneval_resource_limits(timeout, memory_limit_mb),
                    env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                )
            passed = completed.returncode == 0
            status = "passed" if passed else "failed"
            error = output.read_text(encoding="utf-8", errors="replace")[-4000:]
        except subprocess.TimeoutExpired:
            passed = False
            status = "timed_out"
            error = f"exceeded {timeout:.1f}s timeout"
        except Exception as exc:
            passed = False
            status = "executor_error"
            error = repr(exc)
    return {
        "passed": passed,
        "status": status,
        "error": error,
        "execution_seconds": time.monotonic() - started,
    }


def evaluate_humaneval(model, tokenizer, config, source: Optional[str] = None,
                       response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run HumanEval with one deterministic completion and report pass@1."""
    if not source:
        raise ValueError("HumanEval evaluation requires eval_dataset_path.humaneval")
    data = load_samples("humaneval", source, split=_eval_split(config, "humaneval") or "test")
    limit = getattr(config, "num_code_eval_samples", None)
    data = data[:min(limit or len(data), len(data))]
    if not data:
        raise ValueError("HumanEval test split is empty")

    batch_size = max(1, int(getattr(config, "eval_batch_size", 8)))
    max_gen = int(getattr(config, "max_code_gen_tokens", 512))
    temperature = float(getattr(config, "temperature", 0.0))
    timeout = float(getattr(config, "humaneval_timeout", 5.0))
    workers = max(1, int(getattr(config, "humaneval_workers", 4)))
    memory_limit_mb = int(getattr(config, "humaneval_memory_limit_mb", 1024))

    generated: List[Tuple[Dict[str, Any], str, str]] = []
    generation_progress = tqdm(total=len(data), desc="Generate humaneval", unit="sample")
    try:
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start:batch_start + batch_size]
            raw_responses = generate_answers(
                model, tokenizer, [sample["prompt"] for sample in batch], max_gen, temperature,
            )
            for sample, raw_response in zip(batch, raw_responses):
                completion = _extract_humaneval_completion(
                    raw_response, sample["prompt"], sample["metadata"]["entry_point"],
                )
                generated.append((sample, raw_response, completion))
            generation_progress.update(len(batch))
    finally:
        generation_progress.close()

    response_file, response_path = _open_response_file(response_dir, "humaneval")
    outcomes: List[Optional[Dict[str, Any]]] = [None] * len(generated)
    execution_progress = tqdm(total=len(generated), desc="Execute humaneval", unit="sample")
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_humaneval_test, sample, completion, timeout, memory_limit_mb,
                ): index
                for index, (sample, _, completion) in enumerate(generated)
            }
            for future in as_completed(futures):
                index = futures[future]
                outcomes[index] = future.result()
                execution_progress.update(1)
                completed_count = execution_progress.n
                passed_count = sum(
                    bool(outcome and outcome["passed"]) for outcome in outcomes
                )
                execution_progress.set_postfix(pass_at_1=f"{passed_count / completed_count:.4f}")
    finally:
        execution_progress.close()

    passed = 0
    status_counts: Dict[str, int] = {}
    try:
        for index, ((sample, raw_response, completion), outcome) in enumerate(
            zip(generated, outcomes)
        ):
            if outcome is None:
                outcome = {
                    "passed": False,
                    "status": "executor_error",
                    "error": "missing executor result",
                    "execution_seconds": 0.0,
                }
            passed += int(outcome["passed"])
            status = str(outcome["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            if response_file is not None:
                response_file.write(json.dumps({
                    "index": index,
                    "task_id": sample["metadata"]["task_id"],
                    "benchmark": "humaneval",
                    "prompt": sample["prompt"],
                    "raw_response": raw_response,
                    "completion": completion,
                    **outcome,
                }, ensure_ascii=False) + "\n")
        if response_file is not None:
            response_file.flush()
    finally:
        if response_file is not None:
            response_file.close()

    total = len(generated)
    result = {
        "pass_at_1": passed / total,
        "num_passed": passed,
        "num_total": total,
        "status_counts": status_counts,
        "temperature": temperature,
        "num_samples_per_task": 1,
        "execution_timeout_seconds": timeout,
    }
    if response_path is not None:
        result["responses_file"] = str(response_path)
    logger.info(f"Eval humaneval: pass@1={result['pass_at_1']:.4f} ({passed}/{total})")
    return result


def evaluate_alignment(model, tokenizer, config,
                       response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """运行所有配置的选择题/对齐基准。"""
    benchmarks = getattr(config, "alignment_benchmarks", None) or ["arc_easy", "arc_challenge"]
    results: Dict[str, Any] = {}
    for bench in benchmarks:
        src = _eval_source(config, bench)
        if bench in ("arc_easy", "arc_challenge"):
            results[bench] = evaluate_arc(
                model, tokenizer, config, bench.split("_")[1], source=src,
                response_dir=response_dir,
            )
        elif bench in ("mmlu", "hellaswag"):
            results[bench] = evaluate_multiple_choice_benchmark(
                model, tokenizer, config, bench, source=src, response_dir=response_dir,
            )
        elif bench == "gsm8k":
            results[bench] = evaluate_gsm8k(
                model, tokenizer, config, source=src, few_shot=False,
                response_dir=response_dir,
            )
        else:
            logger.warning(f"Unknown alignment benchmark: {bench}")
    return results


def evaluate_reasoning(model, tokenizer, config,
                       response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """运行所有配置的推理基准（GSM8K / MATH / Open-R1）。"""
    benchmarks = getattr(config, "reasoning_benchmarks", None) or ["gsm8k"]
    results: Dict[str, Any] = {}
    for bench in benchmarks:
        src = _eval_source(config, "open_r1" if bench == "open_r1_math" else bench)
        if bench == "gsm8k":
            results[bench] = evaluate_gsm8k(
                model, tokenizer, config, source=src, few_shot=True,
                response_dir=response_dir,
            )
        elif bench == "math":
            results[bench] = evaluate_math(
                model, tokenizer, config, source=src, response_dir=response_dir,
            )
        elif bench in ("open_r1", "open_r1_math"):
            results[bench] = evaluate_open_r1(
                model, tokenizer, config, source=src, response_dir=response_dir,
            )
        else:
            logger.warning(f"Unknown reasoning benchmark: {bench}")
    return results


def evaluate_code(model, tokenizer, config,
                  response_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run configured code-generation benchmarks."""
    benchmarks = getattr(config, "code_benchmarks", None)
    if not benchmarks:
        return {}
    results: Dict[str, Any] = {}
    for bench in benchmarks:
        if bench == "magicoder":
            results[bench] = evaluate_magicoder(
                model, tokenizer, config, source=_eval_source(config, bench),
                response_dir=response_dir,
            )
        elif bench == "humaneval":
            results[bench] = evaluate_humaneval(
                model, tokenizer, config, source=_eval_source(config, bench),
                response_dir=response_dir,
            )
        else:
            logger.warning(f"Unknown code benchmark: {bench}")
    return results


def compute_compacted_prompt_length(S_p_list: List[torch.Tensor],
                                    full_prompt_lengths: List[int]) -> Dict[str, float]:
    """BRIDGE 的 cost-awareness：压缩后 prompt 长度 vs 完整 prompt 长度。"""
    compacted = [int(s.shape[0]) for s in S_p_list]
    full = full_prompt_lengths
    mean_full = sum(full) / max(len(full), 1)
    mean_compact = sum(compacted) / max(len(compacted), 1)
    ratio = mean_compact / mean_full if mean_full > 0 else 0.0
    return {
        "mean_full_prompt_length": mean_full,
        "mean_compacted_prompt_length": mean_compact,
        "prompt_compression_ratio": ratio,
        "prompt_tokens_saved": mean_full - mean_compact,
    }


def compute_performance_per_cost(accuracy: float, wall_clock_seconds: float) -> Dict[str, float]:
    """单位时间的准确率（cost-aware 性能指标）。"""
    return {
        "accuracy": accuracy,
        "wall_clock_seconds": wall_clock_seconds,
        "accuracy_per_second": accuracy / wall_clock_seconds if wall_clock_seconds > 0 else 0.0,
    }
