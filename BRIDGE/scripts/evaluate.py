"""Evaluate benchmarks, instruction following, or aggregate existing results."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from configs import load_config
from evaluation import (
    evaluate_alignment, evaluate_code, evaluate_reasoning,
    generate_answers, prepare_model_for_evaluation,
)
from models.utils import BridgeModel
from utils import get_logger, save_json, auto_select_gpus

logger = get_logger("bridge.eval")


def main_benchmarks():
    p = argparse.ArgumentParser(description="Evaluate a BRIDGE checkpoint")
    p.add_argument("--config", type=str, default="configs/artifact/llama3b_metamath.yaml")
    p.add_argument("--checkpoint", type=str, required=True, help="模型/checkpoint 目录")
    p.add_argument("--model", type=str, default=None, help="基座模型路径（覆盖 config）")
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help="single keeps generation on one GPU; auto enables model-parallel dispatch",
    )
    p.add_argument("--num-eval-samples", type=int, default=None)
    p.add_argument("--num-code-eval-samples", type=int, default=None)
    p.add_argument(
        "--alignment-benchmarks",
        nargs="+",
        default=None,
        help="Optional alignment benchmark subset, for example arc_challenge",
    )
    p.add_argument(
        "--reasoning-benchmarks",
        nargs="+",
        default=None,
        help="Optional reasoning benchmark subset, for example gsm8k math",
    )
    p.add_argument(
        "--code-benchmarks",
        nargs="+",
        default=None,
        help="Optional code benchmark subset, for example humaneval",
    )
    p.add_argument(
        "--multiple-choice-mode",
        choices=("generation", "likelihood"),
        default=None,
        help="Use paper-compatible candidate likelihood for multiple-choice tasks",
    )
    p.add_argument(
        "--likelihood-batch-size",
        type=int,
        default=None,
        help="Candidate sequences per likelihood forward pass",
    )
    p.add_argument(
        "--sections",
        nargs="+",
        choices=("alignment", "reasoning", "code"),
        default=("alignment", "reasoning", "code"),
        help="Evaluation sections to run; defaults to all",
    )
    args = p.parse_args()

    overrides = {}
    if args.model:
        overrides["model_name_or_path"] = args.model
    if args.num_eval_samples is not None:
        if args.num_eval_samples <= 0:
            p.error("--num-eval-samples must be positive")
        overrides["num_eval_samples"] = args.num_eval_samples
    if args.num_code_eval_samples is not None:
        if args.num_code_eval_samples <= 0:
            p.error("--num-code-eval-samples must be positive")
        overrides["num_code_eval_samples"] = args.num_code_eval_samples
    if args.alignment_benchmarks:
        overrides["alignment_benchmarks"] = args.alignment_benchmarks
    if args.reasoning_benchmarks:
        overrides["reasoning_benchmarks"] = args.reasoning_benchmarks
    if args.code_benchmarks:
        overrides["code_benchmarks"] = args.code_benchmarks
    if args.multiple_choice_mode:
        overrides["multiple_choice_mode"] = args.multiple_choice_mode
    if args.likelihood_batch_size is not None:
        if args.likelihood_batch_size <= 0:
            p.error("--likelihood-batch-size must be positive")
        overrides["likelihood_batch_size"] = args.likelihood_batch_size
    cfg = load_config(args.config, overrides=overrides)
    out = Path(args.output) if args.output else Path(args.checkpoint) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    response_root = out.parent / "eval_responses"

    auto_select_gpus(cfg.num_gpus)
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    logger.info(f"Loading full checkpoint '{args.checkpoint}'")
    device_map = {"": 0} if args.device_map == "single" else "auto"
    model = BridgeModel(
        args.checkpoint,
        torch_dtype=dtype,
        device_map=device_map,
        inference_only=True,
    )
    prepare_model_for_evaluation(model)

    checkpoint_path = Path(args.checkpoint)
    tokenizer_source = (
        checkpoint_path
        if (checkpoint_path / "tokenizer_config.json").exists()
        else cfg.model_name_or_path
    )
    logger.info(f"Loading tokenizer from '{tokenizer_source}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    results = {
        "checkpoint": args.checkpoint,
        "config": cfg.to_dict(),
        "evaluation_responses": str(response_root),
        "evaluation_status": "running",
    }
    save_json(results, out)
    try:
        if "alignment" in args.sections:
            results["alignment"] = evaluate_alignment(
                model, tokenizer, cfg, response_dir=response_root / "alignment",
            )
        if "reasoning" in args.sections:
            results["reasoning"] = evaluate_reasoning(
                model, tokenizer, cfg, response_dir=response_root / "reasoning",
            )
        if "code" in args.sections:
            results["code"] = evaluate_code(
                model, tokenizer, cfg, response_dir=response_root / "code",
            )
    except Exception as error:
        results["evaluation_status"] = "failed"
        results["evaluation_error"] = repr(error)
        save_json(results, out)
        raise
    results["evaluation_status"] = "complete"
    save_json(results, out)
    logger.info(f"Evaluation complete: metrics={out}, responses={response_root}")


def parse_instruction_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--benchmark", choices=("ifeval", "ifbench"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--prompt-template",
        choices=("raw", "alpaca"),
        default="raw",
        help="Generation prompt format; scoring always uses the original prompt.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_completed(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        row["prompt"]: row["response"]
        for row in read_jsonl(path)
        if "prompt" in row and "response" in row
    }


def append_responses(path: Path, prompts: list[str], responses: list[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for prompt, response in zip(prompts, responses):
            handle.write(
                json.dumps(
                    {"prompt": prompt, "response": response},
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.flush()


def format_generation_prompt(prompt: str, template: str) -> str:
    if template == "raw":
        return prompt
    if template == "alpaca":
        return (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt.strip()}\n\n"
            "### Response:"
        )
    raise ValueError(f"Unsupported prompt template: {template!r}")


def score_outputs(
    input_path: Path,
    response_path: Path,
    output_dir: Path,
    benchmark: str,
) -> dict[str, Any]:
    if benchmark == "ifeval":
        google_root = PROJECT_ROOT / "external" / "google-research"
        sys.path.insert(0, str(google_root))
        from instruction_following_eval import evaluation_lib  # type: ignore
    else:
        ifbench_root = PROJECT_ROOT / "external" / "IFBench"
        sys.path.insert(0, str(ifbench_root))
        import evaluation_lib  # type: ignore

    inputs = evaluation_lib.read_prompt_list(str(input_path))
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
        str(response_path)
    )
    metrics: dict[str, Any] = {"num_prompts": len(inputs)}
    for mode, scorer in (
        ("strict", evaluation_lib.test_instruction_following_strict),
        ("loose", evaluation_lib.test_instruction_following_loose),
    ):
        outputs = [scorer(item, prompt_to_response) for item in inputs]
        instruction_flags = [
            flag for output in outputs for flag in output.follow_instruction_list
        ]
        metrics[mode] = {
            "prompt_level_accuracy": (
                sum(output.follow_all_instructions for output in outputs) / len(outputs)
            ),
            "instruction_level_accuracy": (
                sum(instruction_flags) / len(instruction_flags)
            ),
        }
        official_output = output_dir / f"official_{mode}.jsonl"
        evaluation_lib.write_outputs(str(official_output), outputs)
    return metrics


def main_instructions() -> None:
    args = parse_instruction_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "responses.jsonl"
    result_path = output_dir / "results.json"

    examples = read_jsonl(input_path)
    completed = read_completed(response_path)
    pending = [row["prompt"] for row in examples if row["prompt"] not in completed]

    state = {
        "benchmark": args.benchmark,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(input_path),
        "responses": str(response_path),
        "num_prompts": len(examples),
        "num_completed": len(completed),
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "evaluation_status": "generating",
        "prompt_template": args.prompt_template,
    }
    save_json(state, result_path)

    if pending:
        dtype = torch.bfloat16
        model = BridgeModel(
            args.checkpoint,
            torch_dtype=dtype,
            device_map={"": 0},
            inference_only=True,
        )
        prepare_model_for_evaluation(model)
        checkpoint = Path(args.checkpoint)
        tokenizer_source = (
            checkpoint
            if (checkpoint / "tokenizer_config.json").exists()
            else Path(args.base_model)
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        generation_config = getattr(model.model, "generation_config", None)
        state["generation_eos_token_id"] = getattr(
            generation_config, "eos_token_id", tokenizer.eos_token_id
        )
        state["generation_pad_token_id"] = tokenizer.pad_token_id
        save_json(state, result_path)

        tokenizer.padding_side = "left"

        progress = tqdm(total=len(pending), desc=f"Generate {args.benchmark}")
        for start in range(0, len(pending), args.batch_size):
            prompts = pending[start : start + args.batch_size]
            generation_prompts = [
                format_generation_prompt(prompt, args.prompt_template)
                for prompt in prompts
            ]
            try:
                responses = generate_answers(
                    model,
                    tokenizer,
                    generation_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
            except torch.OutOfMemoryError:
                if len(prompts) == 1:
                    raise
                torch.cuda.empty_cache()
                responses = []
                for prompt in prompts:
                    responses.extend(
                        generate_answers(
                            model,
                            tokenizer,
                            [
                                format_generation_prompt(
                                    prompt, args.prompt_template
                                )
                            ],
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.0,
                        )
                    )
            append_responses(response_path, prompts, responses)
            progress.update(len(prompts))
            state["num_completed"] = len(completed) + start + len(prompts)
            save_json(state, result_path)
        progress.close()
        del model
        torch.cuda.empty_cache()

    state["evaluation_status"] = "scoring"
    state["num_completed"] = len(examples)
    save_json(state, result_path)
    metrics = score_outputs(
        input_path,
        response_path,
        output_dir,
        args.benchmark,
    )
    state["metrics"] = metrics
    state["evaluation_status"] = "complete"
    save_json(state, result_path)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def checked_fraction(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name}: expected a finite fraction in [0,1]')
    return value


def human_score(result):
    human = result['code']['humaneval']
    if 'pass@1' in human:
        return human['pass@1']
    if 'pass_at_1' in human:
        return human['pass_at_1']
    raise ValueError('HumanEval pass@1 not found; no silent metric fallback')


def summarize(result, metric='raw', target='math', domain=None, ifeval_result=None):
    if metric not in ('raw', 'acc_norm'):
        raise ValueError('metric must be raw or acc_norm')
    if domain not in (None, 'metamath', 'magicoder', 'alpaca'):
        raise ValueError('Unsupported fine-tuning domain')
    scores = {}
    warnings = []
    for task in ('arc_challenge', 'mmlu', 'hellaswag'):
        row = result['alignment'][task]
        key = 'accuracy_raw' if metric == 'raw' or task == 'mmlu' else 'accuracy_normalized'
        if key not in row:
            if task == 'mmlu' and row.get('metric') == 'acc':
                key = 'accuracy'
            else:
                raise ValueError(f'{task}: explicit {key} not available; no silent metric fallback')
        scores[task] = row[key]
    if domain in ('magicoder', 'alpaca'):
        scores['gsm8k'] = result['reasoning']['gsm8k']['accuracy']
        retention_tasks = ('arc_challenge', 'mmlu', 'hellaswag', 'gsm8k')
        if domain == 'magicoder':
            target = 'humaneval'
            scores['target'] = human_score(result)
        else:
            target = 'ifeval_strict_mean'
            if ifeval_result is None or ifeval_result.get('benchmark') != 'ifeval':
                raise ValueError('Alpaca summary requires the explicit IFEval results.json')
            if ifeval_result.get('evaluation_status') != 'complete':
                raise ValueError('IFEval evaluation is not complete')
            strict = ifeval_result['metrics']['strict']
            scores['ifeval_strict_prompt'] = checked_fraction('ifeval_strict_prompt', strict['prompt_level_accuracy'])
            scores['ifeval_strict_instruction'] = checked_fraction('ifeval_strict_instruction', strict['instruction_level_accuracy'])
            scores['target'] = (scores['ifeval_strict_prompt'] + scores['ifeval_strict_instruction']) / 2
            auxiliary_checkpoint = result.get('checkpoint') or result.get('provenance', {}).get('checkpoint')
            target_checkpoint = ifeval_result.get('checkpoint')
            if auxiliary_checkpoint and target_checkpoint:
                if Path(auxiliary_checkpoint).resolve() != Path(target_checkpoint).resolve():
                    raise ValueError('IFEval and general evaluation checkpoint mismatch')
            else:
                warnings.append('Separate evaluation checkpoint binding is not verified; supply original provenance')
    else:
        scores['humaneval'] = human_score(result)
        retention_tasks = ('arc_challenge', 'mmlu', 'hellaswag', 'humaneval')
        if domain == 'metamath':
            target = 'math'
        if target == 'math':
            scores['target'] = result['math']['official_equivalence_accuracy']
        elif target == 'gsm8k':
            scores['target'] = result['reasoning']['gsm8k']['accuracy']
        else:
            raise ValueError('Legacy target must be math or gsm8k')
    for name, value in scores.items():
        checked_fraction(name, value)
    general = sum(scores[k] for k in retention_tasks) / 4
    return dict(multiple_choice_metric=metric, target=target, domain=domain, scores=scores,
                retention_benchmarks=list(retention_tasks),
                general_avg=general, overall_avg=(general+scores['target'])/2,
                units='fraction_0_to_1', number_of_training_seeds=1, warnings=warnings)


def main_summary():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--multiple-choice-metric', required=True, choices=['raw', 'acc_norm'])
    parser.add_argument('--target', choices=['math', 'gsm8k'], help='Legacy target override; use --domain for main tables')
    parser.add_argument('--domain', choices=['metamath', 'magicoder', 'alpaca'])
    parser.add_argument('--ifeval-input', help='Separate official IFEval results.json for Alpaca')
    args = parser.parse_args()
    if args.domain and args.target:
        parser.error('--domain and legacy --target are mutually exclusive')
    if bool(args.ifeval_input) != (args.domain == 'alpaca'):
        parser.error('--ifeval-input is required only for --domain alpaca')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    ifeval = json.loads(Path(args.ifeval_input).read_text()) if args.ifeval_input else None
    data = summarize(json.loads(Path(args.input).read_text()), args.multiple_choice_metric,
                     args.target or 'math', args.domain, ifeval)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


def main():
    commands = {
        "benchmarks": main_benchmarks,
        "instructions": main_instructions,
        "summary": main_summary,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=tuple(commands))
        parser.parse_args()
        return
    command = sys.argv.pop(1)
    sys.argv[0] += " " + command
    commands[command]()


if __name__ == "__main__":
    main()
