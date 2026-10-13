# BRIDGE — Anonymous Code

BRIDGE performs validation-directed token scoring, fixed-interaction alternating
selection and Closure, followed by instruction tuning. Prompts are physically
compacted in original order. Responses remain complete in the forward pass;
only selected response positions receive direct supervision. Final SFT starts
from the pretrained checkpoint, not the warmup checkpoint.

## Install

Use Python 3.10+ and CUDA/PyTorch versions compatible with the available GPUs.

```bash
python -m pip install -r requirements.txt
```

Install requirements-eval.txt for IFEval/IFBench. Their official verifier
libraries and required licenses are under external/.

## Run

Supply original pretrained weights, tokenizer and datasets; they are not bundled.
Run from this directory:

```bash
export BRIDGE_MODEL_PATH=/path/to/pretrained/Llama-3.2-3B
export BRIDGE_DATA_ROOT=/path/to/datasets
python scripts/reproduce.py \
  --config configs/artifact/llama3b_metamath.yaml \
  --gpus 0 --output-dir runs/llama3b_seed42 --dry-run
```


## Evaluate

Only two script entry points are supplied:

- scripts/reproduce.py: complete pipeline; individual stages via
  `stage warmup|direction|scoring|select|train|eval`, or `l2-direction`.
- scripts/evaluate.py: `benchmarks`, `instructions` (IFEval/IFBench), or `summary`.

