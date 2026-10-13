"BRIDGE 通用工具：日志、随机种子、设备管理、文件 IO。"

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


logger = logging.getLogger("bridge")


_GPU_FREE_THRESHOLD = 0.1
_GPU_UTIL_THRESHOLD = 10.0
_MAX_SHARED_GPUS = 4


_LOGGER_CACHE: dict = {}


def get_logger(name: str = "bridge") -> logging.Logger:
    "返回带统一格式的 logger，多次同名调用返回同一实例。"
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        log_file = os.environ.get("BRIDGE_LOG_FILE")
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        logger.setLevel(os.environ.get("BRIDGE_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    _LOGGER_CACHE[name] = logger
    return logger


def set_seed(seed: int) -> None:
    "固定 random / numpy / torch 的随机种子。"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 ``worker_init_fn``，保证多进程可复现。"""
    base = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(base)
    random.seed(base)


def clear_gpu_cache() -> None:
    """清空并同步 CUDA 缓存（评分/训练阶段切换时调用）。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def auto_select_gpus(num_gpus: int) -> None:
    "按 ``cfg.num_gpus`` 限定可见 GPU 数量。"
    requested = max(1, min(int(num_gpus), _MAX_SHARED_GPUS))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if not devices:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is set but contains no GPU ids")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:requested])
        get_logger("bridge.utils").info(
            f"Using configured GPUs: {devices[:requested]} (max {_MAX_SHARED_GPUS})"
        )
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free = []
        for line in result.stdout.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                idx, used, total, util = (
                    int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                )
                if used / total < _GPU_FREE_THRESHOLD and util <= _GPU_UTIL_THRESHOLD:
                    free.append(idx)
        if free:
            selected = free[:requested]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)
            logger = get_logger("bridge.utils")
            logger.info(f"Auto-selected free GPUs: {selected}")
        else:
            raise RuntimeError(
                "No idle GPU satisfies the shared-server limits "
                f"(memory used < {_GPU_FREE_THRESHOLD:.0%}, utilization <= {_GPU_UTIL_THRESHOLD:.0f}%). "
                "Wait for capacity or set CUDA_VISIBLE_DEVICES to GPUs you are authorized to use."
            )
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"GPU detection failed safely: {e}") from e


def debug_tensor_range(
    name: str,
    tensor,
    upper_bound,
    lower_bound: int = 0,
    ignore_values=None,
) -> bool:
    "打印 index 张量的 shape/dtype/device/min/max 并检查是否落在 ``[lower_bound, upper_bound)``。"
    if tensor is None:
        logger.debug(f"[range] {name}: None (skip)")
        return True
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    t = tensor.detach().to(torch.long).cpu()
    if t.numel() == 0:
        logger.debug(f"[range] {name}: empty")
        return True
    ignore = set(ignore_values) if ignore_values else set()
    if ignore:
        ignore_t = torch.tensor(sorted(ignore), dtype=torch.long)
        considered = ~torch.isin(t, ignore_t)
    else:
        considered = torch.ones_like(t, dtype=torch.bool)
    if upper_bound is not None:
        oob = considered & ((t < lower_bound) | (t >= upper_bound))
    else:
        oob = considered & (t < lower_bound)
    n_oob = int(oob.sum().item())
    tmin, tmax = int(t.min().item()), int(t.max().item())
    status = "OK" if n_oob == 0 else f"ERROR: {n_oob} 个 index 越界"
    logger.debug(
        f"[range] {name}: shape={list(t.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"min={tmin} max={tmax} limit=[{lower_bound},{upper_bound}) ignore={sorted(ignore)} -> {status}"
    )
    if n_oob:
        logger.warning(f"[range] {name}: 越界值样本(前8)={t[oob][:8].tolist()}")
    return n_oob == 0


def ensure_dir(path: "str | Path") -> Path:
    """确保目录存在，返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: "str | Path") -> None:
    """将对象保存为 JSON 文件（UTF-8，缩进 2）。"""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
