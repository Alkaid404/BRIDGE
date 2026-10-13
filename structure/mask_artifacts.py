"File and tokenizer fingerprints."

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer: Any, max_sequence_length: int) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(len(tokenizer)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "max_sequence_length": int(max_sequence_length),
        "special_token_ids": {
            name: getattr(tokenizer, name, None)
            for name in (
                "bos_token_id", "eos_token_id", "pad_token_id",
                "unk_token_id", "sep_token_id",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
