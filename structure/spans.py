"Tokenizer alignment and domain-independent closure-span construction."

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from structure.types import ClosureSpan


_DELIMITER_ONLY = re.compile(r"^[\s`~*_#>|:;,.!?\\/()\[\]{}$]+$")
_LATEX_ENV = re.compile(r"\\(begin|end)\{([^{}]+)\}")
_REASONING_CUE = re.compile(
    r"\b(?:so|therefore|hence|thus|because|since|then|first|next|finally|"
    r"remaining|total|difference|let\s+[A-Za-z]|solve\s+for|add(?:ing)?|"
    r"subtract(?:ing)?|multip(?:ly|lying)|divid(?:e|ing)|equation|"
    r"from\s+both\s+sides|years?\s+from\s+now|convert(?:ion)?)\b",
    re.IGNORECASE,
)
_REASONING_SIGNAL = re.compile(
    r"(?:\d|\\(?:frac|sqrt|boxed)\b|[$%]|[A-Za-z]\s*[=<>+\-*/×÷^]|"
    r"[=<>+\-*/×÷^]\s*[A-Za-z0-9]|\blet\s+[A-Za-z]\b|"
    r"\bsolve\s+for\s+[A-Za-z]\b)",
    re.IGNORECASE,
)

_UNIT_WORDS = (
    "second", "minute", "hour", "day", "week", "month", "year",
    "inch", "foot", "feet", "yard", "mile", "meter", "kilometer",
    "gram", "kilogram", "pound", "ounce", "cup", "pint", "quart",
    "gallon", "liter", "dollar", "cent", "calorie", "egg", "book",
    "item", "serving", "person", "people", "dozen", "glass",
)
_UNIT_PATTERN = (
    r"(?:" + "|".join(
        re.escape(word) + (r"s?" if not word.endswith("s") else "")
        for word in _UNIT_WORDS
    ) + r")"
)
_NUMBER_PATTERN = r"(?:\d+(?:,\d{3})*(?:\.\d+)?|\d+\s*/\s*\d+)"
_QUANTITY_PATTERNS = (
    re.compile(
        rf"\$\s*{_NUMBER_PATTERN}"
        rf"(?![\d.A-Za-z_^!{{}}]|\s*[\\()+\-*/=])"
        rf"(?:\s*(?:each|per\s+[A-Za-z]+))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_NUMBER_PATTERN}\s*(?:%|percent\b)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_NUMBER_PATTERN}\s+{_UNIT_PATTERN}"
        rf"(?:\s+(?:per|each|every)\s+(?:{_UNIT_PATTERN}|[A-Za-z]+))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_NUMBER_PATTERN}\s+[A-Za-z]+"
        rf"\s+(?:per|each|every)\s+(?:{_UNIT_PATTERN}|[A-Za-z]+)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_NUMBER_PATTERN}\s+(?:times\s+(?:the\s+)?(?:regular\s+)?"
        r"(?:hourly\s+)?rate|more\s+than|less\s+than|of\s+"
        rf"{_NUMBER_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:half|one\s+third|one\s+quarter|twice)\b"
        r"(?:\s+(?:as\s+many|as\s+much|of)\b)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bevery\s+(?:second|third|fourth)\s+[A-Za-z]+\b",
        re.IGNORECASE,
    ),
)
_QUANTITY_CONNECTOR = re.compile(
    r"(?:\s|[/=+\-*×÷(),])+|"
    r"(?:\s+(?:of|to|than|as|per|each|every|the|regular|hourly|rate)\s+)+",
    re.IGNORECASE,
)


class AlignmentError(ValueError):
    """Raised when offsets cannot be proven to match the scored token indices."""


@dataclass
class AlignedTokens:
    token_ids: List[int]
    offsets: List[Tuple[int, int]]
    token_texts: List[str]


def tokenize_with_verified_offsets(
    text: str,
    tokenizer: Any,
    max_length: int,
    expected_ids: Optional[Sequence[int]] = None,
    expected_length: Optional[int] = None,
    special_token_mode: str = "legacy",
) -> AlignedTokens:
    "Tokenize exactly like ``BridgeDataset`` and verify index identity."
    encode_kwargs: Dict[str, Any] = {}
    effective_max = max_length
    if special_token_mode == "single_bos_eos":
        encode_kwargs["add_special_tokens"] = False
        effective_max = max(max_length - 1, 1)
    try:
        encoded = tokenizer(
            text,
            max_length=effective_max,
            truncation=True,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            **encode_kwargs,
        )
    except Exception as error:
        raise AlignmentError(f"offset_mapping unavailable: {error}") from error
    if "offset_mapping" not in encoded:
        raise AlignmentError("tokenizer did not return offset_mapping")

    ids_value = encoded["input_ids"]
    offsets_value = encoded["offset_mapping"]
    ids = ids_value[0].tolist() if hasattr(ids_value, "ndim") and ids_value.ndim == 2 else list(ids_value)
    raw_offsets = (
        offsets_value[0].tolist()
        if hasattr(offsets_value, "ndim") and offsets_value.ndim == 3
        else list(offsets_value)
    )
    offsets = [(int(item[0]), int(item[1])) for item in raw_offsets]
    if special_token_mode == "single_bos_eos":
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            ids = ids + [int(eos_id)]
            offsets = offsets + [(len(text), len(text))]
    if len(ids) != len(offsets):
        raise AlignmentError(f"token/offset length mismatch: {len(ids)} != {len(offsets)}")
    if expected_length is not None and len(ids) != int(expected_length):
        raise AlignmentError(
            f"token length mismatch: offsets={len(ids)}, scored={int(expected_length)}"
        )
    if expected_ids is not None and ids != [int(value) for value in expected_ids]:
        mismatch = next(
            (i for i, (left, right) in enumerate(zip(ids, expected_ids)) if left != int(right)),
            min(len(ids), len(expected_ids)),
        )
        raise AlignmentError(f"token ID mismatch at response index {mismatch}")

    token_texts: List[str] = []
    for token_id, (start, end) in zip(ids, offsets):
        if end > start:
            token_texts.append(text[start:end])
        elif hasattr(tokenizer, "convert_ids_to_tokens"):
            token_texts.append(str(tokenizer.convert_ids_to_tokens(int(token_id))))
        else:
            token_texts.append("")
    return AlignedTokens(ids, offsets, token_texts)


def is_core_token(
    token_text: str,
    offset: Tuple[int, int],
    exclude_delimiters: bool = True,
) -> bool:
    """Return whether a token carries content under one unified character rule."""
    if offset[1] <= offset[0] or not token_text or token_text.isspace():
        return False
    if exclude_delimiters and _DELIMITER_ONLY.fullmatch(token_text):
        return False
    return any(char.isalnum() or char == "_" for char in token_text) or any(
        char in "+-=<>%&|^@" for char in token_text
    )


def _is_escaped(text: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def _scan_character_boundaries(text: str) -> Tuple[List[Tuple[int, str, float]], bool]:
    """Find safe boundary offsets while tracking paired scopes in one pass."""
    boundaries: List[Tuple[int, str, float]] = []
    bracket_stack: List[str] = []
    latex_stack: List[str] = []
    quote: Optional[str] = None
    fence: Optional[str] = None
    math_mode: Optional[str] = None
    pairs = {")": "(", "]": "[", "}": "{"}
    i = 0
    line_start = True
    while i < len(text):
        if line_start:
            stripped = i
            while stripped < len(text) and text[stripped] in " \t":
                stripped += 1
            marker = text[stripped:stripped + 3]
            if marker in {"```", "~~~"}:
                line_end = text.find("\n", stripped)
                line_end = len(text) if line_end < 0 else line_end + 1
                if fence is None:
                    if not bracket_stack and quote is None and math_mode is None and stripped > 0:
                        boundaries.append((stripped, "code_fence", 1.0))
                    fence = marker
                elif fence == marker:
                    fence = None
                    boundaries.append((line_end, "code_fence", 1.0))
                i = line_end
                line_start = True
                continue
        if fence is not None:
            line_start = text[i] == "\n"
            i += 1
            continue

        env_match = _LATEX_ENV.match(text, i)
        if env_match and quote is None:
            action, env = env_match.groups()
            if action == "begin":
                latex_stack.append(env)
            elif latex_stack and latex_stack[-1] == env:
                latex_stack.pop()
                if not latex_stack and not bracket_stack and math_mode is None:
                    boundaries.append((env_match.end(), "latex_environment", 1.0))
            i = env_match.end()
            line_start = False
            continue

        char = text[i]
        escaped = _is_escaped(text, i)
        if quote is not None:
            if char == quote and not escaped:
                quote = None
            line_start = char == "\n"
            i += 1
            continue
        if char in {'"', "'"} and not escaped:
            apostrophe = (
                char == "'" and i > 0 and i + 1 < len(text)
                and text[i - 1].isalnum() and text[i + 1].isalnum()
            )
            if not apostrophe:
                quote = char
                i += 1
                line_start = False
                continue
        if char == "$" and not escaped:
            marker = "$$" if text.startswith("$$", i) else "$"
            if math_mode is None:
                math_mode = marker
            elif math_mode == marker:
                math_mode = None
                end = i + len(marker)
                if not bracket_stack and not latex_stack:
                    boundaries.append((end, "formula", 1.0))
            i += len(marker)
            line_start = False
            continue
        if math_mode is not None or latex_stack:
            line_start = char == "\n"
            i += 1
            continue

        if char in "([{":
            bracket_stack.append(char)
        elif char in pairs:
            if bracket_stack and bracket_stack[-1] == pairs[char]:
                bracket_stack.pop()
                if not bracket_stack:
                    boundaries.append((i + 1, "paired_close", 0.9))
        elif not bracket_stack:
            if char == "\n":
                boundary_type = "paragraph" if i + 1 < len(text) and text[i + 1] == "\n" else "newline"
                boundaries.append((i + 1, boundary_type, 0.9))
            elif char in ".!?。！？；;":
                boundaries.append((i + 1, "sentence", 0.9))
        line_start = char == "\n"
        i += 1

    unclosed = bool(bracket_stack or latex_stack or quote or fence or math_mode)
    unique: Dict[int, Tuple[str, float]] = {}
    for position, boundary_type, confidence in boundaries:
        if 0 < position <= len(text):
            previous = unique.get(position)
            if previous is None or confidence > previous[1]:
                unique[position] = (boundary_type, confidence)
    return [(position, *unique[position]) for position in sorted(unique)], unclosed


def _char_range(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> Tuple[Optional[int], Optional[int]]:
    nonempty = [offsets[index] for index in range(start, end) if offsets[index][1] > offsets[index][0]]
    if not nonempty:
        return None, None
    return min(item[0] for item in nonempty), max(item[1] for item in nonempty)


def token_indices_for_char_range(
    aligned: AlignedTokens,
    char_start: int,
    char_end: int,
) -> List[int]:
    """Map a text interval to verified semantic token positions."""
    return [
        index
        for index, (start, end) in enumerate(aligned.offsets)
        if end > start and start < char_end and end > char_start
    ]


def _sentence_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"[.!?。！？；;\n]+", text):
        end = match.end()
        if end > cursor and text[cursor:end].strip():
            ranges.append((cursor, end))
        cursor = end
    if cursor < len(text) and text[cursor:].strip():
        ranges.append((cursor, len(text)))
    return ranges


def _reasoning_atomic_ranges(
    sentence: str,
    absolute_start: int,
) -> List[Tuple[int, int]]:
    "Return the cue/operator evidence core of a reasoning sentence."
    cues = list(_REASONING_CUE.finditer(sentence))
    selected: List[Tuple[int, int]] = [
        (match.start(), match.end()) for match in cues
    ]
    evidence_pattern = re.compile(
        r"\\(?:frac|sqrt|boxed|sum|int)\b|"
        r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*\d+)?%?|"
        r"[=<>+\-*/×÷^]"
    )
    evidence = list(evidence_pattern.finditer(sentence))
    for cue in cues:
        if evidence:
            nearest = min(
                evidence,
                key=lambda match: (
                    abs(match.start() - cue.start()),
                    match.start(),
                ),
            )
            selected.append((nearest.start(), nearest.end()))
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(set(selected)):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [
        (absolute_start + start, absolute_start + end)
        for start, end in merged
    ]


def _merge_quantity_ranges(
    text: str,
    ranges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(set(ranges)):
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        gap = text[previous_end:start]
        if (
            start <= previous_end
            or (
                len(gap) <= 48
                and bool(_QUANTITY_CONNECTOR.fullmatch(gap))
            )
        ):
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def build_typed_spans(
    text: str,
    aligned: AlignedTokens,
    include_reasoning_sentence: bool = False,
    include_quantity_unit: bool = False,
    exclude_delimiter_tokens: bool = True,
) -> Tuple[List[ClosureSpan], Dict[str, int]]:
    "Build optional overlapping reasoning and quantity dependency spans."
    candidates: List[Tuple[int, int, str, Dict[str, Any]]] = []
    if include_reasoning_sentence:
        for start, end in _sentence_ranges(text):
            sentence = text[start:end]
            if (
                _REASONING_CUE.search(sentence)
                and _REASONING_SIGNAL.search(sentence)
            ):
                atomic_ranges = _reasoning_atomic_ranges(
                    sentence, start,
                )
                candidates.append((
                    start,
                    end,
                    "reasoning_sentence",
                    {
                        "matched_text": sentence,
                        "atomic_char_ranges": atomic_ranges,
                    },
                ))

    if include_quantity_unit:
        quantity_ranges: List[Tuple[int, int]] = []
        for pattern in _QUANTITY_PATTERNS:
            quantity_ranges.extend(
                (match.start(), match.end())
                for match in pattern.finditer(text)
            )
        for start, end in _merge_quantity_ranges(text, quantity_ranges):
            candidates.append((
                start,
                end,
                "quantity_unit_relation",
                {"matched_text": text[start:end]},
            ))

    spans: List[ClosureSpan] = []
    for char_start, char_end, span_type, metadata in sorted(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    ):
        atomic_ranges = metadata.get(
            "atomic_char_ranges",
            [(char_start, char_end)],
        )
        indices = sorted({
            index
            for range_start, range_end in atomic_ranges
            for index in token_indices_for_char_range(
                aligned, range_start, range_end,
            )
        })
        if not indices:
            continue
        core = [
            index for index in indices
            if is_core_token(
                aligned.token_texts[index],
                aligned.offsets[index],
                exclude_delimiter_tokens,
            )
        ]
        atomic_indices = (
            core if span_type == "reasoning_sentence" and core
            else indices
        )
        spans.append(ClosureSpan(
            span_id=len(spans),
            token_start=min(atomic_indices),
            token_end=max(atomic_indices) + 1,
            token_indices=atomic_indices,
            core_token_indices=core or list(indices),
            char_start=char_start,
            char_end=char_end,
            span_type=span_type,
            confidence=1.0,
            metadata=metadata,
        ))
    return spans, {
        "reasoning_sentence_count": sum(
            span.span_type == "reasoning_sentence" for span in spans
        ),
        "quantity_unit_relation_count": sum(
            span.span_type == "quantity_unit_relation" for span in spans
        ),
    }


def _split_long(intervals: Iterable[Tuple[int, int, str, float]], maximum: int) -> List[Tuple[int, int, str, float]]:
    result: List[Tuple[int, int, str, float]] = []
    for start, end, span_type, confidence in intervals:
        cursor = start
        while end - cursor > maximum:
            result.append((cursor, cursor + maximum, "fixed_window_fallback", min(confidence, 0.5)))
            cursor += maximum
        if cursor < end:
            result.append((cursor, end, span_type, confidence))
    return result


def _merge_short(
    intervals: List[Tuple[int, int, str, float]], minimum: int, maximum: int,
) -> List[Tuple[int, int, str, float]]:
    if len(intervals) < 2:
        return intervals
    merged: List[Tuple[int, int, str, float]] = []
    index = 0
    while index < len(intervals):
        current = intervals[index]
        if current[1] - current[0] < minimum and index + 1 < len(intervals):
            following = intervals[index + 1]
            if following[1] - current[0] <= maximum:
                merged.append((current[0], following[1], following[2], min(current[3], following[3])))
                index += 2
                continue
        if current[1] - current[0] < minimum and merged:
            previous = merged[-1]
            if current[1] - previous[0] <= maximum:
                merged[-1] = (previous[0], current[1], current[2], min(previous[3], current[3]))
                index += 1
                continue
        merged.append(current)
        index += 1
    return merged


def build_closure_spans(
    text: str,
    aligned: AlignedTokens,
    min_span_tokens: int = 4,
    max_span_tokens: int = 96,
    merge_short_spans: bool = True,
    exclude_delimiter_tokens: bool = True,
) -> Tuple[List[ClosureSpan], Dict[str, Any]]:
    """Partition all token positions into closure spans without overlap."""
    started = time.perf_counter()
    token_count = len(aligned.token_ids)
    if token_count == 0:
        return [], {
            "fallback_reason": "empty_response", "uncovered_tokens": 0,
            "overlap_tokens": 0, "construction_wall_time": time.perf_counter() - started,
        }
    boundaries, unclosed = _scan_character_boundaries(text)
    intervals: List[Tuple[int, int, str, float]] = []
    cursor = 0
    token_cursor = 0
    for char_end, span_type, confidence in boundaries:
        while token_cursor < token_count:
            start, end = aligned.offsets[token_cursor]
            if end == 0 or end <= char_end:
                token_cursor += 1
            else:
                break
        if token_cursor > cursor:
            intervals.append((cursor, token_cursor, span_type, confidence))
            cursor = token_cursor
    if cursor < token_count:
        intervals.append((cursor, token_count, "tail", 0.6 if unclosed else 0.8))
    if not intervals:
        intervals = [(0, token_count, "unaligned_fallback", 0.2)]

    intervals = _split_long(intervals, int(max_span_tokens))
    if merge_short_spans:
        intervals = _merge_short(intervals, int(min_span_tokens), int(max_span_tokens))

    spans: List[ClosureSpan] = []
    coverage = [0] * token_count
    for span_id, (start, end, span_type, confidence) in enumerate(intervals):
        if not 0 <= start < end <= token_count:
            raise RuntimeError(f"invalid span interval [{start}, {end})/{token_count}")
        indices = list(range(start, end))
        for token_index in indices:
            coverage[token_index] += 1
        core = [
            token_index for token_index in indices
            if is_core_token(
                aligned.token_texts[token_index], aligned.offsets[token_index], exclude_delimiter_tokens,
            )
        ]
        if not core:
            core = [
                token_index for token_index in indices
                if aligned.offsets[token_index][1] > aligned.offsets[token_index][0]
            ]
        char_start, char_end = _char_range(aligned.offsets, start, end)
        spans.append(ClosureSpan(
            span_id=span_id,
            token_start=start,
            token_end=end,
            token_indices=indices,
            core_token_indices=core,
            char_start=char_start,
            char_end=char_end,
            span_type=span_type,
            confidence=confidence,
            metadata={"unclosed_scope": unclosed},
        ))
    uncovered = sum(value == 0 for value in coverage)
    overlap = sum(value > 1 for value in coverage)
    if uncovered or overlap:
        raise RuntimeError(f"span partition invalid: uncovered={uncovered}, overlap={overlap}")
    lengths = [span.cost for span in spans]
    return spans, {
        "span_count": len(spans),
        "average_span_length": sum(lengths) / len(lengths),
        "max_span_length": max(lengths),
        "fallback_reason": "unclosed_scope" if unclosed else None,
        "uncovered_tokens": uncovered,
        "overlap_tokens": overlap,
        "construction_wall_time": time.perf_counter() - started,
    }
