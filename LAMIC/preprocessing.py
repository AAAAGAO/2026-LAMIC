from __future__ import annotations

import re
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .data import ApiSample


IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
NON_WORD_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
SPLIT_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|(?:\n\s*\n)+|^#+\s+|```", re.MULTILINE)


def split_api_tokens(api: str) -> list[str]:
    tokens: list[str] = []
    normalized = api.replace("::", ".").replace("()", " ")
    for piece in re.split(r"[._()\s]+", normalized):
        if not piece:
            continue
        sub_tokens = [part for part in CAMEL_BOUNDARY.sub(" ", piece).split() if part]
        tokens.append(piece)
        tokens.extend(token.lower() for token in sub_tokens)
    return tokens


def extract_identifiers(fragment: str) -> list[str]:
    return IDENTIFIER_PATTERN.findall(fragment)


def strip_code_for_text(fragment: str) -> str:
    without_fence = CODE_FENCE_PATTERN.sub(" ", fragment)
    without_inline = INLINE_CODE_PATTERN.sub(" ", without_fence)
    return without_inline


def tokenize_text(fragment: str) -> list[str]:
    cleaned = NON_WORD_PATTERN.sub(" ", strip_code_for_text(fragment).lower())
    return [token for token in cleaned.split() if token and token not in ENGLISH_STOP_WORDS]


def build_lexical_document(sample: ApiSample) -> list[str]:
    api_tokens = split_api_tokens(sample.api)
    identifiers = extract_identifiers(sample.fragment)
    text_tokens = tokenize_text(sample.fragment)
    doc: list[str] = []
    doc.extend([sample.api] * 3)
    doc.extend(api_tokens * 2)
    doc.extend(identifiers * 2)
    doc.extend(text_tokens)
    return doc


def normalize_minmax(scores: Iterable[float]) -> list[float]:
    values = list(float(score) for score in scores)
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def semantic_input(api: str, fragment: str) -> str:
    return f"[API] {api}\n[SEP]\n[FRAGMENT] {fragment}"


def split_fragment_boundaries(fragment: str) -> list[str]:
    segments = [segment.strip() for segment in SPLIT_SENTENCE_PATTERN.split(fragment) if segment.strip()]
    return segments or [fragment.strip()]


def build_semantic_chunks(
    api: str,
    fragment: str,
    tokenizer,
    chunk_size: int,
    stride: int,
    semantic_max_length: int,
) -> list[str]:
    prefix = f"[API] {api}\n[SEP]\n"
    segments = split_fragment_boundaries(fragment)
    chunks: list[str] = []
    current = ""
    max_fragment_tokens = semantic_max_length - len(tokenizer.tokenize(prefix)) - 4

    for segment in segments:
        candidate = f"{current}\n{segment}".strip() if current else segment
        if len(tokenizer.tokenize(candidate)) <= max_fragment_tokens:
            current = candidate
            continue
        if current:
            chunks.append(prefix + current)
        current = segment

    if current:
        chunks.append(prefix + current)

    if not chunks:
        return [prefix]

    final_chunks: list[str] = []
    for chunk in chunks:
        fragment_text = chunk[len(prefix) :] if chunk.startswith(prefix) else chunk
        fragment_tokens = tokenizer.tokenize(fragment_text)
        if len(fragment_tokens) <= max_fragment_tokens:
            final_chunks.append(chunk)
            continue
        step = max(1, chunk_size - stride)
        start = 0
        while start < len(fragment_tokens):
            piece_tokens = fragment_tokens[start : start + chunk_size]
            piece = tokenizer.convert_tokens_to_string(piece_tokens)
            final_chunks.append(prefix + piece)
            if start + chunk_size >= len(fragment_tokens):
                break
            start += step
    return final_chunks
