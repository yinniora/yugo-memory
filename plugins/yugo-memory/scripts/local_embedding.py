#!/usr/bin/env python3
"""Deterministic dependency-free local embeddings for recall routing.

These vectors are a signed feature-hashing projection over multilingual terms,
identifier pieces, and bounded character n-grams. They are not a downloaded
neural model. Their purpose is to add typo/paraphrase-tolerant routing to BM25
without a server, model weights, network access, or an upstream plugin.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter

from recall_common import normalize_text, routing_terms


DIMENSIONS = 384
_LATIN_WORD = re.compile(r"[a-z][a-z0-9_+.#/-]{2,96}")

# Small, auditable equivalence classes improve recurring cross-language recall.
# They are routing aids only; raw evidence and structured-anchor guards still
# decide whether a historical claim is answerable.
EQUIVALENCE_GROUPS = (
    ("config", "configuration", "配置"),
    ("checkpoint", "resume", "恢复", "断点", "续跑"),
    ("commit", "revision", "提交", "版本"),
    ("tokenizer", "词表", "分词器"),
    ("archive", "history", "conversation", "归档", "历史", "对话"),
    ("recall", "retrieve", "search", "召回", "检索", "查找"),
    ("earliest", "first", "initial", "最早", "首次", "最初"),
    ("latest", "recent", "current", "最新", "最近", "当前"),
    ("delete", "remove", "purge", "删除", "清除"),
    ("approve", "authorize", "permission", "批准", "授权", "许可"),
    ("failure", "error", "bug", "失败", "报错", "错误"),
    ("training", "pretrain", "训练", "预训练"),
)
_CANONICAL = {
    normalize_text(term): f"concept:{index}"
    for index, group in enumerate(EQUIVALENCE_GROUPS)
    for term in group
}


def _features(text: str) -> Counter[str]:
    normalized = normalize_text(text)
    values: Counter[str] = Counter()
    for term in routing_terms(normalized, max_terms=1600):
        values[f"term:{term}"] += 1
        canonical = _CANONICAL.get(term)
        if canonical:
            values[canonical] += 2
        if len(term) >= 5 and _LATIN_WORD.fullmatch(term):
            padded = f"^{term}$"
            for width in (3, 4):
                for index in range(len(padded) - width + 1):
                    values[f"char:{padded[index:index + width]}"] += 1
    return values


def embed(text: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for feature, count in _features(text).items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "little") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * (1.0 + math.log(float(count)))
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(vector)


def encode(text: str) -> bytes:
    return encode_vector(embed(text))


def encode_vector(vector: tuple[float, ...] | list[float]) -> bytes:
    if len(vector) != DIMENSIONS:
        raise ValueError(f"invalid vector dimensions: expected {DIMENSIONS}, got {len(vector)}")
    return struct.pack(f"<{DIMENSIONS}f", *vector)


def decode(blob: bytes | None) -> tuple[float, ...]:
    if not blob:
        return (0.0,) * DIMENSIONS
    expected = DIMENSIONS * 4
    if len(blob) != expected:
        raise ValueError(f"invalid local embedding blob: expected {expected} bytes, got {len(blob)}")
    return struct.unpack(f"<{DIMENSIONS}f", blob)


def cosine(query_vector: tuple[float, ...], blob: bytes | None) -> float:
    candidate = decode(blob)
    return cosine_vectors(query_vector, candidate)


def cosine_vectors(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
