#!/usr/bin/env python3
"""Dependency-free multi-vector, LSH, graph, and evidence-set primitives.

The implementation borrows structural ideas from late-interaction and
graph-assisted retrieval, but deliberately does not ship or download a neural
model.  Every score is reproducible from local text and every factual answer
must still be verified against the raw transcript.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from local_embedding import cosine_vectors, embed
from recall_common import QueryFeatures, normalize_text


MAX_FACETS = 8
FACET_CHARS = 3_200
LSH_BITS = 64
LSH_BANDS = 8
LSH_BAND_BITS = LSH_BITS // LSH_BANDS
_CLAUSE_RE = re.compile(r"(?:\r?\n+|[。！？!?；;]|\s+(?:and|or|then|with|plus)\s+|以及|并且|然后|同时|还有)", re.IGNORECASE)
_SIMHASH_LAYOUT = tuple(
    (
        dimension % LSH_BITS,
        1.0 if hashlib.blake2b(f"yugo-lsh:{dimension}".encode(), digest_size=1).digest()[0] & 1 else -1.0,
    )
    for dimension in range(384)
)


@dataclass(frozen=True)
class Facet:
    kind: str
    ordinal: int
    text: str


def _bounded_clauses(text: str, limit: int = MAX_FACETS) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    clauses = [item.strip(" ,，") for item in _CLAUSE_RE.split(normalized) if len(item.strip()) >= 3]
    if not clauses:
        clauses = [normalized]
    result: list[str] = []
    for clause in clauses:
        if len(clause) <= FACET_CHARS:
            result.append(clause)
        else:
            for start in range(0, len(clause), FACET_CHARS):
                result.append(clause[start:start + FACET_CHARS])
        if len(result) >= limit:
            break
    return result[:limit]


def exchange_facets(user_message: str, assistant_message: str, part_text: str) -> list[Facet]:
    """Build role-preserving facets for late interaction.

    A whole exchange vector remains useful for coarse routing.  These smaller
    facets prevent one long answer from washing out the short user wording or
    an exact assistant-side decision.
    """

    facets: list[Facet] = []
    user_clauses = _bounded_clauses(user_message, limit=1)
    if user_clauses:
        facets.append(Facet("user", len(facets), user_clauses[0]))
    # The node-level vector already represents the whole part. Sample three
    # locations only when a long part could dilute a short fact.
    if len(part_text) > FACET_CHARS:
        starts = [0, max(0, len(part_text) // 2 - FACET_CHARS // 2), max(0, len(part_text) - FACET_CHARS)]
        for start in dict.fromkeys(starts):
            sample = part_text[start:start + FACET_CHARS]
            if sample.strip():
                facets.append(Facet("assistant-sample", len(facets), sample))
    return facets


def query_facets(query: str, features: QueryFeatures) -> list[str]:
    values: list[str] = []
    values.extend(features.decisive_anchors)
    values.extend(_bounded_clauses(query, limit=5))
    if features.concept_groups:
        values.extend(" ".join(group) for group in features.concept_groups)
    normalized: dict[str, str] = {}
    for value in values:
        key = normalize_text(value)
        if key:
            normalized.setdefault(key, value)
    return list(normalized.values())[:MAX_FACETS]


def late_interaction_score(query_vectors: Sequence[Sequence[float]], facet_vectors: Sequence[Sequence[float]]) -> float:
    """ColBERT-style MaxSim over compact local facets, normalized to [0, 1]."""

    if not query_vectors or not facet_vectors:
        return 0.0
    maxima = [max(0.0, max(cosine_vectors(query, facet) for facet in facet_vectors)) for query in query_vectors]
    return sum(maxima) / len(maxima)


def simhash64(vector: Sequence[float]) -> int:
    """Create deterministic pooled random-projection bits in O(dimensions)."""

    totals = [0.0] * LSH_BITS
    for dimension, value in enumerate(vector):
        if not value:
            continue
        bit, sign = _SIMHASH_LAYOUT[dimension]
        totals[bit] += value * sign
    signature = 0
    for bit, value in enumerate(totals):
        if value >= 0:
            signature |= 1 << bit
    return signature


def bands_from_simhash(signature: int) -> tuple[str, ...]:
    signature &= (1 << LSH_BITS) - 1
    mask = (1 << LSH_BAND_BITS) - 1
    return tuple(f"{band}:{(signature >> (band * LSH_BAND_BITS)) & mask:02x}" for band in range(LSH_BANDS))


def lsh_bands(vector: Sequence[float]) -> tuple[str, ...]:
    return bands_from_simhash(simhash64(vector))


def hamming_similarity(left: int, right: int) -> float:
    return 1.0 - ((left ^ right).bit_count() / LSH_BITS)


def facet_payload(facets: Iterable[Facet]) -> list[tuple[str, int, bytes, int]]:
    """Return only the compact fields needed after indexing."""

    from local_embedding import encode_vector  # avoid duplicate packing logic

    result = []
    for facet in facets:
        vector = embed(facet.text)
        signature = simhash64(vector)
        sqlite_signature = signature if signature < (1 << 63) else signature - (1 << 64)
        result.append((
            facet.kind,
            facet.ordinal,
            encode_vector(vector),
            sqlite_signature,
        ))
    return result
