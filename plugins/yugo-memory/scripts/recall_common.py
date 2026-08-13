#!/usr/bin/env python3
"""Shared, auditable text features for the Yugo Memory recall index."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#-]{1,127}")
ALNUM_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:[A-Za-z]+|[卡层维])(?![A-Za-z0-9])", re.IGNORECASE)
DECISIVE_STRUCTURED_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]{4,64}\b"),
    re.compile(r"\b[0-9a-fA-F]{7,64}\b"),
    re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),
    re.compile(r"(?:^|\s)(--?[A-Za-z][A-Za-z0-9_-]{1,63})(?=\s|=|$)"),
    re.compile(r"(?:~?/|/)(?=[A-Za-z0-9._-]+/)[^\s\"'<>]{2,240}"),
    re.compile(r"https?://[^\s\"'<>]{3,300}"),
    re.compile(r"\bv?\d+(?:\.\d+){1,4}(?:[-+][A-Za-z0-9.-]+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:step|epoch|task|run|job)[_:#-]?\d{3,}\b", re.IGNORECASE),
)
CONTEXTUAL_STRUCTURED_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:[kmgtb]|卡|层|维)(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"\b(?:\d{1,3}(?:,\d{3})+|\d{5,})\b"),
    re.compile(r"\b\d+\.\.\d+\b"),
)
EARLIEST_RE = re.compile(
    r"最早|第一次|首次|起初|一开始|最初|earliest|first(?: time)?|initial(?:ly)?",
    re.IGNORECASE,
)
LATEST_RE = re.compile(
    r"最近|最新|最后一次|最终|目前|现在|most recent|latest|current|final(?:ly)?",
    re.IGNORECASE,
)
TEMPORAL_RE = re.compile(
    r"最近|最新|最早|第一次|首次|上次|之前|后来|当时|最终|目前|现在|今天|昨天|last|latest|recent|before|after|current|final|earliest|first|initial",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(
    r"第\s*(\d+|[零〇一二三四五六七八九十百两]+)\s*(?:次|轮|个|条|段|回|问|对话|消息|交流)"
    r"(?:的)?(?:对话|交流|消息|内容)?"
    r"|\b(\d+)(?:st|nd|rd|th)\b"
    r"|\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "what", "when", "where",
    "which", "into", "then", "than", "have", "has", "how", "why", "our", "your",
    "was", "were", "are", "is", "to", "of", "in", "on", "or", "a", "an",
    "为什", "什么", "为什么", "怎么", "如何", "是否", "请问", "告诉", "帮我",
    "我们", "这个", "那个", "哪些", "一下", "需要",
    "是什", "什么", "是什么", "多少", "各有", "有多", "有多少",
}

# These words express how to order otherwise relevant evidence. Keeping them in
# FTS terms makes "latest X" compete with an unrelated exchange that merely says
# "latest". Remove them from lexical matching and apply time ordering separately.
TEMPORAL_QUERY_TERMS = {
    "最早", "最新", "最近", "首次", "第一", "起初", "最初", "初始", "最终",
    "目前", "当前", "现在", "上次", "之前", "后来", "当时", "earliest", "first",
    "initial", "initially", "latest", "recent", "current", "final", "finally",
}

QUERY_EXPANSIONS = {
    "配置": ("config", "configuration"),
    "冻结": ("最终", "frozen", "freeze", "final"),
    "提交号": ("commit",),
    "提交": ("commit",),
    "断点续跑": ("断点续", "点续跑", "原地续跑", "sqlite", "jsonl", "journal", "resume", "checkpoint"),
    "批准": ("授权", "approved", "authorization"),
    "授权": ("批准", "approved", "authorization"),
    "编号": ("id", "ids", "稳定"),
    "协议": ("protocol",),
    "兼容": ("compatible", "compatibility"),
    "复用": ("reuse",),
}


@dataclass(frozen=True)
class QueryFeatures:
    query: str
    terms: tuple[str, ...]
    anchors: tuple[str, ...]
    decisive_anchors: tuple[str, ...]
    concept_groups: tuple[tuple[str, ...], ...]
    has_temporal_intent: bool
    temporal_direction: str | None
    ordinal_index: int | None
    has_structured_anchor: bool
    has_decisive_anchor: bool


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def content_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8", errors="replace")).hexdigest()


def anchor_sets(text: str) -> tuple[list[str], list[str]]:
    """Return normalized structured anchors and the decisive subset.

    Decisive anchors are values such as paths, commits, URLs, flags, and quoted
    identifiers.  They are indexed separately so an exact value can bypass
    approximate routing without turning ordinary model-size numbers into proof.
    """
    decisive: list[str] = []
    contextual: list[str] = []

    def collect(patterns: tuple[re.Pattern[str], ...], target: list[str]) -> None:
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = next((group for group in match.groups() if group), None) if match.groups() else match.group(0)
                value = (value or "").strip().strip(".,;:!?。；，！？)]}").lower()
                if len(value) >= 2:
                    target.append(value)

    collect(DECISIVE_STRUCTURED_PATTERNS, decisive)
    collect(CONTEXTUAL_STRUCTURED_PATTERNS, contextual)
    for match in re.finditer(r"[`\"']([^`\"'\n]{3,160})[`\"']", text):
        decisive.append(normalize_text(match.group(1)))
    decisive = list(dict.fromkeys(decisive))
    all_anchors = list(dict.fromkeys((*decisive, *contextual)))
    return all_anchors, decisive


def indexable_anchor(value: str) -> bool:
    """Whether a decisive value is stable enough for direct postings/graph edges.

    Quoted natural-language spans remain query guards, but indexing every quote
    from tool output creates a large noisy graph. Stable identifiers keep their
    direct path; ordinary quotes continue through exact phrase and FTS search.
    """

    normalized = normalize_text(value)
    if not normalized:
        return False
    return bool(
        re.fullmatch(r"0x[0-9a-f]{4,64}", normalized)
        or re.fullmatch(r"[0-9a-f]{7,64}", normalized)
        or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", normalized)
        or re.fullmatch(r"--?[a-z][a-z0-9_-]{1,63}", normalized)
        or re.fullmatch(r"https?://[^\s]{3,300}", normalized)
        or re.fullmatch(r"(?:~?/|/)[^\s]{2,240}", normalized)
        or re.fullmatch(r"v?\d+(?:\.\d+){1,4}(?:[-+][a-z0-9.-]+)?", normalized)
        or re.fullmatch(r"(?:step|epoch|task|run|job)[_:#-]?\d{3,}", normalized)
    )


def _structured_anchors(text: str) -> list[str]:
    anchors, _ = anchor_sets(text)
    return anchors


def extract_terms(text: str, max_terms: int = 6000) -> list[str]:
    """Build multilingual sparse keys without depending on a language segmenter.

    Latin/code/path identifiers are preserved while Chinese runs contribute
    overlapping bigrams and short unigrams. Unique terms are ordered by their
    first occurrence so indexing is deterministic.
    """

    normalized = unicodedata.normalize("NFKC", text or "")
    ordered: dict[str, None] = {}

    def add(value: str) -> None:
        value = value.strip().lower()
        if 1 < len(value) <= 300 and value not in STOPWORDS:
            ordered.setdefault(value, None)

    for anchor in _structured_anchors(normalized):
        add(anchor)
        for part in re.split(r"[/_.:=+@#-]+", anchor):
            add(part)

    for match in LATIN_RE.finditer(normalized):
        token = match.group(0)
        add(token)
        for part in re.split(r"[_+.#-]+", token):
            add(part)
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token):
            add(part)

    for match in ALNUM_RE.finditer(normalized):
        add(match.group(0))

    for match in CJK_RE.finditer(normalized):
        run = match.group(0)
        if len(run) == 1:
            add(run)
            continue
        for width in (2, 3):
            if len(run) < width:
                continue
            for index in range(len(run) - width + 1):
                add(run[index:index + width])
                if len(ordered) >= max_terms:
                    return list(ordered)

    return list(ordered)[:max_terms]


def routing_terms(text: str, max_terms: int = 1600) -> list[str]:
    """Extract bounded terms from evenly spaced windows of long text.

    This avoids a large duplicated term store while keeping the head, middle,
    and tail searchable. Stable structured anchors use a separate complete
    posting scan, so this sampling is never the only path to an exact value.
    """

    text = text or ""
    if len(text) <= 8_000:
        return extract_terms(text, max_terms=max_terms)
    window = min(5_000, max(2_000, len(text) // 4))
    starts = [0, max(0, len(text) // 3 - window // 2), max(0, 2 * len(text) // 3 - window // 2), max(0, len(text) - window)]
    per_window = max(64, (max_terms + len(starts) - 1) // len(starts))
    ordered: dict[str, None] = {}
    for start in dict.fromkeys(starts):
        for term in extract_terms(text[start:start + window], max_terms=per_window):
            ordered.setdefault(term, None)
            if len(ordered) >= max_terms:
                return list(ordered)
    return list(ordered)


def query_features(query: str) -> QueryFeatures:
    normalized = normalize_text(query)
    raw_anchors, raw_decisive = anchor_sets(query)
    anchors = tuple(raw_anchors)
    decisive_anchors = tuple(raw_decisive)
    ordinal_match = ORDINAL_RE.search(query)
    ordinal_index = _ordinal_value(ordinal_match) if ordinal_match else None
    lexical_query = ORDINAL_RE.sub(" ", query)
    terms_list = [term for term in extract_terms(lexical_query, max_terms=96) if term not in TEMPORAL_QUERY_TERMS]
    concept_groups: list[tuple[str, ...]] = []
    for phrase, expansions in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            concept_groups.append(tuple(dict.fromkeys((phrase, *expansions))))
            for expansion in expansions:
                if expansion not in terms_list:
                    terms_list.append(expansion)
    terms = tuple(terms_list[:112])
    temporal_direction = (
        "ordinal" if ordinal_index is not None
        else "earliest" if EARLIEST_RE.search(query)
        else "latest" if LATEST_RE.search(query)
        else None
    )
    return QueryFeatures(
        query=normalized,
        terms=terms,
        anchors=anchors,
        decisive_anchors=decisive_anchors,
        concept_groups=tuple(concept_groups),
        has_temporal_intent=bool(TEMPORAL_RE.search(query) or ordinal_index is not None),
        temporal_direction=temporal_direction,
        ordinal_index=ordinal_index,
        has_structured_anchor=bool(anchors),
        has_decisive_anchor=bool(decisive_anchors),
    )


def _ordinal_value(match: re.Match[str]) -> int | None:
    value = next((group for group in match.groups() if group), "")
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 10_000 else None
    english = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    if value.lower() in english:
        return english[value.lower()]
    chinese = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "百" in value:
        left, _, right = value.partition("百")
        number = (chinese.get(left, 1) * 100) + (_chinese_under_hundred(right, chinese) if right else 0)
    else:
        number = _chinese_under_hundred(value, chinese)
    return number if 1 <= number <= 10_000 else None


def _chinese_under_hundred(value: str, digits: dict[str, int]) -> int:
    if not value:
        return 0
    if "十" in value:
        left, _, right = value.partition("十")
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    number = 0
    for char in value:
        number = number * 10 + digits.get(char, 0)
    return number


def term_coverage(terms_text: str, query_terms: tuple[str, ...]) -> float:
    if not query_terms:
        return 0.0
    available = set((terms_text or "").split())
    return sum(1 for term in query_terms if term in available) / len(query_terms)


def concept_coverage(terms_text: str, groups: tuple[tuple[str, ...], ...]) -> float:
    if not groups:
        return 0.0
    available = set((terms_text or "").split())
    return sum(any(term in available for term in group) for group in groups) / len(groups)


def evidence_coverage(terms_text: str, features: QueryFeatures) -> tuple[float, float, float]:
    lexical = term_coverage(terms_text, features.terms)
    concepts = concept_coverage(terms_text, features.concept_groups)
    # Synonyms repair surface-form mismatch; they must not turn one generic word
    # (for example "protocol") into enough evidence for an unrelated question.
    combined = lexical if not features.concept_groups else min(1.0, 0.75 * lexical + 0.25 * concepts)
    return lexical, concepts, combined


def salient_terms(texts: list[str], limit: int = 320) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(set(extract_terms(text, max_terms=1200)))
    return [term for term, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]]
