from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ingestion.book_structure import BookStructure
from ingestion.pdf_extractor import ExtractedPage


@dataclass(frozen=True)
class ChunkingPlan:
    strategy: str
    content_profile: str
    max_tokens: int
    overlap_tokens: int
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "content_profile": self.content_profile,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "source": self.source,
            "reason": self.reason,
        }


class ChunkingStrategySelector:
    """Choose chunk settings per document instead of hard-coding one global size.

    This is a standard hybrid pattern for RAG ingestion:
    1. detect the book/content profile,
    2. chunk on chapter/section/question boundaries first,
    3. use recursive token windows only as fallback.
    """

    PROFILES: dict[str, tuple[int, int, str]] = {
        "math_textbook": (620, 90, "toc_structure_aware"),
        "question_bank": (520, 70, "question_block"),
        "science_textbook": (760, 110, "toc_structure_aware"),
        "grammar": (680, 100, "semantic_then_recursive"),
        "english_literature": (1000, 150, "paragraph_story"),
        "hindi_literature": (900, 140, "paragraph_story"),
        "mixed_textbook": (750, 120, "toc_structure_aware"),
        "unknown": (750, 120, "recursive_token"),
    }

    def __init__(self, default_max_tokens: int, default_overlap_tokens: int, *, auto_enabled: bool = True) -> None:
        self.default_max_tokens = default_max_tokens
        self.default_overlap_tokens = default_overlap_tokens
        self.auto_enabled = auto_enabled

    def select(self, pages: list[ExtractedPage], metadata: dict[str, Any], structure: BookStructure | None = None) -> ChunkingPlan:
        if not self.auto_enabled:
            return ChunkingPlan(
                strategy="fixed_env",
                content_profile="fixed_env",
                max_tokens=self.default_max_tokens,
                overlap_tokens=self.default_overlap_tokens,
                source="env",
                reason="AUTO_CHUNKING_ENABLED=false",
            )

        # Accept LLM recommendation only inside safe limits.
        if structure:
            llm_max = _safe_int(structure.recommended_chunk_max_tokens)
            llm_overlap = _safe_int(structure.recommended_chunk_overlap_tokens)
            if llm_max and 350 <= llm_max <= 1400:
                overlap = llm_overlap if llm_overlap and 40 <= llm_overlap <= min(220, llm_max // 3) else _default_overlap_for(llm_max)
                return ChunkingPlan(
                    strategy=structure.recommended_chunking_strategy or "toc_structure_aware",
                    content_profile=structure.content_profile or self._detect_profile(pages, metadata),
                    max_tokens=llm_max,
                    overlap_tokens=overlap,
                    source=structure.detected_by or "llm",
                    reason="LLM returned safe recommended chunk settings.",
                )

        profile = self._detect_profile(pages, metadata)
        max_tokens, overlap, strategy = self.PROFILES.get(profile, self.PROFILES["unknown"])
        return ChunkingPlan(
            strategy=strategy,
            content_profile=profile,
            max_tokens=max_tokens,
            overlap_tokens=overlap,
            source="heuristic_profile",
            reason="Selected from subject/content statistics.",
        )

    def _detect_profile(self, pages: list[ExtractedPage], metadata: dict[str, Any]) -> str:
        subject = (metadata.get("subject") or "").lower()
        sample = "\n".join((p.cleaned_text or "") for p in pages[: min(len(pages), 30)])
        lower = sample.lower()
        token_total = sum(p.token_count for p in pages[: min(len(pages), 30)]) or 1
        question_hits = len(re.findall(r"\?|exercise|objective questions|fill in|true or false|solve|अभ्यास|प्रश्न", sample, re.I))
        formula_hits = len(re.findall(r"[=+×÷≤≥<>]|\b\d+\s*/\s*\d+\b", sample))
        story_hits = len(re.findall(r"\bstory\b|\bpoem\b|कहानी|कविता", sample, re.I))
        grammar_hits = len(re.findall(r"grammar|vocabulary|noun|verb|adjective|व्याकरण|संज्ञा|सर्वनाम|क्रिया|विशेषण", sample, re.I))
        science_hits = len(re.findall(r"experiment|activity|diagram|observe|science|evs|प्रयोग|विज्ञान", sample, re.I))
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", sample))
        latin_chars = len(re.findall(r"[A-Za-z]", sample))

        if "math" in subject or "गणित" in subject or formula_hits > max(20, token_total // 80):
            if question_hits > 30:
                return "math_textbook"
            return "math_textbook"
        if question_hits > 60 and token_total < 12000:
            return "question_bank"
        if any(x in subject for x in ["science", "evs"]) or science_hits >= 4:
            return "science_textbook"
        if "grammar" in subject or grammar_hits >= 4:
            return "grammar"
        if story_hits >= 2 or "literature" in subject or "english" in subject:
            return "hindi_literature" if devanagari_chars > latin_chars else "english_literature"
        if devanagari_chars and latin_chars and min(devanagari_chars, latin_chars) / max(devanagari_chars, latin_chars) > 0.15:
            return "mixed_textbook"
        return "unknown"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _default_overlap_for(max_tokens: int) -> int:
    if max_tokens <= 550:
        return 70
    if max_tokens <= 750:
        return 100
    if max_tokens <= 1000:
        return 140
    return 180
