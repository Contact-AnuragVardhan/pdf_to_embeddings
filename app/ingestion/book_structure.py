from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ingestion.pdf_extractor import ExtractedPage
from ingestion.structure_detector import StructureState

logger = logging.getLogger(__name__)


@dataclass
class BookChapter:
    chapter_number: str | None = None
    chapter_title: str | None = None
    printed_start_page: int | None = None
    printed_end_page: int | None = None
    pdf_start_page: int | None = None
    pdf_end_page: int | None = None
    confidence: float | None = None
    detected_by: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "printed_start_page": self.printed_start_page,
            "printed_end_page": self.printed_end_page,
            "pdf_start_page": self.pdf_start_page,
            "pdf_end_page": self.pdf_end_page,
            "confidence": self.confidence,
            "detected_by": self.detected_by,
            "metadata": self.metadata or {},
        }


@dataclass
class BookStructure:
    book_title: str | None = None
    subject: str | None = None
    grade: str | None = None
    primary_language: str | None = None
    languages_detected: list[str] = field(default_factory=list)
    publisher: str | None = None
    author: str | None = None
    isbn: str | None = None
    edition: str | None = None
    publication_year: str | None = None
    content_profile: str | None = None
    recommended_chunk_max_tokens: int | None = None
    recommended_chunk_overlap_tokens: int | None = None
    recommended_chunking_strategy: str | None = None
    confidence: float | None = None
    detected_by: str = "unknown"
    chapters: list[BookChapter] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_title": self.book_title,
            "subject": self.subject,
            "grade": self.grade,
            "primary_language": self.primary_language,
            "languages_detected": self.languages_detected,
            "publisher": self.publisher,
            "author": self.author,
            "isbn": self.isbn,
            "edition": self.edition,
            "publication_year": self.publication_year,
            "content_profile": self.content_profile,
            "recommended_chunk_max_tokens": self.recommended_chunk_max_tokens,
            "recommended_chunk_overlap_tokens": self.recommended_chunk_overlap_tokens,
            "recommended_chunking_strategy": self.recommended_chunking_strategy,
            "confidence": self.confidence,
            "detected_by": self.detected_by,
            "chapters": [c.to_dict() for c in self.chapters],
            "metadata": self.metadata or {},
        }

    def normalized_chapters(self) -> list[BookChapter]:
        return normalize_chapters(self.chapters)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _clean_title(value: Any) -> str | None:
    if value is None:
        return None
    title = " ".join(str(value).strip().split())
    return title or None


def structure_from_llm_json(data: dict[str, Any], *, detected_by: str, fallback_metadata: dict[str, Any]) -> BookStructure:
    chapters: list[BookChapter] = []
    for index, raw in enumerate(data.get("chapters") or [], start=1):
        if not isinstance(raw, dict):
            continue
        title = _clean_title(raw.get("chapter_title") or raw.get("title") or raw.get("name"))
        if not title:
            continue
        number = raw.get("chapter_number") or raw.get("number") or str(index)
        chapters.append(
            BookChapter(
                chapter_number=str(number) if number is not None and str(number).strip() else str(index),
                chapter_title=title,
                printed_start_page=_int_or_none(raw.get("printed_start_page") or raw.get("start_page")),
                printed_end_page=_int_or_none(raw.get("printed_end_page")),
                pdf_start_page=_int_or_none(raw.get("pdf_start_page")),
                pdf_end_page=_int_or_none(raw.get("pdf_end_page")),
                confidence=_float_or_none(raw.get("confidence")) or _float_or_none(data.get("confidence")),
                detected_by=detected_by,
                metadata={k: v for k, v in raw.items() if k not in {
                    "chapter_number", "number", "chapter_title", "title", "name",
                    "printed_start_page", "start_page", "printed_end_page",
                    "pdf_start_page", "pdf_end_page", "confidence",
                }},
            )
        )

    languages = data.get("languages_detected") or data.get("languages") or []
    if isinstance(languages, str):
        languages = [x.strip() for x in languages.split(",") if x.strip()]
    if not isinstance(languages, list):
        languages = []

    return BookStructure(
        book_title=_clean_title(data.get("book_title") or data.get("title")) or fallback_metadata.get("book_title"),
        subject=_clean_title(data.get("subject")) or fallback_metadata.get("subject"),
        grade=_clean_title(data.get("grade") or data.get("class_name") or data.get("class")) or fallback_metadata.get("grade") or fallback_metadata.get("class_name"),
        primary_language=_clean_title(data.get("primary_language") or data.get("language")) or fallback_metadata.get("language"),
        languages_detected=[str(x) for x in languages if str(x).strip()],
        publisher=_clean_title(data.get("publisher")) or fallback_metadata.get("publisher"),
        author=_clean_title(data.get("author")) or fallback_metadata.get("author"),
        isbn=_clean_title(data.get("isbn")) or fallback_metadata.get("isbn"),
        edition=_clean_title(data.get("edition")) or fallback_metadata.get("edition"),
        publication_year=_clean_title(data.get("publication_year")) or fallback_metadata.get("publication_year"),
        content_profile=_clean_title(data.get("content_profile")),
        recommended_chunk_max_tokens=_int_or_none(data.get("recommended_chunk_max_tokens")),
        recommended_chunk_overlap_tokens=_int_or_none(data.get("recommended_chunk_overlap_tokens")),
        recommended_chunking_strategy=_clean_title(data.get("recommended_chunking_strategy")),
        confidence=_float_or_none(data.get("confidence")),
        detected_by=detected_by,
        chapters=normalize_chapters(chapters),
        metadata={
            "raw_llm_metadata": data,
        },
    )


def normalize_chapters(chapters: list[BookChapter]) -> list[BookChapter]:
    valid = [c for c in chapters if c.chapter_title]
    valid.sort(key=lambda c: (
        c.pdf_start_page if c.pdf_start_page is not None else 10**9,
        c.printed_start_page if c.printed_start_page is not None else 10**9,
        _int_or_none(c.chapter_number) if _int_or_none(c.chapter_number) is not None else 10**9,
        c.chapter_title or "",
    ))
    for index, chapter in enumerate(valid):
        if not chapter.chapter_number:
            chapter.chapter_number = str(index + 1)
    return valid


def enrich_chapter_page_ranges(chapters: list[BookChapter], pages: list[ExtractedPage]) -> list[BookChapter]:
    """Fill missing pdf_start/pdf_end pages using title search and TOC printed-page offset.

    This helps books where the table of contents gives printed page numbers but the PDF
    has cover/preface pages before printed page 1. It never calls the LLM; it only uses
    chapter titles already detected by the LLM/rules.
    """
    if not chapters:
        return []
    chapters = normalize_chapters([BookChapter(**c.to_dict()) for c in chapters])
    page_text_by_number = {p.page_number: p.cleaned_text or "" for p in pages}
    max_page = max(page_text_by_number) if page_text_by_number else 0

    # Try to locate each chapter title directly in PDF page text.
    last_found = 0
    for chapter in chapters:
        if chapter.pdf_start_page:
            last_found = max(last_found, chapter.pdf_start_page)
            continue
        found = _find_title_start_page(chapter.chapter_title or "", page_text_by_number, min_page=max(1, last_found + 1))
        if found:
            chapter.pdf_start_page = found
            chapter.metadata["pdf_start_page_source"] = "title_scan"
            last_found = found

    # If at least one chapter has both printed and PDF start page, derive offset.
    offsets = [
        c.pdf_start_page - c.printed_start_page
        for c in chapters
        if c.pdf_start_page is not None and c.printed_start_page is not None
    ]
    if offsets:
        # Use the most common offset; it handles front matter in many textbooks.
        offset = max(set(offsets), key=offsets.count)
        for chapter in chapters:
            if not chapter.pdf_start_page and chapter.printed_start_page is not None:
                chapter.pdf_start_page = max(1, chapter.printed_start_page + offset)
                chapter.metadata["pdf_start_page_source"] = "printed_page_offset"
                chapter.metadata["printed_to_pdf_offset"] = offset

    # Last fallback: keep any chapter without pdf_start_page out of page mapping but still store it.
    mapped = [c for c in chapters if c.pdf_start_page]
    mapped.sort(key=lambda c: c.pdf_start_page or 10**9)
    for index, chapter in enumerate(mapped):
        next_start = mapped[index + 1].pdf_start_page if index + 1 < len(mapped) else None
        if not chapter.pdf_end_page:
            chapter.pdf_end_page = (next_start - 1) if next_start else max_page
        if chapter.printed_start_page is not None and chapter.printed_end_page is None:
            next_printed = None
            for later in chapters[index + 1:]:
                if later.printed_start_page is not None:
                    next_printed = later.printed_start_page
                    break
            if next_printed is not None:
                chapter.printed_end_page = next_printed - 1
    return chapters


def _find_title_start_page(title: str, page_text_by_number: dict[int, str], *, min_page: int = 1) -> int | None:
    title_norm = _normalize_for_match(title)
    if not title_norm:
        return None
    for page_number in sorted(page_text_by_number):
        if page_number < min_page:
            continue
        text = page_text_by_number[page_number]
        if not text.strip():
            continue
        head = "\n".join(text.splitlines()[:8])[:700]
        head_norm = _normalize_for_match(head)
        # Exact line/heading at top gets high confidence.
        for line in head.splitlines()[:6]:
            if _normalize_for_match(line) == title_norm:
                return page_number
        # Accept title near the beginning for OCR that merges heading with body.
        if head_norm.startswith(title_norm + " ") or head_norm == title_norm:
            return page_number
    return None


def _normalize_for_match(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\u0900-\u097F]+", " ", text)
    return " ".join(text.split())


class ChapterResolver:
    def __init__(self, chapters: list[BookChapter]) -> None:
        self.chapters = [c for c in normalize_chapters(chapters) if c.pdf_start_page]
        self.chapters.sort(key=lambda c: c.pdf_start_page or 10**9)

    def chapter_for_pdf_page(self, page_number: int) -> BookChapter | None:
        for chapter in self.chapters:
            start = chapter.pdf_start_page or 0
            end = chapter.pdf_end_page or 10**9
            if start <= page_number <= end:
                return chapter
        return None

    def printed_page_for_pdf_page(self, page_number: int) -> int | None:
        chapter = self.chapter_for_pdf_page(page_number)
        if not chapter or chapter.printed_start_page is None or chapter.pdf_start_page is None:
            return None
        return chapter.printed_start_page + (page_number - chapter.pdf_start_page)

    def structure_for_page(self, page_number: int) -> StructureState:
        chapter = self.chapter_for_pdf_page(page_number)
        if not chapter:
            return StructureState()
        return StructureState(
            chapter_number=chapter.chapter_number,
            chapter_title=chapter.chapter_title,
        )
