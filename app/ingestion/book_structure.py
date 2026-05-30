from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
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
    """Repair and complete chapter/page ranges without trusting the LLM blindly.

    Why this is needed:
    - OCR/PyMuPDF often extracts a TOC in two columns: all titles first, then all page
      numbers. An LLM can then attach the wrong printed pages to the right titles.
    - Once ``printed_start_page`` is wrong, raw pages/chunks get wrong printed pages.
    - Answer-key sections are often omitted by LLMs even though they are important for RAG.

    Strategy:
    1. Locate chapter starts by scanning actual page headings.
    2. Anchor printed page 1 to chapter 1 when possible. For normal textbooks this is
       much safer than trusting noisy TOC page numbers.
    3. Recalculate printed_start_page from the PDF page using the anchor offset.
    4. Use the anchor offset to map TOC-only sections such as Answers.
    5. Recalculate all end pages from the next start page.
    """
    if not pages:
        return []

    page_text_by_number = {p.page_number: p.cleaned_text or "" for p in pages}
    max_page = max(page_text_by_number) if page_text_by_number else 0

    # Clone chapters so callers do not see surprising partial mutations.
    cloned = [BookChapter(**c.to_dict()) for c in (chapters or []) if c.chapter_title]

    # If Answers is missing, try to add it from the actual pages. This is a fallback;
    # the preferred path is that RuleBasedStructureDetector parses "Answers" from TOC.
    if not _has_title(cloned, "answers"):
        answers_pdf_page = _detect_answers_start_page(page_text_by_number)
        if answers_pdf_page:
            cloned.append(
                BookChapter(
                    chapter_number=str(_next_chapter_number(cloned)),
                    chapter_title="Answers",
                    pdf_start_page=answers_pdf_page,
                    confidence=0.70,
                    detected_by="rule_based_answers_page",
                    metadata={"pdf_start_page_source": "answers_page_heuristic"},
                )
            )

    # For detection, preserve TOC/chapter-number order. Do not sort by printed pages here
    # because printed pages may be the corrupt values we are trying to repair.
    chapters = _order_chapters_for_detection(cloned)

    # Step 1: locate each chapter title directly in body pages.
    # IMPORTANT: do NOT blindly trust pdf_start_page already supplied by the LLM.
    # In noisy PDFs the LLM can attach the right title to the wrong PDF page.
    # We therefore rescan the actual page headings and override bad existing values.
    # Keep the scan monotonic so repeated words in previous chapters do not hijack a later chapter.
    last_found = 0
    for chapter in chapters:
        scan_from = max(1, last_found + 1)
        found = _find_title_start_page(chapter.chapter_title or "", page_text_by_number, min_page=scan_from)
        if found:
            if chapter.pdf_start_page != found:
                chapter.metadata["original_pdf_start_page"] = chapter.pdf_start_page
                chapter.metadata["pdf_start_page_repaired"] = True
            chapter.pdf_start_page = found
            chapter.metadata["pdf_start_page_source"] = "title_scan"
            last_found = found
        elif chapter.pdf_start_page:
            # Existing value is only a fallback when heading scan fails. Still keep
            # monotonic progress so later scans do not search backwards.
            last_found = max(last_found, chapter.pdf_start_page)

    # Answers headings are sometimes extracted after the answer table, so the title scan
    # may return page 310 even when the answer section visibly starts on page 308.
    answers_start = _detect_answers_start_page(page_text_by_number)
    if answers_start:
        for chapter in chapters:
            if _normalize_for_match(chapter.chapter_title or "") in {"answer", "answers"}:
                if chapter.pdf_start_page is None or answers_start < chapter.pdf_start_page:
                    chapter.metadata["original_pdf_start_page"] = chapter.pdf_start_page
                    chapter.pdf_start_page = answers_start
                    chapter.metadata["pdf_start_page_source"] = "answers_page_heuristic"

    # Step 2: determine offset between printed page and PDF page.
    # Best anchor: chapter 1 starts at printed page 1.
    offset = _infer_pdf_to_printed_offset(chapters)

    # Step 3: if offset is known, map missing pdf pages from printed pages and repair printed pages.
    if offset is not None:
        for chapter in chapters:
            if chapter.pdf_start_page is None and chapter.printed_start_page is not None:
                calculated_pdf = chapter.printed_start_page + offset
                if 1 <= calculated_pdf <= max_page:
                    chapter.pdf_start_page = calculated_pdf
                    chapter.metadata["pdf_start_page_source"] = "printed_page_offset"
                    chapter.metadata["printed_to_pdf_offset"] = offset

        # Recalculate printed pages from the trusted PDF start pages.
        for chapter in chapters:
            if chapter.pdf_start_page is None:
                continue
            repaired_printed = chapter.pdf_start_page - offset
            if repaired_printed >= 1:
                if chapter.printed_start_page != repaired_printed:
                    chapter.metadata["original_printed_start_page"] = chapter.printed_start_page
                    chapter.metadata["printed_start_page_repaired"] = True
                chapter.printed_start_page = repaired_printed
                chapter.metadata["printed_page_source"] = "pdf_page_minus_offset"
                chapter.metadata["printed_to_pdf_offset"] = offset

    # Step 4: recalculate pdf/printed end pages from next chapter start.
    mapped = [c for c in chapters if c.pdf_start_page]
    mapped.sort(key=lambda c: c.pdf_start_page or 10**9)

    for index, chapter in enumerate(mapped):
        next_start = mapped[index + 1].pdf_start_page if index + 1 < len(mapped) else None
        calculated_pdf_end = (next_start - 1) if next_start else max_page
        chapter.pdf_end_page = calculated_pdf_end

        if chapter.printed_start_page is not None:
            if offset is not None and chapter.pdf_end_page is not None:
                chapter.printed_end_page = chapter.pdf_end_page - offset
            else:
                next_printed = mapped[index + 1].printed_start_page if index + 1 < len(mapped) else None
                if next_printed is not None:
                    chapter.printed_end_page = next_printed - 1

    # Return all chapters, but sorted so mapped chapters drive ChapterResolver correctly.
    return normalize_chapters(chapters)


def _has_title(chapters: list[BookChapter], title: str) -> bool:
    wanted = _normalize_for_match(title)
    return any(_normalize_for_match(c.chapter_title or "") == wanted for c in chapters)


def _chapter_number_int(chapter: BookChapter) -> int | None:
    raw = chapter.chapter_number
    if raw is None:
        return None
    match = re.search(r"\d+", str(raw))
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _next_chapter_number(chapters: list[BookChapter]) -> int:
    nums = [_chapter_number_int(c) for c in chapters]
    nums = [n for n in nums if n is not None]
    return (max(nums) + 1) if nums else len(chapters) + 1


def _order_chapters_for_detection(chapters: list[BookChapter]) -> list[BookChapter]:
    indexed = [(idx, c) for idx, c in enumerate(chapters) if c.chapter_title]

    def key(item: tuple[int, BookChapter]) -> tuple[int, int, int]:
        idx, chapter = item
        num = _chapter_number_int(chapter)
        # Numbered chapters should follow chapter-number order; unnumbered sections keep
        # their original order after numbered chapters.
        return (0 if num is not None else 1, num if num is not None else 10**9, idx)

    return [c for _, c in sorted(indexed, key=key)]


def _infer_pdf_to_printed_offset(chapters: list[BookChapter]) -> int | None:
    """Return offset such that pdf_page = printed_page + offset."""
    mapped = [c for c in chapters if c.pdf_start_page is not None]
    if not mapped:
        return None

    # Most reliable for school textbooks: chapter 1 body starts at printed page 1.
    chapter_one_candidates = [c for c in mapped if _chapter_number_int(c) == 1]
    if chapter_one_candidates:
        first = min(chapter_one_candidates, key=lambda c: c.pdf_start_page or 10**9)
        offset = (first.pdf_start_page or 0) - 1
        if offset >= 0:
            for c in chapters:
                c.metadata.setdefault("printed_to_pdf_offset_source", "chapter_1_anchor")
            return offset

    # Fallback: use the most common existing offset, but only if it has enough support.
    offsets: list[int] = []
    for c in mapped:
        if c.printed_start_page is None:
            continue
        offset = c.pdf_start_page - c.printed_start_page
        # Front matter offset is normally small and non-negative. Ignore suspicious values.
        if 0 <= offset <= 80:
            offsets.append(offset)
    if not offsets:
        return None
    best = max(set(offsets), key=offsets.count)
    if offsets.count(best) >= 2:
        for c in chapters:
            c.metadata.setdefault("printed_to_pdf_offset_source", "dominant_existing_offset")
        return best
    return None


def _detect_answers_start_page(page_text_by_number: dict[int, str]) -> int | None:
    """Find answer-key start page when the TOC/LLM omitted it.

    Search only near the end of the book. Require a real "Answers" line, not body text
    such as "choose the correct answer".
    """
    if not page_text_by_number:
        return None
    max_page = max(page_text_by_number)
    min_page = max(1, int(max_page * 0.80))
    answer_heading_page: int | None = None

    for page_number in sorted(page_text_by_number):
        if page_number < min_page:
            continue
        text = page_text_by_number[page_number] or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_lines = lines[:20]
        has_answer_heading = any(_normalize_for_match(line) in {"answer", "answers"} for line in first_lines)
        # Some extraction orders put the centered heading later in the text. Accept it
        # only when the page also looks like an answer key.
        has_later_answer_heading = any(_normalize_for_match(line) in {"answer", "answers"} for line in lines)
        if has_answer_heading or (has_later_answer_heading and _looks_like_answer_key_page(text)):
            answer_heading_page = page_number
            break

    if answer_heading_page is None:
        return None

    # Some PDFs extract the heading after tables, or miss it on the first answer page.
    # Walk back a few pages if they look like answer-key pages and not like Activities.
    start = answer_heading_page
    for page_number in range(answer_heading_page - 1, max(min_page, answer_heading_page - 4) - 1, -1):
        text = page_text_by_number.get(page_number, "") or ""
        if _looks_like_activity_page(text):
            break
        if _looks_like_answer_key_page(text):
            start = page_number
        else:
            break
    return start


def _looks_like_activity_page(text: str) -> bool:
    head = text[:1200]
    return bool(re.search(r"\bActivity\s*[-—]?\s*\d+\b|\bMaterials Required\b|\bProcedure\b|\bStep\s*\d+\b", head, flags=re.I))


def _looks_like_answer_key_page(text: str) -> bool:
    head = text[:2500]
    exercise_count = len(re.findall(r"\bEXERCISE\b", head, flags=re.I))
    option_count = len(re.findall(r"\([a-d]\)", head, flags=re.I))
    roman_count = len(re.findall(r"\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)", head, flags=re.I))
    numbered_count = len(re.findall(r"(?:^|\s)\d{1,2}\.", head))
    return exercise_count >= 2 or (option_count >= 5 and numbered_count >= 5) or (roman_count >= 8 and numbered_count >= 5)

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
        head = "\n".join(text.splitlines()[:10])[:900]
        head_norm = _normalize_for_match(head)
        head_norm_without_leading_number = re.sub(r"^\d+\s+", "", head_norm)

        # Exact line/heading at top gets high confidence.
        for line in head.splitlines()[:8]:
            line_norm = _normalize_for_match(line)
            if line_norm == title_norm:
                return page_number
            # OCR can slightly corrupt short headings, e.g. "Exponenis" for "Exponents".
            if _is_close_heading(line_norm, title_norm):
                return page_number

        # Accept title near the beginning for OCR that merges heading with body, allowing
        # a leading chapter number such as "13 Lines and Angles".
        title_pos = head_norm.find(title_norm)
        title_pos_without_number = head_norm_without_leading_number.find(title_norm)
        if (
            head_norm.startswith(title_norm + " ")
            or head_norm == title_norm
            or head_norm_without_leading_number.startswith(title_norm + " ")
            or head_norm_without_leading_number == title_norm
            # Handles pages that start with a short unit label before the full title,
            # e.g. "RATIO / Ratio and Proportion".
            or 0 <= title_pos <= 60
            or 0 <= title_pos_without_number <= 60
        ):
            return page_number
    return None


def _is_close_heading(line_norm: str, title_norm: str) -> bool:
    if not line_norm or not title_norm:
        return False
    # Compare only plausible heading-sized lines to avoid fuzzy matching body text.
    if len(line_norm.split()) > max(6, len(title_norm.split()) + 3):
        return False
    if len(line_norm) < 4:
        return False
    ratio = SequenceMatcher(None, line_norm, title_norm).ratio()
    return ratio >= 0.78


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
