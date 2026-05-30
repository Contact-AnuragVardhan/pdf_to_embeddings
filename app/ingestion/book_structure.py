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
    # Backward-compatible structure record. For normal textbooks these fields
    # represent chapters. For books like NCERT Poorvi, chapter_* intentionally
    # remains NULL and the record represents a unit/section lesson from the TOC.
    chapter_number: str | None = None
    chapter_title: str | None = None
    unit_number: str | None = None
    unit_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    lesson_title: str | None = None
    structure_type: str = "chapter"  # chapter | section | unit | lesson | answers
    printed_start_page: int | None = None
    printed_end_page: int | None = None
    pdf_start_page: int | None = None
    pdf_end_page: int | None = None
    confidence: float | None = None
    detected_by: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_title(self) -> str | None:
        return self.chapter_title or self.section_title or self.lesson_title or self.unit_title

    @property
    def display_number(self) -> str | None:
        return self.chapter_number or self.section_number or self.unit_number

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "unit_number": self.unit_number,
            "unit_title": self.unit_title,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "lesson_title": self.lesson_title,
            "structure_type": self.structure_type,
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


def _clean_chapter_title_value(
    value: Any,
    *,
    chapter_number: Any | None = None,
    printed_start_page: int | None = None,
) -> str | None:
    """Clean chapter titles returned by an LLM or parsed from a noisy TOC.

    OCR/LLM output frequently turns a TOC row like ``1. Integers 1`` into
    title = ``Integers 1``. If we keep that value, the title scanner can match
    the Contents page instead of the real chapter page and the printed-page
    offset becomes wrong. This cleaner removes only isolated trailing numbers
    that are clearly TOC/page artifacts. It intentionally does not remove
    numbers embedded in real titles such as ``Three-Dimensional Shapes``.
    """
    title = _clean_title(value)
    if not title:
        return None

    title = re.sub(r"\bpage\s+[0-9०-९]+$", "", title, flags=re.I).strip()

    if printed_start_page is not None:
        title = re.sub(rf"\s+{printed_start_page}$", "", title).strip()

    chapter_num = _int_or_none(chapter_number)
    if chapter_num is not None:
        # Strip a duplicated chapter number at the end, e.g. ``Decimals 3``
        # from the TOC row ``3. Decimals 3 36``. Require whitespace before
        # the number so titles such as ``Test Paper-1`` are not damaged.
        title = re.sub(rf"\s+{chapter_num}$", "", title).strip()

    # Strip a leading chapter number accidentally included in the title.
    if chapter_num is not None:
        title = re.sub(rf"^\s*{chapter_num}\s*[.)\-–:,]+\s*", "", title).strip()

    title = title.strip(" .:-–,")
    return _clean_title(title)


def structure_from_llm_json(data: dict[str, Any], *, detected_by: str, fallback_metadata: dict[str, Any]) -> BookStructure:
    chapters: list[BookChapter] = []
    for index, raw in enumerate(data.get("chapters") or [], start=1):
        if not isinstance(raw, dict):
            continue

        number = raw.get("chapter_number") or raw.get("number") or str(index)
        printed_start_page = _int_or_none(raw.get("printed_start_page") or raw.get("start_page"))
        title = _clean_chapter_title_value(
            raw.get("chapter_title") or raw.get("title") or raw.get("name"),
            chapter_number=number,
            printed_start_page=printed_start_page,
        )
        if not title:
            continue

        chapters.append(
            BookChapter(
                chapter_number=str(number) if number is not None and str(number).strip() else str(index),
                chapter_title=title,
                printed_start_page=printed_start_page,
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

    # Optional schema for books that do not call lessons "chapters". Keep
    # chapter_number/chapter_title NULL and store unit/section instead.
    for index, raw in enumerate(data.get("sections") or [], start=1):
        if not isinstance(raw, dict):
            continue
        printed_start_page = _int_or_none(raw.get("printed_start_page") or raw.get("start_page"))
        section_title = _clean_chapter_title_value(
            raw.get("section_title") or raw.get("lesson_title") or raw.get("title") or raw.get("name"),
            chapter_number=None,
            printed_start_page=printed_start_page,
        )
        if not section_title:
            continue
        chapters.append(
            BookChapter(
                chapter_number=None,
                chapter_title=None,
                unit_number=_clean_title(raw.get("unit_number")),
                unit_title=_clean_title(raw.get("unit_title")),
                section_number=_clean_title(raw.get("section_number") or str(index)),
                section_title=section_title,
                lesson_title=_clean_title(raw.get("lesson_title")),
                structure_type="section",
                printed_start_page=printed_start_page,
                printed_end_page=_int_or_none(raw.get("printed_end_page")),
                pdf_start_page=_int_or_none(raw.get("pdf_start_page")),
                pdf_end_page=_int_or_none(raw.get("pdf_end_page")),
                confidence=_float_or_none(raw.get("confidence")) or _float_or_none(data.get("confidence")),
                detected_by=detected_by,
                metadata={k: v for k, v in raw.items() if k not in {
                    "unit_number", "unit_title", "section_number", "section_title",
                    "lesson_title", "title", "name", "printed_start_page", "start_page",
                    "printed_end_page", "pdf_start_page", "pdf_end_page", "confidence",
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
    # The name stays normalize_chapters for backward compatibility, but the list
    # can also contain section-level structures. A valid structure has at least
    # one display title. For section books, chapter fields are intentionally None.
    valid = [c for c in chapters if c.display_title]
    valid.sort(key=lambda c: (
        c.pdf_start_page if c.pdf_start_page is not None else 10**9,
        c.printed_start_page if c.printed_start_page is not None else 10**9,
        c.unit_number or "",
        _int_or_none(c.chapter_number) if _int_or_none(c.chapter_number) is not None else 10**9,
        _int_or_none(c.section_number) if _int_or_none(c.section_number) is not None else 10**9,
        c.display_title or "",
    ))
    for index, chapter in enumerate(valid):
        if not chapter.structure_type:
            chapter.structure_type = "section" if chapter.section_title and not chapter.chapter_title else "chapter"
        # Only auto-fill chapter_number for actual chapter records. For books like
        # Poorvi, chapter_number must remain NULL because structure is section-based.
        if chapter.structure_type == "chapter" and chapter.chapter_title and not chapter.chapter_number:
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
    cloned = [BookChapter(**c.to_dict()) for c in (chapters or []) if c.display_title]

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
        found = _find_title_start_page(chapter.display_title or "", page_text_by_number, min_page=scan_from)
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
            if _normalize_for_match(chapter.display_title or "") in {"answer", "answers"}:
                if chapter.pdf_start_page is None or answers_start < chapter.pdf_start_page:
                    chapter.metadata["original_pdf_start_page"] = chapter.pdf_start_page
                    chapter.pdf_start_page = answers_start
                    chapter.metadata["pdf_start_page_source"] = "answers_page_heuristic"

    # Step 2: determine offset between printed page and PDF page.
    #
    # For section-based books such as NCERT Poorvi, headings can repeat later in
    # transcript/listening pages. A pure title scan may therefore map early
    # sections to later transcript pages (for example, A Bottle of Dew -> page 52
    # instead of page 17). Before using the generic offset inference, detect the
    # dominant printed->PDF offset from near-top heading matches and force that
    # offset for all unit/section records. This path is section-only, so it does
    # not affect chapter-based books such as the maths book.
    section_offset = _infer_section_printed_to_pdf_offset(chapters, page_text_by_number, max_page)
    if section_offset is not None:
        for chapter in chapters:
            if chapter.structure_type != "section" or chapter.printed_start_page is None:
                continue
            calculated_pdf = chapter.printed_start_page + section_offset
            if 1 <= calculated_pdf <= max_page:
                if chapter.pdf_start_page != calculated_pdf:
                    chapter.metadata["original_pdf_start_page"] = chapter.pdf_start_page
                    chapter.metadata["pdf_start_page_repaired_by_section_offset"] = True
                chapter.pdf_start_page = calculated_pdf
                chapter.metadata["pdf_start_page_source"] = "section_printed_page_offset"
                chapter.metadata["printed_to_pdf_offset"] = section_offset

    # Best anchor: chapter 1 starts at printed page 1.
    offset = section_offset if section_offset is not None else _infer_pdf_to_printed_offset(chapters)

    # Step 3: if offset is known, map missing pdf pages from printed pages and repair printed pages.
    if offset is not None:
        for chapter in chapters:
            if chapter.printed_start_page is None:
                continue

            calculated_pdf = chapter.printed_start_page + offset
            if not (1 <= calculated_pdf <= max_page):
                continue

            candidate_text = page_text_by_number.get(calculated_pdf, "") or ""

            # Prefer TOC printed-page + offset when that PDF page really starts with the chapter title.
            # This fixes cases like:
            # printed page 84 + offset 7 = PDF page 91 -> Exponents
            candidate_matches_title = _page_head_matches_title(
                chapter.display_title or "",
                candidate_text,
            )

            if chapter.pdf_start_page is None:
                chapter.pdf_start_page = calculated_pdf
                chapter.metadata["pdf_start_page_source"] = "printed_page_offset"
                chapter.metadata["printed_to_pdf_offset"] = offset
                continue

            # If title scan found a nearby but wrong page, correct it using TOC offset.
            if (
                    candidate_matches_title
                    and abs((chapter.pdf_start_page or calculated_pdf) - calculated_pdf) <= 3
                    and chapter.pdf_start_page != calculated_pdf
            ):
                chapter.metadata["original_pdf_start_page"] = chapter.pdf_start_page
                chapter.metadata["pdf_start_page_repaired_by_toc_offset"] = True
                chapter.pdf_start_page = calculated_pdf
                chapter.metadata["pdf_start_page_source"] = "printed_page_offset_verified_by_heading"
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

def _page_head_matches_title(title: str, text: str) -> bool:
    title_norm = _normalize_for_match(title)
    if not title_norm or not text:
        return False

    head = "\n".join(text.splitlines()[:15])[:1200]
    head_norm = _normalize_for_match(head)
    head_norm_without_leading_number = re.sub(r"^\d+\s+", "", head_norm)

    for line in head.splitlines()[:12]:
        line_norm = _normalize_for_match(line)
        if line_norm == title_norm:
            return True
        if _is_close_heading(line_norm, title_norm):
            return True

    title_pos = head_norm.find(title_norm)
    title_pos_without_number = head_norm_without_leading_number.find(title_norm)

    return (
        head_norm.startswith(title_norm + " ")
        or head_norm == title_norm
        or head_norm_without_leading_number.startswith(title_norm + " ")
        or head_norm_without_leading_number == title_norm
        or 0 <= title_pos <= 80
        or 0 <= title_pos_without_number <= 80
    )

def _has_title(chapters: list[BookChapter], title: str) -> bool:
    wanted = _normalize_for_match(title)
    return any(_normalize_for_match(c.display_title or "") == wanted for c in chapters)


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
    indexed = [(idx, c) for idx, c in enumerate(chapters) if c.display_title]

    def key(item: tuple[int, BookChapter]) -> tuple[int, int, int]:
        idx, chapter = item
        num = _chapter_number_int(chapter)
        # Numbered chapters should follow chapter-number order; unnumbered sections keep
        # their original order after numbered chapters.
        return (0 if num is not None else 1, num if num is not None else 10**9, idx)

    return [c for _, c in sorted(indexed, key=key)]


def _infer_section_printed_to_pdf_offset(
    chapters: list[BookChapter],
    page_text_by_number: dict[int, str],
    max_page: int,
) -> int | None:
    """Infer ``pdf_page = printed_page + offset`` for Unit/Section books.

    Section titles in English literature books often appear again later in
    transcript pages. The generic monotonic title scan can accidentally choose
    those later occurrences. This function looks for section titles very near
    the top of pages, records all plausible offsets, and chooses the dominant
    offset. For Poorvi, it produces 16 because printed page 1 is PDF page 17.
    """
    section_records = [
        c for c in chapters
        if c.structure_type == "section" and c.printed_start_page is not None and c.display_title
    ]
    if len(section_records) < 3:
        return None

    offsets: list[int] = []
    for chapter in sorted(section_records, key=lambda c: c.printed_start_page or 10**9):
        title = chapter.display_title or ""
        printed = chapter.printed_start_page
        if printed is None:
            continue

        first_matching_page: int | None = None
        for page_number in sorted(page_text_by_number):
            text = page_text_by_number.get(page_number, "") or ""
            if not text.strip() or _is_probable_contents_page(text):
                continue
            # Ignore pages before the printed page could possibly occur.
            # This keeps the search cheap and prevents front-matter noise.
            if page_number < printed:
                continue
            if _page_head_matches_title(title, text):
                first_matching_page = page_number
                break

        if first_matching_page is None:
            continue

        offset = first_matching_page - printed
        # School textbook front matter is normally far below 100 PDF pages.
        # Keep the bound loose enough for large prelim sections but tight enough
        # to reject transcript matches far later in the book.
        if 0 <= offset <= min(120, max_page):
            offsets.append(offset)

    if len(offsets) < 3:
        return None

    counts: dict[int, int] = {}
    for offset in offsets:
        counts[offset] = counts.get(offset, 0) + 1

    best_offset, best_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    # Require enough support so one accidental heading does not override a whole
    # book. For 16-section Poorvi, offset 16 gets broad support.
    if best_count >= max(3, int(len(offsets) * 0.40)):
        for c in chapters:
            c.metadata.setdefault("printed_to_pdf_offset_source", "section_heading_offset_mode")
        return best_offset

    return None


def _infer_pdf_to_printed_offset(chapters: list[BookChapter]) -> int | None:
    """Return offset such that pdf_page = printed_page + offset."""
    mapped = [c for c in chapters if c.pdf_start_page is not None]
    if not mapped:
        return None

    # Most reliable for school textbooks: chapter 1 body starts at printed page 1.
    chapter_one_candidates = [c for c in mapped if _chapter_number_int(c) == 1 and c.chapter_title]
    if chapter_one_candidates:
        first = min(chapter_one_candidates, key=lambda c: c.pdf_start_page or 10**9)
        offset = (first.pdf_start_page or 0) - 1
        if offset >= 0:
            for c in chapters:
                c.metadata.setdefault("printed_to_pdf_offset_source", "chapter_1_anchor")
            return offset

    # Section-based books such as NCERT Poorvi do not have chapters. Their first
    # lesson/section normally starts on printed page 1. Use that as the anchor.
    first_printed_page_candidates = [c for c in mapped if c.printed_start_page == 1]
    if first_printed_page_candidates:
        first = min(first_printed_page_candidates, key=lambda c: c.pdf_start_page or 10**9)
        offset = (first.pdf_start_page or 0) - 1
        if offset >= 0:
            for c in chapters:
                c.metadata.setdefault("printed_to_pdf_offset_source", "printed_page_1_structure_anchor")
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

        # Never allow a Contents/Index page to become a chapter start. This was
        # the reason page 6 (Contents) was being tagged as Chapter 1 and the
        # printed page number became PDF page - 5 instead of PDF page - 7.
        if _is_probable_contents_page(text):
            continue

        head = "\n".join(text.splitlines()[:12])[:1100]
        head_norm = _normalize_for_match(head)
        head_norm_without_leading_number = re.sub(r"^\d+\s+", "", head_norm)

        # Exact line/heading at top gets high confidence.
        for line in head.splitlines()[:10]:
            line_norm = _normalize_for_match(line)
            if not line_norm:
                continue
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



def _is_probable_contents_page(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\b(contents|table of contents|index)\b", lower) or "अनुक्रमणिका" in text or "विषय सूची" in text:
        return True

    # Some OCR layers miss the word Contents but contain many TOC-like rows.
    lines = [" ".join(line.strip().split()) for line in text.splitlines() if line.strip()]
    toc_like = 0
    for line in lines[:80]:
        if re.match(r"^\d{1,2}\s*[.)\-–:,]?\s+.{3,80}\s+\d{1,4}$", line):
            toc_like += 1
    return toc_like >= 6


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
            unit_number=chapter.unit_number,
            unit_title=chapter.unit_title,
            lesson_title=chapter.lesson_title,
            section_number=chapter.section_number,
            section_title=chapter.section_title,
            topic=chapter.section_title or chapter.lesson_title or chapter.chapter_title,
        )
