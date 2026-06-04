from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingestion.book_structure import BookChapter, BookStructure, normalize_chapters
from ingestion.language_detector import LanguageDetector
from ingestion.pdf_extractor import ExtractedPage
from ingestion.structure_detector import StructureDetector
from ingestion.text_cleaner import TextCleaner
from utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JsonInputDocument:
    metadata: dict[str, Any]
    pages: list[ExtractedPage]
    book_structure: BookStructure
    warnings: list[str]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None




def _normalize_document_key(value: Any) -> str | None:
    text = _clean_text_value(value)
    if not text:
        return None
    # Keep the key readable but safe for CLI/DB usage. Unicode letters/numbers are preserved.
    text = re.sub(r"[^\w.-]+", "-", text.lower(), flags=re.UNICODE).strip("-._")
    text = re.sub(r"-+", "-", text)
    return text or None


def _first_metadata_value(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value is None or value == "":
            continue
        return value
    return None


def _apply_metadata_aliases(metadata: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "document_key": ["document_key", "documentKey", "book_key", "bookKey", "source_document_key"],
        "school_name": ["school_name", "schoolName", "school", "institution_name", "institution"],
        "class_name": ["class_name", "className", "class", "standard", "grade_name"],
        "grade": ["grade", "grade_name", "class_grade", "class", "standard"],
        "subject": ["subject", "subject_name", "subjectName"],
        "board": ["board", "curriculum", "curriculum_board"],
        "medium": ["medium", "instruction_medium", "medium_of_instruction"],
        "language": ["language", "primary_language", "declared_language"],
        "book_title": ["book_title", "bookTitle", "textbook_name", "textbook", "title"],
        "title": ["title", "book_title", "bookTitle", "textbook_name", "textbook"],
    }
    for target, keys in aliases.items():
        if not metadata.get(target):
            value = _first_metadata_value(metadata, *keys)
            if value is not None:
                metadata[target] = value
    metadata["document_key"] = _normalize_document_key(metadata.get("document_key"))
    return metadata


def _derive_document_key(metadata: dict[str, Any], json_path: Path) -> str:
    parts = [
        metadata.get("school_name"),
        metadata.get("class_name") or metadata.get("grade"),
        metadata.get("subject"),
        metadata.get("book_title") or metadata.get("title") or json_path.stem,
    ]
    key = _normalize_document_key("-".join(str(p) for p in parts if p))
    return key or _normalize_document_key(json_path.stem) or "json-document"


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


def _first_text(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _page_range(start: int | None, end: int | None, fallback_page: int | None = None) -> list[int]:
    if start is None and fallback_page is not None:
        start = fallback_page
    if end is None:
        end = start
    if start is None:
        return []
    if end is None or end < start:
        end = start
    return list(range(start, end + 1))


def _text_fields(raw: dict[str, Any]) -> str | None:
    return _first_text(
        raw,
        "lesson_text",
        "chapter_text",
        "section_text",
        "text",
        "content",
        "cleaned_text",
        "raw_text",
    )


class JsonExtractionInputLoader:
    """Converts an external extraction JSON into the pipeline's internal objects.

    The loader intentionally accepts both the pipeline's own
    *_combined_extraction.json format and a smaller hand-authored JSON format.
    It does not accept embeddings; embeddings are generated later by the normal
    OpenAI embedding flow after pages are converted into chunks.
    """

    MATH_SYMBOL_RE = re.compile(r"[=+×÷≤≥<>√π%−–—]|\d+\s*/\s*\d+|\b\d+(?:\.\d+)?\b")
    C1_CONTROL_RE = re.compile(r"[\x80-\x9F]")

    def __init__(self, cleaner: TextCleaner, language_detector: LanguageDetector, token_counter: TokenCounter) -> None:
        self.cleaner = cleaner
        self.language_detector = language_detector
        self.token_counter = token_counter
        self.structure_detector = StructureDetector()

    def load(self, json_path: Path, cli_metadata: dict[str, Any] | None = None) -> JsonInputDocument:
        raw_payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("Input JSON must be a JSON object at the root.")

        cli_metadata = cli_metadata or {}
        extraction = _as_dict(raw_payload.get("extraction")) or raw_payload
        root_metadata = _as_dict(raw_payload.get("metadata"))
        root_direct_metadata = self._metadata_from_extraction(raw_payload)
        extraction_metadata = self._metadata_from_extraction(extraction)
        metadata = {
            **root_direct_metadata,
            **root_metadata,
            **extraction_metadata,
            **{k: v for k, v in cli_metadata.items() if v is not None},
        }
        metadata = _apply_metadata_aliases(metadata)
        document_key_was_provided = bool(metadata.get("document_key"))
        metadata["title"] = metadata.get("title") or metadata.get("book_title") or extraction.get("book_title") or json_path.stem
        metadata["book_title"] = metadata.get("book_title") or metadata.get("title")
        metadata["grade"] = metadata.get("grade") or metadata.get("class_name")
        metadata["class_name"] = metadata.get("class_name") or metadata.get("grade")
        metadata["document_key"] = metadata.get("document_key") or _derive_document_key(metadata, json_path)
        metadata["source_json_path"] = str(json_path.resolve())
        metadata["source_json_format"] = "combined_extraction" if raw_payload.get("extraction") else "direct_json_extraction"
        if raw_payload.get("source_pdf") and not metadata.get("source_uri"):
            metadata["source_uri"] = str(raw_payload.get("source_pdf"))
        if raw_payload.get("file_hash"):
            metadata["source_file_hash"] = str(raw_payload.get("file_hash"))
        if raw_payload.get("pdf_for_extraction"):
            metadata["pdf_for_extraction"] = str(raw_payload.get("pdf_for_extraction"))

        warnings: list[str] = []
        if not document_key_was_provided:
            warnings.append(f"document_key was not supplied; derived document_key={metadata['document_key']!r} from metadata.")
        book_structure = self._build_book_structure(extraction, metadata)
        pages = self._build_pages(extraction, book_structure, warnings)
        if not pages:
            raise ValueError(
                "JSON input did not contain usable text. Provide extraction.page_extractions, pages, "
                "or chapter/section lesson_text/text fields."
            )

        if not book_structure.chapters:
            warnings.append("No chapter/section structure found in JSON. Ingestion will be page-based only.")
        if not metadata.get("languages_detected"):
            detected = sorted({p.detected_language for p in pages if p.detected_language and p.detected_language != "Unknown"})
            metadata["languages_detected"] = detected
        if not metadata.get("language") and metadata.get("languages_detected"):
            metadata["language"] = "Mixed" if len(metadata["languages_detected"]) > 1 else metadata["languages_detected"][0]

        return JsonInputDocument(
            metadata=metadata,
            pages=pages,
            book_structure=book_structure,
            warnings=warnings,
        )

    def _metadata_from_extraction(self, extraction: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "document_key",
            "documentKey",
            "book_key",
            "title",
            "book_title",
            "school_name",
            "schoolName",
            "school",
            "class_name",
            "className",
            "class",
            "standard",
            "grade",
            "subject",
            "subject_name",
            "subjectName",
            "board",
            "medium",
            "language",
            "publisher",
            "author",
            "isbn",
            "edition",
            "publication_year",
            "copyright_status",
            "license_notes",
            "source_uri",
            "content_profile",
        ]
        return {key: extraction.get(key) for key in keys if extraction.get(key) is not None}

    def _build_book_structure(self, extraction: dict[str, Any], metadata: dict[str, Any]) -> BookStructure:
        chapters: list[BookChapter] = []
        chapters.extend(self._chapter_records(_as_list(extraction.get("chapters"))))
        chapters.extend(self._section_records(_as_list(extraction.get("sections"))))
        chapters.extend(self._section_records(_as_list(extraction.get("content_units"))))

        languages = extraction.get("languages_detected") or metadata.get("languages_detected") or []
        if isinstance(languages, str):
            languages = [x.strip() for x in languages.split(",") if x.strip()]

        recommended = _as_dict(extraction.get("chunking_plan"))
        return BookStructure(
            book_title=_clean_text_value(extraction.get("book_title") or metadata.get("book_title") or metadata.get("title")),
            subject=_clean_text_value(extraction.get("subject") or metadata.get("subject")),
            grade=_clean_text_value(extraction.get("grade") or metadata.get("grade") or metadata.get("class_name")),
            primary_language=_clean_text_value(extraction.get("language") or metadata.get("language")),
            languages_detected=[str(x) for x in languages if str(x).strip()],
            publisher=_clean_text_value(metadata.get("publisher")),
            author=_clean_text_value(metadata.get("author")),
            isbn=_clean_text_value(metadata.get("isbn")),
            edition=_clean_text_value(metadata.get("edition")),
            publication_year=_clean_text_value(metadata.get("publication_year")),
            content_profile=_clean_text_value(extraction.get("content_profile") or metadata.get("content_profile") or recommended.get("content_profile")),
            recommended_chunk_max_tokens=_int_or_none(recommended.get("max_tokens") or extraction.get("recommended_chunk_max_tokens")),
            recommended_chunk_overlap_tokens=_int_or_none(recommended.get("overlap_tokens") or extraction.get("recommended_chunk_overlap_tokens")),
            recommended_chunking_strategy=_clean_text_value(recommended.get("strategy") or extraction.get("recommended_chunking_strategy")),
            confidence=_float_or_none(extraction.get("confidence") or metadata.get("llm_metadata_confidence")) or 1.0,
            detected_by="json_input",
            chapters=normalize_chapters(chapters),
            metadata={"source": "json_input"},
        )

    def _chapter_records(self, items: list[Any]) -> list[BookChapter]:
        records: list[BookChapter] = []
        for index, item in enumerate(items, start=1):
            raw = _as_dict(item)
            if not raw:
                continue
            chapter_number = _clean_text_value(raw.get("chapter_number") or raw.get("number") or raw.get("chapter_id") or index)
            chapter_title = _clean_text_value(raw.get("chapter_title") or raw.get("title") or raw.get("name"))
            child_sections = _as_list(raw.get("sections")) or _as_list(raw.get("lessons"))

            if child_sections:
                # Prefer child section ranges over a broad parent chapter range so page/chunk
                # metadata can be section-specific while still preserving parent chapter data.
                for child_index, child in enumerate(child_sections, start=1):
                    child_raw = _as_dict(child)
                    if not child_raw:
                        continue
                    section_title = _clean_text_value(
                        child_raw.get("section_title")
                        or child_raw.get("lesson_title")
                        or child_raw.get("title")
                        or child_raw.get("name")
                    )
                    if not (chapter_title or section_title):
                        continue
                    records.append(
                        BookChapter(
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            unit_number=_clean_text_value(raw.get("unit_number") or child_raw.get("unit_number")),
                            unit_title=_clean_text_value(raw.get("unit_title") or child_raw.get("unit_title")),
                            section_number=_clean_text_value(child_raw.get("section_number") or child_raw.get("number") or child_index),
                            section_title=section_title,
                            lesson_title=_clean_text_value(child_raw.get("lesson_title")),
                            structure_type="section" if section_title else "chapter",
                            printed_start_page=_int_or_none(child_raw.get("printed_start_page") or child_raw.get("start_printed_page")),
                            printed_end_page=_int_or_none(child_raw.get("printed_end_page") or child_raw.get("end_printed_page")),
                            pdf_start_page=_int_or_none(child_raw.get("start_page") or child_raw.get("pdf_start_page")),
                            pdf_end_page=_int_or_none(child_raw.get("end_page") or child_raw.get("pdf_end_page")),
                            confidence=_float_or_none(child_raw.get("confidence") or raw.get("confidence")) or 1.0,
                            detected_by="json_input",
                            metadata=self._extra_structure_metadata(child_raw),
                        )
                    )
                continue

            if not chapter_title:
                continue
            records.append(
                BookChapter(
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    unit_number=_clean_text_value(raw.get("unit_number")),
                    unit_title=_clean_text_value(raw.get("unit_title")),
                    section_number=_clean_text_value(raw.get("section_number")),
                    section_title=_clean_text_value(raw.get("section_title")),
                    lesson_title=_clean_text_value(raw.get("lesson_title")),
                    structure_type=_clean_text_value(raw.get("structure_type")) or "chapter",
                    printed_start_page=_int_or_none(raw.get("printed_start_page") or raw.get("start_printed_page")),
                    printed_end_page=_int_or_none(raw.get("printed_end_page") or raw.get("end_printed_page")),
                    pdf_start_page=_int_or_none(raw.get("start_page") or raw.get("pdf_start_page")),
                    pdf_end_page=_int_or_none(raw.get("end_page") or raw.get("pdf_end_page")),
                    confidence=_float_or_none(raw.get("confidence")) or 1.0,
                    detected_by="json_input",
                    metadata=self._extra_structure_metadata(raw),
                )
            )
        return records

    def _section_records(self, items: list[Any]) -> list[BookChapter]:
        records: list[BookChapter] = []
        for group_index, item in enumerate(items, start=1):
            raw = _as_dict(item)
            if not raw:
                continue
            lessons = _as_list(raw.get("lessons")) or _as_list(raw.get("sections"))
            if lessons:
                for lesson_index, lesson in enumerate(lessons, start=1):
                    lesson_raw = _as_dict(lesson)
                    if not lesson_raw:
                        continue
                    title = _clean_text_value(
                        lesson_raw.get("section_title")
                        or lesson_raw.get("lesson_title")
                        or lesson_raw.get("title")
                        or lesson_raw.get("name")
                    )
                    if not title:
                        continue
                    records.append(
                        BookChapter(
                            chapter_number=_clean_text_value(lesson_raw.get("chapter_number") or raw.get("chapter_number")),
                            chapter_title=_clean_text_value(lesson_raw.get("chapter_title") or raw.get("chapter_title")),
                            unit_number=_clean_text_value(lesson_raw.get("unit_number") or raw.get("unit_number") or raw.get("chapter_number")),
                            unit_title=_clean_text_value(lesson_raw.get("unit_title") or raw.get("unit_title") or raw.get("chapter_title")),
                            section_number=_clean_text_value(lesson_raw.get("section_number") or lesson_index),
                            section_title=title,
                            lesson_title=_clean_text_value(lesson_raw.get("lesson_title")),
                            structure_type=_clean_text_value(lesson_raw.get("structure_type") or lesson_raw.get("lesson_type")) or "section",
                            printed_start_page=_int_or_none(lesson_raw.get("printed_start_page") or lesson_raw.get("start_printed_page")),
                            printed_end_page=_int_or_none(lesson_raw.get("printed_end_page") or lesson_raw.get("end_printed_page")),
                            pdf_start_page=_int_or_none(lesson_raw.get("start_page") or lesson_raw.get("pdf_start_page")),
                            pdf_end_page=_int_or_none(lesson_raw.get("end_page") or lesson_raw.get("pdf_end_page")),
                            confidence=_float_or_none(lesson_raw.get("confidence") or raw.get("confidence")) or 1.0,
                            detected_by="json_input",
                            metadata=self._extra_structure_metadata(lesson_raw),
                        )
                    )
                continue

            title = _clean_text_value(raw.get("section_title") or raw.get("lesson_title") or raw.get("title") or raw.get("name"))
            if not title:
                continue
            records.append(
                BookChapter(
                    chapter_number=_clean_text_value(raw.get("chapter_number")),
                    chapter_title=_clean_text_value(raw.get("chapter_title")),
                    unit_number=_clean_text_value(raw.get("unit_number") or raw.get("chapter_number")),
                    unit_title=_clean_text_value(raw.get("unit_title") or raw.get("chapter_title")),
                    section_number=_clean_text_value(raw.get("section_number") or group_index),
                    section_title=title,
                    lesson_title=_clean_text_value(raw.get("lesson_title")),
                    structure_type=_clean_text_value(raw.get("structure_type") or raw.get("lesson_type")) or "section",
                    printed_start_page=_int_or_none(raw.get("printed_start_page") or raw.get("start_printed_page")),
                    printed_end_page=_int_or_none(raw.get("printed_end_page") or raw.get("end_printed_page")),
                    pdf_start_page=_int_or_none(raw.get("start_page") or raw.get("pdf_start_page")),
                    pdf_end_page=_int_or_none(raw.get("end_page") or raw.get("pdf_end_page")),
                    confidence=_float_or_none(raw.get("confidence")) or 1.0,
                    detected_by="json_input",
                    metadata=self._extra_structure_metadata(raw),
                )
            )
        return records

    def _extra_structure_metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        ignored = {
            "chapter_number",
            "chapter_title",
            "number",
            "chapter_id",
            "unit_number",
            "unit_title",
            "section_number",
            "section_title",
            "lesson_title",
            "title",
            "name",
            "start_page",
            "end_page",
            "pdf_start_page",
            "pdf_end_page",
            "printed_start_page",
            "printed_end_page",
            "start_printed_page",
            "end_printed_page",
            "lesson_text",
            "chapter_text",
            "section_text",
            "text",
            "content",
            "cleaned_text",
            "raw_text",
            "lessons",
            "sections",
            "pages",
            "page_texts",
            "confidence",
        }
        metadata = {k: v for k, v in raw.items() if k not in ignored}
        metadata["source"] = "json_input"
        return metadata

    def _build_pages(
        self,
        extraction: dict[str, Any],
        book_structure: BookStructure,
        warnings: list[str],
    ) -> list[ExtractedPage]:
        explicit_page_items = (
            _as_list(extraction.get("page_extractions"))
            or _as_list(extraction.get("pages"))
            or _as_list(extraction.get("page_texts"))
        )
        if explicit_page_items:
            pages = self._pages_from_explicit_items(explicit_page_items)
            if pages:
                return pages
            warnings.append("page_extractions/pages were present but no usable page text was found; trying structure text.")

        page_texts: dict[int, list[str]] = {}
        page_metadata: dict[int, dict[str, Any]] = {}
        for item in _as_list(extraction.get("chapters")):
            self._collect_text_from_chapter(_as_dict(item), page_texts, page_metadata)
        for item in _as_list(extraction.get("sections")):
            self._collect_text_from_section_group(_as_dict(item), page_texts, page_metadata)
        for item in _as_list(extraction.get("content_units")):
            self._collect_text_from_section_group(_as_dict(item), page_texts, page_metadata)

        pages: list[ExtractedPage] = []
        for page_number in sorted(page_texts):
            raw_text = "\n\n".join(t.strip() for t in page_texts[page_number] if t and t.strip()).strip()
            if not raw_text:
                continue
            pages.append(self._build_page(page_number, raw_text, metadata=page_metadata.get(page_number)))
        return pages

    def _pages_from_explicit_items(self, items: list[Any]) -> list[ExtractedPage]:
        pages: list[ExtractedPage] = []
        for index, item in enumerate(items, start=1):
            raw = _as_dict(item)
            if not raw:
                continue
            page_number = _int_or_none(raw.get("page_number") or raw.get("pdf_page") or raw.get("page") or index)
            if page_number is None:
                continue
            text = _text_fields(raw)
            if text is None:
                continue
            pages.append(self._build_page(page_number, text, metadata={k: v for k, v in raw.items() if k not in {"text", "raw_text", "cleaned_text"}}))
        pages.sort(key=lambda p: p.page_number)
        return pages

    def _collect_text_from_chapter(
        self,
        raw: dict[str, Any],
        page_texts: dict[int, list[str]],
        page_metadata: dict[int, dict[str, Any]],
    ) -> None:
        if not raw:
            return
        chapter_title = _clean_text_value(raw.get("chapter_title") or raw.get("title") or raw.get("name"))
        direct_text = _text_fields(raw)
        if direct_text:
            start = _int_or_none(raw.get("start_page") or raw.get("pdf_start_page"))
            end = _int_or_none(raw.get("end_page") or raw.get("pdf_end_page"))
            self._add_text_to_pages(
                page_texts,
                page_metadata,
                page_numbers=_page_range(start, end),
                text=direct_text,
                label=chapter_title,
                structure_metadata={
                    "chapter_number": raw.get("chapter_number") or raw.get("number"),
                    "chapter_title": chapter_title,
                    "structure_type": "chapter",
                },
            )
        for child in _as_list(raw.get("sections")) or _as_list(raw.get("lessons")):
            child_raw = _as_dict(child)
            if child_raw:
                child_raw.setdefault("chapter_number", raw.get("chapter_number") or raw.get("number"))
                child_raw.setdefault("chapter_title", chapter_title)
                self._collect_text_from_section_group(child_raw, page_texts, page_metadata)

    def _collect_text_from_section_group(
        self,
        raw: dict[str, Any],
        page_texts: dict[int, list[str]],
        page_metadata: dict[int, dict[str, Any]],
    ) -> None:
        if not raw:
            return
        explicit_pages = _as_list(raw.get("pages")) or _as_list(raw.get("page_texts"))
        if explicit_pages:
            for page in explicit_pages:
                page_raw = _as_dict(page)
                page_number = _int_or_none(page_raw.get("page_number") or page_raw.get("pdf_page") or page_raw.get("page"))
                text = _text_fields(page_raw)
                if page_number is not None and text:
                    self._append_page_text(page_texts, page_metadata, page_number, text, page_raw)

        lessons = _as_list(raw.get("lessons")) or _as_list(raw.get("sections"))
        if lessons:
            for lesson in lessons:
                lesson_raw = _as_dict(lesson)
                if lesson_raw:
                    if raw.get("unit_number") and not lesson_raw.get("unit_number"):
                        lesson_raw["unit_number"] = raw.get("unit_number")
                    if raw.get("unit_title") and not lesson_raw.get("unit_title"):
                        lesson_raw["unit_title"] = raw.get("unit_title")
                    if raw.get("chapter_number") and not lesson_raw.get("chapter_number"):
                        lesson_raw["chapter_number"] = raw.get("chapter_number")
                    if raw.get("chapter_title") and not lesson_raw.get("chapter_title"):
                        lesson_raw["chapter_title"] = raw.get("chapter_title")
                    self._collect_text_from_section_group(lesson_raw, page_texts, page_metadata)
            return

        text = _text_fields(raw)
        if not text:
            return
        start = _int_or_none(raw.get("start_page") or raw.get("pdf_start_page"))
        end = _int_or_none(raw.get("end_page") or raw.get("pdf_end_page"))
        page_numbers = _as_list(raw.get("page_numbers"))
        parsed_page_numbers = [_int_or_none(p) for p in page_numbers]
        parsed_page_numbers = [p for p in parsed_page_numbers if p is not None]
        if not parsed_page_numbers:
            parsed_page_numbers = _page_range(start, end)
        title = _clean_text_value(raw.get("section_title") or raw.get("lesson_title") or raw.get("chapter_title") or raw.get("title") or raw.get("name"))
        self._add_text_to_pages(
            page_texts,
            page_metadata,
            page_numbers=parsed_page_numbers,
            text=text,
            label=title,
            structure_metadata={
                "chapter_number": raw.get("chapter_number"),
                "chapter_title": raw.get("chapter_title"),
                "unit_number": raw.get("unit_number"),
                "unit_title": raw.get("unit_title"),
                "section_number": raw.get("section_number"),
                "section_title": raw.get("section_title") or title,
                "lesson_title": raw.get("lesson_title"),
                "structure_type": raw.get("structure_type") or raw.get("lesson_type") or "section",
                "synthetic_page_from_json_range": True,
            },
        )

    def _add_text_to_pages(
        self,
        page_texts: dict[int, list[str]],
        page_metadata: dict[int, dict[str, Any]],
        *,
        page_numbers: list[int],
        text: str,
        label: str | None,
        structure_metadata: dict[str, Any],
    ) -> None:
        if not page_numbers:
            return
        pieces = self._split_text_across_pages(text, page_numbers)
        for page_number, piece in pieces.items():
            page_text = f"{label}\n{piece}" if label and label not in piece[:200] else piece
            self._append_page_text(page_texts, page_metadata, page_number, page_text, structure_metadata)

    def _append_page_text(
        self,
        page_texts: dict[int, list[str]],
        page_metadata: dict[int, dict[str, Any]],
        page_number: int,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        page_texts.setdefault(page_number, []).append(text)
        if metadata:
            page_metadata.setdefault(page_number, {}).update(metadata)

    def _split_text_across_pages(self, text: str, page_numbers: list[int]) -> dict[int, str]:
        page_numbers = sorted({int(p) for p in page_numbers})
        if not page_numbers:
            return {}
        if len(page_numbers) == 1:
            return {page_numbers[0]: text}

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < len(page_numbers):
            sentences = [p.strip() for p in re.split(r"(?<=[.!?।॥])\s+", text) if p.strip()]
            if len(sentences) >= len(page_numbers):
                paragraphs = sentences
        if len(paragraphs) < len(page_numbers):
            # Last-resort even character split. This is synthetic, but it preserves
            # the caller's start/end page range for downstream chunks and citations.
            chunk_size = max(1, len(text) // len(page_numbers))
            paragraphs = [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size)]

        buckets: list[list[str]] = [[] for _ in page_numbers]
        for idx, paragraph in enumerate(paragraphs):
            bucket_index = min(len(page_numbers) - 1, int(idx * len(page_numbers) / max(len(paragraphs), 1)))
            buckets[bucket_index].append(paragraph)
        return {
            page: "\n\n".join(bucket).strip()
            for page, bucket in zip(page_numbers, buckets)
            if "\n\n".join(bucket).strip()
        }

    def _build_page(self, page_number: int, raw_text: str, metadata: dict[str, Any] | None = None) -> ExtractedPage:
        clean = self.cleaner.clean(raw_text)
        text = clean.cleaned_text
        stats = self.language_detector.detect_with_stats(text)
        classification = self.structure_detector.classify(text)
        word_count = len(text.split()) if text else 0
        token_count = self.token_counter.count(text)
        has_text = bool(text.strip()) and token_count > 3
        garbage_count = len(self.C1_CONTROL_RE.findall(text)) + text.count("�")
        if not has_text:
            quality = "empty"
        elif garbage_count > 0:
            quality = "corrupted"
        elif token_count < 20:
            quality = "low"
        else:
            quality = "ok"
        page_metadata = {
            "source": "json_input",
            "cleaning_notes": clean.notes,
            "garbage_char_count": garbage_count,
        }
        if metadata:
            page_metadata.update(metadata)
        return ExtractedPage(
            page_number=page_number,
            raw_text=raw_text,
            cleaned_text=text,
            detected_language=stats.language,
            word_count=word_count,
            token_count=token_count,
            has_text=has_text,
            has_math=classification.flags.get("has_formula", False),
            has_table_like_text=classification.flags.get("has_table_like_text", False),
            has_devanagari=stats.devanagari_chars > 0,
            has_english=stats.latin_chars > 0,
            extraction_method="json_input",
            extraction_quality=quality,
            metadata=page_metadata,
        )
