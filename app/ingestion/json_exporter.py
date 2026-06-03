from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.book_structure import BookChapter, BookStructure, ChapterResolver, normalize_chapters
from ingestion.chunking_strategy import ChunkingPlan
from ingestion.pdf_extractor import ExtractedPage

logger = logging.getLogger(__name__)


def _safe_id(value: str | None, fallback: str) -> str:
    raw = value or fallback
    raw = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").upper()
    return raw or fallback.upper()


def _page_numbers(start: int | None, end: int | None, pages: list[ExtractedPage]) -> list[int]:
    if start is None or end is None:
        return []
    available = {p.page_number for p in pages}
    return [page_no for page_no in range(start, end + 1) if page_no in available]


def _overlaps(page_start: int | None, page_end: int | None, start: int | None, end: int | None) -> bool:
    if page_start is None or page_end is None or start is None or end is None:
        return False
    return page_start <= end and page_end >= start


def _chunk_title(chunk: dict[str, Any], index: int) -> str:
    for key in ("lesson_title", "section_title", "topic", "chapter_title", "unit_title"):
        value = chunk.get(key)
        if value:
            return str(value)
    chunk_type = str(chunk.get("chunk_type") or "content").replace("_", " ").title()
    return f"{chunk_type} {index}"


def _combine_page_text(pages: list[ExtractedPage], start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return ""
    return "\n\n".join(
        f"[Page {p.page_number}]\n{p.cleaned_text.strip()}"
        for p in pages
        if start <= p.page_number <= end and p.cleaned_text.strip()
    ).strip()


class ExtractionJsonExporter:
    """Builds a human-readable JSON artifact in the same spirit as combined_extraction.json.

    This is intentionally separate from DB persistence. The DB remains the source for
    RAG, while this JSON file is useful for checking book/chapter/section extraction,
    debugging page text, and sharing extraction samples with non-DB consumers.
    """

    def write_combined_extraction(
        self,
        *,
        output_dir: Path,
        original_pdf_path: Path,
        extraction_pdf_path: Path,
        metadata: dict[str, Any],
        pages: list[ExtractedPage],
        chunks: list[dict[str, Any]],
        book_structure: BookStructure,
        chunking_plan: ChunkingPlan,
        file_hash: str,
        warnings: list[str] | None = None,
        dry_run: bool = False,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{original_pdf_path.stem}_combined_extraction.json"
        payload = self.build_combined_extraction(
            original_pdf_path=original_pdf_path,
            extraction_pdf_path=extraction_pdf_path,
            metadata=metadata,
            pages=pages,
            chunks=chunks,
            book_structure=book_structure,
            chunking_plan=chunking_plan,
            file_hash=file_hash,
            warnings=warnings or [],
            dry_run=dry_run,
        )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("JSON extraction artifact written: %s", output_path)
        return output_path

    def build_combined_extraction(
        self,
        *,
        original_pdf_path: Path,
        extraction_pdf_path: Path,
        metadata: dict[str, Any],
        pages: list[ExtractedPage],
        chunks: list[dict[str, Any]],
        book_structure: BookStructure,
        chunking_plan: ChunkingPlan,
        file_hash: str,
        warnings: list[str],
        dry_run: bool,
    ) -> dict[str, Any]:
        resolver = ChapterResolver(book_structure.chapters) if book_structure.chapters else None
        has_real_chapters = any(c.chapter_title for c in book_structure.chapters)
        content_units = (
            self._build_chapter_units(pages, chunks, book_structure)
            if has_real_chapters
            else self._build_section_units(pages, chunks, book_structure)
        )

        extraction: dict[str, Any] = {
            "book_title": metadata.get("book_title") or metadata.get("title") or book_structure.book_title,
            "grade": metadata.get("grade") or book_structure.grade,
            "subject": metadata.get("subject") or book_structure.subject,
            "language": metadata.get("language") or book_structure.primary_language,
            "detected_book_structure": self._describe_structure(book_structure, has_real_chapters),
            "total_pages_observed": len(pages),
            "content_profile": book_structure.content_profile,
            "chunking_plan": chunking_plan.to_dict(),
            "page_extractions": self._build_page_extractions(pages, resolver),
            "global_extraction_notes": self._global_notes(book_structure, warnings),
        }
        if has_real_chapters:
            extraction["chapters"] = content_units
        else:
            extraction["sections"] = content_units

        return {
            "source_pdf": str(original_pdf_path),
            "pdf_for_extraction": str(extraction_pdf_path),
            "file_hash": file_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "total_original_pages": len(pages),
            "model": metadata.get("llm_metadata_model"),
            "metadata": {
                "school_name": metadata.get("school_name"),
                "class_name": metadata.get("class_name"),
                "board": metadata.get("board"),
                "medium": metadata.get("medium"),
                "publisher": metadata.get("publisher") or book_structure.publisher,
                "author": metadata.get("author") or book_structure.author,
                "isbn": metadata.get("isbn") or book_structure.isbn,
                "edition": metadata.get("edition") or book_structure.edition,
                "publication_year": metadata.get("publication_year") or book_structure.publication_year,
                "structure_detected_by": book_structure.detected_by,
                "llm_metadata_confidence": book_structure.confidence,
            },
            "warnings": warnings,
            "extraction": extraction,
        }

    def _build_chapter_units(
        self,
        pages: list[ExtractedPage],
        chunks: list[dict[str, Any]],
        book_structure: BookStructure,
    ) -> list[dict[str, Any]]:
        chapters = [c for c in normalize_chapters(book_structure.chapters) if c.chapter_title]
        units: list[dict[str, Any]] = []
        for idx, chapter in enumerate(chapters, start=1):
            start = chapter.pdf_start_page
            end = chapter.pdf_end_page or chapter.pdf_start_page
            matching_chunks = [
                c for c in chunks
                if _overlaps(c.get("page_start"), c.get("page_end"), start, end)
                and (not chapter.chapter_title or c.get("chapter_title") == chapter.chapter_title or not c.get("chapter_title"))
            ]
            if not matching_chunks and start and end:
                matching_chunks = [c for c in chunks if _overlaps(c.get("page_start"), c.get("page_end"), start, end)]

            lessons = [self._lesson_from_chunk(ch, lesson_index) for lesson_index, ch in enumerate(matching_chunks, start=1)]
            if not lessons and start and end:
                lessons.append(self._lesson_from_pages(pages, chapter, 1))

            chapter_id = f"CH{int(chapter.chapter_number):02d}" if str(chapter.chapter_number or "").isdigit() else _safe_id(chapter.chapter_number, f"CH{idx:02d}")
            units.append(
                {
                    "chapter_id": chapter_id,
                    "container_type": "chapter",
                    "chapter_number": chapter.chapter_number,
                    "chapter_title": chapter.chapter_title,
                    "unit_number": chapter.unit_number,
                    "unit_title": chapter.unit_title,
                    "start_page": start,
                    "end_page": end,
                    "printed_start_page": chapter.printed_start_page,
                    "printed_end_page": chapter.printed_end_page,
                    "page_numbers": _page_numbers(start, end, pages),
                    "lessons": lessons,
                    "chapter_summary": self._summarize_unit(chapter, lessons),
                    "confidence": chapter.confidence,
                    "extraction_notes": self._structure_notes(chapter),
                }
            )
        return units

    def _build_section_units(
        self,
        pages: list[ExtractedPage],
        chunks: list[dict[str, Any]],
        book_structure: BookStructure,
    ) -> list[dict[str, Any]]:
        sections = [c for c in normalize_chapters(book_structure.chapters) if c.display_title]
        grouped: dict[tuple[str | None, str | None], list[BookChapter]] = {}
        for section in sections:
            key = (section.unit_number, section.unit_title or section.chapter_title or "Sections")
            grouped.setdefault(key, []).append(section)

        units: list[dict[str, Any]] = []
        for unit_index, ((unit_number, unit_title), unit_sections) in enumerate(grouped.items(), start=1):
            starts = [s.pdf_start_page for s in unit_sections if s.pdf_start_page]
            ends = [s.pdf_end_page for s in unit_sections if s.pdf_end_page]
            start = min(starts) if starts else None
            end = max(ends) if ends else start
            lessons = []
            for lesson_index, section in enumerate(unit_sections, start=1):
                matching_chunks = self._matching_chunks_for_section(chunks, section)
                if matching_chunks:
                    lesson_text = "\n\n".join(str(c.get("content_clean") or c.get("content") or "").strip() for c in matching_chunks).strip()
                    lesson_type = matching_chunks[0].get("chunk_type") or section.structure_type
                    key_terms = self._merge_terms(matching_chunks)
                else:
                    lesson_text = _combine_page_text(pages, section.pdf_start_page, section.pdf_end_page)
                    lesson_type = section.structure_type
                    key_terms = []
                lessons.append(
                    {
                        "lesson_id": f"{_safe_id(unit_number or unit_title, f'U{unit_index:02d}')}-S{lesson_index:02d}",
                        "lesson_title": section.lesson_title or section.section_title or section.display_title,
                        "lesson_type": lesson_type,
                        "parent_container_type": "unit" if unit_title else "section_group",
                        "unit_number": section.unit_number,
                        "unit_title": section.unit_title,
                        "chapter_number": section.chapter_number or section.unit_number,
                        "chapter_title": section.chapter_title or section.unit_title,
                        "section_number": section.section_number,
                        "section_title": section.section_title or section.display_title,
                        "start_page": section.pdf_start_page,
                        "end_page": section.pdf_end_page,
                        "printed_start_page": section.printed_start_page,
                        "printed_end_page": section.printed_end_page,
                        "page_numbers": _page_numbers(section.pdf_start_page, section.pdf_end_page, pages),
                        "lesson_text": lesson_text,
                        "key_terms": key_terms,
                        "learning_objectives": [],
                        "exercises_or_questions_summary": None,
                        "extraction_notes": self._structure_notes(section),
                    }
                )

            units.append(
                {
                    "section_group_id": _safe_id(unit_number or unit_title, f"U{unit_index:02d}"),
                    "container_type": "unit" if unit_title else "section_group",
                    "chapter_number": unit_number,
                    "chapter_title": unit_title,
                    "unit_number": unit_number,
                    "unit_title": unit_title,
                    "start_page": start,
                    "end_page": end,
                    "page_numbers": _page_numbers(start, end, pages),
                    "lessons": lessons,
                    "section_summary": self._summarize_lessons(lessons),
                    "confidence": self._average_confidence(unit_sections),
                }
            )
        return units

    def _matching_chunks_for_section(self, chunks: list[dict[str, Any]], section: BookChapter) -> list[dict[str, Any]]:
        matches = []
        for chunk in chunks:
            if not _overlaps(chunk.get("page_start"), chunk.get("page_end"), section.pdf_start_page, section.pdf_end_page):
                continue
            if section.unit_title and chunk.get("unit_title") and chunk.get("unit_title") != section.unit_title:
                continue
            if section.section_title and chunk.get("section_title") and chunk.get("section_title") != section.section_title:
                continue
            matches.append(chunk)
        return matches

    def _lesson_from_chunk(self, chunk: dict[str, Any], lesson_index: int) -> dict[str, Any]:
        lesson_title = _chunk_title(chunk, lesson_index)
        return {
            "lesson_id": f"L{lesson_index:03d}",
            "lesson_title": lesson_title,
            "lesson_type": chunk.get("chunk_type"),
            "parent_container_type": "chapter",
            "chapter_number": chunk.get("chapter_number"),
            "chapter_title": chunk.get("chapter_title"),
            "unit_number": chunk.get("unit_number"),
            "unit_title": chunk.get("unit_title"),
            "section_number": chunk.get("section_number"),
            "section_title": chunk.get("section_title"),
            "start_page": chunk.get("page_start"),
            "end_page": chunk.get("page_end"),
            "page_numbers": list(range(int(chunk.get("page_start") or 0), int(chunk.get("page_end") or 0) + 1)) if chunk.get("page_start") and chunk.get("page_end") else [],
            "lesson_text": chunk.get("content_clean") or chunk.get("content"),
            "key_terms": (chunk.get("important_terms") or [])[:20],
            "learning_objectives": [],
            "exercises_or_questions_summary": None,
            "extraction_notes": f"Generated from chunk_index={chunk.get('chunk_index')} and source_label={chunk.get('source_label')}.",
        }

    def _lesson_from_pages(self, pages: list[ExtractedPage], chapter: BookChapter, lesson_index: int) -> dict[str, Any]:
        return {
            "lesson_id": f"L{lesson_index:03d}",
            "lesson_title": chapter.display_title,
            "lesson_type": chapter.structure_type,
            "parent_container_type": chapter.structure_type,
            "chapter_number": chapter.chapter_number,
            "chapter_title": chapter.chapter_title,
            "unit_number": chapter.unit_number,
            "unit_title": chapter.unit_title,
            "section_number": chapter.section_number,
            "section_title": chapter.section_title,
            "start_page": chapter.pdf_start_page,
            "end_page": chapter.pdf_end_page,
            "page_numbers": _page_numbers(chapter.pdf_start_page, chapter.pdf_end_page, pages),
            "lesson_text": _combine_page_text(pages, chapter.pdf_start_page, chapter.pdf_end_page),
            "key_terms": [],
            "learning_objectives": [],
            "exercises_or_questions_summary": None,
            "extraction_notes": "Generated directly from extracted page text because no chunk matched this structure range.",
        }

    def _build_page_extractions(self, pages: list[ExtractedPage], resolver: ChapterResolver | None) -> list[dict[str, Any]]:
        output = []
        for page in pages:
            resolved = resolver.chapter_for_pdf_page(page.page_number).to_dict() if resolver and resolver.chapter_for_pdf_page(page.page_number) else {}
            output.append(
                {
                    "page_number": page.page_number,
                    "printed_page_number": resolver.printed_page_for_pdf_page(page.page_number) if resolver else None,
                    "chapter_number": resolved.get("chapter_number"),
                    "chapter_title": resolved.get("chapter_title"),
                    "unit_number": resolved.get("unit_number"),
                    "unit_title": resolved.get("unit_title"),
                    "lesson_title": resolved.get("lesson_title"),
                    "section_number": resolved.get("section_number"),
                    "section_title": resolved.get("section_title"),
                    "structure_type": resolved.get("structure_type"),
                    "text": page.cleaned_text,
                    "word_count": page.word_count,
                    "token_count": page.token_count,
                    "detected_language": page.detected_language,
                    "extraction_method": page.extraction_method,
                    "extraction_quality": page.extraction_quality,
                    "has_text": page.has_text,
                    "has_math": page.has_math,
                    "has_table_like_text": page.has_table_like_text,
                    "metadata": page.metadata,
                }
            )
        return output

    def _describe_structure(self, book_structure: BookStructure, has_real_chapters: bool) -> str:
        count = len(book_structure.chapters)
        if has_real_chapters:
            return f"Chapter-based textbook structure detected with {count} chapter/structure records."
        if count:
            return f"Section/unit-based textbook structure detected with {count} section records."
        return "No reliable chapter or section structure detected; output is page-based only."

    def _global_notes(self, book_structure: BookStructure, warnings: list[str]) -> str:
        parts = [
            f"Structure detected by {book_structure.detected_by}.",
            "Page numbers use the PDF page positions extracted by the pipeline.",
            "The JSON artifact is generated in addition to DB persistence and embeddings.",
        ]
        if warnings:
            parts.append("Warnings: " + "; ".join(warnings))
        return " ".join(parts)

    def _structure_notes(self, structure: BookChapter) -> str | None:
        notes = []
        if structure.detected_by:
            notes.append(f"detected_by={structure.detected_by}")
        if structure.metadata:
            source = structure.metadata.get("pdf_start_page_source") or structure.metadata.get("printed_to_pdf_offset_source")
            if source:
                notes.append(f"page_mapping={source}")
        return "; ".join(notes) if notes else None

    def _merge_terms(self, chunks: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        terms: list[str] = []
        for chunk in chunks:
            for term in (chunk.get("important_terms") or []) + (chunk.get("keywords") or []):
                value = str(term).strip()
                if value and value.lower() not in seen:
                    seen.add(value.lower())
                    terms.append(value)
        return terms[:30]

    def _summarize_unit(self, chapter: BookChapter, lessons: list[dict[str, Any]]) -> str:
        return (
            f"{chapter.display_title} spans PDF pages {chapter.pdf_start_page}-{chapter.pdf_end_page} "
            f"and produced {len(lessons)} JSON lesson/chunk records."
        )

    def _summarize_lessons(self, lessons: list[dict[str, Any]]) -> str:
        titles = [str(l.get("lesson_title")) for l in lessons[:5] if l.get("lesson_title")]
        suffix = "..." if len(lessons) > 5 else ""
        return f"Contains {len(lessons)} extracted section lesson records: {', '.join(titles)}{suffix}"

    def _average_confidence(self, sections: list[BookChapter]) -> float | None:
        values = [s.confidence for s in sections if s.confidence is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)
