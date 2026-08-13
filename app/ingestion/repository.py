from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from psycopg.rows import dict_row

from db.connection import get_connection
from ingestion.book_structure import BookChapter, BookStructure, BookSubsection, ChapterResolver
from ingestion.embedding_service import EmbeddingRecord, SubsectionEmbeddingRecord

logger = logging.getLogger(__name__)


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _array(value: list[str] | None) -> list[str]:
    return value or []


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.10g}" for v in values) + "]"


def _chapter_rows_for_db(chapters: list[BookChapter]) -> list[BookChapter]:
    """Return the parent rows that belong in ``embeddings_book_chapters``.

    Some production JSONs expose a fine-grained ``section_index`` where several
    prose/poem sections share the same real TOC chapter number/title.  Those
    section rows are useful to the in-memory ``ChapterResolver`` (so chunks and
    raw pages keep section-level metadata), but inserting every one of them into
    ``embeddings_book_chapters`` makes a chapter menu repeat the same chapter.

    Collapse only *clearly hierarchical* repeated chapter groups:
      * same non-empty chapter_number
      * same non-empty chapter_title
      * more than one distinct child section

    Books whose top level is section-based (for example records with no
    chapter_number/chapter_title) and normal one-row-per-chapter books are left
    unchanged. This keeps the loader generic across the existing JSON formats.
    """
    if not chapters:
        return []

    grouped: dict[str, list[BookChapter]] = {}
    for chapter in chapters:
        number = str(chapter.chapter_number or "").strip()
        title = str(chapter.chapter_title or "").strip()
        if number and title:
            grouped.setdefault(number, []).append(chapter)

    repeated_hierarchy_detected = False
    for number, group in grouped.items():
        if len(group) <= 1:
            continue
        titles = {str(item.chapter_title or "").strip() for item in group if str(item.chapter_title or "").strip()}
        child_keys = {
            str(item.section_number or item.section_title or item.lesson_title or "").strip()
            for item in group
            if str(item.section_number or item.section_title or item.lesson_title or "").strip()
        }
        if len(titles) == 1 and len(child_keys) > 1:
            repeated_hierarchy_detected = True
            break

    if not repeated_hierarchy_detected:
        return chapters

    # Once the input is proven to be a chapter->section hierarchy, normalize every
    # real chapter to one parent row, including chapters that happen to have only
    # one child section (for example First Flight Chapter 9, The Proposal).
    collapsible: dict[str, list[BookChapter]] = {}
    for number, group in grouped.items():
        titles = {str(item.chapter_title or "").strip() for item in group if str(item.chapter_title or "").strip()}
        has_child_structure = any(item.section_number or item.section_title or item.lesson_title for item in group)
        if len(titles) == 1 and has_child_structure:
            collapsible[number] = group

    if not collapsible:
        return chapters

    emitted: set[str] = set()
    result: list[BookChapter] = []
    for chapter in chapters:
        number = str(chapter.chapter_number or "").strip()
        group = collapsible.get(number)
        if not group:
            result.append(chapter)
            continue
        if number in emitted:
            continue
        emitted.add(number)

        ordered = sorted(
            group,
            key=lambda item: (
                item.pdf_start_page if item.pdf_start_page is not None else 10**9,
                item.section_number or "",
                item.section_title or "",
            ),
        )
        first = ordered[0]
        last = max(
            ordered,
            key=lambda item: item.pdf_end_page if item.pdf_end_page is not None else -1,
        )
        pdf_starts = [item.pdf_start_page for item in ordered if item.pdf_start_page is not None]
        pdf_ends = [item.pdf_end_page for item in ordered if item.pdf_end_page is not None]

        child_sections = [
            {
                "section_number": item.section_number,
                "section_title": item.section_title,
                "lesson_title": item.lesson_title,
                "structure_type": item.structure_type,
                "printed_start_page": item.printed_start_page,
                "printed_end_page": item.printed_end_page,
                "pdf_start_page": item.pdf_start_page,
                "pdf_end_page": item.pdf_end_page,
            }
            for item in ordered
        ]
        metadata = dict(first.metadata or {})
        metadata.update(
            {
                "collapsed_from_child_sections": True,
                "child_section_count": len(ordered),
                "child_sections": child_sections,
            }
        )

        result.append(
            BookChapter(
                chapter_number=first.chapter_number,
                chapter_title=first.chapter_title,
                unit_number=first.unit_number,
                unit_title=first.unit_title,
                section_number=None,
                section_title=None,
                lesson_title=None,
                section_key=first.chapter_number,
                structure_type="chapter",
                printed_start_page=first.printed_start_page,
                printed_end_page=last.printed_end_page,
                pdf_start_page=min(pdf_starts) if pdf_starts else first.pdf_start_page,
                pdf_end_page=max(pdf_ends) if pdf_ends else last.pdf_end_page,
                confidence=min(
                    [item.confidence for item in ordered if item.confidence is not None],
                    default=first.confidence,
                ),
                detected_by="json_input_parent_chapter_collapse",
                metadata=metadata,
            )
        )

    return result


class RagRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def init_schema(self, schema_path: Path) -> None:
        from db.migrations import init_schema

        init_schema(self.database_url, schema_path)

    def create_ingestion_run(
        self,
        *,
        file_path: str,
        file_hash: str,
        document_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with get_connection(self.database_url) as conn, conn.transaction():
            row = conn.execute(
                """
                INSERT INTO embeddings_ingestion_runs(file_path, file_hash, document_key, status, metadata)
                VALUES (%s, %s, %s, 'running', %s::jsonb)
                RETURNING id
                """,
                (file_path, file_hash, document_key, _json(metadata or {})),
            ).fetchone()
            return str(row[0])

    def finish_ingestion_run(
        self,
        run_id: str,
        *,
        status: str,
        document_id: str | None = None,
        pages_extracted: int = 0,
        chunks_created: int = 0,
        embeddings_created: int = 0,
        error_message: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        with get_connection(self.database_url) as conn, conn.transaction():
            conn.execute(
                """
                UPDATE embeddings_ingestion_runs
                SET status=%s,
                    document_id=%s,
                    finished_at=now(),
                    pages_extracted=%s,
                    chunks_created=%s,
                    embeddings_created=%s,
                    error_message=%s,
                    warnings=%s::jsonb
                WHERE id=%s
                """,
                (status, document_id, pages_extracted, chunks_created, embeddings_created, error_message, _json(warnings or []), run_id),
            )

    def document_exists_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM embeddings_documents WHERE file_hash=%s", (file_hash,))
                row = cur.fetchone()
                return dict(row) if row else None

    def delete_document_by_hash(self, file_hash: str) -> None:
        with get_connection(self.database_url) as conn, conn.transaction():
            conn.execute("DELETE FROM embeddings_documents WHERE file_hash=%s", (file_hash,))

    def document_exists_by_document_key(self, document_key: str) -> dict[str, Any] | None:
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM embeddings_documents WHERE document_key=%s", (document_key,))
                row = cur.fetchone()
                return dict(row) if row else None

    def delete_document_by_document_key(self, document_key: str) -> None:
        with get_connection(self.database_url) as conn, conn.transaction():
            conn.execute("DELETE FROM embeddings_documents WHERE document_key=%s", (document_key,))

    def upsert_document(self, document: dict[str, Any]) -> str:
        return self._upsert_document(document, conflict_column="file_hash")

    def upsert_document_by_document_key(self, document: dict[str, Any]) -> str:
        if not document.get("document_key"):
            raise ValueError("document_key is required for document-key upsert.")
        return self._upsert_document(document, conflict_column="document_key")

    def _upsert_document(self, document: dict[str, Any], *, conflict_column: str) -> str:
        columns = [
            "title", "book_title", "document_key", "normalized_title", "school_name", "class_name", "subject", "grade",
            "board", "medium", "language", "detected_language", "primary_language", "languages_detected",
            "publisher", "edition", "publication_year", "isbn", "author", "source_type", "source_uri", "file_name",
            "file_path", "file_hash", "file_size_bytes", "mime_type", "total_pages", "total_words", "total_tokens",
            "extraction_status", "copyright_status", "license_notes", "llm_metadata_model", "llm_metadata_confidence",
            "structure_detected_by", "content_profile", "chunking_strategy", "chunk_max_tokens", "chunk_overlap_tokens", "metadata",
        ]
        values = []
        for col in columns:
            if col in {"metadata", "languages_detected"}:
                values.append(_json(document.get(col) or ([] if col == "languages_detected" else {})))
            else:
                values.append(document.get(col))
        placeholders = ", ".join(["%s::jsonb" if c in {"metadata", "languages_detected"} else "%s" for c in columns])
        update_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in columns if c != conflict_column)
        with get_connection(self.database_url) as conn, conn.transaction():
            row = conn.execute(
                f"""
                INSERT INTO embeddings_documents({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT({conflict_column}) DO UPDATE SET {update_clause}
                RETURNING id
                """,
                values,
            ).fetchone()
            return str(row[0])

    def insert_book_chapters(self, document_id: str, chapters: list[BookChapter]) -> None:
        if not chapters:
            return
        db_chapters = _chapter_rows_for_db(chapters)
        if len(db_chapters) != len(chapters):
            logger.info(
                "Collapsed %s structure rows to %s parent chapter rows for embeddings_book_chapters; "
                "section-level structures remain available to chunk/page resolution and subsection rows.",
                len(chapters),
                len(db_chapters),
            )
        sql = """
        INSERT INTO embeddings_book_chapters(
            document_id, chapter_number, chapter_title, unit_number, unit_title,
            section_number, section_title, lesson_title, section_key, structure_type,
            printed_start_page, printed_end_page, pdf_start_page, pdf_end_page,
            detected_by, confidence, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT(document_id, section_key) DO UPDATE SET
            chapter_number=EXCLUDED.chapter_number,
            chapter_title=EXCLUDED.chapter_title,
            unit_number=EXCLUDED.unit_number,
            unit_title=EXCLUDED.unit_title,
            section_number=EXCLUDED.section_number,
            section_title=EXCLUDED.section_title,
            lesson_title=EXCLUDED.lesson_title,
            structure_type=EXCLUDED.structure_type,
            printed_start_page=EXCLUDED.printed_start_page,
            printed_end_page=EXCLUDED.printed_end_page,
            pdf_start_page=EXCLUDED.pdf_start_page,
            pdf_end_page=EXCLUDED.pdf_end_page,
            detected_by=EXCLUDED.detected_by,
            confidence=EXCLUDED.confidence,
            metadata=EXCLUDED.metadata
        """
        params = [
            (
                document_id,
                c.chapter_number,
                c.chapter_title,
                c.unit_number,
                c.unit_title,
                c.section_number,
                c.section_title,
                c.lesson_title,
                c.section_key or c.section_number or c.chapter_number or c.display_number or c.display_title,
                c.structure_type,
                c.printed_start_page,
                c.printed_end_page,
                c.pdf_start_page,
                c.pdf_end_page,
                c.detected_by,
                c.confidence,
                _json(c.metadata or {}),
            )
            for c in db_chapters
            if c.display_title
        ]
        with get_connection(self.database_url) as conn, conn.transaction():
            conn.execute("DELETE FROM embeddings_book_chapters WHERE document_id=%s", (document_id,))
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def insert_book_subsections(self, document_id: str, subsections: list[BookSubsection]) -> None:
        """Replace subsection/day/exercise rows for a document.

        These rows are intentionally denormalized with chapter/section fields so
        callers can fetch exact subsection text and page ranges without joining
        back through chunks.
        """
        with get_connection(self.database_url) as conn, conn.transaction():
            conn.execute("DELETE FROM embeddings_book_subsections WHERE document_id=%s", (document_id,))
            if not subsections:
                return
            columns = [
                "document_id", "chapter_number", "chapter_title", "unit_number", "unit_title", "lesson_title",
                "section_number", "section_title", "subsection_number", "subsection_title", "anchor_marker",
                "anchor_pdf_page", "anchor_printed_page", "anchor_detection_method", "anchor_raw_heading",
                "pdf_start_page", "pdf_end_page", "printed_start_page", "printed_end_page", "page_count",
                "page_numbers", "printed_page_numbers", "included_exercises_or_activities", "includes",
                "subsection_text", "subsection_text_plain", "text_length_chars", "include_in_embeddings",
                "embedding_readiness", "text_sources", "quality_flags", "excluded_related_pages", "math_lines", "metadata",
            ]
            placeholders = ", ".join("%s::jsonb" if c in {"excluded_related_pages", "metadata"} else "%s" for c in columns)
            sql = f"""
                INSERT INTO embeddings_book_subsections({', '.join(columns)})
                VALUES ({placeholders})
            """
            params = []
            for ss in subsections:
                params.append(
                    (
                        document_id,
                        ss.chapter_number,
                        ss.chapter_title,
                        ss.unit_number,
                        ss.unit_title,
                        ss.lesson_title,
                        ss.section_number,
                        ss.section_title,
                        ss.subsection_number,
                        ss.subsection_title,
                        ss.anchor_marker,
                        ss.anchor_pdf_page,
                        ss.anchor_printed_page,
                        ss.anchor_detection_method,
                        ss.anchor_raw_heading,
                        ss.pdf_start_page,
                        ss.pdf_end_page,
                        ss.printed_start_page,
                        ss.printed_end_page,
                        ss.page_count,
                        ss.page_numbers or [],
                        ss.printed_page_numbers or [],
                        ss.included_exercises_or_activities or [],
                        ss.includes or [],
                        ss.subsection_text,
                        ss.subsection_text_plain,
                        ss.text_length_chars,
                        ss.include_in_embeddings,
                        ss.embedding_readiness,
                        ss.text_sources or [],
                        ss.quality_flags or [],
                        _json(ss.excluded_related_pages or []),
                        ss.math_lines or [],
                        _json(ss.metadata or {}),
                    )
                )
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def insert_teacher_schedules(self, document_id: str, schedules: list[dict[str, Any]]) -> None:
        """Replace optional real-teacher weekly schedules for a document.

        Teacher schedules are deliberately separate from ``embeddings_book_subsections``.
        Existing structural days/subsections therefore remain untouched for every
        old book, while Teacher Helper can query exact question-targeted page sets.
        """
        parent_known = {
            "schedule_key", "chapter_number", "chapter_title", "unit_number", "unit_title",
            "section_number", "section_title", "lesson_title", "week_start_date",
            "schedule_source", "schedule_type", "exercise", "schedule_note",
            "schedule_is_additive", "structural_subsections_unchanged",
            "teacher_facing_page_system", "internal_page_system", "days", "source_payload",
        }
        day_known = {
            "day", "weekday", "day_type", "activity", "topic", "teaching_book_page_ranges",
            "exercise_book_pages", "exercise", "questions", "range_source", "source_input_warning",
            "selected_book_pages", "selected_pdf_pages", "selected_page_count",
            "selection_is_contiguous", "display_book_pages", "display_pdf_pages",
            "selection_policy", "selected_pages_available",
        }

        parent_sql = """
        INSERT INTO embeddings_teacher_schedules(
            document_id, schedule_key, chapter_number, chapter_title, unit_number, unit_title,
            section_number, section_title, lesson_title, week_start_date, schedule_source,
            schedule_type, exercise, schedule_note, schedule_is_additive,
            structural_subsections_unchanged, teacher_facing_page_system, internal_page_system,
            metadata, source_payload
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s::jsonb, %s::jsonb
        )
        RETURNING id
        """
        day_sql = """
        INSERT INTO embeddings_teacher_schedule_days(
            teacher_schedule_id, document_id, day, weekday, day_type, activity, topic,
            teaching_book_page_ranges, exercise_book_pages, exercise, questions, range_source,
            source_input_warning, selected_book_pages, selected_pdf_pages, selected_page_count,
            selection_is_contiguous, display_book_pages, display_pdf_pages, selection_policy,
            selected_pages_available, metadata, source_payload
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
        )
        """

        with get_connection(self.database_url) as conn, conn.transaction():
            # Deleting parent rows cascades to day rows. Run even when schedules
            # is empty so a re-ingest can intentionally remove old schedule data.
            conn.execute("DELETE FROM embeddings_teacher_schedules WHERE document_id=%s", (document_id,))
            if not schedules:
                return

            for schedule in schedules:
                raw = dict(schedule or {})
                if not raw:
                    continue
                schedule_key = str(raw.get("schedule_key") or "").strip()
                if not schedule_key:
                    schedule_key = "|".join(
                        [
                            f"chapter:{raw.get('chapter_number') or ''}",
                            f"section:{raw.get('section_number') or ''}",
                            f"week:{raw.get('week_start_date') or ''}",
                            f"exercise:{raw.get('exercise') or ''}",
                        ]
                    )
                metadata = {key: value for key, value in raw.items() if key not in parent_known}
                metadata.setdefault("source", "json_input")
                metadata.setdefault("source_kind", "teacher_schedule")
                source_payload = raw.get("source_payload") if isinstance(raw.get("source_payload"), dict) else {
                    key: value for key, value in raw.items() if key not in {"schedule_key", "source_payload"}
                }

                with conn.cursor() as cur:
                    row = cur.execute(
                        parent_sql,
                        (
                            document_id,
                            schedule_key,
                            raw.get("chapter_number"),
                            raw.get("chapter_title"),
                            raw.get("unit_number"),
                            raw.get("unit_title"),
                            raw.get("section_number"),
                            raw.get("section_title"),
                            raw.get("lesson_title"),
                            raw.get("week_start_date"),
                            raw.get("schedule_source"),
                            raw.get("schedule_type"),
                            raw.get("exercise"),
                            raw.get("schedule_note"),
                            _bool_or_none(raw.get("schedule_is_additive")),
                            _bool_or_none(raw.get("structural_subsections_unchanged")),
                            raw.get("teacher_facing_page_system"),
                            raw.get("internal_page_system"),
                            _json(metadata),
                            _json(source_payload),
                        ),
                    ).fetchone()
                teacher_schedule_id = str(row[0])

                day_params = []
                for day_item in raw.get("days") or []:
                    day = dict(day_item) if isinstance(day_item, dict) else {}
                    if not day:
                        continue
                    day_metadata = {key: value for key, value in day.items() if key not in day_known}
                    day_metadata.setdefault("source", "json_input")
                    day_metadata.setdefault("source_kind", "teacher_schedule_day")
                    day_params.append(
                        (
                            teacher_schedule_id,
                            document_id,
                            _int_or_none(day.get("day")),
                            day.get("weekday"),
                            day.get("day_type"),
                            day.get("activity"),
                            day.get("topic"),
                            _json(day.get("teaching_book_page_ranges") if isinstance(day.get("teaching_book_page_ranges"), list) else []),
                            _int_list(day.get("exercise_book_pages")),
                            day.get("exercise") or raw.get("exercise"),
                            _text_list(day.get("questions")),
                            day.get("range_source"),
                            day.get("source_input_warning"),
                            _int_list(day.get("selected_book_pages")),
                            _int_list(day.get("selected_pdf_pages")),
                            _int_or_none(day.get("selected_page_count")),
                            _bool_or_none(day.get("selection_is_contiguous")),
                            day.get("display_book_pages"),
                            day.get("display_pdf_pages"),
                            day.get("selection_policy"),
                            _bool_or_none(day.get("selected_pages_available")),
                            _json(day_metadata),
                            _json(day),
                        )
                    )
                if day_params:
                    with conn.cursor() as cur:
                        cur.executemany(day_sql, day_params)

    def insert_page_extractions(self, document_id: str, page_extractions: list[dict[str, Any]]) -> None:
        """Persist extraction.page_extractions[] without dropping source fields.

        Frequently queried fields are mapped to typed columns. The complete merged
        page object is also stored in source_payload JSONB for lossless audit and
        forward compatibility with new extraction fields.
        """
        if not page_extractions:
            return

        columns = [
            "document_id", "page_number", "pdf_page_number",
            "printed_page_number", "printed_page_label",
            "chapter_type", "structure_type", "chapter_number", "chapter_title",
            "unit_number", "unit_title", "lesson_title",
            "section_number", "section_title",
            "linked_section_number", "linked_section_title",
            "subsection_numbers", "subsection_titles", "topic", "subtopic",
            "content_type", "assignment_status",
            "include_in_chapter_text", "include_in_lesson_text",
            "include_in_embeddings", "embedding_readiness",
            "source_type", "extraction_method", "extraction_quality",
            "detected_language", "has_text", "has_math", "has_table_like_text",
            "text", "text_plain", "production_page_text", "production_safe_text",
            "selectable_text", "raw_extracted_text", "ocr_text",
            "word_count", "token_count", "text_length_chars",
            "production_text_length_chars",
            "page_index_in_parent", "page_count_in_parent",
            "is_first_page", "is_last_page",
            "text_sources", "quality_flags", "production_exclusion_reasons",
            "unresolved_review_items", "reviewed_items_applied",
            "layout_validation", "metadata", "source_payload",
        ]
        json_columns = {"layout_validation", "metadata", "source_payload"}
        array_columns = {
            "subsection_numbers", "subsection_titles", "text_sources",
            "quality_flags", "production_exclusion_reasons",
            "unresolved_review_items", "reviewed_items_applied",
        }
        source_field_names = set(columns) - {"document_id"}
        source_field_names.add("page_text")

        def as_bool(value: Any, default: bool | None = None) -> bool | None:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"true", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "0", "no", "n", "off"}:
                return False
            return default

        def as_text_array(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                return [str(item) for item in value if str(item).strip()]
            text = str(value).strip()
            return [text] if text else []

        placeholders = ", ".join(
            "%s::jsonb" if column in json_columns else "%s"
            for column in columns
        )
        update_columns = [column for column in columns if column not in {"document_id", "page_number"}]
        update_clause = ", ".join(f"{column}=EXCLUDED.{column}" for column in update_columns)
        sql = f"""
            INSERT INTO embeddings_page_extractions({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(document_id, page_number) DO UPDATE SET {update_clause}
        """

        params: list[tuple[Any, ...]] = []
        for source in page_extractions:
            raw = dict(source or {})
            page_number = raw.get("page_number") or raw.get("pdf_page_number")
            if page_number is None:
                continue
            page_number = int(page_number)
            pdf_page_number = int(raw.get("pdf_page_number") or page_number)

            text_value = raw.get("text")
            if text_value is None:
                text_value = raw.get("page_text")
            text_plain = raw.get("text_plain")
            if text_plain is None:
                text_plain = text_value
            production_page_text = raw.get("production_page_text")
            production_safe_text = raw.get("production_safe_text")

            raw_metadata = raw.get("metadata")
            metadata_value = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata_value.setdefault("source", "extraction.page_extractions")
            extra_fields = {
                key: value for key, value in raw.items()
                if key not in source_field_names and key != "metadata"
            }
            if extra_fields:
                metadata_value["unmapped_fields"] = extra_fields

            normalized = dict(raw)
            normalized.update({
                "document_id": document_id,
                "page_number": page_number,
                "pdf_page_number": pdf_page_number,
                "printed_page_number": (
                    None if raw.get("printed_page_number") is None
                    else str(raw.get("printed_page_number"))
                ),
                "text": text_value,
                "text_plain": text_plain,
                "production_page_text": production_page_text,
                "production_safe_text": production_safe_text,
                "include_in_embeddings": as_bool(raw.get("include_in_embeddings"), True),
                "include_in_chapter_text": as_bool(raw.get("include_in_chapter_text")),
                "include_in_lesson_text": as_bool(raw.get("include_in_lesson_text")),
                "has_text": (
                    bool(str(production_page_text or production_safe_text or text_value or "").strip())
                    if raw.get("has_text") is None else as_bool(raw.get("has_text"), False)
                ),
                "has_math": as_bool(raw.get("has_math")),
                "has_table_like_text": as_bool(raw.get("has_table_like_text")),
                "is_first_page": as_bool(raw.get("is_first_page")),
                "is_last_page": as_bool(raw.get("is_last_page")),
                "text_length_chars": (
                    len(str(text_value or ""))
                    if raw.get("text_length_chars") is None else raw.get("text_length_chars")
                ),
                "production_text_length_chars": (
                    len(str(production_page_text or production_safe_text or ""))
                    if raw.get("production_text_length_chars") is None
                    else raw.get("production_text_length_chars")
                ),
                "layout_validation": raw.get("layout_validation") or {},
                "metadata": metadata_value,
                "source_payload": raw,
            })

            values: list[Any] = []
            for column in columns:
                value = normalized.get(column)
                if column in json_columns:
                    values.append(_json(value or {}))
                elif column in array_columns:
                    values.append(_array(as_text_array(value)))
                else:
                    values.append(value)
            params.append(tuple(values))

        if not params:
            return
        with get_connection(self.database_url) as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def insert_pages(self, document_id: str, pages: list[dict[str, Any]]) -> None:
        sql = """
        INSERT INTO embeddings_pages(
            document_id, page_number, raw_text, cleaned_text, detected_language, word_count, token_count,
            has_text, has_math, has_table_like_text, has_devanagari, has_english, extraction_method,
            extraction_quality, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
        )
        ON CONFLICT(document_id, page_number) DO UPDATE SET
            raw_text=EXCLUDED.raw_text,
            cleaned_text=EXCLUDED.cleaned_text,
            detected_language=EXCLUDED.detected_language,
            word_count=EXCLUDED.word_count,
            token_count=EXCLUDED.token_count,
            has_text=EXCLUDED.has_text,
            has_math=EXCLUDED.has_math,
            has_table_like_text=EXCLUDED.has_table_like_text,
            has_devanagari=EXCLUDED.has_devanagari,
            has_english=EXCLUDED.has_english,
            extraction_method=EXCLUDED.extraction_method,
            extraction_quality=EXCLUDED.extraction_quality,
            metadata=EXCLUDED.metadata
        """
        params = [
            (
                document_id, p["page_number"], p.get("raw_text"), p.get("cleaned_text"), p.get("detected_language"),
                p.get("word_count"), p.get("token_count"), p.get("has_text"), p.get("has_math"),
                p.get("has_table_like_text"), p.get("has_devanagari"), p.get("has_english"),
                p.get("extraction_method"), p.get("extraction_quality"), _json(p.get("metadata", {})),
            )
            for p in pages
        ]
        with get_connection(self.database_url) as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def insert_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        columns = [
            "document_id", "page_start", "page_end", "chunk_index", "detected_language",
            "chapter_number", "chapter_title", "unit_number", "unit_title", "lesson_title",
            "section_number", "section_title", "subsection_number", "subsection_title", "topic", "subtopic", "chunk_type", "content_domain", "difficulty_level",
            "pedagogical_role", "content", "content_clean", "content_for_embedding", "summary", "keywords", "important_terms",
            "formulas", "numbers", "question_types", "word_count", "token_count", "char_count", "has_formula",
            "has_numbers", "has_questions", "has_exercises", "has_examples", "has_definition", "has_table_like_text",
            "has_devanagari", "has_english", "source_label", "citation_text", "metadata",
        ]
        placeholders = ", ".join(["%s"] * (len(columns) - 1) + ["%s::jsonb"])
        update_cols = [c for c in columns if c not in {"document_id", "chunk_index"}]
        update_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        sql = f"""
            INSERT INTO embeddings_chunks({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(document_id, chunk_index) DO UPDATE SET {update_clause}
            RETURNING id, chunk_index
        """
        enriched = []
        with get_connection(self.database_url) as conn, conn.transaction():
            for chunk in chunks:
                values = []
                for col in columns:
                    if col == "document_id":
                        values.append(document_id)
                    elif col == "metadata":
                        values.append(_json(chunk.get("metadata", {})))
                    elif col in {"keywords", "important_terms", "formulas", "numbers", "question_types"}:
                        values.append(_array(chunk.get(col)))
                    else:
                        values.append(chunk.get(col))
                row = conn.execute(sql, values).fetchone()
                item = dict(chunk)
                item["id"] = str(row[0])
                enriched.append(item)
        return enriched

    def insert_raw_text_pages(
        self,
        document_id: str,
        pages: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any],
        book_structure: BookStructure | None = None,
    ) -> None:
        """Store raw page text with page/structure metadata for later reference.

        Book-level metadata lives in embeddings_documents and is retrieved by
        joining on document_id. This table keeps only page-specific and
        structure-specific fields.
        """
        if not pages:
            return

        resolver = ChapterResolver(book_structure.chapters) if book_structure else None

        def chapter_for_page(page_number: int) -> dict[str, Any]:
            if resolver:
                resolved = resolver.chapter_for_pdf_page(page_number)
                if resolved:
                    return resolved.to_dict()
            matching = [
                c for c in chunks
                if int(c.get("page_start") or 0) <= page_number <= int(c.get("page_end") or 0)
            ]
            # Prefer a chunk with structure info, then any chunk covering the page.
            matching.sort(key=lambda c: 0 if (c.get("chapter_title") or c.get("section_title") or c.get("unit_title")) else 1)
            if matching:
                return matching[0]
            return {}

        sql = """
        INSERT INTO embeddings_raw_text_pages(
            document_id, chapter_number, chapter_title, unit_number, unit_title, lesson_title,
            section_number, section_title, subsection_number, subsection_title, topic, subtopic,
            page_number, printed_page_number, raw_text, cleaned_text,
            detected_language, word_count, token_count, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
        )
        ON CONFLICT(document_id, page_number) DO UPDATE SET
            chapter_number=EXCLUDED.chapter_number,
            chapter_title=EXCLUDED.chapter_title,
            unit_number=EXCLUDED.unit_number,
            unit_title=EXCLUDED.unit_title,
            lesson_title=EXCLUDED.lesson_title,
            section_number=EXCLUDED.section_number,
            section_title=EXCLUDED.section_title,
            subsection_number=EXCLUDED.subsection_number,
            subsection_title=EXCLUDED.subsection_title,
            topic=EXCLUDED.topic,
            subtopic=EXCLUDED.subtopic,
            printed_page_number=EXCLUDED.printed_page_number,
            raw_text=EXCLUDED.raw_text,
            cleaned_text=EXCLUDED.cleaned_text,
            detected_language=EXCLUDED.detected_language,
            word_count=EXCLUDED.word_count,
            token_count=EXCLUDED.token_count,
            metadata=EXCLUDED.metadata
        """
        params = []
        for page in pages:
            page_number = int(page["page_number"])
            chapter = chapter_for_page(page_number)
            page_metadata = dict(page.get("metadata") or {})
            printed_page_number = resolver.printed_page_for_pdf_page(page_number) if resolver else None
            page_metadata.update({
                "source_table": "embeddings_raw_text_pages",
                "file_metadata_source": metadata.get("path_metadata_source"),
                "printed_page_number": printed_page_number,
                "chapter_detection_source": chapter.get("detected_by"),
                "structure_type": chapter.get("structure_type"),
                "unit_title": chapter.get("unit_title"),
                "section_title": chapter.get("section_title"),
            })
            params.append(
                (
                    document_id,
                    chapter.get("chapter_number"),
                    chapter.get("chapter_title"),
                    chapter.get("unit_number"),
                    chapter.get("unit_title"),
                    chapter.get("lesson_title"),
                    chapter.get("section_number"),
                    chapter.get("section_title"),
                    chapter.get("subsection_number"),
                    chapter.get("subsection_title"),
                    chapter.get("topic"),
                    chapter.get("subtopic"),
                    page_number,
                    printed_page_number,
                    page.get("raw_text"),
                    page.get("cleaned_text"),
                    page.get("detected_language"),
                    page.get("word_count"),
                    page.get("token_count"),
                    _json(page_metadata),
                )
            )
        with get_connection(self.database_url) as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def insert_embeddings(self, records: list[EmbeddingRecord]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO embeddings_vectors(
            chunk_id, document_id, embedding_model, embedding_dimensions, embedding, embedding_input_hash
        ) VALUES (%s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT(chunk_id, embedding_model, embedding_dimensions) DO UPDATE SET
            embedding=EXCLUDED.embedding,
            embedding_input_hash=EXCLUDED.embedding_input_hash,
            created_at=now()
        """
        params = [
            (
                r.chunk_id,
                r.document_id,
                r.embedding_model,
                r.embedding_dimensions,
                to_pgvector(r.embedding),
                r.embedding_input_hash,
            )
            for r in records
        ]
        with get_connection(self.database_url) as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def get_subsections_for_embedding(self, document_id: str) -> list[dict[str, Any]]:
        """Return exact subsection/day/exercise rows for subsection-level vectors.

        Important: subsection vectors are created for every subsection that has text,
        even when include_in_embeddings=false. Some Math subsections are marked
        include_in_embeddings=false because the production chunk policy excludes
        OCR-risky pages until QA, but we still store a subsection vector so callers
        can retrieve every subsection/day/exercise. The include_in_embeddings flag
        and quality_flags are preserved in the vector metadata for downstream filtering.
        """
        sql = """
        SELECT id::text, document_id::text, chapter_number, chapter_title, unit_number, unit_title, lesson_title,
               section_number, section_title, subsection_number, subsection_title, anchor_marker,
               anchor_pdf_page, anchor_printed_page, anchor_detection_method, anchor_raw_heading,
               pdf_start_page, pdf_end_page, printed_start_page, printed_end_page, page_count,
               page_numbers, printed_page_numbers, included_exercises_or_activities, includes,
               subsection_text, subsection_text_plain, text_length_chars, include_in_embeddings,
               embedding_readiness, text_sources, quality_flags, excluded_related_pages, math_lines, metadata
        FROM embeddings_book_subsections
        WHERE document_id = %s
          AND COALESCE(NULLIF(subsection_text_plain, ''), NULLIF(subsection_text, '')) IS NOT NULL
        ORDER BY COALESCE(pdf_start_page, 2147483647), section_number, subsection_number, subsection_title
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (document_id,))
                return [dict(r) for r in cur.fetchall()]

    def insert_subsection_embeddings(self, records: list[SubsectionEmbeddingRecord]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO embeddings_subsection_vectors(
            subsection_id, document_id, embedding_model, embedding_dimensions, embedding,
            embedding_input_hash, content_for_embedding, token_count, text_was_truncated, metadata
        ) VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT(subsection_id, embedding_model, embedding_dimensions) DO UPDATE SET
            document_id=EXCLUDED.document_id,
            embedding=EXCLUDED.embedding,
            embedding_input_hash=EXCLUDED.embedding_input_hash,
            content_for_embedding=EXCLUDED.content_for_embedding,
            token_count=EXCLUDED.token_count,
            text_was_truncated=EXCLUDED.text_was_truncated,
            metadata=EXCLUDED.metadata,
            created_at=now()
        """
        params = [
            (
                r.subsection_id,
                r.document_id,
                r.embedding_model,
                r.embedding_dimensions,
                to_pgvector(r.embedding),
                r.embedding_input_hash,
                r.content_for_embedding,
                r.token_count,
                r.text_was_truncated,
                _json(r.metadata or {}),
            )
            for r in records
        ]
        with get_connection(self.database_url) as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    def get_document_summary(self, document_id: str | None = None, file_hash: str | None = None, document_key: str | None = None) -> dict[str, Any] | None:
        if document_id:
            where = "d.id=%s"
            value = document_id
        elif document_key:
            where = "d.document_key=%s"
            value = document_key
        else:
            where = "d.file_hash=%s"
            value = file_hash
        if not value:
            return None
        # Do not join chunks/vectors/subsections in one query here.
        # Joining all child tables creates a large cross product after subsection vectors are added
        # (for example chunks * vectors * subsections * subsection_vectors), which can make the
        # final CLI summary appear to hang even after ingestion has already completed.
        sql = f"""
        SELECT d.id, d.title, d.book_title, d.school_name, d.class_name, d.subject, d.grade, d.language,
               d.primary_language, d.content_profile, d.chunking_strategy, d.chunk_max_tokens, d.chunk_overlap_tokens,
               d.document_key, d.file_name, d.file_hash, d.total_pages, d.total_words, d.total_tokens,
               (SELECT COUNT(*)::int FROM embeddings_chunks c WHERE c.document_id = d.id) AS chunks,
               (SELECT COUNT(*)::int FROM embeddings_book_subsections s WHERE s.document_id = d.id) AS subsections,
               (SELECT COUNT(*)::int FROM embeddings_teacher_schedules ts WHERE ts.document_id = d.id) AS teacher_schedules,
               (SELECT COUNT(*)::int FROM embeddings_teacher_schedule_days td WHERE td.document_id = d.id) AS teacher_schedule_days,
               (SELECT COUNT(*)::int FROM embeddings_page_extractions pe WHERE pe.document_id = d.id) AS page_extractions,
               (SELECT COUNT(*)::int FROM embeddings_page_extractions pe WHERE pe.document_id = d.id AND pe.include_in_embeddings) AS embedding_eligible_pages,
               (SELECT COUNT(*)::int FROM embeddings_vectors v WHERE v.document_id = d.id) AS embeddings,
               (SELECT COUNT(*)::int FROM embeddings_subsection_vectors sv WHERE sv.document_id = d.id) AS subsection_embeddings,
               (
                   (SELECT COUNT(*)::int FROM embeddings_vectors v WHERE v.document_id = d.id)
                   +
                   (SELECT COUNT(*)::int FROM embeddings_subsection_vectors sv WHERE sv.document_id = d.id)
               ) AS total_embeddings
        FROM embeddings_documents d
        WHERE {where}
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (value,))
                row = cur.fetchone()
                return dict(row) if row else None

    def list_teacher_schedule_days(
        self,
        *,
        document_id: str | None = None,
        document_key: str | None = None,
        chapter_number: str | None = None,
        chapter_title: str | None = None,
        section_number: str | None = None,
        section_title: str | None = None,
        week_start_date: str | None = None,
        exercise: str | None = None,
        weekday: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return teacher-facing weekly schedule days with exact selected pages."""
        clauses, params = self._document_join_filter("ts", document_id=document_id, document_key=document_key)
        exact_filters = {
            "chapter_number": chapter_number,
            "section_number": section_number,
            "exercise": exercise,
        }
        for column, value in exact_filters.items():
            if value is not None:
                clauses.append(f"ts.{column} = %s")
                params.append(value)
        text_filters = {
            "chapter_title": chapter_title,
            "section_title": section_title,
        }
        for column, value in text_filters.items():
            if value:
                clauses.append(f"LOWER(ts.{column}) = LOWER(%s)")
                params.append(value)
        if week_start_date:
            clauses.append("ts.week_start_date = %s::date")
            params.append(week_start_date)
        if weekday:
            clauses.append("LOWER(td.weekday) = LOWER(%s)")
            params.append(weekday)

        sql = f"""
        SELECT d.id::text AS document_id,
               d.document_key,
               d.book_title,
               d.school_name,
               d.class_name,
               d.subject,
               d.grade,
               ts.id::text AS teacher_schedule_id,
               ts.schedule_key,
               ts.chapter_number,
               ts.chapter_title,
               ts.unit_number,
               ts.unit_title,
               ts.section_number,
               ts.section_title,
               ts.lesson_title,
               ts.week_start_date,
               ts.schedule_source,
               ts.schedule_type,
               ts.exercise,
               ts.schedule_note,
               ts.schedule_is_additive,
               ts.structural_subsections_unchanged,
               ts.teacher_facing_page_system,
               ts.internal_page_system,
               td.id::text AS teacher_schedule_day_id,
               td.day,
               td.weekday,
               td.day_type,
               td.activity,
               td.topic,
               td.teaching_book_page_ranges,
               td.exercise_book_pages,
               td.questions,
               td.range_source,
               td.source_input_warning,
               td.selected_book_pages,
               td.selected_pdf_pages,
               td.selected_page_count,
               td.selection_is_contiguous,
               td.display_book_pages,
               td.display_pdf_pages,
               td.selection_policy,
               td.selected_pages_available,
               td.metadata
        FROM embeddings_teacher_schedules ts
        JOIN embeddings_teacher_schedule_days td ON td.teacher_schedule_id = ts.id
        JOIN embeddings_documents d ON d.id = ts.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY ts.week_start_date, COALESCE(td.day, 2147483647), td.weekday
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def vector_search(self, query_embedding: list[float], filters: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        where, params = self._build_filter_sql(filters, table_alias="c")
        params = [to_pgvector(query_embedding)] + params + [limit]
        sql = f"""
        SELECT c.id::text AS chunk_id,
               c.content, c.content_clean, d.book_title, d.school_name, d.class_name, d.subject, d.grade, d.language,
               c.chapter_title, c.unit_title, c.lesson_title, c.section_title, c.subsection_number, c.subsection_title, c.topic, c.chunk_type, c.page_start, c.page_end,
               c.source_label, c.citation_text,
               GREATEST(0, 1 - (v.embedding <=> %s::vector)) AS vector_score,
               0.0::float AS keyword_score
        FROM embeddings_vectors v
        JOIN embeddings_chunks c ON c.id = v.chunk_id
        JOIN embeddings_documents d ON d.id = c.document_id
        {where}
        ORDER BY v.embedding <=> %s::vector
        LIMIT %s
        """
        params = [to_pgvector(query_embedding)] + params[1:-1] + [to_pgvector(query_embedding), limit]
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def keyword_search(self, query: str, filters: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        where, filter_params = self._build_filter_sql(filters, table_alias="c")
        prefix = "WHERE" if not where else where + " AND"
        sql = f"""
        SELECT c.id::text AS chunk_id,
               c.content, c.content_clean, d.book_title, d.school_name, d.class_name, d.subject, d.grade, d.language,
               c.chapter_title, c.unit_title, c.lesson_title, c.section_title, c.subsection_number, c.subsection_title, c.topic, c.chunk_type, c.page_start, c.page_end,
               c.source_label, c.citation_text,
               0.0::float AS vector_score,
               ts_rank_cd(c.search_vector, websearch_to_tsquery('simple', %s))::float AS keyword_score
        FROM embeddings_chunks c
        JOIN embeddings_documents d ON d.id = c.document_id
        {prefix} (
            c.search_vector @@ websearch_to_tsquery('simple', %s)
            OR to_tsvector('simple', coalesce(d.school_name, '') || ' ' || coalesce(d.class_name, '') || ' ' || coalesce(d.book_title, '') || ' ' || coalesce(d.subject, '') || ' ' || coalesce(d.grade, '') || ' ' || coalesce(d.board, '') || ' ' || coalesce(d.language, '')) @@ websearch_to_tsquery('simple', %s)
        )
        ORDER BY keyword_score DESC
        LIMIT %s
        """
        params = [query] + filter_params + [query, query, limit] if where else [query, query, query, limit]
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def get_chapter_text(
        self,
        *,
        document_id: str | None = None,
        document_key: str | None = None,
        chapter_number: str | None = None,
        chapter_title: str | None = None,
        unit_number: str | None = None,
        unit_title: str | None = None,
        section_number: str | None = None,
        section_title: str | None = None,
    ) -> dict[str, Any]:
        """Return page-level text for a chapter/section range from raw pages."""
        clauses, params = self._document_join_filter("p", document_id=document_id, document_key=document_key)
        self._append_structure_filters(
            clauses,
            params,
            alias="p",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            unit_number=unit_number,
            unit_title=unit_title,
            section_number=section_number,
            section_title=section_title,
        )
        sql = f"""
        SELECT d.id::text AS document_id,
               d.document_key,
               d.book_title,
               d.school_name,
               d.class_name,
               d.subject,
               d.grade,
               p.chapter_number,
               p.chapter_title,
               p.unit_number,
               p.unit_title,
               p.section_number,
               p.section_title,
               p.page_number,
               p.printed_page_number,
               p.cleaned_text,
               p.raw_text
        FROM embeddings_raw_text_pages p
        JOIN embeddings_documents d ON d.id = p.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY p.page_number
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = [dict(r) for r in cur.fetchall()]
        text_parts = [(r.get("cleaned_text") or r.get("raw_text") or "").strip() for r in rows]
        text = "\n\n".join(t for t in text_parts if t)
        return {
            "document": self._document_from_rows(rows),
            "filters": {
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "unit_number": unit_number,
                "unit_title": unit_title,
                "section_number": section_number,
                "section_title": section_title,
            },
            "page_count": len(rows),
            "pdf_pages": [r["page_number"] for r in rows],
            "printed_pages": [r["printed_page_number"] for r in rows if r.get("printed_page_number") is not None],
            "pages": rows,
            "text": text,
        }

    def list_subsections(
        self,
        *,
        document_id: str | None = None,
        document_key: str | None = None,
        chapter_number: str | None = None,
        chapter_title: str | None = None,
        unit_number: str | None = None,
        unit_title: str | None = None,
        section_number: str | None = None,
        section_title: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = self._document_join_filter("s", document_id=document_id, document_key=document_key)
        self._append_structure_filters(
            clauses,
            params,
            alias="s",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            unit_number=unit_number,
            unit_title=unit_title,
            section_number=section_number,
            section_title=section_title,
        )
        sql = f"""
        SELECT d.id::text AS document_id,
               d.document_key,
               d.book_title,
               s.chapter_number,
               s.chapter_title,
               s.unit_number,
               s.unit_title,
               s.section_number,
               s.section_title,
               s.subsection_number,
               s.subsection_title,
               s.anchor_marker,
               s.anchor_pdf_page,
               s.anchor_printed_page,
               s.pdf_start_page,
               s.pdf_end_page,
               s.printed_start_page,
               s.printed_end_page,
               s.page_count,
               s.page_numbers,
               s.printed_page_numbers,
               s.includes,
               s.included_exercises_or_activities,
               s.text_length_chars,
               s.include_in_embeddings,
               s.embedding_readiness,
               s.quality_flags
        FROM embeddings_book_subsections s
        JOIN embeddings_documents d ON d.id = s.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY COALESCE(s.pdf_start_page, 2147483647), s.section_number, s.subsection_number, s.subsection_title
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

    def get_subsection_text(
        self,
        *,
        document_id: str | None = None,
        document_key: str | None = None,
        chapter_number: str | None = None,
        chapter_title: str | None = None,
        unit_number: str | None = None,
        unit_title: str | None = None,
        section_number: str | None = None,
        section_title: str | None = None,
        subsection_number: str | None = None,
        subsection_title: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses, params = self._document_join_filter("s", document_id=document_id, document_key=document_key)
        self._append_structure_filters(
            clauses,
            params,
            alias="s",
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            unit_number=unit_number,
            unit_title=unit_title,
            section_number=section_number,
            section_title=section_title,
            subsection_number=subsection_number,
            subsection_title=subsection_title,
        )
        sql = f"""
        SELECT d.id::text AS document_id,
               d.document_key,
               d.book_title,
               d.school_name,
               d.class_name,
               d.subject,
               d.grade,
               s.chapter_number,
               s.chapter_title,
               s.unit_number,
               s.unit_title,
               s.section_number,
               s.section_title,
               s.subsection_number,
               s.subsection_title,
               s.anchor_marker,
               s.anchor_pdf_page,
               s.anchor_printed_page,
               s.anchor_detection_method,
               s.anchor_raw_heading,
               s.pdf_start_page,
               s.pdf_end_page,
               s.printed_start_page,
               s.printed_end_page,
               s.page_count,
               s.page_numbers,
               s.printed_page_numbers,
               s.includes,
               s.included_exercises_or_activities,
               s.text_sources,
               s.quality_flags,
               s.excluded_related_pages,
               s.math_lines,
               s.include_in_embeddings,
               s.embedding_readiness,
               s.subsection_text,
               s.subsection_text_plain,
               s.metadata
        FROM embeddings_book_subsections s
        JOIN embeddings_documents d ON d.id = s.document_id
        WHERE {' AND '.join(clauses)}
        ORDER BY COALESCE(s.pdf_start_page, 2147483647), s.section_number, s.subsection_number, s.subsection_title
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

    def _document_join_filter(self, alias: str, *, document_id: str | None, document_key: str | None) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if document_id:
            clauses.append(f"{alias}.document_id = %s")
            params.append(document_id)
        if document_key:
            clauses.append("d.document_key = %s")
            params.append(document_key)
        if not clauses:
            raise ValueError("Provide document_id or document_key.")
        return clauses, params

    def _append_structure_filters(
        self,
        clauses: list[str],
        params: list[Any],
        *,
        alias: str,
        chapter_number: str | None = None,
        chapter_title: str | None = None,
        unit_number: str | None = None,
        unit_title: str | None = None,
        section_number: str | None = None,
        section_title: str | None = None,
        subsection_number: str | None = None,
        subsection_title: str | None = None,
    ) -> None:
        exact = {
            "chapter_number": chapter_number,
            "unit_number": unit_number,
            "section_number": section_number,
            "subsection_number": subsection_number,
        }
        fuzzy = {
            "chapter_title": chapter_title,
            "unit_title": unit_title,
            "section_title": section_title,
            "subsection_title": subsection_title,
        }
        for field, value in exact.items():
            if value:
                clauses.append(f"{alias}.{field} = %s")
                params.append(value)
        for field, value in fuzzy.items():
            if value:
                clauses.append(f"{alias}.{field} ILIKE %s")
                params.append(f"%{value}%")

    def _document_from_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        row = rows[0]
        return {
            "document_id": row.get("document_id"),
            "document_key": row.get("document_key"),
            "book_title": row.get("book_title"),
            "school_name": row.get("school_name"),
            "class_name": row.get("class_name"),
            "subject": row.get("subject"),
            "grade": row.get("grade"),
        }

    def _build_filter_sql(self, filters: dict[str, Any], table_alias: str = "c") -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        chunk_exact_fields = ["document_id", "chunk_type"]
        document_exact_fields = ["subject", "school_name", "grade", "class_name", "language", "board", "book_title"]
        for field in chunk_exact_fields:
            value = filters.get(field)
            if value:
                clauses.append(f"{table_alias}.{field} = %s")
                params.append(value)
        for field in document_exact_fields:
            value = filters.get(field)
            if value:
                clauses.append(f"d.{field} = %s")
                params.append(value)
        if filters.get("chapter_title"):
            clauses.append(f"{table_alias}.chapter_title ILIKE %s")
            params.append(f"%{filters['chapter_title']}%")
        if filters.get("unit_title"):
            clauses.append(f"{table_alias}.unit_title ILIKE %s")
            params.append(f"%{filters['unit_title']}%")
        if filters.get("section_title"):
            clauses.append(f"{table_alias}.section_title ILIKE %s")
            params.append(f"%{filters['section_title']}%")
        if filters.get("subsection_title"):
            clauses.append(f"{table_alias}.subsection_title ILIKE %s")
            params.append(f"%{filters['subsection_title']}%")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return where, params
