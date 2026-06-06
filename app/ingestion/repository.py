from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from psycopg.rows import dict_row

from db.connection import get_connection
from ingestion.book_structure import BookChapter, BookStructure, BookSubsection, ChapterResolver
from ingestion.embedding_service import EmbeddingRecord

logger = logging.getLogger(__name__)


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _array(value: list[str] | None) -> list[str]:
    return value or []


def to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.10g}" for v in values) + "]"


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
            for c in chapters
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
            "document_id", "page_start", "page_end", "chunk_index", "book_title", "school_name", "class_name",
            "subject", "grade", "board", "medium", "language", "detected_language", "chapter_number",
            "chapter_title", "unit_number", "unit_title", "lesson_title",
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
        """Store raw page text with book/school/chapter metadata for later reference.

        embeddings_pages is kept as the extraction table. This table is a more
        convenient raw-reference table because it repeats the school/class/book
        metadata and adds the best-known chapter for each page.
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
            document_id, school_name, class_name, grade, subject, book_title,
            chapter_number, chapter_title, unit_number, unit_title, lesson_title,
            section_number, section_title, subsection_number, subsection_title, topic, subtopic,
            page_number, printed_page_number, raw_text, cleaned_text,
            detected_language, word_count, token_count, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
        )
        ON CONFLICT(document_id, page_number) DO UPDATE SET
            school_name=EXCLUDED.school_name,
            class_name=EXCLUDED.class_name,
            grade=EXCLUDED.grade,
            subject=EXCLUDED.subject,
            book_title=EXCLUDED.book_title,
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
                    metadata.get("school_name"),
                    metadata.get("class_name"),
                    metadata.get("grade"),
                    metadata.get("subject"),
                    metadata.get("book_title") or metadata.get("title"),
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
        sql = f"""
        SELECT d.id, d.title, d.book_title, d.school_name, d.class_name, d.subject, d.grade, d.language,
               d.primary_language, d.content_profile, d.chunking_strategy, d.chunk_max_tokens, d.chunk_overlap_tokens,
               d.document_key, d.file_name, d.file_hash, d.total_pages, d.total_words, d.total_tokens,
               COUNT(DISTINCT c.id)::int AS chunks,
               COUNT(DISTINCT s.id)::int AS subsections,
               COUNT(DISTINCT v.id)::int AS embeddings
        FROM embeddings_documents d
        LEFT JOIN embeddings_chunks c ON c.document_id=d.id
        LEFT JOIN embeddings_vectors v ON v.document_id=d.id
        LEFT JOIN embeddings_book_subsections s ON s.document_id=d.id
        WHERE {where}
        GROUP BY d.id
        """
        with get_connection(self.database_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (value,))
                row = cur.fetchone()
                return dict(row) if row else None

    def vector_search(self, query_embedding: list[float], filters: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        where, params = self._build_filter_sql(filters, table_alias="c")
        params = [to_pgvector(query_embedding)] + params + [limit]
        sql = f"""
        SELECT c.id::text AS chunk_id,
               c.content, c.content_clean, c.book_title, c.school_name, c.class_name, c.subject, c.grade, c.language,
               c.chapter_title, c.unit_title, c.lesson_title, c.section_title, c.subsection_number, c.subsection_title, c.topic, c.chunk_type, c.page_start, c.page_end,
               c.source_label, c.citation_text,
               GREATEST(0, 1 - (v.embedding <=> %s::vector)) AS vector_score,
               0.0::float AS keyword_score
        FROM embeddings_vectors v
        JOIN embeddings_chunks c ON c.id = v.chunk_id
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
               c.content, c.content_clean, c.book_title, c.school_name, c.class_name, c.subject, c.grade, c.language,
               c.chapter_title, c.unit_title, c.lesson_title, c.section_title, c.subsection_number, c.subsection_title, c.topic, c.chunk_type, c.page_start, c.page_end,
               c.source_label, c.citation_text,
               0.0::float AS vector_score,
               ts_rank_cd(c.search_vector, websearch_to_tsquery('simple', %s))::float AS keyword_score
        FROM embeddings_chunks c
        {prefix} c.search_vector @@ websearch_to_tsquery('simple', %s)
        ORDER BY keyword_score DESC
        LIMIT %s
        """
        params = [query] + filter_params + [query, limit] if where else [query, query, limit]
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
        exact_fields = ["subject", "school_name", "grade", "class_name", "language", "board", "document_id", "chunk_type", "book_title"]
        for field in exact_fields:
            value = filters.get(field)
            if value:
                clauses.append(f"{table_alias}.{field} = %s")
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
