from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import Settings
from ingestion.chunker import MeaningfulChunker
from ingestion.chunking_strategy import ChunkingPlan, ChunkingStrategySelector
from ingestion.embedding_service import OpenAIEmbeddingService
from ingestion.language_detector import LanguageDetector
from ingestion.llm_metadata_detector import LLMMetadataDetector
from ingestion.json_exporter import ExtractionJsonExporter
from ingestion.json_input_loader import JsonExtractionInputLoader
from ingestion.metadata_builder import MetadataBuilder
from ingestion.pdf_extractor import PDFTextExtractor
from ingestion.pdf_preprocessor import PdfPreprocessor
from ingestion.repository import RagRepository
from ingestion.structure_detector import StructureDetector
from ingestion.text_cleaner import TextCleaner
from utils.hashing import sha256_file
from utils.token_counter import TokenCounter
from ingestion.book_structure import BookStructure, ChapterResolver

logger = logging.getLogger(__name__)


def _stable_json_document_hash(document_key: str) -> str:
    """Return a deterministic hash for JSON documents based on document_key, not file contents."""
    return hashlib.sha256(f"json_document_key:{document_key}".encode("utf-8")).hexdigest()


class IngestService:
    def __init__(self, settings: Settings, repository: RagRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.token_counter = TokenCounter(settings.openai_embedding_model)
        self.language_detector = LanguageDetector()
        self.cleaner = TextCleaner()
        self.metadata_builder = MetadataBuilder(self.token_counter, self.language_detector)
        self.structure_detector = StructureDetector()
        self.extractor = PDFTextExtractor(self.cleaner, self.language_detector, self.token_counter)
        self.preprocessor = PdfPreprocessor.from_settings(settings)
        self.metadata_detector = LLMMetadataDetector(settings)
        self.chunking_selector = ChunkingStrategySelector(
            default_max_tokens=settings.chunk_max_tokens,
            default_overlap_tokens=settings.chunk_overlap_tokens,
            auto_enabled=settings.auto_chunking_enabled,
        )
        self.json_exporter = ExtractionJsonExporter()
        self.json_input_loader = JsonExtractionInputLoader(self.cleaner, self.language_detector, self.token_counter)

    def ingest_pdf(
        self,
        pdf_path: Path,
        metadata: dict[str, Any],
        *,
        reindex: bool = False,
        dry_run: bool = False,
        export_json: bool | None = None,
        output_json_dir: Path | None = None,
        log_page_text: bool | None = None,
    ) -> dict[str, Any]:
        original_pdf_path = pdf_path.resolve()
        file_hash = sha256_file(original_pdf_path)
        reindex = reindex or self.settings.reindex_existing

        if not dry_run:
            existing = self.repository.document_exists_by_hash(file_hash)
            if existing and not reindex:
                summary = self.repository.get_document_summary(file_hash=file_hash)
                return {"status": "skipped_existing", "reason": "same file_hash already ingested", "document": summary}
            if existing and reindex:
                logger.info("Reindex requested. Deleting existing document for file hash %s", file_hash)
                self.repository.delete_document_by_hash(file_hash)

        run_id: str | None = None
        warnings: list[str] = []
        pages_count = chunks_count = embeddings_count = 0
        document_id: str | None = None
        json_output_path: Path | None = None
        try:
            if not dry_run:
                run_id = self.repository.create_ingestion_run(file_path=str(original_pdf_path), file_hash=file_hash, metadata=metadata)

            preprocess_result = self.preprocessor.prepare(original_pdf_path, metadata=metadata)
            extraction_pdf_path = preprocess_result.pdf_for_extraction
            if preprocess_result.used_ocr:
                warnings.append(
                    f"OCRmyPDF preprocessing used ({preprocess_result.ocr_language}): {preprocess_result.quality_report.reason}"
                )
            metadata = dict(metadata)
            metadata["original_pdf_path"] = str(preprocess_result.original_pdf)
            metadata["pdf_for_extraction"] = str(preprocess_result.pdf_for_extraction)
            metadata["ocr_preprocessing_used"] = preprocess_result.used_ocr
            metadata["ocr_language"] = preprocess_result.ocr_language
            metadata["ocr_quality_report"] = preprocess_result.quality_report.__dict__

            pages, extraction_warnings = self.extractor.extract(extraction_pdf_path)
            warnings.extend(preprocess_result.warnings)
            warnings.extend(extraction_warnings)
            pages_count = len(pages)
            logger.info("Extracted %s pages from %s", pages_count, extraction_pdf_path)

            book_structure = self.metadata_detector.detect(pages, metadata)
            metadata = self._merge_detected_metadata(metadata, book_structure)
            chunking_plan = self.chunking_selector.select(pages, metadata, book_structure)
            chunker = self._build_chunker(chunking_plan)
            chunks = chunker.chunk_pages(pages, metadata, book_structure=book_structure)
            chunks_count = len(chunks)
            logger.info(
                "Book structure detected_by=%s structures=%s chunks=%s profile=%s",
                book_structure.detected_by,
                len(book_structure.chapters),
                chunks_count,
                chunking_plan.content_profile,
            )
            self._log_page_extractions(pages, book_structure, enabled=log_page_text)
            if self._should_export_json(export_json):
                json_output_path = self.json_exporter.write_combined_extraction(
                    output_dir=output_json_dir or Path(self.settings.json_output_dir),
                    original_pdf_path=original_pdf_path,
                    extraction_pdf_path=extraction_pdf_path,
                    metadata=metadata,
                    pages=pages,
                    chunks=chunks,
                    book_structure=book_structure,
                    chunking_plan=chunking_plan,
                    file_hash=file_hash,
                    warnings=warnings,
                    dry_run=dry_run,
                )

            detected_language = self._dominant_language([p.detected_language for p in pages])
            if book_structure.primary_language and detected_language == "Unknown":
                detected_language = book_structure.primary_language
            total_words = sum(p.word_count for p in pages)
            total_tokens = sum(p.token_count for p in pages)

            if dry_run:
                return {
                    "status": "dry_run_ok",
                    "file": str(original_pdf_path),
                    "pdf_for_extraction": str(extraction_pdf_path),
                    "ocr_preprocessing_used": preprocess_result.used_ocr,
                    "ocr_language": preprocess_result.ocr_language,
                    "ocr_quality_report": preprocess_result.quality_report.__dict__,
                    "file_hash": file_hash,
                    "pages_extracted": pages_count,
                    "chunks_created": chunks_count,
                    "subsections_detected": len(book_structure.subsections),
                    "detected_language": detected_language,
                    "metadata": metadata,
                    "book_structure": book_structure.to_dict(),
                    "chunking_plan": chunking_plan.to_dict(),
                    "warnings": warnings,
                    "json_output": str(json_output_path) if json_output_path else None,
                    "sample_chunks": [
                        {
                            "chunk_index": c["chunk_index"],
                            "page_start": c["page_start"],
                            "page_end": c["page_end"],
                            "chapter_number": c.get("chapter_number"),
                            "chapter_title": c.get("chapter_title"),
                            "unit_title": c.get("unit_title"),
                            "section_title": c.get("section_title"),
                            "chunk_type": c["chunk_type"],
                            "source_label": c["source_label"],
                            "preview": c["content_clean"][:220],
                        }
                        for c in chunks[:8]
                    ],
                }

            document = self._build_document_record(
                pdf_path=original_pdf_path,
                file_hash=file_hash,
                metadata=metadata,
                book_structure=book_structure,
                chunking_plan=chunking_plan,
                detected_language=detected_language,
                total_pages=pages_count,
                total_words=total_words,
                total_tokens=total_tokens,
                extraction_status="extracted" if chunks else "no_chunks_created",
            )
            document_id = self.repository.upsert_document(document)
            self.repository.insert_book_chapters(document_id, book_structure.chapters)
            self.repository.insert_book_subsections(document_id, book_structure.subsections)
            pages_as_dicts = [asdict(p) for p in pages]
            self.repository.insert_pages(document_id, pages_as_dicts)
            chunks_with_ids = self.repository.insert_chunks(document_id, chunks)
            self.repository.insert_raw_text_pages(document_id, pages_as_dicts, chunks_with_ids, metadata, book_structure=book_structure)

            self.settings.validate_for_embedding()
            embedding_service = OpenAIEmbeddingService(self.settings, self.token_counter)
            embeddings = embedding_service.embed_chunks(chunks_with_ids, document_id)
            self.repository.insert_embeddings(embeddings)
            embeddings_count = len(embeddings)

            if run_id:
                self.repository.finish_ingestion_run(
                    run_id,
                    status="completed",
                    document_id=document_id,
                    pages_extracted=pages_count,
                    chunks_created=chunks_count,
                    embeddings_created=embeddings_count,
                    warnings=warnings,
                )
            return {
                "status": "completed",
                "document_id": document_id,
                "file": str(original_pdf_path),
                "pdf_for_extraction": str(extraction_pdf_path),
                "ocr_preprocessing_used": preprocess_result.used_ocr,
                "ocr_language": preprocess_result.ocr_language,
                "ocr_quality_report": preprocess_result.quality_report.__dict__,
                "file_hash": file_hash,
                "pages_extracted": pages_count,
                "chunks_created": chunks_count,
                "embeddings_created": embeddings_count,
                "book_structure": {
                    "detected_by": book_structure.detected_by,
                    "structures_detected": len(book_structure.chapters),
                    "chapters_detected": len([c for c in book_structure.chapters if c.chapter_title]),
                    "sections_detected": len([c for c in book_structure.chapters if c.section_title]),
                    "subsections_detected": len(book_structure.subsections),
                    "content_profile": book_structure.content_profile,
                },
                "chunking_plan": chunking_plan.to_dict(),
                "warnings": warnings,
                "json_output": str(json_output_path) if json_output_path else None,
                "summary": self.repository.get_document_summary(document_id=document_id),
            }
        except Exception as exc:
            logger.exception("Ingestion failed for %s", original_pdf_path)
            if run_id:
                self.repository.finish_ingestion_run(
                    run_id,
                    status="failed",
                    document_id=document_id,
                    pages_extracted=pages_count,
                    chunks_created=chunks_count,
                    embeddings_created=embeddings_count,
                    error_message=str(exc),
                    warnings=warnings,
                )
            raise

    def ingest_json(
        self,
        json_path: Path,
        metadata: dict[str, Any],
        *,
        reindex: bool = False,
        dry_run: bool = False,
        export_json: bool | None = None,
        output_json_dir: Path | None = None,
        log_page_text: bool | None = None,
    ) -> dict[str, Any]:
        """Ingest pre-extracted JSON and then run the normal embedding flow.

        This is a parallel entry point to ``ingest_pdf``. It skips PDF/OCR/LLM
        extraction because the JSON already contains page/chapter/section text,
        then reuses the same chunker, repositories, and embedding service.

        For JSON ingestion, document identity is ``document_key``. The raw JSON
        content hash is stored only as metadata/audit data so editing the JSON does
        not accidentally create a second document when the logical book is the same.
        """
        json_path = json_path.resolve()
        json_input_hash = sha256_file(json_path)
        reindex = reindex or self.settings.reindex_existing

        run_id: str | None = None
        warnings: list[str] = []
        pages_count = chunks_count = embeddings_count = 0
        document_id: str | None = None
        json_output_path: Path | None = None
        document_key: str | None = None
        file_hash: str | None = None

        try:
            loaded = self.json_input_loader.load(json_path, metadata)
            metadata = loaded.metadata
            book_structure = loaded.book_structure
            pages = loaded.pages
            warnings.extend(loaded.warnings)
            pages_count = len(pages)
            document_key = str(metadata.get("document_key") or "").strip()
            if not document_key:
                raise ValueError("JSON ingestion requires document_key. Add metadata.document_key or document_key at the root.")

            # Keep the legacy file_hash column stable for this logical JSON document.
            # The actual uploaded JSON hash is stored separately as json_input_hash.
            file_hash = _stable_json_document_hash(document_key)
            metadata["document_key"] = document_key
            metadata["json_input_hash"] = json_input_hash
            metadata["json_identity_hash"] = file_hash
            metadata["json_identity_strategy"] = "sha256(json_document_key:<document_key>)"
            metadata["source_json_path"] = str(json_path)
            logger.info("Loaded %s JSON text pages from %s with document_key=%s", pages_count, json_path, document_key)

            if not dry_run:
                existing = self.repository.document_exists_by_document_key(document_key)
                if existing and not reindex:
                    summary = self.repository.get_document_summary(document_key=document_key)
                    return {
                        "status": "skipped_existing",
                        "reason": "same document_key already ingested; use --reindex to replace it",
                        "document_key": document_key,
                        "json_input_hash": json_input_hash,
                        "document": summary,
                    }
                if existing and reindex:
                    logger.info("Reindex requested. Deleting existing JSON document for document_key=%s", document_key)
                    self.repository.delete_document_by_document_key(document_key)

                run_id = self.repository.create_ingestion_run(
                    file_path=str(json_path),
                    file_hash=json_input_hash,
                    document_key=document_key,
                    metadata=metadata,
                )

            metadata = self._merge_detected_metadata(metadata, book_structure)
            metadata["document_key"] = document_key
            metadata["source_json_path"] = str(json_path)
            metadata["json_input_hash"] = json_input_hash
            metadata["json_identity_hash"] = file_hash
            metadata["json_identity_strategy"] = "sha256(json_document_key:<document_key>)"
            chunking_plan = self.chunking_selector.select(pages, metadata, book_structure)
            chunker = self._build_chunker(chunking_plan)
            chunks = chunker.chunk_pages(pages, metadata, book_structure=book_structure)
            chunks_count = len(chunks)
            logger.info(
                "JSON book structure detected_by=%s structures=%s chunks=%s profile=%s",
                book_structure.detected_by,
                len(book_structure.chapters),
                chunks_count,
                chunking_plan.content_profile,
            )
            self._log_page_extractions(pages, book_structure, enabled=log_page_text)
            if self._should_export_json(export_json):
                json_output_path = self.json_exporter.write_combined_extraction(
                    output_dir=output_json_dir or Path(self.settings.json_output_dir),
                    original_pdf_path=json_path,
                    extraction_pdf_path=json_path,
                    metadata=metadata,
                    pages=pages,
                    chunks=chunks,
                    book_structure=book_structure,
                    chunking_plan=chunking_plan,
                    file_hash=file_hash,
                    warnings=warnings,
                    dry_run=dry_run,
                )

            detected_language = self._dominant_language([p.detected_language for p in pages])
            if book_structure.primary_language and detected_language == "Unknown":
                detected_language = book_structure.primary_language
            total_words = sum(p.word_count for p in pages)
            total_tokens = sum(p.token_count for p in pages)

            if dry_run:
                return {
                    "status": "dry_run_ok",
                    "file": str(json_path),
                    "source_type": "json_extraction",
                    "document_key": document_key,
                    "file_hash": file_hash,
                    "json_input_hash": json_input_hash,
                    "pages_loaded": pages_count,
                    "chunks_created": chunks_count,
                    "subsections_detected": len(book_structure.subsections),
                    "detected_language": detected_language,
                    "metadata": metadata,
                    "book_structure": book_structure.to_dict(),
                    "chunking_plan": chunking_plan.to_dict(),
                    "warnings": warnings,
                    "json_output": str(json_output_path) if json_output_path else None,
                    "sample_chunks": [
                        {
                            "chunk_index": c["chunk_index"],
                            "page_start": c["page_start"],
                            "page_end": c["page_end"],
                            "chapter_number": c.get("chapter_number"),
                            "chapter_title": c.get("chapter_title"),
                            "unit_title": c.get("unit_title"),
                            "section_title": c.get("section_title"),
                            "chunk_type": c["chunk_type"],
                            "source_label": c["source_label"],
                            "preview": c["content_clean"][:220],
                        }
                        for c in chunks[:8]
                    ],
                }

            document = self._build_document_record(
                pdf_path=json_path,
                file_hash=file_hash,
                metadata=metadata,
                book_structure=book_structure,
                chunking_plan=chunking_plan,
                detected_language=detected_language,
                total_pages=pages_count,
                total_words=total_words,
                total_tokens=total_tokens,
                extraction_status="json_loaded" if chunks else "json_loaded_no_chunks_created",
                source_type="json_extraction",
                mime_type="application/json",
            )
            document_id = self.repository.upsert_document_by_document_key(document)
            self.repository.insert_book_chapters(document_id, book_structure.chapters)
            self.repository.insert_book_subsections(document_id, book_structure.subsections)
            pages_as_dicts = [asdict(p) for p in pages]
            self.repository.insert_pages(document_id, pages_as_dicts)
            chunks_with_ids = self.repository.insert_chunks(document_id, chunks)
            self.repository.insert_raw_text_pages(document_id, pages_as_dicts, chunks_with_ids, metadata, book_structure=book_structure)

            self.settings.validate_for_embedding()
            embedding_service = OpenAIEmbeddingService(self.settings, self.token_counter)
            embeddings = embedding_service.embed_chunks(chunks_with_ids, document_id)
            self.repository.insert_embeddings(embeddings)
            embeddings_count = len(embeddings)

            if run_id:
                self.repository.finish_ingestion_run(
                    run_id,
                    status="completed",
                    document_id=document_id,
                    pages_extracted=pages_count,
                    chunks_created=chunks_count,
                    embeddings_created=embeddings_count,
                    warnings=warnings,
                )
            return {
                "status": "completed",
                "document_id": document_id,
                "document_key": document_key,
                "file": str(json_path),
                "source_type": "json_extraction",
                "file_hash": file_hash,
                "json_input_hash": json_input_hash,
                "pages_loaded": pages_count,
                "chunks_created": chunks_count,
                "embeddings_created": embeddings_count,
                "book_structure": {
                    "detected_by": book_structure.detected_by,
                    "structures_detected": len(book_structure.chapters),
                    "chapters_detected": len([c for c in book_structure.chapters if c.chapter_title]),
                    "sections_detected": len([c for c in book_structure.chapters if c.section_title]),
                    "subsections_detected": len(book_structure.subsections),
                    "content_profile": book_structure.content_profile,
                },
                "chunking_plan": chunking_plan.to_dict(),
                "warnings": warnings,
                "json_output": str(json_output_path) if json_output_path else None,
                "summary": self.repository.get_document_summary(document_id=document_id),
            }
        except Exception as exc:
            logger.exception("JSON ingestion failed for %s", json_path)
            if run_id:
                self.repository.finish_ingestion_run(
                    run_id,
                    status="failed",
                    document_id=document_id,
                    pages_extracted=pages_count,
                    chunks_created=chunks_count,
                    embeddings_created=embeddings_count,
                    error_message=str(exc),
                    warnings=warnings,
                )
            raise

    def _should_export_json(self, override: bool | None) -> bool:
        return self.settings.export_json_enabled if override is None else override

    def _should_log_page_text(self, override: bool | None) -> bool:
        return self.settings.log_extracted_page_text if override is None else override

    def _log_page_extractions(
        self,
        pages: list[Any],
        book_structure: BookStructure,
        *,
        enabled: bool | None = None,
    ) -> None:
        if not self._should_log_page_text(enabled):
            return
        resolver = ChapterResolver(book_structure.chapters) if book_structure.chapters else None
        max_chars = max(0, self.settings.log_page_text_max_chars)
        for page in pages:
            resolved = resolver.chapter_for_pdf_page(page.page_number) if resolver else None
            printed_page = resolver.printed_page_for_pdf_page(page.page_number) if resolver else None
            structure_label = ""
            if resolved:
                structure_label = (
                    resolved.chapter_title
                    or resolved.section_title
                    or resolved.lesson_title
                    or resolved.unit_title
                    or ""
                )
            text = page.cleaned_text or ""
            display_text = text if max_chars == 0 or len(text) <= max_chars else text[:max_chars] + "\n...[truncated for console log]"
            logger.info(
                "\n========== EXTRACTED PAGE %s%s =========="
                "\nstructure_type=%s | structure=%s | lang=%s | words=%s | tokens=%s | quality=%s | method=%s"
                "\n---------- TEXT ----------\n%s"
                "\n======== END PAGE %s ========",
                page.page_number,
                f" / printed {printed_page}" if printed_page is not None else "",
                resolved.structure_type if resolved else None,
                structure_label or None,
                page.detected_language,
                page.word_count,
                page.token_count,
                page.extraction_quality,
                page.extraction_method,
                display_text,
                page.page_number,
            )

    def _build_chunker(self, plan: ChunkingPlan) -> MeaningfulChunker:
        logger.info("Using chunking plan: %s", plan)
        return MeaningfulChunker(
            token_counter=self.token_counter,
            structure_detector=self.structure_detector,
            metadata_builder=self.metadata_builder,
            max_tokens=plan.max_tokens,
            overlap_tokens=plan.overlap_tokens,
        )

    def _merge_detected_metadata(self, metadata: dict[str, Any], book_structure: Any) -> dict[str, Any]:
        merged = dict(metadata)
        # Folder/CLI metadata wins for school/class/declared subject. LLM fills missing fields.
        for target, value in {
            "title": book_structure.book_title,
            "book_title": book_structure.book_title,
            "subject": book_structure.subject,
            "grade": book_structure.grade,
            "language": book_structure.primary_language,
            "publisher": book_structure.publisher,
            "author": book_structure.author,
            "isbn": book_structure.isbn,
            "edition": book_structure.edition,
            "publication_year": book_structure.publication_year,
        }.items():
            if value and not merged.get(target):
                merged[target] = value
        merged["title"] = merged.get("title") or merged.get("book_title")
        merged["book_title"] = merged.get("book_title") or merged.get("title")
        merged["grade"] = merged.get("grade") or merged.get("class_name")
        merged["class_name"] = merged.get("class_name") or merged.get("grade")
        merged["languages_detected"] = book_structure.languages_detected
        merged["content_profile"] = book_structure.content_profile
        merged["llm_metadata_model"] = self.settings.openai_metadata_model if book_structure.detected_by.startswith("llm:") else None
        merged["llm_metadata_confidence"] = book_structure.confidence
        merged["structure_detected_by"] = book_structure.detected_by
        return merged

    def _build_document_record(
        self,
        *,
        pdf_path: Path,
        file_hash: str,
        metadata: dict[str, Any],
        book_structure: Any,
        chunking_plan: ChunkingPlan,
        detected_language: str,
        total_pages: int,
        total_words: int,
        total_tokens: int,
        extraction_status: str,
        source_type: str = "readable_pdf",
        mime_type: str = "application/pdf",
    ) -> dict[str, Any]:
        title = metadata.get("title") or metadata.get("book_title") or pdf_path.stem
        book_title = metadata.get("book_title") or title
        return {
            "title": title,
            "book_title": book_title,
            "document_key": metadata.get("document_key"),
            "normalized_title": " ".join(title.lower().split()),
            "school_name": metadata.get("school_name"),
            "class_name": metadata.get("class_name"),
            "subject": metadata.get("subject"),
            "grade": metadata.get("grade"),
            "board": metadata.get("board"),
            "medium": metadata.get("medium"),
            "language": metadata.get("language"),
            "detected_language": detected_language,
            "primary_language": book_structure.primary_language or detected_language,
            "languages_detected": metadata.get("languages_detected") or [],
            "publisher": metadata.get("publisher"),
            "edition": metadata.get("edition"),
            "publication_year": metadata.get("publication_year"),
            "isbn": metadata.get("isbn"),
            "author": metadata.get("author"),
            "source_type": source_type,
            "source_uri": metadata.get("source_uri"),
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "file_hash": file_hash,
            "file_size_bytes": pdf_path.stat().st_size,
            "mime_type": mime_type,
            "total_pages": total_pages,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "extraction_status": extraction_status,
            "copyright_status": metadata.get("copyright_status"),
            "license_notes": metadata.get("license_notes"),
            "llm_metadata_model": metadata.get("llm_metadata_model"),
            "llm_metadata_confidence": metadata.get("llm_metadata_confidence"),
            "structure_detected_by": metadata.get("structure_detected_by"),
            "content_profile": chunking_plan.content_profile,
            "chunking_strategy": chunking_plan.strategy,
            "chunk_max_tokens": chunking_plan.max_tokens,
            "chunk_overlap_tokens": chunking_plan.overlap_tokens,
            "metadata": {
                "pipeline": "pdf_embedding_pipeline",
                "document_key": metadata.get("document_key"),
                "school_name": metadata.get("school_name"),
                "class_name": metadata.get("class_name"),
                "book_title": book_title,
                "subject": metadata.get("subject"),
                "grade": metadata.get("grade"),
                "path_metadata_source": metadata.get("path_metadata_source"),
                "json_input_hash": metadata.get("json_input_hash"),
                "json_identity_hash": metadata.get("json_identity_hash"),
                "embedding_model": self.settings.openai_embedding_model,
                "embedding_dimensions": self.settings.openai_embedding_dimensions,
                "metadata_model": self.settings.openai_metadata_model,
                "book_structure": book_structure.to_dict(),
                "chunking_plan": chunking_plan.to_dict(),
            },
        }

    def _dominant_language(self, languages: list[str]) -> str:
        counts = {lang: languages.count(lang) for lang in set(languages) if lang and lang != "Unknown"}
        if not counts:
            return "Unknown"
        if "Hindi" in counts and "English" in counts:
            return "Mixed"
        return max(counts.items(), key=lambda x: x[1])[0]
