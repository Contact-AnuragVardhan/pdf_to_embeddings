from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.progress import Progress

from config import Settings
from ingestion.chunker import MeaningfulChunker
from ingestion.embedding_service import OpenAIEmbeddingService
from ingestion.language_detector import LanguageDetector
from ingestion.metadata_builder import MetadataBuilder
from ingestion.pdf_extractor import PDFTextExtractor
from ingestion.repository import RagRepository
from ingestion.structure_detector import StructureDetector
from ingestion.text_cleaner import TextCleaner
from utils.hashing import sha256_file
from utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


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
        self.chunker = MeaningfulChunker(
            token_counter=self.token_counter,
            structure_detector=self.structure_detector,
            metadata_builder=self.metadata_builder,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    def ingest_pdf(self, pdf_path: Path, metadata: dict[str, Any], *, reindex: bool = False, dry_run: bool = False) -> dict[str, Any]:
        pdf_path = pdf_path.resolve()
        file_hash = sha256_file(pdf_path)
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
        try:
            if not dry_run:
                run_id = self.repository.create_ingestion_run(file_path=str(pdf_path), file_hash=file_hash, metadata=metadata)

            pages, extraction_warnings = self.extractor.extract(pdf_path)
            warnings.extend(extraction_warnings)
            pages_count = len(pages)
            chunks = self.chunker.chunk_pages(pages, metadata)
            chunks_count = len(chunks)

            detected_language = self._dominant_language([p.detected_language for p in pages])
            total_words = sum(p.word_count for p in pages)
            total_tokens = sum(p.token_count for p in pages)

            if dry_run:
                return {
                    "status": "dry_run_ok",
                    "file": str(pdf_path),
                    "file_hash": file_hash,
                    "pages_extracted": pages_count,
                    "chunks_created": chunks_count,
                    "detected_language": detected_language,
                    "warnings": warnings,
                    "sample_chunks": [
                        {
                            "chunk_index": c["chunk_index"],
                            "page_start": c["page_start"],
                            "page_end": c["page_end"],
                            "chunk_type": c["chunk_type"],
                            "source_label": c["source_label"],
                            "preview": c["content_clean"][:220],
                        }
                        for c in chunks[:5]
                    ],
                }

            document = self._build_document_record(
                pdf_path=pdf_path,
                file_hash=file_hash,
                metadata=metadata,
                detected_language=detected_language,
                total_pages=pages_count,
                total_words=total_words,
                total_tokens=total_tokens,
                extraction_status="extracted" if chunks else "no_chunks_created",
            )
            document_id = self.repository.upsert_document(document)
            self.repository.insert_pages(document_id, [asdict(p) for p in pages])
            chunks_with_ids = self.repository.insert_chunks(document_id, chunks)

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
                "file": str(pdf_path),
                "file_hash": file_hash,
                "pages_extracted": pages_count,
                "chunks_created": chunks_count,
                "embeddings_created": embeddings_count,
                "warnings": warnings,
                "summary": self.repository.get_document_summary(document_id=document_id),
            }
        except Exception as exc:
            logger.exception("Ingestion failed for %s", pdf_path)
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

    def _build_document_record(
        self,
        *,
        pdf_path: Path,
        file_hash: str,
        metadata: dict[str, Any],
        detected_language: str,
        total_pages: int,
        total_words: int,
        total_tokens: int,
        extraction_status: str,
    ) -> dict[str, Any]:
        title = metadata.get("title") or pdf_path.stem
        return {
            "title": title,
            "normalized_title": " ".join(title.lower().split()),
            "subject": metadata.get("subject"),
            "grade": metadata.get("grade"),
            "board": metadata.get("board"),
            "medium": metadata.get("medium"),
            "language": metadata.get("language"),
            "detected_language": detected_language,
            "publisher": metadata.get("publisher"),
            "edition": metadata.get("edition"),
            "publication_year": metadata.get("publication_year"),
            "isbn": metadata.get("isbn"),
            "author": metadata.get("author"),
            "source_type": "readable_pdf",
            "source_uri": metadata.get("source_uri"),
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "file_hash": file_hash,
            "file_size_bytes": pdf_path.stat().st_size,
            "mime_type": "application/pdf",
            "total_pages": total_pages,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "extraction_status": extraction_status,
            "copyright_status": metadata.get("copyright_status"),
            "license_notes": metadata.get("license_notes"),
            "metadata": {
                "pipeline": "pdf_embedding_pipeline",
                "embedding_model": self.settings.openai_embedding_model,
                "embedding_dimensions": self.settings.openai_embedding_dimensions,
                "chunk_max_tokens": self.settings.chunk_max_tokens,
                "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
            },
        }

    def _dominant_language(self, languages: list[str]) -> str:
        counts = {lang: languages.count(lang) for lang in set(languages) if lang and lang != "Unknown"}
        if not counts:
            return "Unknown"
        if "Hindi" in counts and "English" in counts:
            return "Mixed"
        return max(counts.items(), key=lambda x: x[1])[0]
