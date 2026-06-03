from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    openai_metadata_model: str
    metadata_sample_pages: int
    metadata_max_output_tokens: int
    auto_metadata_enabled: bool
    auto_chunking_enabled: bool
    database_url: str
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    embedding_batch_size: int
    reindex_existing: bool
    log_level: str
    project_root: Path

    # Optional JSON/debug output generated in addition to DB persistence.
    export_json_enabled: bool
    json_output_dir: str
    log_extracted_page_text: bool
    log_page_text_max_chars: int

    # PDF/OCR preprocessing. OCRmyPDF is used before text extraction when the
    # PDF is scanned or has a corrupted text layer.
    pdf_preprocess_with_ocrmypdf: bool
    pdf_preprocess_mode: str
    pdf_ocrmypdf_strategy: str
    pdf_ocr_cache_dir: str
    pdf_ocr_lang: str
    pdf_ocr_dpi: int
    pdf_ocr_jobs: int | None
    pdf_ocr_optimize: int
    pdf_ocr_output_type: str
    pdf_ocr_deskew: bool
    pdf_ocr_rotate_pages: bool
    pdf_ocr_clean: bool
    pdf_ocr_invalidate_cache: bool
    pdf_quality_sample_pages: int

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_embedding_dimensions=_as_int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS"), 3072),
            # Change this freely to gpt-4o-mini, gpt-5.4-mini, etc.; the code is model-name agnostic.
            openai_metadata_model=os.getenv("OPENAI_METADATA_MODEL", "gpt-5.4"),
            metadata_sample_pages=_as_int(os.getenv("METADATA_SAMPLE_PAGES"), 20),
            metadata_max_output_tokens=_as_int(os.getenv("METADATA_MAX_OUTPUT_TOKENS"), 6000),
            auto_metadata_enabled=_as_bool(os.getenv("AUTO_METADATA_ENABLED"), True),
            auto_chunking_enabled=_as_bool(os.getenv("AUTO_CHUNKING_ENABLED"), True),
            database_url=os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/pdf_rag_db"),
            chunk_max_tokens=_as_int(os.getenv("DEFAULT_CHUNK_MAX_TOKENS"), 750),
            chunk_overlap_tokens=_as_int(os.getenv("DEFAULT_CHUNK_OVERLAP_TOKENS"), 120),
            embedding_batch_size=_as_int(os.getenv("EMBEDDING_BATCH_SIZE"), 64),
            reindex_existing=_as_bool(os.getenv("REINDEX_EXISTING"), False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            project_root=Path(__file__).resolve().parent,
            export_json_enabled=_as_bool(os.getenv("EXPORT_JSON_ENABLED"), True),
            json_output_dir=os.getenv("JSON_OUTPUT_DIR", "output/json_exports"),
            log_extracted_page_text=_as_bool(os.getenv("LOG_EXTRACTED_PAGE_TEXT"), False),
            log_page_text_max_chars=_as_int(os.getenv("LOG_PAGE_TEXT_MAX_CHARS"), 12000),
            pdf_preprocess_with_ocrmypdf=_as_bool(os.getenv("PDF_PREPROCESS_WITH_OCRMYPDF"), True),
            pdf_preprocess_mode=os.getenv("PDF_PREPROCESS_MODE", "auto").strip().lower(),
            pdf_ocrmypdf_strategy=os.getenv("PDF_OCRMYPDF_STRATEGY", "force_ocr").strip().lower(),
            pdf_ocr_cache_dir=os.getenv("PDF_OCR_CACHE_DIR", ".ocr_cache"),
            pdf_ocr_lang=os.getenv("PDF_OCR_LANG", "auto").strip() or "auto",
            pdf_ocr_dpi=_as_int(os.getenv("PDF_OCR_DPI"), 300),
            pdf_ocr_jobs=None if not os.getenv("PDF_OCR_JOBS") else _as_int(os.getenv("PDF_OCR_JOBS"), 0),
            pdf_ocr_optimize=_as_int(os.getenv("PDF_OCR_OPTIMIZE"), 1),
            pdf_ocr_output_type=os.getenv("PDF_OCR_OUTPUT_TYPE", "pdf").strip() or "pdf",
            pdf_ocr_deskew=_as_bool(os.getenv("PDF_OCR_DESKEW"), True),
            pdf_ocr_rotate_pages=_as_bool(os.getenv("PDF_OCR_ROTATE_PAGES"), True),
            pdf_ocr_clean=_as_bool(os.getenv("PDF_OCR_CLEAN"), False),
            pdf_ocr_invalidate_cache=_as_bool(os.getenv("PDF_OCR_INVALIDATE_CACHE"), False),
            pdf_quality_sample_pages=_as_int(os.getenv("PDF_QUALITY_SAMPLE_PAGES"), 25),
        )

    def validate_for_embedding(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for ingestion/search that calls the OpenAI embeddings API.")
        if self.openai_embedding_model != "text-embedding-3-large":
            raise ValueError("This project requires OPENAI_EMBEDDING_MODEL=text-embedding-3-large.")
        if self.openai_embedding_dimensions != 3072:
            raise ValueError("This project stores pgvector vector(3072). Set OPENAI_EMBEDDING_DIMENSIONS=3072.")
