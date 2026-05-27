from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.language_detector import LanguageDetector
from ingestion.structure_detector import StructureDetector
from ingestion.text_cleaner import TextCleaner
from utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    page_number: int
    raw_text: str
    cleaned_text: str
    detected_language: str
    word_count: int
    token_count: int
    has_text: bool
    has_math: bool
    has_table_like_text: bool
    has_devanagari: bool
    has_english: bool
    extraction_method: str
    extraction_quality: str
    metadata: dict = field(default_factory=dict)


class PDFTextExtractor:
    def __init__(self, cleaner: TextCleaner, language_detector: LanguageDetector, token_counter: TokenCounter) -> None:
        self.cleaner = cleaner
        self.language_detector = language_detector
        self.token_counter = token_counter
        self.structure_detector = StructureDetector()

    def extract(self, pdf_path: Path) -> tuple[list[ExtractedPage], list[str]]:
        warnings: list[str] = []
        try:
            pages = self._extract_with_pymupdf(pdf_path)
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed for %s: %s. Falling back to pypdf.", pdf_path, exc)
            pages = self._extract_with_pypdf(pdf_path)

        empty_pages = sum(1 for p in pages if not p.has_text)
        if pages and empty_pages / len(pages) >= 0.40:
            warning = "This appears to be scanned PDF. OCR is required before embedding."
            logger.warning(warning)
            warnings.append(warning)
        return pages, warnings

    def _extract_with_pymupdf(self, pdf_path: Path) -> list[ExtractedPage]:
        import fitz

        pages: list[ExtractedPage] = []
        with fitz.open(pdf_path) as doc:
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                pages.append(self._build_page(index, text, "pymupdf"))
        return pages

    def _extract_with_pypdf(self, pdf_path: Path) -> list[ExtractedPage]:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: list[ExtractedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(self._build_page(index, text, "pypdf"))
        return pages

    def _build_page(self, page_number: int, raw_text: str, method: str) -> ExtractedPage:
        clean = self.cleaner.clean(raw_text)
        text = clean.cleaned_text
        stats = self.language_detector.detect_with_stats(text)
        classification = self.structure_detector.classify(text)
        word_count = len(text.split()) if text else 0
        token_count = self.token_counter.count(text)
        has_text = bool(text.strip()) and token_count > 3
        quality = "empty" if not has_text else "low" if token_count < 20 else "ok"
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
            extraction_method=method,
            extraction_quality=quality,
            metadata={"cleaning_notes": clean.notes},
        )
