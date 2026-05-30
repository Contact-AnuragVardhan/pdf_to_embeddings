from __future__ import annotations

import logging
import os
import re
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
    """Extract text from a readable/searchable PDF.

    Recommended production flow:
      Original PDF -> PdfPreprocessor/OCRmyPDF -> searchable PDF -> this extractor.

    This extractor still has optional page OCR fallback, but the best quality comes
    from OCRmyPDF preprocessing because it deskews/rotates/repairs the full PDF first.
    """

    C1_CONTROL_RE = re.compile(r"[\x80-\x9F]")
    DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
    LATIN_RE = re.compile(r"[A-Za-z]")
    MATH_SYMBOL_RE = re.compile(r"[=+×÷≤≥<>√π%−–—]|\d+\s*/\s*\d+|\b\d+(?:\.\d+)?\b")

    def __init__(self, cleaner: TextCleaner, language_detector: LanguageDetector, token_counter: TokenCounter) -> None:
        self.cleaner = cleaner
        self.language_detector = language_detector
        self.token_counter = token_counter
        self.structure_detector = StructureDetector()
        self.text_mode = os.getenv("PDF_TEXT_EXTRACTION_MODE", "native").strip().lower()
        self.ocr_dpi = int(os.getenv("PDF_OCR_DPI", "300"))
        self.ocr_lang = os.getenv("PDF_OCR_LANG", "eng").strip() or "eng"
        if self.ocr_lang.lower() == "auto":
            self.ocr_lang = "eng"

    def extract(self, pdf_path: Path) -> tuple[list[ExtractedPage], list[str]]:
        warnings: list[str] = []
        try:
            pages = self._extract_with_pymupdf(pdf_path)
        except Exception as exc:
            logger.warning("PyMuPDF extraction failed for %s: %s. Falling back to pypdf.", pdf_path, exc)
            pages = self._extract_with_pypdf(pdf_path)

        empty_pages = sum(1 for p in pages if not p.has_text)
        low_pages = sum(1 for p in pages if p.extraction_quality in {"empty", "low", "corrupted"})
        if pages and empty_pages / len(pages) >= 0.40:
            warning = "This appears to be scanned PDF. Enable OCRmyPDF preprocessing before embedding."
            logger.warning(warning)
            warnings.append(warning)
        if pages and low_pages / len(pages) >= 0.25:
            warning = "Many pages have low/corrupted extracted text. Re-run with PDF_PREPROCESS_MODE=always."
            logger.warning(warning)
            warnings.append(warning)
        return pages, warnings

    def _extract_with_pymupdf(self, pdf_path: Path) -> list[ExtractedPage]:
        import fitz

        pages: list[ExtractedPage] = []
        with fitz.open(pdf_path) as doc:
            for index, page in enumerate(doc, start=1):
                native_text = self._extract_page_text_best_effort(page)
                text = native_text
                method = "pymupdf_text_layer"

                if self._should_page_ocr(native_text):
                    ocr_text = self._ocr_page(page)
                    if self._is_ocr_better(native_text, ocr_text):
                        text = ocr_text
                        method = "pymupdf_page_ocr"
                    else:
                        method = "pymupdf_text_layer_ocr_rejected"

                pages.append(self._build_page(index, text, method, native_text=native_text))
        return pages

    def _extract_page_text_best_effort(self, page) -> str:
        # sort=True improves normal reading order in multi-column or scanned OCR layers.
        text = page.get_text("text", sort=True) or ""
        if text.strip():
            return text

        # Fallback: blocks sometimes returns text when plain text extraction is empty.
        try:
            blocks = page.get_text("blocks", sort=True) or []
            block_texts: list[str] = []
            for block in blocks:
                if len(block) >= 5 and isinstance(block[4], str) and block[4].strip():
                    block_texts.append(block[4].strip())
            return "\n".join(block_texts)
        except Exception:
            return text

    def _extract_with_pypdf(self, pdf_path: Path) -> list[ExtractedPage]:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: list[ExtractedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(self._build_page(index, text, "pypdf"))
        return pages

    def _should_page_ocr(self, native_text: str) -> bool:
        # Page-level OCR is a fallback only. Prefer PdfPreprocessor/OCRmyPDF.
        if self.text_mode in {"ocr", "force_ocr", "always_ocr"}:
            return True
        if self.text_mode not in {"auto_ocr", "auto", "ocr_fallback"}:
            return False

        if not native_text.strip():
            return True

        latin = len(self.LATIN_RE.findall(native_text))
        devanagari = len(self.DEVANAGARI_RE.findall(native_text))
        c1_controls = len(self.C1_CONTROL_RE.findall(native_text))
        replacement = native_text.count("�")

        if c1_controls >= 1 or replacement >= 1:
            return True

        # English math books should not suddenly contain isolated Devanagari glyphs.
        if self.ocr_lang == "eng" and latin >= 120 and 2 <= devanagari <= max(40, int(latin * 0.20)):
            return True

        return False

    def _ocr_page(self, page) -> str:
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except Exception as exc:
            logger.warning(
                "Page OCR requested but pytesseract/Pillow/Tesseract is not configured: %s",
                exc,
            )
            return ""

        zoom = self.ocr_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # psm 6 works for textbook pages; preserve_interword_spaces helps formulas/tables.
        config = "--psm 6 -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(image, lang=self.ocr_lang, config=config) or ""

    def _is_ocr_better(self, native_text: str, ocr_text: str) -> bool:
        if not ocr_text or len(ocr_text.strip()) < 30:
            return False
        if not native_text.strip():
            return True

        native_score = self._text_quality_score(native_text)
        ocr_score = self._text_quality_score(ocr_text)
        native_words = len(native_text.split())
        ocr_words = len(ocr_text.split())

        # OCR must be close in amount of text and meaningfully cleaner.
        return ocr_words >= max(20, int(native_words * 0.70)) and ocr_score > native_score + 5

    def _text_quality_score(self, text: str) -> int:
        if not text.strip():
            return 0
        score = 0
        score += min(len(text.split()), 300)
        score += min(len(self.LATIN_RE.findall(text)), 500) // 10
        score += min(len(self.MATH_SYMBOL_RE.findall(text)), 120)
        score -= (len(self.C1_CONTROL_RE.findall(text)) + text.count("�")) * 50
        if self.ocr_lang == "eng":
            score -= len(self.DEVANAGARI_RE.findall(text)) * 8
        return score

    def _build_page(self, page_number: int, raw_text: str, method: str, *, native_text: str | None = None) -> ExtractedPage:
        clean = self.cleaner.clean(raw_text)
        text = clean.cleaned_text
        stats = self.language_detector.detect_with_stats(text)
        classification = self.structure_detector.classify(text)
        word_count = len(text.split()) if text else 0
        token_count = self.token_counter.count(text)
        has_text = bool(text.strip()) and token_count > 3

        garbage_count = len(self.C1_CONTROL_RE.findall(text)) + text.count("�")
        unexpected_devanagari = stats.devanagari_chars > 0 and stats.latin_chars > 120 and self.ocr_lang == "eng"
        if not has_text:
            quality = "empty"
        elif garbage_count > 0 or unexpected_devanagari:
            quality = "corrupted"
        elif token_count < 20:
            quality = "low"
        elif method.endswith("ocr"):
            quality = "ocr"
        else:
            quality = "ok"

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
            metadata={
                "cleaning_notes": clean.notes,
                "native_text_length": len(native_text or "") if native_text is not None else None,
                "garbage_char_count": garbage_count,
            },
        )
