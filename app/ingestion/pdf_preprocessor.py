from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfTextQualityReport:
    total_pages: int
    sampled_pages: int
    empty_pages: int
    pages_with_text: int
    pages_with_control_garbage: int
    pages_with_replacement_chars: int
    pages_with_unexpected_devanagari: int
    avg_chars_per_text_page: float
    should_ocr: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PdfPreprocessResult:
    original_pdf: Path
    pdf_for_extraction: Path
    used_ocr: bool
    ocr_language: str
    ocr_output_pdf: Path | None
    quality_report: PdfTextQualityReport
    warnings: list[str]


@dataclass(frozen=True)
class OcrMyPdfOptions:
    enabled: bool = True
    mode: str = "auto"  # auto | always | never
    ocr_strategy: str = "force_ocr"  # force_ocr | redo_ocr | skip_text
    output_dir: Path = Path(".ocr_cache")
    language: str = "auto"  # auto | eng | hin+eng
    dpi: int = 300
    jobs: int | None = None
    optimize: int = 1
    output_type: str = "pdf"
    deskew: bool = True
    rotate_pages: bool = True
    clean: bool = False
    invalidate_cache: bool = False
    sample_pages: int = 25


class PdfPreprocessor:
    """Prepare a PDF for best text extraction.

    The main idea is:
      1. If the PDF already has a good selectable text layer, keep it.
      2. If it is scanned OR has a bad/corrupted text layer, run OCRmyPDF first.
      3. Extract text from the OCRmyPDF-created searchable PDF.

    This is more reliable than doing page-by-page OCR inside PyMuPDF because OCRmyPDF
    handles deskew, rotation, PDF repair, and creates a proper searchable text layer.
    """

    C1_CONTROL_RE = re.compile(r"[\x80-\x9F]")
    REPLACEMENT_RE = re.compile(r"�")
    DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
    LATIN_RE = re.compile(r"[A-Za-z]")

    def __init__(self, options: OcrMyPdfOptions) -> None:
        self.options = options

    @classmethod
    def from_settings(cls, settings: Any) -> "PdfPreprocessor":
        return cls(
            OcrMyPdfOptions(
                enabled=getattr(settings, "pdf_preprocess_with_ocrmypdf", True),
                mode=getattr(settings, "pdf_preprocess_mode", "auto"),
                ocr_strategy=getattr(settings, "pdf_ocrmypdf_strategy", "force_ocr"),
                output_dir=Path(getattr(settings, "pdf_ocr_cache_dir", ".ocr_cache")),
                language=getattr(settings, "pdf_ocr_lang", "auto"),
                dpi=getattr(settings, "pdf_ocr_dpi", 300),
                jobs=getattr(settings, "pdf_ocr_jobs", None),
                optimize=getattr(settings, "pdf_ocr_optimize", 1),
                output_type=getattr(settings, "pdf_ocr_output_type", "pdf"),
                deskew=getattr(settings, "pdf_ocr_deskew", True),
                rotate_pages=getattr(settings, "pdf_ocr_rotate_pages", True),
                clean=getattr(settings, "pdf_ocr_clean", False),
                invalidate_cache=getattr(settings, "pdf_ocr_invalidate_cache", False),
                sample_pages=getattr(settings, "pdf_quality_sample_pages", 25),
            )
        )

    def prepare(self, pdf_path: Path, metadata: dict[str, Any] | None = None) -> PdfPreprocessResult:
        metadata = metadata or {}
        original_pdf = pdf_path.expanduser().resolve()
        if not original_pdf.exists():
            raise FileNotFoundError(f"PDF not found: {original_pdf}")

        ocr_language = self.resolve_ocr_language(metadata)
        quality = self.inspect_text_quality(original_pdf, metadata=metadata, ocr_language=ocr_language)
        warnings: list[str] = []

        mode = (self.options.mode or "auto").strip().lower()
        enabled = self.options.enabled and mode not in {"never", "off", "false", "0"}
        must_ocr = enabled and mode in {"always", "force", "ocr"}
        should_ocr = enabled and (must_ocr or quality.should_ocr)

        if not enabled:
            return PdfPreprocessResult(original_pdf, original_pdf, False, ocr_language, None, quality, warnings)

        if not should_ocr:
            return PdfPreprocessResult(original_pdf, original_pdf, False, ocr_language, None, quality, warnings)

        self._check_dependencies(ocr_language)
        output_pdf = self._output_path(original_pdf, ocr_language)

        if output_pdf.exists() and not self.options.invalidate_cache:
            logger.info("Using cached OCR PDF: %s", output_pdf)
            return PdfPreprocessResult(original_pdf, output_pdf, True, ocr_language, output_pdf, quality, warnings)

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        command = self._build_ocrmypdf_command(original_pdf, output_pdf, ocr_language)
        logger.info("Running OCRmyPDF: %s", self._safe_command(command))

        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "OCRmyPDF failed. Fix Tesseract/Ghostscript/qpdf installation or set "
                "PDF_PREPROCESS_MODE=never to skip OCR.\n\n"
                f"Command: {self._safe_command(command)}\n\n{error_text}"
            )

        if result.stderr:
            logger.debug("OCRmyPDF stderr: %s", result.stderr.strip())
        if result.stdout:
            logger.debug("OCRmyPDF stdout: %s", result.stdout.strip())

        return PdfPreprocessResult(original_pdf, output_pdf, True, ocr_language, output_pdf, quality, warnings)

    def resolve_ocr_language(self, metadata: dict[str, Any]) -> str:
        configured = (self.options.language or "auto").strip()
        if configured and configured.lower() != "auto":
            return configured

        values = " ".join(
            str(metadata.get(k) or "")
            for k in ("language", "medium", "subject", "title", "book_title")
        ).lower()

        # Hindi books often still contain English numbers/formulas, so use both.
        if any(token in values for token in ["hindi", "हिंदी", "हिन्दी", "devanagari", "sanskrit"]):
            return "hin+eng"

        # Maths/science books in English should NOT use hin+eng because it can turn
        # symbols/1/-1/₹ into Devanagari-looking garbage.
        return "eng"

    def inspect_text_quality(
        self,
        pdf_path: Path,
        *,
        metadata: dict[str, Any] | None = None,
        ocr_language: str = "eng",
    ) -> PdfTextQualityReport:
        import fitz

        metadata = metadata or {}
        allow_devanagari = "hin" in (ocr_language or "").lower() or self._metadata_looks_hindi(metadata)
        sample_limit = max(1, int(self.options.sample_pages or 25))

        total_pages = 0
        sampled_pages = 0
        empty_pages = 0
        pages_with_text = 0
        pages_with_control_garbage = 0
        pages_with_replacement_chars = 0
        pages_with_unexpected_devanagari = 0
        text_lengths: list[int] = []
        sample_details: list[dict[str, Any]] = []

        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            candidate_indexes = self._sample_indexes(total_pages, sample_limit)
            for page_index in candidate_indexes:
                page = doc[page_index]
                text = page.get_text("text", sort=True) or ""
                sampled_pages += 1
                stripped = text.strip()
                text_len = len(stripped)
                if not stripped:
                    empty_pages += 1
                else:
                    pages_with_text += 1
                    text_lengths.append(text_len)

                control_count = len(self.C1_CONTROL_RE.findall(text))
                replacement_count = len(self.REPLACEMENT_RE.findall(text))
                latin_count = len(self.LATIN_RE.findall(text))
                devanagari_count = len(self.DEVANAGARI_RE.findall(text))

                if control_count >= 1:
                    pages_with_control_garbage += 1
                if replacement_count >= 1:
                    pages_with_replacement_chars += 1
                unexpected_devanagari = (
                    not allow_devanagari
                    and latin_count >= 80
                    and devanagari_count >= 2
                    and devanagari_count <= max(40, int(latin_count * 0.20))
                )
                if unexpected_devanagari:
                    pages_with_unexpected_devanagari += 1

                if len(sample_details) < 8:
                    sample_details.append(
                        {
                            "page": page_index + 1,
                            "text_len": text_len,
                            "latin": latin_count,
                            "devanagari": devanagari_count,
                            "control": control_count,
                            "replacement": replacement_count,
                            "preview": stripped[:220],
                        }
                    )

        avg_chars = sum(text_lengths) / max(len(text_lengths), 1)
        empty_ratio = empty_pages / max(sampled_pages, 1)
        corrupted_pages = pages_with_control_garbage + pages_with_replacement_chars + pages_with_unexpected_devanagari
        corrupted_ratio = corrupted_pages / max(sampled_pages, 1)

        should_ocr = False
        reasons: list[str] = []
        if empty_ratio >= 0.35:
            should_ocr = True
            reasons.append(f"{empty_pages}/{sampled_pages} sampled pages have no text")
        if pages_with_text > 0 and avg_chars < 120:
            should_ocr = True
            reasons.append(f"very little extracted text, avg {avg_chars:.0f} chars/page")
        if corrupted_ratio >= 0.12:
            should_ocr = True
            reasons.append(f"{corrupted_pages}/{sampled_pages} sampled pages look corrupted")
        if pages_with_control_garbage or pages_with_replacement_chars:
            should_ocr = True
            reasons.append("control/replacement characters found")
        if pages_with_unexpected_devanagari >= 2:
            should_ocr = True
            reasons.append("unexpected Devanagari glyphs found in English-looking pages")

        return PdfTextQualityReport(
            total_pages=total_pages,
            sampled_pages=sampled_pages,
            empty_pages=empty_pages,
            pages_with_text=pages_with_text,
            pages_with_control_garbage=pages_with_control_garbage,
            pages_with_replacement_chars=pages_with_replacement_chars,
            pages_with_unexpected_devanagari=pages_with_unexpected_devanagari,
            avg_chars_per_text_page=round(avg_chars, 2),
            should_ocr=should_ocr,
            reason="; ".join(reasons) if reasons else "existing text layer looks usable",
            details={"samples": sample_details, "allow_devanagari": allow_devanagari},
        )

    def _metadata_looks_hindi(self, metadata: dict[str, Any]) -> bool:
        joined = " ".join(str(v or "") for v in metadata.values()).lower()
        return any(token in joined for token in ["hindi", "हिंदी", "हिन्दी", "devanagari"])

    def _sample_indexes(self, total_pages: int, sample_limit: int) -> list[int]:
        if total_pages <= 0:
            return []
        # First pages + evenly spaced pages. This catches contents and chapter pages.
        first = list(range(min(total_pages, min(10, sample_limit))))
        remaining = sample_limit - len(first)
        if remaining <= 0:
            return first
        if total_pages <= len(first):
            return first
        step = max(1, (total_pages - 1) // remaining)
        spaced = list(range(len(first), total_pages, step))[:remaining]
        return sorted(set(first + spaced))[:sample_limit]

    def _output_path(self, input_pdf: Path, language: str) -> Path:
        safe_lang = re.sub(r"[^A-Za-z0-9_-]+", "_", language)
        output_dir = self.options.output_dir.expanduser().resolve()
        return output_dir / f"{input_pdf.stem}.ocr-{safe_lang}.pdf"

    def _build_ocrmypdf_command(self, input_pdf: Path, output_pdf: Path, language: str) -> list[str]:
        command = [sys.executable, "-m", "ocrmypdf", "--language", language]

        strategy = (self.options.ocr_strategy or "force_ocr").strip().lower().replace("-", "_")
        if strategy in {"force", "force_ocr", "always"}:
            command.append("--force-ocr")
        elif strategy in {"redo", "redo_ocr"}:
            command.append("--redo-ocr")
        elif strategy in {"skip", "skip_text", "skip_text_pages"}:
            command.append("--skip-text")
        else:
            raise ValueError(f"Invalid PDF_OCRMYPDF_STRATEGY: {self.options.ocr_strategy}")

        if self.options.deskew:
            command.append("--deskew")
        if self.options.rotate_pages:
            command.append("--rotate-pages")
        if self.options.clean:
            command.append("--clean")
        if self.options.optimize is not None:
            command.extend(["--optimize", str(self.options.optimize)])
        if self.options.output_type:
            command.extend(["--output-type", self.options.output_type])
        if self.options.jobs:
            command.extend(["--jobs", str(self.options.jobs)])

        # OCRmyPDF estimates resolution from PDF. For image-only books, --oversample
        # improves tiny math symbols without changing pages that already have enough DPI.
        if self.options.dpi and self.options.dpi >= 150:
            command.extend(["--oversample", str(self.options.dpi)])

        command.extend([str(input_pdf), str(output_pdf)])
        return command

    def _check_dependencies(self, language: str) -> None:
        if shutil.which("tesseract") is None:
            raise RuntimeError("Tesseract OCR is not found in PATH. Install it and reopen the terminal/PyCharm.")

        # OCRmyPDF finds Ghostscript as gs/gswin64c depending on OS.
        if os.name == "nt":
            gs_ok = shutil.which("gswin64c") or shutil.which("gswin32c") or shutil.which("gs")
        else:
            gs_ok = shutil.which("gs")
        if gs_ok is None:
            raise RuntimeError("Ghostscript is not found in PATH. OCRmyPDF needs Ghostscript.")

        try:
            import ocrmypdf  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Python package ocrmypdf is missing. Run: pip install ocrmypdf") from exc

        result = subprocess.run(["tesseract", "--list-langs"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        langs_output = (result.stdout or result.stderr or "").lower()
        needed = [part.strip().lower() for part in language.split("+") if part.strip()]
        missing = [lang for lang in needed if lang not in langs_output]
        if missing:
            raise RuntimeError(
                f"Tesseract language data missing: {', '.join(missing)}. "
                f"Available output was:\n{langs_output}"
            )

    def _safe_command(self, command: list[str]) -> str:
        return " ".join(f'"{part}"' if " " in part else part for part in command)
