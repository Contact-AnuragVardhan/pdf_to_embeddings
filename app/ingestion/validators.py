from __future__ import annotations

from pathlib import Path
from typing import Any


VALID_LANGUAGES = {"Hindi", "English", "Mixed", "Hinglish", "Unknown", None}


def validate_pdf_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")


def validate_ingest_args(pdf_path: Path, metadata: dict[str, Any]) -> None:
    validate_pdf_file(pdf_path)
    if not metadata.get("title"):
        raise ValueError("--title is required unless using ingest-folder, where the PDF file name is used.")
    language = metadata.get("language")
    if language not in VALID_LANGUAGES:
        raise ValueError(f"Unsupported --language={language}. Use Hindi, English, Mixed, Hinglish, or omit it.")
