from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


def _normalized_parts(path: Path | str) -> list[str]:
    """Return path parts in a way that also understands Windows-style backslashes.

    This lets dry-runs/tests parse examples like
    input\\Mother Miracle School\\Class-7\\Maths_RSAgarwal.pdf even on Linux/macOS.
    On Windows, pathlib already handles backslashes, but this normalization is harmless.
    """
    raw = str(path).replace("\\", "/")
    return [part for part in PurePosixPath(raw).parts if part not in {"", "."}]


def _relative_parts(pdf_path: Path, input_root: Path | None = None) -> list[str]:
    if input_root is not None:
        try:
            return _normalized_parts(pdf_path.resolve().relative_to(input_root.resolve()))
        except Exception:
            pass
    return _normalized_parts(pdf_path)


def _clean_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def parse_book_filename(pdf_path: Path) -> dict[str, str | None]:
    """Parse <Subject>_<Title Of Book>.pdf.

    Examples:
      Maths_RSAgarwal.pdf -> subject=Maths, title=RSAgarwal
      English_Honeycomb.pdf -> subject=English, title=Honeycomb
      Hindi_Vyakaran_Rachna.pdf -> subject=Hindi, title=Vyakaran Rachna
    """
    stem = pdf_path.stem.strip()
    if "_" not in stem:
        return {"subject": None, "title": _clean_title(stem), "book_title": _clean_title(stem)}

    subject, title = stem.split("_", 1)
    subject = _clean_title(subject)
    title = _clean_title(title) or _clean_title(stem)
    return {"subject": subject, "title": title, "book_title": title}


def derive_metadata_from_path(pdf_path: Path, input_root: Path | None = None) -> dict[str, Any]:
    """Derive school/class/book metadata from the requested folder convention.

    Expected folder shape:
      input/<School Name>/<Class-Grade>/<Subject>_<Book Title>.pdf

    Example:
      input/Mother Miracle School/Class-7/Maths_RSAgarwal.pdf
        school_name = Mother Miracle School
        class_name  = Class-7
        grade       = Class-7
        subject     = Maths
        title       = RSAgarwal
        book_title  = RSAgarwal
    """
    parts = _relative_parts(pdf_path, input_root)
    file_meta = parse_book_filename(pdf_path)

    school_name: str | None = None
    class_name: str | None = None

    # When input_root is supplied, the relative path should be:
    # <school>/<class>/<file>.pdf. Without it, use the last two folders.
    if input_root is not None and len(parts) >= 3:
        school_name = parts[-3]
        class_name = parts[-2]
    else:
        parent = pdf_path.parent
        if parent.name:
            class_name = parent.name
        if parent.parent.name and parent.parent != parent:
            school_name = parent.parent.name

    return {
        "school_name": _clean_title(school_name),
        "class_name": _clean_title(class_name),
        "grade": _clean_title(class_name),
        "subject": file_meta.get("subject"),
        "title": file_meta.get("title"),
        "book_title": file_meta.get("book_title"),
        "path_metadata_source": "folder_and_filename",
    }


def merge_metadata(*sources: dict[str, Any]) -> dict[str, Any]:
    """Merge metadata dictionaries, keeping later non-empty values."""
    merged: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            if value is not None and value != "":
                merged[key] = value
            elif key not in merged:
                merged[key] = value
    return merged
