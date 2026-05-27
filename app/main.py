from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

from config import Settings
from db.migrations import init_schema
from ingestion.ingest_service import IngestService
from ingestion.path_metadata import derive_metadata_from_path, merge_metadata
from ingestion.repository import RagRepository
from ingestion.validators import validate_ingest_args
from search.rag_search import RagSearchService
from utils.logging_config import configure_logging
import json

console = Console()
logger = logging.getLogger(__name__)


def add_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", help="Book title. For folder ingestion this is derived from <Subject>_<Book Title>.pdf.")
    parser.add_argument("--book-title", help="Book title/name. Defaults to title parsed from <Subject>_<Book Title>.pdf.")
    parser.add_argument("--subject", help="Subject, e.g. Hindi, English, Maths, Science, EVS. Defaults to filename prefix before underscore.")
    parser.add_argument("--school-name", help="School name. Defaults to input/<School Name>/<Class-Grade>/<PDF>.")
    parser.add_argument("--class-name", help="Class/grade folder name, e.g. Class-7. Defaults to input/<School>/<Class-Grade>/<PDF>.")
    parser.add_argument("--grade", help="Grade/class, e.g. Class-7 or 7. Defaults to --class-name/path class folder.")
    parser.add_argument("--board", help="Board, e.g. CBSE, NCERT, ICSE.")
    parser.add_argument("--medium", help="Medium, e.g. Hindi or English.")
    parser.add_argument("--language", help="Declared language, e.g. Hindi, English, Mixed.")
    parser.add_argument("--publisher")
    parser.add_argument("--edition")
    parser.add_argument("--publication-year")
    parser.add_argument("--isbn")
    parser.add_argument("--author")
    parser.add_argument("--copyright-status", default="unknown")
    parser.add_argument("--license-notes")
    parser.add_argument("--source-uri")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdf_embedding_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="Create/upgrade PostgreSQL schema.")
    init_db.set_defaults(handler=handle_init_db)

    ingest = sub.add_parser("ingest", help="Ingest one readable PDF.")
    ingest.add_argument("--pdf", required=True, help="Path to a readable/selectable PDF.")
    add_metadata_args(ingest)
    ingest.add_argument("--reindex", action="store_true", help="Delete existing document by file_hash and ingest again.")
    ingest.add_argument("--dry-run", action="store_true", help="Extract/chunk only. Do not write DB or call OpenAI.")
    ingest.add_argument("--validate-only", action="store_true", help="Validate CLI/file inputs only.")
    ingest.set_defaults(handler=handle_ingest)

    folder = sub.add_parser("ingest-folder", help="Ingest all PDFs in a folder.")
    folder.add_argument("--input-dir", required=True, help="Folder containing PDF files.")
    add_metadata_args(folder)
    folder.add_argument("--reindex", action="store_true")
    folder.add_argument("--dry-run", action="store_true")
    folder.add_argument("--validate-only", action="store_true")
    folder.set_defaults(handler=handle_ingest_folder)

    search = sub.add_parser("search", help="Hybrid RAG search over ingested chunks.")
    search.add_argument("--query", required=True)
    search.add_argument("--subject")
    search.add_argument("--school-name")
    search.add_argument("--grade")
    search.add_argument("--class-name")
    search.add_argument("--book-title")
    search.add_argument("--language")
    search.add_argument("--board")
    search.add_argument("--document-id")
    search.add_argument("--chapter-title")
    search.add_argument("--chunk-type")
    search.add_argument("--top-k", type=int, default=8)
    search.set_defaults(handler=handle_search)
    return parser


def metadata_from_args(
    args: argparse.Namespace,
    *,
    pdf_path: Path | None = None,
    input_root: Path | None = None,
    title_fallback: str | None = None,
) -> dict[str, Any]:
    path_metadata = derive_metadata_from_path(pdf_path, input_root) if pdf_path else {}
    cli_metadata = {
        "title": args.title or args.book_title,
        "book_title": args.book_title or args.title,
        "subject": args.subject,
        "school_name": args.school_name,
        "class_name": args.class_name,
        "grade": args.grade or args.class_name,
        "board": args.board,
        "medium": args.medium,
        "language": args.language,
        "publisher": args.publisher,
        "edition": args.edition,
        "publication_year": args.publication_year,
        "isbn": args.isbn,
        "author": args.author,
        "copyright_status": args.copyright_status,
        "license_notes": args.license_notes,
        "source_uri": args.source_uri,
    }
    metadata = merge_metadata(path_metadata, cli_metadata)
    # Keep the existing title field as the book title used throughout the older code.
    metadata["title"] = metadata.get("title") or metadata.get("book_title") or title_fallback
    metadata["book_title"] = metadata.get("book_title") or metadata.get("title")
    metadata["grade"] = metadata.get("grade") or metadata.get("class_name")
    metadata["class_name"] = metadata.get("class_name") or metadata.get("grade")
    return metadata


def handle_init_db(args: argparse.Namespace, settings: Settings) -> None:
    init_schema(settings.database_url, settings.project_root / "db" / "schema.sql")
    console.print("[green]Database schema initialized successfully.[/green]")


def handle_ingest(args: argparse.Namespace, settings: Settings) -> None:
    pdf_path = Path(args.pdf)
    metadata = metadata_from_args(args, pdf_path=pdf_path, title_fallback=pdf_path.stem)
    validate_ingest_args(pdf_path, metadata)
    if args.validate_only:
        console.print("[green]Validation passed.[/green]")
        return

    repository = RagRepository(settings.database_url)
    service = IngestService(settings, repository)
    result = service.ingest_pdf(
        pdf_path=pdf_path,
        metadata=metadata,
        reindex=args.reindex,
        dry_run=args.dry_run,
    )
    console.print_json(
        json=json.dumps(result, default=str, ensure_ascii=False, indent=2)
    )


def handle_ingest_folder(args: argparse.Namespace, settings: Settings) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {input_dir}.[/yellow]")
        return

    repository = RagRepository(settings.database_url)
    service = IngestService(settings, repository)
    results: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        metadata = metadata_from_args(args, pdf_path=pdf_path, input_root=input_dir, title_fallback=pdf_path.stem)
        validate_ingest_args(pdf_path, metadata)
        if args.validate_only:
            results.append({"file": str(pdf_path), "status": "validated"})
            continue
        results.append(service.ingest_pdf(pdf_path, metadata, reindex=args.reindex, dry_run=args.dry_run))
    console.print_json(
        json=json.dumps(results, default=str, ensure_ascii=False, indent=2)
    )


def handle_search(args: argparse.Namespace, settings: Settings) -> None:
    settings.validate_for_embedding()
    repository = RagRepository(settings.database_url)
    service = RagSearchService(settings, repository)
    results = service.search(
        query=args.query,
        top_k=args.top_k,
        filters={
            "subject": args.subject,
            "school_name": args.school_name,
            "grade": args.grade,
            "class_name": args.class_name,
            "book_title": args.book_title,
            "language": args.language,
            "board": args.board,
            "document_id": args.document_id,
            "chapter_title": args.chapter_title,
            "chunk_type": args.chunk_type,
        },
    )
    service.print_results(results)


def main() -> None:
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args, settings)
    except Exception as exc:
        logger.exception("Command failed")
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
