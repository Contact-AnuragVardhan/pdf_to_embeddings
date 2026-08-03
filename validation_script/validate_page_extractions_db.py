from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row


def _printed(value: Any) -> str | None:
    return None if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify extraction.page_extractions[] against embeddings_page_extractions."
    )
    parser.add_argument("--json", required=True, help="Production extraction JSON used for ingest-json.")
    parser.add_argument("--document-key", help="Defaults to metadata.document_key in the JSON.")
    args = parser.parse_args()

    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required in the environment or .env file.")

    json_path = Path(args.json).resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    extraction = payload.get("extraction") or payload
    expected = extraction.get("page_extractions") or []
    document_key = args.document_key or (payload.get("metadata") or {}).get("document_key") or extraction.get("document_key")
    if not document_key:
        raise SystemExit("document_key was not found. Pass --document-key explicitly.")
    if not expected:
        raise SystemExit("The JSON has no extraction.page_extractions[] records.")

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT pe.*
            FROM embeddings_page_extractions pe
            JOIN embeddings_documents d ON d.id = pe.document_id
            WHERE d.document_key = %s
            ORDER BY pe.page_number
            """,
            (document_key,),
        ).fetchall()

    errors: list[str] = []
    expected_by_page = {int(item.get("page_number") or item.get("pdf_page_number")): item for item in expected}
    actual_by_page = {int(item["page_number"]): item for item in rows}

    if len(rows) != len(expected):
        errors.append(f"row count mismatch: JSON={len(expected)} DB={len(rows)}")
    missing = sorted(set(expected_by_page) - set(actual_by_page))
    extra = sorted(set(actual_by_page) - set(expected_by_page))
    if missing:
        errors.append(f"missing DB pages: {missing}")
    if extra:
        errors.append(f"unexpected DB pages: {extra}")

    compare_fields = [
        "pdf_page_number",
        "printed_page_label",
        "chapter_number",
        "chapter_title",
        "section_number",
        "section_title",
        "linked_section_number",
        "linked_section_title",
        "content_type",
        "assignment_status",
        "include_in_chapter_text",
        "include_in_lesson_text",
        "include_in_embeddings",
        "embedding_readiness",
        "source_type",
        "extraction_method",
        "text",
        "text_plain",
        "production_page_text",
        "production_safe_text",
        "selectable_text",
        "raw_extracted_text",
        "ocr_text",
        "text_length_chars",
    ]
    for page_number in sorted(set(expected_by_page) & set(actual_by_page)):
        source = expected_by_page[page_number]
        stored = actual_by_page[page_number]
        if _printed(source.get("printed_page_number")) != stored.get("printed_page_number"):
            errors.append(f"page {page_number}: printed_page_number mismatch")
        for field in compare_fields:
            expected_value = source.get(field)
            actual_value = stored.get(field)
            if field == "pdf_page_number" and expected_value is None:
                expected_value = page_number
            if expected_value != actual_value:
                errors.append(f"page {page_number}: {field} mismatch")
        if stored.get("source_payload", {}).get("page_number") != page_number:
            errors.append(f"page {page_number}: source_payload is missing the original page number")

    expected_excluded = sum(1 for item in expected if item.get("include_in_embeddings") is False)
    actual_excluded = sum(1 for item in rows if item.get("include_in_embeddings") is False)
    print(f"document_key: {document_key}")
    print(f"JSON page_extractions: {len(expected)}")
    print(f"DB page_extractions:   {len(rows)}")
    print(f"Audit-only pages:      JSON={expected_excluded}, DB={actual_excluded}")

    if errors:
        print("FAILED")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... and {len(errors) - 100} more")
        return 1

    print("PASSED: page_extractions were stored losslessly for the checked fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
