#!/usr/bin/env python
"""
Fixed validator for Maths_RSAgarwal JSON + DB ingestion.

Usage:
  python validation_script/validate_math_rsaggarwal_db_fixed.py --json "json_input/Maths_RSAgarwal_math_aware_extraction_v4_production_safe.json"
  python validation_script/validate_math_rsaggarwal_db_fixed.py --json "json_input/Maths_RSAgarwal_math_aware_extraction_v4_production_safe.json" --db

Fixes:
  1. Does not treat pages as empty just because production_text is blank.
     It falls back to text/text_plain/ocr_text.
  2. Uses embeddings_book_subsections.pdf_start_page/pdf_end_page,
     with fallback support if your schema has start_page/end_page.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any


def first_non_empty_str(obj: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def text_from_chapter(ch: dict[str, Any]) -> str:
    return first_non_empty_str(
        ch,
        (
            "production_lesson_text",
            "lesson_text",
            "section_text",
            "chapter_text",
            "text_plain",
            "text",
        ),
    )


def text_from_subsection(sub: dict[str, Any]) -> str:
    return first_non_empty_str(
        sub,
        (
            "production_subsection_text",
            "subsection_text",
            "subsection_text_plain",
            "text_plain",
            "text",
        ),
    )


def raw_text_from_page(page: dict[str, Any]) -> str:
    # Important: production_text can intentionally be empty for pages excluded
    # from production embeddings. That does NOT mean the page has no extracted text.
    return first_non_empty_str(
        page,
        (
            "text",
            "text_plain",
            "ocr_text",
            "production_text",
            "page_text",
        ),
    )


def production_text_from_page(page: dict[str, Any]) -> str:
    return first_non_empty_str(page, ("production_text",))


def page_number_from_page(page: dict[str, Any]) -> int | None:
    for key in ("page_number", "pdf_page", "pdf_page_number"):
        value = page.get(key)
        if isinstance(value, int):
            return value
    return None


def get_sub_range(sub: dict[str, Any]) -> tuple[int | None, int | None]:
    start = (
        sub.get("start_page")
        or sub.get("start_pdf_page")
        or sub.get("pdf_start_page")
    )
    end = (
        sub.get("end_page")
        or sub.get("end_pdf_page")
        or sub.get("pdf_end_page")
    )

    if start is None and isinstance(sub.get("pdf_pages"), dict):
        start = sub["pdf_pages"].get("start")
    if end is None and isinstance(sub.get("pdf_pages"), dict):
        end = sub["pdf_pages"].get("end")

    return start if isinstance(start, int) else None, end if isinstance(end, int) else None


def section_key(ch: dict[str, Any]) -> str:
    return str(ch.get("chapter_number") or ch.get("section_number") or "")


def validate_json(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata") or {}
    extraction = data.get("extraction") or {}
    chapters = extraction.get("section_index") or []

    if not isinstance(chapters, list):
        errors.append("extraction.section_index is missing or is not a list")
        chapters = []

    document_key = metadata.get("document_key")
    if not document_key:
        errors.append("metadata.document_key is missing")

    page_extractions = extraction.get("page_extractions") or data.get("page_extractions") or []
    if not isinstance(page_extractions, list):
        warnings.append("page_extractions is missing or is not a list")
        page_extractions = []

    total_pdf_pages = extraction.get("total_pdf_pages")
    content_start = extraction.get("content_start_page")
    content_end = extraction.get("content_end_page")

    all_subsections: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ch in chapters:
        if isinstance(ch, dict):
            for sub in ch.get("subsections") or []:
                if isinstance(sub, dict):
                    all_subsections.append((ch, sub))

    chapter_key_counter = collections.Counter()
    subsection_key_counter = collections.Counter()

    chapter_bad_ranges = []
    chapter_no_text = []
    chapter_bad_indexed_pages = []

    for ch in chapters:
        if not isinstance(ch, dict):
            continue

        ch_key = section_key(ch)
        chapter_key_counter[ch_key] += 1

        start = ch.get("start_page")
        end = ch.get("end_page")

        if not isinstance(start, int) or not isinstance(end, int) or start > end:
            chapter_bad_ranges.append((ch_key, start, end))
            continue

        indexed_pages = ch.get("indexed_page_numbers") or ch.get("page_numbers") or []
        if indexed_pages:
            if min(indexed_pages) < start or max(indexed_pages) > end:
                chapter_bad_indexed_pages.append((ch_key, start, end, indexed_pages[:3], indexed_pages[-3:]))

        if not text_from_chapter(ch).strip() and not ch.get("text_length_chars"):
            chapter_no_text.append(ch_key)

    duplicate_chapters = [key for key, count in chapter_key_counter.items() if key and count > 1]

    subsection_bad_ranges = []
    subsection_outside_parent = []
    subsection_no_text = []
    subsection_no_pages = []

    for ch, sub in all_subsections:
        ch_key = section_key(ch)
        sub_key = str(sub.get("subsection_number") or "")
        subsection_key_counter[(ch_key, sub_key)] += 1

        start, end = get_sub_range(sub)
        if not isinstance(start, int) or not isinstance(end, int) or start > end:
            subsection_bad_ranges.append((ch_key, sub_key, start, end))
            continue

        ch_start = ch.get("start_page")
        ch_end = ch.get("end_page")
        if isinstance(ch_start, int) and isinstance(ch_end, int) and (start < ch_start or end > ch_end):
            subsection_outside_parent.append((ch_key, sub_key, start, end, ch_start, ch_end))

        page_numbers = sub.get("page_numbers") or sub.get("production_indexed_page_numbers") or []
        if not page_numbers:
            subsection_no_pages.append((ch_key, sub_key, start, end))

        if not text_from_subsection(sub).strip():
            subsection_no_text.append((ch_key, sub_key, start, end))

    duplicate_subsections = [
        key for key, count in subsection_key_counter.items()
        if key[0] and key[1] and count > 1
    ]

    subsection_gaps_or_overlaps = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue

        ranges = []
        for sub in ch.get("subsections") or []:
            if not isinstance(sub, dict):
                continue
            start, end = get_sub_range(sub)
            if isinstance(start, int) and isinstance(end, int):
                ranges.append((start, end, str(sub.get("subsection_number") or "")))

        if not ranges:
            continue

        ranges.sort()
        ch_key = section_key(ch)
        ch_start = ch.get("start_page")
        ch_end = ch.get("end_page")

        if isinstance(ch_start, int) and ranges[0][0] > ch_start:
            subsection_gaps_or_overlaps.append((ch_key, "gap_at_start", ch_start, ranges[0]))
        if isinstance(ch_end, int) and ranges[-1][1] < ch_end:
            subsection_gaps_or_overlaps.append((ch_key, "gap_at_end", ranges[-1], ch_end))

        for left, right in zip(ranges, ranges[1:]):
            left_start, left_end, left_no = left
            right_start, right_end, right_no = right
            if right_start > left_end + 1:
                subsection_gaps_or_overlaps.append((ch_key, "gap", left_end + 1, right_start - 1, left_no, right_no))
            if right_start <= left_end:
                subsection_gaps_or_overlaps.append((ch_key, "overlap", right_start, left_end, left_no, right_no))

    page_numbers = []
    empty_raw_pages = []
    empty_raw_content_pages = []
    production_text_empty_but_raw_exists = []

    for page in page_extractions:
        if not isinstance(page, dict):
            continue

        pn = page_number_from_page(page)
        if pn is None:
            continue

        page_numbers.append(pn)

        raw_text = raw_text_from_page(page)
        prod_text = production_text_from_page(page)

        if not raw_text.strip():
            empty_raw_pages.append(pn)
            if isinstance(content_start, int) and isinstance(content_end, int) and content_start <= pn <= content_end:
                empty_raw_content_pages.append(pn)

        if raw_text.strip() and not prod_text.strip():
            production_text_empty_but_raw_exists.append(pn)

    if total_pdf_pages and page_numbers:
        if len(set(page_numbers)) != total_pdf_pages:
            warnings.append(
                f"page_extractions unique page count {len(set(page_numbers))} does not match total_pdf_pages {total_pdf_pages}"
            )
        if min(page_numbers) != 1 or max(page_numbers) != total_pdf_pages:
            warnings.append(f"page_extractions range is {min(page_numbers)}-{max(page_numbers)}, expected 1-{total_pdf_pages}")

    error_groups = {
        "duplicate_chapters": duplicate_chapters,
        "chapter_bad_ranges": chapter_bad_ranges,
        "chapter_bad_indexed_pages": chapter_bad_indexed_pages,
        "chapter_no_text": chapter_no_text,
        "duplicate_subsections": duplicate_subsections,
        "subsection_bad_ranges": subsection_bad_ranges,
        "subsection_outside_parent": subsection_outside_parent,
        "subsection_no_pages": subsection_no_pages,
        "subsection_no_text": subsection_no_text,
        "subsection_gaps_or_overlaps": subsection_gaps_or_overlaps,
        "empty_raw_content_pages": empty_raw_content_pages,
    }

    for name, values in error_groups.items():
        if values:
            errors.append(f"{name}: {len(values)} problem(s), examples={values[:10]}")

    summary = {
        "document_key": document_key,
        "book_title": extraction.get("book_title"),
        "subject": extraction.get("subject"),
        "structure_type": extraction.get("structure_type"),
        "total_pdf_pages": total_pdf_pages,
        "content_start_page": content_start,
        "content_end_page": content_end,
        "answers_start_page": extraction.get("answers_start_page"),
        "answers_end_page": extraction.get("answers_end_page"),
        "chapters": len(chapters),
        "subsections": len(all_subsections),
        "page_extractions": len(page_extractions),
        "unique_page_numbers": len(set(page_numbers)),
        "empty_raw_pages": empty_raw_pages,
        "empty_raw_content_pages": empty_raw_content_pages,
        "production_text_empty_but_raw_exists_count": len(production_text_empty_but_raw_exists),
        "production_text_empty_but_raw_exists_examples": production_text_empty_but_raw_exists[:20],
        "chapters_with_subsection_counts": [
            {
                "chapter_number": section_key(ch),
                "chapter_title": ch.get("chapter_title") or ch.get("section_title"),
                "start_page": ch.get("start_page"),
                "end_page": ch.get("end_page"),
                "subsections": len(ch.get("subsections") or []),
            }
            for ch in chapters
            if isinstance(ch, dict)
        ],
    }

    return summary, errors, warnings


def table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        """
        select column_name
        from information_schema.columns
        where table_schema = 'public'
          and table_name = %s
        """,
        (table_name,),
    )
    return {row[0] for row in cur.fetchall()}


def validate_db(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        from dotenv import load_dotenv
        import psycopg
    except Exception as exc:
        return [f"DB validation requires python-dotenv and psycopg. Details: {exc}"], warnings

    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return ["DATABASE_URL is missing from environment/.env"], warnings

    document_key = summary["document_key"]
    if not document_key:
        return ["Cannot validate DB because JSON has no document_key"], warnings

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select id, document_key, title from embeddings_documents where document_key = %s",
                (document_key,),
            )
            docs = cur.fetchall()

            if len(docs) != 1:
                errors.append(f"Expected exactly 1 document row for {document_key}, found {len(docs)}")
                return errors, warnings

            doc_id = docs[0][0]

            queries = {
                "chapters": "select count(*) from embeddings_book_chapters where document_id = %s",
                "subsections": "select count(*) from embeddings_book_subsections where document_id = %s",
                "raw_pages": "select count(*) from embeddings_raw_text_pages where document_id = %s",
                "chunks": "select count(*) from embeddings_chunks where document_id = %s",
                "vectors": """
                    select count(*)
                    from embeddings_vectors v
                    join embeddings_chunks c on c.id = v.chunk_id
                    where c.document_id = %s
                """,
            }

            db_counts = {}
            for name, sql in queries.items():
                cur.execute(sql, (doc_id,))
                db_counts[name] = cur.fetchone()[0]

            expected_chapters = summary["chapters"]
            expected_subsections = summary["subsections"]
            expected_pages = summary["page_extractions"]

            if db_counts["chapters"] != expected_chapters:
                errors.append(f"DB chapters={db_counts['chapters']} but JSON chapters={expected_chapters}")

            if db_counts["subsections"] != expected_subsections:
                errors.append(f"DB subsections={db_counts['subsections']} but JSON subsections={expected_subsections}")

            # Your ingestion console showed 324 pages. JSON has 325 because PDF page 7 is blank.
            # So raw_pages can be either 325 or 324 depending on whether blank pages are stored.
            if db_counts["raw_pages"] not in (expected_pages, expected_pages - 1):
                warnings.append(
                    f"DB raw_pages={db_counts['raw_pages']} but JSON page_extractions={expected_pages}. "
                    "This is OK only if your loader intentionally excludes blank/non-content pages."
                )

            if db_counts["chunks"] <= 0:
                errors.append("DB chunks=0; embedding pipeline did not create chunks")

            if db_counts["vectors"] <= 0:
                errors.append("DB vectors=0; embeddings were not created")

            if db_counts["chunks"] != db_counts["vectors"]:
                errors.append(f"DB chunks={db_counts['chunks']} but vectors={db_counts['vectors']}; embeddings incomplete")

            sub_cols = table_columns(cur, "embeddings_book_subsections")
            chapter_cols = table_columns(cur, "embeddings_book_chapters")

            # Support both schema names, but your current table uses pdf_start_page/pdf_end_page.
            s_start = "pdf_start_page" if "pdf_start_page" in sub_cols else "start_page"
            s_end = "pdf_end_page" if "pdf_end_page" in sub_cols else "end_page"
            c_start = "start_page" if "start_page" in chapter_cols else "pdf_start_page"
            c_end = "end_page" if "end_page" in chapter_cols else "pdf_end_page"

            cur.execute(
                f"""
                select count(*)
                from embeddings_book_subsections s
                join embeddings_book_chapters c
                  on c.document_id = s.document_id
                 and coalesce(c.chapter_number, '') = coalesce(s.chapter_number, '')
                 and coalesce(c.section_number, '') = coalesce(s.section_number, '')
                where s.document_id = %s
                  and (s.{s_start} < c.{c_start} or s.{s_end} > c.{c_end})
                """,
                (doc_id,),
            )
            outside_parent = cur.fetchone()[0]
            if outside_parent:
                errors.append(f"DB has {outside_parent} subsection rows outside parent chapter/section range")

            cur.execute(
                """
                select count(*)
                from embeddings_book_subsections s
                left join embeddings_book_chapters c
                  on c.document_id = s.document_id
                 and coalesce(c.chapter_number, '') = coalesce(s.chapter_number, '')
                 and coalesce(c.section_number, '') = coalesce(s.section_number, '')
                where s.document_id = %s
                  and c.id is null
                """,
                (doc_id,),
            )
            orphan_subsections = cur.fetchone()[0]
            if orphan_subsections:
                errors.append(f"DB has {orphan_subsections} orphan subsection rows with no matching parent chapter/section")

            cur.execute(
                """
                select count(*)
                from embeddings_book_subsections
                where document_id = %s
                  and coalesce(text_length_chars, 0) = 0
                """,
                (doc_id,),
            )
            empty_sub_text = cur.fetchone()[0]
            if empty_sub_text:
                errors.append(f"DB has {empty_sub_text} subsections with zero text_length_chars")

            print("\nDB COUNTS")
            for key, value in db_counts.items():
                print(f"  {key}: {value}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to production extraction JSON")
    parser.add_argument("--db", action="store_true", help="Also compare JSON counts against PostgreSQL ingestion tables")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
        return 2

    summary, errors, warnings = validate_json(json_path)

    print("JSON SUMMARY")
    for key in (
        "document_key",
        "book_title",
        "subject",
        "structure_type",
        "total_pdf_pages",
        "content_start_page",
        "content_end_page",
        "answers_start_page",
        "answers_end_page",
        "chapters",
        "subsections",
        "page_extractions",
        "unique_page_numbers",
        "empty_raw_pages",
        "empty_raw_content_pages",
        "production_text_empty_but_raw_exists_count",
        "production_text_empty_but_raw_exists_examples",
    ):
        print(f"  {key}: {summary.get(key)}")

    print("\nCHAPTER/SUBSECTION SUMMARY")
    for ch in summary["chapters_with_subsection_counts"]:
        print(
            f"  Chapter {ch['chapter_number']}: "
            f"{ch['chapter_title']} | PDF {ch['start_page']}-{ch['end_page']} | "
            f"subsections={ch['subsections']}"
        )

    if args.db:
        db_errors, db_warnings = validate_db(summary)
        errors.extend(db_errors)
        warnings.extend(db_warnings)

    if warnings:
        print("\nWARNINGS")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nFAILED")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nPASSED: JSON validation completed successfully" + (" and DB matches expected counts." if args.db else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
