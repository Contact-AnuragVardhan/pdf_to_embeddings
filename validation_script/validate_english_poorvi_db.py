#!/usr/bin/env python
"""
Validate English Poorvi production JSON + optional PostgreSQL DB ingestion.

Usage:
  python validation_script/validate_english_poorvi_db.py --json "json_input/English_Poorvi_production_ready.json"
  python validation_script/validate_english_poorvi_db.py --json "json_input/English_Poorvi_production_ready.json" --db

This validator is designed for the Poorvi JSON shape:
  metadata.document_key = mother-miracle-class-6-english-poorvi
  extraction.structure_type = unit_section
  extraction.section_index[] = lesson/section rows
  extraction.section_index[].subsections[] = Day-level subsection rows

It validates:
  - document metadata
  - section/lesson count
  - subsection count
  - page extraction count
  - blank page handling
  - subsection ranges inside parent lesson ranges
  - no subsection gaps/overlaps inside each lesson
  - DB table counts after ingestion
  - DB subsection parent linkage
  - chunks and embeddings count match
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_DOCUMENT_KEY = "mother-miracle-class-6-english-poorvi"
EXPECTED_STRUCTURE_TYPE = "unit_section"
EXPECTED_CHAPTERS_OR_SECTIONS = 16
EXPECTED_SUBSECTIONS = 77
EXPECTED_TOTAL_PDF_PAGES = 180
EXPECTED_CONTENT_START_PAGE = 17
EXPECTED_CONTENT_END_PAGE = 180


def first_non_empty_str(obj: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def text_from_section(section: dict[str, Any]) -> str:
    return first_non_empty_str(
        section,
        (
            "production_lesson_text",
            "lesson_text",
            "production_section_text",
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
    # Page-level production_text may be absent/blank in some files.
    # For validating whether extraction exists, use raw/selectable/clean text first.
    return first_non_empty_str(
        page,
        (
            "text",
            "text_plain",
            "clean_text",
            "selectable_text",
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


def get_range(obj: dict[str, Any]) -> tuple[int | None, int | None]:
    start = (
        obj.get("start_page")
        or obj.get("start_pdf_page")
        or obj.get("pdf_start_page")
    )
    end = (
        obj.get("end_page")
        or obj.get("end_pdf_page")
        or obj.get("pdf_end_page")
    )

    if start is None and isinstance(obj.get("pdf_pages"), dict):
        start = obj["pdf_pages"].get("start")
    if end is None and isinstance(obj.get("pdf_pages"), dict):
        end = obj["pdf_pages"].get("end")

    return start if isinstance(start, int) else None, end if isinstance(end, int) else None


def section_key(section: dict[str, Any]) -> str:
    return str(section.get("section_number") or section.get("chapter_number") or "")


def validate_json(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata") or {}
    extraction = data.get("extraction") or {}
    sections = extraction.get("section_index") or []

    if not isinstance(sections, list):
        errors.append("extraction.section_index is missing or is not a list")
        sections = []

    document_key = metadata.get("document_key")
    structure_type = extraction.get("structure_type")
    total_pdf_pages = extraction.get("total_pdf_pages")
    content_start = extraction.get("content_start_page")
    content_end = extraction.get("content_end_page")

    if document_key != EXPECTED_DOCUMENT_KEY:
        warnings.append(f"document_key is {document_key!r}; expected {EXPECTED_DOCUMENT_KEY!r}")

    if structure_type != EXPECTED_STRUCTURE_TYPE:
        warnings.append(f"structure_type is {structure_type!r}; expected {EXPECTED_STRUCTURE_TYPE!r}")

    if total_pdf_pages != EXPECTED_TOTAL_PDF_PAGES:
        warnings.append(f"total_pdf_pages is {total_pdf_pages}; expected {EXPECTED_TOTAL_PDF_PAGES}")

    if content_start != EXPECTED_CONTENT_START_PAGE or content_end != EXPECTED_CONTENT_END_PAGE:
        warnings.append(
            f"content range is {content_start}-{content_end}; "
            f"expected {EXPECTED_CONTENT_START_PAGE}-{EXPECTED_CONTENT_END_PAGE}"
        )

    page_extractions = extraction.get("page_extractions") or data.get("page_extractions") or []
    if not isinstance(page_extractions, list):
        warnings.append("page_extractions is missing or is not a list")
        page_extractions = []

    all_subsections: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for section in sections:
        if isinstance(section, dict):
            for sub in section.get("subsections") or []:
                if isinstance(sub, dict):
                    all_subsections.append((section, sub))

    if len(sections) != EXPECTED_CHAPTERS_OR_SECTIONS:
        errors.append(f"section_index count is {len(sections)}; expected {EXPECTED_CHAPTERS_OR_SECTIONS}")

    if len(all_subsections) != EXPECTED_SUBSECTIONS:
        errors.append(f"subsection count is {len(all_subsections)}; expected {EXPECTED_SUBSECTIONS}")

    if len(page_extractions) != EXPECTED_TOTAL_PDF_PAGES:
        warnings.append(f"page_extractions count is {len(page_extractions)}; expected {EXPECTED_TOTAL_PDF_PAGES}")

    section_counter = collections.Counter()
    subsection_counter = collections.Counter()

    section_bad_ranges = []
    section_no_text = []
    section_bad_page_arrays = []

    for section in sections:
        if not isinstance(section, dict):
            continue

        skey = section_key(section)
        section_counter[skey] += 1

        start, end = get_range(section)
        if not isinstance(start, int) or not isinstance(end, int) or start > end:
            section_bad_ranges.append((skey, start, end))
            continue

        page_count = section.get("page_count")
        if isinstance(page_count, int) and page_count != (end - start + 1):
            section_bad_page_arrays.append((skey, "page_count_mismatch", page_count, end - start + 1))

        indexed_pages = section.get("indexed_page_numbers") or section.get("page_numbers") or []
        if indexed_pages:
            if min(indexed_pages) < start or max(indexed_pages) > end:
                section_bad_page_arrays.append((skey, "indexed_pages_outside_range", start, end))
            if len(indexed_pages) != (end - start + 1):
                section_bad_page_arrays.append((skey, "indexed_pages_count_mismatch", len(indexed_pages), end - start + 1))

        if not text_from_section(section).strip() and not section.get("text_length_chars"):
            section_no_text.append(skey)

    duplicate_sections = [key for key, count in section_counter.items() if key and count > 1]

    subsection_bad_ranges = []
    subsection_outside_parent = []
    subsection_no_text = []
    subsection_no_pages = []
    subsection_bad_anchor = []
    subsection_not_ready_for_embedding = []

    for section, sub in all_subsections:
        skey = section_key(section)
        subkey = str(sub.get("subsection_number") or "")
        subsection_counter[(skey, subkey)] += 1

        start, end = get_range(sub)
        if not isinstance(start, int) or not isinstance(end, int) or start > end:
            subsection_bad_ranges.append((skey, subkey, start, end))
            continue

        parent_start, parent_end = get_range(section)
        if isinstance(parent_start, int) and isinstance(parent_end, int):
            if start < parent_start or end > parent_end:
                subsection_outside_parent.append((skey, subkey, start, end, parent_start, parent_end))

        anchor_pdf_page = sub.get("anchor_pdf_page")
        if isinstance(anchor_pdf_page, int) and not (start <= anchor_pdf_page <= end):
            subsection_bad_anchor.append((skey, subkey, anchor_pdf_page, start, end))

        page_numbers = sub.get("page_numbers") or sub.get("production_indexed_page_numbers") or []
        if not page_numbers:
            subsection_no_pages.append((skey, subkey, start, end))
        else:
            if min(page_numbers) < start or max(page_numbers) > end:
                subsection_no_pages.append((skey, subkey, "page_numbers_outside_range", start, end))

        if not text_from_subsection(sub).strip():
            subsection_no_text.append((skey, subkey, start, end))

        if sub.get("include_in_embeddings") is False:
            subsection_not_ready_for_embedding.append((skey, subkey, sub.get("embedding_readiness")))

    duplicate_subsections = [
        key for key, count in subsection_counter.items()
        if key[0] and key[1] and count > 1
    ]

    subsection_gaps_or_overlaps = []
    for section in sections:
        if not isinstance(section, dict):
            continue

        ranges = []
        for sub in section.get("subsections") or []:
            if not isinstance(sub, dict):
                continue
            start, end = get_range(sub)
            if isinstance(start, int) and isinstance(end, int):
                ranges.append((start, end, str(sub.get("subsection_number") or "")))

        if not ranges:
            continue

        ranges.sort()
        skey = section_key(section)
        section_start, section_end = get_range(section)

        if isinstance(section_start, int) and ranges[0][0] > section_start:
            subsection_gaps_or_overlaps.append((skey, "gap_at_start", section_start, ranges[0]))
        if isinstance(section_end, int) and ranges[-1][1] < section_end:
            subsection_gaps_or_overlaps.append((skey, "gap_at_end", ranges[-1], section_end))

        for left, right in zip(ranges, ranges[1:]):
            left_start, left_end, left_no = left
            right_start, right_end, right_no = right
            if right_start > left_end + 1:
                subsection_gaps_or_overlaps.append((skey, "gap", left_end + 1, right_start - 1, left_no, right_no))
            if right_start <= left_end:
                subsection_gaps_or_overlaps.append((skey, "overlap", right_start, left_end, left_no, right_no))

    page_numbers = []
    empty_raw_pages = []
    empty_raw_content_pages = []
    production_text_empty_but_raw_exists = []
    main_text_empty_pages = []

    for page in page_extractions:
        if not isinstance(page, dict):
            continue

        pn = page_number_from_page(page)
        if pn is None:
            continue

        page_numbers.append(pn)

        raw_text = raw_text_from_page(page)
        prod_text = production_text_from_page(page)
        main_text = first_non_empty_str(page, ("text", "raw_text", "cleaned_text", "page_text"))
        if not main_text.strip():
            main_text_empty_pages.append(pn)

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

    # These are actual failures for production ingestion.
    error_groups = {
        "duplicate_sections": duplicate_sections,
        "section_bad_ranges": section_bad_ranges,
        "section_bad_page_arrays": section_bad_page_arrays,
        "section_no_text": section_no_text,
        "duplicate_subsections": duplicate_subsections,
        "subsection_bad_ranges": subsection_bad_ranges,
        "subsection_outside_parent": subsection_outside_parent,
        "subsection_bad_anchor": subsection_bad_anchor,
        "subsection_no_pages": subsection_no_pages,
        "subsection_no_text": subsection_no_text,
        "subsection_gaps_or_overlaps": subsection_gaps_or_overlaps,
        "subsection_not_ready_for_embedding": subsection_not_ready_for_embedding,
        "empty_raw_content_pages": empty_raw_content_pages,
    }

    for name, values in error_groups.items():
        if values:
            errors.append(f"{name}: {len(values)} problem(s), examples={values[:10]}")

    unit_counts = collections.Counter()
    for section in sections:
        if isinstance(section, dict):
            unit_counts[str(section.get("unit_number") or "")] += 1

    summary = {
        "document_key": document_key,
        "book_title": extraction.get("book_title"),
        "subject": extraction.get("subject"),
        "structure_type": structure_type,
        "total_pdf_pages": total_pdf_pages,
        "content_start_page": content_start,
        "content_end_page": content_end,
        "chapters_or_sections": len(sections),
        "subsections": len(all_subsections),
        "page_extractions": len(page_extractions),
        "unique_page_numbers": len(set(page_numbers)),
        "empty_raw_pages": empty_raw_pages,
        "empty_raw_content_pages": empty_raw_content_pages,
        "main_text_empty_pages": main_text_empty_pages,
        "production_text_empty_but_raw_exists_count": len(production_text_empty_but_raw_exists),
        "production_text_empty_but_raw_exists_examples": production_text_empty_but_raw_exists[:20],
        "unit_counts": dict(sorted(unit_counts.items())),
        "sections_with_subsection_counts": [
            {
                "section_number": section_key(section),
                "section_title": section.get("section_title") or section.get("chapter_title"),
                "unit_number": section.get("unit_number"),
                "unit_title": section.get("unit_title"),
                "chapter_number": section.get("chapter_number"),
                "chapter_title": section.get("chapter_title"),
                "start_page": get_range(section)[0],
                "end_page": get_range(section)[1],
                "subsections": len(section.get("subsections") or []),
            }
            for section in sections
            if isinstance(section, dict)
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
                "chapters_or_sections": "select count(*) from embeddings_book_chapters where document_id = %s",
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

            expected_sections = summary["chapters_or_sections"]
            expected_subsections = summary["subsections"]
            expected_pages = summary["page_extractions"]
            empty_page_count = max(
                len(summary.get("empty_raw_pages") or []),
                len(summary.get("main_text_empty_pages") or []),
            )

            if db_counts["chapters_or_sections"] != expected_sections:
                errors.append(
                    f"DB chapters/sections={db_counts['chapters_or_sections']} but JSON section_index={expected_sections}"
                )

            if db_counts["subsections"] != expected_subsections:
                errors.append(f"DB subsections={db_counts['subsections']} but JSON subsections={expected_subsections}")

            # Poorvi JSON has 180 page_extractions; blank/front-matter pages may be skipped by ingestion.
            allowed_raw_pages = {expected_pages, expected_pages - empty_page_count}
            if db_counts["raw_pages"] not in allowed_raw_pages:
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
                errors.append(f"DB has {outside_parent} subsection rows outside parent lesson/section range")

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
                errors.append(f"DB has {orphan_subsections} orphan subsection rows with no matching parent lesson/section")

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

            cur.execute(
                f"""
                select section_number, section_title, chapter_number, {c_start}, {c_end}
                from embeddings_book_chapters
                where document_id = %s
                order by {c_start} nulls last, section_number
                limit 5
                """,
                (doc_id,),
            )
            sample_sections = cur.fetchall()

            cur.execute(
                f"""
                select section_number, subsection_number, subsection_title, {s_start}, {s_end}, text_length_chars
                from embeddings_book_subsections
                where document_id = %s
                order by {s_start} nulls last, section_number, subsection_number
                limit 5
                """,
                (doc_id,),
            )
            sample_subsections = cur.fetchall()

            print("\nDB COUNTS")
            for key, value in db_counts.items():
                print(f"  {key}: {value}")

            print("\nDB SAMPLE SECTIONS")
            for row in sample_sections:
                print(f"  {row}")

            print("\nDB SAMPLE SUBSECTIONS")
            for row in sample_subsections:
                print(f"  {row}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to English Poorvi production JSON")
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
        "chapters_or_sections",
        "subsections",
        "page_extractions",
        "unique_page_numbers",
        "empty_raw_pages",
        "empty_raw_content_pages",
        "main_text_empty_pages",
        "production_text_empty_but_raw_exists_count",
        "production_text_empty_but_raw_exists_examples",
        "unit_counts",
    ):
        print(f"  {key}: {summary.get(key)}")

    print("\nSECTION/SUBSECTION SUMMARY")
    for section in summary["sections_with_subsection_counts"]:
        print(
            f"  Section {section['section_number']}: "
            f"{section['section_title']} | "
            f"{section['chapter_number']} / {section['chapter_title']} | "
            f"PDF {section['start_page']}-{section['end_page']} | "
            f"subsections={section['subsections']}"
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

    print("\nPASSED: English Poorvi JSON validation completed successfully" + (" and DB matches expected counts." if args.db else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
