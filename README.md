# PDF Embedding Pipeline with School/Class/Book Metadata, LLM Chapter Detection, Auto Chunking, and RAG Validation

This pipeline ingests readable/selectable PDFs into PostgreSQL + pgvector for RAG search.

It supports folders like:

```text
input/Mother Miracle School/Class-7/Maths_RSAgarwal.pdf
```

The path is parsed into:

```text
school_name = Mother Miracle School
class_name  = Class-7
grade       = Class-7
subject     = Maths
book_title  = RSAgarwal
```

All CLI commands should be run from the **project root** using:

```powershell
python app/main.py ...
```

Do not use `python main.py` unless you intentionally keep a root-level compatibility wrapper.

## What was added

- Recursive folder ingestion.
- School/class/subject/book metadata from folder and file name.
- LLM metadata detection using `OPENAI_METADATA_MODEL`, default `gpt-5.4`.
- Model-name-agnostic metadata call; you can change to `gpt-4o-mini` or `gpt-5.4-mini` in `.env`.
- `embeddings_book_chapters` table.
- `embeddings_raw_text_pages` now includes chapter and printed page mapping.
- Auto chunking per book using LLM recommendation + safe heuristic fallback.
- Deterministic fallback if LLM metadata detection fails.
- RAG validation commands to verify table counts, metadata, chapter detection, raw page mapping, and semantic search results.

## Important OpenAI model settings

Embeddings stay fixed because the DB column is `vector(3072)`:

```env
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072
```

Metadata/chapter detection is separate and can be changed:

```env
OPENAI_METADATA_MODEL=gpt-5.4
# or
OPENAI_METADATA_MODEL=gpt-4o-mini
# or
OPENAI_METADATA_MODEL=gpt-5.4-mini
```

## Setup

From project root:

```powershell
cd D:\Project\JaltaSitaraApps\TeacherHelperProject\pdf_to_embeddings\pdf_embedding_pipeline
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_key

OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072

OPENAI_METADATA_MODEL=gpt-5.4
AUTO_METADATA_ENABLED=true
METADATA_SAMPLE_PAGES=20
METADATA_MAX_OUTPUT_TOKENS=6000

AUTO_CHUNKING_ENABLED=true
DEFAULT_CHUNK_MAX_TOKENS=750
DEFAULT_CHUNK_OVERLAP_TOKENS=120

DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
EMBEDDING_BATCH_SIZE=64
REINDEX_EXISTING=false
LOG_LEVEL=INFO
```

## Fresh reset command

This deletes only the embedding pipeline tables.

```powershell
python -c "from dotenv import load_dotenv; import os, psycopg; load_dotenv(); conn=psycopg.connect(os.environ['DATABASE_URL'], autocommit=True); cur=conn.cursor(); cur.execute('DROP TABLE IF EXISTS public.embeddings_vectors CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_raw_text_pages CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_book_subsections CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_book_chapters CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_chunks CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_pages CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_ingestion_runs CASCADE'); cur.execute('DROP TABLE IF EXISTS public.embeddings_documents CASCADE'); conn.close(); print('Embedding tables deleted')"
```

Then recreate:

```powershell
python app/main.py init-db
```


## JSON output and page-wise console logs

The ingestion flow now writes a JSON artifact in addition to saving rows in PostgreSQL.
The file is written after page extraction, book-structure detection, and chunking, so it is available for both normal ingestion and `--dry-run`.

Default output location:

```text
output/json_exports/<pdf_name>_combined_extraction.json
```

Run with page-wise text printed in the console:

```powershell
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English" --log-page-text
```

Use a custom JSON output folder:

```powershell
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English" --output-json-dir "output/extractions"
```

Disable JSON output for a run:

```powershell
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English" --no-json-output
```

Environment settings:

```env
EXPORT_JSON_ENABLED=true
JSON_OUTPUT_DIR=output/json_exports
LOG_EXTRACTED_PAGE_TEXT=false
LOG_PAGE_TEXT_MAX_CHARS=12000
```

JSON shape:

- Chapter-based books write `extraction.chapters`.
- Unit/section-based books write `extraction.sections`.
- Every JSON includes `extraction.page_extractions` so you can verify text page by page.
- The normal DB save and embedding flow is unchanged.


## Ingest pre-extracted JSON instead of reading the PDF again

Use this when another process has already extracted book text chapter-wise or section-wise. The JSON should **not** contain embeddings. This command bypasses OCR/PDF extraction, saves the document, book structures, page text, raw page text, chunks, and then creates embeddings from the extracted text.

JSON ingestion uses `document_key` as the stable logical identity. Reindexing deletes/replaces by `document_key`, not by the JSON file hash.

```powershell
python app/main.py ingest-json --json "samples/json_ingest_sample.json" --document-key "mother-miracle-class-7-maths-sample-maths-book" --board "CBSE" --language "English"
```

For a safe check without DB writes or OpenAI calls:

```powershell
python app/main.py ingest-json --json "samples/json_ingest_sample.json" --dry-run --no-json-output
```

For reprocessing the same logical document, even if the JSON file contents changed:

```powershell
python app/main.py ingest-json --json "samples/json_ingest_sample.json" --reindex
```

The JSON importer accepts either:

- the pipeline's own `*_combined_extraction.json` output, or
- a smaller JSON with `document_key`, `metadata`, `extraction.chapters` or `extraction.sections`, and optional `extraction.page_extractions`.

Metadata mapping order:

1. Root-level fields such as `document_key`, `school_name`, `class_name`, `grade`, `subject`, `board`, `language`.
2. `metadata` object.
3. `extraction` object.
4. CLI arguments, which override JSON values when provided.

Accepted aliases include `school`, `schoolName`, `class`, `standard`, `subject_name`, `subjectName`, `book_key`, and `documentKey`. If `document_key` is not supplied, the loader derives one from `school_name + class_name/grade + subject + book_title`, but it is better to provide it explicitly.

Minimal chapter/section JSON shape:

```json
{
  "document_key": "mother-miracle-class-7-maths-sample-maths-book",
  "metadata": {
    "school_name": "Mother Miracle School",
    "class_name": "Class-7",
    "grade": "Class-7",
    "board": "CBSE",
    "medium": "English",
    "book_title": "Sample Maths Book",
    "subject": "Maths",
    "language": "English"
  },
  "extraction": {
    "book_title": "Sample Maths Book",
    "subject": "Maths",
    "language": "English",
    "chapters": [
      {
        "chapter_number": "1",
        "chapter_title": "Integers",
        "start_page": 5,
        "end_page": 8,
        "printed_start_page": 1,
        "printed_end_page": 4,
        "lessons": [
          {
            "section_number": "1.1",
            "section_title": "Introduction to Integers",
            "start_page": 5,
            "end_page": 6,
            "printed_start_page": 1,
            "printed_end_page": 2,
            "lesson_text": "Full extracted text for this section..."
          }
        ]
      }
    ],
    "page_extractions": [
      {
        "page_number": 5,
        "printed_page_number": 1,
        "chapter_number": "1",
        "chapter_title": "Integers",
        "section_number": "1.1",
        "section_title": "Introduction to Integers",
        "text": "Exact extracted text for PDF page 5..."
      }
    ]
  }
}
```

Notes:

- `document_key` is stored in `embeddings_documents.document_key` and `embeddings_ingestion_runs.document_key`.
- The physical JSON file hash is stored as `json_input_hash` in metadata for audit. It is not used as the JSON document identity.
- `page_extractions` is recommended because it stores exact page-wise text in `embeddings_pages` and `embeddings_raw_text_pages`.
- If `page_extractions` is missing, the importer synthesizes page records from each chapter/section `start_page`, `end_page`, and `lesson_text`/`text`.
- For chapter-based books use `extraction.chapters`.
- For unit/section-based books use `extraction.sections`; each section lesson can include `unit_number`, `unit_title`, `section_number`, `section_title`, `start_page`, `end_page`, and `lesson_text`.

## Ingest all PDFs

Use this folder format:

```text
input\Mother Miracle School\Class-7\Maths_RSAgarwal.pdf
```

Run ingestion:

```powershell
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English"
```

For reprocessing the same files:

```powershell
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English" --reindex
```

## Dry run

Dry run extracts, detects metadata/chapter structure, selects chunking plan, and chunks, but does not write DB or create embeddings.

```powershell
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English" --dry-run
```

## Search

```powershell
python app/main.py search --query "integers addition rules" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5
```

## How chapter detection works

1. The code extracts sample pages from the PDF.
2. The LLM reads front matter, title page, contents/index pages, and sample body pages.
3. The LLM returns strict JSON for book title, language, publisher, chapters, printed page numbers, and chunking recommendation.
4. The code verifies/fills `pdf_start_page` by scanning actual extracted pages for chapter titles.
5. Page ranges are calculated and stored in `embeddings_book_chapters`.
6. Raw pages and chunks get `chapter_number`, `chapter_title`, `page_number`, and `printed_page_number`.

If the LLM fails or API key is missing, the code falls back to rule-based TOC detection.

## How auto chunking works

`DEFAULT_CHUNK_MAX_TOKENS` and `DEFAULT_CHUNK_OVERLAP_TOKENS` are now fallbacks only.

When `AUTO_CHUNKING_ENABLED=true`, each book gets a chunking profile:

```text
math_textbook       -> around 620 / 90
question_bank       -> around 520 / 70
science_textbook    -> around 760 / 110
english_literature  -> around 1000 / 150
hindi_literature    -> around 900 / 140
grammar             -> around 680 / 100
mixed_textbook      -> around 750 / 120
```

The actual selected values are stored in:

```text
embeddings_documents.chunking_strategy
embeddings_documents.chunk_max_tokens
embeddings_documents.chunk_overlap_tokens
embeddings_documents.content_profile
```

# RAG Search Validation Checklist

Use this section after a fresh reset and ingestion to check whether RAG search will work correctly.

RAG search is working fine only when these four things are true:

```text
1. Documents are stored.
2. Chapters/pages/raw text are stored correctly.
3. Embeddings are created.
4. Search returns the correct chapter/page/chunk for real questions.
```

## 1. Run ingestion first

From project root:

```powershell
cd D:\Project\JaltaSitaraApps\TeacherHelperProject\pdf_to_embeddings\pdf_embedding_pipeline
.venv\Scripts\activate
```

Fresh run:

```powershell
python app/main.py init-db
python app/main.py ingest-folder --input-dir "input" --board "CBSE" --language "English"
```

## 2. Check table counts

```powershell
python -c "from dotenv import load_dotenv; import os, psycopg; load_dotenv(); conn=psycopg.connect(os.environ['DATABASE_URL']); tables=['embeddings_documents','embeddings_book_chapters','embeddings_raw_text_pages','embeddings_chunks','embeddings_vectors']; [print(t, conn.execute(f'SELECT count(*) FROM public.{t}').fetchone()[0]) for t in tables]; conn.close()"
```

Expected:

```text
embeddings_documents       > 0
embeddings_book_chapters   > 0
embeddings_raw_text_pages  > 0
embeddings_chunks          > 0
embeddings_vectors         > 0
```

If `embeddings_vectors = 0`, RAG semantic search will not work.

## 3. Check document metadata

```powershell
python -c "from dotenv import load_dotenv; import os, psycopg; load_dotenv(); conn=psycopg.connect(os.environ['DATABASE_URL']); rows=conn.execute('SELECT id, school_name, class_name, grade, subject, book_title, chunking_strategy, chunk_max_tokens, chunk_overlap_tokens FROM public.embeddings_documents ORDER BY created_at DESC LIMIT 10').fetchall(); [print(r) for r in rows]; conn.close()"
```

You should see something like:

```text
Mother Miracle School
Class-7
Maths
RSAgarwal
math_textbook
620
90
```

## 4. Check chapter detection

```powershell
python -c "from dotenv import load_dotenv; import os, psycopg; load_dotenv(); conn=psycopg.connect(os.environ['DATABASE_URL']); rows=conn.execute('SELECT chapter_number, chapter_title, printed_start_page, pdf_start_page, pdf_end_page, confidence FROM public.embeddings_book_chapters ORDER BY document_id, chapter_number LIMIT 30').fetchall(); [print(r) for r in rows]; conn.close()"
```

For `Maths_RSAgarwal.pdf`, you should see chapters like:

```text
1 Integers
2 Fractions
3 Decimals
4 Rational Numbers
```

If the chapter table is empty, LLM/fallback chapter detection did not work correctly.

## 5. Check raw pages have chapter names

```powershell
python -c "from dotenv import load_dotenv; import os, psycopg; load_dotenv(); conn=psycopg.connect(os.environ['DATABASE_URL']); rows=conn.execute('SELECT page_number, printed_page_number, chapter_number, chapter_title, LEFT(cleaned_text,120) FROM public.embeddings_raw_text_pages WHERE chapter_title IS NOT NULL ORDER BY page_number LIMIT 20').fetchall(); [print(r) for r in rows]; conn.close()"
```

Good result:

```text
8, 1, 1, Integers, ...
9, 2, 1, Integers, ...
23, 16, 2, Fractions, ...
```

Bad result:

```text
chapter_number = null
chapter_title = null
```

If chapter values are still null, chapter mapping failed.

## 6. Run actual RAG search tests

### Test 1: Integers

```powershell
python app/main.py search --query "addition rules for integers" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5
```

Expected result should mention:

```text
Integers
Addition of Integers
Rules
Examples
```

### Test 2: Fractions

```powershell
python app/main.py search --query "reciprocal of a fraction" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5
```

Expected result should come from:

```text
Fractions
Division of Fractions
Reciprocal
```

### Test 3: Decimals

```powershell
python app/main.py search --query "multiplication of decimal by decimal" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5
```

Expected result should come from:

```text
Decimals
Multiplication of Decimals
```

## 7. Check if search is returning useful chunks

A good RAG result should have:

```text
Correct book
Correct subject
Correct class
Correct chapter
Correct page range
Useful text preview
High score compared to other results
```

A bad result looks like:

```text
query: reciprocal of fraction
result: Integers chapter
```

or:

```text
query: multiplication of decimals
result: book preface / contents page
```

## 8. Best quick pass/fail rule

Your RAG search is working fine if these three commands return correct-looking results:

```powershell
python app/main.py search --query "addition rules for integers" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5

python app/main.py search --query "reciprocal of a fraction" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5

python app/main.py search --query "multiplication of decimal by decimal" --school-name "Mother Miracle School" --class-name "Class-7" --subject "Maths" --book-title "RSAgarwal" --top-k 5
```

If all three return the right chapter/page content, retrieval is good enough for first testing.

## Subsection/day/exercise storage

The pipeline now stores subsection ranges from production JSON in a separate table:

```text
embeddings_book_subsections
```

This table is populated from JSON shapes like:

```text
extraction.section_index[*].subsections[*]
extraction.chapters[*].subsections[*]
extraction.chapters[*].lessons[*].subsections[*]
```

It stores exact subsection/page metadata such as `subsection_number`, `subsection_title`, parent chapter/section fields, PDF page range, printed page range, page arrays, includes/activities, quality flags, math lines, and the production subsection text.

After pulling this update, run the schema migration:

```powershell
python app/main.py init-db
```

Then reindex any JSON document so subsection rows are backfilled:

```powershell
python app/main.py ingest-json --json "json_input/English_Poorvi.json" --reindex
python app/main.py ingest-json --json "json_input/Maths_RSAgarwal.json" --reindex
```

List subsection ranges for a lesson/chapter:

```powershell
python app/main.py list-subsections `
  --document-key "mother-miracle-class-6-english-poorvi" `
  --section-title "A Bottle of Dew"

python app/main.py list-subsections `
  --document-key "mother-miracle-class-7-maths-rsaggarwal" `
  --chapter-number "1"
```

Fetch exact subsection text and pages:

```powershell
python app/main.py subsection-text `
  --document-key "mother-miracle-class-6-english-poorvi" `
  --section-number "1.1" `
  --subsection-number "1.1.1"

python app/main.py subsection-text `
  --document-key "mother-miracle-class-7-maths-rsaggarwal" `
  --chapter-number "1" `
  --subsection-number "1.1"
```

Fetch broader chapter/section page text from `embeddings_raw_text_pages`:

```powershell
python app/main.py chapter-text `
  --document-key "mother-miracle-class-7-maths-rsaggarwal" `
  --chapter-number "1"

python app/main.py chapter-text `
  --document-key "mother-miracle-class-6-english-poorvi" `
  --section-number "1.1"
```

Useful SQL checks:

```sql
SELECT d.document_key, count(*) AS subsection_count
FROM embeddings_book_subsections s
JOIN embeddings_documents d ON d.id = s.document_id
GROUP BY d.document_key
ORDER BY d.document_key;

SELECT subsection_number, subsection_title, pdf_start_page, pdf_end_page,
       printed_start_page, printed_end_page, page_numbers, printed_page_numbers
FROM embeddings_book_subsections s
JOIN embeddings_documents d ON d.id = s.document_id
WHERE d.document_key = 'mother-miracle-class-7-maths-rsaggarwal'
  AND s.chapter_number = '1'
ORDER BY s.pdf_start_page;
```
