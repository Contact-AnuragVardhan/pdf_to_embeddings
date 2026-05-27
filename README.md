# pdf_embedding_pipeline

Standalone CLI project to ingest readable/selectable PDF books into PostgreSQL + pgvector for future Teacher Helper RAG search.

It supports Hindi, English, mixed Hindi-English, maths, grammar, stories, poems, examples, exercises, definitions, formulas, numbers, and table-like text. It does **not** depend on the Teacher Helper app.

## What it stores

All database table names start with `embeddings_`:

| Table | Purpose |
|---|---|
| `embeddings_documents` | One row per PDF/book with school/book metadata and file hash |
| `embeddings_pages` | One row per extracted PDF page with raw/cleaned text and page flags |
| `embeddings_chunks` | Meaningful RAG chunks with content, metadata, flags, keywords, formulas, and citations |
| `embeddings_vectors` | `text-embedding-3-large` vectors stored as `vector(3072)` |
| `embeddings_ingestion_runs` | Audit trail for ingestion status, counts, warnings, and errors |

Important design choices:

- Original chunks are stored, not only embeddings.
- Page numbers and source labels are preserved for citation.
- `metadata jsonb` is included for future expansion.
- `content_for_embedding` is stored separately from `content_clean`.
- Hybrid search is used: vector search + PostgreSQL full-text search + metadata scoring.
- Metadata filters support future Teacher Helper use cases: subject, grade, board, language, chapter, topic, page range, and chunk type.
- Formulas, numbers, Hindi punctuation, Devanagari, and math symbols are preserved instead of over-cleaned.

## Limits

This pipeline is for **readable/selectable PDFs**. If most pages have no selectable text, the tool logs:

```text
This appears to be scanned PDF. OCR is required before embedding.
```

OCR is intentionally not implemented in this version. OCR scanned books first, then run this pipeline.

Use copyrighted books only when you have the proper license/permission for indexing and app usage.

## PostgreSQL setup

Install PostgreSQL and pgvector. Then create a database:

```sql
CREATE DATABASE pdf_rag_db;
```

The project schema enables:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

The vector table uses:

```sql
embedding vector(3072) NOT NULL
```

The schema tries to create an HNSW cosine index. If HNSW is unavailable, it falls back to IVFFlat.

## Python setup

Windows PowerShell / Command Prompt:

```bat
cd pdf_embedding_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS:

```bash
cd pdf_embedding_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_key
DATABASE_URL=postgresql://user:password@localhost:5432/pdf_rag_db
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072
```

## Initialize database

```bash
python main.py init-db
```

## Ingest one PDF

Windows Command Prompt:

```bat
python main.py ingest ^
  --pdf "input/Hindi Vyakaran Rachna.pdf" ^
  --title "Hindi Vyakaran Rachna" ^
  --subject "Hindi" ^
  --grade "5" ^
  --board "CBSE" ^
  --medium "Hindi" ^
  --language "Hindi" ^
  --publisher "Unknown" ^
  --copyright-status "licensed/internal"
```

Linux/macOS:

```bash
python main.py ingest \
  --pdf "input/Hindi Vyakaran Rachna.pdf" \
  --title "Hindi Vyakaran Rachna" \
  --subject "Hindi" \
  --grade "5" \
  --board "CBSE" \
  --medium "Hindi" \
  --language "Hindi" \
  --publisher "Unknown" \
  --copyright-status "licensed/internal"
```

## Reindex one PDF

```bash
python main.py ingest --pdf "input/book.pdf" --title "Book" --subject "Maths" --grade "5" --reindex
```

## Batch ingest a folder

```bash
python main.py ingest-folder \
  --input-dir "input" \
  --subject "Hindi" \
  --grade "5" \
  --board "CBSE" \
  --language "Hindi"
```

## Dry run / validate only

Dry run extracts and chunks but does not insert into the database or call OpenAI:

```bash
python main.py ingest --pdf "input/book.pdf" --title "Book" --dry-run
```

Validate only checks file existence/type and metadata arguments:

```bash
python main.py ingest --pdf "input/book.pdf" --title "Book" --validate-only
```

## Search

```bash
python main.py search \
  --query "संज्ञा पर 40 मिनट का lesson plan" \
  --subject "Hindi" \
  --grade "5" \
  --language "Hindi" \
  --top-k 8
```

Search returns:

- chunk id
- content preview
- book title
- subject
- grade
- language
- chapter title
- section title
- topic
- chunk type
- page range
- source label
- citation text
- final weighted score

## Metadata fields

Document metadata captures book-level values from CLI plus automatically computed file and extraction details.

Chunk metadata captures:

- book/school metadata: book title, subject, grade, board, medium, language
- structure metadata: chapter, unit, lesson, section, subsection, topic
- content classification: chunk type, content domain, difficulty level, pedagogical role
- content features: formulas, numbers, question types, important terms, keywords
- citation metadata: source label and page citation

## Opening in PyCharm

Open the `pdf_embedding_pipeline` folder in PyCharm, create a Python interpreter from `.venv`, install requirements, create `.env`, then run `main.py` with the CLI arguments above.
