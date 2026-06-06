$ErrorActionPreference = "Stop"

# Run from the folder where this script lives
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Step 1: Dropping embeddings tables only..." -ForegroundColor Cyan

@'
from dotenv import load_dotenv
from pathlib import Path
import os
import sys
import psycopg

env_path = Path.cwd() / ".env"
load_dotenv(dotenv_path=env_path)

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    print(f"DATABASE_URL not found. Checked .env at: {env_path}", file=sys.stderr)
    sys.exit(1)

conn = psycopg.connect(database_url)
conn.autocommit = True

sql = """
DROP TABLE IF EXISTS public.embeddings_subsection_vectors CASCADE;
DROP TABLE IF EXISTS public.embeddings_vectors CASCADE;
DROP TABLE IF EXISTS public.embeddings_chunks CASCADE;
DROP TABLE IF EXISTS public.embeddings_raw_text_pages CASCADE;
DROP TABLE IF EXISTS public.embeddings_pages CASCADE;
DROP TABLE IF EXISTS public.embeddings_book_subsections CASCADE;
DROP TABLE IF EXISTS public.embeddings_book_chapters CASCADE;
DROP TABLE IF EXISTS public.embeddings_ingestion_runs CASCADE;
DROP TABLE IF EXISTS public.embeddings_documents CASCADE;
"""

conn.execute(sql)
conn.close()

print("Dropped embeddings tables only.")
'@ | python

if ($LASTEXITCODE -ne 0) {
    throw "Step 1 failed: could not drop embeddings tables."
}

Write-Host ""
Write-Host "Step 2: Recreating schema using app/main.py init-db..." -ForegroundColor Cyan

python app/main.py init-db

if ($LASTEXITCODE -ne 0) {
    throw "Step 2 failed: init-db failed."
}

Write-Host ""
Write-Host "Step 3: Applying section_key unique-constraint fix..." -ForegroundColor Cyan

@'
from dotenv import load_dotenv
from pathlib import Path
import os
import sys
import psycopg

env_path = Path.cwd() / ".env"
load_dotenv(dotenv_path=env_path)

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    print(f"DATABASE_URL not found. Checked .env at: {env_path}", file=sys.stderr)
    sys.exit(1)

conn = psycopg.connect(database_url)
conn.autocommit = True

sql = """
ALTER TABLE public.embeddings_book_chapters
ADD COLUMN IF NOT EXISTS section_key TEXT;

UPDATE public.embeddings_book_chapters
SET section_key = COALESCE(
    NULLIF(section_number, ''),
    NULLIF(chapter_number, ''),
    NULLIF(chapter_title, ''),
    id::text
)
WHERE section_key IS NULL OR section_key = '';

ALTER TABLE public.embeddings_book_chapters
DROP CONSTRAINT IF EXISTS embeddings_book_chapters_document_id_chapter_number_chapter_key;

ALTER TABLE public.embeddings_book_chapters
DROP CONSTRAINT IF EXISTS embeddings_book_chapters_document_id_chapter_number_chapter_title_key;

ALTER TABLE public.embeddings_book_chapters
DROP CONSTRAINT IF EXISTS embeddings_book_chapters_document_id_chapter_number_key;

ALTER TABLE public.embeddings_book_chapters
DROP CONSTRAINT IF EXISTS embeddings_book_chapters_document_id_section_key_key;

ALTER TABLE public.embeddings_book_chapters
ADD CONSTRAINT embeddings_book_chapters_document_id_section_key_key
UNIQUE (document_id, section_key);
"""

conn.execute(sql)
conn.close()

print("Fixed embeddings_book_chapters unique constraint.")
'@ | python

if ($LASTEXITCODE -ne 0) {
    throw "Step 3 failed: could not apply section_key constraint fix."
}

Write-Host ""
Write-Host "Reset complete. Now rerun ingest-json commands:" -ForegroundColor Green
Write-Host 'python app/main.py ingest-json --json "json_input/English_Poorvi_production_ready.json" --reindex'
Write-Host 'python app/main.py ingest-json --json "json_input/Maths_RSAgarwal_math_aware_extraction_v4_production_safe.json" --reindex'
