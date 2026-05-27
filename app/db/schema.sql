CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION embeddings_touch_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS embeddings_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title text NOT NULL,
    book_title text,
    normalized_title text,
    school_name text,
    class_name text,
    subject text,
    grade text,
    board text,
    medium text,
    language text,
    detected_language text,
    primary_language text,
    languages_detected jsonb DEFAULT '[]'::jsonb,
    publisher text,
    edition text,
    publication_year text,
    isbn text,
    author text,
    source_type text DEFAULT 'readable_pdf',
    source_uri text,
    file_name text NOT NULL,
    file_path text NOT NULL,
    file_hash text UNIQUE NOT NULL,
    file_size_bytes bigint,
    mime_type text DEFAULT 'application/pdf',
    total_pages int,
    total_words int,
    total_tokens int,
    extraction_status text,
    copyright_status text,
    license_notes text,
    llm_metadata_model text,
    llm_metadata_confidence numeric,
    structure_detected_by text,
    content_profile text,
    chunking_strategy text,
    chunk_max_tokens int,
    chunk_overlap_tokens int,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS book_title text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS school_name text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS class_name text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS primary_language text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS languages_detected jsonb DEFAULT '[]'::jsonb;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS llm_metadata_model text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS llm_metadata_confidence numeric;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS structure_detected_by text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS content_profile text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS chunking_strategy text;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS chunk_max_tokens int;
ALTER TABLE embeddings_documents ADD COLUMN IF NOT EXISTS chunk_overlap_tokens int;
UPDATE embeddings_documents SET book_title = title WHERE book_title IS NULL;

DROP TRIGGER IF EXISTS embeddings_documents_touch_updated_at ON embeddings_documents;
CREATE TRIGGER embeddings_documents_touch_updated_at
BEFORE UPDATE ON embeddings_documents
FOR EACH ROW EXECUTE FUNCTION embeddings_touch_updated_at();

CREATE TABLE IF NOT EXISTS embeddings_pages (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES embeddings_documents(id) ON DELETE CASCADE,
    page_number int NOT NULL,
    raw_text text,
    cleaned_text text,
    detected_language text,
    word_count int,
    token_count int,
    has_text boolean DEFAULT true,
    has_math boolean DEFAULT false,
    has_table_like_text boolean DEFAULT false,
    has_devanagari boolean DEFAULT false,
    has_english boolean DEFAULT false,
    extraction_method text DEFAULT 'pymupdf',
    extraction_quality text,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now(),
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS embeddings_book_chapters (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES embeddings_documents(id) ON DELETE CASCADE,
    chapter_number text,
    chapter_title text NOT NULL,
    printed_start_page int,
    printed_end_page int,
    pdf_start_page int,
    pdf_end_page int,
    detected_by text,
    confidence numeric,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(document_id, chapter_number, chapter_title)
);

DROP TRIGGER IF EXISTS embeddings_book_chapters_touch_updated_at ON embeddings_book_chapters;
CREATE TRIGGER embeddings_book_chapters_touch_updated_at
BEFORE UPDATE ON embeddings_book_chapters
FOR EACH ROW EXECUTE FUNCTION embeddings_touch_updated_at();

CREATE TABLE IF NOT EXISTS embeddings_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES embeddings_documents(id) ON DELETE CASCADE,
    page_start int NOT NULL,
    page_end int NOT NULL,
    chunk_index int NOT NULL,

    book_title text,
    school_name text,
    class_name text,
    subject text,
    grade text,
    board text,
    medium text,
    language text,
    detected_language text,

    chapter_number text,
    chapter_title text,
    unit_title text,
    lesson_title text,
    section_title text,
    subsection_title text,
    topic text,
    subtopic text,

    chunk_type text,
    content_domain text,
    difficulty_level text,
    pedagogical_role text,

    content text NOT NULL,
    content_clean text NOT NULL,
    content_for_embedding text NOT NULL,
    summary text,
    keywords text[],
    important_terms text[],
    formulas text[],
    numbers text[],
    question_types text[],

    word_count int,
    token_count int,
    char_count int,

    has_formula boolean DEFAULT false,
    has_numbers boolean DEFAULT false,
    has_questions boolean DEFAULT false,
    has_exercises boolean DEFAULT false,
    has_examples boolean DEFAULT false,
    has_definition boolean DEFAULT false,
    has_table_like_text boolean DEFAULT false,
    has_devanagari boolean DEFAULT false,
    has_english boolean DEFAULT false,

    source_label text,
    citation_text text,

    search_vector tsvector,

    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(document_id, chunk_index)
);

ALTER TABLE embeddings_chunks ADD COLUMN IF NOT EXISTS school_name text;
ALTER TABLE embeddings_chunks ADD COLUMN IF NOT EXISTS class_name text;
ALTER TABLE embeddings_chunks ADD COLUMN IF NOT EXISTS book_title text;

CREATE TABLE IF NOT EXISTS embeddings_raw_text_pages (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES embeddings_documents(id) ON DELETE CASCADE,
    school_name text,
    class_name text,
    grade text,
    subject text,
    book_title text,
    chapter_number text,
    chapter_title text,
    page_number int NOT NULL,
    printed_page_number int,
    raw_text text,
    cleaned_text text,
    detected_language text,
    word_count int,
    token_count int,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(document_id, page_number)
);

ALTER TABLE embeddings_raw_text_pages ADD COLUMN IF NOT EXISTS printed_page_number int;

DROP TRIGGER IF EXISTS embeddings_raw_text_pages_touch_updated_at ON embeddings_raw_text_pages;
CREATE TRIGGER embeddings_raw_text_pages_touch_updated_at
BEFORE UPDATE ON embeddings_raw_text_pages
FOR EACH ROW EXECUTE FUNCTION embeddings_touch_updated_at();

CREATE OR REPLACE FUNCTION embeddings_chunks_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('simple', coalesce(NEW.school_name, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(NEW.class_name, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(NEW.book_title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(NEW.chapter_title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(NEW.topic, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(NEW.section_title, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(NEW.content_clean, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS embeddings_chunks_touch_updated_at ON embeddings_chunks;
CREATE TRIGGER embeddings_chunks_touch_updated_at
BEFORE UPDATE ON embeddings_chunks
FOR EACH ROW EXECUTE FUNCTION embeddings_touch_updated_at();

DROP TRIGGER IF EXISTS embeddings_chunks_search_vector_trigger ON embeddings_chunks;
CREATE TRIGGER embeddings_chunks_search_vector_trigger
BEFORE INSERT OR UPDATE OF school_name, class_name, book_title, chapter_title, topic, section_title, content_clean
ON embeddings_chunks
FOR EACH ROW EXECUTE FUNCTION embeddings_chunks_search_vector_update();

CREATE TABLE IF NOT EXISTS embeddings_vectors (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id uuid REFERENCES embeddings_chunks(id) ON DELETE CASCADE,
    document_id uuid REFERENCES embeddings_documents(id) ON DELETE CASCADE,
    embedding_model text NOT NULL DEFAULT 'text-embedding-3-large',
    embedding_dimensions int NOT NULL DEFAULT 3072,
    embedding vector(3072) NOT NULL,
    embedding_input_hash text NOT NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE(chunk_id, embedding_model, embedding_dimensions)
);

CREATE TABLE IF NOT EXISTS embeddings_ingestion_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES embeddings_documents(id) ON DELETE SET NULL,
    file_path text,
    file_hash text,
    status text,
    started_at timestamptz DEFAULT now(),
    finished_at timestamptz,
    pages_extracted int DEFAULT 0,
    chunks_created int DEFAULT 0,
    embeddings_created int DEFAULT 0,
    error_message text,
    warnings jsonb DEFAULT '[]'::jsonb,
    metadata jsonb DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS embeddings_documents_file_hash_idx ON embeddings_documents(file_hash);
CREATE INDEX IF NOT EXISTS embeddings_documents_school_class_idx ON embeddings_documents(school_name, class_name);
CREATE INDEX IF NOT EXISTS embeddings_documents_subject_grade_language_idx ON embeddings_documents(subject, grade, language);
CREATE INDEX IF NOT EXISTS embeddings_documents_title_idx ON embeddings_documents(title);
CREATE INDEX IF NOT EXISTS embeddings_documents_book_title_idx ON embeddings_documents(book_title);
CREATE INDEX IF NOT EXISTS embeddings_documents_content_profile_idx ON embeddings_documents(content_profile);

CREATE INDEX IF NOT EXISTS embeddings_pages_document_page_idx ON embeddings_pages(document_id, page_number);

CREATE INDEX IF NOT EXISTS embeddings_book_chapters_document_idx ON embeddings_book_chapters(document_id);
CREATE INDEX IF NOT EXISTS embeddings_book_chapters_title_idx ON embeddings_book_chapters(chapter_title);
CREATE INDEX IF NOT EXISTS embeddings_book_chapters_pdf_range_idx ON embeddings_book_chapters(document_id, pdf_start_page, pdf_end_page);

CREATE INDEX IF NOT EXISTS embeddings_raw_text_pages_document_page_idx ON embeddings_raw_text_pages(document_id, page_number);
CREATE INDEX IF NOT EXISTS embeddings_raw_text_pages_printed_page_idx ON embeddings_raw_text_pages(document_id, printed_page_number);
CREATE INDEX IF NOT EXISTS embeddings_raw_text_pages_chapter_idx ON embeddings_raw_text_pages(chapter_title);
CREATE INDEX IF NOT EXISTS embeddings_raw_text_pages_school_class_idx ON embeddings_raw_text_pages(school_name, class_name);

CREATE INDEX IF NOT EXISTS embeddings_chunks_document_idx ON embeddings_chunks(document_id);
CREATE INDEX IF NOT EXISTS embeddings_chunks_school_class_idx ON embeddings_chunks(school_name, class_name);
CREATE INDEX IF NOT EXISTS embeddings_chunks_subject_grade_language_idx ON embeddings_chunks(subject, grade, language);
CREATE INDEX IF NOT EXISTS embeddings_chunks_chapter_title_idx ON embeddings_chunks(chapter_title);
CREATE INDEX IF NOT EXISTS embeddings_chunks_topic_idx ON embeddings_chunks(topic);
CREATE INDEX IF NOT EXISTS embeddings_chunks_chunk_type_idx ON embeddings_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS embeddings_chunks_search_vector_idx ON embeddings_chunks USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS embeddings_chunks_metadata_idx ON embeddings_chunks USING GIN(metadata);

CREATE INDEX IF NOT EXISTS embeddings_vectors_document_idx ON embeddings_vectors(document_id);

DO $$
BEGIN
    BEGIN
        EXECUTE 'CREATE INDEX IF NOT EXISTS embeddings_vectors_hnsw_idx ON embeddings_vectors USING hnsw (embedding vector_cosine_ops)';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'HNSW index creation failed or is unsupported; falling back to IVFFlat. Error: %', SQLERRM;
        BEGIN
            EXECUTE 'CREATE INDEX IF NOT EXISTS embeddings_vectors_ivfflat_idx ON embeddings_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'IVFFlat index creation also failed. Vector search still works without ANN index. Error: %', SQLERRM;
        END;
    END;
END;
$$;
