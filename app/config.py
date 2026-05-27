from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    openai_metadata_model: str
    metadata_sample_pages: int
    metadata_max_output_tokens: int
    auto_metadata_enabled: bool
    auto_chunking_enabled: bool
    database_url: str
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    embedding_batch_size: int
    reindex_existing: bool
    log_level: str
    project_root: Path

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_embedding_dimensions=_as_int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS"), 3072),
            # Change this freely to gpt-4o-mini, gpt-5.4-mini, etc.; the code is model-name agnostic.
            openai_metadata_model=os.getenv("OPENAI_METADATA_MODEL", "gpt-5.4"),
            metadata_sample_pages=_as_int(os.getenv("METADATA_SAMPLE_PAGES"), 20),
            metadata_max_output_tokens=_as_int(os.getenv("METADATA_MAX_OUTPUT_TOKENS"), 6000),
            auto_metadata_enabled=_as_bool(os.getenv("AUTO_METADATA_ENABLED"), True),
            auto_chunking_enabled=_as_bool(os.getenv("AUTO_CHUNKING_ENABLED"), True),
            database_url=os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/pdf_rag_db"),
            chunk_max_tokens=_as_int(os.getenv("DEFAULT_CHUNK_MAX_TOKENS"), 750),
            chunk_overlap_tokens=_as_int(os.getenv("DEFAULT_CHUNK_OVERLAP_TOKENS"), 120),
            embedding_batch_size=_as_int(os.getenv("EMBEDDING_BATCH_SIZE"), 64),
            reindex_existing=_as_bool(os.getenv("REINDEX_EXISTING"), False),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            project_root=Path(__file__).resolve().parent,
        )

    def validate_for_embedding(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for ingestion/search that calls the OpenAI embeddings API.")
        if self.openai_embedding_model != "text-embedding-3-large":
            raise ValueError("This project requires OPENAI_EMBEDDING_MODEL=text-embedding-3-large.")
        if self.openai_embedding_dimensions != 3072:
            raise ValueError("This project stores pgvector vector(3072). Set OPENAI_EMBEDDING_DIMENSIONS=3072.")
