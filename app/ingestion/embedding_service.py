from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from rich.progress import Progress
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from config import Settings
from utils.hashing import sha256_text
from utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    document_id: str
    embedding_model: str
    embedding_dimensions: int
    embedding: list[float]
    embedding_input_hash: str


@dataclass(frozen=True)
class SubsectionEmbeddingRecord:
    subsection_id: str
    document_id: str
    embedding_model: str
    embedding_dimensions: int
    embedding: list[float]
    embedding_input_hash: str
    content_for_embedding: str
    token_count: int
    text_was_truncated: bool
    metadata: dict[str, Any]


class OpenAIEmbeddingService:
    MODEL_MAX_INPUT_TOKENS = 8192

    def __init__(self, settings: Settings, token_counter: TokenCounter) -> None:
        settings.validate_for_embedding()
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.token_counter = token_counter

    def embed_chunks(self, chunks: list[dict], document_id: str) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        prepared: list[tuple[str, str]] = []
        for chunk in chunks:
            text = (chunk.get("content_for_embedding") or "").strip()
            if not text:
                continue
            token_count = self.token_counter.count(text)
            if token_count > self.MODEL_MAX_INPUT_TOKENS:
                raise ValueError(
                    f"Chunk {chunk.get('id') or chunk.get('chunk_index')} has {token_count} tokens, "
                    f"above embedding max {self.MODEL_MAX_INPUT_TOKENS}. Reduce DEFAULT_CHUNK_MAX_TOKENS."
                )
            prepared.append((str(chunk["id"]), text))

        with Progress() as progress:
            task = progress.add_task("Generating embeddings", total=len(prepared))
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                inputs = [text for _, text in batch]
                vectors = self._embed_batch(inputs)
                for (chunk_id, text), vector in zip(batch, vectors):
                    if len(vector) != self.settings.openai_embedding_dimensions:
                        raise ValueError(
                            f"Embedding dimension mismatch for chunk {chunk_id}: "
                            f"expected {self.settings.openai_embedding_dimensions}, got {len(vector)}"
                        )
                    records.append(
                        EmbeddingRecord(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            embedding_model=self.settings.openai_embedding_model,
                            embedding_dimensions=self.settings.openai_embedding_dimensions,
                            embedding=vector,
                            embedding_input_hash=sha256_text(text),
                        )
                    )
                progress.update(task, advance=len(batch))
        return records

    def embed_subsections(self, subsections: list[dict[str, Any]], document_id: str) -> list[SubsectionEmbeddingRecord]:
        """Generate one embedding per subsection/day/exercise row.

        This complements chunk-level embeddings. Each subsection vector is built
        from the exact subsection text plus compact structural/page metadata so
        downstream lesson-planning can retrieve at day/exercise granularity.
        """
        records: list[SubsectionEmbeddingRecord] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        prepared: list[tuple[dict[str, Any], str, int, bool, dict[str, Any]]] = []

        for subsection in subsections:
            text = self._build_subsection_embedding_input(subsection).strip()
            if not text:
                continue
            text, token_count, was_truncated = self._fit_embedding_input(text)
            metadata = self._subsection_embedding_metadata(subsection, token_count, was_truncated)
            prepared.append((subsection, text, token_count, was_truncated, metadata))

        with Progress() as progress:
            task = progress.add_task("Generating subsection embeddings", total=len(prepared))
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                inputs = [text for _, text, _, _, _ in batch]
                vectors = self._embed_batch(inputs)
                for (subsection, text, token_count, was_truncated, metadata), vector in zip(batch, vectors):
                    if len(vector) != self.settings.openai_embedding_dimensions:
                        raise ValueError(
                            f"Embedding dimension mismatch for subsection {subsection.get('id')}: "
                            f"expected {self.settings.openai_embedding_dimensions}, got {len(vector)}"
                        )
                    records.append(
                        SubsectionEmbeddingRecord(
                            subsection_id=str(subsection["id"]),
                            document_id=document_id,
                            embedding_model=self.settings.openai_embedding_model,
                            embedding_dimensions=self.settings.openai_embedding_dimensions,
                            embedding=vector,
                            embedding_input_hash=sha256_text(text),
                            content_for_embedding=text,
                            token_count=token_count,
                            text_was_truncated=was_truncated,
                            metadata=metadata,
                        )
                    )
                progress.update(task, advance=len(batch))
        return records

    def _build_subsection_embedding_input(self, subsection: dict[str, Any]) -> str:
        page_numbers = subsection.get("page_numbers") or []
        printed_page_numbers = subsection.get("printed_page_numbers") or []
        includes = subsection.get("includes") or subsection.get("included_exercises_or_activities") or []
        labels = [
            f"Chapter: {subsection.get('chapter_number') or ''} {subsection.get('chapter_title') or ''}".strip(),
            f"Unit: {subsection.get('unit_number') or ''} {subsection.get('unit_title') or ''}".strip(),
            f"Section/Lesson: {subsection.get('section_number') or ''} {subsection.get('section_title') or subsection.get('lesson_title') or ''}".strip(),
            f"Subsection: {subsection.get('subsection_number') or ''} {subsection.get('subsection_title') or subsection.get('anchor_marker') or ''}".strip(),
            f"PDF pages: {subsection.get('pdf_start_page') or ''}-{subsection.get('pdf_end_page') or ''}".strip(" -"),
            f"Printed pages: {subsection.get('printed_start_page') or ''}-{subsection.get('printed_end_page') or ''}".strip(" -"),
            f"Page numbers: {', '.join(str(p) for p in page_numbers)}" if page_numbers else "",
            f"Printed page numbers: {', '.join(str(p) for p in printed_page_numbers)}" if printed_page_numbers else "",
            f"Includes: {', '.join(str(x) for x in includes)}" if includes else "",
        ]
        header = "\n".join(label for label in labels if label and not label.endswith(":"))
        body = (subsection.get("subsection_text_plain") or subsection.get("subsection_text") or "").strip()
        return f"{header}\n\nSubsection text:\n{body}" if header else body

    def _fit_embedding_input(self, text: str) -> tuple[str, int, bool]:
        token_count = self.token_counter.count(text)
        if token_count <= self.MODEL_MAX_INPUT_TOKENS:
            return text, token_count, False
        trimmed = self.token_counter.split_by_tokens(text, self.MODEL_MAX_INPUT_TOKENS, overlap_tokens=0)[0]
        return trimmed, self.token_counter.count(trimmed), True

    def _subsection_embedding_metadata(
        self,
        subsection: dict[str, Any],
        token_count: int,
        was_truncated: bool,
    ) -> dict[str, Any]:
        return {
            "source_table": "embeddings_book_subsections",
            "embedding_granularity": "subsection",
            "subsection_id": str(subsection.get("id")),
            "chapter_number": subsection.get("chapter_number"),
            "chapter_title": subsection.get("chapter_title"),
            "unit_number": subsection.get("unit_number"),
            "unit_title": subsection.get("unit_title"),
            "lesson_title": subsection.get("lesson_title"),
            "section_number": subsection.get("section_number"),
            "section_title": subsection.get("section_title"),
            "subsection_number": subsection.get("subsection_number"),
            "subsection_title": subsection.get("subsection_title"),
            "anchor_marker": subsection.get("anchor_marker"),
            "anchor_pdf_page": subsection.get("anchor_pdf_page"),
            "anchor_printed_page": subsection.get("anchor_printed_page"),
            "pdf_start_page": subsection.get("pdf_start_page"),
            "pdf_end_page": subsection.get("pdf_end_page"),
            "printed_start_page": subsection.get("printed_start_page"),
            "printed_end_page": subsection.get("printed_end_page"),
            "page_numbers": subsection.get("page_numbers") or [],
            "printed_page_numbers": subsection.get("printed_page_numbers") or [],
            "includes": subsection.get("includes") or [],
            "included_exercises_or_activities": subsection.get("included_exercises_or_activities") or [],
            "embedding_token_count": token_count,
            "embedding_input_truncated": was_truncated,
            "embedding_model": self.settings.openai_embedding_model,
            "embedding_dimensions": self.settings.openai_embedding_dimensions,
        }

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, TimeoutError, ConnectionError)),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        logger.info("Embedding batch of %s chunks", len(inputs))
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=inputs,
            dimensions=self.settings.openai_embedding_dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if self.token_counter.count(query) > self.MODEL_MAX_INPUT_TOKENS:
            raise ValueError("Search query is too large for the embedding model.")
        return self._embed_batch([query])[0]
