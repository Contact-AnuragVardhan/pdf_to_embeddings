from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

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
