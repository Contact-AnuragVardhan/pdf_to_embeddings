from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, model: str = "text-embedding-3-large") -> None:
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            logger.debug("Falling back to cl100k_base token encoding for model %s", model)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        chunks: list[str] = []
        start = 0
        safe_overlap = max(0, min(overlap_tokens, max_tokens // 3))
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(self.encoding.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = max(0, end - safe_overlap)
        return chunks
