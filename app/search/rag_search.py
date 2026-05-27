from __future__ import annotations

import re
from typing import Any

from rich.console import Console
from rich.table import Table

from config import Settings
from ingestion.embedding_service import OpenAIEmbeddingService
from ingestion.repository import RagRepository
from utils.token_counter import TokenCounter


class RagSearchService:
    def __init__(self, settings: Settings, repository: RagRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.token_counter = TokenCounter(settings.openai_embedding_model)
        self.embedding_service = OpenAIEmbeddingService(settings, self.token_counter)
        self.console = Console()

    def search(self, *, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = {k: v for k, v in (filters or {}).items() if v}
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.repository.vector_search(query_embedding, filters, limit=max(top_k * 3, top_k))
        keyword_results = self.repository.keyword_search(query, filters, limit=max(top_k * 3, top_k))
        merged = self._merge(vector_results, keyword_results)
        for row in merged.values():
            row["metadata_score"] = self._metadata_score(row, query, filters)
            row["final_score"] = (
                0.70 * float(row.get("vector_score") or 0)
                + 0.20 * float(row.get("keyword_score") or 0)
                + 0.10 * float(row.get("metadata_score") or 0)
            )
        ranked = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
        return ranked[:top_k]

    def _merge(self, vector_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in vector_results + keyword_results:
            chunk_id = row["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = row
            else:
                merged[chunk_id]["vector_score"] = max(float(merged[chunk_id].get("vector_score") or 0), float(row.get("vector_score") or 0))
                merged[chunk_id]["keyword_score"] = max(float(merged[chunk_id].get("keyword_score") or 0), float(row.get("keyword_score") or 0))
        return merged

    def _metadata_score(self, row: dict[str, Any], query: str, filters: dict[str, Any]) -> float:
        score = 0.0
        if filters.get("subject") and row.get("subject") == filters["subject"]:
            score += 0.25
        if filters.get("grade") and row.get("grade") == filters["grade"]:
            score += 0.20
        if filters.get("language") and row.get("language") == filters["language"]:
            score += 0.20
        query_terms = {t.lower() for t in re.findall(r"[A-Za-z\u0900-\u097F]{3,}", query)}
        text = " ".join(str(row.get(k) or "") for k in ["topic", "chapter_title", "section_title", "chunk_type"]).lower()
        if any(term in text for term in query_terms):
            score += 0.20
        if row.get("chunk_type") in {"definition", "explanation", "example", "activity", "grammar_rule", "worked_example"}:
            score += 0.15
        return min(score, 1.0)

    def print_results(self, results: list[dict[str, Any]]) -> None:
        if not results:
            self.console.print("[yellow]No results found.[/yellow]")
            return
        table = Table(title="RAG Search Results", show_lines=True)
        table.add_column("#", justify="right", width=3)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Source", width=32)
        table.add_column("Type", width=14)
        table.add_column("Pages", width=9)
        table.add_column("Preview", width=70)
        for i, row in enumerate(results, start=1):
            preview = " ".join((row.get("content_clean") or row.get("content") or "").split())[:260]
            pages = f"{row.get('page_start')}-{row.get('page_end')}"
            table.add_row(
                str(i),
                f"{float(row.get('final_score') or 0):.4f}",
                row.get("source_label") or row.get("book_title") or "",
                row.get("chunk_type") or "",
                pages,
                preview,
            )
        self.console.print(table)
        for row in results:
            self.console.print(f"[dim]{row.get('citation_text')} | chunk_id={row.get('chunk_id')}[/dim]")
