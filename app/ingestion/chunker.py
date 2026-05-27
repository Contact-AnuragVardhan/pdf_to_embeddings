from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from ingestion.book_structure import BookStructure, ChapterResolver
from ingestion.metadata_builder import MetadataBuilder
from ingestion.pdf_extractor import ExtractedPage
from ingestion.structure_detector import StructureDetector, StructureState
from utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ParagraphUnit:
    text: str
    page_number: int
    structure: StructureState


class MeaningfulChunker:
    def __init__(
        self,
        token_counter: TokenCounter,
        structure_detector: StructureDetector,
        metadata_builder: MetadataBuilder,
        max_tokens: int = 750,
        overlap_tokens: int = 120,
        min_useful_tokens: int = 80,
    ) -> None:
        self.token_counter = token_counter
        self.structure_detector = structure_detector
        self.metadata_builder = metadata_builder
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_useful_tokens = min_useful_tokens

    def chunk_pages(self, pages: list[ExtractedPage], metadata: dict[str, Any], book_structure: BookStructure | None = None) -> list[dict[str, Any]]:
        units = self._build_units(pages, book_structure)
        chunks: list[dict[str, Any]] = []
        current: list[ParagraphUnit] = []
        current_tokens = 0
        chunk_index = 0

        for unit in units:
            unit_tokens = self.token_counter.count(unit.text)
            if unit_tokens > self.max_tokens:
                if current:
                    chunk_index = self._flush(current, metadata, chunks, chunk_index)
                    current = []
                    current_tokens = 0
                for split in self._split_large_unit(unit):
                    chunk_index = self._flush([split], metadata, chunks, chunk_index)
                continue

            new_chapter = self._starts_new_chapter(unit, current)
            would_exceed = current and current_tokens + unit_tokens > self.max_tokens
            if current and (new_chapter or would_exceed):
                chunk_index = self._flush(current, metadata, chunks, chunk_index)
                current = self._overlap_tail(current) if not new_chapter else []
                current_tokens = sum(self.token_counter.count(u.text) for u in current)

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            self._flush(current, metadata, chunks, chunk_index)
        return chunks

    def _build_units(self, pages: list[ExtractedPage], book_structure: BookStructure | None = None) -> list[ParagraphUnit]:
        resolver = ChapterResolver(book_structure.chapters) if book_structure else None
        state = StructureState()
        units: list[ParagraphUnit] = []
        for page in pages:
            if not page.cleaned_text.strip():
                continue
            # Prefer TOC/LLM page-to-chapter mapping over fragile heading regex.
            page_state = resolver.structure_for_page(page.page_number) if resolver else StructureState()
            if page_state.chapter_title:
                state = page_state
            paragraphs = self._split_paragraphs(page.cleaned_text)
            for para in paragraphs:
                heading = self.structure_detector.detect_heading(para)
                state = self.structure_detector.update_state(state, heading)
                # Do not let generic topic headings erase the LLM/TOC chapter mapping.
                if page_state.chapter_title:
                    state.chapter_number = page_state.chapter_number
                    state.chapter_title = page_state.chapter_title
                units.append(ParagraphUnit(para, page.page_number, StructureState(**state.as_dict())))
        return units

    def _split_paragraphs(self, text: str) -> list[str]:
        blocks = re.split(r"\n\s*\n", text)
        paragraphs: list[str] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if self.token_counter.count(block) <= self.max_tokens:
                paragraphs.append(block)
                continue
            parts = re.split(r"(?<=[.!?।॥])\s+", block)
            buffer: list[str] = []
            for part in parts:
                if not part.strip():
                    continue
                candidate = " ".join(buffer + [part.strip()])
                if buffer and self.token_counter.count(candidate) > self.max_tokens:
                    paragraphs.append(" ".join(buffer).strip())
                    buffer = [part.strip()]
                else:
                    buffer.append(part.strip())
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
        return paragraphs

    def _starts_new_chapter(self, unit: ParagraphUnit, current: list[ParagraphUnit]) -> bool:
        if not current:
            return False
        return bool(
            unit.structure.chapter_title
            and unit.structure.chapter_title != current[-1].structure.chapter_title
        )

    def _split_large_unit(self, unit: ParagraphUnit) -> list[ParagraphUnit]:
        pieces = self.token_counter.split_by_tokens(unit.text, self.max_tokens, self.overlap_tokens)
        return [ParagraphUnit(piece, unit.page_number, unit.structure) for piece in pieces if piece.strip()]

    def _overlap_tail(self, current: list[ParagraphUnit]) -> list[ParagraphUnit]:
        if self.overlap_tokens <= 0:
            return []
        tail: list[ParagraphUnit] = []
        tokens = 0
        for unit in reversed(current):
            unit_tokens = self.token_counter.count(unit.text)
            if tokens + unit_tokens > self.overlap_tokens:
                break
            tail.insert(0, unit)
            tokens += unit_tokens
        return tail

    def _flush(
        self,
        units: list[ParagraphUnit],
        metadata: dict[str, Any],
        chunks: list[dict[str, Any]],
        chunk_index: int,
    ) -> int:
        content = "\n\n".join(u.text for u in units).strip()
        if not content:
            return chunk_index
        token_count = self.token_counter.count(content)
        classification = self.structure_detector.classify(content, metadata.get("subject"))
        is_small_but_useful = classification.chunk_type in {"formula", "definition", "table", "vocabulary", "grammar_rule"}
        if token_count < self.min_useful_tokens and chunks and not is_small_but_useful:
            prev = chunks[-1]
            if prev.get("chapter_title") == units[0].structure.chapter_title and prev["token_count"] + token_count <= self.max_tokens:
                prev["content"] += "\n\n" + content
                prev["content_clean"] = prev["content"]
                prev["page_end"] = max(prev["page_end"], units[-1].page_number)
                enriched = self.metadata_builder.enrich_chunk(
                    base={
                        "chunk_index": prev["chunk_index"],
                        "page_start": prev["page_start"],
                        "page_end": prev["page_end"],
                        "content": prev["content"],
                        "content_clean": prev["content_clean"],
                    },
                    metadata=metadata,
                    structure=units[-1].structure,
                    classification=self.structure_detector.classify(prev["content"], metadata.get("subject")),
                )
                prev.update(enriched)
                return chunk_index
        base = {
            "chunk_index": chunk_index,
            "page_start": min(u.page_number for u in units),
            "page_end": max(u.page_number for u in units),
            "content": content,
            "content_clean": content,
        }
        enriched = self.metadata_builder.enrich_chunk(
            base=base,
            metadata=metadata,
            structure=units[-1].structure,
            classification=classification,
        )
        chunks.append(enriched)
        return chunk_index + 1
