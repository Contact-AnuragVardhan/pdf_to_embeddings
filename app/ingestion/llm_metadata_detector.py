from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import Settings
from ingestion.book_structure import BookChapter, BookStructure, enrich_chapter_page_ranges, structure_from_llm_json
from ingestion.pdf_extractor import ExtractedPage

logger = logging.getLogger(__name__)


class LLMMetadataDetector:
    """Detect document/book structure once per PDF using a model-agnostic OpenAI call.

    The detector is intentionally best-effort. If the model name changes from gpt-5.4
    to gpt-4o-mini or gpt-5.4-mini, the calling code remains the same. If a model
    rejects json_object or temperature, fallback calls are attempted automatically.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def detect(self, pages: list[ExtractedPage], metadata: dict[str, Any]) -> BookStructure:
        fallback = RuleBasedStructureDetector().detect(pages, metadata)
        if not self.settings.auto_metadata_enabled:
            fallback.detected_by = "rule_based_disabled_llm"
            fallback.chapters = enrich_chapter_page_ranges(fallback.chapters, pages)
            return fallback
        if not self.client:
            logger.warning("OPENAI_API_KEY is not set. Skipping LLM metadata detection and using rule-based fallback.")
            fallback.detected_by = "rule_based_no_openai_key"
            fallback.chapters = enrich_chapter_page_ranges(fallback.chapters, pages)
            return fallback

        prompt = self._build_prompt(pages, metadata, fallback)
        try:
            data = self._call_model(prompt)
            structure = structure_from_llm_json(
                data,
                detected_by=f"llm:{self.settings.openai_metadata_model}",
                fallback_metadata=metadata,
            )
            structure.chapters = enrich_chapter_page_ranges(structure.chapters, pages)
            # Merge fallback chapters if LLM found none.
            if not structure.chapters and fallback.chapters:
                structure.chapters = enrich_chapter_page_ranges(fallback.chapters, pages)
                structure.metadata["llm_warning"] = "LLM returned no usable chapters; used rule-based TOC fallback."
            return self._merge_with_path_metadata(structure, metadata)
        except Exception as exc:
            logger.exception("LLM metadata detection failed. Falling back to rule-based detector: %s", exc)
            fallback.detected_by = "rule_based_after_llm_failure"
            fallback.metadata["llm_error"] = str(exc)
            fallback.chapters = enrich_chapter_page_ranges(fallback.chapters, pages)
            return fallback

    def _merge_with_path_metadata(self, structure: BookStructure, metadata: dict[str, Any]) -> BookStructure:
        # Folder/file path metadata is treated as high-trust for school/class/declared subject.
        structure.subject = metadata.get("subject") or structure.subject
        structure.grade = metadata.get("grade") or metadata.get("class_name") or structure.grade
        structure.book_title = structure.book_title or metadata.get("book_title") or metadata.get("title")
        structure.primary_language = metadata.get("language") or structure.primary_language
        return structure

    def _build_prompt(self, pages: list[ExtractedPage], metadata: dict[str, Any], fallback: BookStructure) -> str:
        samples = self._sample_pages_for_prompt(pages)
        compact_pages = []
        for page in samples:
            text = " ".join((page.cleaned_text or "").split())
            compact_pages.append({
                "pdf_page_number": page.page_number,
                "detected_language": page.detected_language,
                "token_count": page.token_count,
                "text": text[:3500],
            })

        return json.dumps(
            {
                "task": "Detect textbook/book metadata, table-of-contents chapter structure, languages, and recommended chunking profile for a RAG embedding pipeline.",
                "instructions": [
                    "Return only valid JSON. Do not wrap in markdown.",
                    "Use null for unknown values.",
                    "chapter_number may be string or number.",
                    "printed_start_page is the page number printed in the book/table of contents.",
                    "pdf_start_page is the actual PDF page number if visible from the provided samples; if you cannot know it, return null.",
                    "For different publishers/languages, infer chapters from Contents/Index/Table of Contents/अनुक्रमणिका/विषय सूची/সূচিপত্র when present.",
                    "Prefer deterministic, citation-friendly structure over creative guesses.",
                    "Recommend chunk settings based on content type: math/question books should usually have smaller chunks; literature/story books larger chunks; grammar/rule books medium chunks.",
                ],
                "expected_json_schema": {
                    "book_title": "string|null",
                    "subject": "string|null",
                    "grade": "string|null",
                    "primary_language": "string|null",
                    "languages_detected": ["string"],
                    "publisher": "string|null",
                    "author": "string|null",
                    "isbn": "string|null",
                    "edition": "string|null",
                    "publication_year": "string|null",
                    "content_profile": "math_textbook|science_textbook|english_literature|hindi_literature|grammar|question_bank|mixed_textbook|unknown",
                    "recommended_chunking_strategy": "toc_structure_aware|semantic_then_recursive|question_block|paragraph_story|recursive_token",
                    "recommended_chunk_max_tokens": "integer between 350 and 1400",
                    "recommended_chunk_overlap_tokens": "integer between 40 and 220",
                    "confidence": "number 0 to 1",
                    "chapters": [
                        {
                            "chapter_number": "string|null",
                            "chapter_title": "string",
                            "printed_start_page": "integer|null",
                            "pdf_start_page": "integer|null",
                            "confidence": "number 0 to 1"
                        }
                    ]
                },
                "path_or_cli_metadata_high_trust": metadata,
                "rule_based_fallback_chapters": [c.to_dict() for c in fallback.chapters[:80]],
                "sample_pages": compact_pages,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _sample_pages_for_prompt(self, pages: list[ExtractedPage]) -> list[ExtractedPage]:
        if not pages:
            return []
        selected: dict[int, ExtractedPage] = {}
        # Front matter often contains title/copyright/contents.
        for p in pages[: min(len(pages), self.settings.metadata_sample_pages)]:
            selected[p.page_number] = p
        # Explicit contents pages, wherever they appear early in the book.
        toc_patterns = re.compile(r"\b(contents|table of contents|index)\b|अनुक्रमणिका|विषय\s*सूची|সূচিপত্র", re.I)
        for p in pages[: min(len(pages), max(self.settings.metadata_sample_pages, 40))]:
            if toc_patterns.search(p.cleaned_text or ""):
                selected[p.page_number] = p
                # include next couple of pages in case TOC spans pages
                for q in pages[p.page_number : min(len(pages), p.page_number + 3)]:
                    selected[q.page_number] = q
        # Add a few later pages so the LLM sees body page style.
        for idx in [20, 40, 60, 100]:
            if idx <= len(pages):
                selected[pages[idx - 1].page_number] = pages[idx - 1]
        return [selected[k] for k in sorted(selected)]

    def _call_model(self, prompt: str) -> dict[str, Any]:
        assert self.client is not None
        messages = [
            {"role": "system", "content": "You are a careful multilingual textbook structure extraction engine. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        model = self.settings.openai_metadata_model
        attempts = [
            {"response_format": {"type": "json_object"}, "temperature": 0, "max_completion_tokens": self.settings.metadata_max_output_tokens},
            {"response_format": {"type": "json_object"}, "max_completion_tokens": self.settings.metadata_max_output_tokens},
            {"max_completion_tokens": self.settings.metadata_max_output_tokens},
            {"max_tokens": self.settings.metadata_max_output_tokens},
        ]
        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                text = response.choices[0].message.content or "{}"
                return self._parse_json(text)
            except Exception as exc:
                last_exc = exc
                logger.warning("Metadata model call attempt failed for model %s with kwargs %s: %s", model, sorted(kwargs), exc)
        raise RuntimeError(f"All metadata model calls failed for {model}: {last_exc}")

    def _parse_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Metadata LLM response must be a JSON object.")
        return data


class RuleBasedStructureDetector:
    """Cheap fallback that extracts common TOC lines without using an LLM."""

    TOC_LINE_RE = re.compile(
        r"^\s*(?P<num>[0-9०-९]+|[IVXivx]+)?\s*[.)\-–:]?\s*(?P<title>[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F0-9 ,&()\-/]+?)\s+(?P<page>[0-9०-९]{1,4})\s*$"
    )

    def detect(self, pages: list[ExtractedPage], metadata: dict[str, Any]) -> BookStructure:
        chapters: list[BookChapter] = []
        for page in pages[:40]:
            text = page.cleaned_text or ""
            if not self._looks_like_contents_page(text):
                continue
            for line in text.splitlines():
                parsed = self._parse_toc_line(line)
                if not parsed:
                    continue
                chapters.append(parsed)
        # De-duplicate by title.
        seen: set[str] = set()
        unique: list[BookChapter] = []
        for index, chapter in enumerate(chapters, start=1):
            key = (chapter.chapter_title or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            chapter.chapter_number = chapter.chapter_number or str(index)
            chapter.detected_by = "rule_based_toc"
            chapter.confidence = chapter.confidence or 0.55
            unique.append(chapter)

        structure = BookStructure(
            book_title=metadata.get("book_title") or metadata.get("title"),
            subject=metadata.get("subject"),
            grade=metadata.get("grade") or metadata.get("class_name"),
            primary_language=metadata.get("language"),
            languages_detected=[],
            content_profile=None,
            recommended_chunking_strategy=None,
            detected_by="rule_based",
            confidence=0.45 if unique else 0.2,
            chapters=unique,
            metadata={"note": "Fallback rule-based metadata detector."},
        )
        return structure

    def _looks_like_contents_page(self, text: str) -> bool:
        lower = text.lower()
        return (
            "contents" in lower
            or "table of contents" in lower
            or "अनुक्रमणिका" in text
            or "विषय सूची" in text
            or sum(1 for line in text.splitlines() if self._parse_toc_line(line)) >= 5
        )

    def _parse_toc_line(self, line: str) -> BookChapter | None:
        clean = " ".join(line.strip().split())
        if len(clean) < 4 or len(clean) > 140:
            return None
        match = self.TOC_LINE_RE.match(clean)
        if not match:
            return None
        title = match.group("title").strip(" .:-–")
        if not title or title.lower() in {"contents", "answers"}:
            return None
        # Avoid lines that are mostly numbers/noise.
        alpha_count = len(re.findall(r"[A-Za-z\u0900-\u097F]", title))
        if alpha_count < 3:
            return None
        return BookChapter(
            chapter_number=match.group("num"),
            chapter_title=title,
            printed_start_page=_devanagari_int(match.group("page")),
            detected_by="rule_based_toc",
            confidence=0.55,
        )


def _devanagari_int(value: str | None) -> int | None:
    if not value:
        return None
    trans = str.maketrans("०१२३४५६७८९", "0123456789")
    try:
        return int(value.translate(trans))
    except Exception:
        return None
