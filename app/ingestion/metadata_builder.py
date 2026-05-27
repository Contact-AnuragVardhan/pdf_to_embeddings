from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ingestion.language_detector import LanguageDetector
from ingestion.structure_detector import ContentClassification, StructureState
from utils.hashing import sha256_text
from utils.token_counter import TokenCounter


HINDI_IMPORTANT_TERMS = {
    "संज्ञा", "सर्वनाम", "क्रिया", "विशेषण", "संधि", "समास", "वचन", "लिंग", "काल",
    "भिन्न", "दशमलव", "कोण", "रेखा", "त्रिभुज", "वर्ग", "क्षेत्रफल", "परिमाप", "परिभाषा",
    "नियम", "उदाहरण", "अभ्यास", "प्रश्न", "कहानी", "कविता", "शब्दार्थ",
}
ENGLISH_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were", "have", "has",
    "into", "your", "you", "but", "not", "can", "will", "shall", "what", "when", "where", "which",
    "chapter", "unit", "lesson", "section", "page", "book",
}
HINDI_STOPWORDS = {"और", "यह", "वह", "है", "हैं", "था", "थे", "को", "का", "की", "के", "में", "से", "पर", "एक", "लिए"}


class MetadataBuilder:
    FORMULA_RE = re.compile(r"[A-Za-zअ-ह०-९0-9π√()/%+\-×÷\s]+\s*[=<>≤≥]\s*[A-Za-zअ-ह०-९0-9π√()/%+\-×÷\s]+|\b\d+\s*/\s*\d+\b")
    NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b|[०-९]+")
    WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}|[\u0900-\u097F]{2,}")

    def __init__(self, token_counter: TokenCounter, language_detector: LanguageDetector) -> None:
        self.token_counter = token_counter
        self.language_detector = language_detector

    def build_source_label(self, book_title: str, structure: StructureState, page_start: int, page_end: int) -> str:
        section = structure.chapter_title or structure.section_title or structure.topic
        pages = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        if section:
            return f"{book_title}, {section}, {pages}"
        return f"{book_title}, {pages}"

    def build_citation_text(self, book_title: str, page_start: int, page_end: int) -> str:
        pages = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        return f"Source: {book_title}, {pages}"

    def build_content_for_embedding(
        self,
        *,
        book_title: str,
        school_name: str | None,
        class_name: str | None,
        subject: str | None,
        grade: str | None,
        board: str | None,
        language: str | None,
        structure: StructureState,
        chunk_type: str,
        page_start: int,
        page_end: int,
        content_clean: str,
    ) -> str:
        return "\n".join(
            [
                f"School: {school_name or ''}",
                f"Class: {class_name or grade or ''}",
                f"Book: {book_title or ''}",
                f"Subject: {subject or ''}",
                f"Grade: {grade or class_name or ''}",
                f"Board: {board or ''}",
                f"Language: {language or ''}",
                f"Chapter: {structure.chapter_title or ''}",
                f"Section: {structure.section_title or ''}",
                f"Topic: {structure.topic or ''}",
                f"Chunk Type: {chunk_type}",
                f"Pages: {page_start}-{page_end}",
                "Text:",
                content_clean,
            ]
        ).strip()

    def enrich_chunk(
        self,
        *,
        base: dict[str, Any],
        metadata: dict[str, Any],
        structure: StructureState,
        classification: ContentClassification,
    ) -> dict[str, Any]:
        text = base["content_clean"]
        detected = self.language_detector.detect_with_stats(text)
        keywords = self.extract_keywords(text)
        formulas = self.extract_formulas(text)
        numbers = self.extract_numbers(text)
        question_types = self.extract_question_types(text)
        book_title = metadata.get("book_title") or metadata.get("title") or base.get("book_title") or "Untitled Book"
        content_for_embedding = self.build_content_for_embedding(
            book_title=book_title,
            school_name=metadata.get("school_name"),
            class_name=metadata.get("class_name"),
            subject=metadata.get("subject"),
            grade=metadata.get("grade"),
            board=metadata.get("board"),
            language=metadata.get("language") or detected.language,
            structure=structure,
            chunk_type=classification.chunk_type,
            page_start=base["page_start"],
            page_end=base["page_end"],
            content_clean=text,
        )
        flags = classification.flags
        chunk = {
            **base,
            "book_title": book_title,
            "school_name": metadata.get("school_name"),
            "class_name": metadata.get("class_name"),
            "subject": metadata.get("subject"),
            "grade": metadata.get("grade"),
            "board": metadata.get("board"),
            "medium": metadata.get("medium"),
            "language": metadata.get("language") or detected.language,
            "detected_language": detected.language,
            **structure.as_dict(),
            "chunk_type": classification.chunk_type,
            "content_domain": classification.content_domain,
            "difficulty_level": classification.difficulty_level,
            "pedagogical_role": classification.pedagogical_role,
            "content_for_embedding": content_for_embedding,
            "summary": self.simple_summary(text),
            "keywords": keywords,
            "important_terms": self.extract_important_terms(text, keywords),
            "formulas": formulas,
            "numbers": numbers,
            "question_types": question_types,
            "word_count": len(text.split()),
            "token_count": self.token_counter.count(text),
            "char_count": len(text),
            "has_formula": bool(formulas) or flags.get("has_formula", False),
            "has_numbers": bool(numbers) or flags.get("has_numbers", False),
            "has_questions": bool(question_types) or flags.get("has_questions", False),
            "has_exercises": flags.get("has_exercises", False),
            "has_examples": flags.get("has_examples", False),
            "has_definition": flags.get("has_definition", False),
            "has_table_like_text": flags.get("has_table_like_text", False),
            "has_devanagari": detected.devanagari_chars > 0,
            "has_english": detected.latin_chars > 0,
            "source_label": self.build_source_label(book_title, structure, base["page_start"], base["page_end"]),
            "citation_text": self.build_citation_text(book_title, base["page_start"], base["page_end"]),
            "metadata": {
                "embedding_input_hash": sha256_text(content_for_embedding),
                "school_name": metadata.get("school_name"),
                "class_name": metadata.get("class_name"),
                "book_title": book_title,
                "source_pages": list(range(base["page_start"], base["page_end"] + 1)),
            },
        }
        return chunk

    def extract_formulas(self, text: str) -> list[str]:
        return sorted({m.group(0).strip() for m in self.FORMULA_RE.finditer(text) if len(m.group(0).strip()) <= 160})[:25]

    def extract_numbers(self, text: str) -> list[str]:
        return sorted(set(self.NUMBER_RE.findall(text)))[:50]

    def extract_question_types(self, text: str) -> list[str]:
        found: list[str] = []
        checks = [
            (r"\?", "question"),
            (r"multiple choice|choose|tick|सही विकल्प", "multiple_choice"),
            (r"fill in|रिक्त स्थान", "fill_in_blank"),
            (r"true or false|सही या गलत", "true_false"),
            (r"short answer|लघु उत्तर", "short_answer"),
            (r"solve|हल करें", "solve"),
        ]
        for pattern, label in checks:
            if re.search(pattern, text, re.I):
                found.append(label)
        return found

    def extract_keywords(self, text: str, max_keywords: int = 20) -> list[str]:
        words = [w.strip("'’-") for w in self.WORD_RE.findall(text)]
        normalized = []
        for word in words:
            lower = word.lower()
            if lower in ENGLISH_STOPWORDS or word in HINDI_STOPWORDS:
                continue
            if len(word) < 3:
                continue
            normalized.append(word)
        counts = Counter(normalized)
        for word in list(counts):
            if word.istitle() or word in HINDI_IMPORTANT_TERMS:
                counts[word] += 2
        return [word for word, _ in counts.most_common(max_keywords)]

    def extract_important_terms(self, text: str, keywords: list[str]) -> list[str]:
        terms = {term for term in HINDI_IMPORTANT_TERMS if term in text}
        terms.update(word for word in keywords if word[:1].isupper())
        return sorted(terms)[:25]

    def simple_summary(self, text: str, max_chars: int = 280) -> str:
        clean = " ".join(text.split())
        return clean[:max_chars].rstrip() + ("..." if len(clean) > max_chars else "")
