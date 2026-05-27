from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class CleanResult:
    cleaned_text: str
    notes: list[str]


class TextCleaner:
    CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    SPACE_RE = re.compile(r"[ \t\u00A0]+");
    BLANK_LINE_RE = re.compile(r"\n\s*\n\s*\n+")
    QUESTION_START_RE = re.compile(r"^\s*(\(?\d+\)?[.)]|[A-Da-d][.)]|[कखगघ][.)]|प्रश्न\s*\d*)")
    FORMULA_LINE_RE = re.compile(r"[=+×÷≤≥<>√π%]|\d+\s*/\s*\d+")

    def clean(self, text: str | None) -> CleanResult:
        notes: list[str] = []
        if not text:
            return CleanResult("", ["empty_text"])
        original_len = len(text)
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self.CONTROL_RE.sub("", text)
        text = self.SPACE_RE.sub(" ", text)
        text = self._fix_hyphenated_line_breaks(text)
        text = self._join_safe_broken_lines(text)
        text = self.BLANK_LINE_RE.sub("\n\n", text)
        text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if len(text) != original_len:
            notes.append("normalized_spacing_and_line_breaks")
        return CleanResult(text, notes)

    def _fix_hyphenated_line_breaks(self, text: str) -> str:
        return re.sub(r"([A-Za-z])[-‐]\n([A-Za-z])", r"\1\2", text)

    def _join_safe_broken_lines(self, text: str) -> str:
        lines = text.split("\n")
        if not lines:
            return text
        short_lines = sum(1 for line in lines if 0 < len(line.strip()) <= 45)
        preserve_short_lines = short_lines / max(len(lines), 1) > 0.55
        if preserve_short_lines:
            return text

        out: list[str] = []
        for line in lines:
            current = line.strip()
            if not current:
                out.append("")
                continue
            if not out or not out[-1].strip():
                out.append(current)
                continue
            prev = out[-1]
            if self._should_join(prev, current):
                out[-1] = prev.rstrip() + " " + current
            else:
                out.append(current)
        return "\n".join(out)

    def _should_join(self, prev: str, current: str) -> bool:
        if self.QUESTION_START_RE.match(current):
            return False
        if self.FORMULA_LINE_RE.search(prev) or self.FORMULA_LINE_RE.search(current):
            return False
        if prev.endswith((".", "?", "!", "।", "॥", ":", ";")):
            return False
        if len(prev) < 25 or len(current) < 20:
            return False
        if current[:1].isupper() and not prev.endswith(","):
            return False
        return True
