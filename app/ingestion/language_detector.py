from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageStats:
    language: str
    devanagari_chars: int
    latin_chars: int
    digits: int
    total_letters: int


class LanguageDetector:
    DEVANAGARI_START = "\u0900"
    DEVANAGARI_END = "\u097F"

    def detect_with_stats(self, text: str | None) -> LanguageStats:
        if not text or len(text.strip()) < 8:
            return LanguageStats("Unknown", 0, 0, 0, 0)
        devanagari = 0
        latin = 0
        digits = 0
        for ch in text:
            if self.DEVANAGARI_START <= ch <= self.DEVANAGARI_END:
                devanagari += 1
            elif ("a" <= ch.lower() <= "z"):
                latin += 1
            elif ch.isdigit():
                digits += 1
        total_letters = devanagari + latin
        if total_letters < 5:
            language = "Unknown"
        else:
            dev_ratio = devanagari / total_letters
            latin_ratio = latin / total_letters
            if devanagari >= 10 and latin >= 10 and dev_ratio >= 0.20 and latin_ratio >= 0.20:
                language = "Mixed"
            elif dev_ratio >= 0.55:
                language = "Hindi"
            elif latin_ratio >= 0.55:
                language = "English"
            else:
                language = "Mixed"
        return LanguageStats(language, devanagari, latin, digits, total_letters)

    def detect(self, text: str | None) -> str:
        return self.detect_with_stats(text).language

    def has_devanagari(self, text: str | None) -> bool:
        return self.detect_with_stats(text).devanagari_chars > 0

    def has_english(self, text: str | None) -> bool:
        return self.detect_with_stats(text).latin_chars > 0
