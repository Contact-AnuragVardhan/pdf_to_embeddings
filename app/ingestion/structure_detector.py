from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class StructureState:
    chapter_number: str | None = None
    chapter_title: str | None = None
    unit_title: str | None = None
    lesson_title: str | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    topic: str | None = None
    subtopic: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class HeadingDetection:
    level: str | None
    title: str | None
    number: str | None = None


@dataclass(frozen=True)
class ContentClassification:
    chunk_type: str
    pedagogical_role: str
    content_domain: str | None = None
    difficulty_level: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)


class StructureDetector:
    HINDI_HEADINGS = [
        "अध्याय", "पाठ", "इकाई", "विषय", "भाग", "अभ्यास", "उदाहरण", "प्रश्नावली",
        "गतिविधि", "व्याकरण", "कविता", "कहानी", "परिभाषा", "नियम", "सारांश",
    ]
    ENGLISH_HEADINGS = [
        "Chapter", "Unit", "Lesson", "Section", "Exercise", "Activity", "Example", "Practice",
        "Summary", "Review", "Poem", "Story", "Grammar", "Vocabulary", "Definition", "Formula",
    ]

    CHAPTER_RE = re.compile(r"^\s*(Chapter|अध्याय)\s*([0-9०-९IVXivx.-]+)?\s*[:\-–.]?\s*(.*)$", re.I)
    UNIT_RE = re.compile(r"^\s*(Unit|इकाई)\s*([0-9०-९IVXivx.-]+)?\s*[:\-–.]?\s*(.*)$", re.I)
    LESSON_RE = re.compile(r"^\s*(Lesson|पाठ)\s*([0-9०-९IVXivx.-]+)?\s*[:\-–.]?\s*(.*)$", re.I)
    SECTION_RE = re.compile(r"^\s*(Section|Exercise|Activity|Practice|Review|अभ्यास|प्रश्नावली|गतिविधि|विषय|भाग)\s*([0-9०-९IVXivx.-]+)?\s*[:\-–.]?\s*(.*)$", re.I)

    FORMULA_RE = re.compile(r"([A-Za-zअ-ह०-९0-9π√]+\s*[=+×÷≤≥<>]\s*[A-Za-zअ-ह०-९0-9π√()/%+\-×÷\s]+)|\b\d+\s*/\s*\d+\b")
    QUESTION_RE = re.compile(r"(\?|प्रश्न|उत्तर दें|Choose|Tick|Fill in|निम्नलिखित|हल करें|Solve)", re.I)
    TABLE_RE = re.compile(r"(\|.+\|)|(\S+\s{2,}\S+\s{2,}\S+)")

    def detect_heading(self, paragraph: str) -> HeadingDetection:
        line = " ".join(paragraph.strip().split())
        if not line or len(line) > 160:
            return HeadingDetection(None, None)
        for regex, level in [
            (self.CHAPTER_RE, "chapter_title"),
            (self.UNIT_RE, "unit_title"),
            (self.LESSON_RE, "lesson_title"),
            (self.SECTION_RE, "section_title"),
        ]:
            match = regex.match(line)
            if match:
                number = match.group(2) or None
                title = (match.group(3) or line).strip() or line
                return HeadingDetection(level, title, number)
        if any(line.lower().startswith(h.lower()) for h in self.ENGLISH_HEADINGS) or any(line.startswith(h) for h in self.HINDI_HEADINGS):
            return HeadingDetection("section_title", line)
        words = line.split()
        if 2 <= len(words) <= 10 and not self.QUESTION_RE.search(line) and not line.endswith((".", "।")):
            if line.istitle() or re.search(r"[\u0900-\u097F]", line):
                return HeadingDetection("topic", line)
        return HeadingDetection(None, None)

    def update_state(self, state: StructureState, heading: HeadingDetection) -> StructureState:
        if not heading.level or not heading.title:
            return state
        data = state.as_dict()
        data[heading.level] = heading.title
        if heading.level == "chapter_title":
            data["chapter_number"] = heading.number
            data["section_title"] = None
            data["subsection_title"] = None
            data["topic"] = None
            data["subtopic"] = None
        elif heading.level in {"unit_title", "lesson_title"}:
            data["section_title"] = None
            data["topic"] = None
        elif heading.level == "section_title":
            data["topic"] = heading.title
        elif heading.level == "topic":
            data["topic"] = heading.title
        return StructureState(**data)

    def classify(self, text: str, subject: str | None = None) -> ContentClassification:
        lower = text.lower()
        flags = {
            "has_formula": bool(self.FORMULA_RE.search(text)),
            "has_numbers": bool(re.search(r"\d|[०-९]", text)),
            "has_questions": bool(self.QUESTION_RE.search(text)),
            "has_exercises": bool(re.search(r"\bexercise\b|अभ्यास|प्रश्नावली|practice", text, re.I)),
            "has_examples": bool(re.search(r"\bexample\b|उदाहरण", text, re.I)),
            "has_definition": bool(re.search(r"\bdefinition\b|is called|refers to|परिभाषा|कहलाता|कहलाती|कहते हैं", text, re.I)),
            "has_table_like_text": bool(self.TABLE_RE.search(text)),
        }
        if "poem" in lower or "कविता" in text:
            return ContentClassification("poem", "reading_passage", "language", flags=flags)
        if "story" in lower or "कहानी" in text:
            return ContentClassification("story", "reading_passage", "language", flags=flags)
        if re.search(r"vocabulary|शब्दार्थ|शब्दावली", text, re.I):
            return ContentClassification("vocabulary", "reference", "language", flags=flags)
        if re.search(r"grammar|व्याकरण|संज्ञा|सर्वनाम|क्रिया|विशेषण|संधि|समास", text, re.I):
            if flags["has_examples"]:
                return ContentClassification("grammar_rule", "concept_teaching", "language", flags=flags)
            return ContentClassification("grammar_rule", "reference", "language", flags=flags)
        if flags["has_exercises"] and flags["has_questions"]:
            return ContentClassification("question_set", "assessment", self._domain(subject), flags=flags)
        if flags["has_questions"]:
            return ContentClassification("question_set", "assessment", self._domain(subject), flags=flags)
        if flags["has_examples"] and flags["has_formula"]:
            return ContentClassification("worked_example", "concept_teaching", "math", flags=flags)
        if flags["has_examples"]:
            return ContentClassification("example", "concept_teaching", self._domain(subject), flags=flags)
        if flags["has_definition"]:
            return ContentClassification("definition", "concept_teaching", self._domain(subject), flags=flags)
        if flags["has_formula"]:
            return ContentClassification("formula", "reference", "math", flags=flags)
        if flags["has_table_like_text"]:
            return ContentClassification("table", "reference", self._domain(subject), flags=flags)
        if re.search(r"activity|गतिविधि", text, re.I):
            return ContentClassification("activity", "activity", self._domain(subject), flags=flags)
        if re.search(r"summary|सारांश", text, re.I):
            return ContentClassification("summary", "reference", self._domain(subject), flags=flags)
        return ContentClassification("explanation", "concept_teaching", self._domain(subject), flags=flags)

    def _domain(self, subject: str | None) -> str | None:
        if not subject:
            return None
        s = subject.lower()
        if any(x in s for x in ["math", "गणित"]):
            return "math"
        if any(x in s for x in ["hindi", "english", "grammar", "language"]):
            return "language"
        if "science" in s or "evs" in s:
            return "science"
        return subject
