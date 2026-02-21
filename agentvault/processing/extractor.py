"""Extractor — extracts structured facts and knowledge from text."""

from __future__ import annotations

import logging
import re
from typing import Any

from agentvault.core.types import MemoryType

logger = logging.getLogger(__name__)


class Extractor:
    """Extracts structured information from raw text.

    Uses pattern matching and heuristics to identify:
    - Facts and preferences (semantic memories)
    - Events and interactions (episodic memories)
    - Procedures and workflows (procedural memories)
    - Tags and importance indicators
    """

    # Patterns indicating preferences
    PREFERENCE_PATTERNS = [
        r"(?:I |the user |they )?(?:prefer|like|want|love|enjoy|choose)s?\s+(.+)",
        r"(?:I |the user |they )?(?:don't like|dislike|hate|avoid)s?\s+(.+)",
        r"(.+)\s+(?:is|are)\s+(?:better|preferred|favorite|best)",
    ]

    # Patterns indicating procedures/steps
    PROCEDURE_PATTERNS = [
        r"(?:step\s+\d+|first|then|next|finally|after that)[:\s]+(.+)",
        r"(?:to\s+\w+),?\s+(?:you\s+)?(?:need to|should|must|have to)\s+(.+)",
        r"(?:run|execute|type|enter|use)\s+[`\"](.+?)[`\"]",
    ]

    # Patterns indicating events
    EVENT_PATTERNS = [
        r"(?:on|at|during)\s+(?:[\w\s,]+\d{4}|\d{1,2}/\d{1,2})[,:\s]+(.+)",
        r"(?:yesterday|today|last\s+\w+|this\s+\w+)[,:\s]+(.+)",
        r"(?:I |the user |they )?(?:asked|requested|said|mentioned|told)\s+(.+)",
    ]

    # Words indicating high importance
    IMPORTANCE_KEYWORDS = {
        "critical": 0.9,
        "important": 0.8,
        "essential": 0.85,
        "must": 0.8,
        "always": 0.7,
        "never": 0.7,
        "key": 0.7,
        "crucial": 0.85,
        "significant": 0.7,
        "priority": 0.8,
    }

    def classify_memory_type(self, text: str) -> MemoryType:
        """Classify what type of memory a text represents.

        Args:
            text: The text to classify.

        Returns:
            The detected MemoryType.
        """
        text_lower = text.lower()

        for pattern in self.PROCEDURE_PATTERNS:
            if re.search(pattern, text_lower):
                return MemoryType.PROCEDURAL

        for pattern in self.EVENT_PATTERNS:
            if re.search(pattern, text_lower):
                return MemoryType.EPISODIC

        for pattern in self.PREFERENCE_PATTERNS:
            if re.search(pattern, text_lower):
                return MemoryType.SEMANTIC

        return MemoryType.SEMANTIC

    def estimate_importance(self, text: str) -> float:
        """Estimate the importance of a text based on content analysis.

        Args:
            text: The text to analyze.

        Returns:
            An importance score between 0 and 1.
        """
        text_lower = text.lower()
        max_importance = 0.5

        for keyword, importance in self.IMPORTANCE_KEYWORDS.items():
            if keyword in text_lower:
                max_importance = max(max_importance, importance)

        if len(text) > 200:
            max_importance = min(1.0, max_importance + 0.05)
        if any(c in text for c in ["!", "?", "***", "IMPORTANT", "NOTE"]):
            max_importance = min(1.0, max_importance + 0.1)

        return max_importance

    def extract_tags(self, text: str) -> list[str]:
        """Extract relevant tags from text content.

        Args:
            text: The text to extract tags from.

        Returns:
            A list of extracted tags.
        """
        tags: list[str] = []
        text_lower = text.lower()

        tech_keywords = [
            "python", "javascript", "typescript", "react", "vue", "angular",
            "docker", "kubernetes", "aws", "gcp", "azure", "api", "database",
            "sql", "nosql", "redis", "git", "ci/cd", "testing", "deploy",
            "frontend", "backend", "fullstack", "machine learning", "ai",
            "langchain", "crewai", "openai", "anthropic",
        ]
        for kw in tech_keywords:
            if kw in text_lower:
                tags.append(kw)

        action_keywords = [
            "refactor", "fix", "bug", "feature", "update", "deploy",
            "migrate", "optimize", "test", "review", "debug",
        ]
        for kw in action_keywords:
            if kw in text_lower:
                tags.append(kw)

        return list(set(tags))

    def extract_facts(self, text: str) -> list[dict[str, Any]]:
        """Extract structured facts from text.

        Attempts to parse subject-predicate-object triples from text.

        Args:
            text: The text to extract facts from.

        Returns:
            List of dicts with 'subject', 'predicate', 'object', 'content'.
        """
        facts: list[dict[str, Any]] = []

        spo_patterns = [
            r"(\w[\w\s]*?)\s+(prefers?|likes?|uses?|wants?|needs?|is|are|has|have)\s+(\w[\w\s]*?)(?:\.|$|,)",
            r"(\w[\w\s]*?)\s+(?:should|must|always|never)\s+(\w+)\s+(\w[\w\s]*?)(?:\.|$|,)",
        ]

        for pattern in spo_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    subject = groups[0].strip()
                    predicate = groups[1].strip()
                    obj = groups[2].strip()
                    if len(subject) > 1 and len(obj) > 1:
                        facts.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "content": match.group(0).strip(),
                        })

        return facts

    def extract_steps(self, text: str) -> list[str]:
        """Extract ordered steps from a text describing a procedure.

        Args:
            text: The text to extract steps from.

        Returns:
            List of step strings.
        """
        steps: list[str] = []

        numbered = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+)", text)
        if numbered:
            return [s.strip() for s in numbered]

        bulleted = re.findall(r"(?:^|\n)\s*[-*]\s+(.+)", text)
        if bulleted:
            return [s.strip() for s in bulleted]

        for pattern in self.PROCEDURE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            steps.extend(m.strip() for m in matches if m.strip())

        return steps

    def auto_extract(self, text: str) -> dict[str, Any]:
        """Automatically extract all available information from text.

        Returns a comprehensive extraction result including type,
        importance, tags, facts, and steps.

        Args:
            text: The text to analyze.

        Returns:
            Dict with all extracted information.
        """
        return {
            "memory_type": self.classify_memory_type(text),
            "importance": self.estimate_importance(text),
            "tags": self.extract_tags(text),
            "facts": self.extract_facts(text),
            "steps": self.extract_steps(text),
        }
