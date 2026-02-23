"""Tests for the Extractor — type classification, fact/step extraction, importance."""

from agentvault.core.types import MemoryType
from agentvault.processing.extractor import Extractor


class TestClassifyMemoryType:
    """Tests for classify_memory_type()."""

    def test_procedural_step_pattern(self, extractor):
        assert extractor.classify_memory_type(
            "Step 1: install dependencies"
        ) == MemoryType.PROCEDURAL

    def test_procedural_to_do_pattern(self, extractor):
        assert extractor.classify_memory_type(
            "To deploy, you need to run npm build"
        ) == MemoryType.PROCEDURAL

    def test_procedural_run_command(self, extractor):
        assert extractor.classify_memory_type(
            'run `git push origin main`'
        ) == MemoryType.PROCEDURAL

    def test_episodic_yesterday(self, extractor):
        assert extractor.classify_memory_type(
            "Yesterday, the user asked to fix the login"
        ) == MemoryType.EPISODIC

    def test_episodic_user_asked(self, extractor):
        assert extractor.classify_memory_type(
            "The user asked to refactor the auth module"
        ) == MemoryType.EPISODIC

    def test_semantic_preference(self, extractor):
        assert extractor.classify_memory_type(
            "The user prefers dark mode"
        ) == MemoryType.SEMANTIC

    def test_semantic_default_fallback(self, extractor):
        assert extractor.classify_memory_type(
            "Python 3.11 supports tomllib"
        ) == MemoryType.SEMANTIC


class TestEstimateImportance:
    """Tests for estimate_importance()."""

    def test_baseline_importance(self, extractor):
        score = extractor.estimate_importance("some random text")
        assert score == 0.5

    def test_critical_keyword(self, extractor):
        score = extractor.estimate_importance("This is a critical security issue")
        assert score >= 0.9

    def test_important_keyword(self, extractor):
        score = extractor.estimate_importance("This is an important feature")
        assert score >= 0.8

    def test_long_text_bonus(self, extractor):
        short = extractor.estimate_importance("short")
        long_text = "a" * 250
        long_score = extractor.estimate_importance(long_text)
        assert long_score >= short

    def test_exclamation_bonus(self, extractor):
        base = extractor.estimate_importance("test note")
        with_bang = extractor.estimate_importance("test note!")
        assert with_bang > base


class TestExtractTags:
    """Tests for extract_tags()."""

    def test_tech_keywords(self, extractor):
        tags = extractor.extract_tags("Uses Python with Docker and Redis")
        assert "python" in tags
        assert "docker" in tags
        assert "redis" in tags

    def test_action_keywords(self, extractor):
        tags = extractor.extract_tags("Need to refactor and fix the bug")
        assert "refactor" in tags
        assert "fix" in tags
        assert "bug" in tags

    def test_no_tags(self, extractor):
        tags = extractor.extract_tags("Hello world")
        assert tags == []

    def test_deduplication(self, extractor):
        tags = extractor.extract_tags("deploy deploy deploy")
        assert tags.count("deploy") == 1


class TestExtractFacts:
    """Tests for extract_facts() — SPO triple extraction."""

    def test_preference_fact(self, extractor):
        facts = extractor.extract_facts("The user prefers TypeScript over JavaScript")
        assert len(facts) >= 1
        assert any(f["predicate"] == "prefers" for f in facts)

    def test_is_fact(self, extractor):
        facts = extractor.extract_facts("Python is a programming language")
        assert len(facts) >= 1

    def test_no_facts(self, extractor):
        facts = extractor.extract_facts("Hello")
        assert facts == []

    def test_multiple_facts(self, extractor):
        text = "The team uses React. The backend uses FastAPI."
        facts = extractor.extract_facts(text)
        assert len(facts) >= 2


class TestExtractSteps:
    """Tests for extract_steps()."""

    def test_numbered_steps(self, extractor):
        text = "1. Install deps\n2. Run build\n3. Deploy"
        steps = extractor.extract_steps(text)
        assert len(steps) == 3
        assert steps[0] == "Install deps"

    def test_bulleted_steps(self, extractor):
        text = "- Clone repo\n- Install packages\n- Start dev server"
        steps = extractor.extract_steps(text)
        assert len(steps) == 3

    def test_no_steps(self, extractor):
        steps = extractor.extract_steps("Just a regular sentence")
        assert len(steps) == 0


class TestAutoExtract:
    """Tests for auto_extract() — the full pipeline."""

    def test_returns_all_keys(self, extractor):
        result = extractor.auto_extract("The user prefers Python")
        assert "memory_type" in result
        assert "importance" in result
        assert "tags" in result
        assert "facts" in result
        assert "steps" in result

    def test_procedural_auto(self, extractor):
        result = extractor.auto_extract(
            "Step 1: install Python. Then create a venv. Finally run tests."
        )
        assert result["memory_type"] == MemoryType.PROCEDURAL
        assert len(result["steps"]) >= 1
