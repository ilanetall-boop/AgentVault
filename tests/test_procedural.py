"""Tests for procedural memory."""


from agentvault.core.procedural import ProceduralMemoryManager
from agentvault.core.types import MemoryType, ProceduralMemory


class TestProceduralMemoryManager:
    """Tests for the ProceduralMemoryManager."""

    def setup_method(self):
        self.manager = ProceduralMemoryManager()

    def test_create_memory(self):
        """Test creating a procedural memory."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Deploy to Vercel: build then deploy",
            name="deploy_vercel",
            steps=["npm run build", "vercel deploy --prod"],
            learned_from="conversation_123",
            importance=0.7,
            tags=["deploy", "vercel"],
        )
        assert isinstance(memory, ProceduralMemory)
        assert memory.type == MemoryType.PROCEDURAL
        assert memory.name == "deploy_vercel"
        assert len(memory.steps) == 2
        assert memory.steps[0] == "npm run build"
        assert memory.learned_from == "conversation_123"

    def test_record_success(self):
        """Test recording a successful execution."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Test procedure",
            importance=0.5,
        )
        original_importance = memory.importance
        memory = self.manager.record_success(memory)
        assert memory.success_count == 1
        assert memory.importance > original_importance

    def test_record_failure(self):
        """Test recording a failed execution."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Test procedure",
            importance=0.5,
        )
        memory = self.manager.record_failure(memory)
        assert memory.failure_count == 1

    def test_multiple_failures_decrease_importance(self):
        """Test that many failures decrease importance."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Bad procedure",
            importance=0.5,
        )
        for _ in range(5):
            memory = self.manager.record_failure(memory)
        assert memory.importance < 0.5

    def test_success_rate(self):
        """Test success rate calculation."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Test procedure",
        )
        memory.success_count = 8
        memory.failure_count = 2
        rate = self.manager.get_success_rate(memory)
        assert rate == 0.8

    def test_success_rate_zero_executions(self):
        """Test success rate with no executions."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Never executed",
        )
        assert self.manager.get_success_rate(memory) == 0.0

    def test_merge_procedures(self):
        """Test merging two similar procedures."""
        proc_a = ProceduralMemory(
            agent_id="agent-1",
            content="Deploy to prod",
            name="deploy",
            steps=["build", "deploy"],
            tags=["deploy"],
            success_count=3,
            failure_count=1,
            access_count=5,
        )
        proc_b = ProceduralMemory(
            agent_id="agent-1",
            content="Deploy to production with tests",
            name="deploy",
            steps=["test", "build", "deploy", "verify"],
            tags=["production"],
            success_count=2,
            failure_count=0,
            access_count=3,
        )
        merged = self.manager.merge_procedures(proc_a, proc_b)
        assert merged.id == proc_a.id
        assert len(merged.steps) == 4  # kept longer list
        assert merged.success_count == 5  # combined
        assert merged.failure_count == 1
        assert merged.access_count == 8
        assert "deploy" in merged.tags
        assert "production" in merged.tags

    def test_extract_procedure_info(self):
        """Test extracting procedure info."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Git workflow",
            name="git_push",
            steps=["git add .", "git commit -m 'msg'", "git push"],
        )
        memory.success_count = 10
        memory.failure_count = 1
        info = self.manager.extract_procedure_info(memory)
        assert info["name"] == "git_push"
        assert len(info["steps"]) == 3
        assert info["success_rate"] > 0.9
