"""Tests for the Ranker — scoring, ranking, weight adjustment."""

import math
from datetime import datetime, timedelta

import pytest

from agentvault.core.types import Memory, MemoryType
from agentvault.processing.ranker import Ranker


def _mem(
    importance: float = 0.5,
    access_count: int = 0,
    created_hours_ago: float = 0,
) -> Memory:
    """Helper to create a Memory with controllable fields."""
    now = datetime.utcnow()
    m = Memory(
        agent_id="test",
        type=MemoryType.SEMANTIC,
        content="test",
        importance=importance,
        access_count=access_count,
    )
    m.created_at = now - timedelta(hours=created_hours_ago)
    m.last_accessed = now - timedelta(hours=created_hours_ago)
    return m


class TestRankerScore:
    """Tests for score()."""

    def test_score_bounds(self, ranker):
        """Score is always between 0 and 1."""
        m = _mem(importance=1.0, access_count=100)
        score = ranker.score(m, similarity=1.0)
        assert 0.0 <= score <= 1.0

    def test_score_all_zeros(self, ranker):
        """Score with all-zero inputs is 0."""
        m = _mem(importance=0.0, access_count=0, created_hours_ago=10000)
        score = ranker.score(m, similarity=0.0)
        assert score >= 0.0

    def test_higher_similarity_higher_score(self, ranker):
        """Higher similarity should increase score."""
        m = _mem()
        low = ranker.score(m, similarity=0.1)
        high = ranker.score(m, similarity=0.9)
        assert high > low

    def test_higher_importance_higher_score(self, ranker):
        """Higher importance should increase score."""
        low = ranker.score(_mem(importance=0.1), similarity=0.5)
        high = ranker.score(_mem(importance=0.9), similarity=0.5)
        assert high > low


class TestRecencyScore:
    """Tests for _recency_score()."""

    def test_recent_memory_high_score(self, ranker):
        """A memory accessed just now should score near 1.0."""
        m = _mem(created_hours_ago=0)
        now = datetime.utcnow()
        score = ranker._recency_score(m, now)
        assert score >= 0.99

    def test_old_memory_low_score(self, ranker):
        """A very old memory should score near 0."""
        m = _mem(created_hours_ago=500)
        now = datetime.utcnow()
        score = ranker._recency_score(m, now)
        assert score < 0.01

    def test_decay_formula(self, ranker):
        """Score follows exp(-decay * hours)."""
        hours = 10
        m = _mem(created_hours_ago=hours)
        now = datetime.utcnow()
        expected = math.exp(-ranker.recency_decay * hours)
        actual = ranker._recency_score(m, now)
        assert abs(actual - expected) < 0.01


class TestFrequencyScore:
    """Tests for _frequency_score()."""

    def test_zero_accesses(self, ranker):
        assert ranker._frequency_score(_mem(access_count=0)) == 0.0

    def test_positive_accesses(self, ranker):
        score = ranker._frequency_score(_mem(access_count=10))
        assert 0 < score <= 1.0

    def test_capped_at_one(self, ranker):
        score = ranker._frequency_score(_mem(access_count=1000))
        assert score <= 1.0

    def test_diminishing_returns(self, ranker):
        s10 = ranker._frequency_score(_mem(access_count=10))
        s20 = ranker._frequency_score(_mem(access_count=20))
        s30 = ranker._frequency_score(_mem(access_count=30))
        # Increments should shrink (log scale)
        assert (s20 - s10) > (s30 - s20)


class TestRank:
    """Tests for rank()."""

    def test_sorted_descending(self, ranker):
        m1 = _mem(importance=0.1)
        m2 = _mem(importance=0.9)
        ranked = ranker.rank([m1, m2], [0.5, 0.5], top_k=2)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self, ranker):
        memories = [_mem() for _ in range(5)]
        sims = [0.5] * 5
        ranked = ranker.rank(memories, sims, top_k=3)
        assert len(ranked) == 3

    def test_mismatched_lengths_raises(self, ranker):
        with pytest.raises(ValueError):
            ranker.rank([_mem()], [0.5, 0.6])


class TestAdjustWeights:
    """Tests for adjust_weights()."""

    def test_normalization(self, ranker):
        ranker.adjust_weights(similarity_weight=1.0, recency_weight=1.0)
        total = (
            ranker.similarity_weight
            + ranker.recency_weight
            + ranker.importance_weight
            + ranker.frequency_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_partial_adjustment(self, ranker):
        original_rec = ranker.recency_weight
        ranker.adjust_weights(similarity_weight=0.8)
        # Weights should still sum to 1.0
        total = (
            ranker.similarity_weight
            + ranker.recency_weight
            + ranker.importance_weight
            + ranker.frequency_weight
        )
        assert abs(total - 1.0) < 0.001
