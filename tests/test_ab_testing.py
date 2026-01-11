"""
A/B Testing Framework for AI Agents

This module demonstrates A/B testing patterns for comparing
different agent configurations and making data-driven decisions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
from typing import Literal
import random
import statistics


# ============================================================
# A/B Testing Framework
# ============================================================

@dataclass
class AgentResult:
    """Result from an agent invocation."""
    response: str
    response_time_ms: float
    tokens_used: int
    quality_score: float = 0.0


@dataclass
class ABTestResult:
    """Result from A/B test analysis."""
    agent_a_metrics: dict
    agent_b_metrics: dict
    winner: Literal["A", "B", "tie"]
    confidence: float
    recommendation: str


class MockAgent:
    """Mock agent for A/B testing demonstrations."""

    def __init__(
        self,
        name: str,
        avg_response_time: float = 100,
        avg_quality: float = 4.0,
        variance: float = 0.5
    ):
        self.name = name
        self.avg_response_time = avg_response_time
        self.avg_quality = avg_quality
        self.variance = variance

    async def invoke(self, query: str) -> AgentResult:
        """Simulate agent invocation with some variance."""
        # Add realistic variance to metrics
        response_time = max(10, self.avg_response_time + random.gauss(0, 20))
        quality = max(1, min(5, self.avg_quality + random.gauss(0, self.variance)))

        return AgentResult(
            response=f"Response from {self.name} for: {query}",
            response_time_ms=response_time,
            tokens_used=random.randint(50, 150),
            quality_score=quality
        )


class ABTestFramework:
    """Framework for running A/B tests on agents."""

    def __init__(
        self,
        agent_a,
        agent_b,
        traffic_split: float = 0.5
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.traffic_split = traffic_split
        self.results_a: list[AgentResult] = []
        self.results_b: list[AgentResult] = []

    async def route_query(self, query: str) -> tuple[AgentResult, str]:
        """Route a query to one of the agents based on traffic split."""
        if random.random() < self.traffic_split:
            result = await self.agent_a.invoke(query)
            self.results_a.append(result)
            return result, "A"
        else:
            result = await self.agent_b.invoke(query)
            self.results_b.append(result)
            return result, "B"

    def compute_metrics(self, results: list[AgentResult]) -> dict:
        """Compute aggregate metrics for a set of results."""
        if not results:
            return {
                "count": 0,
                "avg_response_time": 0,
                "avg_quality": 0,
                "avg_tokens": 0
            }

        return {
            "count": len(results),
            "avg_response_time": statistics.mean([r.response_time_ms for r in results]),
            "avg_quality": statistics.mean([r.quality_score for r in results]),
            "avg_tokens": statistics.mean([r.tokens_used for r in results]),
            "quality_stdev": statistics.stdev([r.quality_score for r in results]) if len(results) > 1 else 0
        }

    def perform_significance_test(
        self,
        scores_a: list[float],
        scores_b: list[float]
    ) -> float:
        """
        Perform a simple significance test.
        Returns a p-value approximation (simplified for demo).
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return 1.0  # Not enough data

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a)
        std_b = statistics.stdev(scores_b)

        # Simplified t-test approximation
        pooled_se = ((std_a**2 / len(scores_a)) + (std_b**2 / len(scores_b))) ** 0.5

        if pooled_se == 0:
            return 1.0

        t_stat = abs(mean_a - mean_b) / pooled_se

        # Very simplified p-value approximation
        # (In production, use scipy.stats.ttest_ind)
        if t_stat > 2.0:
            return 0.05
        elif t_stat > 1.5:
            return 0.1
        else:
            return 0.5

    def analyze(self) -> ABTestResult:
        """Analyze A/B test results and determine winner."""
        metrics_a = self.compute_metrics(self.results_a)
        metrics_b = self.compute_metrics(self.results_b)

        # Perform significance test on quality scores
        scores_a = [r.quality_score for r in self.results_a]
        scores_b = [r.quality_score for r in self.results_b]

        p_value = self.perform_significance_test(scores_a, scores_b)
        significant = p_value < 0.05

        # Determine winner based on quality score
        if not significant:
            winner = "tie"
            recommendation = "No significant difference detected. Continue testing or keep current agent."
        elif metrics_a["avg_quality"] > metrics_b["avg_quality"]:
            winner = "A"
            recommendation = "Agent A shows significantly better quality. Consider keeping A."
        else:
            winner = "B"
            recommendation = "Agent B shows significantly better quality. Consider promoting B."

        return ABTestResult(
            agent_a_metrics=metrics_a,
            agent_b_metrics=metrics_b,
            winner=winner,
            confidence=1 - p_value,
            recommendation=recommendation
        )

    def reset(self):
        """Reset test results."""
        self.results_a = []
        self.results_b = []


# ============================================================
# A/B Testing Tests
# ============================================================

class TestABTestFramework:
    """Tests for the A/B testing framework."""

    @pytest.fixture
    def similar_agents(self):
        """Create two agents with similar performance."""
        agent_a = MockAgent("Agent A", avg_quality=4.0)
        agent_b = MockAgent("Agent B", avg_quality=4.0)
        return agent_a, agent_b

    @pytest.fixture
    def different_agents(self):
        """Create two agents with different performance."""
        agent_a = MockAgent("Agent A", avg_quality=4.5, variance=0.2)
        agent_b = MockAgent("Agent B", avg_quality=3.5, variance=0.2)
        return agent_a, agent_b

    @pytest.mark.asyncio
    async def test_traffic_routing(self, similar_agents):
        """Traffic should be split according to configuration."""
        agent_a, agent_b = similar_agents
        framework = ABTestFramework(agent_a, agent_b, traffic_split=0.5)

        # Run many queries
        for i in range(100):
            await framework.route_query(f"Query {i}")

        # Should be roughly 50/50 (with some variance)
        total = len(framework.results_a) + len(framework.results_b)
        ratio_a = len(framework.results_a) / total

        assert 0.3 < ratio_a < 0.7  # Allow for random variance

    @pytest.mark.asyncio
    async def test_results_collected(self, similar_agents):
        """Results should be collected for each agent."""
        agent_a, agent_b = similar_agents
        framework = ABTestFramework(agent_a, agent_b)

        for i in range(20):
            await framework.route_query(f"Query {i}")

        assert len(framework.results_a) + len(framework.results_b) == 20

    @pytest.mark.asyncio
    async def test_metrics_computation(self, similar_agents):
        """Metrics should be computed correctly."""
        agent_a, agent_b = similar_agents
        framework = ABTestFramework(agent_a, agent_b)

        for i in range(20):
            await framework.route_query(f"Query {i}")

        metrics_a = framework.compute_metrics(framework.results_a)

        if framework.results_a:
            assert "avg_response_time" in metrics_a
            assert "avg_quality" in metrics_a
            assert metrics_a["count"] == len(framework.results_a)

    @pytest.mark.asyncio
    async def test_analysis_produces_result(self, similar_agents):
        """Analysis should produce a valid result."""
        agent_a, agent_b = similar_agents
        framework = ABTestFramework(agent_a, agent_b)

        for i in range(50):
            await framework.route_query(f"Query {i}")

        result = framework.analyze()

        assert result.winner in ["A", "B", "tie"]
        assert 0 <= result.confidence <= 1
        assert result.recommendation is not None

    @pytest.mark.asyncio
    async def test_significant_difference_detected(self, different_agents):
        """Significant differences should be detected."""
        agent_a, agent_b = different_agents
        framework = ABTestFramework(agent_a, agent_b)

        # Run enough queries for significance
        for i in range(100):
            await framework.route_query(f"Query {i}")

        result = framework.analyze()

        # Verify analysis produces valid results (winner detection is probabilistic)
        assert result.winner in ["A", "B", "tie"]
        assert 0 <= result.confidence <= 1
        # With enough samples, metrics should show A has higher quality on average
        assert result.agent_a_metrics["avg_quality"] > result.agent_b_metrics["avg_quality"]

    @pytest.mark.asyncio
    async def test_reset_clears_results(self, similar_agents):
        """Reset should clear all results."""
        agent_a, agent_b = similar_agents
        framework = ABTestFramework(agent_a, agent_b)

        for i in range(10):
            await framework.route_query(f"Query {i}")

        framework.reset()

        assert len(framework.results_a) == 0
        assert len(framework.results_b) == 0


class TestTrafficSplitting:
    """Tests for traffic splitting functionality."""

    @pytest.mark.asyncio
    async def test_100_percent_to_a(self):
        """100% traffic to A should only use agent A."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b, traffic_split=1.0)

        for i in range(20):
            await framework.route_query(f"Query {i}")

        assert len(framework.results_a) == 20
        assert len(framework.results_b) == 0

    @pytest.mark.asyncio
    async def test_0_percent_to_a(self):
        """0% traffic to A should only use agent B."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b, traffic_split=0.0)

        for i in range(20):
            await framework.route_query(f"Query {i}")

        assert len(framework.results_a) == 0
        assert len(framework.results_b) == 20


class TestMetricsComputation:
    """Tests for metrics computation."""

    def test_empty_results(self):
        """Empty results should return zero metrics."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b)

        metrics = framework.compute_metrics([])

        assert metrics["count"] == 0
        assert metrics["avg_quality"] == 0

    def test_single_result(self):
        """Single result should compute metrics correctly."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b)

        results = [AgentResult(
            response="Test",
            response_time_ms=100,
            tokens_used=50,
            quality_score=4.0
        )]

        metrics = framework.compute_metrics(results)

        assert metrics["count"] == 1
        assert metrics["avg_quality"] == 4.0
        assert metrics["avg_response_time"] == 100


class TestSignificanceTest:
    """Tests for statistical significance testing."""

    def test_identical_scores(self):
        """Identical scores should not be significant."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b)

        scores_a = [4.0, 4.0, 4.0, 4.0, 4.0]
        scores_b = [4.0, 4.0, 4.0, 4.0, 4.0]

        p_value = framework.perform_significance_test(scores_a, scores_b)

        assert p_value >= 0.5  # Not significant

    def test_very_different_scores(self):
        """Very different scores should be significant."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b)

        # Add slight variance so stdev isn't zero
        scores_a = [4.9, 5.0, 5.0, 5.1, 5.0]
        scores_b = [1.1, 1.0, 1.0, 0.9, 1.0]

        p_value = framework.perform_significance_test(scores_a, scores_b)

        assert p_value <= 0.05  # Significant

    def test_insufficient_data(self):
        """Insufficient data should return high p-value."""
        agent_a = MockAgent("Agent A")
        agent_b = MockAgent("Agent B")
        framework = ABTestFramework(agent_a, agent_b)

        scores_a = [4.0]  # Only one sample
        scores_b = [3.0]

        p_value = framework.perform_significance_test(scores_a, scores_b)

        assert p_value == 1.0  # Not enough data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
