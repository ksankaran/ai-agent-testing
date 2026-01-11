"""
Evaluation Tests for AI Agents

This module demonstrates evaluation patterns including
LLM-as-judge, metrics collection, and quality assessment.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
from typing import Optional
import json
import re


# ============================================================
# Evaluation Framework
# ============================================================

@dataclass
class EvaluationResult:
    """Result from evaluating an agent response."""
    query: str
    response: str
    scores: dict = field(default_factory=dict)
    feedback: str = ""
    passed: bool = True


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating responses."""
    name: str
    description: str
    weight: float = 1.0
    min_passing_score: float = 3.0


class MockJudgeLLM:
    """Mock LLM for evaluation (simulates LLM-as-judge)."""

    def __init__(self, scores: dict = None):
        self.default_scores = scores or {
            "relevance": 4,
            "accuracy": 4,
            "helpfulness": 4,
            "clarity": 4
        }

    async def evaluate(self, query: str, response: str, criteria: list[str]) -> dict:
        """Evaluate a response against criteria."""
        scores = {}
        feedback = []

        for criterion in criteria:
            # Simulate evaluation logic
            score = self._calculate_score(query, response, criterion)
            scores[criterion] = score

            if score < 3:
                feedback.append(f"{criterion}: needs improvement")
            elif score >= 4:
                feedback.append(f"{criterion}: good")

        return {
            "scores": scores,
            "feedback": "; ".join(feedback),
            "average_score": sum(scores.values()) / len(scores) if scores else 0
        }

    def _calculate_score(self, query: str, response: str, criterion: str) -> int:
        """Calculate score based on simple heuristics (mock)."""
        # In reality, this would call an LLM
        base_score = self.default_scores.get(criterion, 3)

        # Simple heuristics for testing
        if criterion == "relevance":
            # Check if response relates to query
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words & response_words)
            if overlap > 2:
                base_score = min(5, base_score + 1)

        if criterion == "clarity":
            # Check response length and structure
            if 20 < len(response) < 500:
                base_score = min(5, base_score + 1)

        if criterion == "helpfulness":
            # Check for helpful indicators
            helpful_words = ["help", "here", "you can", "try", "suggest"]
            if any(word in response.lower() for word in helpful_words):
                base_score = min(5, base_score + 1)

        return base_score


class Evaluator:
    """Evaluator for agent responses."""

    def __init__(self, judge_llm: MockJudgeLLM = None):
        self.judge = judge_llm or MockJudgeLLM()
        self.default_criteria = ["relevance", "accuracy", "helpfulness", "clarity"]

    async def evaluate_single(
        self,
        query: str,
        response: str,
        criteria: list[str] = None
    ) -> EvaluationResult:
        """Evaluate a single query-response pair."""
        criteria = criteria or self.default_criteria

        evaluation = await self.judge.evaluate(query, response, criteria)

        passed = all(
            score >= 3 for score in evaluation["scores"].values()
        )

        return EvaluationResult(
            query=query,
            response=response,
            scores=evaluation["scores"],
            feedback=evaluation["feedback"],
            passed=passed
        )

    async def evaluate_batch(
        self,
        test_cases: list[dict]
    ) -> dict:
        """Evaluate multiple test cases and aggregate results."""
        results = []

        for case in test_cases:
            result = await self.evaluate_single(
                query=case["query"],
                response=case["response"],
                criteria=case.get("criteria", self.default_criteria)
            )
            results.append(result)

        # Aggregate metrics
        if not results:
            return {"error": "No test cases provided"}

        all_scores = {criterion: [] for criterion in self.default_criteria}
        for result in results:
            for criterion, score in result.scores.items():
                if criterion in all_scores:
                    all_scores[criterion].append(score)

        return {
            "total_cases": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
            "average_scores": {
                criterion: sum(scores) / len(scores) if scores else 0
                for criterion, scores in all_scores.items()
            },
            "detailed_results": results
        }


# ============================================================
# Test Datasets
# ============================================================

HAPPY_PATH_CASES = [
    {
        "query": "What is Python?",
        "response": "Python is a high-level programming language known for its simplicity and readability. It's widely used for web development, data science, and automation.",
        "expected_pass": True
    },
    {
        "query": "How do I create a list in Python?",
        "response": "You can create a list in Python using square brackets: my_list = [1, 2, 3]. You can also use the list() constructor.",
        "expected_pass": True
    },
    {
        "query": "Explain machine learning",
        "response": "Machine learning is a subset of AI where systems learn patterns from data. It includes supervised learning, unsupervised learning, and reinforcement learning.",
        "expected_pass": True
    }
]

EDGE_CASES = [
    {
        "query": "",
        "response": "I'd be happy to help, but I didn't receive a question. Could you please provide more details?",
        "expected_pass": True
    },
    {
        "query": "??????????",
        "response": "I'm not sure I understand your question. Could you rephrase it?",
        "expected_pass": True
    },
    {
        "query": "a" * 1000,
        "response": "It seems like your message might have been corrupted. Please try again with your actual question.",
        "expected_pass": True
    }
]

FAILURE_CASES = [
    {
        "query": "What is the capital of France?",
        "response": ".",
        "expected_pass": False
    },
    {
        "query": "Explain quantum computing",
        "response": "I don't know.",
        "expected_pass": False
    }
]


# ============================================================
# Evaluation Tests
# ============================================================

class TestEvaluator:
    """Tests for the evaluation framework."""

    @pytest.fixture
    def evaluator(self):
        return Evaluator()

    @pytest.mark.asyncio
    async def test_evaluate_good_response(self, evaluator):
        """Good responses should pass evaluation."""
        result = await evaluator.evaluate_single(
            query="What is Python?",
            response="Python is a programming language known for its simplicity and readability."
        )

        assert result.passed is True
        assert result.scores["relevance"] >= 3

    @pytest.mark.asyncio
    async def test_evaluate_poor_response(self, evaluator):
        """Poor responses should fail evaluation."""
        # Use a judge that gives low scores
        strict_judge = MockJudgeLLM(scores={
            "relevance": 2,
            "accuracy": 2,
            "helpfulness": 2,
            "clarity": 2
        })
        strict_evaluator = Evaluator(judge_llm=strict_judge)

        result = await strict_evaluator.evaluate_single(
            query="Explain AI",
            response="No."
        )

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, evaluator):
        """All scores should be between 1 and 5."""
        result = await evaluator.evaluate_single(
            query="Test query",
            response="Test response"
        )

        for score in result.scores.values():
            assert 1 <= score <= 5

    @pytest.mark.asyncio
    async def test_feedback_generated(self, evaluator):
        """Evaluation should generate feedback."""
        result = await evaluator.evaluate_single(
            query="Test query",
            response="Here is a helpful response for you."
        )

        assert result.feedback is not None
        assert len(result.feedback) > 0


class TestBatchEvaluation:
    """Tests for batch evaluation."""

    @pytest.fixture
    def evaluator(self):
        return Evaluator()

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, evaluator):
        """Batch evaluation should process all cases."""
        test_cases = [
            {"query": "What is AI?", "response": "AI is artificial intelligence."},
            {"query": "What is ML?", "response": "ML is machine learning."},
        ]

        results = await evaluator.evaluate_batch(test_cases)

        assert results["total_cases"] == 2
        assert "average_scores" in results
        assert "pass_rate" in results

    @pytest.mark.asyncio
    async def test_batch_pass_rate_calculation(self, evaluator):
        """Pass rate should be calculated correctly."""
        # All good responses
        test_cases = [
            {"query": "Q1", "response": "Here is a helpful answer for you."},
            {"query": "Q2", "response": "Let me help you with that question."},
        ]

        results = await evaluator.evaluate_batch(test_cases)

        assert 0 <= results["pass_rate"] <= 1

    @pytest.mark.asyncio
    async def test_empty_batch(self, evaluator):
        """Empty batch should be handled gracefully."""
        results = await evaluator.evaluate_batch([])

        assert "error" in results


class TestHappyPathCases:
    """Tests using happy path dataset."""

    @pytest.fixture
    def evaluator(self):
        return Evaluator()

    @pytest.mark.asyncio
    async def test_happy_path_cases_pass(self, evaluator):
        """Happy path cases should generally pass."""
        for case in HAPPY_PATH_CASES:
            result = await evaluator.evaluate_single(
                query=case["query"],
                response=case["response"]
            )

            # Most happy path cases should pass
            if case["expected_pass"]:
                assert result.passed is True, f"Failed for query: {case['query']}"


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def evaluator(self):
        return Evaluator()

    @pytest.mark.asyncio
    async def test_edge_cases_handled(self, evaluator):
        """Edge cases should be handled without errors."""
        for case in EDGE_CASES:
            # Should not raise an exception
            result = await evaluator.evaluate_single(
                query=case["query"],
                response=case["response"]
            )

            assert result is not None
            assert result.scores is not None


# ============================================================
# Metrics Collection Tests
# ============================================================

@dataclass
class AgentMetrics:
    """Metrics collected during agent execution."""
    task_completed: bool
    response_time_ms: float
    tokens_used: int
    tools_called: list = field(default_factory=list)
    error_occurred: bool = False
    error_message: Optional[str] = None


class MetricsCollector:
    """Collects and aggregates agent metrics."""

    def __init__(self):
        self.metrics: list[AgentMetrics] = []

    def record(self, metric: AgentMetrics):
        """Record a single metric."""
        self.metrics.append(metric)

    def summary(self) -> dict:
        """Generate summary statistics."""
        if not self.metrics:
            return {}

        return {
            "total_runs": len(self.metrics),
            "completion_rate": sum(m.task_completed for m in self.metrics) / len(self.metrics),
            "avg_response_time": sum(m.response_time_ms for m in self.metrics) / len(self.metrics),
            "error_rate": sum(m.error_occurred for m in self.metrics) / len(self.metrics),
            "avg_tokens": sum(m.tokens_used for m in self.metrics) / len(self.metrics),
        }

    def clear(self):
        """Clear all recorded metrics."""
        self.metrics = []


class TestMetricsCollector:
    """Tests for metrics collection."""

    def test_record_and_summarize(self):
        """Metrics should be recorded and summarized correctly."""
        collector = MetricsCollector()

        collector.record(AgentMetrics(
            task_completed=True,
            response_time_ms=100,
            tokens_used=50
        ))
        collector.record(AgentMetrics(
            task_completed=True,
            response_time_ms=200,
            tokens_used=100
        ))

        summary = collector.summary()

        assert summary["total_runs"] == 2
        assert summary["completion_rate"] == 1.0
        assert summary["avg_response_time"] == 150

    def test_error_rate_calculation(self):
        """Error rate should be calculated correctly."""
        collector = MetricsCollector()

        collector.record(AgentMetrics(
            task_completed=True, response_time_ms=100,
            tokens_used=50, error_occurred=False
        ))
        collector.record(AgentMetrics(
            task_completed=False, response_time_ms=100,
            tokens_used=50, error_occurred=True
        ))

        summary = collector.summary()

        assert summary["error_rate"] == 0.5

    def test_empty_collector(self):
        """Empty collector should return empty summary."""
        collector = MetricsCollector()

        summary = collector.summary()

        assert summary == {}

    def test_clear_metrics(self):
        """Clear should remove all metrics."""
        collector = MetricsCollector()

        collector.record(AgentMetrics(
            task_completed=True, response_time_ms=100, tokens_used=50
        ))
        collector.clear()

        assert len(collector.metrics) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
