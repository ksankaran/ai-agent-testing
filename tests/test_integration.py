"""
Integration Tests for AI Agent Workflows

This module demonstrates integration testing patterns,
verifying that agent components work together correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Callable


# ============================================================
# Agent Simulation Framework
# ============================================================

@dataclass
class AgentResult:
    """Result from agent execution."""
    response: str
    tools_used: list = field(default_factory=list)
    steps_completed: int = 0
    error_handled: bool = False
    error_message: str = None


@dataclass
class Tool:
    """A tool that the agent can use."""
    name: str
    description: str
    function: Callable


class SimpleAgent:
    """A simple agent for testing purposes."""

    def __init__(self, tools: list[Tool], llm=None):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm or self._create_default_llm()

    def _create_default_llm(self):
        """Create a default mock LLM."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content="Default response"))
        return mock

    async def invoke(self, query: str) -> AgentResult:
        """Process a query and return results."""
        tools_used = []
        steps = 0

        # Simple keyword-based tool selection for testing
        query_lower = query.lower()

        try:
            if any(word in query_lower for word in ["calculate", "math", "compute", "*", "+", "-", "/"]):
                if "calculator" in self.tools:
                    tools_used.append("calculator")
                    steps += 1

            if any(word in query_lower for word in ["search", "find", "look up", "weather", "news"]):
                if "search" in self.tools:
                    tools_used.append("search")
                    steps += 1

            if any(word in query_lower for word in ["research", "analyze", "investigate"]):
                if "search" in self.tools:
                    tools_used.append("search")
                    steps += 1
                if "summarize" in self.tools:
                    tools_used.append("summarize")
                    steps += 1

            # Get LLM response
            llm_response = await self.llm.ainvoke(query)
            steps += 1

            return AgentResult(
                response=llm_response.content,
                tools_used=tools_used,
                steps_completed=steps
            )

        except Exception as e:
            return AgentResult(
                response=f"I encountered an error: {str(e)}",
                tools_used=tools_used,
                steps_completed=steps,
                error_handled=True,
                error_message=str(e)
            )


class FailingTool:
    """A tool that always fails, for testing error handling."""

    def __init__(self, error_message: str = "Tool execution failed"):
        self.name = "failing_tool"
        self.description = "A tool that fails"
        self.error_message = error_message

    def __call__(self, *args, **kwargs):
        raise RuntimeError(self.error_message)


# ============================================================
# Test Fixtures
# ============================================================

def calculator_function(expression: str) -> str:
    """Calculator tool function."""
    try:
        allowed = set("0123456789+-*/(). ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Invalid expression"
    except Exception:
        return "Calculation error"


def search_function(query: str) -> str:
    """Mock search function."""
    return f"Search results for: {query}"


def summarize_function(text: str) -> str:
    """Mock summarize function."""
    return f"Summary: {text[:50]}..."


@pytest.fixture
def basic_tools():
    """Create basic set of tools for testing."""
    return [
        Tool("calculator", "Performs math calculations", calculator_function),
        Tool("search", "Searches for information", search_function),
    ]


@pytest.fixture
def research_tools():
    """Create tools for research agent testing."""
    return [
        Tool("search", "Searches for information", search_function),
        Tool("summarize", "Summarizes text", summarize_function),
    ]


@pytest.fixture
def mock_llm_calculator():
    """Create LLM that suggests using calculator."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(
        content="The result of the calculation is 100."
    ))
    return mock


@pytest.fixture
def mock_llm_helpful():
    """Create a helpful LLM response."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(
        content="I'd be happy to help you with that request."
    ))
    return mock


# ============================================================
# Integration Tests - Tool Selection
# ============================================================

class TestToolSelection:
    """Tests verifying agent selects appropriate tools."""

    @pytest.mark.asyncio
    async def test_selects_calculator_for_math(self, basic_tools, mock_llm_calculator):
        """Agent should use calculator for math questions."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_calculator)

        result = await agent.invoke("Calculate 25 * 4")

        assert "calculator" in result.tools_used

    @pytest.mark.asyncio
    async def test_selects_search_for_lookup(self, basic_tools, mock_llm_helpful):
        """Agent should use search for information queries."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Search for Python tutorials")

        assert "search" in result.tools_used

    @pytest.mark.asyncio
    async def test_no_tools_for_simple_chat(self, basic_tools, mock_llm_helpful):
        """Agent should not use tools for simple conversation."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Hello, how are you?")

        assert len(result.tools_used) == 0

    @pytest.mark.asyncio
    async def test_multiple_tools_for_complex_query(self, research_tools, mock_llm_helpful):
        """Agent should use multiple tools for complex research queries."""
        agent = SimpleAgent(tools=research_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Research and analyze Python trends")

        assert "search" in result.tools_used
        assert "summarize" in result.tools_used
        assert result.steps_completed >= 2


# ============================================================
# Integration Tests - Multi-Step Workflows
# ============================================================

class TestMultiStepWorkflows:
    """Tests for multi-step agent workflows."""

    @pytest.mark.asyncio
    async def test_research_workflow_steps(self, research_tools, mock_llm_helpful):
        """Research workflow should complete multiple steps."""
        agent = SimpleAgent(tools=research_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Research machine learning applications")

        assert result.steps_completed >= 2
        assert len(result.tools_used) >= 1

    @pytest.mark.asyncio
    async def test_workflow_produces_response(self, basic_tools, mock_llm_helpful):
        """Workflow should produce a non-empty response."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Help me with a calculation")

        assert result.response is not None
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_steps_tracked_correctly(self, basic_tools, mock_llm_helpful):
        """All steps should be tracked in the result."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Calculate something")

        # At minimum: tool selection + LLM response
        assert result.steps_completed >= 1


# ============================================================
# Integration Tests - Error Handling
# ============================================================

class TestErrorHandling:
    """Tests for error handling in agent workflows."""

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, basic_tools):
        """Agent should handle LLM errors gracefully."""
        failing_llm = MagicMock()
        failing_llm.ainvoke = AsyncMock(side_effect=Exception("LLM API error"))

        agent = SimpleAgent(tools=basic_tools, llm=failing_llm)

        result = await agent.invoke("Hello")

        assert result.error_handled is True
        assert "error" in result.response.lower()

    @pytest.mark.asyncio
    async def test_continues_after_partial_failure(self, basic_tools, mock_llm_helpful):
        """Agent should continue even if some steps fail."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        # Query that might trigger multiple tools
        result = await agent.invoke("Search for something")

        # Should still produce a response
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_error_message_captured(self, basic_tools):
        """Error messages should be captured in result."""
        failing_llm = MagicMock()
        failing_llm.ainvoke = AsyncMock(side_effect=ValueError("Specific error"))

        agent = SimpleAgent(tools=basic_tools, llm=failing_llm)

        result = await agent.invoke("Hello")

        assert result.error_handled is True
        assert "Specific error" in result.error_message


# ============================================================
# Integration Tests - Response Quality
# ============================================================

class TestResponseQuality:
    """Tests verifying response quality."""

    @pytest.mark.asyncio
    async def test_response_addresses_query(self, basic_tools, mock_llm_helpful):
        """Response should be relevant to the query."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Help me understand Python")

        assert result.response is not None
        assert len(result.response) > 10

    @pytest.mark.asyncio
    async def test_response_not_empty_after_error(self, basic_tools):
        """Even after errors, response should not be empty."""
        failing_llm = MagicMock()
        failing_llm.ainvoke = AsyncMock(side_effect=Exception("Error"))

        agent = SimpleAgent(tools=basic_tools, llm=failing_llm)

        result = await agent.invoke("Hello")

        assert result.response is not None
        assert len(result.response) > 0


# ============================================================
# Integration Tests - State Consistency
# ============================================================

class TestStateConsistency:
    """Tests verifying state remains consistent across operations."""

    @pytest.mark.asyncio
    async def test_tools_used_matches_steps(self, basic_tools, mock_llm_helpful):
        """Number of tools used should be consistent with steps."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result = await agent.invoke("Calculate 5 + 5")

        # Tools used + LLM call = steps
        assert result.steps_completed >= len(result.tools_used)

    @pytest.mark.asyncio
    async def test_multiple_invocations_independent(self, basic_tools, mock_llm_helpful):
        """Multiple invocations should be independent."""
        agent = SimpleAgent(tools=basic_tools, llm=mock_llm_helpful)

        result1 = await agent.invoke("Calculate something")
        result2 = await agent.invoke("Search for something")

        # Results should be independent
        assert result1.tools_used != result2.tools_used


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
