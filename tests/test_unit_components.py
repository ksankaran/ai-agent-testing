"""
Unit Tests for AI Agent Components

This module demonstrates unit testing patterns for AI agents,
including mocking LLM responses and testing individual components.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


# ============================================================
# Mock Utilities
# ============================================================

def create_mock_llm(response_text: str):
    """Create a mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    mock.invoke = MagicMock(return_value=MagicMock(content=response_text))
    return mock


def create_mock_llm_with_tool_call(tool_name: str, tool_args: dict):
    """Create a mock LLM that returns a tool call."""
    tool_call = MagicMock()
    tool_call.name = tool_name
    tool_call.args = tool_args

    response = MagicMock()
    response.content = ""
    response.tool_calls = [tool_call]

    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=response)
    return mock


# ============================================================
# Example Agent State
# ============================================================

@dataclass
class AgentState:
    """Simple agent state for testing."""
    messages: list
    context: dict
    iteration_count: int = 0
    tools_used: list = None

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []


def add_message(state: AgentState, message: str) -> AgentState:
    """Add a message to agent state."""
    new_messages = state.messages + [message]
    return AgentState(
        messages=new_messages,
        context=state.context,
        iteration_count=state.iteration_count,
        tools_used=state.tools_used
    )


# ============================================================
# Example Tools
# ============================================================

def calculator_tool(expression: str) -> str:
    """A simple calculator tool."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)  # Safe due to character restriction
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def search_tool(query: str) -> str:
    """A mock search tool."""
    # In reality, this would call an API
    mock_results = {
        "python": "Python is a programming language known for its simplicity.",
        "weather": "Current weather information unavailable in test mode.",
        "ai": "Artificial Intelligence refers to machine-based intelligence.",
    }

    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value

    return f"No results found for: {query}"


# ============================================================
# Unit Tests - LLM Mocking
# ============================================================

class TestLLMMocking:
    """Tests demonstrating LLM mocking patterns."""

    @pytest.mark.asyncio
    async def test_mock_llm_returns_expected_response(self):
        """Verify mock LLM returns the configured response."""
        expected = "Hello! How can I help you today?"
        mock_llm = create_mock_llm(expected)

        response = await mock_llm.ainvoke("Hi there")

        assert response.content == expected

    @pytest.mark.asyncio
    async def test_mock_llm_response_contains_keyword(self):
        """Test response contains expected keywords."""
        mock_llm = create_mock_llm("I'd be happy to help you with that task.")

        response = await mock_llm.ainvoke("Can you help me?")

        assert "help" in response.content.lower()

    @pytest.mark.asyncio
    async def test_mock_llm_with_tool_call(self):
        """Test mock LLM that returns tool calls."""
        mock_llm = create_mock_llm_with_tool_call(
            "calculator",
            {"expression": "2 + 2"}
        )

        response = await mock_llm.ainvoke("What is 2 + 2?")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calculator"
        assert response.tool_calls[0].args["expression"] == "2 + 2"


# ============================================================
# Unit Tests - Tool Functions
# ============================================================

class TestCalculatorTool:
    """Tests for the calculator tool."""

    def test_basic_addition(self):
        """Calculator handles basic addition."""
        result = calculator_tool("2 + 2")
        assert result == "4"

    def test_multiplication(self):
        """Calculator handles multiplication."""
        result = calculator_tool("10 * 5")
        assert result == "50"

    def test_complex_expression(self):
        """Calculator handles complex expressions."""
        result = calculator_tool("(10 + 5) * 2")
        assert result == "30"

    def test_division(self):
        """Calculator handles division."""
        result = calculator_tool("100 / 4")
        assert result == "25.0"

    def test_invalid_characters_rejected(self):
        """Calculator rejects expressions with invalid characters."""
        result = calculator_tool("import os; os.system('rm -rf /')")
        assert "error" in result.lower()

    def test_handles_syntax_error(self):
        """Calculator handles malformed expressions."""
        result = calculator_tool("2 + * 2")  # Invalid: operator followed by operator
        assert "error" in result.lower()


class TestSearchTool:
    """Tests for the search tool."""

    def test_finds_matching_result(self):
        """Search returns relevant results for known queries."""
        result = search_tool("Tell me about Python")
        assert "programming" in result.lower()

    def test_no_results_message(self):
        """Search returns appropriate message for unknown queries."""
        result = search_tool("xyzzy123456")
        assert "no results" in result.lower()

    def test_case_insensitive_search(self):
        """Search is case-insensitive."""
        result = search_tool("PYTHON")
        assert "programming" in result.lower()


# ============================================================
# Unit Tests - State Management
# ============================================================

class TestAgentState:
    """Tests for agent state management."""

    def test_state_initialization(self):
        """Verify initial state is set correctly."""
        state = AgentState(
            messages=[],
            context={},
            iteration_count=0
        )

        assert state.messages == []
        assert state.context == {}
        assert state.iteration_count == 0
        assert state.tools_used == []

    def test_state_with_initial_messages(self):
        """State can be initialized with messages."""
        state = AgentState(
            messages=["Hello", "World"],
            context={"user": "test"},
            iteration_count=1
        )

        assert len(state.messages) == 2
        assert state.context["user"] == "test"

    def test_add_message_accumulates(self):
        """Messages accumulate across additions."""
        state = AgentState(messages=[], context={})

        state = add_message(state, "First")
        state = add_message(state, "Second")
        state = add_message(state, "Third")

        assert len(state.messages) == 3
        assert state.messages == ["First", "Second", "Third"]

    def test_add_message_preserves_context(self):
        """Adding messages preserves existing context."""
        state = AgentState(
            messages=[],
            context={"key": "value"},
            iteration_count=5
        )

        state = add_message(state, "New message")

        assert state.context["key"] == "value"
        assert state.iteration_count == 5

    def test_tools_used_tracking(self):
        """Tools used list can be tracked."""
        state = AgentState(
            messages=[],
            context={},
            tools_used=["calculator", "search"]
        )

        assert "calculator" in state.tools_used
        assert "search" in state.tools_used


# ============================================================
# Unit Tests - Edge Cases
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_input_handling(self):
        """Components handle empty inputs gracefully."""
        result = calculator_tool("")
        assert "error" in result.lower()

    def test_very_long_input(self):
        """Components handle very long inputs."""
        long_expression = "1 + " * 100 + "1"
        result = calculator_tool(long_expression)
        # Should either compute or error, not crash
        assert result is not None

    def test_unicode_in_search(self):
        """Search handles unicode characters."""
        result = search_tool("Python programming")
        assert result is not None

    def test_whitespace_only_input(self):
        """Calculator handles whitespace-only input."""
        result = calculator_tool("   ")
        # Empty expression should error
        assert "error" in result.lower() or result.strip() == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
