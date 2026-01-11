"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures and configuration
for all test modules in the testing suite.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field
from typing import Callable


# ============================================================
# Shared Data Classes
# ============================================================

@dataclass
class AgentState:
    """Shared agent state for testing."""
    messages: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    iteration_count: int = 0
    tools_used: list = field(default_factory=list)


@dataclass
class Tool:
    """Tool definition for agents."""
    name: str
    description: str
    function: Callable


# ============================================================
# Mock LLM Factories
# ============================================================

@pytest.fixture
def mock_llm():
    """Create a basic mock LLM."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(
        content="This is a mock response."
    ))
    mock.invoke = MagicMock(return_value=MagicMock(
        content="This is a mock response."
    ))
    return mock


@pytest.fixture
def mock_llm_factory():
    """Factory for creating mock LLMs with custom responses."""
    def _create(response_text: str):
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
        mock.invoke = MagicMock(return_value=MagicMock(content=response_text))
        return mock
    return _create


@pytest.fixture
def mock_llm_with_tool_calls():
    """Factory for creating mock LLMs that return tool calls."""
    def _create(tool_name: str, tool_args: dict):
        tool_call = MagicMock()
        tool_call.name = tool_name
        tool_call.args = tool_args

        response = MagicMock()
        response.content = ""
        response.tool_calls = [tool_call]

        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=response)
        return mock
    return _create


# ============================================================
# Tool Fixtures
# ============================================================

def _calculator_function(expression: str) -> str:
    """Calculator implementation."""
    try:
        allowed = set("0123456789+-*/(). ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"


def _search_function(query: str) -> str:
    """Mock search implementation."""
    return f"Search results for: {query}"


def _summarize_function(text: str) -> str:
    """Mock summarize implementation."""
    return f"Summary: {text[:50]}..."


@pytest.fixture
def calculator_tool():
    """Provide calculator tool."""
    return Tool(
        name="calculator",
        description="Performs mathematical calculations",
        function=_calculator_function
    )


@pytest.fixture
def search_tool():
    """Provide search tool."""
    return Tool(
        name="search",
        description="Searches for information",
        function=_search_function
    )


@pytest.fixture
def summarize_tool():
    """Provide summarize tool."""
    return Tool(
        name="summarize",
        description="Summarizes text content",
        function=_summarize_function
    )


@pytest.fixture
def all_tools(calculator_tool, search_tool, summarize_tool):
    """Provide all available tools."""
    return [calculator_tool, search_tool, summarize_tool]


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def sample_queries():
    """Provide sample queries for testing."""
    return [
        "What is 2 + 2?",
        "Tell me about Python",
        "Search for machine learning tutorials",
        "Summarize this text",
        "Hello, how are you?",
    ]


@pytest.fixture
def sample_responses():
    """Provide sample responses for testing."""
    return [
        "The result is 4.",
        "Python is a programming language.",
        "Here are some machine learning resources...",
        "In summary, the text discusses...",
        "I'm doing well, thank you for asking!",
    ]


@pytest.fixture
def query_response_pairs(sample_queries, sample_responses):
    """Provide paired queries and responses."""
    return list(zip(sample_queries, sample_responses))


# ============================================================
# Configuration
# ============================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
