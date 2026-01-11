"""
Live Integration Tests with Real LLM Calls

These tests make actual API calls to OpenAI.
Run with: pytest tests/test_live.py -v

Requires OPENAI_API_KEY environment variable.
"""

import pytest
import os
from dataclasses import dataclass

# Skip all tests in this module if no API key
pytestmark = pytest.mark.live


@dataclass
class SimpleAgent:
    """A simple agent that uses OpenAI for responses."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = "gpt-4o-mini"
        self.tools = {
            "calculator": self._calculator,
            "get_weather": self._get_weather,
        }

    def _calculator(self, expression: str) -> str:
        """Evaluate a math expression."""
        try:
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return "Invalid expression"
        except Exception as e:
            return f"Error: {e}"

    def _get_weather(self, city: str) -> str:
        """Mock weather tool."""
        return f"The weather in {city} is 72°F and sunny."

    def invoke(self, query: str) -> dict:
        """Process a query with tool support."""
        tools_spec = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate mathematical expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        messages = [{"role": "user", "content": query}]
        tools_used = []

        # First call - may return tool calls
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools_spec,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Handle tool calls if any
        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tools_used.append(tool_name)

                import json
                args = json.loads(tool_call.function.arguments)

                # Execute tool
                if tool_name == "calculator":
                    result = self._calculator(args["expression"])
                elif tool_name == "get_weather":
                    result = self._get_weather(args["city"])
                else:
                    result = "Unknown tool"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Get final response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            message = response.choices[0].message

        return {
            "response": message.content,
            "tools_used": tools_used
        }


class TestLiveToolSelection:
    """Test that the agent selects appropriate tools with real LLM."""

    @pytest.fixture
    def agent(self):
        return SimpleAgent()

    def test_selects_calculator_for_math(self, agent):
        """Agent should use calculator for math questions."""
        result = agent.invoke("What is 125 * 8?")

        assert "calculator" in result["tools_used"]
        assert "1000" in result["response"]

    def test_selects_weather_for_weather_query(self, agent):
        """Agent should use weather tool for weather questions."""
        result = agent.invoke("What's the weather in Tokyo?")

        assert "get_weather" in result["tools_used"]
        assert "Tokyo" in result["response"] or "72" in result["response"]

    def test_no_tool_for_general_question(self, agent):
        """Agent should not use tools for general questions."""
        result = agent.invoke("What is the capital of France?")

        assert len(result["tools_used"]) == 0
        assert "Paris" in result["response"]


class TestLiveResponseQuality:
    """Test response quality with real LLM."""

    @pytest.fixture
    def agent(self):
        return SimpleAgent()

    def test_response_is_relevant(self, agent):
        """Response should be relevant to the query."""
        result = agent.invoke("Explain what Python is in one sentence.")

        response = result["response"].lower()
        assert "python" in response or "programming" in response or "language" in response

    def test_response_follows_instructions(self, agent):
        """Agent should follow specific instructions."""
        result = agent.invoke("Say 'HELLO' in all caps and nothing else.")

        assert "HELLO" in result["response"]

    def test_handles_complex_math(self, agent):
        """Agent should handle complex calculations correctly."""
        result = agent.invoke("Calculate (15 + 25) * 3 - 10")

        assert "calculator" in result["tools_used"]
        assert "110" in result["response"]


class TestLiveErrorHandling:
    """Test error handling with real LLM."""

    @pytest.fixture
    def agent(self):
        return SimpleAgent()

    def test_handles_ambiguous_query(self, agent):
        """Agent should handle ambiguous queries gracefully."""
        result = agent.invoke("Calculate the thing")

        # Should either ask for clarification or not crash
        assert result["response"] is not None
        assert len(result["response"]) > 0


class TestLiveConsistency:
    """Test response consistency with real LLM."""

    @pytest.fixture
    def agent(self):
        return SimpleAgent()

    def test_math_consistency(self, agent):
        """Math results should be consistent across calls."""
        results = []
        for _ in range(3):
            result = agent.invoke("What is 50 + 50?")
            results.append(result)

        # All should use calculator and contain 100
        for result in results:
            assert "calculator" in result["tools_used"]
            assert "100" in result["response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
