# AI Agent Testing Examples

A collection of pytest examples demonstrating testing patterns for AI agents and LLM-powered systems.

## What's Included

- **Unit Tests** — Mocking LLMs, testing tools, state management
- **Integration Tests** — Tool selection, multi-step workflows, error handling
- **Evaluation Tests** — LLM-as-judge patterns, metrics collection
- **A/B Testing** — Traffic splitting, statistical significance testing

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ksankaran/ai-agent-testing.git
cd ai-agent-testing
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_unit_components.py -v
pytest tests/test_integration.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_ab_testing.py -v
```

Run with coverage:
```bash
pip install pytest-cov
pytest tests/ --cov=tests --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_unit_components.py  # Unit tests for agent components
├── test_integration.py      # Integration tests for workflows
├── test_evaluation.py       # LLM-as-judge evaluation tests
└── test_ab_testing.py       # A/B testing framework tests
```

## License

MIT
