# 🚀 Concurrent Multi-Provider LLM Client

A production-ready Python package that provides a unified interface for interacting with multiple Large Language Model (LLM) providers concurrently.

The project demonstrates modern Python software engineering practices including:

- Production project structure (`src/` layout)
- `pyproject.toml` packaging
- Concurrent execution using `asyncio`
- Provider abstraction
- Unit testing with `pytest`
- Static analysis using `mypy`
- Linting with `ruff`
- Git pre-commit hooks
- SOLID principles
- Clean Architecture

---

# Features

- 🔹 Unified interface for multiple LLM providers
- 🔹 OpenAI, Anthropic and Google Provider
- 🔹 Async concurrent request execution
- 🔹 Custom exception handling
- 🔹 Strongly typed response models
- 🔹 Production-ready project structure
- 🔹 Unit tested with pytest
- 🔹 Static type checking
- 🔹 Automatic linting before commits

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Async | asyncio |
| Testing | pytest |
| Type Checking | mypy |
| Linting | ruff |
| Hooks | pre-commit |
| Packaging | pyproject.toml |
| Environment | uv |
| Version Control | Git |

---

# Project Structure

```text
.
├── docs/
├── src/
│   └── llm_client/
│       ├── api/
│       ├── core/
│       ├── db/
│       ├── models/
│           └── response_model.py
│       ├── services/
│       │   ├── llm_service.py
│       │   └── providers/
│       │       ├── openai_provider.py
│       │       ├── anthropic_provider.py
│       │       └── google_provider.py
│       ├── utils/
│       ├── config.py
│       ├── exceptions.py
│       └── main.py
│
├── tests/
│   ├── unit/
│   │   ├── test_client.py
│   │   ├── test_complete_all.py
│   │   ├── test_exception.py
│   │   └── test_providers.py
│   └── integration/
│
├── .env
├── .env.example
├── .gitignore
├── .pre-commit-config.toml
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

---

# Architecture

```text
                      User
                        │
                        ▼
                LLM Service
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   OpenAI Provider  Anthropic     Google Gemini
         │              │              │
         └──────────────┼──────────────┘
                        ▼
                Response Model
```

---

# Software Design

The project follows several software engineering best practices.

## Single Responsibility Principle

Each provider is responsible only for communicating with its own API.

```
OpenAI Provider
↓

OpenAI only
```

```
Anthropic Provider
↓

Anthropic only
```

```
Google Provider
↓

Gemini only
```

---

## Open/Closed Principle

New providers can be added without modifying existing providers.

Example:

```
LLM Service

↓

Provider

↓

OpenAI

Anthropic

Gemini

Future Provider
```

---

## Liskov Substitution Principle

Every provider exposes the same public behaviour, allowing them to be used interchangeably.

---

## Interface Segregation Principle

Each module contains only functionality related to its own responsibility.

---

## Dependency Inversion Principle

The service depends on provider abstractions instead of concrete implementations.

---

# Installation

Clone the repository

```bash
git clone https://github.com/harshilkatharva/genai-engineer-journey
```

Install dependencies

```bash
uv sync
```

Activate virtual environment

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

---

# Configuration

Create a `.env` file. or you get demo from .env.example

```text
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

---

# Usage

Example

```python
from llm_client.services.llm_service import LLMService

service = LLMService()

response = service.complete(
    provider="openai",
    prompt="Explain SOLID Principles."
)

print(response)
```

---

# Running Tests

Run all tests

```bash
pytest
```

Run unit tests

```bash
pytest tests/unit
```

Run with coverage

```bash
pytest --cov=src
```

---

# Type Checking

```bash
mypy src
```

---

# Linting

```bash
ruff check .
```

Automatically fix issues

```bash
ruff check . --fix
```

---

# Pre-commit

Install hooks

```bash
pre-commit install
```

Run manually

```bash
pre-commit run --all-files
```

---

# Git Workflow

The project was developed using a production-style Git workflow.

1. Create feature branch

```bash
git switch -c module-02
```

2. Commit small logical changes

```bash
git add .

git commit -m "Add Google provider"
```

3. Push

```bash
git push origin module-02
```

4. Open Pull Request

5. Review

6. Merge into main

---

# Learning Outcomes

This project was built to practice production Python software engineering concepts.

Topics covered:

- Async Programming
- asyncio
- Python Packaging
- src Layout
- pyproject.toml
- SOLID Principles
- Clean Architecture
- Git Workflow
- Ruff
- Mypy
- Pytest
- Pre-commit
- Concurrent Programming

---

# Author
**Harshil Kareliya**    