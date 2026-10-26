#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Running pre-push checks for module-18-plan_and_execute_research_assistant ==="

# 1. Ruff linting
echo "--> Checking ruff rules..."
uv run --directory "$MODULE_DIR" ruff check .

# 2. Ruff formatting
echo "--> Checking ruff formatting..."
uv run --directory "$MODULE_DIR" ruff format --check .

# 3. Mypy type-checking
echo "--> Checking mypy types..."
uv run --directory "$MODULE_DIR" mypy .

# 4. Pytest test suite
echo "--> Running test suite..."
uv run --directory "$MODULE_DIR" pytest -v

echo "=== All module-18-plan_and_execute_research_assistant checks passed successfully! ==="
