#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Running pre-push checks for module-17-ai_agents ==="

# 1. Ruff linting
echo "--> Checking ruff rules..."
uv run --directory "$MODULE_DIR" ruff check .

# 2. Ruff formatting
echo "--> Checking ruff formatting..."
uv run --directory "$MODULE_DIR" ruff format --check .

# 3. Mypy type-checking
echo "--> Checking mypy types..."
uv run --directory "$MODULE_DIR" mypy src tests

# 4. Pytest test suite
echo "--> Running test suite..."
uv run --directory "$MODULE_DIR" pytest -v

echo "=== All module-17-ai_agents checks passed successfully! ==="
