#!/usr/bin/env bash
set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PACKAGE_DIR="$REPO_ROOT/packages/genai-core"

echo "=== Running pre-push checks for genai-core ==="

# 1. Ruff linting
echo "--> Checking ruff rules..."
uv run --directory "$PACKAGE_DIR" ruff check .

# 2. Ruff formatting
echo "--> Checking ruff formatting..."
uv run --directory "$PACKAGE_DIR" ruff format --check .

# 3. Mypy type-checking
echo "--> Checking mypy types..."
uv run --directory "$PACKAGE_DIR" mypy src tests

echo "=== All genai-core pre-push checks passed successfully! ==="
