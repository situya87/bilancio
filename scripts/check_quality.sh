#!/usr/bin/env bash

set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

RUFF_TARGETS=(
  src/bilancio/ui/cli
  src/bilancio/ui/sweep_setup.py
  tests/test_smoke.py
  tests/ui/test_cli.py
  tests/ui/test_cli_integration.py
  tests/ui/test_cli_run_coverage.py
  tests/ui/test_cli_sweep.py
  tests/ui/test_sweep_setup.py
)

MYPY_TARGETS=(
  src/bilancio/ui/cli
  src/bilancio/ui/sweep_setup.py
)

PYTHONPATH=src "$PYTHON_BIN" -m ruff check "${RUFF_TARGETS[@]}"
PYTHONPATH=src "$PYTHON_BIN" -m ruff format --check "${RUFF_TARGETS[@]}"
PYTHONPATH=src "$PYTHON_BIN" -m mypy "${MYPY_TARGETS[@]}"
PYTHONPATH=src "$PYTHON_BIN" -m pytest \
  tests/ \
  --ignore=tests/benchmark \
  --no-cov \
  -m "not slow" \
  -q
