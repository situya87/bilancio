# Contributing to Bilancio

Thank you for your interest in contributing to Bilancio! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vlad-ds/bilancio.git
   cd bilancio
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create and activate a virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install the package in development mode:**
   ```bash
   uv pip install -e ".[dev,analysis,cloud]"
   ```

### Installation Profiles

Use the smallest install that matches the work you are doing:

- `core`: `uv pip install -e .`
- `dev`: `uv pip install -e ".[dev]"`
- `analysis/viz`: `uv pip install -e ".[analysis]"`
- `cloud`: `uv pip install -e ".[cloud]"`
- `notebooks`: `uv pip install -e ".[analysis,notebooks]"`

Core scenario workflows (`bilancio validate` / `bilancio run`) are expected to work with the
`core` profile alone. Optional analysis and cloud commands should give clear guidance when an
extra is missing.

## Development Workflow

### Running Tests

```bash
# Run the normal non-benchmark suite
uv run pytest tests/ --ignore=tests/benchmark -m "not slow" -v

# Run with coverage
uv run pytest tests/ --cov=bilancio --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_balances.py -v
```

### Quality Checks

PR CI enforces:

- Ruff on the current clean CLI/sweep surface
- mypy on the same CLI/sweep surface
- the full normal non-benchmark test suite

Run the same commands locally before pushing:

```bash
scripts/check_quality.sh
```

Equivalent manual commands:

```bash
PYTHONPATH=src python -m ruff check \
  src/bilancio/ui/cli \
  src/bilancio/ui/sweep_setup.py \
  tests/test_smoke.py \
  tests/ui/test_cli.py \
  tests/ui/test_cli_integration.py \
  tests/ui/test_cli_run_coverage.py \
  tests/ui/test_cli_sweep.py \
  tests/ui/test_sweep_setup.py
PYTHONPATH=src python -m ruff format --check \
  src/bilancio/ui/cli \
  src/bilancio/ui/sweep_setup.py \
  tests/test_smoke.py \
  tests/ui/test_cli.py \
  tests/ui/test_cli_integration.py \
  tests/ui/test_cli_run_coverage.py \
  tests/ui/test_cli_sweep.py \
  tests/ui/test_sweep_setup.py
PYTHONPATH=src python -m mypy src/bilancio/ui/cli src/bilancio/ui/sweep_setup.py
PYTHONPATH=src python -m pytest tests/ --ignore=tests/benchmark --no-cov -m "not slow"
```

Ruff and mypy are enforced on the clean CLI/sweep surface first, while the rest of the repo is
cleaned incrementally. Expand that scope only when the next target stays green by default.

### Pre-commit

Install and run the shared hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and single-purpose
- Use `ruff format` as the canonical formatter

### Project Structure

```
bilancio/
├── src/bilancio/
│   ├── core/           # Core data structures and utilities
│   ├── domain/         # Domain models (agents, instruments)
│   ├── ops/            # Operations on domain objects
│   ├── engines/        # Computation engines
│   ├── analysis/       # Analysis & analytics tools
│   ├── config/         # Configuration and scenario loading
│   ├── dealer/         # Dealer pricing and trading logic
│   └── ui/             # CLI and visualization
├── tests/              # Test suites
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── examples/           # Example scenarios and notebooks
└── docs/               # Documentation
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure tests pass:
   ```bash
   uv run pytest tests/ -v
   ```

3. **Commit with clear messages:**
   ```bash
   git commit -m "feat: add new feature description"
   ```

4. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### What to Include in a PR

- Clear description of changes
- Test coverage for new functionality
- Updated documentation if applicable
- Reference to related issues (if any)

## Reporting Issues

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (Python version, OS, etc.)
6. **Relevant code or error messages**

## Architecture Guidelines

### Double-Entry Invariants

Bilancio enforces strict double-entry bookkeeping. All financial instruments must:
- Have exactly one asset holder and one liability issuer
- Balance at the system level (total assets = total liabilities)

### Agent Types

When adding new agent types:
- Inherit from the base `Agent` class
- Register in `bilancio/domain/agents/__init__.py`
- Update `PolicyEngine` if new holding/issuing rules needed
- Add tests for agent-specific behavior

### Instrument Types

When adding new instruments:
- Inherit from the base `Instrument` class
- Implement `validate_type_invariants()` method
- Register in appropriate module under `bilancio/domain/instruments/`
- Update policy rules if needed

## Questions?

If you have questions about contributing, please open an issue with the "question" label.

Thank you for contributing to Bilancio!
