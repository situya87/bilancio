#!/usr/bin/env python3
"""Lint script to detect common test anti-patterns.

Scans all Python files under tests/ for:
  1. Tautological assertions: `or True` in assert statements
  2. Over-broad exception catches: `pytest.raises(Exception)`
  3. Bare exception swallowing: `except Exception: pass`

Exits non-zero if any violations are found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _check_or_true(node: ast.Assert, filepath: str, violations: list[str]) -> None:
    """Detect `assert ... or True` patterns."""
    test = node.test
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for val in test.values:
            if isinstance(val, ast.Constant) and val.value is True:
                violations.append(
                    f"{filepath}:{node.lineno}: tautological assertion "
                    f"('or True' in assert)"
                )
                return


def _check_pytest_raises_exception(
    node: ast.With, filepath: str, violations: list[str]
) -> None:
    """Detect `pytest.raises(Exception)` (over-broad catch)."""
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        # Match pytest.raises(...)
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "raises":
            if isinstance(func.value, ast.Name) and func.value.id == "pytest":
                for arg in call.args:
                    if isinstance(arg, ast.Name) and arg.id == "Exception":
                        violations.append(
                            f"{filepath}:{node.lineno}: over-broad exception catch "
                            f"(pytest.raises(Exception))"
                        )


def _check_bare_except_pass(
    node: ast.ExceptHandler, filepath: str, violations: list[str]
) -> None:
    """Detect `except Exception: pass` (bare exception swallowing)."""
    if node.type is None:
        # bare `except:` without a type
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            violations.append(
                f"{filepath}:{node.lineno}: bare 'except: pass' swallows all errors"
            )
        return

    if isinstance(node.type, ast.Name) and node.type.id == "Exception":
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            violations.append(
                f"{filepath}:{node.lineno}: bare 'except Exception: pass' "
                f"swallows all errors"
            )


def scan_file(filepath: Path) -> list[str]:
    """Scan a single Python file for anti-patterns."""
    violations: list[str] = []
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            _check_or_true(node, str(filepath), violations)
        elif isinstance(node, ast.With):
            _check_pytest_raises_exception(node, str(filepath), violations)
        elif isinstance(node, ast.ExceptHandler):
            _check_bare_except_pass(node, str(filepath), violations)

    return violations


def main() -> int:
    """Scan all test files and report violations."""
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    if not tests_dir.is_dir():
        print(f"ERROR: tests directory not found at {tests_dir}", file=sys.stderr)
        return 1

    all_violations: list[str] = []
    for pyfile in sorted(tests_dir.rglob("*.py")):
        all_violations.extend(scan_file(pyfile))

    if all_violations:
        print(f"Found {len(all_violations)} test anti-pattern(s):\n")
        for v in all_violations:
            print(f"  {v}")
        print()
        return 1

    print("No test anti-patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
