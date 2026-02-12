"""Lightweight loaders for analytics inputs (events JSONL, balances CSV).

Stdlib only; keeps parsing minimal and robust to schema changes.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def _to_decimal(val: object) -> Decimal:
    """Best-effort Decimal conversion for numbers represented as int/float/str."""
    if val is None:
        return Decimal("0")
    if isinstance(val, Decimal):
        return val
    # Normalize bools to Decimal 0/1
    if isinstance(val, bool):
        return Decimal(int(val))
    # JSONL may encode Decimals as numbers or strings
    try:
        return Decimal(str(val))
    except (ValueError, ArithmeticError, TypeError):
        return Decimal("0")


def read_events_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    """Yield events (dict) from a JSONL file in recorded order.

    Ensures numeric fields like amount/day/due_day are normalized to Python types
    (Decimal for amounts, int for day counters when available).
    """
    p = Path(path)
    with p.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            evt = json.loads(line)
            # Normalize common fields if present
            if "amount" in evt:
                evt["amount"] = _to_decimal(evt["amount"])
            if "day" in evt and evt["day"] is not None:
                try:
                    evt["day"] = int(evt["day"])  # day indices are integers in the engine
                except (ValueError, TypeError):
                    pass
            if "due_day" in evt and evt["due_day"] is not None:
                try:
                    evt["due_day"] = int(evt["due_day"])  # due_day recorded on creation
                except (ValueError, TypeError):
                    pass
            yield evt


def read_balances_csv(path: Path | str) -> List[Dict[str, Any]]:
    """Read balances CSV produced by export.writers.write_balances_csv.

    Returns a list of dict rows. Numeric fields remain as strings unless parsed explicitly
    by downstream code. Rows with ad-hoc summary fields (e.g., item_type) are preserved.
    """
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

