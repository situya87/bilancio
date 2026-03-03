"""Lightweight loaders for analytics inputs (events JSONL, balances CSV,
dealer/bank state CSVs).

Stdlib only for core loaders; pandas used only for snapshot loaders.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


def read_events_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
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


def read_balances_csv(path: Path | str) -> list[dict[str, Any]]:
    """Read balances CSV produced by export.writers.write_balances_csv.

    Returns a list of dict rows. Numeric fields remain as strings unless parsed explicitly
    by downstream code. Rows with ad-hoc summary fields (e.g., item_type) are preserved.
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_csv(run_dir: Path, filename: str) -> Path | None:
    """Locate a CSV file in run_dir or run_dir/out/."""
    for candidate in [run_dir / filename, run_dir / "out" / filename]:
        if candidate.exists():
            return candidate
    return None


def load_dealer_snapshots(run_dir: Path | str) -> Any:
    """Load dealer_state.csv from a run directory as a pandas DataFrame.

    Returns None if the file is not found or pandas is not available.
    """
    run_dir = Path(run_dir)
    csv_path = _find_csv(run_dir, "dealer_state.csv")
    if csv_path is None:
        return None
    try:
        import pandas as pd
        return pd.read_csv(csv_path)
    except Exception as exc:
        logger.debug("Failed to load dealer_state.csv: %s", exc)
        return None


def load_bank_snapshots(run_dir: Path | str) -> Any:
    """Load bank_state.csv from a run directory as a pandas DataFrame.

    Returns None if the file is not found or pandas is not available.
    """
    run_dir = Path(run_dir)
    csv_path = _find_csv(run_dir, "bank_state.csv")
    if csv_path is None:
        return None
    try:
        import pandas as pd
        return pd.read_csv(csv_path)
    except Exception as exc:
        logger.debug("Failed to load bank_state.csv: %s", exc)
        return None
