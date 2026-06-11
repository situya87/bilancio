"""Invariant checks for clean-core balance exports."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bilancio.core.errors import ValidationError


def assert_clean_core_invariants(rows: list[dict[str, Any]]) -> None:
    """Check core accounting invariants for clean-core balance rows."""
    system_row = next((row for row in rows if row.get("agent_id") == "SYSTEM"), None)
    if system_row is None:
        raise ValidationError("clean-core invariant failed: missing SYSTEM balance row")

    assets = Decimal(str(system_row.get("total_financial_assets", 0)))
    liabilities = Decimal(str(system_row.get("total_financial_liabilities", 0)))
    if assets != liabilities:
        raise ValidationError(
            f"clean-core invariant failed: system assets {assets} != liabilities {liabilities}"
        )

    for row in rows:
        for key, value in row.items():
            if key == "net_financial" or key.startswith("liabilities_"):
                continue
            if key.startswith(("assets_", "inventory_", "nonfinancial_")):
                amount = Decimal(str(value))
                if amount < 0:
                    agent_id = row.get("agent_id", "<unknown>")
                    raise ValidationError(
                        f"clean-core invariant failed: negative {key} for {agent_id}: {amount}"
                    )
