"""Tests for T-account row builder and maturity parsing."""

from decimal import Decimal

from bilancio.analysis.visualization import build_t_account_rows, parse_day_from_maturity
from bilancio.config.apply import create_agent
from bilancio.engines.system import System


def test_parse_day_from_maturity_helper():
    assert parse_day_from_maturity("Day 10") == 10
    # Unknown or malformed sorts to end (inf)
    assert parse_day_from_maturity("Soon") > 10**9
    assert parse_day_from_maturity(None) > 10**9
    assert parse_day_from_maturity("Day X") > 10**9


def test_build_t_account_rows_sorting_by_maturity():
    sys = System()
    # Two agents with reciprocal delivery obligations at different days
    f1 = create_agent(type("Spec", (), {"id": "F1", "kind": "firm", "name": "Firm One"}))
    f2 = create_agent(type("Spec", (), {"id": "F2", "kind": "firm", "name": "Firm Two"}))
    sys.add_agent(f1)
    sys.add_agent(f2)

    # F1 owes F2: 10 items due Day 2
    sys.create_delivery_obligation(
        "F1", "F2", sku="ITEM", quantity=10, unit_price=Decimal("5"), due_day=2
    )
    # F2 owes F1: 5 items due Day 1
    sys.create_delivery_obligation(
        "F2", "F1", sku="ITEM", quantity=5, unit_price=Decimal("5"), due_day=1
    )

    acct_f1 = build_t_account_rows(sys, "F1")
    # For F1: assets should include receivable from F2 due Day 1 before inventory/financials; liabilities Day 2 obligation
    asset_names = [r.name for r in acct_f1.assets]
    liab_names = [r.name for r in acct_f1.liabilities]
    # Receivable row must be present
    assert any(n.endswith("receivable") for n in asset_names)
    # Obligation row must be present
    assert any(n.endswith("obligation") for n in liab_names)

    # Sorting by maturity: for obligations, Day 2 sorts after Day 1; for assets, receivable is a group keyed by day
    # Build keys using the same helper
    asset_due_days = [
        parse_day_from_maturity(r.maturity) for r in acct_f1.assets if r.name.endswith("receivable")
    ]
    assert asset_due_days == sorted(asset_due_days)
