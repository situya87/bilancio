"""Deterministic parity scenarios for the rating agency slice.

Covers interactions the random corpus is unlikely to hit: stock inventory
and delivery obligations feeding the rating balance-sheet score, the
realistic (seeded-noise) information profile, and the lender pricing off
the published rating registry (signal-based mode, no kappa).
"""

from __future__ import annotations

import pytest

from bilancio.config.models import ScenarioConfig
from bilancio_v2.parity import compare_runs


def rating_scenario(info_profile: str) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "version": 1,
            "name": f"rating-{info_profile}",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "RATER", "kind": "rating_agency", "name": "Rater"},
                {"id": "LENDER", "kind": "non_bank_lender", "name": "Lender"},
                {"id": "F1", "kind": "firm", "name": "Firm 1"},
                {"id": "F2", "kind": "firm", "name": "Firm 2"},
                {"id": "H1", "kind": "household", "name": "House 1"},
            ],
            "initial_actions": [
                {"mint_cash": {"to": "LENDER", "amount": 400}},
                {"mint_cash": {"to": "F1", "amount": 80}},
                {"mint_cash": {"to": "F2", "amount": 120}},
                {"mint_cash": {"to": "H1", "amount": 50}},
                # Stock + delivery obligations enter the rating balance sheet.
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 10, "unit_price": 25}},
                {
                    "create_delivery_obligation": {
                        "from": "F1",
                        "to": "F2",
                        "sku": "WIDGET",
                        "quantity": 4,
                        "unit_price": 25,
                        "due_day": 2,
                    }
                },
                {"create_payable": {"from": "F1", "to": "F2", "amount": 150, "due_day": 2}},
                {"create_payable": {"from": "F2", "to": "H1", "amount": 200, "due_day": 3}},
                {"create_payable": {"from": "H1", "to": "F1", "amount": 180, "due_day": 3}},
            ],
            "rating_agency": {
                "enabled": True,
                "info_profile": info_profile,
                "coverage_fraction": "1.0",
            },
            # Signal-based lender (no kappa): prices off the rating registry.
            "lender": {"enabled": True, "maturity_days": 2, "horizon": 3},
            "run": {"max_days": 10, "quiet_days": 2, "default_handling": "expel-agent"},
        }
    )


@pytest.mark.parametrize("info_profile", ["omniscient", "realistic"])
def test_rating_with_inventory_runs_identically(info_profile: str) -> None:
    report = compare_runs(rating_scenario(info_profile))
    assert report.ok, "\n".join(report.diffs)
