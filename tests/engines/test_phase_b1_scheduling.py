from pathlib import Path

from bilancio.config.apply import apply_to_system
from bilancio.config.loaders import load_yaml
from bilancio.engines.simulation import run_day
from bilancio.engines.system import System

SCENARIO = """
version: 1
name: B1-before-B2
agents:
  - {id: CB, kind: central_bank, name: CB}
  - {id: F1, kind: firm, name: F1}
  - {id: F2, kind: firm, name: F2}
initial_actions:
  - {mint_cash: {to: F1, amount: 100}}
  - {create_stock: {owner: F2, sku: X, quantity: 1, unit_price: "100"}}
run: {mode: until_stable, max_days: 5}
"""


def test_b1_executes_before_b2(tmp_path: Path):
    # Create config file
    p = tmp_path / "s.yaml"
    p.write_text(SCENARIO)
    cfg = load_yaml(p)
    sys = System()
    apply_to_system(cfg, sys)

    # Schedule actions for current day (0): create obligations due today
    sys.state.scheduled_actions_by_day[0] = [
        {
            "create_delivery_obligation": {
                "from": "F2",
                "to": "F1",
                "sku": "X",
                "quantity": 1,
                "unit_price": "100",
                "due_day": 0,
            }
        },
        {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 0}},
    ]

    # Run day 0 and verify obligations were settled same day
    run_day(sys)
    # No open obligations should remain
    assert not any(
        c.kind in ("payable", "delivery_obligation") for c in sys.state.contracts.values()
    )
    # And events should include settled events for day 0
    day0_events = [e for e in sys.state.events if e.get("day") == 0]
    kinds = {e.get("kind") for e in day0_events}
    assert "DeliveryObligationSettled" in kinds
    assert "PayableSettled" in kinds
