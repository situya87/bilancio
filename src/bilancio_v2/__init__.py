"""Bilancio v2: an event-journaled, plugin-based rebuild of the simulation kernel.

Public API::

    from bilancio_v2 import load_scenario, run_scenario

    result = run_scenario(load_scenario("examples/scenarios/payment_demo.yaml"))
    result.events          # legacy-compatible event dicts
    result.ledger.cash     # final balances

Scenarios use the existing YAML schema (``bilancio.config``). The kernel
currently covers the payment/settlement core (cash, reserves, deposits,
payables, deliveries, CB loans, interbank netting, defaults with expulsion
and pro-rata recovery); dealer/lender/rating/banking behavior plugins are
rejected explicitly until rebuilt. Observable behavior is verified
event-for-event against the existing engine by ``tests/v2/test_parity.py``.
"""

from __future__ import annotations

from pathlib import Path

from bilancio.config.loaders import load_yaml
from bilancio.config.models import ScenarioConfig
from bilancio_v2.engine import (
    RunResult,
    Runtime,
    UnsupportedScenarioError,
    prepare_scenario,
    run_day,
    run_scenario,
    run_until_stable,
)
from bilancio_v2.ledger import InvariantViolation, Ledger

__all__ = [
    "InvariantViolation",
    "Ledger",
    "RunResult",
    "Runtime",
    "UnsupportedScenarioError",
    "load_scenario",
    "prepare_scenario",
    "run_day",
    "run_scenario",
    "run_until_stable",
]


def load_scenario(path: Path | str) -> ScenarioConfig:
    """Load and validate a scenario YAML using the shared config schema."""
    return load_yaml(path)
