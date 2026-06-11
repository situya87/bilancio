"""Characterization tests for public scenario CLI behavior.

These tests intentionally pin the observable scenario contract used by the
examples: CLI invocation, exported event stream, and final balance CSV values.
They are a parity harness for future core rewrites.

Test intent:
- Keep the public `bilancio run` CLI compatible with existing scenario YAML.
- Verify clean-core, legacy, and auto engine routing preserve exported artifacts.
- Catch user-visible regressions in CSV, JSONL, HTML, display, and invariant
  behavior before scenario changes reach users.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.scenarios.ring.compiler import (
    _to_yaml_ready,
    compile_ring_explorer_balanced,
)
from bilancio.ui.cli import cli

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "scenarios"
EXERCISE_SCENARIOS_DIR = PROJECT_ROOT / "examples" / "exercise_scenarios" / "yaml"
KALECKI_SCENARIOS_DIR = PROJECT_ROOT / "examples" / "kalecki"
TRACKED_EXAMPLE_SCENARIOS = (
    "default_handling_demo.yaml",
    "firm_delivery.yaml",
    "interbank_netting.yaml",
    "intraday_netting.yaml",
    "kalecki_with_dealer.yaml",
    "payment_demo.yaml",
    "rich_simulation.yaml",
    "ring_with_action_specs.yaml",
    "sasa_scenario.yaml",
    "simple_bank.yaml",
    "simple_dealer.yaml",
    "simple_nbfi.yaml",
    "two_banks_interbank.yaml",
    "two_jurisdictions.yaml",
)
TRACKED_EXERCISE_SCENARIOS = (
    "ex1_cash_for_goods.yaml",
    "ex2_two_firms_cash_purchase.yaml",
    "ex3_iou_assignment.yaml",
    "ex4_generic_claim_transfer.yaml",
    "ex5_deferred_exchange.yaml",
    "ex6_goods_now_cash_later.yaml",
    "ex7_cash_now_goods_later.yaml",
)
TRACKED_KALECKI_SCENARIOS = ("kalecki_ring_baseline.yaml",)


def _run_scenario_with_exports(
    tmp_path: Path,
    scenario_name: str,
    *,
    max_days: int = 5,
    engine: str | None = "legacy",
    scenarios_dir: Path = EXAMPLES_DIR,
    require_stable: bool = True,
) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, str]]]:
    engine_label = engine or "default"
    balances_path = tmp_path / f"{engine_label}.{scenario_name}.balances.csv"
    events_path = tmp_path / f"{engine_label}.{scenario_name}.events.jsonl"
    args = [
        "run",
        str(scenarios_dir / scenario_name),
        "--max-days",
        str(max_days),
        "--quiet-days",
        "2",
        "--check-invariants",
        "daily",
        "--show",
        "summary",
        "--export-balances",
        str(balances_path),
        "--export-events",
        str(events_path),
    ]
    if engine is not None:
        args[2:2] = ["--engine", engine]

    runner = CliRunner()
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, result.output
    if require_stable:
        assert "OK System reached stable state" in result.output
    assert balances_path.exists()
    assert events_path.exists()

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    balance_rows = _read_balance_export(balances_path)

    return result.output, events, balance_rows


def _read_balance_export(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["agent_id"]: row for row in csv.DictReader(f) if row["agent_id"] != "SYSTEM"}


def _assert_balance_exports_match(
    legacy_balances: dict[str, dict[str, str]],
    clean_balances: dict[str, dict[str, str]],
) -> None:
    assert clean_balances.keys() == legacy_balances.keys()
    for agent_id, legacy_row in legacy_balances.items():
        clean_row = clean_balances[agent_id]
        for field in set(legacy_row) | set(clean_row):
            if field == "agent_id":
                continue
            assert _amount(clean_row, field) == _amount(legacy_row, field), (agent_id, field)


def _amount(row: dict[str, str], field: str) -> Decimal:
    value = row.get(field, "")
    return Decimal(value or "0")


def _event_exists(events: list[dict[str, Any]], **expected: Any) -> bool:
    return any(all(event.get(key) == value for key, value in expected.items()) for event in events)


def _active_dealer_action_specs_scenario(
    *,
    mode: str = "active",
    n_banks: int = 0,
) -> dict[str, Any]:
    config = RingExplorerGeneratorConfig.model_validate(
        {
            "version": 1,
            "generator": "ring_explorer_v1",
            "name_prefix": "Active Dealer Characterization",
            "params": {
                "n_agents": 3,
                "seed": 123,
                "kappa": "0.5",
                "Q_total": "300",
                "liquidity": {"allocation": {"mode": "uniform"}},
                "inequality": {
                    "scheme": "dirichlet",
                    "concentration": "1.0",
                    "monotonicity": "0",
                },
                "maturity": {"days": 2, "mode": "lead_lag", "mu": "0"},
            },
            "compile": {"emit_yaml": False},
        }
    )
    return _to_yaml_ready(
        compile_ring_explorer_balanced(
            config,
            mode=mode,
            n_banks=n_banks,
            emit_action_specs=True,
            kappa=Decimal("0.5"),
        )
    )


def _active_dealer_generated_combo_scenario(
    *,
    generated_mode: str,
    n_banks: int,
    enable_bank_lending: bool,
    enable_lender: bool,
) -> dict[str, Any]:
    agents: list[dict[str, str]] = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
        {"id": "H1", "kind": "household", "name": "H1"},
        {"id": "H2", "kind": "household", "name": "H2"},
        {"id": "dealer_short", "kind": "household", "name": "Dealer Short"},
        {"id": "vbt_short", "kind": "household", "name": "VBT Short"},
    ]
    initial_actions: list[dict[str, dict[str, Any]]] = [
        {"mint_cash": {"to": "H1", "amount": 20}},
        {"mint_cash": {"to": "H2", "amount": 20}},
        {"mint_cash": {"to": "dealer_short", "amount": 20}},
        {"mint_cash": {"to": "vbt_short", "amount": 20}},
    ]
    balanced_config: dict[str, Any] = {
        "mode": generated_mode,
        "n_banks": n_banks,
        "kappa": "1.0",
        "maturity_days": 5,
        "Q_total": 50,
        "enable_banking": n_banks > 0,
        "enable_bank_lending": enable_bank_lending,
        "trader_bank_assignments": {"H1": ["B1", "B2"], "H2": ["B2", "B1"]},
    }
    if n_banks:
        agents[1:1] = [
            {"id": "B1", "kind": "bank", "name": "Bank One"},
            {"id": "B2", "kind": "bank", "name": "Bank Two"},
        ]
        initial_actions.extend(
            [
                {"mint_reserves": {"to": "B1", "amount": 1000}},
                {"mint_reserves": {"to": "B2", "amount": 1000}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 20}},
                {"deposit_cash": {"customer": "H2", "bank": "B2", "amount": 20}},
            ]
        )
    if enable_lender:
        agents.append({"id": "lender", "kind": "non_bank_lender", "name": "Lender"})
        initial_actions.append({"mint_cash": {"to": "lender", "amount": 100}})
        balanced_config["enable_lender"] = True

    initial_actions.extend(
        [
            {
                "create_payable": {
                    "from": "H1",
                    "to": "H2",
                    "amount": 35,
                    "due_day": 2,
                    "maturity_distance": 2,
                }
            },
            {
                "create_payable": {
                    "from": "H2",
                    "to": "H1",
                    "amount": 15,
                    "due_day": 1,
                    "maturity_distance": 1,
                }
            },
        ]
    )

    scenario: dict[str, Any] = {
        "version": 1,
        "name": f"Active Dealer {generated_mode}",
        "agents": agents,
        "initial_actions": initial_actions,
        "dealer": {
            "enabled": True,
            "ticket_size": 1,
            "buckets": {
                "short": {"tau_min": 1, "tau_max": 3, "M": "1.0", "O": "0.20"},
                "mid": {"tau_min": 4, "tau_max": 8, "M": "1.0", "O": "0.30"},
                "long": {"tau_min": 9, "tau_max": 999, "M": "1.0", "O": "0.40"},
            },
            "dealer_share": "0.05",
            "vbt_share": "0.20",
        },
        "balanced_dealer": {
            "enabled": True,
            "mode": "active",
            "face_value": 1,
            "outside_mid_ratio": "0.75",
            "rollover_enabled": False,
        },
        "_balanced_config": balanced_config,
        "run": {
            "mode": "until_stable",
            "default_handling": "fail-fast",
            "rollover_enabled": False,
            "quiet_days": 1,
            "show": {"events": "summary"},
        },
    }
    if enable_lender:
        scenario["lender"] = {
            "enabled": True,
            "base_rate": "0.05",
            "risk_premium_scale": "0.20",
            "max_single_exposure": "1.0",
            "max_total_exposure": "1.0",
            "maturity_days": 3,
            "kappa": "1.0",
            "min_coverage_ratio": "0",
        }
    return scenario


def _normalize_unstable_contract_ids(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for event in events:
        item = dict(event)
        if "loan_id" in item:
            item["loan_id"] = "<loan>"
        if "reserve_id" in item:
            item["reserve_id"] = "<reserve>"
        if "instr_id" in item:
            item["instr_id"] = "<instrument>"
        if "contract_id" in item:
            item["contract_id"] = "<contract>"
        normalized.append(item)
    return normalized


UNSTABLE_EVENT_ID_FIELDS = {
    "cash_piece_ids",
    "contract_id",
    "deposit_id",
    "id",
    "instr_id",
    "keep",
    "loan_id",
    "new_id",
    "new_payable",
    "obligation_id",
    "old_payable",
    "original_id",
    "payable_id",
    "pid",
    "removed",
    "reserve_id",
    "stock_id",
    "ticket_id",
    "trigger_contract",
}


def _strip_unstable_event_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_unstable_event_fields(item)
            for key, item in value.items()
            if key not in UNSTABLE_EVENT_ID_FIELDS
        }
    if isinstance(value, list):
        return [_strip_unstable_event_fields(item) for item in value]
    return value


def _assert_event_exports_match(
    legacy_events: list[dict[str, Any]],
    clean_events: list[dict[str, Any]],
    *,
    ordered: bool = True,
) -> None:
    clean_normalized = [_strip_unstable_event_fields(event) for event in clean_events]
    legacy_normalized = [_strip_unstable_event_fields(event) for event in legacy_events]
    if ordered:
        assert clean_normalized == legacy_normalized
        return

    def signature(event: dict[str, Any]) -> str:
        return json.dumps(event, sort_keys=True)

    assert Counter(signature(event) for event in clean_normalized) == Counter(
        signature(event) for event in legacy_normalized
    )


def _cash_fragment_normalized_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    pending_cash_transfer: dict[str, Any] | None = None
    pending_cash_key: tuple[Any, ...] | None = None

    def flush_pending() -> None:
        nonlocal pending_cash_transfer, pending_cash_key
        if pending_cash_transfer is not None:
            normalized.append(pending_cash_transfer)
            pending_cash_transfer = None
            pending_cash_key = None

    for raw_event in events:
        event = _strip_unstable_event_fields(raw_event)
        if event.get("kind") == "InstrumentMerged":
            continue
        if event.get("kind") != "CashTransferred":
            flush_pending()
            normalized.append(event)
            continue

        cash_key = (
            event.get("day"),
            event.get("phase"),
            event.get("frm"),
            event.get("to"),
        )
        if pending_cash_transfer is None or pending_cash_key != cash_key:
            flush_pending()
            pending_cash_transfer = dict(event)
            pending_cash_key = cash_key
            continue

        amount = Decimal(str(pending_cash_transfer["amount"])) + Decimal(str(event["amount"]))
        pending_cash_transfer["amount"] = int(amount) if amount == amount.to_integral_value() else str(amount)

    flush_pending()
    return normalized


def test_documented_module_entrypoint_validates_scenario() -> None:
    """The documented ``python -m bilancio.ui.cli`` invocation remains supported."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bilancio.ui.cli",
            "validate",
            str(EXAMPLES_DIR / "sasa_scenario.yaml"),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Configuration is valid" in result.stdout


@pytest.mark.parametrize("scenario_name", TRACKED_EXAMPLE_SCENARIOS)
def test_tracked_example_scenarios_validate(scenario_name: str) -> None:
    """Every checked-in example scenario remains loadable and applicable."""
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(EXAMPLES_DIR / scenario_name)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


@pytest.mark.parametrize("scenario_name", ("sasa_scenario.yaml", "firm_delivery.yaml"))
def test_clean_core_cli_engine_matches_legacy_exports(tmp_path: Path, scenario_name: str) -> None:
    """The opt-in clean-core CLI path preserves event payloads and balance exports."""
    _, legacy_events, legacy_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=8,
        engine="legacy",
    )
    output, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=8,
        engine="clean-core",
    )

    assert "Engine: clean-core" in output
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(legacy_balances, clean_balances)


@pytest.mark.parametrize("scenario_name", ("simple_dealer.yaml", "kalecki_with_dealer.yaml"))
def test_clean_core_cli_engine_matches_legacy_exports_for_defaulting_generator_examples(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    _, legacy_events, legacy_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=10,
        engine="legacy",
        require_stable=False,
    )
    output, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=10,
        engine="clean-core",
        require_stable=False,
    )

    assert "Engine: clean-core" in output
    assert "Insufficient funds to settle payable" in output
    assert "Simulation stopped after an error" in output
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(legacy_balances, clean_balances)


def test_clean_core_cli_engine_matches_legacy_claim_transfer_exports(tmp_path: Path) -> None:
    """The opt-in clean-core CLI path preserves claim-transfer scenario exports."""
    _, legacy_events, legacy_balances = _run_scenario_with_exports(
        tmp_path,
        "ex3_iou_assignment.yaml",
        max_days=8,
        engine="legacy",
        scenarios_dir=EXERCISE_SCENARIOS_DIR,
    )
    output, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        "ex3_iou_assignment.yaml",
        max_days=8,
        engine="clean-core",
        scenarios_dir=EXERCISE_SCENARIOS_DIR,
    )

    assert "Engine: clean-core" in output
    assert "ClaimTransferred" in {event["kind"] for event in clean_events}
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(legacy_balances, clean_balances)


def test_clean_core_cli_engine_matches_legacy_nbfi_exports(tmp_path: Path) -> None:
    """The opt-in clean-core CLI path preserves the cash-only NBFI example."""
    _, legacy_events, legacy_balances = _run_scenario_with_exports(
        tmp_path,
        "simple_nbfi.yaml",
        max_days=10,
        engine="legacy",
    )
    output, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        "simple_nbfi.yaml",
        max_days=10,
        engine="clean-core",
    )

    assert "Engine: clean-core" in output
    assert "NonBankLoanCreated" in {event["kind"] for event in clean_events}
    _assert_event_exports_match(legacy_events, clean_events, ordered=False)
    _assert_balance_exports_match(legacy_balances, clean_balances)


@pytest.mark.parametrize("scenario_name", TRACKED_EXERCISE_SCENARIOS)
def test_clean_core_cli_engine_matches_legacy_exercise_exports(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    _, legacy_events, legacy_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=10,
        engine="legacy",
        scenarios_dir=EXERCISE_SCENARIOS_DIR,
    )
    output, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=10,
        engine="clean-core",
        scenarios_dir=EXERCISE_SCENARIOS_DIR,
    )

    assert "Engine: clean-core" in output
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(legacy_balances, clean_balances)


@pytest.mark.parametrize("scenario_name", TRACKED_KALECKI_SCENARIOS)
def test_clean_core_cli_engine_matches_legacy_kalecki_exports(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    _, legacy_events, legacy_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=10,
        engine="legacy",
        scenarios_dir=KALECKI_SCENARIOS_DIR,
    )
    output, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        scenario_name,
        max_days=10,
        engine="clean-core",
        scenarios_dir=KALECKI_SCENARIOS_DIR,
    )

    assert "Engine: clean-core" in output
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(legacy_balances, clean_balances)


def test_auto_engine_uses_clean_core_for_supported_scenario(tmp_path: Path) -> None:
    output, auto_events, auto_balances = _run_scenario_with_exports(
        tmp_path,
        "simple_nbfi.yaml",
        max_days=10,
        engine="auto",
    )
    _, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        "simple_nbfi.yaml",
        max_days=10,
        engine="clean-core",
    )

    assert "Engine: clean-core" in output
    _assert_event_exports_match(clean_events, auto_events)
    _assert_balance_exports_match(clean_balances, auto_balances)


def test_default_cli_engine_uses_auto_clean_core_for_supported_scenario(tmp_path: Path) -> None:
    output, default_events, default_balances = _run_scenario_with_exports(
        tmp_path,
        "simple_nbfi.yaml",
        max_days=10,
        engine=None,
    )
    _, clean_events, clean_balances = _run_scenario_with_exports(
        tmp_path,
        "simple_nbfi.yaml",
        max_days=10,
        engine="clean-core",
    )

    assert "Engine: clean-core" in output
    _assert_event_exports_match(clean_events, default_events)
    _assert_balance_exports_match(clean_balances, default_balances)


def test_auto_engine_uses_clean_core_for_direct_dealer_marker(tmp_path: Path) -> None:
    scenario_path = tmp_path / "dealer_direct.yaml"
    scenario_path.write_text(
        """
version: 1
name: Direct Dealer Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
dealer:
  enabled: true
run:
  mode: until_stable
  default_handling: fail-fast
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--agents",
            "H1,H2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    assert "SubphaseB_Dealer" in result.output


def test_default_cli_engine_uses_clean_core_for_direct_dealer_marker(tmp_path: Path) -> None:
    scenario_path = tmp_path / "dealer_direct_default.yaml"
    scenario_path.write_text(
        """
version: 1
name: Direct Dealer Default Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
dealer:
  enabled: true
run:
  mode: until_stable
  default_handling: fail-fast
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--agents",
            "H1,H2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    assert "SubphaseB_Dealer" in result.output


def test_auto_engine_uses_clean_core_for_generator_dealer_scenarios(tmp_path: Path) -> None:
    runner = CliRunner()
    legacy_events_path = tmp_path / "legacy_generator_dealer" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_generator_dealer" / "balances.csv"
    auto_events_path = tmp_path / "auto_generator_dealer" / "events.jsonl"
    auto_balances_path = tmp_path / "auto_generator_dealer" / "balances.csv"
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_dealer.yaml"),
            "--engine",
            "legacy",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_dealer.yaml"),
            "--engine",
            "auto",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(auto_events_path),
            "--export-balances",
            str(auto_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    auto_events = [json.loads(line) for line in auto_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, auto_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(auto_balances_path),
    )


def test_auto_engine_uses_clean_core_for_balanced_dealer_metadata(tmp_path: Path) -> None:
    scenario_path = tmp_path / "balanced_dealer_auto.yaml"
    scenario_path.write_text(
        """
version: 1
name: Balanced Dealer Auto Fallback
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
balanced_dealer:
  enabled: true
run:
  mode: until_stable
  default_handling: fail-fast
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy.events.jsonl"
    legacy_balances_path = tmp_path / "legacy.balances.csv"
    clean_events_path = tmp_path / "clean.events.jsonl"
    clean_balances_path = tmp_path / "clean.balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--agents",
            "H1,H2",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--agents",
            "H1,H2",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_clean_core_engine_supports_direct_dealer_marker_and_metrics(tmp_path: Path) -> None:
    scenario_path = tmp_path / "dealer_direct_clean_core.yaml"
    scenario_path.write_text(
        """
version: 1
name: Direct Dealer Clean Core Guard
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
dealer:
  enabled: true
run:
  mode: until_stable
  default_handling: fail-fast
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_events_path = tmp_path / "legacy" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy" / "balances.csv"
    clean_events_path = tmp_path / "clean" / "events.jsonl"
    clean_balances_path = tmp_path / "clean" / "balances.csv"
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    assert "SubphaseB_Dealer" in [event["kind"] for event in clean_events]
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 0
    assert "Traceback" not in result.output


def test_clean_core_engine_accepts_balanced_dealer_metadata(tmp_path: Path) -> None:
    scenario_path = tmp_path / "balanced_dealer_clean_core.yaml"
    scenario_path.write_text(
        """
version: 1
name: Balanced Dealer Clean Core Guard
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
balanced_dealer:
  enabled: true
run:
  mode: until_stable
  default_handling: fail-fast
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "OK System reached stable state" in result.output
    assert "Traceback" not in result.output


def test_clean_core_engine_matches_legacy_balanced_dealer_passive_metrics(tmp_path: Path) -> None:
    scenario_path = tmp_path / "balanced_dealer_passive.yaml"
    scenario_path.write_text(
        """
version: 1
name: Balanced Passive Direct
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
  - id: dealer_short
    kind: household
    name: Dealer Short
  - id: vbt_short
    kind: household
    name: VBT Short
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: dealer_short, amount: 25}
  - mint_cash: {to: vbt_short, amount: 50}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
  - create_payable: {from: H1, to: dealer_short, amount: 25, due_day: 1}
  - create_payable: {from: H1, to: vbt_short, amount: 50, due_day: 1}
dealer:
  enabled: true
  ticket_size: 1
  dealer_share: "0.05"
  vbt_share: "0.20"
balanced_dealer:
  enabled: true
  mode: passive
  face_value: 1
  outside_mid_ratio: "0.75"
  vbt_share_per_bucket: "0.20"
  dealer_share_per_bucket: "0.05"
  rollover_enabled: false
run:
  mode: until_stable
  default_handling: expel-agent
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_passive" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_passive" / "balances.csv"
    clean_events_path = tmp_path / "clean_passive" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_passive" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 0
    assert clean_metrics["dealer_total_pnl"] == 24.3625


def test_clean_core_engine_matches_legacy_active_dealer_action_specs(tmp_path: Path) -> None:
    scenario_path = tmp_path / "active_dealer_action_specs.yaml"
    scenario_path.write_text(
        yaml.safe_dump(_active_dealer_action_specs_scenario(), sort_keys=False),
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_active" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_active" / "balances.csv"
    clean_events_path = tmp_path / "clean_active" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_active" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 5
    assert clean_metrics["total_buy_trades"] == 5
    assert clean_metrics["interior_trades"] == 3
    assert clean_metrics["passthrough_trades"] == 2


def test_clean_core_engine_matches_legacy_action_specs_with_default_dealer_config(
    tmp_path: Path,
) -> None:
    scenario = _active_dealer_action_specs_scenario()
    scenario.pop("dealer", None)
    scenario_path = tmp_path / "active_dealer_action_specs_default_dealer.yaml"
    scenario_path.write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_default_dealer" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_default_dealer" / "balances.csv"
    clean_events_path = tmp_path / "clean_default_dealer" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_default_dealer" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 5


def test_clean_core_engine_matches_legacy_active_dealer_sell_trade(tmp_path: Path) -> None:
    scenario_path = tmp_path / "active_dealer_sell_trade.yaml"
    scenario_path.write_text(
        """
version: 1
name: Active Dealer Sell Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
  - id: dealer_short
    kind: household
    name: Dealer Short
  - id: vbt_short
    kind: household
    name: VBT Short
initial_actions:
  - mint_cash: {to: H1, amount: 20}
  - mint_cash: {to: dealer_short, amount: 20}
  - mint_cash: {to: vbt_short, amount: 20}
  - create_payable: {from: H1, to: H2, amount: 10, due_day: 2, maturity_distance: 2}
  - create_payable: {from: H2, to: H1, amount: 15, due_day: 1, maturity_distance: 1}
dealer:
  enabled: true
  ticket_size: 1
  buckets:
    short: {tau_min: 1, tau_max: 3, M: "1.0", O: "0.20"}
    mid: {tau_min: 4, tau_max: 8, M: "1.0", O: "0.30"}
    long: {tau_min: 9, tau_max: 999, M: "1.0", O: "0.40"}
  dealer_share: "0.05"
  vbt_share: "0.20"
balanced_dealer:
  enabled: true
  mode: active
  face_value: 1
  outside_mid_ratio: "0.75"
  rollover_enabled: false
action_specs:
  - kind: household
    profile_type: trader
    profile_params:
      buy_reserve_fraction: "1.0"
      trading_motive: liquidity_only
    actions:
      - {action: settle, phase: B2_Settlement}
      - {action: sell_ticket, phase: B_Dealer}
      - {action: buy_ticket, phase: B_Dealer}
run:
  mode: until_stable
  default_handling: fail-fast
  rollover_enabled: false
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_sell" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_sell" / "balances.csv"
    clean_events_path = tmp_path / "clean_sell" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_sell" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "1",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "1",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 1
    assert clean_metrics["total_sell_trades"] == 1
    assert clean_metrics["liquidity_driven_sales"] == 1


def test_clean_core_engine_matches_legacy_active_dealer_multiday_default_metrics(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "active_dealer_multiday_default.yaml"
    scenario_path.write_text(
        """
version: 1
name: Active Dealer Multi Day Default Metrics
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
  - id: dealer_short
    kind: household
    name: Dealer Short
  - id: vbt_short
    kind: household
    name: VBT Short
initial_actions:
  - mint_cash: {to: H1, amount: 20}
  - mint_cash: {to: dealer_short, amount: 20}
  - mint_cash: {to: vbt_short, amount: 20}
  - create_payable: {from: H1, to: H2, amount: 10, due_day: 2, maturity_distance: 2}
  - create_payable: {from: H2, to: H1, amount: 15, due_day: 1, maturity_distance: 1}
dealer:
  enabled: true
  ticket_size: 1
  buckets:
    short: {tau_min: 1, tau_max: 3, M: "1.0", O: "0.20"}
    mid: {tau_min: 4, tau_max: 8, M: "1.0", O: "0.30"}
    long: {tau_min: 9, tau_max: 999, M: "1.0", O: "0.40"}
  dealer_share: "0.05"
  vbt_share: "0.20"
balanced_dealer:
  enabled: true
  mode: active
  face_value: 1
  outside_mid_ratio: "0.75"
  rollover_enabled: false
action_specs:
  - kind: household
    profile_type: trader
    profile_params:
      buy_reserve_fraction: "1.0"
      trading_motive: liquidity_only
    actions:
      - {action: settle, phase: B2_Settlement}
      - {action: sell_ticket, phase: B_Dealer}
      - {action: buy_ticket, phase: B_Dealer}
run:
  mode: until_stable
  default_handling: expel-agent
  rollover_enabled: false
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_multiday_default" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_multiday_default" / "balances.csv"
    clean_events_path = tmp_path / "clean_multiday_default" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_multiday_default" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "2",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "3",
            "--quiet-days",
            "2",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    assert "ObligationDefaulted" in [event["kind"] for event in clean_events]
    assert "PayableSettled" in [event["kind"] for event in clean_events]
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["vbt_mid_final"] == {"long": 0.25, "mid": 0.25, "short": 0.25}


def test_clean_core_engine_matches_legacy_active_dealer_sell_passthrough(tmp_path: Path) -> None:
    scenario_path = tmp_path / "active_dealer_sell_passthrough.yaml"
    scenario_path.write_text(
        """
version: 1
name: Active Dealer Sell Passthrough Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
  - id: dealer_short
    kind: household
    name: Dealer Short
  - id: vbt_short
    kind: household
    name: VBT Short
initial_actions:
  - mint_cash: {to: H1, amount: 20}
  - mint_cash: {to: vbt_short, amount: 30}
  - create_payable: {from: H1, to: H2, amount: 10, due_day: 2, maturity_distance: 2}
  - create_payable: {from: H2, to: H1, amount: 15, due_day: 1, maturity_distance: 1}
  - create_payable: {from: H1, to: dealer_short, amount: 1, due_day: 2, maturity_distance: 2}
dealer:
  enabled: true
  ticket_size: 1
  buckets:
    short: {tau_min: 1, tau_max: 3, M: "1.0", O: "0.20"}
    mid: {tau_min: 4, tau_max: 8, M: "1.0", O: "0.30"}
    long: {tau_min: 9, tau_max: 999, M: "1.0", O: "0.40"}
  dealer_share: "0.05"
  vbt_share: "0.20"
balanced_dealer:
  enabled: true
  mode: active
  face_value: 1
  outside_mid_ratio: "0.75"
  rollover_enabled: false
action_specs:
  - kind: household
    profile_type: trader
    profile_params:
      buy_reserve_fraction: "1.0"
      trading_motive: liquidity_only
    actions:
      - {action: settle, phase: B2_Settlement}
      - {action: sell_ticket, phase: B_Dealer}
      - {action: buy_ticket, phase: B_Dealer}
run:
  mode: until_stable
  default_handling: fail-fast
  rollover_enabled: false
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_sell_passthrough" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_sell_passthrough" / "balances.csv"
    clean_events_path = tmp_path / "clean_sell_passthrough" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_sell_passthrough" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "1",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "1",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 2
    assert clean_metrics["total_sell_trades"] == 1
    assert clean_metrics["passthrough_trades"] == 1


def test_clean_core_engine_matches_legacy_direct_balanced_active_dealer(tmp_path: Path) -> None:
    scenario_path = tmp_path / "direct_balanced_active_dealer.yaml"
    scenario_path.write_text(
        """
version: 1
name: Direct Balanced Active Dealer Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
  - id: dealer_short
    kind: household
    name: Dealer Short
  - id: vbt_short
    kind: household
    name: VBT Short
initial_actions:
  - mint_cash: {to: H1, amount: 20}
  - mint_cash: {to: dealer_short, amount: 20}
  - mint_cash: {to: vbt_short, amount: 20}
  - create_payable: {from: H1, to: H2, amount: 10, due_day: 2, maturity_distance: 2}
  - create_payable: {from: H2, to: H1, amount: 15, due_day: 1, maturity_distance: 1}
dealer:
  enabled: true
  ticket_size: 1
  buckets:
    short: {tau_min: 1, tau_max: 3, M: "1.0", O: "0.20"}
    mid: {tau_min: 4, tau_max: 8, M: "1.0", O: "0.30"}
    long: {tau_min: 9, tau_max: 999, M: "1.0", O: "0.40"}
  dealer_share: "0.05"
  vbt_share: "0.20"
balanced_dealer:
  enabled: true
  mode: active
  face_value: 1
  outside_mid_ratio: "0.75"
  rollover_enabled: false
run:
  mode: until_stable
  default_handling: fail-fast
  rollover_enabled: false
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_direct_active" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_direct_active" / "balances.csv"
    clean_events_path = tmp_path / "clean_direct_active" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_direct_active" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "1",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "1",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics
    assert clean_metrics["total_trades"] == 2
    assert clean_metrics["total_sell_trades"] == 1
    assert clean_metrics["total_buy_trades"] == 1


def test_auto_engine_uses_clean_core_for_active_dealer_with_passive_banking(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "active_dealer_passive_banking.yaml"
    scenario_path.write_text(
        """
version: 1
name: Active Dealer Passive Banking
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: B2
    kind: bank
    name: Bank Two
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
  - id: dealer_short
    kind: household
    name: Dealer Short
  - id: vbt_short
    kind: household
    name: VBT Short
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_reserves: {to: B2, amount: 1000}
  - mint_cash: {to: H1, amount: 20}
  - mint_cash: {to: H2, amount: 20}
  - mint_cash: {to: dealer_short, amount: 20}
  - mint_cash: {to: vbt_short, amount: 20}
  - deposit_cash: {customer: H1, bank: B1, amount: 20}
  - deposit_cash: {customer: H2, bank: B2, amount: 20}
  - create_payable: {from: H1, to: H2, amount: 10, due_day: 2, maturity_distance: 2}
  - create_payable: {from: H2, to: H1, amount: 15, due_day: 1, maturity_distance: 1}
dealer:
  enabled: true
  ticket_size: 1
  buckets:
    short: {tau_min: 1, tau_max: 3, M: "1.0", O: "0.20"}
    mid: {tau_min: 4, tau_max: 8, M: "1.0", O: "0.30"}
    long: {tau_min: 9, tau_max: 999, M: "1.0", O: "0.40"}
  dealer_share: "0.05"
  vbt_share: "0.20"
balanced_dealer:
  enabled: true
  mode: active
  face_value: 1
  outside_mid_ratio: "0.75"
  rollover_enabled: false
_balanced_config:
  mode: active
  n_banks: 2
  kappa: "1.0"
  maturity_days: 5
  Q_total: 25
  enable_banking: true
  enable_bank_lending: false
  trader_bank_assignments:
    H1: [B1, B2]
    H2: [B2, B1]
run:
  mode: until_stable
  default_handling: fail-fast
  rollover_enabled: false
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_active_banking" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_active_banking" / "balances.csv"
    clean_events_path = tmp_path / "clean_active_banking" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_active_banking" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics


def test_auto_engine_uses_clean_core_for_active_dealer_with_bank_lending(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "active_dealer_bank_lending.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            _active_dealer_generated_combo_scenario(
                generated_mode="bank_dealer",
                n_banks=2,
                enable_bank_lending=True,
                enable_lender=False,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy_active_bank_lending" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_active_bank_lending" / "balances.csv"
    clean_events_path = tmp_path / "clean_active_bank_lending" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_active_bank_lending" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    clean_kinds = {event["kind"] for event in clean_events}
    _assert_event_exports_match(legacy_events, clean_events)
    assert "BankLoanIssued" in clean_kinds
    assert "BankLoanRepaid" in clean_kinds
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    clean_metrics = json.loads((clean_events_path.parent / "dealer_metrics.json").read_text())
    assert clean_metrics == legacy_metrics


@pytest.mark.parametrize(
    ("generated_mode", "n_banks", "enable_bank_lending", "enable_lender"),
    (
        ("active", 0, False, True),
        ("active", 2, True, False),
        ("active", 2, True, True),
        ("nbfi_dealer", 0, False, True),
        ("bank_dealer", 2, True, True),
        ("bank_dealer_nbfi", 2, True, True),
    ),
)
def test_auto_engine_uses_clean_core_for_active_dealer_with_nbfi(
    tmp_path: Path,
    generated_mode: str,
    n_banks: int,
    enable_bank_lending: bool,
    enable_lender: bool,
) -> None:
    scenario_path = tmp_path / f"active_dealer_{generated_mode}.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            _active_dealer_generated_combo_scenario(
                generated_mode=generated_mode,
                n_banks=n_banks,
                enable_bank_lending=enable_bank_lending,
                enable_lender=enable_lender,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / f"legacy_{generated_mode}" / "events.jsonl"
    legacy_balances_path = tmp_path / f"legacy_{generated_mode}" / "balances.csv"
    auto_events_path = tmp_path / f"auto_{generated_mode}" / "events.jsonl"
    auto_balances_path = tmp_path / f"auto_{generated_mode}" / "balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    auto_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(auto_events_path),
            "--export-balances",
            str(auto_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert auto_result.exit_code == 0, auto_result.output
    assert "Engine: clean-core" in auto_result.output
    assert "auto fallback" not in auto_result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    auto_events = [json.loads(line) for line in auto_events_path.read_text().splitlines()]
    auto_kinds = {event["kind"] for event in auto_events}
    assert _cash_fragment_normalized_events(auto_events) == _cash_fragment_normalized_events(
        legacy_events
    )
    if enable_lender:
        assert {"NonBankLoanCreated", "NonBankLoanRepaid"} <= auto_kinds
    if enable_bank_lending:
        assert {"BankLoanIssued", "BankLoanRepaid"} <= auto_kinds
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(auto_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    auto_metrics = json.loads((auto_events_path.parent / "dealer_metrics.json").read_text())
    assert auto_metrics == legacy_metrics


def test_auto_engine_uses_clean_core_for_generated_balanced_active_dealer(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_balanced_active_auto.yaml"
    scenario_path.write_text(
        yaml.safe_dump(_active_dealer_action_specs_scenario(), sort_keys=False),
        encoding="utf-8",
    )
    legacy_events_path = tmp_path / "generated_active_legacy" / "events.jsonl"
    legacy_balances_path = tmp_path / "generated_active_legacy" / "balances.csv"
    auto_events_path = tmp_path / "generated_active_auto" / "events.jsonl"
    auto_balances_path = tmp_path / "generated_active_auto" / "balances.csv"

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    auto_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(auto_events_path),
            "--export-balances",
            str(auto_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert auto_result.exit_code == 0, auto_result.output
    assert "Engine: clean-core" in auto_result.output
    assert "auto fallback" not in auto_result.output

    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    auto_events = [json.loads(line) for line in auto_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, auto_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(auto_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    auto_metrics = json.loads((auto_events_path.parent / "dealer_metrics.json").read_text())
    assert auto_metrics == legacy_metrics


@pytest.mark.parametrize(
    ("generated_mode", "n_banks"),
    (
        ("nbfi_dealer", 0),
        ("bank_dealer", 1),
        ("bank_dealer_nbfi", 1),
    ),
)
def test_auto_engine_uses_clean_core_for_generated_noop_dealer_action_specs(
    tmp_path: Path,
    generated_mode: str,
    n_banks: int,
) -> None:
    scenario_path = tmp_path / f"generated_{generated_mode}_auto.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            _active_dealer_action_specs_scenario(mode=generated_mode, n_banks=n_banks),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    legacy_events_path = tmp_path / f"generated_{generated_mode}_legacy" / "events.jsonl"
    legacy_balances_path = tmp_path / f"generated_{generated_mode}_legacy" / "balances.csv"
    auto_events_path = tmp_path / f"generated_{generated_mode}_auto" / "events.jsonl"
    auto_balances_path = tmp_path / f"generated_{generated_mode}_auto" / "balances.csv"

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    auto_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(auto_events_path),
            "--export-balances",
            str(auto_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert auto_result.exit_code == 0, auto_result.output
    assert "Engine: clean-core" in auto_result.output
    assert "auto fallback" not in auto_result.output

    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    auto_events = [json.loads(line) for line in auto_events_path.read_text().splitlines()]
    event_kinds = {event["kind"] for event in auto_events}
    _assert_event_exports_match(legacy_events, auto_events)
    assert "SubphaseB_Dealer" in event_kinds
    if "nbfi" in generated_mode:
        assert "SubphaseB_Lending" in event_kinds
    if "bank" in generated_mode:
        assert "SubphaseB_BankLending" in event_kinds
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(auto_balances_path),
    )
    legacy_metrics = json.loads((legacy_events_path.parent / "dealer_metrics.json").read_text())
    auto_metrics = json.loads((auto_events_path.parent / "dealer_metrics.json").read_text())
    assert auto_metrics == legacy_metrics


def test_clean_core_engine_rejects_dealer_action_specs_without_traceback(tmp_path: Path) -> None:
    scenario_path = tmp_path / "dealer_action_specs_clean_core.yaml"
    scenario_path.write_text(
        """
version: 1
name: Dealer Action Specs Clean Core Guard
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: H1
  - id: H2
    kind: household
    name: H2
action_specs:
  - kind: household
    profile_type: trader
    actions:
      - {action: sell_ticket, phase: B_Dealer}
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
run:
  mode: until_stable
  default_handling: fail-fast
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 1
    assert "Type: ConfigurationError" in result.output
    assert "action_specs request B_Dealer phase" in result.output
    assert "Unexpected error" not in result.output
    assert "Traceback" not in result.output


def test_auto_engine_uses_clean_core_for_action_specs_lending_phase(tmp_path: Path) -> None:
    scenario_path = tmp_path / "action_specs_lending.yaml"
    scenario_path.write_text(
        """
version: 1
name: Action Specs Lending Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
  - id: lender
    kind: non_bank_lender
    name: Lender
action_specs:
  - kind: non_bank_lender
    profile_type: lender
    actions:
      - {action: lend, phase: B_Lending}
    information: omniscient
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: lender, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 115, due_day: 1}
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "NonBankLoanCreated" in result.output


def test_auto_engine_uses_clean_core_for_realistic_lending_action_specs(tmp_path: Path) -> None:
    scenario_path = tmp_path / "realistic_action_specs_lending.yaml"
    scenario_path.write_text(
        """
version: 1
name: Realistic Action Specs Lending Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
  - id: lender
    kind: non_bank_lender
    name: Lender
action_specs:
  - kind: non_bank_lender
    profile_type: lender
    actions:
      - {action: lend, phase: B_Lending}
    information: realistic
    profile_params:
      kappa: "0.5"
      risk_aversion: "0.3"
      planning_horizon: 5
      profit_target: "0.05"
      max_loan_maturity: 3
      min_coverage_ratio: "0"
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: lender, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 115, due_day: 1}
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    assert "NonBankLoanCreated" in result.output


def test_clean_core_engine_runs_realistic_lending_action_specs(tmp_path: Path) -> None:
    scenario_path = tmp_path / "realistic_action_specs_lending_clean_core.yaml"
    scenario_path.write_text(
        """
version: 1
name: Realistic Action Specs Lending Clean Core Guard
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
  - id: lender
    kind: non_bank_lender
    name: Lender
action_specs:
  - kind: non_bank_lender
    profile_type: lender
    actions:
      - {action: lend, phase: B_Lending}
    information: realistic
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: lender, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 115, due_day: 1}
run:
  mode: until_stable
  quiet_days: 1
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "NonBankLoanCreated" in result.output
    assert "Unexpected error" not in result.output
    assert "Traceback" not in result.output


def test_auto_engine_uses_clean_core_for_blind_lending_action_specs(tmp_path: Path) -> None:
    scenario_path = tmp_path / "blind_action_specs_lending.yaml"
    scenario_path.write_text(
        """
version: 1
name: Blind Action Specs Lending Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
  - id: lender
    kind: non_bank_lender
    name: Lender
action_specs:
  - kind: non_bank_lender
    profile_type: lender
    actions:
      - {action: lend, phase: B_Lending}
    information: blind
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: lender, amount: 1000}
  - create_payable: {from: H1, to: H2, amount: 115, due_day: 1}
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "NonBankLoanCreated" in result.output


def test_auto_engine_uses_clean_core_for_action_specs_rating_phase(tmp_path: Path) -> None:
    scenario_path = tmp_path / "action_specs_rating.yaml"
    scenario_path.write_text(
        """
version: 1
name: Action Specs Rating Smoke
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: Household One
  - id: RA
    kind: rating_agency
    name: Rating Agency
action_specs:
  - kind: rating_agency
    profile_type: rating
    actions:
      - {action: rate, phase: B_Rating}
    information: omniscient
initial_actions:
  - mint_cash: {to: H1, amount: 100}
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "2",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "RatingsPublished" in result.output


def test_auto_engine_uses_clean_core_for_unbanked_payee_deposit_settlement(tmp_path: Path) -> None:
    scenario_path = tmp_path / "unbanked_payee_deposit_settlement.yaml"
    scenario_path.write_text(
        """
version: 1
name: Unbanked Payee Deposit Settlement
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "IntraBankPayment" in result.output
    assert "PayableSettled" in result.output


def test_auto_engine_keeps_clean_core_for_noop_banking_flag(tmp_path: Path) -> None:
    scenario_path = tmp_path / "noop_banking_flag.yaml"
    scenario_path.write_text(
        """
version: 1
name: No-op Banking Flag
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_cash: {to: H1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
run:
  mode: until_stable
  quiet_days: 1
  enable_banking: true
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "PayableSettled" in result.output


def test_auto_engine_uses_clean_core_for_single_bank_generated_banking_config(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_banking.yaml"
    legacy_events_path = tmp_path / "generated_banking.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_banking.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_banking.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_banking.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Banking Config
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    event_kinds = [event["kind"] for event in clean_events]
    assert event_kinds == [event["kind"] for event in legacy_events]
    assert "SubphaseB_BankQuotes" in event_kinds
    assert "SubphaseC_InterbankAuction" in event_kinds
    assert "InterbankAuction" in event_kinds
    auction = next(event for event in clean_events if event["kind"] == "InterbankAuction")
    legacy_auction = next(event for event in legacy_events if event["kind"] == "InterbankAuction")
    assert "phase" not in auction
    assert auction == legacy_auction
    assert event_kinds[-1] == "CBFinalSettlementEnd"
    assert clean_events[-1] == legacy_events[-1]
    assert clean_events[-1]["cb_loans_outstanding_post_final"] == 0
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_two_bank_generated_banking_config(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_two_bank.yaml"
    legacy_events_path = tmp_path / "generated_two_bank.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_two_bank.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_two_bank.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_two_bank.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Two Bank Config
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: B2
    kind: bank
    name: Bank Two
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_reserves: {to: B2, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B2, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
_balanced_config:
  n_banks: 2
  kappa: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_auctions = [event for event in legacy_events if event["kind"] == "InterbankAuction"]
    clean_auctions = [event for event in clean_events if event["kind"] == "InterbankAuction"]
    assert clean_auctions == legacy_auctions
    assert any(event["kind"] == "InterbankCleared" for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_adaptive_generated_banking_config(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_adaptive_banking.yaml"
    legacy_events_path = tmp_path / "generated_adaptive_banking.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_adaptive_banking.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_adaptive_banking.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_adaptive_banking.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Adaptive Banking Config
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: B2
    kind: bank
    name: Bank Two
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_reserves: {to: B2, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B2, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
_balanced_config:
  n_banks: 2
  kappa: "0.5"
  adaptive_corridor: true
  mu: "0.4"
  concentration: "0.8"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_auctions = [event for event in legacy_events if event["kind"] == "InterbankAuction"]
    clean_auctions = [event for event in clean_events if event["kind"] == "InterbankAuction"]
    assert clean_auctions == legacy_auctions
    assert clean_auctions
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_banking_static_cb_loan(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_banking_static_cb_loan.yaml"
    legacy_events_path = tmp_path / "generated_banking_static_cb_loan.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_banking_static_cb_loan.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_banking_static_cb_loan.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_banking_static_cb_loan.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Banking Static CB Loan
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 200}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
  - create_cb_loan: {bank: B1, amount: 100, rate: "0.03", issuance_day: 10, alias: CBL1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_cb_events = [event for event in legacy_events if event["kind"].startswith("CB")]
    clean_cb_events = [event for event in clean_events if event["kind"].startswith("CB")]
    assert _normalize_unstable_contract_ids(clean_cb_events) == _normalize_unstable_contract_ids(
        legacy_cb_events
    )
    assert clean_cb_events[-1]["loans_repaid"] == 1
    assert clean_cb_events[-1]["cb_loans_outstanding_post_final"] == -100
    assert clean_cb_events[-1]["cb_reserves_final"] == 97
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_banking_cb_loan_default(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_banking_cb_loan_default.yaml"
    legacy_events_path = tmp_path / "generated_banking_cb_loan_default.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_banking_cb_loan_default.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_banking_cb_loan_default.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_banking_cb_loan_default.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Banking CB Loan Default
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 50}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
  - create_cb_loan: {bank: B1, amount: 100, rate: "0.03", issuance_day: 10, alias: CBL1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_resolution_events = legacy_events[legacy_events.index(next(
        event for event in legacy_events if event["kind"] == "CBFinalSettlementStart"
    )) :]
    clean_resolution_events = clean_events[clean_events.index(next(
        event for event in clean_events if event["kind"] == "CBFinalSettlementStart"
    )) :]
    assert _normalize_unstable_contract_ids(clean_resolution_events) == (
        _normalize_unstable_contract_ids(legacy_resolution_events)
    )
    assert clean_events[-1]["bank_defaults"] == 1
    assert clean_events[-1]["loans_written_off"] == 1
    assert clean_events[-1]["cb_reserves_final"] == 0
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_multi_key_generated_bank_lending_actions(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "generated_bank_lending_noop.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_noop.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_noop.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_noop.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_noop.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Noop
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - unknown_dynamic: {ignored: true}
    mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    assert "SubphaseB_BankLending" in {event["kind"] for event in clean_events}
    assert not any(event["kind"].startswith("BankLoan") for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_that_issues_and_defaults_loan(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_active.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_active.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_active.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_active.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_active.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Active
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    assert any(event["kind"] == "BankLoanIssued" for event in clean_events)
    assert any(event["kind"] == "BankLoanDefault" for event in clean_events)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_that_repays_loan(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_repaid.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_repaid.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_repaid.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_repaid.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_repaid.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Repaid
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
  - create_payable: {from: H2, to: H1, amount: 60, due_day: 2}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "6",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "6",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    assert any(event["kind"] == "BankLoanRepaid" for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_two_bank_generated_bank_lending_default(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_two_bank_lending_default.yaml"
    legacy_events_path = tmp_path / "generated_two_bank_lending_default.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_two_bank_lending_default.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_two_bank_lending_default.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_two_bank_lending_default.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Two Bank Lending Default
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: B2
    kind: bank
    name: Bank Two
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_reserves: {to: B2, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B2, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 2
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
  trader_bank_assignments:
    H1: [B1, B2]
    H2: [B2, B1]
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    assert any(event["kind"] == "ClientPayment" for event in clean_events)
    assert any(event["kind"] == "InterbankCleared" for event in clean_events)
    assert any(event["kind"] == "BankLoanDefault" for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_with_reserve_transfer(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_reserve_transfer.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_reserve_transfer.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_reserve_transfer.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_reserve_transfer.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_reserve_transfer.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Reserve Transfer
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: B2
    kind: bank
    name: Bank Two
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_reserves: {to: B2, amount: 1000}
  - transfer_reserves: {from_bank: B1, to_bank: B2, amount: 10}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B2, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 2
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
  trader_bank_assignments:
    H1: [B1, B2]
    H2: [B2, B1]
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events, ordered=False)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    assert any(event["kind"] == "ReservesTransferred" for event in clean_events)
    assert any(event["kind"] == "BankLoanIssued" for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_two_bank_generated_bank_lending_repaid(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_two_bank_lending_repaid.yaml"
    legacy_events_path = tmp_path / "generated_two_bank_lending_repaid.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_two_bank_lending_repaid.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_two_bank_lending_repaid.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_two_bank_lending_repaid.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Two Bank Lending Repaid
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: B2
    kind: bank
    name: Bank Two
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_reserves: {to: B2, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B2, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
  - create_payable: {from: H2, to: H1, amount: 60, due_day: 2}
_balanced_config:
  n_banks: 2
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
  trader_bank_assignments:
    H1: [B1, B2]
    H2: [B2, B1]
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "6",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "6",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    assert any(event["kind"] == "BankLoanRepaid" for event in clean_events)
    assert any(
        event["kind"] == "ClientPayment"
        and event.get("payer_bank") != event.get("payee_bank")
        for event in clean_events
    )
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_coverage_rejection(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_coverage.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_coverage.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_coverage.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_coverage.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_coverage.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Coverage Rejection
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 60}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B1, amount: 60}
  - create_payable: {from: H2, to: H1, amount: 60, due_day: 1}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  min_coverage_ratio: "0.5"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_rejections = [
        event for event in legacy_events if event["kind"] == "BankLoanRejectedCoverage"
    ]
    clean_rejections = [
        event for event in clean_events if event["kind"] == "BankLoanRejectedCoverage"
    ]
    assert clean_rejections == legacy_rejections
    assert not any(event["kind"].startswith("BankLoanIssued") for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_credit_risk_loading(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_credit_risk.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_credit_risk.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_credit_risk.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_credit_risk.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_credit_risk.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Credit Risk
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  credit_risk_loading: "0.5"
  max_borrower_risk: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    issued = next(event for event in clean_bank_events if event["kind"] == "BankLoanIssued")
    assert Decimal(str(issued["rate"])) > Decimal("0")
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_rationing(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_rationing.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_rationing.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_rationing.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_rationing.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_rationing.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Rationing
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 60}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B1, amount: 60}
  - create_payable: {from: H2, to: H1, amount: 60, due_day: 1}
  - create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  credit_risk_loading: "0.5"
  max_borrower_risk: "0.10"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "4",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_rationed = [
        event for event in legacy_events if event["kind"] == "BankLoanRationed"
    ]
    clean_rationed = [
        event for event in clean_events if event["kind"] == "BankLoanRationed"
    ]
    assert clean_rationed == legacy_rationed
    assert not any(event["kind"] == "BankLoanIssued" for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_cb_cutoff(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_cb_cutoff.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_cb_cutoff.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_cb_cutoff.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_cb_cutoff.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_cb_cutoff.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending CB Cutoff
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H2, amount: 150}
  - deposit_cash: {customer: H2, bank: B1, amount: 150}
  - create_payable: {from: H1, to: H2, amount: 100, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "0.5"
  enable_bank_lending: true
  maturity_days: 2
  cb_lending_cutoff_day: 1
  Q_total: 50
run:
  mode: until_stable
  default_handling: expel-agent
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    assert [event for event in clean_events if event["kind"] == "CBLendingFreezeActivated"] == [
        event for event in legacy_events if event["kind"] == "CBLendingFreezeActivated"
    ]
    assert not any(event["kind"] == "CBLendingFreezeStability" for event in clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_generated_bank_lending_scheduled_payable(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_bank_lending_scheduled_payable.yaml"
    legacy_events_path = tmp_path / "generated_bank_lending_scheduled_payable.legacy.events.jsonl"
    legacy_balances_path = tmp_path / "generated_bank_lending_scheduled_payable.legacy.balances.csv"
    clean_events_path = tmp_path / "generated_bank_lending_scheduled_payable.clean.events.jsonl"
    clean_balances_path = tmp_path / "generated_bank_lending_scheduled_payable.clean.balances.csv"
    scenario_path.write_text(
        """
version: 1
name: Generated Bank Lending Scheduled Payable
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - mint_cash: {to: H2, amount: 1}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - deposit_cash: {customer: H2, bank: B1, amount: 1}
  - create_payable: {from: H2, to: H1, amount: 1, due_day: 1}
scheduled_actions:
  - day: 1
    action:
      create_payable: {from: H1, to: H2, amount: 150, due_day: 1}
_balanced_config:
  n_banks: 1
  kappa: "1.0"
  enable_bank_lending: true
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  default_handling: expel-agent
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "5",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    legacy_bank_events = [
        event for event in legacy_events if event["kind"].startswith("BankLoan")
    ]
    clean_bank_events = [
        event for event in clean_events if event["kind"].startswith("BankLoan")
    ]
    assert _normalize_unstable_contract_ids(clean_bank_events) == _normalize_unstable_contract_ids(
        legacy_bank_events
    )
    assert any(event["kind"] == "BankLoanIssued" and event["day"] == 1 for event in clean_bank_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_auto_engine_uses_clean_core_for_mismatched_generated_banking_config(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "generated_mismatched_banks.yaml"
    scenario_path.write_text(
        """
version: 1
name: Generated Mismatched Bank Config
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
_balanced_config:
  n_banks: 2
  kappa: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    legacy_events_path = tmp_path / "legacy.events.jsonl"
    legacy_balances_path = tmp_path / "legacy.balances.csv"
    clean_events_path = tmp_path / "clean.events.jsonl"
    clean_balances_path = tmp_path / "clean.balances.csv"
    runner = CliRunner()
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "legacy",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "auto",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "auto fallback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_clean_core_engine_accepts_mismatched_generated_banking_config(tmp_path: Path) -> None:
    scenario_path = tmp_path / "generated_banking_clean_core.yaml"
    scenario_path.write_text(
        """
version: 1
name: Generated Banking Clean Core Guard
agents:
  - id: CB
    kind: central_bank
    name: Central Bank
  - id: B1
    kind: bank
    name: Bank One
  - id: H1
    kind: household
    name: Household One
  - id: H2
    kind: household
    name: Household Two
initial_actions:
  - mint_reserves: {to: B1, amount: 1000}
  - mint_cash: {to: H1, amount: 100}
  - deposit_cash: {customer: H1, bank: B1, amount: 100}
  - create_payable: {from: H1, to: H2, amount: 50, due_day: 1}
_balanced_config:
  n_banks: 2
  kappa: "1.0"
  maturity_days: 5
  Q_total: 50
run:
  mode: until_stable
  quiet_days: 1
  show:
    events: summary
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(scenario_path),
            "--engine",
            "clean-core",
            "--max-days",
            "3",
            "--quiet-days",
            "1",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "OK System reached stable state" in result.output
    assert "Traceback" not in result.output


def test_default_cli_engine_uses_clean_core_for_t_account_display() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_bank.yaml"),
            "--max-days",
            "3",
            "--show",
            "summary",
            "--t-account",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: clean-core" in result.output
    assert "Smith Family [H1]" in result.output
    assert "bank_deposit" in result.output


def test_clean_core_cli_engine_writes_html_report(tmp_path: Path) -> None:
    html_path = tmp_path / "clean_core.html"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_nbfi.yaml"),
            "--engine",
            "clean-core",
            "--max-days",
            "10",
            "--html",
            str(html_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Exported HTML report" in result.output
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Bilancio Simulation" in html
    assert "NBFI Lending Demo" in html
    assert "NonBankLoanCreated" in html


def test_clean_core_cli_engine_html_respects_agents_filter(tmp_path: Path) -> None:
    html_path = tmp_path / "filtered.html"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_nbfi.yaml"),
            "--engine",
            "clean-core",
            "--max-days",
            "10",
            "--agents",
            "H2,lender",
            "--html",
            str(html_path),
        ],
    )

    assert result.exit_code == 0, result.output
    html = html_path.read_text(encoding="utf-8")
    final_balances = html.split("<section><h2>Final Balances</h2>", maxsplit=1)[1].split("</section>", maxsplit=1)[0]
    assert "<td>H2</td>" in final_balances
    assert "<td>lender</td>" in final_balances
    assert "<td>H3</td>" not in final_balances
    assert "<td>SYSTEM</td>" not in final_balances


def test_clean_core_cli_engine_html_respects_t_account(tmp_path: Path) -> None:
    html_path = tmp_path / "t_account.html"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_bank.yaml"),
            "--engine",
            "clean-core",
            "--max-days",
            "5",
            "--agents",
            "H1",
            "--t-account",
            "--html",
            str(html_path),
        ],
    )

    assert result.exit_code == 0, result.output
    html = html_path.read_text(encoding="utf-8")
    assert "Final T-Accounts" in html
    assert "Smith Family [H1] (household)" in html
    assert "bank_deposit" in html
    assert "First National Bank [B1]" in html


def test_clean_core_cli_engine_summary_respects_agents_filter() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_nbfi.yaml"),
            "--engine",
            "clean-core",
            "--max-days",
            "10",
            "--agents",
            "H2,lender",
            "--show",
            "summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Final Balances (clean-core)" in result.output
    balances_section = result.output.split("Final Balances (clean-core)", maxsplit=1)[1].split(
        "Event Summary",
        maxsplit=1,
    )[0]
    assert "H2" in balances_section
    assert "lender" in balances_section
    assert "H3" not in balances_section
    assert "Event Summary (clean-core)" in result.output
    assert "NonBankLoanCreated" in result.output


def test_clean_core_cli_engine_table_show_prints_events() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_bank.yaml"),
            "--engine",
            "clean-core",
            "--max-days",
            "5",
            "--show",
            "table",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Final Balances (clean-core)" in result.output
    assert "Events (clean-core)" in result.output
    assert "Setup (Day 0)" in result.output
    assert "Phase B" in result.output
    assert "Phase C" in result.output
    assert "PayableSettled" in result.output


def test_clean_core_cli_engine_supports_step_mode(tmp_path: Path) -> None:
    events_path = tmp_path / "step.events.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_bank.yaml"),
            "--engine",
            "clean-core",
            "--mode",
            "step",
            "--max-days",
            "3",
            "--export-events",
            str(events_path),
        ],
        input="y\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Run day 1?" in result.output
    assert "Simulation stopped by user" in result.output
    assert "Simulation stopped before stability" in result.output
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert "PhaseA" in {event["kind"] for event in events}


def test_clean_core_cli_engine_exports_after_default_errors_without_traceback(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    legacy_events_path = tmp_path / "legacy_default_error" / "events.jsonl"
    legacy_balances_path = tmp_path / "legacy_default_error" / "balances.csv"
    clean_events_path = tmp_path / "clean_default_error" / "events.jsonl"
    clean_balances_path = tmp_path / "clean_default_error" / "balances.csv"
    legacy_result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_dealer.yaml"),
            "--engine",
            "legacy",
            "--max-days",
            "10",
            "--show",
            "summary",
            "--export-events",
            str(legacy_events_path),
            "--export-balances",
            str(legacy_balances_path),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "simple_dealer.yaml"),
            "--engine",
            "clean-core",
            "--max-days",
            "10",
            "--show",
            "summary",
            "--export-events",
            str(clean_events_path),
            "--export-balances",
            str(clean_balances_path),
        ],
    )

    assert legacy_result.exit_code == 0, legacy_result.output
    assert result.exit_code == 0, result.output
    assert "Insufficient funds to settle payable" in result.output
    assert "Simulation stopped after an error" in result.output
    assert "Traceback" not in result.output
    legacy_events = [json.loads(line) for line in legacy_events_path.read_text().splitlines()]
    clean_events = [json.loads(line) for line in clean_events_path.read_text().splitlines()]
    _assert_event_exports_match(legacy_events, clean_events)
    _assert_balance_exports_match(
        _read_balance_export(legacy_balances_path),
        _read_balance_export(clean_balances_path),
    )


def test_clean_core_cli_engine_preflights_scheduled_aliases(tmp_path: Path) -> None:
    scenario_path = tmp_path / "bad_alias.yaml"
    scenario_path.write_text(
        """
version: 1
name: Bad Alias
agents:
  - {id: F1, kind: firm, name: Firm One}
  - {id: F2, kind: firm, name: Firm Two}
initial_actions: []
scheduled_actions:
  - day: 1
    action:
      transfer_claim: {contract_alias: GHOST, to_agent: F1}
run:
  mode: until_stable
  max_days: 2
  quiet_days: 1
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(scenario_path), "--engine", "clean-core"])

    assert result.exit_code == 1
    assert "unknown alias 'GHOST'" in result.output
    assert "Traceback" not in result.output


def test_clean_core_cli_engine_honors_default_handling_override() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLES_DIR / "default_handling_demo.yaml"),
            "--engine",
            "clean-core",
            "--default-handling",
            "fail-fast",
            "--max-days",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "Default handling mode: fail-fast" in result.output
    assert "Insufficient funds to settle payable" in result.output
    assert "Simulation stopped after an error" in result.output
    assert "Traceback" not in result.output


def test_simple_bank_cli_export_contract(tmp_path: Path) -> None:
    """The one-bank example settles the household payable inside one bank."""
    output, events, balances = _run_scenario_with_exports(tmp_path, "simple_bank.yaml")

    assert "Simple Banking System" in output
    assert len(events) == 28
    assert Counter(event["kind"] for event in events) == Counter(
        {
            "PhaseA": 4,
            "PhaseB": 4,
            "SubphaseB1": 4,
            "SubphaseB2": 4,
            "PhaseC": 4,
            "CashMinted": 2,
            "CashDeposited": 2,
            "ReservesMinted": 1,
            "PayableCreated": 1,
            "IntraBankPayment": 1,
            "PayableSettled": 1,
        }
    )
    assert _event_exists(events, kind="IntraBankPayment", day=1, payer="H1", payee="H2", bank="B1", amount=500)
    assert _event_exists(events, kind="PayableSettled", day=1, debtor="H1", creditor="H2", amount=500)

    assert _amount(balances["CB"], "liabilities_cash") == Decimal("3500")
    assert _amount(balances["CB"], "liabilities_reserve_deposit") == Decimal("10000")
    assert _amount(balances["B1"], "assets_cash") == Decimal("2800")
    assert _amount(balances["B1"], "assets_reserve_deposit") == Decimal("10000")
    assert _amount(balances["B1"], "liabilities_bank_deposit") == Decimal("2800")
    assert _amount(balances["H1"], "assets_bank_deposit") == Decimal("1300")
    assert _amount(balances["H1"], "assets_cash") == Decimal("200")
    assert _amount(balances["H2"], "assets_bank_deposit") == Decimal("1500")
    assert _amount(balances["H2"], "assets_cash") == Decimal("500")


def test_sasa_scenario_cli_export_contract(tmp_path: Path) -> None:
    """The two-bank example splits payment across cash and interbank settlement."""
    output, events, balances = _run_scenario_with_exports(tmp_path, "sasa_scenario.yaml")

    assert "sasa_scenario" in output
    assert len(events) == 33
    assert Counter(event["kind"] for event in events) == Counter(
        {
            "PhaseA": 4,
            "PhaseB": 4,
            "SubphaseB1": 4,
            "SubphaseB2": 4,
            "PhaseC": 4,
            "ReservesMinted": 2,
            "CashMinted": 2,
            "CashDeposited": 2,
            "PayableCreated": 1,
            "CashTransferred": 1,
            "ClientPayment": 1,
            "PayableSettled": 1,
            "ReservesTransferred": 1,
            "InstrumentMerged": 1,
            "InterbankCleared": 1,
        }
    )
    assert _event_exists(events, kind="CashTransferred", day=1, frm="FIRM_A", to="FIRM_B", amount=100)
    assert _event_exists(
        events,
        kind="ClientPayment",
        day=1,
        payer="FIRM_A",
        payer_bank="BANK_A",
        payee="FIRM_B",
        payee_bank="BANK_B",
        amount=50,
    )
    assert _event_exists(events, kind="ReservesTransferred", day=1, frm="BANK_A", to="BANK_B", amount=50)
    assert _event_exists(events, kind="InterbankCleared", day=1, debtor_bank="BANK_A", creditor_bank="BANK_B", amount=50)

    assert _amount(balances["BANK_A"], "assets_reserve_deposit") == Decimal("950")
    assert _amount(balances["BANK_A"], "liabilities_bank_deposit") == Decimal("50")
    assert _amount(balances["BANK_B"], "assets_reserve_deposit") == Decimal("1050")
    assert _amount(balances["BANK_B"], "liabilities_bank_deposit") == Decimal("150")
    assert _amount(balances["FIRM_A"], "assets_bank_deposit") == Decimal("50")
    assert _amount(balances["FIRM_A"], "total_financial_liabilities") == Decimal("0")
    assert _amount(balances["FIRM_B"], "assets_bank_deposit") == Decimal("150")
    assert _amount(balances["FIRM_B"], "assets_cash") == Decimal("100")


def test_interbank_netting_cli_export_contract(tmp_path: Path) -> None:
    """The interbank example clears only the daily net reserve positions."""
    output, events, balances = _run_scenario_with_exports(tmp_path, "interbank_netting.yaml")

    assert "Interbank Netting Example" in output
    assert len(events) == 59
    assert Counter(event["kind"] for event in events) == Counter(
        {
            "PhaseA": 5,
            "PhaseB": 5,
            "SubphaseB1": 5,
            "SubphaseB2": 5,
            "PhaseC": 5,
            "CashDeposited": 4,
            "CashMinted": 4,
            "ClientPayment": 6,
            "PayableCreated": 6,
            "PayableSettled": 6,
            "ReservesMinted": 2,
            "ReservesTransferred": 2,
            "InterbankCleared": 2,
            "InstrumentMerged": 2,
        }
    )
    assert _event_exists(events, kind="InterbankCleared", day=1, debtor_bank="B1", creditor_bank="B2", amount=700)
    assert _event_exists(events, kind="InterbankCleared", day=2, debtor_bank="B2", creditor_bank="B1", amount=600)
    assert _event_exists(events, kind="ReservesTransferred", day=1, frm="B1", to="B2", amount=700)
    assert _event_exists(events, kind="ReservesTransferred", day=2, frm="B2", to="B1", amount=600)

    assert _amount(balances["B1"], "assets_reserve_deposit") == Decimal("9900")
    assert _amount(balances["B1"], "liabilities_bank_deposit") == Decimal("8900")
    assert _amount(balances["B2"], "assets_reserve_deposit") == Decimal("10100")
    assert _amount(balances["B2"], "liabilities_bank_deposit") == Decimal("9100")
    assert _amount(balances["H1"], "assets_bank_deposit") == Decimal("5000")
    assert _amount(balances["H2"], "assets_bank_deposit") == Decimal("4800")
    assert _amount(balances["H3"], "assets_bank_deposit") == Decimal("3900")
    assert _amount(balances["H4"], "assets_bank_deposit") == Decimal("4300")


def test_firm_delivery_cli_export_contract(tmp_path: Path) -> None:
    """The delivery example settles goods and matching payables over three days."""
    output, events, balances = _run_scenario_with_exports(tmp_path, "firm_delivery.yaml", max_days=8)

    assert "Firm with Delivery Obligations" in output
    assert len(events) == 65
    assert Counter(event["kind"] for event in events) == Counter(
        {
            "PhaseA": 6,
            "PhaseB": 6,
            "SubphaseB1": 6,
            "SubphaseB2": 6,
            "PhaseC": 6,
            "CashDeposited": 4,
            "CashMinted": 4,
            "DeliveryObligationCancelled": 3,
            "DeliveryObligationCreated": 3,
            "DeliveryObligationSettled": 3,
            "IntraBankPayment": 3,
            "PayableCreated": 3,
            "PayableSettled": 3,
            "ReservesMinted": 1,
            "StockCreated": 2,
            "StockSplit": 3,
            "StockTransferred": 3,
        }
    )
    assert _event_exists(events, kind="StockTransferred", day=1, frm="F1", to="H1", sku="WIDGET", qty=10)
    assert _event_exists(events, kind="StockTransferred", day=2, frm="F1", to="F2", sku="WIDGET", qty=20)
    assert _event_exists(events, kind="StockTransferred", day=3, frm="F2", to="H2", sku="GADGET", qty=5)
    assert _event_exists(events, kind="PayableSettled", day=1, debtor="H1", creditor="F1", amount=250)
    assert _event_exists(events, kind="PayableSettled", day=2, debtor="F2", creditor="F1", amount=500)
    assert _event_exists(events, kind="PayableSettled", day=3, debtor="H2", creditor="F2", amount=500)

    assert _amount(balances["H1"], "inventory_WIDGET_quantity") == Decimal("10")
    assert _amount(balances["H1"], "inventory_WIDGET_value") == Decimal("250.0")
    assert _amount(balances["H2"], "inventory_GADGET_quantity") == Decimal("5")
    assert _amount(balances["H2"], "inventory_GADGET_value") == Decimal("500.0")
    assert _amount(balances["F1"], "inventory_WIDGET_quantity") == Decimal("70")
    assert _amount(balances["F1"], "assets_bank_deposit") == Decimal("8750")
    assert _amount(balances["F2"], "inventory_WIDGET_quantity") == Decimal("20")
    assert _amount(balances["F2"], "inventory_GADGET_quantity") == Decimal("45")
    assert _amount(balances["F2"], "assets_bank_deposit") == Decimal("6000")


def test_intraday_netting_cli_export_contract(tmp_path: Path) -> None:
    """The intraday example preserves gross mutual cash flows and goods exchange."""
    output, events, balances = _run_scenario_with_exports(tmp_path, "intraday_netting.yaml", max_days=8)

    assert "Intraday Netting Example" in output
    assert len(events) == 58
    assert Counter(event["kind"] for event in events) == Counter(
        {
            "PhaseA": 5,
            "PhaseB": 5,
            "SubphaseB1": 5,
            "SubphaseB2": 5,
            "PhaseC": 5,
            "CashDeposited": 3,
            "CashMinted": 3,
            "CashTransferred": 3,
            "DeliveryObligationCancelled": 2,
            "DeliveryObligationCreated": 2,
            "DeliveryObligationSettled": 2,
            "InstrumentMerged": 2,
            "IntraBankPayment": 1,
            "PayableCreated": 4,
            "PayableSettled": 4,
            "ReservesMinted": 1,
            "StockCreated": 2,
            "StockSplit": 2,
            "StockTransferred": 2,
        }
    )
    assert _event_exists(events, kind="CashTransferred", day=1, frm="F1", to="F2", amount=2000)
    assert _event_exists(events, kind="CashTransferred", day=1, frm="F2", to="F1", amount=1500)
    assert _event_exists(events, kind="StockTransferred", day=2, frm="F1", to="F2", sku="WIDGET", qty=10)
    assert _event_exists(events, kind="StockTransferred", day=2, frm="F2", to="F1", sku="GADGET", qty=15)

    assert _amount(balances["F1"], "assets_bank_deposit") == Decimal("8500")
    assert _amount(balances["F1"], "assets_cash") == Decimal("1500")
    assert _amount(balances["F1"], "inventory_WIDGET_quantity") == Decimal("90")
    assert _amount(balances["F1"], "inventory_GADGET_quantity") == Decimal("15")
    assert _amount(balances["F2"], "assets_bank_deposit") == Decimal("8000")
    assert _amount(balances["F2"], "assets_cash") == Decimal("2200")
    assert _amount(balances["F2"], "inventory_GADGET_quantity") == Decimal("85")
    assert _amount(balances["F2"], "inventory_WIDGET_quantity") == Decimal("10")
    assert _amount(balances["H1"], "assets_bank_deposit") == Decimal("3500")
    assert _amount(balances["H1"], "assets_cash") == Decimal("1300")


def test_two_jurisdictions_cli_export_contract(tmp_path: Path) -> None:
    """The jurisdiction example loads config and settles the cross-firm payable."""
    output, events, balances = _run_scenario_with_exports(tmp_path, "two_jurisdictions.yaml", max_days=8)

    assert "Two-Jurisdiction Banking System" in output
    assert len(events) == 38
    assert Counter(event["kind"] for event in events) == Counter(
        {
            "PhaseA": 6,
            "PhaseB": 6,
            "SubphaseB1": 6,
            "SubphaseB2": 6,
            "PhaseC": 6,
            "CashMinted": 2,
            "ReservesMinted": 2,
            "PayableCreated": 1,
            "CashTransferred": 1,
            "PayableSettled": 1,
            "InstrumentMerged": 1,
        }
    )
    assert _event_exists(events, kind="CashTransferred", day=3, frm="F_US", to="F_EU", amount=1000)
    assert _event_exists(events, kind="PayableSettled", day=3, debtor="F_US", creditor="F_EU", amount=1000)

    assert _amount(balances["CB_US"], "liabilities_cash") == Decimal("9000")
    assert _amount(balances["CB_US"], "liabilities_reserve_deposit") == Decimal("18000")
    assert _amount(balances["CB_EU"], "net_financial") == Decimal("0")
    assert _amount(balances["B_US"], "assets_reserve_deposit") == Decimal("10000")
    assert _amount(balances["B_EU"], "assets_reserve_deposit") == Decimal("8000")
    assert _amount(balances["F_US"], "assets_cash") == Decimal("4000")
    assert _amount(balances["F_EU"], "assets_cash") == Decimal("5000")
