"""Parity tests between the legacy simulation engine and clean core.

Test intent:
- Preserve YAML scenario compatibility across the clean-core rewrite.
- Prove final balance rows and event payloads match legacy behavior for the
  supported scenario surface.
- Protect accounting and default-handling semantics while the clean core grows.
"""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from bilancio.analysis.balances import as_rows as legacy_balance_rows
from bilancio.config import apply_to_system, load_yaml
from bilancio.config.models import (
    ActionDefConfig,
    ActionSpecConfig,
    AgentSpec,
    LenderScenarioConfig,
    RatingAgencyScenarioConfig,
    ScenarioConfig,
    ScheduledAction,
)
from bilancio.core.errors import DefaultError
from bilancio.engines.clean_core import balance_rows as clean_balance_rows
from bilancio.engines.clean_core import run_basic_scenario
from bilancio.engines.simulation import run_until_stable
from bilancio.engines.system import System
from bilancio.engines.termination import StopReason

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "scenarios"
EXERCISE_SCENARIOS_DIR = Path(__file__).parent.parent.parent / "examples" / "exercise_scenarios" / "yaml"
KALECKI_SCENARIOS_DIR = Path(__file__).parent.parent.parent / "examples" / "kalecki"

CLEAN_CORE_SUPPORTED_SCENARIOS = (
    "default_handling_demo.yaml",
    "firm_delivery.yaml",
    "interbank_netting.yaml",
    "intraday_netting.yaml",
    "payment_demo.yaml",
    "rich_simulation.yaml",
    "ring_with_action_specs.yaml",
    "sasa_scenario.yaml",
    "simple_bank.yaml",
    "simple_nbfi.yaml",
    "two_banks_interbank.yaml",
    "two_jurisdictions.yaml",
)

CLEAN_CORE_SUPPORTED_DEFAULTING_SCENARIOS = (
    "kalecki_with_dealer.yaml",
    "simple_dealer.yaml",
)

CLEAN_CORE_SUPPORTED_EXERCISE_SCENARIOS = (
    "ex1_cash_for_goods.yaml",
    "ex2_two_firms_cash_purchase.yaml",
    "ex3_iou_assignment.yaml",
    "ex4_generic_claim_transfer.yaml",
    "ex5_deferred_exchange.yaml",
    "ex6_goods_now_cash_later.yaml",
    "ex7_cash_now_goods_later.yaml",
)

CLEAN_CORE_SUPPORTED_KALECKI_SCENARIOS = (
    "kalecki_ring_baseline.yaml",
)

CLEAN_CORE_LENDER_FEATURE_VARIANTS = (
    ("coverage_gate", {"min_coverage_ratio": Decimal("0.20")}),
    ("graduated_coverage", {"min_coverage_ratio": Decimal("0.80"), "coverage_mode": "graduated"}),
    ("maturity_matching", {"maturity_matching": True, "min_loan_maturity": 1, "max_loan_maturity": 5}),
    ("borrower_loan_cap", {"max_loans_per_borrower_per_day": 1}),
    ("cascade_ranking", {"ranking_mode": "cascade"}),
    ("blended_ranking", {"ranking_mode": "blended", "cascade_weight": Decimal("0.35")}),
    ("stress_risk_premium", {"stress_risk_premium_scale": Decimal("0.20")}),
    ("collateralized_terms", {"collateralized_terms": True, "collateral_advance_rate": Decimal("0.50")}),
    ("marginal_relief_gate", {"marginal_relief_min_ratio": Decimal("2.0")}),
    ("daily_expected_loss_budget", {"daily_expected_loss_budget_ratio": Decimal("0.15")}),
    ("run_expected_loss_budget", {"run_expected_loss_budget_ratio": Decimal("0.20")}),
    ("adaptive_capital_conservation", {"adaptive_capital_conservation": True}),
    ("adaptive_risk_aversion_flag", {"adaptive_risk_aversion": True}),
    ("adaptive_loan_maturity_flag", {"adaptive_loan_maturity": True}),
    ("adaptive_rates_flag", {"adaptive_rates": True}),
    ("adaptive_prevention_flag", {"adaptive_prevention": True}),
    ("noisy_liability_information", {"info_liabilities_visibility": "noisy"}),
    ("hidden_liability_information", {"info_liabilities_visibility": "none"}),
    (
        "sampled_history_information",
        {"info_history_visibility": "noisy", "info_history_sample_rate": Decimal("0.5")},
    ),
    ("network_information_flag", {"info_network_visibility": "perfect"}),
    ("market_information_flag", {"info_market_visibility": "perfect"}),
)


def _run_legacy(config: ScenarioConfig, *, max_days: int, quiet_days: int) -> tuple[Any, System]:
    system = System(default_mode=config.run.default_handling)
    apply_to_system(config, system)
    system.state.rollover_enabled = config.run.rollover_enabled
    system.state.estimate_logging_enabled = config.run.estimate_logging
    for scheduled in config.scheduled_actions:
        system.state.scheduled_actions_by_day.setdefault(scheduled.day, []).append(scheduled.action)

    result = run_until_stable(
        system,
        max_days=max_days,
        quiet_days=quiet_days,
        enable_dealer=bool(config.dealer and config.dealer.enabled),
        enable_lender=system.state.lender_config is not None,
        enable_rating=system.state.rating_config is not None,
    )
    return result, system


def _decimal(value: Any) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    return Decimal(str(value))


def _assert_balance_rows_match(legacy_rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]]) -> None:
    legacy_by_agent = {row["agent_id"]: row for row in legacy_rows}
    clean_by_agent = {row["agent_id"]: row for row in clean_rows}
    assert clean_by_agent.keys() == legacy_by_agent.keys()

    for agent_id, legacy_row in legacy_by_agent.items():
        clean_row = clean_by_agent[agent_id]
        assert clean_row["agent_id"] == legacy_row["agent_id"]
        for key in set(legacy_row) | set(clean_row):
            if key == "agent_id":
                continue
            assert _decimal(clean_row.get(key)) == _decimal(legacy_row.get(key)), (agent_id, key)


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


def _assert_event_payloads_match(
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

    def signature_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: signature_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [signature_value(item) for item in value]
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
        return value

    def signature(event: dict[str, Any]) -> str:
        return json.dumps(signature_value(event), default=str, sort_keys=True)

    assert Counter(signature(event) for event in clean_normalized) == Counter(
        signature(event) for event in legacy_normalized
    )


def _non_bank_lender_event_signature(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    fields = {
        "kind",
        "day",
        "phase",
        "lender_id",
        "borrower_id",
        "amount",
        "rate",
        "p_default",
        "maturity_day",
        "principal",
        "interest",
        "total_repaid",
        "amount_owed",
        "cash_available",
        "coverage",
        "min_coverage",
        "count",
        "limit",
        "scope",
        "expected_loss",
        "budget_cap",
        "budget_used",
        "realized_loss",
        "realized_ratio",
        "threshold",
        "preventive",
        "at_risk_receivables",
    }
    return [
        {field: str(event[field]) for field in fields if field in event}
        for event in events
        if str(event.get("kind", "")).startswith("NonBank")
    ]


def _rating_event_signature(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    fields = {"kind", "day", "phase", "agency_id", "n_rated", "n_eligible", "ratings"}
    return [
        {field: str(event[field]) for field in fields if field in event}
        for event in events
        if event.get("kind") in {"SubphaseB_Rating", "RatingsPublished"}
    ]


def _estimate_log_signature(estimates: list[Any]) -> list[dict[str, str]]:
    return [
        {
            "value": str(estimate.value),
            "estimator_id": estimate.estimator_id,
            "target_id": estimate.target_id,
            "target_type": estimate.target_type,
            "estimation_day": str(estimate.estimation_day),
            "method": estimate.method,
        }
        for estimate in estimates
    ]


def test_clean_core_matches_legacy_multi_key_initial_action_dispatch_order() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {
                    "unknown_dynamic": {"ignored": True},
                    "mint_reserves": {"to": "B1", "amount": 10000},
                },
                *config.initial_actions[1:],
            ],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


@pytest.mark.parametrize("scenario_name", CLEAN_CORE_SUPPORTED_SCENARIOS)
def test_clean_core_matches_legacy_engine_for_supported_examples(scenario_name: str) -> None:
    config = load_yaml(EXAMPLES_DIR / scenario_name)

    legacy_result, legacy_system = _run_legacy(config, max_days=10, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(
        legacy_system.state.events,
        clean_result.events,
        ordered=scenario_name != "simple_nbfi.yaml",
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_omniscient_rating_agency() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                *config.agents,
                AgentSpec(id="RA", kind="rating_agency", name="Rating Agency"),
            ],
            "rating_agency": RatingAgencyScenarioConfig(
                enabled=True,
                info_profile="omniscient",
                coverage_fraction=Decimal("1.0"),
            ),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert _rating_event_signature(clean_result.events) == (
        _rating_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_deposit_payment_to_unbanked_payee() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": 1000}},
                {"mint_cash": {"to": "H1", "amount": 100}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 100}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 50, "due_day": 1}},
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=1)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=1)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert any(
        event.get("kind") == "IntraBankPayment" and event.get("payee") == "H2"
        for event in clean_result.events
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_scheduled_contract_alias_collision() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
                AgentSpec(id="F2", kind="firm", name="Firm Two"),
                AgentSpec(id="F3", kind="firm", name="Firm Three"),
            ],
            "initial_actions": [
                {
                    "create_payable": {
                        "from": "F1",
                        "to": "F2",
                        "amount": 30,
                        "due_day": 3,
                        "alias": "OLD",
                    }
                },
                {"mint_cash": {"to": "F1", "amount": 100}},
            ],
            "scheduled_actions": [
                ScheduledAction(
                    day=1,
                    action={
                        "create_payable": {
                            "from": "F1",
                            "to": "F2",
                            "amount": 70,
                            "due_day": 2,
                            "alias": "NEW",
                        }
                    },
                ),
                ScheduledAction(
                    day=1,
                    action={"transfer_claim": {"contract_alias": "NEW", "to_agent": "F3"}},
                ),
            ],
            "run": config.run.model_copy(update={"default_handling": "fail-fast"}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=6, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=6, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    claim_events = [event for event in clean_result.events if event["kind"] == "ClaimTransferred"]
    assert claim_events == [
        {
            "kind": "ClaimTransferred",
            "day": 1,
            "phase": "simulation",
            "contract_id": "PAY_1",
            "frm": "F2",
            "to": "F3",
            "contract_kind": "payable",
            "amount": Decimal("70"),
            "due_day": 2,
            "sku": None,
            "alias": "NEW",
        }
    ]
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_delivery_across_stock_lots() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
                AgentSpec(id="F2", kind="firm", name="Firm Two"),
            ],
            "initial_actions": [
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 3, "unit_price": "1"}},
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 7, "unit_price": "1"}},
                {
                    "create_delivery_obligation": {
                        "from": "F1",
                        "to": "F2",
                        "sku": "WIDGET",
                        "quantity": 10,
                        "unit_price": "1",
                        "due_day": 1,
                    }
                },
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events, ordered=False)
    assert sum(1 for event in clean_result.events if event["kind"] == "StockTransferred") == 2
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_delivery_settlement_alias_exports() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
                AgentSpec(id="F2", kind="firm", name="Firm Two"),
            ],
            "initial_actions": [
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 5, "unit_price": "1"}},
                {
                    "create_delivery_obligation": {
                        "from": "F1",
                        "to": "F2",
                        "sku": "WIDGET",
                        "quantity": 5,
                        "unit_price": "1",
                        "due_day": 1,
                        "alias": "DELIVERY_ALIAS",
                    }
                },
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    def delivery_alias_signature(events: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        return [
            (str(event["kind"]), event.get("alias"))
            for event in events
            if event["kind"]
            in {
                "DeliveryObligationCreated",
                "DeliveryObligationCancelled",
                "DeliveryObligationSettled",
            }
        ]

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert delivery_alias_signature(clean_result.events) == delivery_alias_signature(
        legacy_system.state.events
    )
    assert delivery_alias_signature(clean_result.events) == [
        ("DeliveryObligationCreated", "DELIVERY_ALIAS"),
        ("DeliveryObligationCancelled", "DELIVERY_ALIAS"),
        ("DeliveryObligationSettled", "DELIVERY_ALIAS"),
    ]
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_delivery_default_in_expel_mode() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
                AgentSpec(id="F2", kind="firm", name="Firm Two"),
                AgentSpec(id="F3", kind="firm", name="Firm Three"),
            ],
            "initial_actions": [
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 3, "unit_price": "1"}},
                {
                    "create_delivery_obligation": {
                        "from": "F1",
                        "to": "F2",
                        "sku": "WIDGET",
                        "quantity": 10,
                        "unit_price": "1",
                        "due_day": 1,
                        "alias": "DELIVERY_SHORT",
                    }
                },
            ],
            "scheduled_actions": [],
            "run": config.run.model_copy(update={"default_handling": "expel-agent"}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    def default_signature(events: list[dict[str, Any]]) -> list[dict[str, str]]:
        fields = {
            "kind",
            "contract_id",
            "alias",
            "debtor",
            "creditor",
            "contract_kind",
            "settlement_kind",
            "delivered_quantity",
            "required_quantity",
            "shortfall",
            "sku",
            "qty",
            "agent",
            "trigger_contract",
            "mode",
        }
        return [
            {
                field: "<contract>" if field in {"contract_id", "trigger_contract"} else str(event[field])
                for field in fields
                if field in event
            }
            for event in events
            if event["kind"] in {"PartialSettlement", "ObligationDefaulted", "AgentDefaulted"}
        ]

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert default_signature(clean_result.events) == default_signature(legacy_system.state.events)
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_reassignment_weights_include_delivery_liabilities() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
                AgentSpec(id="F2", kind="firm", name="Firm Two"),
                AgentSpec(id="F3", kind="firm", name="Firm Three"),
                AgentSpec(id="F4", kind="firm", name="Firm Four"),
            ],
            "initial_actions": [
                {"mint_cash": {"to": "F4", "amount": 100}},
                {"create_payable": {"from": "F1", "to": "F2", "amount": 100, "due_day": 1}},
                {"create_payable": {"from": "F4", "to": "F1", "amount": 100, "due_day": 3}},
                {
                    "create_delivery_obligation": {
                        "from": "F1",
                        "to": "F3",
                        "sku": "WIDGET",
                        "quantity": 100,
                        "unit_price": "1",
                        "due_day": 4,
                    }
                },
            ],
            "scheduled_actions": [],
            "run": config.run.model_copy(update={"default_handling": "expel-agent"}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=7, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=7, quiet_days=2)

    def reassignment_signature(events: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        return sorted(
            (
                event["debtor"],
                event["new_creditor"],
                str(event["amount"]),
            )
            for event in events
            if event["kind"] == "ReceivableReassigned"
        )

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    assert reassignment_signature(clean_result.events) == reassignment_signature(
        legacy_system.state.events
    )
    assert reassignment_signature(clean_result.events) == [
        ("F4", "F2", "50"),
        ("F4", "F3", "50"),
    ]
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_transfer_stock_first_lot_insufficient() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
                AgentSpec(id="F2", kind="firm", name="Firm Two"),
            ],
            "initial_actions": [
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 3, "unit_price": "1"}},
                {"create_stock": {"owner": "F1", "sku": "WIDGET", "quantity": 10, "unit_price": "1"}},
            ],
            "scheduled_actions": [
                ScheduledAction(
                    day=1,
                    action={
                        "transfer_stock": {
                            "from_agent": "F1",
                            "to_agent": "F2",
                            "sku": "WIDGET",
                            "quantity": 5,
                        }
                    },
                )
            ],
        }
    )

    with pytest.raises(ValueError, match="Insufficient stock: 3 < 5"):
        _run_legacy(config, max_days=3, quiet_days=3)
    with pytest.raises(ValueError, match="Insufficient stock: 3 < 5"):
        run_basic_scenario(config, max_days=3, quiet_days=3)


def test_clean_core_matches_legacy_for_blind_rating_action_specs() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                *config.agents,
                AgentSpec(id="RA", kind="rating_agency", name="Rating Agency"),
            ],
            "action_specs": [
                ActionSpecConfig(
                    kind="rating_agency",
                    profile_type="rating",
                    actions=[ActionDefConfig(action="rate", phase="B_Rating")],
                    information="blind",
                    profile_params={"coverage_fraction": "1.0", "no_data_prior": "0.25"},
                )
            ],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert _rating_event_signature(clean_result.events) == (
        _rating_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_rating_estimate_logging() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                *config.agents,
                AgentSpec(id="RA", kind="rating_agency", name="Rating Agency"),
            ],
            "rating_agency": RatingAgencyScenarioConfig(
                enabled=True,
                info_profile="omniscient",
                coverage_fraction=Decimal("1.0"),
            ),
            "run": config.run.model_copy(update={"estimate_logging": True}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    assert _estimate_log_signature(clean_result.state.estimate_log) == (
        _estimate_log_signature(legacy_system.state.estimate_log)
    )


def test_clean_core_matches_legacy_for_rating_registry_lender_pricing() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="RA", kind="rating_agency", name="Rating Agency"),
                AgentSpec(id="lender", kind="non_bank_lender", name="Lender"),
                AgentSpec(id="H1", kind="household", name="Household One"),
                AgentSpec(id="H2", kind="household", name="Household Two"),
            ],
            "initial_actions": [
                {"mint_cash": {"to": "lender", "amount": 100}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 100, "due_day": 1}},
            ],
            "scheduled_actions": [],
            "rating_agency": RatingAgencyScenarioConfig(
                enabled=True,
                info_profile="omniscient",
                coverage_fraction=Decimal("1.0"),
            ),
            "lender": LenderScenarioConfig(
                enabled=True,
                max_single_exposure=Decimal("1.0"),
                max_total_exposure=Decimal("1.0"),
                horizon=3,
            ),
            "run": config.run.model_copy(update={"default_handling": "fail-fast"}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert _rating_event_signature(clean_result.events) == (
        _rating_event_signature(legacy_system.state.events)
    )
    assert _non_bank_lender_event_signature(clean_result.events) == (
        _non_bank_lender_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_blind_lending_action_specs() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="lender", kind="non_bank_lender", name="Lender"),
                AgentSpec(id="H1", kind="household", name="Household One"),
                AgentSpec(id="H2", kind="household", name="Household Two"),
            ],
            "initial_actions": [
                {"mint_cash": {"to": "H1", "amount": 100}},
                {"mint_cash": {"to": "lender", "amount": 1000}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 115, "due_day": 1}},
            ],
            "scheduled_actions": [],
            "lender": None,
            "action_specs": [
                ActionSpecConfig(
                    kind="non_bank_lender",
                    profile_type="lender",
                    actions=[ActionDefConfig(action="lend", phase="B_Lending")],
                    information="blind",
                )
            ],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    assert any(event["kind"] == "PayableSettled" for event in clean_result.events)
    assert _non_bank_lender_event_signature(clean_result.events) == (
        _non_bank_lender_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_initial_decision_for_realistic_lending_action_specs() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="lender", kind="non_bank_lender", name="Lender"),
                AgentSpec(id="H1", kind="household", name="Household One"),
                AgentSpec(id="H2", kind="household", name="Household Two"),
            ],
            "initial_actions": [
                {"mint_cash": {"to": "lender", "amount": 1000}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 115, "due_day": 1}},
            ],
            "scheduled_actions": [],
            "lender": None,
            "action_specs": [
                ActionSpecConfig(
                    kind="non_bank_lender",
                    profile_type="lender",
                    actions=[ActionDefConfig(action="lend", phase="B_Lending")],
                    information="realistic",
                    profile_params={
                        "kappa": "0.5",
                        "risk_aversion": "0.3",
                        "planning_horizon": 5,
                        "profit_target": "0.05",
                        "max_loan_maturity": 3,
                        "min_coverage_ratio": "0",
                    },
                )
            ],
        }
    )

    _, legacy_system = _run_legacy(config, max_days=1, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=1, quiet_days=2)

    clean_lending_events = _non_bank_lender_event_signature(clean_result.events)
    legacy_lending_events = _non_bank_lender_event_signature(legacy_system.state.events)

    assert clean_lending_events[:2] == legacy_lending_events[:2]


def test_clean_core_matches_legacy_for_lender_stop_loss_gate() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="lender", kind="non_bank_lender", name="Lender"),
                AgentSpec(id="H1", kind="household", name="Household One"),
                AgentSpec(id="H2", kind="household", name="Household Two"),
            ],
            "initial_actions": [
                {"mint_cash": {"to": "lender", "amount": 200}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 100, "due_day": 1}},
            ],
            "scheduled_actions": [
                ScheduledAction(
                    day=2,
                    action={"create_payable": {"from": "H1", "to": "H2", "amount": 50, "due_day": 3}},
                )
            ],
            "lender": LenderScenarioConfig(
                enabled=True,
                base_rate=Decimal("0"),
                risk_premium_scale=Decimal("0"),
                max_single_exposure=Decimal("0.50"),
                max_total_exposure=Decimal("1.0"),
                maturity_days=1,
                horizon=3,
                stop_loss_realized_ratio=Decimal("0.20"),
            ),
            "run": config.run.model_copy(update={"default_handling": "expel-agent"}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=6, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=6, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert _non_bank_lender_event_signature(clean_result.events) == (
        _non_bank_lender_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_preventive_lending() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                AgentSpec(id="CB", kind="central_bank", name="Central Bank"),
                AgentSpec(id="lender", kind="non_bank_lender", name="Lender"),
                AgentSpec(id="F0", kind="firm", name="Firm Zero"),
                AgentSpec(id="F1", kind="firm", name="Firm One"),
            ],
            "initial_actions": [
                {"mint_cash": {"to": "lender", "amount": 500}},
                {"mint_cash": {"to": "F1", "amount": 100}},
                {"create_payable": {"from": "F1", "to": "F0", "amount": 100, "due_day": 2}},
            ],
            "scheduled_actions": [],
            "lender": LenderScenarioConfig(
                enabled=True,
                max_single_exposure=Decimal("1.0"),
                max_total_exposure=Decimal("1.0"),
                maturity_days=3,
                horizon=3,
                kappa=Decimal("0.5"),
                preventive_lending=True,
                prevention_threshold=Decimal("0.04"),
            ),
            "run": config.run.model_copy(update={"default_handling": "expel-agent"}),
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=6, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=6, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert any(
        event.get("kind") == "NonBankLoanCreatedPreventive"
        for event in clean_result.events
    )
    assert _non_bank_lender_event_signature(clean_result.events) == (
        _non_bank_lender_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_hidden_cash_lender_information() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    config = config.model_copy(
        update={"lender": config.lender.model_copy(update={"info_cash_visibility": "none"})}
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=10, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    assert _non_bank_lender_event_signature(clean_result.events) == (
        _non_bank_lender_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_supports_noisy_cash_lender_information() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    config = config.model_copy(
        update={
            "lender": config.lender.model_copy(
                update={
                    "info_cash_visibility": "noisy",
                    "info_cash_noise": Decimal("0.10"),
                }
            )
        }
    )

    clean_result = run_basic_scenario(config, max_days=10, quiet_days=2)
    rows = clean_balance_rows(clean_result)

    assert clean_result.reached_stable is True
    assert any(event["kind"] == "NonBankLoanCreated" for event in clean_result.events)
    assert rows
    assert all(
        _decimal(row["net_financial"]) >= Decimal("0")
        for row in rows
        if row["agent_id"] == "lender"
    )


@pytest.mark.parametrize(("variant_name", "lender_update"), CLEAN_CORE_LENDER_FEATURE_VARIANTS)
def test_clean_core_matches_legacy_for_lender_feature_variants(
    variant_name: str,
    lender_update: dict[str, Any],
) -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    config = config.model_copy(
        update={"lender": config.lender.model_copy(update=lender_update)}
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=10, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED, variant_name
    assert clean_result.reached_stable is True, variant_name
    assert clean_result.final_day == legacy_result.stop_day, variant_name
    _assert_event_payloads_match(
        legacy_system.state.events,
        clean_result.events,
        ordered=False,
    )
    assert _non_bank_lender_event_signature(clean_result.events) == (
        _non_bank_lender_event_signature(legacy_system.state.events)
    )
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_direct_payment_actions() -> None:
    config = load_yaml(EXAMPLES_DIR / "sasa_scenario.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "BANK_A", "amount": 1000}},
                {"mint_reserves": {"to": "BANK_B", "amount": 1000}},
                {"transfer_reserves": {"from_bank": "BANK_B", "to_bank": "BANK_A", "amount": 10}},
                {"mint_cash": {"to": "FIRM_A", "amount": 500}},
                {"mint_cash": {"to": "FIRM_B", "amount": 200}},
                {"transfer_cash": {"from_agent": "FIRM_A", "to_agent": "FIRM_B", "amount": 50}},
                {"deposit_cash": {"customer": "FIRM_A", "bank": "BANK_A", "amount": 300}},
                {"deposit_cash": {"customer": "FIRM_B", "bank": "BANK_B", "amount": 100}},
                {"withdraw_cash": {"customer": "FIRM_B", "bank": "BANK_B", "amount": 25}},
                {"client_payment": {"payer": "FIRM_A", "payee": "FIRM_B", "amount": 75}},
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert {"CashWithdrawn", "ClientPayment", "InterbankCleared"} <= {
        event["kind"] for event in clean_result.events
    }
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_bank_cash_burn() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": 1000}},
                {"mint_cash": {"to": "H1", "amount": 100}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": 80}},
                {"burn_bank_cash": {"bank": "B1"}},
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert "BankCashBurned" in {event["kind"] for event in clean_result.events}
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_simple_rollover() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "run": config.run.model_copy(
                update={"rollover_enabled": True, "max_days": 10, "quiet_days": 2}
            )
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=10, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_static_cb_loan_action() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": 200}},
                {"create_cb_loan": {"bank": "B1", "amount": 100, "rate": "0.03", "issuance_day": 0, "alias": "CBL1"}},
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=5, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert "CBLoanCreated" in {event["kind"] for event in clean_result.events}
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_due_cb_loan_repayment() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": 300}},
                {"create_cb_loan": {"bank": "B1", "amount": 100, "rate": "0.03", "issuance_day": 0, "alias": "CBL1"}},
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=6, quiet_days=3)
    clean_result = run_basic_scenario(config, max_days=6, quiet_days=3)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert "CBLoanRepaid" in {event["kind"] for event in clean_result.events}
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


def test_clean_core_matches_legacy_for_cb_loan_refinancing() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": 50}},
                {"create_cb_loan": {"bank": "B1", "amount": 100, "rate": "0.03", "issuance_day": 0, "alias": "CBL1"}},
            ],
            "scheduled_actions": [],
        }
    )

    legacy_result, legacy_system = _run_legacy(config, max_days=6, quiet_days=3)
    clean_result = run_basic_scenario(config, max_days=6, quiet_days=3)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    assert [event["kind"] for event in clean_result.events].count("CBLoanCreated") == 2
    assert "CBLoanRepaid" in {event["kind"] for event in clean_result.events}
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


@pytest.mark.parametrize("scenario_name", CLEAN_CORE_SUPPORTED_DEFAULTING_SCENARIOS)
def test_clean_core_matches_legacy_default_failure_for_supported_examples(scenario_name: str) -> None:
    config = load_yaml(EXAMPLES_DIR / scenario_name)

    with pytest.raises(DefaultError, match="Insufficient funds to settle payable"):
        _run_legacy(config, max_days=10, quiet_days=2)
    with pytest.raises(DefaultError, match="Insufficient funds to settle payable"):
        run_basic_scenario(config, max_days=10, quiet_days=2)


@pytest.mark.parametrize("scenario_name", CLEAN_CORE_SUPPORTED_EXERCISE_SCENARIOS)
def test_clean_core_matches_legacy_engine_for_exercise_scenarios(scenario_name: str) -> None:
    config = load_yaml(EXERCISE_SCENARIOS_DIR / scenario_name)

    legacy_result, legacy_system = _run_legacy(config, max_days=10, quiet_days=2)
    clean_result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )


@pytest.mark.parametrize("scenario_name", CLEAN_CORE_SUPPORTED_KALECKI_SCENARIOS)
def test_clean_core_matches_legacy_engine_for_kalecki_scenarios(scenario_name: str) -> None:
    config = load_yaml(KALECKI_SCENARIOS_DIR / scenario_name)

    legacy_result, legacy_system = _run_legacy(config, max_days=10, quiet_days=1)
    clean_result = run_basic_scenario(config, max_days=10, quiet_days=1)

    assert legacy_result.stop_reason is StopReason.STABILITY_REACHED
    assert clean_result.reached_stable is True
    assert clean_result.final_day == legacy_result.stop_day
    _assert_event_payloads_match(legacy_system.state.events, clean_result.events)
    _assert_balance_rows_match(
        legacy_balance_rows(legacy_system),
        clean_balance_rows(clean_result),
    )
