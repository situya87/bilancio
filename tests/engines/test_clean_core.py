"""Tests for the clean-room rebuild engine slice."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from bilancio.config.loaders import load_yaml
from bilancio.config.models import (
    ActionDefConfig,
    ActionSpecConfig,
    AgentSpec,
    DealerConfig,
    RatingAgencyScenarioConfig,
)
from bilancio.core.errors import ConfigurationError, DefaultError, ValidationError
from bilancio.engines import clean_core_actions as action_helpers
from bilancio.engines import clean_core_banking as banking_helpers
from bilancio.engines import clean_core_central_bank as cb_helpers
from bilancio.engines import clean_core_dealer as dealer_helpers
from bilancio.engines import clean_core_lender as lender_helpers
from bilancio.engines import clean_core_lending_phase as lending_phase_helpers
from bilancio.engines import clean_core_settlement as settlement_helpers
from bilancio.engines.clean_core import (
    CleanBankingConfig,
    CleanStabilityTracker,
    balance_rows,
    prepare_scenario,
    run_basic_scenario,
    run_day,
    run_runtime_until_stable,
    t_account_rows,
    update_runtime_stability,
    write_balances_csv,
    write_events_jsonl,
    write_html_report,
)
from bilancio.engines.clean_core_cash import (
    take_cash_lots as cash_take_cash_lots,
)
from bilancio.engines.clean_core_cash import (
    transfer_cash_with_events as cash_transfer_cash_with_events,
)
from bilancio.engines.clean_core_compat import (
    build_clean_core_banking_config,
    clean_core_auto_fallback_reason,
)
from bilancio.engines.clean_core_config import (
    build_lender_config as config_build_lender_config,
)
from bilancio.engines.clean_core_config import (
    clean_core_configuration_error_reason as config_clean_core_configuration_error_reason,
)
from bilancio.engines.clean_core_config import (
    clean_core_unsupported_reason as config_clean_core_unsupported_reason,
)
from bilancio.engines.clean_core_config import (
    select_action_payload as config_select_action_payload,
)
from bilancio.engines.clean_core_contracts import (
    action_references_agent as contracts_action_references_agent,
)
from bilancio.engines.clean_core_contracts import (
    contract_id_for_alias as contracts_contract_id_for_alias,
)
from bilancio.engines.clean_core_contracts import (
    transfer_delivery_claim as contracts_transfer_delivery_claim,
)
from bilancio.engines.clean_core_contracts import (
    transfer_payable_claim as contracts_transfer_payable_claim,
)
from bilancio.engines.clean_core_exports import (
    write_balances_csv as write_export_balances_csv,
)
from bilancio.engines.clean_core_exports import (
    write_events_jsonl as write_export_events_jsonl,
)
from bilancio.engines.clean_core_exports import (
    write_html_report as write_export_html_report,
)
from bilancio.engines.clean_core_interbank import (
    clean_interbank_auction_summary as interbank_clean_interbank_auction_summary,
)
from bilancio.engines.clean_core_interbank import (
    client_payment_flows_for_day as interbank_client_payment_flows_for_day,
)
from bilancio.engines.clean_core_interbank import (
    initial_banking_reserve_targets as interbank_initial_banking_reserve_targets,
)
from bilancio.engines.clean_core_interbank import (
    net_interbank_flows as interbank_net_interbank_flows,
)
from bilancio.engines.clean_core_interbank import (
    primary_bank_for_customer as interbank_primary_bank_for_customer,
)
from bilancio.engines.clean_core_invariants import assert_clean_core_invariants
from bilancio.engines.clean_core_inventory import (
    deliver_stock_for_obligation as inventory_deliver_stock_for_obligation,
)
from bilancio.engines.clean_core_inventory import (
    first_stock_lot_by_sku as inventory_first_stock_lot_by_sku,
)
from bilancio.engines.clean_core_inventory import (
    move_stock_lot as inventory_move_stock_lot,
)
from bilancio.engines.clean_core_rating import compute_rating as rating_compute_rating
from bilancio.engines.clean_core_rating import run_rating_phase as rating_run_rating_phase
from bilancio.engines.clean_core_rollover import (
    rollover_settled_payables as rollover_rollover_settled_payables,
)
from bilancio.engines.clean_core_stability import (
    defaults_on_day as stability_defaults_on_day,
)
from bilancio.engines.clean_core_stability import (
    has_pending_future_obligations as stability_has_pending_future_obligations,
)
from bilancio.engines.clean_core_stability import (
    impacted_on_day as stability_impacted_on_day,
)
from bilancio.engines.clean_core_stability import (
    update_runtime_stability as stability_update_runtime_stability,
)
from bilancio.engines.clean_core_types import (
    CleanAgent as TypeCleanAgent,
)
from bilancio.engines.clean_core_types import (
    CleanBankLoan as TypeCleanBankLoan,
)
from bilancio.engines.clean_core_types import (
    CleanCBLoan as TypeCleanCBLoan,
)
from bilancio.engines.clean_core_types import (
    CleanDealerBucketConfig as TypeCleanDealerBucketConfig,
)
from bilancio.engines.clean_core_types import (
    CleanDealerConfig as TypeCleanDealerConfig,
)
from bilancio.engines.clean_core_types import (
    CleanDeliveryObligation as TypeCleanDeliveryObligation,
)
from bilancio.engines.clean_core_types import (
    CleanNonBankLoan as TypeCleanNonBankLoan,
)
from bilancio.engines.clean_core_types import (
    CleanPayable as TypeCleanPayable,
)
from bilancio.engines.clean_core_types import (
    CleanRatingConfig as TypeCleanRatingConfig,
)
from bilancio.engines.clean_core_types import (
    CleanRunResult as TypeCleanRunResult,
)
from bilancio.engines.clean_core_types import (
    CleanState as TypeCleanState,
)
from bilancio.engines.clean_core_types import (
    CleanStockLot as TypeCleanStockLot,
)
from bilancio.engines.clean_core_views import (
    balance_rows as view_balance_rows,
)
from bilancio.engines.clean_core_views import (
    t_account_rows as view_t_account_rows,
)
from bilancio.engines.termination import StopReason

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "scenarios"


def _rows_by_agent(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["agent_id"]: row for row in rows}


def test_clean_core_invariants_accept_balanced_rows() -> None:
    result = run_basic_scenario(load_yaml(EXAMPLES_DIR / "simple_bank.yaml"))

    assert_clean_core_invariants(balance_rows(result))


def test_clean_core_invariants_reject_missing_system_row() -> None:
    with pytest.raises(ValidationError, match="missing SYSTEM balance row"):
        assert_clean_core_invariants([{"agent_id": "H1", "assets_cash": 1}])


def test_clean_core_invariants_reject_system_imbalance() -> None:
    rows = [
        {
            "agent_id": "SYSTEM",
            "total_financial_assets": "10",
            "total_financial_liabilities": "9",
        }
    ]

    with pytest.raises(ValidationError, match="system assets 10 != liabilities 9"):
        assert_clean_core_invariants(rows)


def test_clean_core_invariants_reject_negative_assets() -> None:
    rows = [
        {
            "agent_id": "SYSTEM",
            "total_financial_assets": "0",
            "total_financial_liabilities": "0",
        },
        {"agent_id": "H1", "assets_cash": "-1"},
    ]

    with pytest.raises(ValidationError, match="negative assets_cash for H1"):
        assert_clean_core_invariants(rows)


def test_clean_core_compat_has_no_generated_banking_config_without_banks() -> None:
    assert build_clean_core_banking_config(EXAMPLES_DIR / "simple_bank.yaml") is None


def test_clean_core_compat_builds_generated_banking_config(tmp_path: Path) -> None:
    path = tmp_path / "generated_banking.yaml"
    path.write_text(
        """
version: 1
name: Generated Banking
agents:
  - {id: CB, kind: central_bank, name: Central Bank}
  - {id: B1, kind: bank, name: Bank One}
  - {id: H1, kind: household, name: Household One}
initial_actions: []
_balanced_config:
  n_banks: 1
  kappa: "0.5"
  maturity_days: 7
  credit_risk_loading: "0.1"
  max_borrower_risk: "0.8"
  min_coverage_ratio: "0.2"
  enable_bank_lending: true
  trader_bank_assignments:
    H1: [B1]
  infra_bank_assignments:
    vbt_short: B1
run:
  mode: until_stable
  max_days: 5
  quiet_days: 1
""".lstrip()
    )
    banking_config = build_clean_core_banking_config(path)

    assert banking_config is not None
    assert banking_config.kappa == Decimal("0.5")
    assert banking_config.maturity_days == 7
    assert banking_config.credit_risk_loading == Decimal("0.1")
    assert banking_config.max_borrower_risk == Decimal("0.8")
    assert banking_config.min_coverage_ratio == Decimal("0.2")
    assert banking_config.enable_bank_lending is True
    assert banking_config.trader_bank_assignments == {"H1": ["B1"]}
    assert banking_config.infra_bank_assignments == {"vbt_short": "B1"}


def test_clean_core_compat_auto_accepts_supported_scenario() -> None:
    assert clean_core_auto_fallback_reason(EXAMPLES_DIR / "simple_bank.yaml", None) is None


def test_clean_core_compat_auto_reports_clean_core_unsupported_reason(tmp_path: Path) -> None:
    path = tmp_path / "unsupported_action.yaml"
    path.write_text(
        """
version: 1
name: Unsupported Action
agents:
  - {id: F1, kind: firm, name: Firm One}
initial_actions:
  - unsupported_action: {}
run:
  mode: until_stable
  max_days: 1
  quiet_days: 1
""".lstrip()
    )

    reason = clean_core_auto_fallback_reason(path, None)

    assert reason == "clean core does not support initial action: unsupported_action"


def test_clean_core_config_module_handles_support_checks_and_action_precedence(tmp_path: Path) -> None:
    action_name, payload = config_select_action_payload(
        {
            "unsupported_action": {},
            "mint_cash": {"to": "F1", "amount": 5},
        }
    )
    assert action_name == "mint_cash"
    assert payload == {"to": "F1", "amount": 5}

    supported_config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    assert config_clean_core_unsupported_reason(supported_config) is None
    assert config_build_lender_config(supported_config) is None

    path = tmp_path / "unsupported_direct.yaml"
    path.write_text(
        """
version: 1
name: Unsupported Direct
agents:
  - {id: F1, kind: firm, name: Firm One}
initial_actions:
  - unsupported_action: {}
run:
  mode: until_stable
  max_days: 1
  quiet_days: 1
""".lstrip()
    )
    unsupported_config = load_yaml(path)

    assert (
        config_clean_core_unsupported_reason(unsupported_config)
        == "clean core does not support initial action: unsupported_action"
    )


def test_clean_core_type_module_exposes_state_and_computed_properties() -> None:
    state = TypeCleanState()
    state.day = 3

    state.log("ExampleEvent", amount=Decimal("10"))
    state.log_setup("SetupEvent", amount=Decimal("2"))
    result = TypeCleanRunResult(state, final_day=3, reached_stable=True)

    assert result.events == [
        {"kind": "ExampleEvent", "day": 3, "phase": "simulation", "amount": Decimal("10")},
        {"kind": "SetupEvent", "day": 0, "phase": "setup", "amount": Decimal("2")},
    ]
    assert result.stop_reason is StopReason.STABILITY_REACHED
    assert result.stop_day == 3
    assert TypeCleanCBLoan(
        id="loan-1",
        bank="B1",
        central_bank="CB",
        amount=Decimal("100"),
        rate=Decimal("0.07"),
        issuance_day=1,
    ).interest_amount == Decimal("7")
    assert TypeCleanBankLoan(
        id="bank-loan-1",
        bank="B1",
        borrower="F1",
        amount=Decimal("100"),
        rate=Decimal("0.025"),
        issuance_day=1,
        maturity_day=5,
    ).repayment_amount == Decimal("102")
    assert TypeCleanStockLot(
        id="stock-1",
        owner="F1",
        sku="ITEM",
        quantity=4,
        unit_price=Decimal("2.50"),
    ).value == Decimal("10.00")


def test_clean_core_cash_module_splits_transfers_and_merges_lots() -> None:
    state = TypeCleanState()
    state.day = 4
    state.cash["A"] = Decimal("10")
    state.cash_lots["A"] = [Decimal("4"), Decimal("6")]
    state.cash["B"] = Decimal("2")
    state.cash_lots["B"] = [Decimal("2")]
    events: list[dict[str, Any]] = []

    transferred = cash_transfer_cash_with_events(
        state,
        "A",
        "B",
        Decimal("5"),
        log=lambda kind, **payload: events.append({"kind": kind, **payload}),
    )

    assert transferred == Decimal("5")
    assert state.cash["A"] == Decimal("5")
    assert state.cash_lots["A"] == [Decimal("5")]
    assert state.cash["B"] == Decimal("7")
    assert state.cash_lots["B"] == [Decimal("7")]
    assert [event["kind"] for event in events] == [
        "CashTransferred",
        "CashTransferred",
        "InstrumentMerged",
        "InstrumentMerged",
    ]

    fallback_state = TypeCleanState()
    fallback_state.cash["C"] = Decimal("3")
    assert cash_take_cash_lots(fallback_state, "C", Decimal("2")) == [Decimal("2")]
    assert fallback_state.cash_lots["C"] == [Decimal("1")]


def test_clean_core_inventory_module_splits_and_delivers_stock_lots() -> None:
    state = TypeCleanState()
    state.day = 2
    state.stocks["S0"] = TypeCleanStockLot(
        id="S0",
        owner="F1",
        sku="ITEM",
        quantity=5,
        unit_price=Decimal("3"),
    )
    events: list[dict[str, Any]] = []

    moving_id = inventory_move_stock_lot(
        state,
        state.stocks["S0"],
        "F1",
        "F2",
        2,
        log=lambda kind, **payload: events.append({"kind": kind, **payload}),
    )

    assert moving_id == "S_split_S0_2_1"
    assert state.stocks["S0"].quantity == 3
    assert state.stocks[moving_id].owner == "F2"
    assert [event["kind"] for event in events] == ["StockSplit"]

    state.stocks["S1"] = TypeCleanStockLot(
        id="S1",
        owner="F1",
        sku="ITEM",
        quantity=4,
        unit_price=Decimal("3"),
    )
    assert inventory_first_stock_lot_by_sku(state, "F1", "ITEM").id == "S0"

    delivered = inventory_deliver_stock_for_obligation(
        state,
        debtor="F1",
        creditor="F2",
        sku="ITEM",
        quantity=5,
    )

    assert delivered == 5
    assert state.stocks["S0"].owner == "F2"
    assert state.stocks["S1"].quantity == 2
    assert state.stocks["S_split_S1_2_3"].owner == "F2"
    assert [event["kind"] for event in state.events] == [
        "StockTransferred",
        "StockSplit",
        "StockTransferred",
    ]


def test_clean_core_rating_module_publishes_balance_sheet_ratings() -> None:
    state = TypeCleanState()
    state.day = 1
    state.agents = {
        "RA": TypeCleanAgent(id="RA", kind="rating_agency", name="Rating Agency"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
        "F2": TypeCleanAgent(id="F2", kind="firm", name="Firm Two"),
    }
    state.cash["F1"] = Decimal("100")
    state.payables.append(
        TypeCleanPayable(
            id="P1",
            debtor="F1",
            creditor="F2",
            amount=Decimal("80"),
            due_day=2,
            maturity_distance=1,
        )
    )
    config = TypeCleanRatingConfig(
        info_profile="omniscient",
        lookback_window=3,
        balance_sheet_weight=Decimal("1"),
        history_weight=Decimal("0"),
        conservatism_bias=Decimal("0"),
        coverage_fraction=Decimal("1.0"),
    )

    events = rating_run_rating_phase(state, config)

    assert events == [
        {
            "kind": "RatingsPublished",
            "day": 1,
            "agency_id": "RA",
            "n_rated": 2,
            "n_eligible": 2,
            "ratings": {"F2": "0.02", "F1": "0.4500"},
        }
    ]
    assert state.rating_registry == {
        "F1": rating_compute_rating(state, "F1", config),
        "F2": rating_compute_rating(state, "F2", config),
    }


def test_clean_core_interbank_module_nets_flows_and_builds_auction_state() -> None:
    state = TypeCleanState()
    state.day = 2
    state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "B2": TypeCleanAgent(id="B2", kind="bank", name="Bank Two"),
        "H1": TypeCleanAgent(id="H1", kind="household", name="Household One"),
        "H2": TypeCleanAgent(id="H2", kind="household", name="Household Two"),
    }
    state.deposits[("H1", "B1")] = Decimal("100")
    state.deposits[("H2", "B2")] = Decimal("50")
    state.reserves["B1"] = Decimal("80")
    state.reserves["B2"] = Decimal("10")
    state.events.extend(
        [
            {
                "kind": "ClientPayment",
                "day": 2,
                "payer_bank": "B1",
                "payee_bank": "B2",
                "amount": Decimal("100"),
            },
            {
                "kind": "ClientPayment",
                "day": 2,
                "payer_bank": "B2",
                "payee_bank": "B1",
                "amount": Decimal("40"),
            },
            {
                "kind": "ClientPayment",
                "day": 1,
                "payer_bank": "B1",
                "payee_bank": "B2",
                "amount": Decimal("999"),
            },
            {
                "kind": "ClientPayment",
                "day": 2,
                "payer_bank": "B1",
                "payee_bank": "B1",
                "amount": Decimal("999"),
            },
        ]
    )

    flows = interbank_client_payment_flows_for_day(state)
    net_flows = interbank_net_interbank_flows(flows)

    assert interbank_primary_bank_for_customer(state, "H1") == "B1"
    assert interbank_primary_bank_for_customer(state, "missing") is None
    assert flows == {
        ("B1", "B2"): Decimal("100"),
        ("B2", "B1"): Decimal("40"),
    }
    assert net_flows == {("B1", "B2"): Decimal("60")}
    assert interbank_initial_banking_reserve_targets(
        state,
        CleanBankingConfig(reserve_target_ratio=Decimal("0.10")),
    ) == {"B1": 10, "B2": 5}

    auction = interbank_clean_interbank_auction_summary(
        state,
        CleanBankingConfig(reserve_targets={"B1": 50, "B2": 50}),
        net_flows,
    )
    positions = {
        row["bank_id"]: row["position"]
        for row in auction["market_state"]["positions"]
    }

    assert positions == {"B1": -30, "B2": 20}
    assert auction["market_state"]["borrower_bids"][0]["bank_id"] == "B1"
    assert auction["market_state"]["lender_asks"][0]["bank_id"] == "B2"


def test_clean_core_contracts_module_resolves_aliases_and_transfers_claims() -> None:
    state = TypeCleanState()
    state.payables.append(
        TypeCleanPayable(
            id="P1",
            debtor="F1",
            creditor="F2",
            amount=Decimal("25"),
            due_day=3,
            maturity_distance=2,
            alias="PAY_ALIAS",
        )
    )
    state.delivery_obligations.append(
        TypeCleanDeliveryObligation(
            id="D1",
            debtor="F1",
            creditor="F2",
            sku="ITEM",
            quantity=4,
            unit_price=Decimal("3"),
            due_day=5,
            alias="DELIVERY_ALIAS",
        )
    )
    events: list[dict[str, Any]] = []

    assert contracts_contract_id_for_alias(state, "PAY_ALIAS") == "P1"
    assert contracts_contract_id_for_alias(state, "DELIVERY_ALIAS") == "D1"
    assert contracts_contract_id_for_alias(state, "missing") is None
    assert contracts_action_references_agent(
        {"create_payable": {"from": "F1", "to": "F2"}},
        "F1",
    )
    assert not contracts_action_references_agent(
        {"create_payable": {"from": "F1", "to": "F2"}},
        "F3",
    )

    payable_transferred = contracts_transfer_payable_claim(
        state,
        "P1",
        "F3",
        alias="PAY_ALIAS",
        log=lambda kind, **payload: events.append({"kind": kind, **payload}),
    )
    delivery_transferred = contracts_transfer_delivery_claim(
        state,
        "D1",
        "F4",
        alias="DELIVERY_ALIAS",
        log=lambda kind, **payload: events.append({"kind": kind, **payload}),
    )

    assert payable_transferred is True
    assert delivery_transferred is True
    assert contracts_transfer_payable_claim(
        state,
        "missing",
        "F5",
        alias=None,
        log=lambda kind, **payload: events.append({"kind": kind, **payload}),
    ) is False
    assert state.payables[0].creditor == "F3"
    assert state.delivery_obligations[0].creditor == "F4"
    assert events == [
        {
            "kind": "ClaimTransferred",
            "contract_id": "P1",
            "frm": "F2",
            "to": "F3",
            "contract_kind": "payable",
            "amount": Decimal("25"),
            "due_day": 3,
            "sku": None,
            "alias": "PAY_ALIAS",
        },
        {
            "kind": "ClaimTransferred",
            "contract_id": "D1",
            "frm": "F2",
            "to": "F4",
            "contract_kind": "delivery_obligation",
            "amount": 4,
            "due_day": 5,
            "sku": "ITEM",
            "alias": "DELIVERY_ALIAS",
        },
    ]


def test_clean_core_rollover_module_creates_full_and_partial_replacement_payables() -> None:
    state = TypeCleanState()
    state.day = 3
    state.agents = {
        "D": TypeCleanAgent(id="D", kind="firm", name="Debtor"),
        "C": TypeCleanAgent(id="C", kind="firm", name="Creditor"),
    }
    state.payables.append(
        TypeCleanPayable(
            id="P_existing",
            debtor="D",
            creditor="C",
            amount=Decimal("1"),
            due_day=4,
            maturity_distance=1,
        )
    )
    state.deposits[("C", "B1")] = Decimal("40")
    state.cash["C"] = Decimal("10")
    state.cash_lots["C"] = [Decimal("10")]

    def pay_with_deposit(
        state: TypeCleanState,
        payer: str,
        payee: str,
        amount: Decimal,
    ) -> Decimal:
        paid = min(state.deposits[(payer, "B1")], amount)
        if paid <= 0:
            return Decimal("0")
        state.deposits[(payer, "B1")] -= paid
        state.deposits[(payee, "B1")] += paid
        state.log("IntraBankPayment", payer=payer, payee=payee, bank="B1", amount=paid)
        return paid

    created = rollover_rollover_settled_payables(
        state,
        [
            ("D", "C", Decimal("50"), 2),
            ("D", "C", Decimal("30"), 1),
            ("missing", "C", Decimal("10"), 1),
        ],
        pay_with_deposit=pay_with_deposit,
        transfer_cash_with_events=cash_transfer_cash_with_events,
    )

    assert created == ["PAY_rollover_1", "PAY_rollover_2"]
    assert [(payable.id, payable.due_day) for payable in state.payables] == [
        ("P_existing", 4),
        ("PAY_rollover_1", 6),
        ("PAY_rollover_2", 5),
    ]
    assert state.deposits[("C", "B1")] == Decimal("0")
    assert state.deposits[("D", "B1")] == Decimal("40")
    assert state.cash["C"] == Decimal("0")
    assert state.cash["D"] == Decimal("10")
    assert [event["kind"] for event in state.events] == [
        "IntraBankPayment",
        "CashTransferred",
        "PayableRolledOver",
        "RolloverPartial",
    ]
    assert state.events[-1]["cash_transferred"] == Decimal("0")


def test_clean_core_lender_module_measures_risk_terms_and_ranking() -> None:
    config = config_build_lender_config(load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml"))
    assert config is not None
    state = TypeCleanState()
    state.day = 2
    state.agents = {
        "lender": TypeCleanAgent(id="lender", kind="non_bank_lender", name="Lender"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
        "F2": TypeCleanAgent(id="F2", kind="firm", name="Firm Two"),
    }
    state.cash["F1"] = Decimal("20")
    state.deposits[("F1", "B1")] = Decimal("30")
    state.payables.extend(
        [
            TypeCleanPayable(
                id="P_due",
                debtor="F1",
                creditor="F2",
                amount=Decimal("100"),
                due_day=3,
                maturity_distance=1,
            ),
            TypeCleanPayable(
                id="P_receivable",
                debtor="F2",
                creditor="F1",
                amount=Decimal("40"),
                due_day=4,
                maturity_distance=2,
            ),
        ]
    )
    state.non_bank_loans.append(
        TypeCleanNonBankLoan(
            id="NL1",
            lender="lender",
            borrower="F1",
            amount=Decimal("10"),
            rate=Decimal("0.10"),
            issuance_day=0,
            maturity_days=2,
        )
    )
    state.bank_loans.append(
        TypeCleanBankLoan(
            id="BL1",
            bank="B1",
            borrower="F1",
            amount=Decimal("20"),
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=3,
        )
    )
    state.rating_registry["F1"] = Decimal("0.40")
    state.rating_registry["F2"] = Decimal("0.80")

    assert lender_helpers.active_lender_id(state) == "lender"
    assert lender_helpers.agent_liquid_assets(state, "F1") == Decimal("50")
    assert lender_helpers.upcoming_obligations(state, "F1", horizon=2) == Decimal("132")
    assert lender_helpers.quality_adjusted_receivables(state, "F1", horizon=5) == Decimal("40")
    assert lender_helpers.assess_non_bank_borrower(
        state,
        "F1",
        Decimal("50"),
        Decimal("0.10"),
        horizon=2,
    ) == Decimal("-21") / Decimal("55")
    assert lender_helpers.observe_lender_counterparty_liquidity(state, config, "F1") == (
        Decimal("111"),
        Decimal("50"),
    )
    assert lender_helpers.lender_observed_default_probability(state, config, "F1") == Decimal("0.40")
    assert lender_helpers.lender_default_probability(
        state,
        config,
        "F1",
        Decimal("111"),
        Decimal("50"),
    ) == Decimal("0.8222222222222222222222222220")
    assert lender_helpers.lender_loan_rate(config, Decimal("0.40")) == Decimal("0.138")
    assert lender_helpers.preventive_lender_loan_rate(config, Decimal("0.40")) == Decimal("0.138")
    assert lender_helpers.nearest_receivable_day(state, "F1", max_horizon=5) == 4
    assert lender_helpers.downstream_obligation_total(state, "F1") == Decimal("100")
    assert lender_helpers.receivables_at_risk(
        state,
        config,
        "F1",
        horizon=5,
        threshold=Decimal("0.5"),
    ) == Decimal("40")
    assert lender_helpers.count_existing_non_bank_loans(state, "lender", "F1") == 1
    assert lender_helpers.resolve_non_bank_loan_terms(
        state,
        config,
        {"borrower_id": "F1", "rate": Decimal("0.10"), "p_default": Decimal("0.80")},
    ) == (Decimal("0.10"), 2)

    opportunities = [
        {"expected_profit": 0.1, "downstream": 10, "coverage_ratio": Decimal("0.2"), "p_default": Decimal("0.10")},
        {"expected_profit": 0.2, "downstream": 1, "coverage_ratio": Decimal("1.0"), "p_default": Decimal("0.20")},
    ]
    lender_helpers.rank_lending_opportunities(opportunities, config)
    assert [opportunity["expected_profit"] for opportunity in opportunities] == [0.2, 0.1]

    matching_config = replace(config, maturity_matching=True)
    assert lender_helpers.resolve_preventive_non_bank_loan_maturity(
        state,
        matching_config,
        "F1",
        Decimal("0.80"),
    ) == 2


def test_clean_core_banking_module_quotes_and_screens_bank_lending() -> None:
    state = TypeCleanState()
    state.day = 1
    state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "B2": TypeCleanAgent(id="B2", kind="bank", name="Bank Two"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
        "F2": TypeCleanAgent(id="F2", kind="firm", name="Firm Two"),
    }
    state.cash["F1"] = Decimal("10")
    state.deposits[("F1", "B1")] = Decimal("20")
    state.deposits[("F2", "B2")] = Decimal("100")
    state.reserves["B1"] = Decimal("500")
    state.reserves["B2"] = Decimal("200")
    state.payables.extend(
        [
            TypeCleanPayable(
                id="P_due",
                debtor="F1",
                creditor="F2",
                amount=Decimal("80"),
                due_day=1,
                maturity_distance=1,
            ),
            TypeCleanPayable(
                id="P_recv",
                debtor="F2",
                creditor="F1",
                amount=Decimal("40"),
                due_day=4,
                maturity_distance=2,
            ),
        ]
    )
    state.bank_loans.append(
        TypeCleanBankLoan(
            id="BL0",
            bank="B1",
            borrower="F1",
            amount=Decimal("30"),
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=3,
        )
    )
    config = CleanBankingConfig(
        reserve_target_ratio=Decimal("0.10"),
        trader_bank_assignments={"F1": ["B1"]},
    )
    profile = banking_helpers.clean_bank_profile(config)
    quote, params = banking_helpers.clean_bank_quote(state, "B1", config, profile)

    assert banking_helpers.clean_agent_banks(state, "F1", config) == ["B1"]
    assert banking_helpers.clean_bank_loan_maturity(config) == 5
    assert banking_helpers.clean_bank_deposits_total(state, "B1") == Decimal("20")
    assert banking_helpers.clean_bank_withdrawal_forecast(state, "B1", n_banks=2) == Decimal("10")
    assert banking_helpers.clean_bank_settlement_forecast(state, "B1") == Decimal("80")
    assert banking_helpers.find_clean_bank_borrowers(state, horizon=5) == [("F1", Decimal("81"))]
    assert banking_helpers.clean_cheapest_loan_bank(state, "F1", config) is not None
    assert params.reserve_target == 2
    assert quote.day == state.day
    assert banking_helpers.clean_bank_borrower_rate(
        Decimal("0.05"),
        "F1",
        banking_helpers.clean_bank_profile(
            CleanBankingConfig(credit_risk_loading=Decimal("0.2")),
        ),
        state.day,
    ) == Decimal("0.080")
    assert banking_helpers.assess_clean_bank_borrower(
        state,
        "F1",
        Decimal("50"),
        Decimal("0.10"),
        loan_maturity=5,
    ) == Decimal("-41") / Decimal("55")
    assert banking_helpers.clean_bank_can_lend(
        state,
        "B1",
        "F1",
        Decimal("20"),
        profile,
        params,
    ) is True


def test_clean_core_banking_module_runs_bank_lending_phase() -> None:
    state = TypeCleanState()
    state.day = 0
    state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
        "F2": TypeCleanAgent(id="F2", kind="firm", name="Firm Two"),
    }
    state.reserves["B1"] = Decimal("1000")
    state.deposits[("F2", "B1")] = Decimal("1000")
    state.payables.append(
        TypeCleanPayable(
            id="P1",
            debtor="F1",
            creditor="F2",
            amount=Decimal("100"),
            due_day=1,
            maturity_distance=1,
        )
    )

    banking_helpers.run_bank_lending_phase(state, CleanBankingConfig())

    assert len(state.bank_loans) == 1
    loan = state.bank_loans[0]
    assert loan.id == "BL_0"
    assert loan.bank == "B1"
    assert loan.borrower == "F1"
    assert loan.amount == Decimal("100")
    assert loan.maturity_day == 5
    assert state.deposits[("F1", "B1")] == Decimal("100")
    assert _event_exists(
        state.events,
        kind="BankLoanIssued",
        bank="B1",
        borrower="F1",
        amount=Decimal("100"),
        loan_id="BL_0",
    )


def test_clean_core_banking_module_repays_and_defaults_bank_loans() -> None:
    repayment_state = TypeCleanState()
    repayment_state.day = 2
    repayment_state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "B2": TypeCleanAgent(id="B2", kind="bank", name="Bank Two"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
    }
    repayment_state.reserves["B1"] = Decimal("100")
    repayment_state.reserves["B2"] = Decimal("100")
    repayment_state.deposits[("F1", "B1")] = Decimal("50")
    repayment_state.deposits[("F1", "B2")] = Decimal("60")
    repayment_state.bank_loans.append(
        TypeCleanBankLoan(
            id="BL0",
            bank="B1",
            borrower="F1",
            amount=Decimal("100"),
            rate=Decimal("0.10"),
            issuance_day=0,
            maturity_day=2,
        )
    )

    banking_helpers.repay_due_bank_loans(repayment_state)

    assert repayment_state.bank_loans[0].settled is True
    assert repayment_state.deposits[("F1", "B1")] == Decimal("0")
    assert repayment_state.deposits[("F1", "B2")] == Decimal("0")
    assert repayment_state.reserves["B1"] == Decimal("160")
    assert repayment_state.reserves["B2"] == Decimal("40")
    assert _event_exists(
        repayment_state.events,
        kind="BankLoanRepaid",
        bank="B1",
        borrower="F1",
        principal=Decimal("100"),
        repayment=Decimal("110"),
    )

    default_state = TypeCleanState()
    default_state.day = 2
    default_state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
    }
    default_state.deposits[("F1", "B1")] = Decimal("20")
    default_state.bank_loans.append(
        TypeCleanBankLoan(
            id="BL1",
            bank="B1",
            borrower="F1",
            amount=Decimal("100"),
            rate=Decimal("0.10"),
            issuance_day=0,
            maturity_day=2,
        )
    )

    banking_helpers.repay_due_bank_loans(default_state)

    assert default_state.bank_loans[0].settled is True
    assert default_state.deposits[("F1", "B1")] == Decimal("0")
    assert "F1" in default_state.bank_defaulted_borrowers
    assert _event_exists(
        default_state.events,
        kind="BankLoanDefault",
        bank="B1",
        borrower="F1",
        repayment_due=Decimal("110"),
        recovered=Decimal("20"),
    )


def test_clean_core_central_bank_module_repays_due_cb_loans() -> None:
    state = TypeCleanState()
    state.day = 2
    state.central_bank_id = "CB"
    state.agents = {
        "CB": TypeCleanAgent(id="CB", kind="central_bank", name="Central Bank"),
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
    }
    state.reserves["B1"] = Decimal("103")
    state.cb_reserves_outstanding = Decimal("100")
    state.cb_loans_outstanding = Decimal("100")
    state.cb_loans.append(
        TypeCleanCBLoan(
            id="CBL0",
            bank="B1",
            central_bank="CB",
            amount=Decimal("100"),
            rate=Decimal("0.03"),
            issuance_day=0,
        )
    )

    cb_helpers.repay_due_cb_loans(state)

    assert state.cb_loans[0].settled is True
    assert state.reserves["B1"] == Decimal("0.00")
    assert state.cb_loans_outstanding == Decimal("0")
    assert state.cb_interest_total_paid == Decimal("3.00")
    assert _event_exists(
        state.events,
        kind="CBLoanRepaid",
        bank_id="B1",
        loan_id="CBL0",
        principal=Decimal("100"),
        total_repaid=Decimal("103.00"),
    )


def test_clean_core_central_bank_module_resolves_failed_bank_deposits() -> None:
    state = TypeCleanState()
    state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "B2": TypeCleanAgent(id="B2", kind="bank", name="Bank Two"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
    }
    state.reserves["B1"] = Decimal("0")
    state.reserves["B2"] = Decimal("20")
    state.cb_reserves_outstanding = Decimal("20")
    state.deposits[("F1", "B2")] = Decimal("40")
    state.deposits[("F1", "B1")] = Decimal("0")

    cb_helpers.resolve_clean_failed_bank(state, "B2")
    cb_helpers.write_off_clean_bank_liabilities(state, "B2")

    assert state.reserves["B2"] == Decimal("0")
    assert state.reserves["B1"] == Decimal("20")
    assert state.deposits[("F1", "B1")] == Decimal("20")
    assert ("F1", "B2") not in state.deposits
    assert _event_exists(
        state.events,
        kind="ReservesTransferred",
        frm="B2",
        to="B1",
        amount=Decimal("20"),
    )
    assert _event_exists(state.events, kind="BankResolutionCompleted", bank_id="B2")


def test_clean_core_dealer_module_initializes_marker_metrics() -> None:
    state = TypeCleanState()
    state.agents = {
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
        "F2": TypeCleanAgent(id="F2", kind="firm", name="Firm Two"),
    }
    state.cash["F1"] = Decimal("100")
    state.payables.append(
        TypeCleanPayable(
            id="P1",
            debtor="F1",
            creditor="F2",
            amount=Decimal("50"),
            due_day=1,
            maturity_distance=1,
        )
    )
    config = TypeCleanDealerConfig(
        ticket_size=Decimal("10"),
        buckets=(
            TypeCleanDealerBucketConfig(
                name="short",
                tau_min=0,
                tau_max=3,
                mid=Decimal("0.90"),
                spread=Decimal("0.05"),
            ),
        ),
        dealer_share=Decimal("0.25"),
        vbt_share=Decimal("0.75"),
        risk_enabled=False,
        lookback_window=3,
        smoothing_alpha=Decimal("1"),
        initial_prior=Decimal("0.15"),
    )
    state.dealer_config = config

    dealer_helpers._initialize_clean_dealer_marker(state, config)

    assert "dealer_short" in state.agents
    assert "vbt_short" in state.agents
    assert state.dealer_metrics is not None
    assert state.dealer_metrics.initial_total_debt == Decimal("50")
    assert state.dealer_metrics.initial_total_money == Decimal("100")
    assert state.dealer_metrics.initial_equity_by_bucket["short"] == Decimal("25.00")
    assert dealer_helpers.dealer_metrics_summary(state)["initial_total_debt"] == 50.0


def test_clean_core_actions_module_applies_yaml_actions() -> None:
    state = TypeCleanState()
    state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
    }

    assert action_helpers._as_decimal("10.9") == Decimal("10")

    action_helpers._apply_action(
        state,
        {"mint_cash": {"to": "F1", "amount": "100.9", "alias": "seed_cash"}},
        index=0,
        setup=True,
    )
    action_helpers._apply_action(
        state,
        {"deposit_cash": {"customer": "F1", "bank": "B1", "amount": 40}},
        index=1,
        setup=False,
    )

    assert state.cash["F1"] == Decimal("60")
    assert state.cash["B1"] == Decimal("40")
    assert state.deposits[("F1", "B1")] == Decimal("40")
    assert _event_exists(
        state.events,
        kind="CashMinted",
        phase="setup",
        to="F1",
        amount=Decimal("100"),
        alias="seed_cash",
    )
    assert _event_exists(
        state.events,
        kind="CashDeposited",
        phase="simulation",
        customer="F1",
        bank="B1",
        amount=Decimal("40"),
    )


def test_clean_core_settlement_module_settles_payable_with_deposit() -> None:
    state = TypeCleanState()
    state.agents = {
        "B1": TypeCleanAgent(id="B1", kind="bank", name="Bank One"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
        "F2": TypeCleanAgent(id="F2", kind="firm", name="Firm Two"),
    }
    state.deposits[("F1", "B1")] = Decimal("75")
    payable = TypeCleanPayable(
        id="P1",
        debtor="F1",
        creditor="F2",
        amount=Decimal("50"),
        due_day=0,
        maturity_distance=1,
    )
    state.payables.append(payable)

    settled, rollover_info = settlement_helpers._settle_payable(
        state,
        payable,
        {"firm": ["bank_deposit"]},
    )

    assert settled is True
    assert rollover_info is None
    assert payable.settled is True
    assert state.deposits[("F1", "B1")] == Decimal("25")
    assert state.deposits[("F2", "B1")] == Decimal("50")
    assert _event_exists(
        state.events,
        kind="IntraBankPayment",
        payer="F1",
        payee="F2",
        bank="B1",
        amount=Decimal("50"),
    )
    assert _event_exists(
        state.events,
        kind="PayableSettled",
        contract_id="P1",
        debtor="F1",
        creditor="F2",
        amount=Decimal("50"),
    )


def test_clean_core_lending_phase_module_creates_and_repays_non_bank_loan() -> None:
    state = TypeCleanState()
    state.agents = {
        "lender": TypeCleanAgent(id="lender", kind="non_bank_lender", name="Lender"),
        "F1": TypeCleanAgent(id="F1", kind="firm", name="Firm One"),
    }
    state.cash["lender"] = Decimal("100")
    state.cash_lots["lender"].append(Decimal("100"))

    loan_id = lending_phase_helpers._create_non_bank_loan(
        state,
        lender_id="lender",
        borrower_id="F1",
        amount=Decimal("40"),
        rate=Decimal("0.10"),
        maturity_days=1,
    )

    assert loan_id == "NBL_0"
    assert state.cash["lender"] == Decimal("60")
    assert state.cash["F1"] == Decimal("40")
    assert len(state.non_bank_loans) == 1
    assert _event_exists(
        state.events,
        kind="NonBankLoanCreated",
        lender_id="lender",
        borrower_id="F1",
        amount=Decimal("40"),
        loan_id="NBL_0",
    )

    state.cash["F1"] += Decimal("10")
    state.cash_lots["F1"].append(Decimal("10"))
    state.day = 1

    assert lending_phase_helpers._repay_due_non_bank_loans(state) is True
    assert state.non_bank_loans[0].settled is True
    assert state.cash["F1"] == Decimal("6")
    assert state.cash["lender"] == Decimal("104")
    assert _event_exists(
        state.events,
        kind="NonBankLoanRepaid",
        loan_id="NBL_0",
        borrower_id="F1",
        lender_id="lender",
        total_repaid=Decimal("44"),
    )


def _amount(row: dict[str, Any], field: str) -> Decimal:
    return Decimal(str(row.get(field, 0)))


def _event_exists(events: list[dict[str, Any]], **expected: Any) -> bool:
    return any(all(event.get(key) == value for key, value in expected.items()) for event in events)


def test_clean_core_matches_simple_bank_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 4
    assert result.stop_reason is StopReason.STABILITY_REACHED
    assert result.stop_day == result.final_day
    assert len(result.stability_snapshots) == result.final_day
    assert result.stability_snapshots[-1].consecutive_quiet >= 2
    assert len(result.events) == 28
    assert Counter(event["kind"] for event in result.events) == Counter(
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
    assert _event_exists(result.events, kind="IntraBankPayment", day=1, payer="H1", payee="H2", bank="B1", amount=500)
    assert _event_exists(result.events, kind="PayableSettled", day=1, debtor="H1", creditor="H2", amount=500)

    assert _amount(rows["CB"], "liabilities_cash") == Decimal("3500")
    assert _amount(rows["CB"], "liabilities_reserve_deposit") == Decimal("10000")
    assert _amount(rows["B1"], "assets_cash") == Decimal("2800")
    assert _amount(rows["B1"], "assets_reserve_deposit") == Decimal("10000")
    assert _amount(rows["B1"], "liabilities_bank_deposit") == Decimal("2800")
    assert _amount(rows["H1"], "assets_bank_deposit") == Decimal("1300")
    assert _amount(rows["H1"], "assets_cash") == Decimal("200")
    assert _amount(rows["H2"], "assets_bank_deposit") == Decimal("1500")
    assert _amount(rows["H2"], "assets_cash") == Decimal("500")


def test_clean_core_records_max_days_stop_reason_and_snapshots() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_cash": {"to": "H1", "amount": 100}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 10, "due_day": 10}},
            ],
            "scheduled_actions": [],
        }
    )

    result = run_basic_scenario(config, max_days=2, quiet_days=1)

    assert result.reached_stable is False
    assert result.stop_reason is StopReason.MAX_DAYS_REACHED
    assert result.stop_day == 2
    assert [snapshot.day for snapshot in result.stability_snapshots] == [0, 1]
    assert result.stability_snapshots[-1].has_open_obligations is True


def test_clean_core_runtime_until_stable_reports_progress_and_snapshots() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    runtime = prepare_scenario(config)
    calls: list[tuple[int, int]] = []
    day_calls: list[int] = []

    result = run_runtime_until_stable(
        runtime,
        max_days=5,
        quiet_days=2,
        progress_callback=lambda current, total: calls.append((current, total)),
        day_callback=lambda _runtime, day: day_calls.append(day),
    )

    assert result.reached_stable is True
    assert result.stop_reason is StopReason.STABILITY_REACHED
    assert calls == [(day, 5) for day in range(1, result.final_day + 1)]
    assert day_calls == list(range(result.final_day))
    assert len(result.stability_snapshots) == result.final_day


def test_clean_core_step_stability_uses_shared_tracker() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    runtime = prepare_scenario(config)
    tracker = CleanStabilityTracker()

    for day in range(3):
        impactful = run_day(runtime, day)
        stable = update_runtime_stability(
            runtime,
            day=day,
            impactful=impactful,
            quiet_days=1,
            tracker=tracker,
        )

    assert stable is True
    assert runtime.state.quiet_days == 1
    assert [snapshot.day for snapshot in tracker.snapshots] == [0, 1, 2]
    assert tracker.snapshots[-1].consecutive_quiet == 1
    assert tracker.snapshots[-1].has_open_obligations is False


def test_clean_core_stability_module_tracks_events_and_updates_runtime() -> None:
    state = TypeCleanState()
    state.day = 1
    state.payables.append(
        TypeCleanPayable(
            id="payable-1",
            debtor="F1",
            creditor="F2",
            amount=Decimal("10"),
            due_day=2,
            maturity_distance=1,
        )
    )
    state.events.extend(
        [
            {"kind": "PayableSettled", "day": 1, "phase": "simulation"},
            {"kind": "ObligationDefaulted", "day": 1, "phase": "simulation"},
        ]
    )

    assert stability_has_pending_future_obligations(state) is True
    assert stability_impacted_on_day(state, 1) == 1
    assert stability_defaults_on_day(state, 1) == 1

    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    runtime = prepare_scenario(config)
    tracker = CleanStabilityTracker()

    for day in range(3):
        impactful = run_day(runtime, day)
        stable = stability_update_runtime_stability(
            runtime,
            day=day,
            impactful=impactful,
            quiet_days=1,
            tracker=tracker,
        )

    assert stable is True
    assert runtime.state.quiet_days == 1
    assert [snapshot.day for snapshot in tracker.snapshots] == [0, 1, 2]
    assert tracker.snapshots[-1].has_open_obligations is False


def test_clean_core_builds_t_account_rows_from_simple_bank_state() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")

    result = run_basic_scenario(config, max_days=5, quiet_days=2)

    household_rows = t_account_rows(result.state, "H1")
    assert household_rows["assets"] == [
        {
            "name": "cash",
            "quantity": None,
            "value_minor": 200,
            "counterparty_name": "-",
            "maturity": "on-demand",
            "id_or_alias": "cash:H1:0",
        },
        {
            "name": "bank_deposit",
            "quantity": None,
            "value_minor": 1300,
            "counterparty_name": "First National Bank [B1]",
            "maturity": "on-demand",
            "id_or_alias": "deposit:H1:B1",
        },
    ]
    assert household_rows["liabs"] == []

    central_bank_rows = t_account_rows(result.state, "CB")
    assert {
        "name": "reserve_deposit",
        "quantity": None,
        "value_minor": 10000,
        "counterparty_name": "First National Bank [B1]",
        "maturity": "on-demand",
        "id_or_alias": "reserve:B1",
    } in central_bank_rows["liabs"]


def test_clean_core_view_module_builds_balances_and_t_accounts_directly() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    result = run_basic_scenario(config, max_days=5, quiet_days=2)

    rows = _rows_by_agent(view_balance_rows(result))
    assert rows == _rows_by_agent(balance_rows(result))
    assert rows["SYSTEM"]["total_financial_assets"] == rows["SYSTEM"]["total_financial_liabilities"]

    household_rows = view_t_account_rows(result.state, "H1")
    assert household_rows == t_account_rows(result.state, "H1")
    assert [row["name"] for row in household_rows["assets"]] == ["cash", "bank_deposit"]
    assert household_rows["liabs"] == []


def test_clean_core_rolls_over_settled_payables() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "run": config.run.model_copy(
                update={"rollover_enabled": True, "max_days": 10, "quiet_days": 2}
            )
        }
    )

    result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert result.reached_stable is True
    assert result.final_day == 2
    assert [event["kind"] for event in result.events] == [
        "ReservesMinted",
        "CashMinted",
        "CashMinted",
        "CashDeposited",
        "CashDeposited",
        "PayableCreated",
        "PhaseA",
        "PhaseB",
        "SubphaseB1",
        "SubphaseB2",
        "PhaseC",
        "PhaseA",
        "PhaseB",
        "SubphaseB1",
        "SubphaseB2",
        "IntraBankPayment",
        "PayableSettled",
        "SubphaseB_Rollover",
        "IntraBankPayment",
        "PayableRolledOver",
        "PhaseC",
    ]
    open_payables = [payable for payable in result.state.payables if not payable.settled]
    assert len(open_payables) == 1
    rolled = open_payables[0]
    assert rolled.debtor == "H1"
    assert rolled.creditor == "H2"
    assert rolled.amount == Decimal("500")
    assert rolled.due_day == 2
    assert rolled.maturity_distance == 1


def test_clean_core_coerces_non_integral_financial_amounts_like_legacy(tmp_path: Path) -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_reserves": {"to": "B1", "amount": Decimal("100.25")}},
                {"mint_cash": {"to": "H1", "amount": Decimal("20.50")}},
                {"mint_cash": {"to": "H2", "amount": Decimal("1.25")}},
                {"deposit_cash": {"customer": "H1", "bank": "B1", "amount": Decimal("20.50")}},
                {"deposit_cash": {"customer": "H2", "bank": "B1", "amount": Decimal("1.25")}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": Decimal("12.75"), "due_day": 1}},
            ]
        }
    )

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))
    events_path = tmp_path / "events.jsonl"
    write_events_jsonl(result, events_path)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]

    assert result.reached_stable is True
    assert _event_exists(
        result.events,
        kind="IntraBankPayment",
        day=1,
        payer="H1",
        payee="H2",
        bank="B1",
        amount=12,
    )
    assert _amount(rows["CB"], "liabilities_cash") == Decimal("21")
    assert _amount(rows["CB"], "liabilities_reserve_deposit") == Decimal("100")
    assert _amount(rows["B1"], "assets_cash") == Decimal("21")
    assert _amount(rows["B1"], "assets_reserve_deposit") == Decimal("100")
    assert _amount(rows["H1"], "assets_bank_deposit") == Decimal("8")
    assert _amount(rows["H2"], "assets_bank_deposit") == Decimal("13")
    assert next(event for event in events if event["kind"] == "PayableCreated")["amount"] == 12


def test_clean_core_matches_sasa_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "sasa_scenario.yaml")

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 4
    assert len(result.events) == 33
    assert Counter(event["kind"] for event in result.events) == Counter(
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
    assert _event_exists(result.events, kind="CashTransferred", day=1, frm="FIRM_A", to="FIRM_B", amount=100)
    assert _event_exists(
        result.events,
        kind="ClientPayment",
        day=1,
        payer="FIRM_A",
        payer_bank="BANK_A",
        payee="FIRM_B",
        payee_bank="BANK_B",
        amount=50,
    )
    assert _event_exists(result.events, kind="ReservesTransferred", day=1, frm="BANK_A", to="BANK_B", amount=50)
    assert _event_exists(
        result.events,
        kind="InterbankCleared",
        day=1,
        debtor_bank="BANK_A",
        creditor_bank="BANK_B",
        amount=50,
    )

    assert _amount(rows["BANK_A"], "assets_reserve_deposit") == Decimal("950")
    assert _amount(rows["BANK_A"], "liabilities_bank_deposit") == Decimal("50")
    assert _amount(rows["BANK_B"], "assets_reserve_deposit") == Decimal("1050")
    assert _amount(rows["BANK_B"], "liabilities_bank_deposit") == Decimal("150")
    assert _amount(rows["FIRM_A"], "assets_bank_deposit") == Decimal("50")
    assert _amount(rows["FIRM_A"], "total_financial_liabilities") == Decimal("0")
    assert _amount(rows["FIRM_B"], "assets_bank_deposit") == Decimal("150")
    assert _amount(rows["FIRM_B"], "assets_cash") == Decimal("100")


def test_clean_core_matches_interbank_netting_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "interbank_netting.yaml")

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 5
    assert len(result.events) == 59
    assert Counter(event["kind"] for event in result.events) == Counter(
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
    assert _event_exists(result.events, kind="InterbankCleared", day=1, debtor_bank="B1", creditor_bank="B2", amount=700)
    assert _event_exists(result.events, kind="InterbankCleared", day=2, debtor_bank="B2", creditor_bank="B1", amount=600)

    assert _amount(rows["B1"], "assets_reserve_deposit") == Decimal("9900")
    assert _amount(rows["B1"], "liabilities_bank_deposit") == Decimal("8900")
    assert _amount(rows["B2"], "assets_reserve_deposit") == Decimal("10100")
    assert _amount(rows["B2"], "liabilities_bank_deposit") == Decimal("9100")
    assert _amount(rows["H1"], "assets_bank_deposit") == Decimal("5000")
    assert _amount(rows["H2"], "assets_bank_deposit") == Decimal("4800")
    assert _amount(rows["H3"], "assets_bank_deposit") == Decimal("3900")
    assert _amount(rows["H4"], "assets_bank_deposit") == Decimal("4300")


def test_clean_core_matches_two_jurisdictions_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "two_jurisdictions.yaml")

    result = run_basic_scenario(config, max_days=8, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 6
    assert len(result.events) == 38
    assert Counter(event["kind"] for event in result.events) == Counter(
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
    assert _event_exists(result.events, kind="CashTransferred", day=3, frm="F_US", to="F_EU", amount=1000)
    assert _event_exists(result.events, kind="PayableSettled", day=3, debtor="F_US", creditor="F_EU", amount=1000)

    assert _amount(rows["CB_US"], "liabilities_cash") == Decimal("9000")
    assert _amount(rows["CB_US"], "liabilities_reserve_deposit") == Decimal("18000")
    assert _amount(rows["CB_EU"], "net_financial") == Decimal("0")
    assert _amount(rows["B_US"], "assets_reserve_deposit") == Decimal("10000")
    assert _amount(rows["B_EU"], "assets_reserve_deposit") == Decimal("8000")
    assert _amount(rows["F_US"], "assets_cash") == Decimal("4000")
    assert _amount(rows["F_EU"], "assets_cash") == Decimal("5000")


def test_clean_core_matches_firm_delivery_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "firm_delivery.yaml")

    result = run_basic_scenario(config, max_days=8, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 6
    assert len(result.events) == 65
    assert Counter(event["kind"] for event in result.events) == Counter(
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
    assert _event_exists(result.events, kind="StockTransferred", day=1, frm="F1", to="H1", sku="WIDGET", qty=10)
    assert _event_exists(result.events, kind="StockTransferred", day=2, frm="F1", to="F2", sku="WIDGET", qty=20)
    assert _event_exists(result.events, kind="StockTransferred", day=3, frm="F2", to="H2", sku="GADGET", qty=5)
    assert _event_exists(result.events, kind="PayableSettled", day=1, debtor="H1", creditor="F1", amount=250)
    assert _event_exists(result.events, kind="PayableSettled", day=2, debtor="F2", creditor="F1", amount=500)
    assert _event_exists(result.events, kind="PayableSettled", day=3, debtor="H2", creditor="F2", amount=500)

    assert _amount(rows["H1"], "inventory_WIDGET_quantity") == Decimal("10")
    assert _amount(rows["H1"], "inventory_WIDGET_value") == Decimal("250")
    assert _amount(rows["H2"], "inventory_GADGET_quantity") == Decimal("5")
    assert _amount(rows["H2"], "inventory_GADGET_value") == Decimal("500")
    assert _amount(rows["F1"], "inventory_WIDGET_quantity") == Decimal("70")
    assert _amount(rows["F1"], "assets_bank_deposit") == Decimal("8750")
    assert _amount(rows["F2"], "inventory_WIDGET_quantity") == Decimal("20")
    assert _amount(rows["F2"], "inventory_GADGET_quantity") == Decimal("45")
    assert _amount(rows["F2"], "assets_bank_deposit") == Decimal("6000")


def test_clean_core_matches_intraday_netting_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "intraday_netting.yaml")

    result = run_basic_scenario(config, max_days=8, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 5
    assert len(result.events) == 58
    assert Counter(event["kind"] for event in result.events) == Counter(
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
    assert _event_exists(result.events, kind="CashTransferred", day=1, frm="F1", to="F2", amount=2000)
    assert _event_exists(result.events, kind="CashTransferred", day=1, frm="F2", to="F1", amount=1500)
    assert _event_exists(result.events, kind="StockTransferred", day=2, frm="F1", to="F2", sku="WIDGET", qty=10)
    assert _event_exists(result.events, kind="StockTransferred", day=2, frm="F2", to="F1", sku="GADGET", qty=15)

    assert _amount(rows["F1"], "assets_bank_deposit") == Decimal("8500")
    assert _amount(rows["F1"], "assets_cash") == Decimal("1500")
    assert _amount(rows["F1"], "inventory_WIDGET_quantity") == Decimal("90")
    assert _amount(rows["F1"], "inventory_GADGET_quantity") == Decimal("15")
    assert _amount(rows["F2"], "assets_bank_deposit") == Decimal("8000")
    assert _amount(rows["F2"], "assets_cash") == Decimal("2200")
    assert _amount(rows["F2"], "inventory_GADGET_quantity") == Decimal("85")
    assert _amount(rows["F2"], "inventory_WIDGET_quantity") == Decimal("10")
    assert _amount(rows["H1"], "assets_bank_deposit") == Decimal("3500")
    assert _amount(rows["H1"], "assets_cash") == Decimal("1300")


def test_clean_core_matches_payment_demo_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "payment_demo.yaml")

    result = run_basic_scenario(config, max_days=10, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 6
    assert len(result.events) == 67
    assert Counter(event["kind"] for event in result.events) == Counter(
        {
            "PhaseA": 6,
            "PhaseB": 6,
            "SubphaseB1": 6,
            "SubphaseB2": 6,
            "PhaseC": 6,
            "CashDeposited": 4,
            "CashMinted": 4,
            "ClientPayment": 4,
            "InstrumentMerged": 2,
            "InterbankCleared": 2,
            "IntraBankPayment": 3,
            "PayableCreated": 7,
            "PayableSettled": 7,
            "ReservesMinted": 2,
            "ReservesTransferred": 2,
        }
    )

    assert _amount(rows["BANK_A"], "assets_reserve_deposit") == Decimal("6300")
    assert _amount(rows["BANK_A"], "liabilities_bank_deposit") == Decimal("4300")
    assert _amount(rows["BANK_B"], "assets_reserve_deposit") == Decimal("13700")
    assert _amount(rows["BANK_B"], "liabilities_bank_deposit") == Decimal("11700")
    assert _amount(rows["ALICE"], "assets_bank_deposit") == Decimal("3600")
    assert _amount(rows["BOB"], "assets_bank_deposit") == Decimal("700")
    assert _amount(rows["CHARLIE"], "assets_bank_deposit") == Decimal("4600")
    assert _amount(rows["DIANA"], "assets_bank_deposit") == Decimal("7100")


def test_clean_core_matches_two_banks_interbank_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "two_banks_interbank.yaml")

    result = run_basic_scenario(config, max_days=10, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 5
    assert len(result.events) == 45
    assert Counter(event["kind"] for event in result.events) == Counter(
        {
            "PhaseA": 5,
            "PhaseB": 5,
            "SubphaseB1": 5,
            "SubphaseB2": 5,
            "PhaseC": 5,
            "CashDeposited": 3,
            "CashMinted": 3,
            "ClientPayment": 2,
            "InstrumentMerged": 1,
            "InterbankCleared": 1,
            "IntraBankPayment": 1,
            "PayableCreated": 3,
            "PayableSettled": 3,
            "ReservesMinted": 2,
            "ReservesTransferred": 1,
        }
    )
    assert _event_exists(result.events, kind="InterbankCleared", day=1, debtor_bank="B1", creditor_bank="B2", amount=200)

    assert _amount(rows["B1"], "assets_reserve_deposit") == Decimal("19800")
    assert _amount(rows["B1"], "liabilities_bank_deposit") == Decimal("6800")
    assert _amount(rows["B2"], "assets_reserve_deposit") == Decimal("15200")
    assert _amount(rows["B2"], "liabilities_bank_deposit") == Decimal("3700")
    assert _amount(rows["H1"], "assets_bank_deposit") == Decimal("3800")
    assert _amount(rows["H2"], "assets_bank_deposit") == Decimal("3700")
    assert _amount(rows["H3"], "assets_bank_deposit") == Decimal("3000")


def test_clean_core_matches_rich_simulation_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "rich_simulation.yaml")

    result = run_basic_scenario(config, max_days=10, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 7
    assert len(result.events) == 131
    assert Counter(event["kind"] for event in result.events) == Counter(
        {
            "PhaseA": 7,
            "PhaseB": 7,
            "SubphaseB1": 7,
            "SubphaseB2": 7,
            "PhaseC": 7,
            "CashDeposited": 6,
            "CashMinted": 6,
            "CashTransferred": 6,
            "ClientPayment": 4,
            "DeliveryObligationCancelled": 4,
            "DeliveryObligationCreated": 4,
            "DeliveryObligationSettled": 4,
            "InstrumentMerged": 8,
            "InterbankCleared": 2,
            "IntraBankPayment": 6,
            "PayableCreated": 16,
            "PayableSettled": 16,
            "ReservesMinted": 2,
            "ReservesTransferred": 2,
            "StockCreated": 2,
            "StockSplit": 4,
            "StockTransferred": 4,
        }
    )

    assert _amount(rows["BANK1"], "assets_reserve_deposit") == Decimal("49800")
    assert _amount(rows["BANK2"], "assets_reserve_deposit") == Decimal("50200")
    assert _amount(rows["H1"], "inventory_TECH_STOCK_quantity") == Decimal("100")
    assert _amount(rows["H2"], "inventory_RETAIL_STOCK_quantity") == Decimal("50")
    assert _amount(rows["H3"], "inventory_TECH_STOCK_quantity") == Decimal("150")
    assert _amount(rows["H4"], "inventory_RETAIL_STOCK_quantity") == Decimal("75")
    assert _amount(rows["FIRM1"], "inventory_TECH_STOCK_quantity") == Decimal("750")
    assert _amount(rows["FIRM1"], "assets_bank_deposit") == Decimal("15300")
    assert _amount(rows["FIRM2"], "inventory_RETAIL_STOCK_quantity") == Decimal("375")
    assert _amount(rows["FIRM2"], "assets_bank_deposit") == Decimal("15600")


def test_clean_core_matches_ring_with_action_specs_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "ring_with_action_specs.yaml")

    result = run_basic_scenario(config, max_days=10, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 6
    assert len(result.events) == 54
    assert Counter(event["kind"] for event in result.events) == Counter(
        {
            "PhaseA": 6,
            "PhaseB": 6,
            "SubphaseB1": 6,
            "SubphaseB2": 6,
            "PhaseC": 6,
            "CashMinted": 5,
            "CashTransferred": 5,
            "InstrumentMerged": 4,
            "PayableCreated": 5,
            "PayableSettled": 5,
        }
    )

    assert _amount(rows["CB"], "liabilities_cash") == Decimal("1000")
    for agent_id in ["H1", "H2", "H3", "H4", "H5"]:
        assert _amount(rows[agent_id], "assets_cash") == Decimal("200")


def test_clean_core_matches_default_handling_contract() -> None:
    config = load_yaml(EXAMPLES_DIR / "default_handling_demo.yaml")

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert result.final_day == 2
    assert len(result.events) == 19
    assert Counter(event["kind"] for event in result.events) == Counter(
        {
            "PhaseA": 2,
            "PhaseB": 2,
            "SubphaseB1": 2,
            "SubphaseB2": 2,
            "PhaseC": 2,
            "CashMinted": 1,
            "CashTransferred": 1,
            "PayableCreated": 2,
            "PartialSettlement": 1,
            "ObligationDefaulted": 1,
            "AgentDefaulted": 1,
            "ObligationWrittenOff": 1,
            "ScheduledActionCancelled": 1,
        }
    )
    assert _event_exists(
        result.events,
        kind="PartialSettlement",
        day=1,
        debtor="F1",
        creditor="F2",
        amount_paid=60,
        shortfall=40,
        original_amount=100,
    )
    assert _event_exists(
        result.events,
        kind="ObligationDefaulted",
        day=1,
        debtor="F1",
        creditor="F2",
        shortfall=40,
        amount_paid=60,
        amount=40,
    )
    assert _event_exists(result.events, kind="AgentDefaulted", day=1, agent="F1", shortfall=40, mode="expel-agent")
    assert _event_exists(result.events, kind="ObligationWrittenOff", day=1, debtor="F1", creditor="F3", amount=50)
    assert _event_exists(
        result.events,
        kind="ScheduledActionCancelled",
        day=1,
        agent="F1",
        scheduled_day=2,
        action="mint_cash",
        mode="expel-agent",
    )

    assert _amount(rows["CB"], "liabilities_cash") == Decimal("60")
    assert _amount(rows["F1"], "total_financial_assets") == Decimal("0")
    assert _amount(rows["F1"], "total_financial_liabilities") == Decimal("0")
    assert _amount(rows["F2"], "assets_cash") == Decimal("60")
    assert _amount(rows["F3"], "total_financial_assets") == Decimal("0")


def test_clean_core_writes_legacy_shaped_exports(tmp_path: Path) -> None:
    config = load_yaml(EXAMPLES_DIR / "firm_delivery.yaml")
    result = run_basic_scenario(config, max_days=8, quiet_days=2)

    balances_path = tmp_path / "balances.csv"
    events_path = tmp_path / "events.jsonl"
    write_balances_csv(result, balances_path)
    write_events_jsonl(result, events_path)

    with balances_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows_by_agent = {
        row["agent_id"]: row
        for row in rows
        if row["agent_id"] != "SYSTEM" and row.get("item_type") != "summary"
    }
    summary_rows = [row for row in rows if row["agent_id"] == "SYSTEM" and row.get("item_type") == "summary"]

    assert len(rows) == 10
    assert [row["item_name"] for row in summary_rows] == ["Total Assets", "Total Liabilities", "Total Equity"]
    assert Decimal(rows_by_agent["F1"]["assets_bank_deposit"]) == Decimal("8750")
    assert Decimal(rows_by_agent["F1"]["inventory_WIDGET_quantity"]) == Decimal("70")
    assert Decimal(rows_by_agent["F2"]["inventory_GADGET_value"]) == Decimal("4500.0")

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 65
    assert events[0]["kind"] == "ReservesMinted"
    assert any(event["kind"] == "DeliveryObligationSettled" for event in events)


def test_clean_core_writes_html_report(tmp_path: Path) -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    result = run_basic_scenario(config, max_days=10, quiet_days=2)
    html_path = tmp_path / "clean_nbfi.html"

    write_html_report(result, config, html_path, max_days=10, quiet_days=2)

    html = html_path.read_text(encoding="utf-8")
    assert "Bilancio Simulation" in html
    assert "NBFI Lending Demo" in html
    assert "Final Balances" in html
    assert "NonBankLoanCreated" in html
    assert "assets_cash" in html


def test_clean_core_writes_t_account_html_report(tmp_path: Path) -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    html_path = tmp_path / "clean_t_account.html"

    write_html_report(
        result,
        config,
        html_path,
        max_days=5,
        quiet_days=2,
        agent_ids=["H1"],
        t_account=True,
    )

    html = html_path.read_text(encoding="utf-8")
    assert "Final T-Accounts" in html
    assert "Smith Family [H1] (household)" in html
    assert "bank_deposit" in html
    assert "First National Bank [B1]" in html


def test_clean_core_export_module_writes_outputs_directly(tmp_path: Path) -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    balances_path = tmp_path / "direct_balances.csv"
    events_path = tmp_path / "direct_events.jsonl"
    html_path = tmp_path / "direct_report.html"

    write_export_balances_csv(result, balances_path)
    write_export_events_jsonl(result, events_path)
    write_export_html_report(
        result,
        config,
        html_path,
        max_days=5,
        quiet_days=2,
        agent_ids=["H1"],
        t_account=True,
    )

    with balances_path.open(newline="") as f:
        balance_rows = list(csv.DictReader(f))
    assert any(row["agent_id"] == "SYSTEM" for row in balance_rows)

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert any(event["kind"] == "PayableSettled" for event in events)

    html = html_path.read_text(encoding="utf-8")
    assert "Final T-Accounts" in html
    assert "Smith Family [H1]" in html


def test_clean_core_rejects_unsupported_action() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                *config.initial_actions,
                {"create_bank_loan": {"bank": "B1", "borrower": "H1", "amount": 10}},
            ]
        }
    )

    with pytest.raises(NotImplementedError, match="create_bank_loan"):
        run_basic_scenario(config, max_days=5, quiet_days=2)


def test_clean_core_fail_fast_default_raises_legacy_default_error() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"mint_cash": {"to": "H1", "amount": 10}},
                {"create_payable": {"from": "H1", "to": "H2", "amount": 15, "due_day": 1}},
            ]
        }
    )

    with pytest.raises(DefaultError, match="Insufficient funds to settle payable"):
        run_basic_scenario(config, max_days=5, quiet_days=2)


def test_clean_core_supports_direct_dealer_marker_and_metrics() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(update={"dealer": DealerConfig(enabled=True)})

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert _event_exists(result.events, kind="SubphaseB_Dealer")
    assert "dealer_short" in rows
    assert "vbt_short" in rows
    assert result.state.dealer_metrics is not None
    summary = result.state.dealer_metrics.summary()
    assert summary["total_trades"] == 0
    assert summary["initial_total_money"] > 0


def test_clean_core_rejects_unsupported_action_specs_phase() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "action_specs": [
                ActionSpecConfig(
                    kind="household",
                    profile_type="trader",
                    actions=[ActionDefConfig(action="sell_ticket", phase="B_Dealer")],
                )
            ]
        }
    )

    assert (
        config_clean_core_configuration_error_reason(config)
        == "action_specs request B_Dealer phase but no balanced_dealer config is present. "
        "Include a balanced_dealer section with enabled=true in the scenario, "
        "or use the ring compiler with emit_action_specs=True which emits both."
    )

    with pytest.raises(ConfigurationError, match="action_specs request B_Dealer phase"):
        run_basic_scenario(config, max_days=5, quiet_days=2)


def test_clean_core_supports_lending_action_specs_phase() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    config = config.model_copy(
        update={
            "lender": None,
            "action_specs": [
                ActionSpecConfig(
                    kind="non_bank_lender",
                    profile_type="lender",
                    actions=[ActionDefConfig(action="lend", phase="B_Lending")],
                    information="omniscient",
                    profile_params={
                        "base_rate": "0.05",
                        "risk_premium_scale": "0.20",
                        "max_single_exposure": "0.30",
                        "max_total_exposure": "0.80",
                        "maturity_days": 3,
                        "horizon": 3,
                        "kappa": "0.5",
                        "risk_aversion": "0.3",
                        "planning_horizon": 5,
                        "profit_target": "0.05",
                        "min_coverage_ratio": "0",
                    },
                )
            ],
        }
    )

    result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert _event_exists(result.events, kind="SubphaseB_Lending")
    assert _event_exists(result.events, kind="NonBankLoanCreated", lender_id="lender")


def test_clean_core_uses_lender_profile_when_lending_action_specs_include_borrowers() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    action_spec_config = config.model_copy(
        update={
            "lender": None,
            "action_specs": [
                ActionSpecConfig(
                    kind="household",
                    profile_type="trader",
                    profile_params={
                        "risk_aversion": "0.5",
                        "buy_reserve_fraction": "0.5",
                        "trading_motive": "liquidity_then_earning",
                    },
                    actions=[
                        ActionDefConfig(action="settle", phase="B2_Settlement"),
                        ActionDefConfig(action="borrow", phase="B_Lending"),
                    ],
                ),
                ActionSpecConfig(
                    kind="non_bank_lender",
                    profile_type="lender",
                    actions=[ActionDefConfig(action="lend", phase="B_Lending")],
                    information="omniscient",
                    profile_params={
                        "base_rate": "0.05",
                        "risk_premium_scale": "0.20",
                        "max_single_exposure": "0.30",
                        "max_total_exposure": "0.80",
                        "maturity_days": 3,
                        "horizon": 3,
                        "kappa": "0.5",
                        "risk_aversion": "0.3",
                        "planning_horizon": 5,
                        "profit_target": "0.05",
                        "min_coverage_ratio": "0",
                    },
                ),
            ],
        }
    )

    expected = run_basic_scenario(config, max_days=10, quiet_days=2)
    actual = run_basic_scenario(action_spec_config, max_days=10, quiet_days=2)

    assert [event["kind"] for event in actual.events] == [
        event["kind"] for event in expected.events
    ]
    assert _rows_by_agent(balance_rows(actual)) == _rows_by_agent(balance_rows(expected))


def test_clean_core_supports_realistic_lending_action_specs_information() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    config = config.model_copy(
        update={
            "lender": None,
            "action_specs": [
                ActionSpecConfig(
                    kind="non_bank_lender",
                    profile_type="lender",
                    actions=[ActionDefConfig(action="lend", phase="B_Lending")],
                    information="realistic",
                    profile_params={
                        "base_rate": "0.05",
                        "risk_premium_scale": "0.20",
                        "max_single_exposure": "0.30",
                        "max_total_exposure": "0.80",
                        "maturity_days": 3,
                        "horizon": 3,
                        "kappa": "0.5",
                        "risk_aversion": "0.3",
                        "planning_horizon": 5,
                        "profit_target": "0.05",
                        "min_coverage_ratio": "0",
                    },
                )
            ],
        }
    )

    runtime = prepare_scenario(config)
    assert runtime.lender_config is not None
    assert runtime.lender_config.info_cash_visibility == "noisy"
    assert runtime.lender_config.info_cash_noise == Decimal("0.15")
    assert runtime.lender_config.info_liabilities_visibility == "noisy"
    assert runtime.lender_config.info_history_visibility == "noisy"
    assert runtime.lender_config.info_history_sample_rate == Decimal("0.7")

    result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert result.reached_stable is True
    assert _event_exists(result.events, kind="SubphaseB_Lending")
    assert _event_exists(result.events, kind="NonBankLoanCreated", lender_id="lender")


def test_clean_core_cb_lending_cutoff_defaults_bank_when_refinancing_needed() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "initial_actions": [
                {"create_cb_loan": {"bank": "B1", "amount": 100}},
            ],
            "scheduled_actions": [],
        }
    )
    runtime = prepare_scenario(
        config,
        banking_config=CleanBankingConfig(cb_lending_cutoff_day=1),
    )

    run_day(runtime, 0)
    run_day(runtime, 1)
    run_day(runtime, 2)

    assert runtime.state.cb_lending_frozen is True
    assert "B1" in runtime.state.defaulted_agent_ids
    assert _event_exists(runtime.state.events, kind="CBLendingFreezeActivated", cutoff_day=1)
    assert _event_exists(runtime.state.events, kind="CBLendingFrozen", bank_id="B1", amount=103)
    assert _event_exists(runtime.state.events, kind="BankDefaultCBFreeze", bank_id="B1", amount=100)
    assert _event_exists(runtime.state.events, kind="CBLoanFreezeWrittenOff", bank_id="B1", amount=100)


def test_clean_core_supports_rating_action_specs_phase() -> None:
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
                    information="omniscient",
                    profile_params={"coverage_fraction": "1.0", "no_data_prior": "0.25"},
                )
            ],
        }
    )

    result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert _event_exists(result.events, kind="SubphaseB_Rating")
    assert _event_exists(result.events, kind="RatingsPublished", agency_id="RA", n_rated=2)


def test_clean_core_rejects_unsupported_rating_action_specs_profile_params() -> None:
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
                    information="omniscient",
                    profile_params={"unsupported": 1},
                )
            ],
        }
    )

    with pytest.raises(NotImplementedError, match="profile params: unsupported"):
        run_basic_scenario(config, max_days=5, quiet_days=2)


def test_clean_core_supports_omniscient_rating_agency() -> None:
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

    result = run_basic_scenario(config, max_days=5, quiet_days=2)

    assert result.reached_stable is True
    assert "SubphaseB_Rating" in [event["kind"] for event in result.events]
    rating_events = [event for event in result.events if event["kind"] == "RatingsPublished"]
    assert rating_events
    assert rating_events[0]["agency_id"] == "RA"
    assert rating_events[0]["n_rated"] == 2
    assert set(result.state.rating_registry) == {"H1", "H2"}


def test_clean_core_supports_realistic_rating_agency() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_bank.yaml")
    config = config.model_copy(
        update={
            "agents": [
                *config.agents,
                AgentSpec(id="RA", kind="rating_agency", name="Rating Agency"),
            ],
            "rating_agency": RatingAgencyScenarioConfig(enabled=True),
        }
    )

    result = run_basic_scenario(config, max_days=5, quiet_days=2)
    rating_events = [event for event in result.events if event["kind"] == "RatingsPublished"]

    assert result.reached_stable is True
    assert rating_events
    assert all(Decimal("0.01") <= Decimal(value) <= Decimal("0.99") for value in rating_events[0]["ratings"].values())


def test_clean_core_supports_simple_cash_lender_scenario() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    result = run_basic_scenario(config, max_days=10, quiet_days=2)
    rows = _rows_by_agent(balance_rows(result))

    assert result.reached_stable is True
    assert Counter(event["kind"] for event in result.events)["NonBankLoanCreated"] == 8
    assert Counter(event["kind"] for event in result.events)["NonBankLoanRepaid"] == 3
    assert _amount(rows["lender"], "assets_cash") == Decimal("295")
    assert _amount(rows["lender"], "assets_non_bank_loan") == Decimal("0")
    assert _amount(rows["H2"], "assets_cash") == Decimal("214")


def test_clean_core_supports_noisy_cash_lender_information() -> None:
    config = load_yaml(EXAMPLES_DIR / "simple_nbfi.yaml")
    assert config.lender is not None
    config = config.model_copy(
        update={"lender": config.lender.model_copy(update={"info_cash_visibility": "noisy"})}
    )

    result = run_basic_scenario(config, max_days=10, quiet_days=2)

    assert result.reached_stable is True
    assert Counter(event["kind"] for event in result.events)["NonBankLoanCreated"] > 0
