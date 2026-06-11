"""Clean-room scenario engine used to rebuild Bilancio in tested slices.

This module is intentionally scoped.  It implements the public scenario
contract covered by the characterization tests: cash/reserve creation,
deposits, payables, delivery obligations, basic CB loans, claim transfer,
bank-mediated settlement, scenario-level ratings, the scenario-level
non-bank lender contract, a dealer marker/metrics slice, and the first
balanced active dealer trading slice. Richer market/network information
subsystems still fail fast until rebuilt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from bilancio.config.models import ScenarioConfig
from bilancio.engines import clean_core_actions as _actions
from bilancio.engines import clean_core_banking as _banking
from bilancio.engines import clean_core_central_bank as _central_bank
from bilancio.engines import clean_core_contracts as _contracts
from bilancio.engines import clean_core_dealer as _dealer
from bilancio.engines import clean_core_interbank as _interbank
from bilancio.engines import clean_core_lender as _lender
from bilancio.engines import clean_core_lending_phase as _lending_phase
from bilancio.engines import clean_core_rating as _rating
from bilancio.engines import clean_core_settlement as _settlement
from bilancio.engines.clean_core_cash import require_at_least as _require_at_least
from bilancio.engines.clean_core_config import (
    build_dealer_config as _build_dealer_config,
)
from bilancio.engines.clean_core_config import (
    build_lender_config as _build_lender_config,
)
from bilancio.engines.clean_core_config import (
    build_rating_config as _build_rating_config,
)
from bilancio.engines.clean_core_config import (
    clean_core_configuration_error_reason as _clean_core_configuration_error_reason,
)
from bilancio.engines.clean_core_config import (
    clean_core_unsupported_reason as _clean_core_unsupported_reason,
)
from bilancio.engines.clean_core_config import (
    reject_unsupported_subsystems as _reject_unsupported_subsystems,
)
from bilancio.engines.clean_core_stability import (
    has_pending_future_obligations as _has_pending_future_obligations,
)
from bilancio.engines.clean_core_stability import (
    update_runtime_stability as _update_runtime_stability,
)
from bilancio.engines.clean_core_types import (
    CleanAgent,
    CleanBankingConfig,
    CleanDealerConfig,
    CleanLenderConfig,
    CleanRatingConfig,
    CleanRunResult,
    CleanScenarioRuntime,
    CleanStabilityTracker,
    CleanState,
)
from bilancio.engines.termination import StopReason

ZERO = Decimal("0")

_run_rating_phase = _rating.run_rating_phase
_log_rating_estimates = _rating.log_rating_estimates
_active_rating_agency_id = _rating.active_rating_agency_id
_compute_rating = _rating.compute_rating
_rating_noisy_decimal = _rating.rating_noisy_decimal
_raw_net_worth = _rating.raw_net_worth
_raw_total_assets = _rating.raw_total_assets
_raw_total_liabilities = _rating.raw_total_liabilities
_raw_default_probability = _rating.raw_default_probability
_primary_bank_for_customer = _interbank.primary_bank_for_customer
_client_payment_flows_for_day = _interbank.client_payment_flows_for_day
_net_interbank_flows = _interbank.net_interbank_flows
_initial_banking_reserve_targets = _interbank.initial_banking_reserve_targets
_clean_interbank_auction_summary = _interbank.clean_interbank_auction_summary
_action_references_agent = _contracts.action_references_agent
_contract_id_for_alias = _contracts.contract_id_for_alias
_transfer_payable_claim = _contracts.transfer_payable_claim
_transfer_delivery_claim = _contracts.transfer_delivery_claim
_rank_lending_opportunities = _lender.rank_lending_opportunities
_lending_cascade_score = _lender.lending_cascade_score
_active_lender_id = _lender.active_lender_id
_agent_liquid_assets = _lender.agent_liquid_assets
_upcoming_obligations = _lender.upcoming_obligations
_quality_adjusted_receivables = _lender.quality_adjusted_receivables
_assess_non_bank_borrower = _lender.assess_non_bank_borrower
_lender_uses_information = _lender.lender_uses_information
_observe_lender_counterparty_liquidity = _lender.observe_lender_counterparty_liquidity
_lender_observed_default_probability = _lender.lender_observed_default_probability
_lender_noisy_decimal = _lender.lender_noisy_decimal
_lender_default_probability = _lender.lender_default_probability
_lender_profile_default_probability = _lender.lender_profile_default_probability
_lender_loan_rate = _lender.lender_loan_rate
_preventive_lender_loan_rate = _lender.preventive_lender_loan_rate
_resolve_non_bank_loan_terms = _lender.resolve_non_bank_loan_terms
_resolve_preventive_non_bank_loan_maturity = (
    _lender.resolve_preventive_non_bank_loan_maturity
)
_nearest_receivable_day = _lender.nearest_receivable_day
_downstream_obligation_total = _lender.downstream_obligation_total
_receivables_at_risk = _lender.receivables_at_risk
_lender_signal_default_probability = _lender.lender_signal_default_probability
_count_existing_non_bank_loans = _lender.count_existing_non_bank_loans
_clean_bank_borrower_rate = _banking.clean_bank_borrower_rate
_find_clean_bank_borrowers = _banking.find_clean_bank_borrowers
_clean_cheapest_loan_bank = _banking.clean_cheapest_loan_bank
_clean_agent_banks = _banking.clean_agent_banks
_clean_bank_profile = _banking.clean_bank_profile
_clean_bank_loan_maturity = _banking.clean_bank_loan_maturity
_clean_bank_quote = _banking.clean_bank_quote
_clean_bank_deposits_total = _banking.clean_bank_deposits_total
_clean_bank_withdrawal_forecast = _banking.clean_bank_withdrawal_forecast
_clean_bank_settlement_forecast = _banking.clean_bank_settlement_forecast
_assess_clean_bank_borrower = _banking.assess_clean_bank_borrower
_clean_bank_can_lend = _banking.clean_bank_can_lend
_run_bank_lending_phase = _banking.run_bank_lending_phase
_create_clean_bank_loan = _banking.create_clean_bank_loan
_repay_due_bank_loans = _banking.repay_due_bank_loans
_agent_deposits_total = _banking.agent_deposits_total
_debit_clean_bank_loan_repayment = _banking.debit_clean_bank_loan_repayment
_decrease_clean_deposit = _banking.decrease_clean_deposit
_run_clean_bank_loan_winddown = _banking.run_clean_bank_loan_winddown
_has_outstanding_clean_bank_loans = _banking.has_outstanding_clean_bank_loans
finalize_banking_marker_events = _central_bank.finalize_banking_marker_events
_run_clean_final_cb_settlement = _central_bank.run_clean_final_cb_settlement
_repay_due_cb_loans = _central_bank.repay_due_cb_loans
_refinance_cb_loan_repayment = _central_bank.refinance_cb_loan_repayment
_repay_clean_cb_loan = _central_bank.repay_clean_cb_loan
_write_off_frozen_cb_loan = _central_bank.write_off_frozen_cb_loan
_write_off_clean_cb_loan = _central_bank.write_off_clean_cb_loan
_resolve_clean_failed_bank = _central_bank.resolve_clean_failed_bank
_find_clean_surviving_bank = _central_bank.find_clean_surviving_bank
_write_off_clean_bank_liabilities = _central_bank.write_off_clean_bank_liabilities
_as_decimal = _actions._as_decimal
_unique_clean_id = _actions._unique_clean_id
_apply_action = _actions._apply_action


def clean_core_unsupported_reason(config: ScenarioConfig) -> str | None:
    """Return a user-facing reason when a scenario still needs the legacy engine."""
    return _clean_core_unsupported_reason(config)


def clean_core_configuration_error_reason(config: ScenarioConfig) -> str | None:
    """Return a legacy-compatible setup error for invalid clean-core scenarios."""
    return _clean_core_configuration_error_reason(config)


def prepare_scenario(
    config: ScenarioConfig,
    *,
    banking_config: CleanBankingConfig | None = None,
) -> CleanScenarioRuntime:
    """Apply scenario setup and return a runtime that can be stepped day by day."""
    _reject_unsupported_subsystems(config)
    rating_config = _build_rating_config(config)
    lender_config = _build_lender_config(config)
    dealer_config = _build_dealer_config(config)

    state = CleanState(
        default_mode=config.run.default_handling,
        rollover_enabled=config.run.rollover_enabled,
        estimate_logging_enabled=config.run.estimate_logging,
        dealer_config=dealer_config,
    )
    for agent_spec in config.agents:
        state.agents[agent_spec.id] = CleanAgent(
            id=agent_spec.id,
            kind=str(agent_spec.kind),
            name=agent_spec.name,
        )
        if state.central_bank_id is None and str(agent_spec.kind) == "central_bank":
            state.central_bank_id = agent_spec.id

    policy_order = {
        "household": ["bank_deposit", "cash"],
        "firm": ["cash", "bank_deposit"],
        "bank": ["reserve_deposit"],
        "treasury": ["reserve_deposit"],
        "central_bank": ["reserve_deposit"],
        "non_bank_lender": ["cash"],
        "rating_agency": [],
    }
    if config.policy_overrides and config.policy_overrides.mop_rank:
        policy_order.update(config.policy_overrides.mop_rank)

    for scheduled in config.scheduled_actions:
        state.scheduled_actions_by_day.setdefault(scheduled.day, []).append(scheduled.action)

    for index, action in enumerate(config.initial_actions):
        _apply_action(state, action, index=index, setup=True)

    state.cb_reserves_initial = state.cb_reserves_outstanding
    if dealer_config is not None:
        if dealer_config.balanced_active:
            _initialize_clean_active_dealer_subsystem(state, dealer_config)
        else:
            _initialize_clean_dealer_marker(state, dealer_config)

    if banking_config is not None and not banking_config.reserve_targets:
        banking_config = replace(
            banking_config,
            reserve_targets=_initial_banking_reserve_targets(state, banking_config),
        )

    return CleanScenarioRuntime(
        state=state,
        policy_order=policy_order,
        lender_config=lender_config,
        rating_config=rating_config,
        banking_config=banking_config,
        dealer_config=dealer_config,
    )


_initialize_clean_dealer_marker = _dealer._initialize_clean_dealer_marker
_capture_clean_dealer_marker_snapshots = _dealer._capture_clean_dealer_marker_snapshots
_initialize_clean_active_dealer_subsystem = _dealer._initialize_clean_active_dealer_subsystem
_clean_active_dealer_risk_params = _dealer._clean_active_dealer_risk_params
_clean_active_convert_payables_to_tickets = _dealer._clean_active_convert_payables_to_tickets
_clean_active_add_ticket_for_payable = _dealer._clean_active_add_ticket_for_payable
_clean_active_categorize_market_maker_tickets = _dealer._clean_active_categorize_market_maker_tickets
_clean_active_initialize_market_makers = _dealer._clean_active_initialize_market_makers
_clean_active_mu_tilt_factors = _dealer._clean_active_mu_tilt_factors
_clean_active_initialize_traders = _dealer._clean_active_initialize_traders
_clean_active_capture_initial_debt_to_money = _dealer._clean_active_capture_initial_debt_to_money
_clean_dealer_default_probability = _dealer._clean_dealer_default_probability
_clean_update_dealer_risk_history = _dealer._clean_update_dealer_risk_history
_clean_dealer_face_by_bucket = _dealer._clean_dealer_face_by_bucket
_clean_dealer_inventory_count = _dealer._clean_dealer_inventory_count
_clean_balanced_vbt_mid = _dealer._clean_balanced_vbt_mid
dealer_metrics_summary = _dealer.dealer_metrics_summary
_clean_passive_dealer_pnl = _dealer._clean_passive_dealer_pnl
_clean_dealer_bucket_for_tau = _dealer._clean_dealer_bucket_for_tau
_run_clean_active_dealer_phase = _dealer._run_clean_active_dealer_phase
_clean_active_liquid_balance = _dealer._clean_active_liquid_balance
_clean_active_sync_cash_from_state = _dealer._clean_active_sync_cash_from_state
_clean_active_cleanup_orphaned_tickets = _dealer._clean_active_cleanup_orphaned_tickets
_clean_active_ingest_new_payables = _dealer._clean_active_ingest_new_payables
_clean_active_update_ticket_maturities = _dealer._clean_active_update_ticket_maturities
_clean_active_remove_ticket_from_holdings = _dealer._clean_active_remove_ticket_from_holdings
_clean_active_move_ticket_to_bucket = _dealer._clean_active_move_ticket_to_bucket
_clean_active_execute_trading_rounds = _dealer._clean_active_execute_trading_rounds
_clean_active_sync_payable_ownership = _dealer._clean_active_sync_payable_ownership
_clean_active_sync_cash_to_state = _dealer._clean_active_sync_cash_to_state
_clean_active_sync_one_cash_balance = _dealer._clean_active_sync_one_cash_balance
_clean_active_adjust_deposit_balance = _dealer._clean_active_adjust_deposit_balance
_clean_active_select_sync_deposit_bank = _dealer._clean_active_select_sync_deposit_bank
_clean_payable_for_ticket = _dealer._clean_payable_for_ticket
_clean_active_reassign_payable_owner = _dealer._clean_active_reassign_payable_owner


def run_day(runtime: CleanScenarioRuntime, day: int) -> bool:
    """Run a single clean-core day and return whether it had impactful settlement events."""
    runtime.state.day = day
    if (
        runtime.banking_config is not None
        and runtime.banking_config.cb_lending_cutoff_day is not None
        and day >= runtime.banking_config.cb_lending_cutoff_day
        and not runtime.state.cb_lending_frozen
    ):
        runtime.state.cb_lending_frozen = True
        runtime.state.log(
            "CBLendingFreezeActivated",
            cutoff_day=runtime.banking_config.cb_lending_cutoff_day,
        )
    return _run_day(
        runtime.state,
        runtime.policy_order,
        runtime.lender_config,
        runtime.rating_config,
        runtime.banking_config,
        runtime.dealer_config,
    )


def has_pending_future_obligations(state: CleanState) -> bool:
    """Return whether any unsettled obligation or loan remains after the current day."""
    return _has_pending_future_obligations(state)


def run_basic_scenario(
    config: ScenarioConfig,
    *,
    max_days: int | None = None,
    quiet_days: int | None = None,
) -> CleanRunResult:
    """Run the first clean-core scenario subset.

    Supported initial actions are:
    ``mint_reserves``, ``mint_cash``, ``deposit_cash``, ``create_payable``,
    ``create_stock``, and ``create_delivery_obligation``. Payables settle by
    the configured means-of-payment order for the debtor's agent kind,
    including intrabank payment and two-bank reserve netting. Delivery
    obligations settle by transferring inventory lots.
    """
    runtime = prepare_scenario(config)

    effective_max_days = max_days if max_days is not None else config.run.max_days
    effective_quiet_days = quiet_days if quiet_days is not None else config.run.quiet_days

    return run_runtime_until_stable(
        runtime,
        max_days=effective_max_days,
        quiet_days=effective_quiet_days,
    )


def run_runtime_until_stable(
    runtime: CleanScenarioRuntime,
    *,
    max_days: int,
    quiet_days: int,
    progress_callback: Callable[[int, int], None] | None = None,
    day_callback: Callable[[CleanScenarioRuntime, int], None] | None = None,
) -> CleanRunResult:
    """Run a prepared clean-core runtime with the legacy-compatible stop rule."""
    state = runtime.state
    reached_stable = False
    final_day = 0
    stability = CleanStabilityTracker()
    for day in range(max_days):
        impactful = run_day(runtime, day)
        if progress_callback:
            progress_callback(day + 1, max_days)
        stable_today = update_runtime_stability(
            runtime,
            day=day,
            impactful=impactful,
            quiet_days=quiet_days,
            tracker=stability,
        )
        if day_callback:
            day_callback(runtime, day)

        if stable_today:
            reached_stable = True
            final_day = day + 1
            break
        final_day = day + 1

    return CleanRunResult(
        state=state,
        final_day=final_day,
        reached_stable=reached_stable,
        stop_reason=(
            StopReason.STABILITY_REACHED
            if reached_stable
            else StopReason.MAX_DAYS_REACHED
        ),
        stability_snapshots=tuple(stability.snapshots),
    )


def update_runtime_stability(
    runtime: CleanScenarioRuntime,
    *,
    day: int,
    impactful: bool,
    quiet_days: int,
    tracker: CleanStabilityTracker,
) -> bool:
    return _update_runtime_stability(
        runtime,
        day=day,
        impactful=impactful,
        quiet_days=quiet_days,
        tracker=tracker,
    )


def balance_rows(result: CleanRunResult) -> list[dict[str, Any]]:
    """Return rows shaped like ``bilancio.analysis.balances.as_rows`` for this slice."""
    from bilancio.engines.clean_core_views import balance_rows as _balance_rows

    return _balance_rows(result)


def t_account_rows(state: CleanState, agent_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return detailed T-account rows for a clean-core agent."""
    from bilancio.engines.clean_core_views import t_account_rows as _t_account_rows

    return _t_account_rows(state, agent_id)


def write_balances_csv(result: CleanRunResult, path: Path) -> None:
    """Write clean-core balances using the same CSV shape as the legacy export."""
    from bilancio.engines.clean_core_exports import write_balances_csv as _write

    _write(result, path)


def write_events_jsonl(result: CleanRunResult, path: Path) -> None:
    """Write clean-core events as JSONL, preserving Decimal values like legacy exports."""
    from bilancio.engines.clean_core_exports import write_events_jsonl as _write

    _write(result, path)


def write_html_report(
    result: CleanRunResult,
    config: ScenarioConfig,
    path: Path,
    *,
    max_days: int,
    quiet_days: int,
    agent_ids: list[str] | None = None,
    t_account: bool = False,
) -> None:
    """Write a readable clean-core HTML report for the public ``--html`` CLI flag."""
    from bilancio.engines.clean_core_exports import write_html_report as _write

    _write(
        result,
        config,
        path,
        max_days=max_days,
        quiet_days=quiet_days,
        agent_ids=agent_ids,
        t_account=t_account,
    )


def _run_day(
    state: CleanState,
    policy_order: dict[str, list[str]],
    lender_config: CleanLenderConfig | None,
    rating_config: CleanRatingConfig | None,
    banking_config: CleanBankingConfig | None = None,
    dealer_config: CleanDealerConfig | None = None,
) -> bool:
    state.log("PhaseA")
    state.log("PhaseB")
    state.log("SubphaseB1")
    for index, action in enumerate(state.scheduled_actions_by_day.get(state.day, [])):
        _apply_action(state, action, index=index, setup=False)

    impactful = False
    if rating_config is not None:
        state.log("SubphaseB_Rating")
        state.events.extend(_run_rating_phase(state, rating_config))
        if state.estimate_logging_enabled:
            _log_rating_estimates(state)

    if banking_config is not None:
        state.log("SubphaseB_BankQuotes")

    if lender_config is not None:
        state.log("SubphaseB_Lending")
        _run_lending_phase(state, lender_config, banking_config=banking_config)

    if banking_config is not None and banking_config.enable_bank_lending:
        state.log("SubphaseB_BankLending")
        _run_bank_lending_phase(state, banking_config)

    if dealer_config is not None:
        state.log("SubphaseB_Dealer")
        if dealer_config.balanced_active:
            impactful = (
                _run_clean_active_dealer_phase(
                    state,
                    dealer_config,
                    banking_config=banking_config,
                )
                or impactful
            )
        elif not dealer_config.balanced_passive:
            _capture_clean_dealer_marker_snapshots(state, dealer_config)

    state.log("SubphaseB2")

    settled_for_rollover: list[tuple[str, str, Decimal, int]] = []
    for payable in list(state.payables):
        if payable.settled or payable.due_day != state.day:
            continue
        settled, rollover_info = _settle_payable(
            state,
            payable,
            policy_order,
            banking_config=banking_config,
        )
        impactful = settled or impactful
        if rollover_info is not None:
            settled_for_rollover.append(rollover_info)

    for obligation in state.delivery_obligations:
        if obligation.settled or obligation.due_day != state.day:
            continue
        impactful = _settle_delivery_obligation(state, obligation) or impactful

    if state.rollover_enabled and settled_for_rollover:
        state.log("SubphaseB_Rollover")
        _rollover_settled_payables(state, settled_for_rollover)

    state.log("PhaseC")
    interbank_flows = _net_interbank_flows(_client_payment_flows_for_day(state))
    if banking_config is not None:
        state.log("SubphaseC_InterbankAuction")
        state.events.append(
            {
                "kind": "InterbankAuction",
                "day": state.day,
                **_clean_interbank_auction_summary(state, banking_config, interbank_flows),
            }
        )

    for (from_bank, to_bank), amount in interbank_flows.items():
        _require_at_least(state.reserves[from_bank], amount, f"{from_bank} reserves")
        state.reserves[from_bank] -= amount
        state.reserves[to_bank] += amount
        state.log("ReservesTransferred", frm=from_bank, to=to_bank, amount=amount)
        state.log("InstrumentMerged", keep=f"reserve:{to_bank}", removed=f"transfer:{from_bank}:{to_bank}:{state.day}")
        state.log("InterbankCleared", debtor_bank=from_bank, creditor_bank=to_bank, amount=amount)
        impactful = True

    _repay_due_cb_loans(state)
    _repay_due_non_bank_loans(state)
    _repay_due_bank_loans(state)
    return impactful


_run_lending_phase = _lending_phase._run_lending_phase
_apply_lender_capital_conservation = _lending_phase._apply_lender_capital_conservation
_realized_non_bank_loan_loss = _lending_phase._realized_non_bank_loan_loss
_collect_lending_opportunities = _lending_phase._collect_lending_opportunities
_collect_preventive_lending_opportunities = _lending_phase._collect_preventive_lending_opportunities
_execute_preventive_lending_opportunities = _lending_phase._execute_preventive_lending_opportunities
_create_non_bank_loan = _lending_phase._create_non_bank_loan
_repay_due_non_bank_loans = _lending_phase._repay_due_non_bank_loans
_settle_payable = _settlement._settle_payable
_pay_with_deposit = _settlement._pay_with_deposit
_pay_with_routed_deposits = _settlement._pay_with_routed_deposits
_select_clean_receive_bank = _settlement._select_clean_receive_bank
_handle_payable_default = _settlement._handle_payable_default
_rollover_settled_payables = _settlement._rollover_settled_payables
_expel_agent = _settlement._expel_agent
_collect_creditor_weights = _settlement._collect_creditor_weights
_distribute_pro_rata_recovery = _settlement._distribute_pro_rata_recovery
_write_off_liabilities = _settlement._write_off_liabilities
_reassign_receivables = _settlement._reassign_receivables
_cancel_scheduled_actions_for_agent = _settlement._cancel_scheduled_actions_for_agent
_settle_delivery_obligation = _settlement._settle_delivery_obligation
_handle_delivery_default = _settlement._handle_delivery_default
