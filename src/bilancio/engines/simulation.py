"""Simulation engines for financial scenario analysis."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from bilancio.core.errors import DefaultError, SimulationHalt, ValidationError
from bilancio.core.performance import fast_atomic_scope
from bilancio.domain.agent import AgentKind
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.cb_loan import CBLoan
from bilancio.engines.clearing import settle_intraday_nets
from bilancio.engines.settlement import _remove_contract, rollover_settled_payables, settle_due

if TYPE_CHECKING:
    from bilancio.core.performance import PerformanceConfig
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)

IMPACT_EVENTS = {
    "PayableSettled",
    "DeliveryObligationSettled",
    "InterbankCleared",
    "InterbankOvernightCreated",
    "InterbankAuctionTrade",
}

DEFAULT_EVENTS = {
    "ObligationDefaulted",
    "ObligationWrittenOff",
    "AgentDefaulted",
    "BankDefaultCBFreeze",
}


@dataclass
class DayReport:
    day: int
    impacted: int
    notes: str = ""


def _impacted_today(system: System, day: int) -> int:
    return sum(
        1 for e in system.state.events if e.get("day") == day and e.get("kind") in IMPACT_EVENTS
    )


def _defaults_today(system: System, day: int) -> int:
    """Count default events that occurred on a given day."""
    return sum(
        1 for e in system.state.events if e.get("day") == day and e.get("kind") in DEFAULT_EVENTS
    )


def _has_open_obligations(system: System) -> bool:
    for c in system.state.contracts.values():
        if c.kind in (InstrumentKind.PAYABLE, InstrumentKind.DELIVERY_OBLIGATION):
            return True
    return False


def _has_outstanding_bank_loans(banking_sub: Any) -> bool:
    """Check if any bank has outstanding loans."""
    for bank_state in banking_sub.banks.values():
        if bank_state.outstanding_loans:
            return True
    return False


class SimulationEngine(Protocol):
    """Protocol for simulation engines that can run financial scenarios."""

    def run(self, scenario: Any) -> Any:
        """
        Run a simulation for a given scenario.

        Args:
            scenario: The scenario to simulate

        Returns:
            Simulation results
        """
        ...


class MonteCarloEngine:
    """Monte Carlo simulation engine for financial scenarios."""

    def __init__(
        self,
        num_simulations: int = 1000,
        n_simulations: int | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize the Monte Carlo engine.

        Args:
            num_simulations: Number of simulation runs to perform
            n_simulations: Alternative parameter name for num_simulations (for compatibility)
            random_seed: Optional seed for reproducible results
        """
        # Support both parameter names for compatibility
        if n_simulations is not None:
            self.num_simulations = n_simulations
            self.n_simulations = n_simulations
        else:
            self.num_simulations = num_simulations
            self.n_simulations = num_simulations

        if random_seed is not None:
            random.seed(random_seed)

    def run(self, scenario: Any) -> dict[str, Any]:
        """
        Run Monte Carlo simulation for a scenario.

        This is a placeholder implementation. A real implementation would:
        1. Extract parameters and distributions from the scenario
        2. Generate random samples according to those distributions
        3. Run the scenario multiple times with different random inputs
        4. Aggregate and return statistical results

        Args:
            scenario: The scenario to simulate

        Returns:
            Dictionary containing simulation results and statistics
        """
        # Placeholder implementation
        results = []

        for i in range(self.num_simulations):
            # In a real implementation, this would:
            # - Sample from probability distributions
            # - Apply scenario logic with sampled values
            # - Calculate outcome metrics

            # For now, just generate dummy results
            result = {
                "run_id": i,
                "outcome": random.gauss(100, 20),  # Placeholder random outcome
                "scenario": scenario,
            }
            results.append(result)

        # Calculate summary statistics
        outcomes = [r["outcome"] for r in results]
        summary = {
            "num_simulations": self.num_simulations,
            "mean": sum(outcomes) / len(outcomes),
            "min": min(outcomes),
            "max": max(outcomes),
            "results": results,
        }

        return summary

    def set_num_simulations(self, num_simulations: int) -> None:
        """Update the number of simulations to run."""
        self.num_simulations = num_simulations


def _log_rating_estimates(system: System, current_day: int) -> None:
    """Log rating agency estimates to the estimate log."""
    from bilancio.information.estimates import Estimate

    rating_registry = getattr(system.state, "rating_registry", None)
    if not rating_registry:
        return

    for agent_id, p_default in rating_registry.items():
        est = Estimate(
            value=p_default,
            estimator_id="rating_agency",
            target_id=agent_id,
            target_type="agent",
            estimation_day=current_day,
            method="rating_agency_published",
            inputs={"source": "rating_registry"},
        )
        system.log_estimate(est)


def _log_dealer_estimates(system: System, current_day: int) -> None:
    """Log dealer risk assessor estimates to the estimate log."""

    dealer_sub = system.state.dealer_subsystem
    if dealer_sub is None or dealer_sub.risk_assessor is None:
        return

    assessor = dealer_sub.risk_assessor
    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            continue
        est = assessor.estimate_default_prob_detail(
            agent_id,
            current_day,
            estimator_id="dealer_risk_assessor",
        )
        system.log_estimate(est)


def _run_cb_backstop(system: System, banking_sub: Any, current_day: int) -> None:
    """CB end-of-day backstop: automatically lend from CB to bring bank reserves up to target.

    Per paper Part E, at end of each day the CB checks each bank's reserves
    and automatically lends to cover any shortfall below target.
    """
    from bilancio.engines.banking_subsystem import _get_bank_reserves

    if system.state.cb_lending_frozen:
        return

    for bank_id, bank_state in banking_sub.banks.items():
        bank_agent = system.state.agents.get(bank_id)
        if bank_agent is None or bank_agent.defaulted:
            continue

        reserves = _get_bank_reserves(system, bank_id)
        target = bank_state.pricing_params.reserve_target
        shortfall = target - reserves

        if shortfall > 0:
            try:
                system.cb_lend_reserves(bank_id, shortfall, current_day)
                system.log(
                    "CBBackstopLoan",
                    bank_id=bank_id,
                    amount=shortfall,
                    day=current_day,
                    reserves_before=reserves,
                    reserve_target=target,
                )
                logger.debug(
                    "CB backstop: bank=%s shortfall=%d day=%d",
                    bank_id, shortfall, current_day,
                )
            except (ValueError, ValidationError, RuntimeError):
                # CB lending frozen or other issue - log and continue
                logger.warning(
                    "CB backstop failed for bank=%s shortfall=%d",
                    bank_id, shortfall,
                )

def run_day(
    system: System,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
    enable_bank_lending: bool = False,
    performance: PerformanceConfig | None = None,
) -> None:
    """
    Run a single day's simulation with three phases.

    Phase A: Log PhaseA event (noop for now)
    Phase B: Settle obligations due on the current day using settle_due
    Phase C: Clear intraday nets for the current day using settle_intraday_nets

    Finally, increment the system day counter.

    Args:
        system: System instance to run the day for
        enable_dealer: If True, run dealer trading phase between scheduled actions and settlements
        enable_lender: If True, run non-bank lending phase
        enable_rating: If True, run rating agency phase
        enable_banking: If True, run banking infrastructure (bank quotes, interbank, deposits, interbank repayments)
        enable_bank_lending: If True, run bank lending phases (loan origination and repayment)

    Note: Rollover is controlled by system.state.rollover_enabled (Plan 024)
    """
    current_day = system.state.day
    rollover_enabled = getattr(system.state, "rollover_enabled", False)
    fast = performance.fast_atomic if performance else False
    logger.debug("run_day: day=%d phase=%s", current_day, system.state.phase)

    # Phase A: Log PhaseA event (reserved)
    system.log("PhaseA")

    # Phase B: two subphases — B1 scheduled actions, B2 settlements
    system.log("PhaseB")  # Phase B bucket marker
    # B1: Execute scheduled actions for this day (if any)
    system.log("SubphaseB1")
    try:
        actions_today = system.state.scheduled_actions_by_day.get(current_day, [])
        if actions_today:
            # Lazy import to avoid heavy imports at module load
            from bilancio.config.apply import apply_action

            agents = system.state.agents
            for action_dict in actions_today:
                apply_action(system, action_dict, agents)
    except (ValueError, ValidationError, DefaultError, SimulationHalt, KeyError, AttributeError):
        # Allow scheduled-action errors to bubble via apply_action's own error handling
        # but keep guard to ensure the simulation loop stability
        raise

    with fast_atomic_scope(system, fast):
        # SubphaseB_Rating: Run rating agency phase (optional)
        if enable_rating and system.state.rating_config is not None:
            system.log("SubphaseB_Rating")
            from bilancio.engines.rating import run_rating_phase

            rating_events = run_rating_phase(system, current_day, system.state.rating_config)
            system.state.events.extend(rating_events)

            # Log rating estimates if enabled
            if system.state.estimate_logging_enabled:
                _log_rating_estimates(system, current_day)

        # SubphaseB_BankQuotes: Refresh bank quotes (optional)
        banking_sub = getattr(system.state, "banking_subsystem", None)
        if enable_banking and banking_sub is not None:
            system.log("SubphaseB_BankQuotes")
            banking_sub.update_cb_corridor(system)
            banking_sub.refresh_all_quotes(system, current_day)

    # SubphaseB_Lending: Non-bank lending phase (optional)
    # Runs BEFORE dealer trading so firms can borrow to cover shortfalls
    # before deciding whether to sell claims on the secondary market.
    if enable_lender and system.state.lender_config is not None:
        system.log("SubphaseB_Lending")
        from bilancio.engines.lending import run_lending_phase

        lending_events = run_lending_phase(system, current_day, system.state.lender_config)
        system.state.events.extend(lending_events)

    # SubphaseB_BankLending: Bank lending to traders (optional)
    if enable_bank_lending and banking_sub is not None:
        system.log("SubphaseB_BankLending")
        from bilancio.engines.bank_lending import run_bank_lending_phase

        bank_lending_events = run_bank_lending_phase(system, current_day, banking_sub)
        system.state.events.extend(bank_lending_events)

    # SubphaseB_Dealer: Run dealer trading phase (optional)
    if enable_dealer and system.state.dealer_subsystem is not None:
        system.log("SubphaseB_Dealer")
        # Lazy import to avoid circular dependencies
        from bilancio.engines.dealer_integration import (
            run_dealer_trading_phase,
            sync_dealer_to_system,
        )

        # Thread performance Option B (prune_ineligible) to the subsystem
        if performance and performance.prune_ineligible:
            system.state.dealer_subsystem.prune_ineligible = True

        # Thread performance Option C (cache_dealer_quotes) to the subsystem
        if performance and performance.cache_dealer_quotes:
            system.state.dealer_subsystem.cache_dealer_quotes = True

        # Thread performance Option F (preview_buy) to the subsystem
        if performance and performance.preview_buy:
            system.state.dealer_subsystem.preview_buy = True

        # Thread performance Option G (dirty_bucket_recompute) to the subsystem
        if performance and performance.dirty_bucket_recompute:
            system.state.dealer_subsystem.dirty_bucket_recompute = True

        # Thread performance Option H (incremental_intentions) to the subsystem
        if performance and performance.incremental_intentions:
            system.state.dealer_subsystem.incremental_intentions = True

        # Thread performance Option D (matching_order) to the subsystem
        if performance and performance.matching_order != "random":
            system.state.dealer_subsystem.matching_order = performance.matching_order

        # Thread performance Option E (dealer_backend) to the subsystem
        if performance and performance.dealer_backend == "native":
            from bilancio.dealer.kernel_native import (
                NATIVE_AVAILABLE,
                recompute_dealer_state_native,
            )

            if NATIVE_AVAILABLE:
                system.state.dealer_subsystem._recompute_fn = recompute_dealer_state_native
                if system.state.dealer_subsystem.executor is not None:
                    system.state.dealer_subsystem.executor._recompute_fn = (
                        recompute_dealer_state_native
                    )
            else:
                logger.warning(
                    "dealer_backend='native' requested but Rust extension unavailable; "
                    "falling back to Python"
                )

        # Run dealer trading and collect events
        dealer_events = run_dealer_trading_phase(system.state.dealer_subsystem, system, current_day)
        system.state.events.extend(dealer_events)

        # Sync dealer state back to main system
        sync_dealer_to_system(system.state.dealer_subsystem, system)

        # Log dealer risk estimates if enabled
        if system.state.estimate_logging_enabled:
            _log_dealer_estimates(system, current_day)

    # B2: Automated settlements due today
    system.log("SubphaseB2")
    settled_for_rollover = settle_due(system, current_day, rollover_enabled=rollover_enabled)

    # Refresh bank quotes after settlements (event-driven pricing)
    if enable_banking and banking_sub is not None:
        banking_sub.refresh_all_quotes(system, current_day)

    # Plan 024: Rollover - create new payables for settled ones
    with fast_atomic_scope(system, fast):
        if rollover_enabled and settled_for_rollover:
            system.log("SubphaseB_Rollover")
            dealer_active = enable_dealer and system.state.dealer_subsystem is not None
            rollover_settled_payables(
                system, current_day, settled_for_rollover, dealer_active=dealer_active
            )

    # Phase C: Clear intraday nets + interbank auction
    system.log("PhaseC")
    if enable_banking and banking_sub is not None:
        from bilancio.engines.clearing import (
            compute_bank_net_obligations,
            compute_combined_nets,
            settle_nets_with_funding,
        )
        from bilancio.engines.interbank import (
            compute_interbank_obligations,
            finalize_interbank_repayments,
            run_interbank_auction,
        )

        # 1. Identify overnight loan repayment obligations (DO NOT settle yet)
        ib_obligations = compute_interbank_obligations(current_day, banking_sub)

        # 2-3. Compute combined bilateral nets and per-bank positions
        combined_nets = compute_combined_nets(system, current_day, ib_obligations)
        net_obligations = compute_bank_net_obligations(combined_nets)

        # 4. Call auction
        system.log("SubphaseC_InterbankAuction")
        auction_events = run_interbank_auction(
            system, current_day, banking_sub, net_obligations,
        )
        system.state.events.extend(auction_events)

        # 5. Settle combined nets (interbank-funded + CB fallback)
        settle_nets_with_funding(system, current_day, combined_nets)

        # 6. Finalize overnight repayments (remove from book)
        repay_events = finalize_interbank_repayments(
            system, current_day, banking_sub, ib_obligations,
        )
        system.state.events.extend(repay_events)

        # 7. CB backstop (top up remaining deficit banks to target)
        _run_cb_backstop(system, banking_sub, current_day)
        banking_sub.refresh_all_quotes(system, current_day)
    else:
        settle_intraday_nets(system, current_day)

    # Phase D: CB corridor maintenance (interest + loan repayment)
    with fast_atomic_scope(system, fast):
        has_cb = any(agent.kind == AgentKind.CENTRAL_BANK for agent in system.state.agents.values())
        if has_cb:
            system.credit_reserve_interest(current_day)
            for loan_id in system.get_cb_loans_due(current_day):
                loan = system.state.contracts.get(loan_id)
                if loan is None:
                    continue
                assert isinstance(loan, CBLoan)
                bank_id = loan.liability_issuer_id
                try:
                    system.cb_repay_loan(loan_id, bank_id)
                except (ValueError, ValidationError):
                    # Bank can't repay — try refinancing
                    repayment = loan.repayment_amount
                    try:
                        system.cb_lend_reserves(bank_id, repayment, current_day)
                        system.cb_repay_loan(loan_id, bank_id)
                        logger.debug(
                            "CB loan refinanced: bank=%s loan=%s amount=%d",
                            bank_id, loan_id, repayment,
                        )
                    except (ValueError, ValidationError):
                        # Refinancing failed (frozen or cap exceeded)
                        if system.state.cb_lending_frozen:
                            # Bank defaults: write off loan, mark as defaulted
                            bank_agent = system.state.agents.get(bank_id)
                            if bank_agent and not bank_agent.defaulted:
                                bank_agent.defaulted = True
                                system.state.defaulted_agent_ids.add(bank_id)
                                system.log(
                                    "BankDefaultCBFreeze",
                                    bank_id=bank_id,
                                    loan_id=loan_id,
                                    amount=loan.amount,
                                    day=current_day,
                                )
                                logger.info(
                                    "Bank %s defaulted (CB frozen, can't repay %d)",
                                    bank_id, loan.amount,
                                )
                            # Per-loan writeoff event (emitted for every loan,
                            # not just the first one that triggers bank default)
                            system.log(
                                "CBLoanFreezeWrittenOff",
                                bank_id=bank_id,
                                loan_id=loan_id,
                                amount=loan.amount,
                                day=current_day,
                            )
                            _remove_contract(system, loan_id)
                        else:
                            logger.warning(
                                "CB loan repayment failed even after refinancing: bank=%s loan=%s",
                                bank_id, loan_id,
                            )

    # Non-bank loan repayment
    has_lender = any(
        agent.kind == AgentKind.NON_BANK_LENDER for agent in system.state.agents.values()
    )
    if has_lender:
        from bilancio.engines.lending import run_loan_repayments

        run_loan_repayments(system, current_day)

    # Bank loan repayments
    if enable_bank_lending and banking_sub is not None:
        from bilancio.engines.bank_lending import run_bank_loan_repayments

        bank_repay_events = run_bank_loan_repayments(system, current_day, banking_sub)
        system.state.events.extend(bank_repay_events)

        # Refresh bank quotes after loan repayments (event-driven pricing)
        banking_sub.refresh_all_quotes(system, current_day)

    # Deposit interest accrual
    if enable_banking and banking_sub is not None:
        from bilancio.engines.bank_interest import accrue_deposit_interest

        interest_events = accrue_deposit_interest(system, current_day, banking_sub)
        system.state.events.extend(interest_events)

    # (Interbank repayments and CB backstop are now handled in Phase C)

    # Increment system day
    system.state.day += 1


def run_final_cb_settlement(system: System) -> dict[str, Any]:
    """Force all banks to repay outstanding CB loans after the main simulation loop.

    Banks that cannot repay are marked as defaulted and their loans are written off.
    This makes CB-funded bank fragility observable rather than silently rolling over.

    No cascading is triggered — by this point all trader activity is done.

    Args:
        system: The system after the main simulation loop has finished.

    Returns:
        Dict with settlement summary metrics.
    """
    pre_outstanding = system.state.cb_loans_outstanding
    loans_attempted = 0
    loans_repaid = 0
    loans_written_off = 0
    bank_defaults = 0
    total_written_off_amount = 0
    defaulted_banks: set[str] = set()

    system.log(
        "CBFinalSettlementStart",
        cb_loans_outstanding=pre_outstanding,
    )

    # Collect all CB_LOAN contracts
    cb_loans = [
        (cid, contract)
        for cid, contract in list(system.state.contracts.items())
        if contract.kind == InstrumentKind.CB_LOAN and isinstance(contract, CBLoan)
    ]

    for loan_id, loan in cb_loans:
        # Skip if loan was already removed (e.g., all loans for a defaulted bank)
        if loan_id not in system.state.contracts:
            continue

        bank_id = loan.liability_issuer_id

        # If bank already defaulted in this phase, write off remaining loans
        if bank_id in defaulted_banks:
            total_written_off_amount += loan.amount
            loans_written_off += 1
            loans_attempted += 1
            system.log(
                "CBFinalSettlementWrittenOff",
                loan_id=loan_id,
                bank_id=bank_id,
                amount=loan.amount,
                reason="bank_already_defaulted",
            )
            _remove_contract(system, loan_id)
            continue

        loans_attempted += 1
        try:
            system.cb_repay_loan(loan_id, bank_id)
            loans_repaid += 1
            system.log(
                "CBFinalSettlementRepaid",
                loan_id=loan_id,
                bank_id=bank_id,
            )
        except (ValidationError, ValueError):
            # Bank can't repay — default
            total_written_off_amount += loan.amount
            loans_written_off += 1
            defaulted_banks.add(bank_id)

            # Mark bank as defaulted
            bank_agent = system.state.agents.get(bank_id)
            if bank_agent and not bank_agent.defaulted:
                bank_agent.defaulted = True
                system.state.defaulted_agent_ids.add(bank_id)
                bank_defaults += 1
                system.log(
                    "CBFinalSettlementBankDefault",
                    bank_id=bank_id,
                    loan_id=loan_id,
                    shortfall=loan.amount,
                )

            # Write off this loan
            system.log(
                "CBFinalSettlementWrittenOff",
                loan_id=loan_id,
                bank_id=bank_id,
                amount=loan.amount,
                reason="insufficient_reserves",
            )
            _remove_contract(system, loan_id)

            # Write off ALL remaining CB loans for this bank
            for other_id, other_loan in list(system.state.contracts.items()):
                if (
                    other_id != loan_id
                    and other_loan.kind == InstrumentKind.CB_LOAN
                    and isinstance(other_loan, CBLoan)
                    and other_loan.liability_issuer_id == bank_id
                    and other_id in system.state.contracts
                ):
                    total_written_off_amount += other_loan.amount
                    loans_written_off += 1
                    loans_attempted += 1
                    system.log(
                        "CBFinalSettlementWrittenOff",
                        loan_id=other_id,
                        bank_id=bank_id,
                        amount=other_loan.amount,
                        reason="bank_already_defaulted",
                    )
                    _remove_contract(system, other_id)

            # Resolve the failed bank: distribute remaining reserves to depositors,
            # then write off all remaining liabilities (BankDeposit, interbank, etc.)
            from bilancio.engines.settlement import _resolve_failed_bank, _write_off_liabilities
            _resolve_failed_bank(system, bank_id)
            _write_off_liabilities(system, bank_id)

    post_outstanding = system.state.cb_loans_outstanding

    system.log(
        "CBFinalSettlementEnd",
        loans_attempted=loans_attempted,
        loans_repaid=loans_repaid,
        loans_written_off=loans_written_off,
        bank_defaults=bank_defaults,
        total_written_off_amount=total_written_off_amount,
        cb_loans_outstanding_pre_final=pre_outstanding,
        cb_loans_outstanding_post_final=post_outstanding,
        cb_reserves_initial=system.state.cb_reserves_initial,
        cb_reserves_final=system.state.cb_reserves_outstanding,
        cb_interest_total_paid=system.state.cb_interest_total_paid,
        cb_loans_created_count=system.state.cb_loans_created_count,
    )

    return {
        "loans_attempted": loans_attempted,
        "loans_repaid": loans_repaid,
        "loans_written_off": loans_written_off,
        "bank_defaults": bank_defaults,
        "total_written_off_amount": total_written_off_amount,
        "cb_loans_outstanding_pre_final": pre_outstanding,
        "cb_loans_outstanding_post_final": post_outstanding,
    }


def _run_bank_loan_winddown(system: System, banking_sub: Any) -> int:
    """Run bank loan repayments until all outstanding loans mature.

    After the main simulation loop reaches stability, continue running
    bank loan repayments only (no new lending) until all outstanding
    bank loans have matured.

    Args:
        system: System instance
        banking_sub: BankingSubsystem with bank states

    Returns:
        Number of wind-down days executed.
    """
    if not _has_outstanding_bank_loans(banking_sub):
        return 0

    # Count initial outstanding loans
    initial_loans = sum(
        len(bs.outstanding_loans) for bs in banking_sub.banks.values()
    )

    system.log(
        "BankLoanWinddownStart",
        day=system.state.day,
        outstanding_loans=initial_loans,
    )

    winddown_days = 0
    # Compute cap from the latest maturity day across all outstanding loans,
    # plus a small buffer.  Falls back to 50 if no maturity info available.
    max_maturity = 0
    for bs in banking_sub.banks.values():
        for loan in bs.outstanding_loans.values():
            if loan.maturity_day > max_maturity:
                max_maturity = loan.maturity_day
    # We need at most (max_maturity - current_day + 1) days, but use
    # include_overdue=True so one pass can clear all overdue loans.
    # Add a buffer of 5 for interbank settlement rounds.
    max_winddown = max(max_maturity - system.state.day + 5, 10)

    for _ in range(max_winddown):
        if not _has_outstanding_bank_loans(banking_sub):
            break

        current_day = system.state.day

        system.log("BankLoanWinddownDay", day=current_day)

        # Run bank loan repayments only (include_overdue catches loans
        # that matured during the main loop but were never settled)
        from bilancio.engines.bank_lending import run_bank_loan_repayments
        bank_repay_events = run_bank_loan_repayments(
            system, current_day, banking_sub, include_overdue=True
        )
        system.state.events.extend(bank_repay_events)

        # Run interbank repayments (simplified: identify + finalize during wind-down)
        from bilancio.engines.interbank import (
            compute_interbank_obligations,
            finalize_interbank_repayments,
        )
        ib_obligations = compute_interbank_obligations(current_day, banking_sub)
        # During wind-down, interbank repayments are simple reserve transfers
        # (no auction needed — just settle maturing loans directly)
        for borrower, lender, repayment, _loan in ib_obligations:
            try:
                system.transfer_reserves(borrower, lender, repayment)
            except (ValueError, ValidationError):
                # Borrower can't repay — CB backstop will handle
                logger.warning(
                    "Wind-down interbank repayment failed: %s -> %s amount=%d",
                    borrower, lender, repayment,
                )
        ib_repay_events = finalize_interbank_repayments(
            system, current_day, banking_sub, ib_obligations,
        )
        system.state.events.extend(ib_repay_events)

        # CB loan repayments during wind-down
        has_cb = any(a.kind == AgentKind.CENTRAL_BANK for a in system.state.agents.values())
        if has_cb:
            system.credit_reserve_interest(current_day)
            for loan_id in system.get_cb_loans_due(current_day):
                loan = system.state.contracts.get(loan_id)
                if loan is None:
                    continue
                bank_id = loan.liability_issuer_id
                try:
                    system.cb_repay_loan(loan_id, bank_id)
                except (ValueError, ValidationError):
                    # CB is frozen during wind-down, so refinancing will fail
                    bank_agent = system.state.agents.get(bank_id)
                    if bank_agent and not bank_agent.defaulted:
                        bank_agent.defaulted = True
                        system.state.defaulted_agent_ids.add(bank_id)
                        system.log(
                            "BankDefaultWinddown",
                            bank_id=bank_id,
                            loan_id=loan_id,
                            day=current_day,
                        )
                        logger.info(
                            "Bank %s defaulted during wind-down (can't repay CB loan)",
                            bank_id,
                        )
                        # Resolve failed bank: distribute reserves, write off liabilities
                        from bilancio.engines.settlement import _resolve_failed_bank, _write_off_liabilities
                        _resolve_failed_bank(system, bank_id)
                        _write_off_liabilities(system, bank_id)
                    _remove_contract(system, loan_id)

        system.state.day += 1
        winddown_days += 1

    # Count remaining loans (should be 0)
    remaining_loans = sum(
        len(bs.outstanding_loans) for bs in banking_sub.banks.values()
    )

    system.log(
        "BankLoanWinddownEnd",
        day=system.state.day,
        winddown_days=winddown_days,
        initial_loans=initial_loans,
        remaining_loans=remaining_loans,
    )

    logger.info(
        "Bank loan wind-down: %d days, %d→%d loans",
        winddown_days, initial_loans, remaining_loans,
    )

    return winddown_days


def run_until_stable(
    system: System,
    max_days: int = 365,
    quiet_days: int = 2,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
    enable_bank_lending: bool = False,
    enable_final_cb_settlement: bool | None = None,
    cb_lending_cutoff_day: int | None = None,
    performance: PerformanceConfig | None = None,
) -> list[DayReport]:
    """
    Advance day by day until the system is stable:
    - No impactful events happen for `quiet_days` consecutive days, AND
    - No outstanding payables or delivery obligations remain.

    Args:
        system: System instance to run
        max_days: Maximum number of days to run
        quiet_days: Number of consecutive quiet days needed for stability
        enable_dealer: If True, run dealer trading phase each day
        enable_lender: If True, run non-bank lending phase each day
        enable_rating: If True, run rating agency phase each day
        enable_banking: If True, run banking infrastructure each day
        enable_bank_lending: If True, run bank lending phases each day
        enable_final_cb_settlement: If True, force banks to repay all CB loans
            after the main loop. None = auto (enabled when banking is active).

    Note: Rollover is controlled by system.state.rollover_enabled (Plan 024)
    """
    # Capture initial reserves for CB stress metrics
    system.state.cb_reserves_initial = system.state.cb_reserves_outstanding

    reports = []
    consecutive_quiet = 0
    consecutive_no_defaults = 0
    rollover_enabled = getattr(system.state, "rollover_enabled", False)

    for _ in range(max_days):
        day_before = system.state.day
        # Activate CB lending freeze when cutoff day is reached
        if (
            cb_lending_cutoff_day is not None
            and day_before >= cb_lending_cutoff_day
            and not system.state.cb_lending_frozen
        ):
            system.state.cb_lending_frozen = True
            system.log("CBLendingFreezeActivated", day=day_before, cutoff_day=cb_lending_cutoff_day)
            logger.info("CB lending frozen at day %d (cutoff=%d)", day_before, cb_lending_cutoff_day)
        run_day(
            system,
            enable_dealer=enable_dealer,
            enable_lender=enable_lender,
            enable_rating=enable_rating,
            enable_banking=enable_banking,
            enable_bank_lending=enable_bank_lending,
            performance=performance,
        )
        impacted = _impacted_today(system, day_before)
        defaults = _defaults_today(system, day_before)
        reports.append(DayReport(day=day_before, impacted=impacted))

        if impacted == 0:
            consecutive_quiet += 1
        else:
            consecutive_quiet = 0

        if defaults == 0:
            consecutive_no_defaults += 1
        else:
            consecutive_no_defaults = 0

        # Rollover mode: stability = no defaults for quiet_days consecutive days
        # (settlements are expected and fine in rollover scenarios)
        # Non-rollover: stability = no impact events + no open obligations
        if rollover_enabled:
            stability_condition = consecutive_no_defaults >= quiet_days
        else:
            stability_condition = consecutive_quiet >= quiet_days and not _has_open_obligations(
                system
            )

        if stability_condition:
            if enable_banking and not system.state.cb_lending_frozen:
                system.state.cb_lending_frozen = True
                system.log("CBLendingFreezeStability", day=system.state.day)
                logger.info("CB lending frozen at stability (day %d)", system.state.day)
            break

    # Bank loan wind-down: run remaining bank loan repayments after main loop
    if enable_bank_lending:
        banking_sub = getattr(system.state, "banking_subsystem", None)
        if banking_sub is not None:
            _run_bank_loan_winddown(system, banking_sub)

    # Final CB settlement: force banks to repay all outstanding CB loans
    should_final = (enable_final_cb_settlement is not False) and enable_banking
    if should_final:
        has_cb = any(a.kind == AgentKind.CENTRAL_BANK for a in system.state.agents.values())
        if has_cb:
            final_result = run_final_cb_settlement(system)
            if final_result["loans_attempted"] > 0:
                logger.info(
                    "final CB settlement: %d/%d repaid, %d written off, %d bank defaults",
                    final_result["loans_repaid"],
                    final_result["loans_attempted"],
                    final_result["loans_written_off"],
                    final_result["bank_defaults"],
                )

    logger.info("simulation complete: %d days", system.state.day)
    return reports
