"""Simulation engines for financial scenario analysis."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from bilancio.core.errors import DefaultError, SimulationHalt, ValidationError
from bilancio.domain.agent import AgentKind
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.clearing import settle_intraday_nets
from bilancio.engines.settlement import rollover_settled_payables, settle_due

if TYPE_CHECKING:
    from bilancio.engines.system import System

logger = logging.getLogger(__name__)

IMPACT_EVENTS = {
    "PayableSettled",
    "DeliveryObligationSettled",
    "InterbankCleared",
    "InterbankOvernightCreated",
}

DEFAULT_EVENTS = {
    "ObligationDefaulted",
    "ObligationWrittenOff",
    "AgentDefaulted",
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


def run_day(
    system: System,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
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
        enable_banking: If True, run banking subphases (bank quotes, bank lending, interbank, repayments)

    Note: Rollover is controlled by system.state.rollover_enabled (Plan 024)
    """
    current_day = system.state.day
    rollover_enabled = getattr(system.state, "rollover_enabled", False)
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
    if enable_banking and banking_sub is not None:
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

    # Plan 024: Rollover - create new payables for settled ones
    if rollover_enabled and settled_for_rollover:
        system.log("SubphaseB_Rollover")
        dealer_active = enable_dealer and system.state.dealer_subsystem is not None
        rollover_settled_payables(
            system, current_day, settled_for_rollover, dealer_active=dealer_active
        )

    # Phase C: Clear intraday nets for the current day
    system.log("PhaseC")  # optional: helps timeline
    settle_intraday_nets(system, current_day)

    # SubphaseC_Interbank: Interbank lending (optional)
    if enable_banking and banking_sub is not None:
        system.log("SubphaseC_Interbank")
        from bilancio.engines.interbank import run_interbank_lending

        interbank_events = run_interbank_lending(system, current_day, banking_sub)
        system.state.events.extend(interbank_events)

    # Phase D: CB corridor maintenance (interest + loan repayment)
    has_cb = any(agent.kind == AgentKind.CENTRAL_BANK for agent in system.state.agents.values())
    if has_cb:
        system.credit_reserve_interest(current_day)
        for loan_id in system.get_cb_loans_due(current_day):
            loan = system.state.contracts.get(loan_id)
            if loan is None:
                continue
            system.cb_repay_loan(loan_id, loan.liability_issuer_id)

    # Non-bank loan repayment
    has_lender = any(
        agent.kind == AgentKind.NON_BANK_LENDER for agent in system.state.agents.values()
    )
    if has_lender:
        from bilancio.engines.lending import run_loan_repayments

        run_loan_repayments(system, current_day)

    # Bank loan repayments
    if enable_banking and banking_sub is not None:
        from bilancio.engines.bank_lending import run_bank_loan_repayments

        bank_repay_events = run_bank_loan_repayments(system, current_day, banking_sub)
        system.state.events.extend(bank_repay_events)

    # Deposit interest accrual
    if enable_banking and banking_sub is not None:
        from bilancio.engines.bank_interest import accrue_deposit_interest

        interest_events = accrue_deposit_interest(system, current_day, banking_sub)
        system.state.events.extend(interest_events)

    # Interbank loan repayments
    if enable_banking and banking_sub is not None:
        from bilancio.engines.interbank import run_interbank_repayments

        ib_repay_events = run_interbank_repayments(system, current_day, banking_sub)
        system.state.events.extend(ib_repay_events)

    # Increment system day
    system.state.day += 1


def run_until_stable(
    system: System,
    max_days: int = 365,
    quiet_days: int = 2,
    enable_dealer: bool = False,
    enable_lender: bool = False,
    enable_rating: bool = False,
    enable_banking: bool = False,
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
        enable_banking: If True, run banking subphases each day

    Note: Rollover is controlled by system.state.rollover_enabled (Plan 024)
    """
    reports = []
    consecutive_quiet = 0
    consecutive_no_defaults = 0
    rollover_enabled = getattr(system.state, "rollover_enabled", False)

    for _ in range(max_days):
        day_before = system.state.day
        run_day(
            system,
            enable_dealer=enable_dealer,
            enable_lender=enable_lender,
            enable_rating=enable_rating,
            enable_banking=enable_banking,
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
            break

    logger.info("simulation complete: %d days", system.state.day)
    return reports
