"""Non-bank lending engine: strategy and execution.

Implements the lending phase where a non-bank lender provides cash loans
to traders who need liquidity to settle upcoming obligations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System
    from bilancio.information.profile import InformationProfile
    from bilancio.decision.protocols import (
        PortfolioStrategy, CounterpartyScreener,
        InstrumentSelector, TransactionPricer,
    )

logger = logging.getLogger(__name__)


@dataclass
class LendingConfig:
    """Configuration for the non-bank lending strategy."""
    base_rate: Decimal = Decimal("0.05")
    risk_premium_scale: Decimal = Decimal("0.20")
    max_single_exposure: Decimal = Decimal("0.15")
    max_total_exposure: Decimal = Decimal("0.80")
    maturity_days: int = 2
    horizon: int = 3
    min_shortfall: int = 1
    max_default_prob: Decimal = Decimal("0.50")
    information_profile: Optional["InformationProfile"] = None
    # Decision protocol overrides (None = auto-construct from scalar params)
    portfolio_strategy: Optional["PortfolioStrategy"] = None
    counterparty_screener: Optional["CounterpartyScreener"] = None
    instrument_selector: Optional["InstrumentSelector"] = None
    transaction_pricer: Optional["TransactionPricer"] = None


def _resolve_protocols(
    config: LendingConfig,
) -> tuple["PortfolioStrategy", "CounterpartyScreener", "InstrumentSelector", "TransactionPricer"]:
    """Build effective protocols: explicit overrides or defaults from scalar params."""
    from bilancio.decision.protocols import (
        FixedPortfolioStrategy, ThresholdScreener,
        FixedMaturitySelector, LinearPricer,
    )
    portfolio = config.portfolio_strategy or FixedPortfolioStrategy(
        max_exposure_fraction=config.max_total_exposure,
        base_return=config.base_rate,
    )
    screener = config.counterparty_screener or ThresholdScreener(
        max_default_prob=config.max_default_prob,
    )
    selector = config.instrument_selector or FixedMaturitySelector(
        maturity_days=config.maturity_days,
    )
    pricer = config.transaction_pricer or LinearPricer(
        risk_premium_scale=config.risk_premium_scale,
    )
    return portfolio, screener, selector, pricer


def run_lending_phase(
    system: "System",
    current_day: int,
    lending_config: Optional[LendingConfig] = None,
) -> List[Dict[str, Any]]:
    """Run the non-bank lending phase.

    For each eligible borrower with a shortfall, the lender offers a loan
    priced by risk. Loans are ranked by expected profit and capped by
    exposure limits.

    When an ``information_profile`` is set on the config, the lender
    observes counterparties through an InformationService that may add
    noise or block access.  When it is ``None`` (default), the lender
    has perfect (omniscient) information — identical to the original
    behavior.

    Args:
        system: The simulation system
        current_day: Current simulation day
        lending_config: Lending parameters (uses defaults if None)

    Returns:
        List of lending event dicts for aggregation
    """
    from bilancio.domain.agent import AgentKind
    from bilancio.domain.instruments.base import InstrumentKind
    from bilancio.domain.instruments.non_bank_loan import NonBankLoan

    config = lending_config or LendingConfig()
    portfolio, screener, selector, pricer = _resolve_protocols(config)
    events: List[Dict[str, Any]] = []

    # Find the lender
    lender_id = None
    for agent_id, agent in system.state.agents.items():
        if agent.kind == AgentKind.NON_BANK_LENDER and not agent.defaulted:
            lender_id = agent_id
            break

    if lender_id is None:
        return events

    # Build information service if profile is configured
    info = None
    if config.information_profile is not None:
        from bilancio.information.service import InformationService
        info = InformationService(
            system, config.information_profile, observer_id=lender_id,
        )

    # Calculate lender's available capital (own data — always perfect)
    lender_cash = _get_agent_cash(system, lender_id)
    existing_loan_exposure = (
        info.get_loan_exposure(lender_id) if info is not None
        else _get_loan_exposure(system, lender_id)
    )
    initial_capital = lender_cash + existing_loan_exposure

    if initial_capital <= 0:
        return events

    max_total = portfolio.max_exposure(initial_capital)
    available = min(lender_cash, max_total - existing_loan_exposure)
    if available <= 0:
        return events

    # Build raw default probs only when not using info service
    default_probs = None if info is not None else _estimate_default_probs(system, current_day)

    # Identify borrowers with shortfalls
    opportunities: List[Dict[str, Any]] = []
    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            continue
        if agent.kind not in (AgentKind.HOUSEHOLD, AgentKind.FIRM):
            continue

        # Calculate upcoming obligations within horizon
        if info is not None:
            upcoming_due = info.get_counterparty_obligations(agent_id, current_day, config.horizon)
            if upcoming_due is None:
                continue  # Can't observe → skip
            agent_cash = info.get_counterparty_cash(agent_id, current_day)
            if agent_cash is None:
                agent_cash = 0  # Conservative: assume no cash
        else:
            upcoming_due = _get_upcoming_obligations(system, agent_id, current_day, config.horizon)
            agent_cash = _get_agent_cash(system, agent_id)

        shortfall = upcoming_due - agent_cash

        if shortfall < config.min_shortfall:
            continue

        # Risk assessment
        if info is not None:
            p_default = info.get_default_probability(agent_id, current_day)
            if p_default is None:
                p_default = Decimal("0.15")  # Prior when unobservable
        else:
            p_default = default_probs.get(agent_id, Decimal("0.15"))
        if not screener.is_eligible(p_default):
            continue

        # Price the loan
        base_rate = portfolio.target_return()
        rate = pricer.price(base_rate, p_default)

        # Check per-borrower exposure limit (own data — always perfect)
        if info is not None:
            borrower_existing = info.get_borrower_exposure(lender_id, agent_id)
        else:
            borrower_existing = _get_borrower_exposure(system, lender_id, agent_id)
        max_single = int(config.max_single_exposure * initial_capital)
        max_to_this_borrower = max_single - borrower_existing
        if max_to_this_borrower <= 0:
            continue

        loan_amount = min(shortfall, max_to_this_borrower)

        # Expected profit for ranking
        expected_profit = float(rate) * (1.0 - float(p_default))

        opportunities.append({
            "borrower_id": agent_id,
            "amount": loan_amount,
            "rate": rate,
            "p_default": p_default,
            "expected_profit": expected_profit,
            "shortfall": shortfall,
        })

    # Rank by expected profit descending
    opportunities.sort(key=lambda x: x["expected_profit"], reverse=True)

    # Execute loans
    remaining_capital = available
    for opp in opportunities:
        if remaining_capital <= 0:
            break

        loan_amount = min(opp["amount"], remaining_capital)
        if loan_amount <= 0:
            continue

        try:
            loan_id = system.nonbank_lend_cash(
                lender_id=lender_id,
                borrower_id=opp["borrower_id"],
                amount=loan_amount,
                rate=opp["rate"],
                day=current_day,
                maturity_days=selector.select_maturity(),
            )
            remaining_capital -= loan_amount
            events.append({
                "kind": "NonBankLoanCreated",
                "day": current_day,
                "lender_id": lender_id,
                "borrower_id": opp["borrower_id"],
                "amount": loan_amount,
                "rate": str(opp["rate"]),
                "loan_id": loan_id,
                "p_default": str(opp["p_default"]),
            })
            logger.debug(
                "Loan created: %s -> %s, amount=%d, rate=%s",
                lender_id, opp["borrower_id"], loan_amount, opp["rate"],
            )
        except Exception as e:
            logger.warning("Failed to create loan to %s: %s", opp["borrower_id"], e)
            continue

    return events


def run_loan_repayments(system: "System", current_day: int) -> List[Dict[str, Any]]:
    """Process repayments for all non-bank loans due today.

    Args:
        system: The simulation system
        current_day: Current simulation day

    Returns:
        List of repayment event dicts
    """
    events: List[Dict[str, Any]] = []

    due_loans = system.get_nonbank_loans_due(current_day)
    for loan_id in due_loans:
        loan = system.state.contracts.get(loan_id)
        if loan is None:
            continue

        borrower_id = loan.liability_issuer_id
        try:
            repaid = system.nonbank_repay_loan(loan_id, borrower_id)
            events.append({
                "kind": "NonBankLoanRepaid" if repaid else "NonBankLoanDefaulted",
                "day": current_day,
                "loan_id": loan_id,
                "borrower_id": borrower_id,
                "repaid": repaid,
            })
        except Exception as e:
            logger.warning("Loan repayment failed for %s: %s", loan_id, e)

    return events


# ── Helper functions ─────────────────────────────────────────────────

def _get_agent_cash(system: "System", agent_id: str) -> int:
    """Get total cash held by an agent."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None and contract.kind == InstrumentKind.CASH:
            total += contract.amount
    return total


def _get_loan_exposure(system: "System", lender_id: str) -> int:
    """Get total outstanding loan principal for a lender."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(lender_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None and contract.kind == InstrumentKind.NON_BANK_LOAN:
            total += contract.amount
    return total


def _get_borrower_exposure(system: "System", lender_id: str, borrower_id: str) -> int:
    """Get existing loan exposure from lender to a specific borrower."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(lender_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if (contract is not None
                and contract.kind == InstrumentKind.NON_BANK_LOAN
                and contract.liability_issuer_id == borrower_id):
            total += contract.amount
    return total


def _get_upcoming_obligations(
    system: "System", agent_id: str, current_day: int, horizon: int
) -> int:
    """Get total obligations due within horizon days for an agent."""
    from bilancio.domain.instruments.base import InstrumentKind
    from bilancio.domain.instruments.non_bank_loan import NonBankLoan

    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    for cid in agent.liability_ids:
        contract = system.state.contracts.get(cid)
        if contract is None:
            continue
        due_day = contract.due_day
        if due_day is None:
            # NonBankLoans use maturity_day property
            if isinstance(contract, NonBankLoan):
                due_day = contract.maturity_day
            else:
                continue
        if current_day <= due_day <= current_day + horizon:
            if contract.kind == InstrumentKind.PAYABLE:
                total += contract.amount
            elif isinstance(contract, NonBankLoan):
                total += contract.repayment_amount
    return total


def _estimate_default_probs(
    system: "System", current_day: int
) -> Dict[str, Decimal]:
    """Estimate default probability per agent.

    Reuses dealer subsystem's RiskAssessor if available,
    otherwise uses a simple heuristic based on default history.
    """
    probs: Dict[str, Decimal] = {}

    # Try to use the dealer subsystem's risk assessor
    dealer_sub = system.state.dealer_subsystem
    if dealer_sub is not None and hasattr(dealer_sub, 'risk_assessor') and dealer_sub.risk_assessor is not None:
        assessor = dealer_sub.risk_assessor
        for agent_id in system.state.agents:
            agent = system.state.agents[agent_id]
            if agent.defaulted:
                probs[agent_id] = Decimal("1.0")
                continue
            p = assessor.estimate_default_prob(agent_id)
            probs[agent_id] = Decimal(str(p)) if p is not None else Decimal("0.15")
        return probs

    # Try rating registry (institutional source)
    rating_registry = getattr(system.state, 'rating_registry', None)
    if rating_registry:
        for agent_id, agent in system.state.agents.items():
            if agent.defaulted:
                probs[agent_id] = Decimal("1.0")
            elif agent_id in rating_registry:
                probs[agent_id] = rating_registry[agent_id]
            else:
                probs[agent_id] = Decimal("0.15")
        return probs

    # Fallback: simple heuristic based on defaulted agents in the system
    n_agents = len(system.state.agents)
    n_defaulted = len(system.state.defaulted_agent_ids)
    base_rate = Decimal(str(n_defaulted / max(n_agents, 1)))

    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            probs[agent_id] = Decimal("1.0")
        else:
            # Use system-wide base + small per-agent noise
            probs[agent_id] = max(Decimal("0.01"), min(Decimal("0.99"), base_rate + Decimal("0.05")))

    return probs
