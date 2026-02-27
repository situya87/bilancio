"""Non-bank lending engine: strategy and execution.

Implements the lending phase where a non-bank lender provides cash loans
to traders who need liquidity to settle upcoming obligations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bilancio.core.errors import ValidationError

if TYPE_CHECKING:
    from bilancio.decision.profiles import LenderProfile
    from bilancio.decision.protocols import (
        CounterpartyScreener,
        InstrumentSelector,
        PortfolioStrategy,
        TransactionPricer,
    )
    from bilancio.decision.risk_assessment import RiskAssessor
    from bilancio.engines.system import System
    from bilancio.information.profile import InformationProfile

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
    information_profile: InformationProfile | None = None
    lender_profile: LenderProfile | None = None
    max_ring_maturity: int | None = None  # computed at ring creation time
    risk_assessor: RiskAssessor | None = None  # persists across days for Bayesian learning
    initial_prior: Decimal = Decimal("0.15")  # κ-informed default probability prior
    min_coverage_ratio: Decimal = Decimal("0")
    # Decision protocol overrides (None = auto-construct from scalar params)
    portfolio_strategy: PortfolioStrategy | None = None
    counterparty_screener: CounterpartyScreener | None = None
    instrument_selector: InstrumentSelector | None = None
    transaction_pricer: TransactionPricer | None = None


def _resolve_protocols(
    config: LendingConfig,
) -> tuple[PortfolioStrategy, CounterpartyScreener, InstrumentSelector, TransactionPricer]:
    """Build effective protocols: explicit overrides or defaults from scalar params."""
    from bilancio.decision.protocols import (
        FixedMaturitySelector,
        FixedPortfolioStrategy,
        LinearPricer,
        ThresholdScreener,
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
    system: System,
    current_day: int,
    lending_config: LendingConfig | None = None,
) -> list[dict[str, Any]]:
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

    config = lending_config or LendingConfig()
    profile = config.lender_profile
    portfolio, screener, selector, pricer = _resolve_protocols(config)
    events: list[dict[str, Any]] = []

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
            system,
            config.information_profile,
            observer_id=lender_id,
        )

    # Calculate lender's available capital (own data — always perfect)
    lender_cash = _get_agent_cash(system, lender_id)
    existing_loan_exposure = (
        info.get_loan_exposure(lender_id)
        if info is not None
        else _get_loan_exposure(system, lender_id)
    )
    performing_exposure = _get_performing_loan_exposure(system, lender_id)
    initial_capital = lender_cash + performing_exposure

    if initial_capital <= 0:
        return events

    max_total = portfolio.max_exposure(initial_capital)
    available = min(lender_cash, max_total - existing_loan_exposure)
    if available <= 0:
        return events

    # Build raw default probs only when not using info service
    default_probs = None if info is not None else _estimate_default_probs(system, current_day)

    # Identify borrowers with shortfalls
    opportunities: list[dict[str, Any]] = []
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
                p_default = config.initial_prior  # Prior when unobservable
        else:
            assert default_probs is not None
            p_default = default_probs.get(agent_id, config.initial_prior)
        if not screener.is_eligible(p_default):
            continue

        # Coverage gate: reject if borrower's balance sheet coverage is too low
        if config.min_coverage_ratio > 0:
            coverage = _assess_borrower_nbfi(
                system, agent_id, min(shortfall, int(config.max_single_exposure * initial_capital)),
                config.lender_profile.profit_target if config.lender_profile else config.base_rate,
                current_day, config.horizon,
            )
            if coverage < config.min_coverage_ratio:
                events.append({
                    "kind": "NonBankLoanRejectedCoverage",
                    "day": current_day,
                    "lender_id": lender_id,
                    "borrower_id": agent_id,
                    "coverage": str(coverage),
                    "min_coverage": str(config.min_coverage_ratio),
                })
                continue

        # If LenderProfile available, use kappa-aware pricing
        if profile is not None:
            # Compute coverage ratio: (cash + receivables) / upcoming_obligations
            receivables = _quality_adjusted_receivables(
                system, agent_id, current_day, profile.planning_horizon
            )
            total_resources = max(agent_cash + receivables, 0)
            coverage = Decimal(str(total_resources)) / Decimal(str(max(upcoming_due, 1)))
            # Coverage-based default estimate (original heuristic)
            p_coverage = profile.base_default_estimate * (
                Decimal("1") / max(coverage, Decimal("0.01"))
            )
            p_coverage = max(Decimal("0.01"), min(Decimal("0.95"), p_coverage))

            # Blend with Bayesian posterior when assessor is available
            if (
                config.risk_assessor is not None
                and profile.risk_assessment_params is not None
            ):
                p_bayesian = config.risk_assessor.estimate_default_prob(
                    agent_id, current_day
                )
                # Count observations within the same lookback window used by the
                # posterior (estimate_default_prob).  Using lifetime history would
                # lock w_bayes=1 after enough old observations, permanently
                # suppressing the coverage component even when recent data is stale.
                tracker = config.risk_assessor.belief_tracker
                window_start = current_day - tracker.lookback_window
                issuer_hist = tracker.issuer_history.get(agent_id, [])
                n = sum(1 for day, _ in issuer_hist if day >= window_start)
                w_bayes = min(
                    Decimal("1"),
                    Decimal(str(n)) / Decimal(str(profile.warmup_observations)),
                )
                p_default = w_bayes * p_bayesian + (Decimal("1") - w_bayes) * p_coverage
            else:
                p_default = p_coverage

            p_default = max(Decimal("0.01"), min(Decimal("0.95"), p_default))

        # Price the loan
        if profile is not None:
            base_rate = profile.profit_target
            rate = base_rate + profile.risk_premium_scale * p_default
        else:
            base_rate = portfolio.target_return()
            rate = pricer.price(base_rate, p_default)

        # Anchor rate to CB corridor when banking subsystem is active
        banking_sub = getattr(system.state, "banking_subsystem", None)
        if banking_sub is not None:
            from decimal import Decimal as D
            r_floor = banking_sub.bank_profile.r_floor(banking_sub.kappa)
            omega = banking_sub.bank_profile.corridor_width(banking_sub.kappa)
            p_0 = D("1") / (D("1") + banking_sub.kappa)
            if p_0 > D("0"):
                rate = r_floor + omega * (p_default / p_0)
            else:
                rate = r_floor + omega

        # Borrow-vs-sell: skip this loan if selling is cheaper
        selling_cost = _expected_selling_cost(system, agent_id, current_day)
        if selling_cost is not None and selling_cost < rate:
            continue  # Selling a claim is cheaper than borrowing

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

        opportunities.append(
            {
                "borrower_id": agent_id,
                "amount": loan_amount,
                "rate": rate,
                "p_default": p_default,
                "expected_profit": expected_profit,
                "shortfall": shortfall,
            }
        )

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
            effective_maturity = selector.select_maturity()
            if profile is not None and config.max_ring_maturity is not None:
                effective_maturity = min(profile.max_loan_maturity, config.max_ring_maturity)
            # Align NBFI loan maturity with bank loan maturity when banking is active
            banking_sub_mat = getattr(system.state, "banking_subsystem", None)
            if banking_sub_mat is not None:
                effective_maturity = banking_sub_mat.loan_maturity
            loan_id = system.nonbank_lend(
                lender_id=lender_id,
                borrower_id=opp["borrower_id"],
                amount=loan_amount,
                rate=opp["rate"],
                day=current_day,
                maturity_days=effective_maturity,
            )
            remaining_capital -= loan_amount
            events.append(
                {
                    "kind": "NonBankLoanCreated",
                    "day": current_day,
                    "lender_id": lender_id,
                    "borrower_id": opp["borrower_id"],
                    "amount": loan_amount,
                    "rate": str(opp["rate"]),
                    "loan_id": loan_id,
                    "p_default": str(opp["p_default"]),
                }
            )
            logger.debug(
                "Loan created: %s -> %s, amount=%d, rate=%s",
                lender_id,
                opp["borrower_id"],
                loan_amount,
                opp["rate"],
            )
        except (ValidationError, ValueError, KeyError) as e:
            logger.warning("Failed to create loan to %s: %s", opp["borrower_id"], e)
            continue

    return events


def run_loan_repayments(system: System, current_day: int) -> list[dict[str, Any]]:
    """Process repayments for all non-bank loans due today.

    Args:
        system: The simulation system
        current_day: Current simulation day

    Returns:
        List of repayment event dicts
    """
    events: list[dict[str, Any]] = []

    # Get assessor from lending config (if available) for Bayesian learning
    lender_config = getattr(system.state, "lender_config", None)
    assessor = lender_config.risk_assessor if lender_config is not None else None

    due_loans = system.get_nonbank_loans_due(current_day)
    for loan_id in due_loans:
        loan = system.state.contracts.get(loan_id)
        if loan is None:
            continue

        borrower_id = loan.liability_issuer_id
        try:
            repaid = system.nonbank_repay_loan(loan_id, borrower_id)
            events.append(
                {
                    "kind": "NonBankLoanRepaid" if repaid else "NonBankLoanDefaulted",
                    "day": current_day,
                    "loan_id": loan_id,
                    "borrower_id": borrower_id,
                    "repaid": repaid,
                }
            )
            # Update NBFI BeliefTracker with loan outcome
            if assessor is not None:
                assessor.update_history(current_day, borrower_id, defaulted=not repaid)
        except (ValidationError, ValueError, KeyError) as e:
            logger.warning("Loan repayment failed for %s: %s", loan_id, e)

    return events


# ── Helper functions ─────────────────────────────────────────────────


def _get_agent_cash(system: System, agent_id: str) -> int:
    """Get total cash held by an agent."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None and contract.kind in (InstrumentKind.CASH, InstrumentKind.BANK_DEPOSIT):
            total += contract.amount
    return total


def _get_loan_exposure(system: System, lender_id: str) -> int:
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


def _get_borrower_exposure(system: System, lender_id: str, borrower_id: str) -> int:
    """Get existing loan exposure from lender to a specific borrower."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(lender_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if (
            contract is not None
            and contract.kind == InstrumentKind.NON_BANK_LOAN
            and contract.liability_issuer_id == borrower_id
        ):
            total += contract.amount
    return total


def _get_upcoming_obligations(system: System, agent_id: str, current_day: int, horizon: int) -> int:
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


def _expected_selling_cost(system: System, agent_id: str, current_day: int) -> Decimal | None:
    """Estimate the haircut from selling a claim on the secondary market.

    Returns the expected cost as a fraction (e.g., 0.15 = 15% haircut),
    or None if no dealer is available (can't sell).
    """
    dealer_sub = system.state.dealer_subsystem
    if dealer_sub is None:
        return None  # No dealer → can't sell

    # Find the agent's cheapest-to-sell payable (receivable = asset payable)
    from bilancio.domain.instruments.base import InstrumentKind

    agent = system.state.agents.get(agent_id)
    if agent is None:
        return None

    best_haircut: Decimal | None = None
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None or contract.kind != InstrumentKind.PAYABLE:
            continue
        if contract.due_day is None or contract.due_day <= current_day:
            continue
        # Get dealer bid price for this claim
        debtor_id = contract.liability_issuer_id
        if not hasattr(dealer_sub, "risk_assessor") or dealer_sub.risk_assessor is None:
            continue
        p_default = dealer_sub.risk_assessor.estimate_default_prob(debtor_id, current_day)
        if p_default is None:
            p_default = Decimal("0.15")
        else:
            p_default = Decimal(str(p_default))
        # VBT credit-adjusted mid: M = outside_mid_ratio × (1 - P_default)
        # Bid = M - spread/2. Haircut = 1 - bid_price_ratio
        # Approximate: haircut ≈ P_default + spread/2
        spread = Decimal("0.05")  # typical dealer spread
        haircut = p_default + spread / 2
        haircut = max(Decimal("0.01"), min(Decimal("0.99"), haircut))
        if best_haircut is None or haircut < best_haircut:
            best_haircut = haircut

    return best_haircut


def _get_receivables_due_within(
    system: System, agent_id: str, current_day: int, horizon: int
) -> int:
    """Get total receivables (asset payables) due within horizon days."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None or contract.kind != InstrumentKind.PAYABLE:
            continue
        due_day = contract.due_day
        if due_day is not None and current_day <= due_day <= current_day + horizon:
            total += contract.amount
    return total


def _quality_adjusted_receivables(
    system: System, agent_id: str, current_day: int, horizon: int
) -> int:
    """Receivables due within horizon, excluding defaulted counterparties."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return 0
    defaulted = system.state.defaulted_agent_ids
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is None or contract.kind != InstrumentKind.PAYABLE:
            continue
        due_day = contract.due_day
        if due_day is not None and current_day <= due_day <= current_day + horizon:
            if contract.liability_issuer_id not in defaulted:
                total += contract.amount
    return total


def _get_performing_loan_exposure(system: System, lender_id: str) -> int:
    """Loan exposure excluding loans to defaulted borrowers."""
    from bilancio.domain.instruments.base import InstrumentKind

    total = 0
    agent = system.state.agents.get(lender_id)
    if agent is None:
        return 0
    defaulted = system.state.defaulted_agent_ids
    for cid in agent.asset_ids:
        contract = system.state.contracts.get(cid)
        if contract is not None and contract.kind == InstrumentKind.NON_BANK_LOAN:
            if contract.liability_issuer_id not in defaulted:
                total += contract.amount
    return total


def _assess_borrower_nbfi(
    system: System,
    agent_id: str,
    loan_amount: int,
    rate: Decimal,
    current_day: int,
    horizon: int,
) -> Decimal:
    """Assess NBFI borrower's repayment capacity via balance sheet analysis.

    Mirrors bank_lending._assess_borrower: projects cash position at repayment
    by combining liquid assets, quality-adjusted receivables, and obligations.

    Returns coverage ratio: net_resources / loan_repayment.
    """
    agent = system.state.agents.get(agent_id)
    if agent is None:
        return Decimal("-1")

    loan_repayment = int(Decimal(loan_amount) * (Decimal("1") + rate))
    if loan_repayment <= 0:
        return Decimal("999")

    liquid = _get_agent_cash(system, agent_id)
    quality_receivables = _quality_adjusted_receivables(
        system, agent_id, current_day, horizon
    )
    obligations = _get_upcoming_obligations(system, agent_id, current_day, horizon)

    net_resources = liquid + quality_receivables - obligations
    return Decimal(net_resources) / Decimal(loan_repayment)


def _estimate_default_probs(
    system: System,
    current_day: int,
    log_estimates: bool = False,
    channel_bindings: tuple[Any, ...] = (),
) -> dict[str, Decimal]:
    """Estimate default probability per agent.

    When *channel_bindings* are provided, sources are tried in priority
    order.  Otherwise the legacy waterfall (dealer → registry → heuristic)
    is used.

    Reuses dealer subsystem's RiskAssessor if available,
    otherwise uses a simple heuristic based on default history.
    """
    if channel_bindings:
        bindings = sorted(
            [b for b in channel_bindings if b.category == "default_prob"],
            key=lambda b: b.priority,
        )
        for binding in bindings:
            result = _try_default_prob_source(
                system,
                current_day,
                binding.source,
                log_estimates,
            )
            if result is not None:
                return result
        # All declared sources unavailable — heuristic fallback
        return _default_prob_heuristic(system, current_day)

    # No bindings: legacy waterfall
    return (
        _default_prob_from_dealer(system, current_day, log_estimates)
        or _default_prob_from_registry(system, current_day)
        or _default_prob_heuristic(system, current_day)
    )


def _try_default_prob_source(
    system: System,
    current_day: int,
    source: str,
    log_estimates: bool = False,
) -> dict[str, Decimal] | None:
    """Dispatch to a named default-prob source."""
    if source == "dealer_risk_assessor":
        return _default_prob_from_dealer(system, current_day, log_estimates)
    if source == "rating_registry":
        return _default_prob_from_registry(system, current_day)
    if source == "system_heuristic":
        return _default_prob_heuristic(system, current_day)
    return None


def _default_prob_from_dealer(
    system: System,
    current_day: int,
    log_estimates: bool = False,
) -> dict[str, Decimal] | None:
    """Use the dealer subsystem's RiskAssessor for default probs."""
    dealer_sub = system.state.dealer_subsystem
    if (
        dealer_sub is None
        or not hasattr(dealer_sub, "risk_assessor")
        or dealer_sub.risk_assessor is None
    ):
        return None
    probs: dict[str, Decimal] = {}
    assessor = dealer_sub.risk_assessor
    for agent_id in system.state.agents:
        agent = system.state.agents[agent_id]
        if agent.defaulted:
            probs[agent_id] = Decimal("1.0")
            continue
        if log_estimates and hasattr(assessor, "estimate_default_prob_detail"):
            est = assessor.estimate_default_prob_detail(
                agent_id,
                current_day,
                estimator_id="lender",
            )
            system.log_estimate(est)
            probs[agent_id] = est.value
        else:
            p = assessor.estimate_default_prob(agent_id, current_day)
            probs[agent_id] = Decimal(str(p)) if p is not None else Decimal("0.15")
    return probs


def _default_prob_from_registry(
    system: System,
    current_day: int,
) -> dict[str, Decimal] | None:
    """Use the rating registry for default probs."""
    rating_registry = getattr(system.state, "rating_registry", None)
    if not rating_registry:
        return None
    probs: dict[str, Decimal] = {}
    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            probs[agent_id] = Decimal("1.0")
        elif agent_id in rating_registry:
            probs[agent_id] = rating_registry[agent_id]
        else:
            probs[agent_id] = Decimal("0.15")
    return probs


def _default_prob_heuristic(
    system: System,
    current_day: int,
) -> dict[str, Decimal]:
    """System-wide heuristic: base_rate + margin."""
    probs: dict[str, Decimal] = {}
    n_agents = len(system.state.agents)
    n_defaulted = len(system.state.defaulted_agent_ids)
    base_rate = Decimal(str(n_defaulted / max(n_agents, 1)))
    for agent_id, agent in system.state.agents.items():
        if agent.defaulted:
            probs[agent_id] = Decimal("1.0")
        else:
            probs[agent_id] = max(
                Decimal("0.01"),
                min(Decimal("0.99"), base_rate + Decimal("0.05")),
            )
    return probs
