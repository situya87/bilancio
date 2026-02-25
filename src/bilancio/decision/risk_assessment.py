"""
Risk assessment module for traders in dealer subsystem.

This module enables traders to:
1. Estimate default probabilities for issuers
2. Compute expected values of holding receivables
3. Make rational buy/sell decisions based on price vs expected value

The pipeline is decomposed into four independently usable stages:
- **BeliefTracker** — Bayesian default probability estimation
- **EVValuer** — expected hold-value computation
- **PositionAssessor** — urgency-adjusted risk thresholds
- **TradeGate** — accept/reject gate for buy and sell decisions

``RiskAssessor`` is a convenience wrapper that composes all four.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bilancio.core.ids import AgentId

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bilancio.dealer.models import Ticket
    from bilancio.decision.protocols import InstrumentValuer
    from bilancio.information.estimates import Estimate


@dataclass
class RiskAssessmentParams:
    """
    Parameters for trader risk assessment.

    Attributes:
        lookback_window: Number of days to look back for default history
        smoothing_alpha: Laplace smoothing parameter (handles small samples)
        base_risk_premium: Seller premium (fraction of face value).
            Default 0: selling converts uncertainty to certainty, so no premium needed.
        urgency_sensitivity: How much liquidity urgency reduces threshold
        use_issuer_specific: If True, track per-issuer rates; if False, use system-wide
        buy_premium_multiplier: Buyers require higher premium than sellers
        buy_risk_premium: Buyer premium (fraction of face value).
            Default 0.01: buyers demand a premium for taking on default risk.
    """

    lookback_window: int = 5
    smoothing_alpha: Decimal = Decimal("1.0")
    base_risk_premium: Decimal = Decimal(
        "0"
    )  # Seller premium: 0 (selling converts uncertainty to certainty)
    urgency_sensitivity: Decimal = Decimal("0.10")  # 10% sensitivity
    use_issuer_specific: bool = False
    buy_premium_multiplier: Decimal = Decimal("1.0")  # Buyers use same premium as sellers
    buy_risk_premium: Decimal = Decimal("0.01")  # Buyer premium: 1%
    default_observability: Decimal = Decimal("1.0")  # 0=ignore observed defaults, 1=full tracking
    initial_prior: Decimal = Decimal(
        "0.15"
    )  # No-history default prior (can be overridden by informedness)


# ---------------------------------------------------------------------------
# Stage 1: Belief Tracker
# ---------------------------------------------------------------------------


class BeliefTracker:
    """Tracks payment outcomes and estimates default probabilities.

    Uses Laplace-smoothed Bayesian estimation with configurable
    lookback window and observability friction.
    """

    def __init__(
        self,
        lookback_window: int = 5,
        smoothing_alpha: Decimal = Decimal("1.0"),
        default_observability: Decimal = Decimal("1.0"),
        initial_prior: Decimal = Decimal("0.15"),
        use_issuer_specific: bool = False,
        information_service: Any = None,  # Optional InformationService
    ):
        self.lookback_window = lookback_window
        self.smoothing_alpha = smoothing_alpha
        self.default_observability = default_observability
        self.initial_prior = initial_prior
        self.use_issuer_specific = use_issuer_specific
        self._information_service = information_service

        # Track system-wide default history: (day, issuer_id, defaulted)
        self.payment_history: list[tuple[int, AgentId, bool]] = []

        # Track per-issuer history (if issuer-specific enabled)
        self.issuer_history: dict[AgentId, list[tuple[int, bool]]] = {}

    @property
    def information_service(self) -> Any:
        """The InformationService used for queries, if any."""
        return self._information_service

    @information_service.setter
    def information_service(self, value: Any) -> None:
        self._information_service = value

    def update_history(self, day: int, issuer_id: AgentId, defaulted: bool) -> None:
        """
        Record a payment outcome (success or default).

        Should be called at the end of each day after settlements are processed.

        Args:
            day: Simulation day
            issuer_id: Agent who was supposed to make payment
            defaulted: True if payment failed, False if succeeded
        """
        # Add to system-wide history
        self.payment_history.append((day, issuer_id, defaulted))

        # Add to per-issuer history (if enabled)
        if self.use_issuer_specific:
            if issuer_id not in self.issuer_history:
                self.issuer_history[issuer_id] = []
            self.issuer_history[issuer_id].append((day, defaulted))

    def estimate_default_prob(self, issuer_id: AgentId, current_day: int) -> Decimal:
        """
        Estimate probability that issuer will default on obligations.

        Uses recent payment history within the lookback window.
        Applies Laplace smoothing to handle small samples.

        When an InformationService is attached, delegates to it instead
        of using internal payment history.

        Args:
            issuer_id: Agent whose default probability to estimate
            current_day: Current simulation day

        Returns:
            Estimated default probability in [0, 1]
        """
        # Delegate to InformationService when available
        if self._information_service is not None:
            p = self._information_service.get_default_probability(issuer_id, current_day)
            if p is not None:
                return p
            # Fall through to internal estimation if service returns None

        window_start = current_day - self.lookback_window

        if self.use_issuer_specific and issuer_id in self.issuer_history:
            # Use issuer-specific history
            history = self.issuer_history[issuer_id]
            recent = [(d, defaulted) for d, defaulted in history if d >= window_start]
        else:
            # Use system-wide history
            recent = [
                (d, defaulted)
                for d, agent_id, defaulted in self.payment_history
                if d >= window_start
            ]

        if not recent:
            # No recent data: use configured initial prior.
            # Default 0.15 reflects moderate uncertainty with zero observations,
            # allowing initial trades at market prices before data arrives.
            # When informedness (alpha) is used, this is a blended prior
            # incorporating the kappa-implied default probability.
            return self.initial_prior

        # Count defaults and total payments
        defaults = sum(1 for _, defaulted in recent if defaulted)
        total = len(recent)

        # Laplace smoothing: (alpha + defaults) / (2*alpha + total)
        # This prevents extreme estimates from small samples
        alpha = self.smoothing_alpha
        p_default = (alpha + Decimal(defaults)) / (Decimal(2) * alpha + Decimal(total))

        # Blend observed rate with prior based on observability:
        # obs=1.0 (default): full tracking of observed defaults
        # obs=0.0: always return initial_prior regardless of observed data
        if self.default_observability == Decimal("1"):
            return p_default
        return self.initial_prior + self.default_observability * (
            p_default - self.initial_prior
        )

    def estimate_default_prob_detail(
        self,
        issuer_id: AgentId,
        current_day: int,
        estimator_id: str = "system",
    ) -> "Estimate":
        """Return an Estimate wrapping estimate_default_prob() with provenance."""
        # Delegate to InformationService when available
        if self._information_service is not None:
            detail = self._information_service.get_default_probability_detail(issuer_id, current_day)
            if detail is not None:
                return detail
            # Fall through to internal estimation if service returns None

        from bilancio.information.estimates import Estimate

        value = self.estimate_default_prob(issuer_id, current_day)

        window_start = current_day - self.lookback_window

        if self.use_issuer_specific and issuer_id in self.issuer_history:
            recent = [
                (d, defaulted)
                for d, defaulted in self.issuer_history[issuer_id]
                if d >= window_start
            ]
        else:
            recent = [
                (d, defaulted) for d, _, defaulted in self.payment_history if d >= window_start
            ]

        defaults_count = sum(1 for _, defaulted in recent if defaulted)
        total_observations = len(recent)
        used_prior = total_observations == 0

        return Estimate(
            value=value,
            estimator_id=estimator_id,
            target_id=str(issuer_id),
            target_type="agent",
            estimation_day=current_day,
            method="bayesian_default_prob",
            inputs={
                "defaults_count": defaults_count,
                "total_observations": total_observations,
                "used_prior": used_prior,
            },
            metadata={
                "lookback_window": self.lookback_window,
                "smoothing_alpha": str(self.smoothing_alpha),
                "initial_prior": str(self.initial_prior),
                "default_observability": str(self.default_observability),
            },
        )


# ---------------------------------------------------------------------------
# Stage 2: EV Valuer
# ---------------------------------------------------------------------------


class EVValuer:
    """Computes expected hold value: EV = (1 - P_default) x face.

    Uses a BeliefTracker (or any object with an ``estimate_default_prob``
    method) as its belief source.  When an ``instrument_valuer`` is
    injected, delegates to it instead of using the built-in formula.
    """

    def __init__(self, belief_source: BeliefTracker, instrument_valuer: InstrumentValuer | None = None):
        self.belief_source = belief_source
        self.instrument_valuer = instrument_valuer

    def expected_value(self, ticket: Ticket, current_day: int) -> Decimal:
        """
        Compute expected value of holding a ticket to maturity.

        EV = (1 - P(default)) * face_value

        When an ``instrument_valuer`` is injected, delegates to it instead
        of using the built-in formula.

        Args:
            ticket: Ticket (receivable) to value
            current_day: Current simulation day

        Returns:
            Expected payoff from holding (in same units as face value)
        """
        if self.instrument_valuer is not None:
            result: Decimal = self.instrument_valuer.value_decimal(ticket, current_day)
            return result
        p_default = self.belief_source.estimate_default_prob(ticket.issuer_id, current_day)
        ev = (Decimal(1) - p_default) * ticket.face
        return ev

    def expected_value_detail(
        self,
        ticket: Ticket,
        current_day: int,
        estimator_id: str = "system",
    ) -> "Estimate":
        """Return an Estimate wrapping expected_value() with provenance.

        When an ``instrument_valuer`` is injected, delegates to its
        ``value()`` method instead of using the built-in formula.
        """
        if self.instrument_valuer is not None:
            ev_est = self.instrument_valuer.value(ticket, current_day)
            return cast("Estimate", ev_est)
        from bilancio.information.estimates import Estimate

        value = self.expected_value(ticket, current_day)
        p_default_est = self.belief_source.estimate_default_prob_detail(
            ticket.issuer_id,
            current_day,
            estimator_id,
        )

        return Estimate(
            value=value,
            estimator_id=estimator_id,
            target_id=str(ticket.id),
            target_type="instrument",
            estimation_day=current_day,
            method="ev_hold",
            inputs={
                "p_default_estimate": p_default_est,
                "face_value": str(ticket.face),
            },
            metadata={
                "issuer_id": str(ticket.issuer_id),
                "maturity_day": ticket.maturity_day,
                "bucket_id": str(ticket.bucket_id),
            },
        )


# ---------------------------------------------------------------------------
# Stage 3: Position Assessor
# ---------------------------------------------------------------------------


class PositionAssessor:
    """Computes urgency-adjusted risk thresholds from liquidity position.

    threshold_eff = base - urgency_sensitivity * (shortfall / wealth)
    """

    def __init__(
        self,
        base_risk_premium: Decimal = Decimal("0"),
        urgency_sensitivity: Decimal = Decimal("0.10"),
    ):
        self.base_risk_premium = base_risk_premium
        self.urgency_sensitivity = urgency_sensitivity

    def compute_effective_threshold(
        self, cash: Decimal, shortfall: Decimal, asset_value: Decimal
    ) -> Decimal:
        """
        Compute effective risk premium threshold based on liquidity urgency.

        When trader has severe liquidity needs, threshold decreases
        (willing to accept worse prices).

        threshold_eff = threshold_base - urgency_sensitivity * (shortfall / wealth)

        Args:
            cash: Current cash holdings
            shortfall: Immediate payment shortfall (positive if needs cash)
            asset_value: Total value of receivables held

        Returns:
            Effective risk premium threshold (can be negative if desperate)
        """
        wealth = cash + asset_value

        if wealth <= 0:
            # Desperate situation: accept any price
            return Decimal("-1.0")

        if shortfall <= 0:
            # No urgency: use base threshold
            return self.base_risk_premium

        # Compute urgency ratio: shortfall as fraction of wealth
        urgency_ratio = shortfall / wealth

        # Reduce threshold based on urgency
        threshold_eff = (
            self.base_risk_premium - self.urgency_sensitivity * urgency_ratio
        )

        return threshold_eff


# ---------------------------------------------------------------------------
# Stage 4: Trade Gate
# ---------------------------------------------------------------------------


class TradeGate:
    """Accept/reject gate for buy and sell trade decisions.

    Combines valuation, position assessment, and threshold logic
    to produce boolean trade decisions.
    """

    def __init__(
        self,
        valuer: EVValuer,
        position_assessor: PositionAssessor,
        buy_risk_premium: Decimal = Decimal("0.01"),
        buy_premium_multiplier: Decimal = Decimal("1.0"),
        initial_prior: Decimal = Decimal("0.15"),
    ):
        self.valuer = valuer
        self.position_assessor = position_assessor
        self.buy_risk_premium = buy_risk_premium
        self.buy_premium_multiplier = buy_premium_multiplier
        self.initial_prior = initial_prior

    def should_sell(
        self,
        ticket: Ticket,
        dealer_bid: Decimal,
        current_day: int,
        trader_cash: Decimal,
        trader_shortfall: Decimal,
        trader_asset_value: Decimal,
    ) -> bool:
        """
        Decide whether trader should sell ticket to dealer at offered price.

        Decision rule:
        Accept if: dealer_offer >= expected_value + threshold

        Where threshold is adjusted for liquidity urgency.

        Args:
            ticket: Ticket being considered for sale
            dealer_bid: Dealer's bid price (unit price, per face=1)
            current_day: Current simulation day
            trader_cash: Trader's current cash
            trader_shortfall: Trader's immediate shortfall
            trader_asset_value: Total expected value of trader's receivables

        Returns:
            True if trader should accept the sale, False to reject
        """
        # Expected value if hold
        ev_hold = self.valuer.expected_value(ticket, current_day)

        # Dealer's offer (scaled to ticket face value)
        dealer_offer = dealer_bid * ticket.face

        # Compute effective threshold (urgency-adjusted)
        threshold = self.position_assessor.compute_effective_threshold(
            cash=trader_cash, shortfall=trader_shortfall, asset_value=trader_asset_value
        )
        threshold_absolute = threshold * ticket.face

        # Accept if dealer offer meets or exceeds expected value + threshold
        should_accept = dealer_offer >= (ev_hold + threshold_absolute)

        return should_accept

    def should_buy(
        self,
        ticket: Ticket,
        dealer_ask: Decimal,
        current_day: int,
        trader_cash: Decimal,
        trader_shortfall: Decimal,
        trader_asset_value: Decimal,
    ) -> bool:
        """
        Decide whether trader should buy ticket from dealer at offered price.

        Decision rule:
        Accept if: expected_value >= dealer_cost + threshold

        The threshold combines a base premium with a liquidity-adjusted component:
        traders with more of their wealth in illiquid receivables demand a higher
        premium to deploy scarce settlement cash.  The adjustment scales with the
        issuer's estimated default probability, creating a natural feedback loop
        (more defaults -> higher threshold -> fewer buys -> less liquidity drain).

        Args:
            ticket: Ticket being considered for purchase
            dealer_ask: Dealer's ask price (unit price)
            current_day: Current simulation day
            trader_cash: Trader's current cash
            trader_shortfall: Trader's shortfall (positive means needs cash)
            trader_asset_value: Total expected value of trader's receivables

        Returns:
            True if trader should accept the purchase, False to reject
        """
        # Expected value if buy
        ev_hold = self.valuer.expected_value(ticket, current_day)

        # Dealer's cost
        dealer_cost = dealer_ask * ticket.face

        # Base buy threshold from profile
        buy_threshold = self.buy_risk_premium

        # Liquidity-adjusted threshold: deploying settlement cash into illiquid
        # receivables carries an opportunity cost proportional to the issuer's
        # default risk and the trader's own cash scarcity.
        #
        # The premium uses a blended default estimate (empirical + prior) / 2
        # to prevent the threshold from dropping too fast when early settlements
        # succeed.  This reflects model uncertainty: traders stay cautious
        # until they have extensive evidence of system health.
        #
        # The minimum factor of 0.75 ensures even cash-rich traders demand at
        # least 75% of blended-P as a liquidity premium.  Cash-scarce traders
        # (low cash_ratio) demand more.
        total_position = trader_cash + trader_asset_value
        if total_position > 0 and trader_cash > 0:
            cash_ratio = trader_cash / total_position
            p_empirical = self.valuer.belief_source.estimate_default_prob(
                ticket.issuer_id, current_day
            )
            p_blended = (p_empirical + self.initial_prior) / 2
            liquidity_factor = max(Decimal("0.75"), Decimal(1) - cash_ratio)
            buy_threshold += p_blended * liquidity_factor
        elif total_position <= 0:
            return False  # Insolvent — do not buy

        threshold_absolute = buy_threshold * ticket.face

        # Accept if expected value exceeds cost by at least threshold
        should_accept = ev_hold >= (dealer_cost + threshold_absolute)

        return should_accept


# ---------------------------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------------------------


class RiskAssessor:
    """
    Risk assessment module for traders.

    Convenience wrapper that composes BeliefTracker, EVValuer,
    PositionAssessor, and TradeGate into the familiar single-class API.
    All public methods delegate to the appropriate component.
    """

    def __init__(self, params: RiskAssessmentParams, instrument_valuer: InstrumentValuer | None = None):
        """
        Initialize risk assessor.

        Args:
            params: Risk assessment parameters
            instrument_valuer: Optional InstrumentValuer to delegate
                expected_value / expected_value_detail calls to.
                When None (default), built-in logic is used.
        """
        self.params = params
        self.instrument_valuer = instrument_valuer

        # Compose pipeline stages
        self.belief_tracker = BeliefTracker(
            lookback_window=params.lookback_window,
            smoothing_alpha=params.smoothing_alpha,
            default_observability=params.default_observability,
            initial_prior=params.initial_prior,
            use_issuer_specific=params.use_issuer_specific,
        )
        self.valuer = EVValuer(self.belief_tracker, instrument_valuer)
        self.position_assessor = PositionAssessor(
            base_risk_premium=params.base_risk_premium,
            urgency_sensitivity=params.urgency_sensitivity,
        )
        self.trade_gate = TradeGate(
            valuer=self.valuer,
            position_assessor=self.position_assessor,
            buy_risk_premium=params.buy_risk_premium,
            buy_premium_multiplier=params.buy_premium_multiplier,
            initial_prior=params.initial_prior,
        )

    # -- Backward-compatible attribute access --------------------------------
    # ``payment_history`` and ``issuer_history`` are accessed directly by
    # callers (tests, diagnostics).  We expose them as properties that
    # delegate to the underlying BeliefTracker so mutations are shared.

    @property
    def payment_history(self) -> list[tuple[int, AgentId, bool]]:
        return self.belief_tracker.payment_history

    @payment_history.setter
    def payment_history(self, value: list[tuple[int, AgentId, bool]]) -> None:
        self.belief_tracker.payment_history = value

    @property
    def issuer_history(self) -> dict[AgentId, list[tuple[int, bool]]]:
        return self.belief_tracker.issuer_history

    @issuer_history.setter
    def issuer_history(self, value: dict[AgentId, list[tuple[int, bool]]]) -> None:
        self.belief_tracker.issuer_history = value

    # -- Delegated methods ---------------------------------------------------

    def update_history(self, day: int, issuer_id: AgentId, defaulted: bool) -> None:
        """
        Record a payment outcome (success or default).

        Should be called at the end of each day after settlements are processed.

        Args:
            day: Simulation day
            issuer_id: Agent who was supposed to make payment
            defaulted: True if payment failed, False if succeeded
        """
        self.belief_tracker.update_history(day, issuer_id, defaulted)

    def estimate_default_prob(self, issuer_id: AgentId, current_day: int) -> Decimal:
        """
        Estimate probability that issuer will default on obligations.

        Uses recent payment history within the lookback window.
        Applies Laplace smoothing to handle small samples.

        Args:
            issuer_id: Agent whose default probability to estimate
            current_day: Current simulation day

        Returns:
            Estimated default probability in [0, 1]
        """
        return self.belief_tracker.estimate_default_prob(issuer_id, current_day)

    def estimate_default_prob_detail(
        self,
        issuer_id: AgentId,
        current_day: int,
        estimator_id: str = "system",
    ) -> "Estimate":
        """Return an Estimate wrapping estimate_default_prob() with provenance."""
        return self.belief_tracker.estimate_default_prob_detail(
            issuer_id, current_day, estimator_id
        )

    def expected_value(self, ticket: Ticket, current_day: int) -> Decimal:
        """
        Compute expected value of holding a ticket to maturity.

        EV = (1 - P(default)) * face_value

        When an ``instrument_valuer`` is injected, delegates to it instead
        of using the built-in formula.

        Args:
            ticket: Ticket (receivable) to value
            current_day: Current simulation day

        Returns:
            Expected payoff from holding (in same units as face value)
        """
        return self.valuer.expected_value(ticket, current_day)

    def expected_value_detail(
        self,
        ticket: Ticket,
        current_day: int,
        estimator_id: str = "system",
    ) -> "Estimate":
        """Return an Estimate wrapping expected_value() with provenance.

        When an ``instrument_valuer`` is injected, delegates to its
        ``value()`` method instead of using the built-in formula.
        """
        return self.valuer.expected_value_detail(ticket, current_day, estimator_id)

    def compute_effective_threshold(
        self, cash: Decimal, shortfall: Decimal, asset_value: Decimal
    ) -> Decimal:
        """
        Compute effective risk premium threshold based on liquidity urgency.

        When trader has severe liquidity needs, threshold decreases
        (willing to accept worse prices).

        threshold_eff = threshold_base - urgency_sensitivity * (shortfall / wealth)

        Args:
            cash: Current cash holdings
            shortfall: Immediate payment shortfall (positive if needs cash)
            asset_value: Total value of receivables held

        Returns:
            Effective risk premium threshold (can be negative if desperate)
        """
        return self.position_assessor.compute_effective_threshold(cash, shortfall, asset_value)

    def should_sell(
        self,
        ticket: Ticket,
        dealer_bid: Decimal,
        current_day: int,
        trader_cash: Decimal,
        trader_shortfall: Decimal,
        trader_asset_value: Decimal,
    ) -> bool:
        """
        Decide whether trader should sell ticket to dealer at offered price.

        Decision rule:
        Accept if: dealer_offer >= expected_value + threshold

        Where threshold is adjusted for liquidity urgency.

        Args:
            ticket: Ticket being considered for sale
            dealer_bid: Dealer's bid price (unit price, per face=1)
            current_day: Current simulation day
            trader_cash: Trader's current cash
            trader_shortfall: Trader's immediate shortfall
            trader_asset_value: Total expected value of trader's receivables

        Returns:
            True if trader should accept the sale, False to reject
        """
        return self.trade_gate.should_sell(
            ticket, dealer_bid, current_day, trader_cash, trader_shortfall, trader_asset_value
        )

    def should_buy(
        self,
        ticket: Ticket,
        dealer_ask: Decimal,
        current_day: int,
        trader_cash: Decimal,
        trader_shortfall: Decimal,
        trader_asset_value: Decimal,
    ) -> bool:
        """
        Decide whether trader should buy ticket from dealer at offered price.

        Decision rule:
        Accept if: expected_value >= dealer_cost + threshold

        The threshold combines a base premium with a liquidity-adjusted component:
        traders with more of their wealth in illiquid receivables demand a higher
        premium to deploy scarce settlement cash.  The adjustment scales with the
        issuer's estimated default probability, creating a natural feedback loop
        (more defaults -> higher threshold -> fewer buys -> less liquidity drain).

        Args:
            ticket: Ticket being considered for purchase
            dealer_ask: Dealer's ask price (unit price)
            current_day: Current simulation day
            trader_cash: Trader's current cash
            trader_shortfall: Trader's shortfall (positive means needs cash)
            trader_asset_value: Total expected value of trader's receivables

        Returns:
            True if trader should accept the purchase, False to reject
        """
        return self.trade_gate.should_buy(
            ticket, dealer_ask, current_day, trader_cash, trader_shortfall, trader_asset_value
        )

    def get_diagnostics(self, current_day: int) -> dict[str, Any]:
        """
        Get diagnostic information about risk assessor state.

        Useful for debugging and analysis.

        Args:
            current_day: Current simulation day

        Returns:
            Dictionary with diagnostic information
        """
        window_start = current_day - self.params.lookback_window

        # System-wide statistics
        recent_payments = [
            (d, issuer_id, defaulted)
            for d, issuer_id, defaulted in self.payment_history
            if d >= window_start
        ]

        total_payments = len(recent_payments)
        total_defaults = sum(1 for _, _, defaulted in recent_payments if defaulted)

        system_default_rate = (
            Decimal(total_defaults) / Decimal(total_payments) if total_payments > 0 else Decimal(0)
        )

        return {
            "total_payment_history_size": len(self.payment_history),
            "recent_payments_count": total_payments,
            "recent_defaults_count": total_defaults,
            "system_default_rate": float(system_default_rate),
            "lookback_window": self.params.lookback_window,
            "base_risk_premium": float(self.params.base_risk_premium),
            "issuer_specific_enabled": self.params.use_issuer_specific,
            "issuers_tracked": len(self.issuer_history) if self.params.use_issuer_specific else 0,
        }
