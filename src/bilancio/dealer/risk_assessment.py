"""
Risk assessment module for traders in dealer subsystem.

This module enables traders to:
1. Estimate default probabilities for issuers
2. Compute expected values of holding receivables
3. Make rational buy/sell decisions based on price vs expected value
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List

from bilancio.core.ids import AgentId
from .models import Ticket


@dataclass
class RiskAssessmentParams:
    """
    Parameters for trader risk assessment.

    Attributes:
        lookback_window: Number of days to look back for default history
        smoothing_alpha: Laplace smoothing parameter (handles small samples)
        base_risk_premium: Minimum premium required to sell (fraction of face value)
        urgency_sensitivity: How much liquidity urgency reduces threshold
        use_issuer_specific: If True, track per-issuer rates; if False, use system-wide
        buy_premium_multiplier: Buyers require higher premium than sellers
    """

    lookback_window: int = 5
    smoothing_alpha: Decimal = Decimal("1.0")
    base_risk_premium: Decimal = Decimal("0.02")  # 2% premium
    urgency_sensitivity: Decimal = Decimal("0.10")  # 10% sensitivity
    use_issuer_specific: bool = False
    buy_premium_multiplier: Decimal = Decimal("1.0")  # Buyers use same premium as sellers


class RiskAssessor:
    """
    Risk assessment module for traders.

    Tracks payment outcomes (defaults vs successes) and uses this history
    to estimate default probabilities, compute expected values, and make
    rational trading decisions.
    """

    def __init__(self, params: RiskAssessmentParams):
        """
        Initialize risk assessor.

        Args:
            params: Risk assessment parameters
        """
        self.params = params

        # Track system-wide default history: (day, issuer_id, defaulted)
        self.payment_history: List[tuple[int, AgentId, bool]] = []

        # Track per-issuer history (if issuer-specific enabled)
        self.issuer_history: Dict[AgentId, List[tuple[int, bool]]] = {}

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
        if self.params.use_issuer_specific:
            if issuer_id not in self.issuer_history:
                self.issuer_history[issuer_id] = []
            self.issuer_history[issuer_id].append((day, defaulted))

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
        window_start = current_day - self.params.lookback_window

        if self.params.use_issuer_specific and issuer_id in self.issuer_history:
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
            # No recent data: assume uncertain default rate as prior
            # 25% reflects genuine uncertainty with zero observations,
            # allowing initial trades at market prices before data arrives
            return Decimal("0.25")

        # Count defaults and total payments
        defaults = sum(1 for _, defaulted in recent if defaulted)
        total = len(recent)

        # Laplace smoothing: (alpha + defaults) / (2*alpha + total)
        # This prevents extreme estimates from small samples
        alpha = self.params.smoothing_alpha
        p_default = (alpha + Decimal(defaults)) / (Decimal(2) * alpha + Decimal(total))

        return p_default

    def expected_value(self, ticket: Ticket, current_day: int) -> Decimal:
        """
        Compute expected value of holding a ticket to maturity.

        EV = (1 - P(default)) * face_value

        Args:
            ticket: Ticket (receivable) to value
            current_day: Current simulation day

        Returns:
            Expected payoff from holding (in same units as face value)
        """
        p_default = self.estimate_default_prob(ticket.issuer_id, current_day)
        ev = (Decimal(1) - p_default) * ticket.face
        return ev

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
            return self.params.base_risk_premium

        # Compute urgency ratio: shortfall as fraction of wealth
        urgency_ratio = shortfall / wealth

        # Reduce threshold based on urgency
        threshold_eff = (
            self.params.base_risk_premium
            - self.params.urgency_sensitivity * urgency_ratio
        )

        return threshold_eff

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
        ev_hold = self.expected_value(ticket, current_day)

        # Dealer's offer (scaled to ticket face value)
        dealer_offer = dealer_bid * ticket.face

        # Compute effective threshold (urgency-adjusted)
        threshold = self.compute_effective_threshold(
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

        Buying typically requires higher threshold than selling (bid-ask asymmetry).

        Args:
            ticket: Ticket being considered for purchase
            dealer_ask: Dealer's ask price (unit price)
            current_day: Current simulation day
            trader_cash: Trader's current cash
            trader_shortfall: Trader's shortfall (negative means surplus)
            trader_asset_value: Total expected value of trader's receivables

        Returns:
            True if trader should accept the purchase, False to reject
        """
        # Expected value if buy
        ev_hold = self.expected_value(ticket, current_day)

        # Dealer's cost
        dealer_cost = dealer_ask * ticket.face

        # For buying, use higher threshold (bid-ask asymmetry)
        buy_threshold = self.params.base_risk_premium * self.params.buy_premium_multiplier
        threshold_absolute = buy_threshold * ticket.face

        # Accept if expected value exceeds cost by at least threshold
        should_accept = ev_hold >= (dealer_cost + threshold_absolute)

        return should_accept

    def get_diagnostics(self, current_day: int) -> Dict[str, any]:
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
            Decimal(total_defaults) / Decimal(total_payments)
            if total_payments > 0
            else Decimal(0)
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
