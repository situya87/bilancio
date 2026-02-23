"""Pre-simulation trade viability check.

Pure-math check that verifies both buy and sell trades are viable
for a given set of parameters, without running a full simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bilancio.dealer.priors import kappa_informed_prior


@dataclass
class ViabilityReport:
    """Result of a pre-simulation trade viability check."""

    sell_viable: bool
    buy_viable: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def all_viable(self) -> bool:
        return self.sell_viable and self.buy_viable


def check_trade_viability(
    kappa: Decimal,
    face_value: Decimal,
    n_agents: int,
    dealer_share: Decimal,
    vbt_share: Decimal,
    layoff_threshold: Decimal,
    buy_premium: Decimal,
    maturity_days: int,
    outside_mid_ratio: Decimal = Decimal("1.0"),
    base_spread: Decimal = Decimal("0.04"),
) -> ViabilityReport:
    """Check whether buy and sell trades are viable for given parameters.

    Uses the kernel math (midline/inside width formulas) to verify that:
    - Sell check: urgent seller (shortfall > 0) accepts dealer bid
    - Buy check: at inventory_ratio = layoff_threshold, surplus buyer
      finds ask low enough that EV/face > ask + buy_premium

    Args:
        kappa: System liquidity ratio L0/S1
        face_value: Ticket face value S
        n_agents: Number of agents in ring
        dealer_share: Fraction of claims held by dealer per bucket
        vbt_share: Fraction of claims held by VBT per bucket
        layoff_threshold: Inventory ratio below which VBT injects cash
        buy_premium: Minimum gain required for a buy (e.g. 0.01)
        maturity_days: Payment horizon
        outside_mid_ratio: Base M/S ratio before credit adjustment

    Returns:
        ViabilityReport with sell_viable, buy_viable, and diagnostics
    """
    diagnostics: dict[str, Any] = {}

    # Compute shared prior
    P_prior = kappa_informed_prior(kappa)
    diagnostics["P_prior"] = float(P_prior)

    # Credit-adjusted mid: M = outside_mid_ratio * (1 - P_default)
    M = outside_mid_ratio * (Decimal(1) - P_prior)
    diagnostics["M"] = float(M)

    S = face_value

    # Estimate total claims per bucket (rough: total debt / num_buckets)
    # Total debt ~ n_agents * face_value (each agent has one payable)
    total_claims_per_bucket = Decimal(n_agents) * face_value / Decimal(3)
    n_dealer_tickets = int(total_claims_per_bucket * dealer_share / face_value)
    n_vbt_tickets = int(total_claims_per_bucket * vbt_share / face_value)
    diagnostics["n_dealer_tickets"] = n_dealer_tickets
    diagnostics["n_vbt_tickets"] = n_vbt_tickets

    # Estimate dealer capital: cash = n_tickets * M * S
    dealer_cash = Decimal(n_dealer_tickets) * M * S
    dealer_x = S * n_dealer_tickets

    # Compute V, K_star, X_star
    V = M * dealer_x + dealer_cash
    if M * S > 0:
        K_star = int(V / (M * S))
    else:
        K_star = 0
    X_star = S * K_star

    diagnostics["K_star"] = K_star
    diagnostics["X_star"] = float(X_star)
    diagnostics["dealer_inventory_ratio"] = (
        float(Decimal(n_dealer_tickets) / max(1, K_star)) if K_star > 0 else 0.0
    )

    # VBT outside spread (per-bucket: short=0.04, mid=0.08, long=0.12)
    outside_spread = base_spread

    # --- Sell viability ---
    # Compute dealer bid at current inventory
    # lambda = S / (X_star + S)
    if X_star + S > 0:
        lambda_ = S / (X_star + S)
    else:
        lambda_ = Decimal(1)
    inside_width = lambda_ * outside_spread

    # Midline: p(x) = M - O/(X*+2S) * (x - X*/2)
    if X_star + 2 * S > 0:
        slope = outside_spread / (X_star + 2 * S)
        midline = M - slope * (dealer_x - X_star / 2)
    else:
        midline = M

    bid = midline - inside_width / 2
    # VBT bid B = M - O/2
    B = M - outside_spread / 2
    bid = max(B, bid)

    diagnostics["dealer_bid"] = float(bid)
    diagnostics["vbt_bid"] = float(B)

    # Sell is viable if bid > 0 (urgent sellers accept any positive price)
    sell_viable = bid > Decimal(0)
    diagnostics["sell_viable_reason"] = "bid > 0" if sell_viable else "bid <= 0"

    # --- Buy viability ---
    # At inventory_ratio = layoff_threshold, what is the ask?
    if K_star > 0 and layoff_threshold > 0:
        target_inventory = int(layoff_threshold * K_star)
        target_x = S * target_inventory

        # Midline at target inventory
        if X_star + 2 * S > 0:
            slope_buy = outside_spread / (X_star + 2 * S)
            midline_buy = M - slope_buy * (target_x - X_star / 2)
        else:
            midline_buy = M

        ask_at_threshold = midline_buy + inside_width / 2
        # VBT ask A = M + O/2
        A = M + outside_spread / 2
        ask = min(A, ask_at_threshold)

        diagnostics["ask_at_threshold"] = float(ask)
        diagnostics["vbt_ask"] = float(A)

        # EV for buyer: M (credit-adjusted mid)
        buyer_ev = M
        buyer_gain = buyer_ev - ask
        diagnostics["buyer_gain"] = float(buyer_gain)
        diagnostics["buy_premium"] = float(buy_premium)

        buy_viable = buyer_gain > buy_premium
        diagnostics["buy_viable_reason"] = (
            f"gain={float(buyer_gain):.4f} > premium={float(buy_premium):.4f}"
            if buy_viable
            else f"gain={float(buyer_gain):.4f} <= premium={float(buy_premium):.4f}"
        )
    else:
        buy_viable = False
        diagnostics["buy_viable_reason"] = "K_star=0 or layoff_threshold=0"

    return ViabilityReport(
        sell_viable=sell_viable,
        buy_viable=buy_viable,
        diagnostics=diagnostics,
    )
