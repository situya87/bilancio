"""
Pricing kernel for Banks-as-Dealers.

Implements the Treynor-style dealer pricing from Section 6 of the specification.

The bank earns reserves by taking liquidity risk. Over horizon H=10 days,
liquidity risk is the chance that reserve outflows exceed inflows, forcing
end-of-day CB borrowing at cost i_B.

The bank sets rates only (r_D, r_L) and updates them ticket-by-ticket.
The corridor and backstop discipline prices and guarantee settlement.
"""

from dataclasses import dataclass
from decimal import Decimal

from bilancio.banking.types import Quote


@dataclass
class PricingParams:
    """
    Parameters for the pricing kernel.

    All rates are 2-day effective rates.
    """

    # === Central Bank Corridor ===
    # Floor: 2-day reserve remuneration rate
    reserve_remuneration_rate: Decimal  # i_R^(2) = r_t (lower pin)

    # Ceiling: 2-day CB borrowing rate
    cb_borrowing_rate: Decimal  # i_B = r̄_t (upper pin)

    # === Bank Internal Parameters ===
    # Target reserves at t+2 (midpoint of internal band)
    reserve_target: int  # R^tar

    # Half-width of internal reserve band around R^tar
    symmetric_capacity: int  # X*

    # Standard ticket size on funding plane
    ticket_size: int  # S_fund

    # System-level minimum reserves
    reserve_floor: int  # R_min

    # === Risk Loadings ===
    # Sensitivity of midline to cash-tightness L*
    alpha: Decimal = Decimal("0.005")

    # Sensitivity of midline to risk index ρ
    gamma: Decimal = Decimal("0.002")

    # =========================================================================
    # Derived quantities
    # =========================================================================

    @property
    def outside_width(self) -> Decimal:
        """
        Ω^(2) = r̄_t - r_t

        Outside spread between CB ceiling and floor on 2-day funding plane.
        """
        return self.cb_borrowing_rate - self.reserve_remuneration_rate

    @property
    def layoff_probability(self) -> Decimal:
        """
        λ = S_fund / (2X* + S_fund)

        Steady-state probability that a marginal unit of 2-day liquidity
        is laid off to CB (rather than warehoused) for symmetric band limits.
        """
        return Decimal(self.ticket_size) / Decimal(2 * self.symmetric_capacity + self.ticket_size)

    @property
    def inside_width(self) -> Decimal:
        """
        I^(2) = λ × Ω^(2)

        Symmetric inside spread on 2-day funding plane.
        This is the Treynor identity: inside spread = expected layoff cost.
        """
        return self.layoff_probability * self.outside_width

    @property
    def outside_mid(self) -> Decimal:
        """
        M^(2) = (r_t + r̄_t) / 2

        Midpoint between CB floor and ceiling.
        """
        return (self.reserve_remuneration_rate + self.cb_borrowing_rate) / 2


def compute_midline(inventory: int, params: PricingParams) -> Decimal:
    """
    Compute the symmetric linear midline on the 2-day funding plane.

    m^(2)(x) = M^(2) - [Ω^(2) / 2(X* + S_fund)] × x

    The midline is pinned to:
    - m^(2)(X* + S) = r_t (floor) at the long limit
    - m^(2)(-X* - S) = r̄_t (ceiling) at the short limit

    Args:
        inventory: x = R_{t+2} - R^tar (current inventory position)
        params: Pricing parameters

    Returns:
        The midline rate at this inventory position
    """
    # Slope: how much the midline moves per unit of inventory
    slope = params.outside_width / Decimal(2 * (params.symmetric_capacity + params.ticket_size))

    return params.outside_mid - slope * inventory


def compute_tilted_midline(
    inventory: int,
    cash_tightness: Decimal,
    risk_index: Decimal,
    params: PricingParams,
) -> Decimal:
    """
    Compute the bank's risk-tilted midline.

    m^(2)_bank(x) = m^(2)(x) + α×L* + γ×ρ

    The risk tilts push the midline up when:
    - L* is high (projected reserves close to floor)
    - ρ is high (shortfall probability or book-gap pressure)

    Args:
        inventory: x = R_{t+2} - R^tar
        cash_tightness: L* (10-day shortfall metric)
        risk_index: ρ (shortfall risk + book-gap index)
        params: Pricing parameters

    Returns:
        The tilted midline rate
    """
    base_midline = compute_midline(inventory, params)

    # Risk tilts
    tilt = params.alpha * cash_tightness + params.gamma * risk_index

    return base_midline + tilt


def compute_quotes(
    inventory: int,
    cash_tightness: Decimal,
    risk_index: Decimal,
    params: PricingParams,
    day: int,
    ticket_number: int = 0,
) -> Quote:
    """
    Compute deposit rate (bid) and loan rate (ask).

    Inside quotes:
        r^(2)_bid = m^(2)_bank - I^(2)/2
        r^(2)_ask = m^(2)_bank + I^(2)/2

    Posted rates (with ceiling discipline):
        r_D = min(r^(2)_bid, r̄_t)  # deposit rate
        r_L = r^(2)_ask             # loan rate

    Args:
        inventory: x = R_{t+2} - R^tar
        cash_tightness: L* (10-day shortfall metric)
        risk_index: ρ (shortfall risk + book-gap index)
        params: Pricing parameters
        day: Current day
        ticket_number: Which ticket these quotes are for

    Returns:
        Quote with deposit and loan rates
    """
    # Tilted midline
    midline = compute_tilted_midline(inventory, cash_tightness, risk_index, params)

    # Inside quotes
    half_width = params.inside_width / 2
    r_bid = midline - half_width  # Deposit rate (what bank pays)
    r_ask = midline + half_width  # Loan rate (what bank charges)

    # Ceiling discipline on deposits
    # Bank won't pay more than CB ceiling for deposits
    r_deposit = min(r_bid, params.cb_borrowing_rate)

    # Floor discipline on deposits (can't pay negative... usually)
    r_deposit = max(r_deposit, Decimal("0"))

    return Quote(
        deposit_rate=r_deposit,
        loan_rate=r_ask,
        day=day,
        ticket_number=ticket_number,
        inventory=inventory,
        cash_tightness=cash_tightness,
        risk_index=risk_index,
        midline=midline,
    )


def compute_inventory(
    projected_reserves_t2: int,
    reserve_target: int,
) -> int:
    """
    Compute the funding-plane inventory coordinate.

    x = R̂(t+2) - R^tar

    This is the 2-day inventory: how far are projected t+2 reserves
    from the target?

    Args:
        projected_reserves_t2: Projected reserves at t+2
        reserve_target: Target reserves

    Returns:
        Inventory coordinate x
    """
    return projected_reserves_t2 - reserve_target


# =============================================================================
# Utility functions for risk calculation
# =============================================================================


def simple_risk_index(
    cash_tightness: Decimal,
    loan_gap_ratio: Decimal | None = None,
    deposit_gap_ratio: Decimal | None = None,
    weight_reserve: Decimal = Decimal("1.0"),
    weight_loan: Decimal = Decimal("0.5"),
    weight_deposit: Decimal = Decimal("0.5"),
) -> Decimal:
    """
    Compute a simple risk index ρ.

    ρ = ω_R × ρ^R + ω_L × ρ^L + ω_D × ρ^D

    Where:
    - ρ^R: Reserve shortfall component (derived from L*)
    - ρ^L: Loan book gap (expected vs target)
    - ρ^D: Deposit book gap (expected vs target)

    For now, we use a simplified version:
    - ρ = cash_tightness (normalized)

    Later we can add loan/deposit book pressures.
    """
    risk = weight_reserve * cash_tightness

    if loan_gap_ratio is not None:
        risk += weight_loan * abs(loan_gap_ratio)

    if deposit_gap_ratio is not None:
        risk += weight_deposit * abs(deposit_gap_ratio)

    return risk
