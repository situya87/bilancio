"""
Reserve projection and cash-tightness calculation.

Implements the 10-day reserve path projection from Section 6.1.3 of the specification.

The Treasury desk builds a full 10-day reserve trajectory R̂(h) for h ∈ {t, ..., t+10}
after each ticket event, combining:
1. Current balance sheet
2. Scheduled central-bank and loan legs
3. Remaining expected withdrawals for today
"""

from dataclasses import dataclass
from decimal import Decimal

from bilancio.banking.state import BankDealerState, CentralBankParams


@dataclass
class ReserveProjection:
    """
    10-day reserve path projection.

    R̂(h) for h ∈ {t, t+1, ..., t+10}
    """

    base_day: int
    path: dict[int, int]  # day -> projected reserves

    # Diagnostic: what legs contributed
    legs_by_day: dict[int, list[str]]

    def at_horizon(self, h: int) -> int:
        """Get projected reserves at horizon h."""
        return self.path.get(h, self.path[max(self.path.keys())])

    @property
    def min_reserves(self) -> int:
        """Minimum reserves over the projection horizon."""
        return min(self.path.values())

    @property
    def min_day(self) -> int:
        """Day when minimum reserves occur."""
        min_val = self.min_reserves
        for day, val in self.path.items():
            if val == min_val:
                return day
        return self.base_day

    def __str__(self) -> str:
        lines = [f"Reserve Projection from day {self.base_day}:"]
        for day in sorted(self.path.keys()):
            legs = self.legs_by_day.get(day, [])
            leg_str = ", ".join(legs) if legs else "carry"
            lines.append(f"  Day {day}: {self.path[day]:>12,} ({leg_str})")
        lines.append(f"  Min: {self.min_reserves:,} on day {self.min_day}")
        return "\n".join(lines)


def project_reserves(
    state: BankDealerState,
    cb_params: CentralBankParams,
    horizon: int = 10,
) -> ReserveProjection:
    """
    Build a 10-day reserve projection starting from current state.

    R̂(t) = R^(k)_t - W^rem_t  (current reserves minus remaining expected withdrawals)

    R̂(t+s) = R̂(t+s-1) + ΔR^old_{t+s} + E[ΔR^new_{t+s} | r_D, r_L]

    Where:
    - ΔR^old_{t+s}: Scheduled reserve legs from pre-t cohorts
      (CB remuneration/repayment, loan repayments)
    - ΔR^new_{t+s}: Expected legs from today's rate-sensitive flows
      (often set to 0 for simplicity)

    Args:
        state: Current BankDealer state
        cb_params: Central bank parameters (for remuneration calculation)
        horizon: Number of days to project (default 10)

    Returns:
        ReserveProjection with path and diagnostic info
    """
    t = state.current_day
    path = {}
    legs_by_day = {}

    # Starting point: current reserves minus remaining expected withdrawals
    # R̂(t) = R_t - W^rem_t
    r_hat = state.reserves - state.withdrawals_remaining
    path[t] = r_hat
    legs_by_day[t] = [f"R={state.reserves}", f"-W^rem={state.withdrawals_remaining}"]

    # Get all scheduled legs for the projection horizon
    scheduled = state.get_scheduled_legs(t + 1, t + horizon)

    # Build a lookup: day -> list of (amount, description)
    legs_lookup: dict[int, list[tuple[int, str]]] = {}
    for leg in scheduled:
        if leg.day not in legs_lookup:
            legs_lookup[leg.day] = []
        legs_lookup[leg.day].append((leg.amount, f"{leg.leg_type}:{leg.amount}"))

    # CB remuneration at t+2 (if we're before day t+2)
    # R ↑ i_R^(2) × R_base,t
    # For simplicity, use current reserves as the base
    cb_rem_day = t + 2
    if cb_rem_day <= t + horizon:
        # Simple model: remuneration on current reserves
        rem_amount = int(cb_params.reserve_remuneration_rate * state.reserves)
        if rem_amount > 0:
            if cb_rem_day not in legs_lookup:
                legs_lookup[cb_rem_day] = []
            legs_lookup[cb_rem_day].append((rem_amount, f"CB_rem:{rem_amount}"))

    # Project forward
    for s in range(1, horizon + 1):
        day = t + s
        legs_for_day = legs_lookup.get(day, [])

        # Sum all legs for this day
        delta_r = sum(amount for amount, _ in legs_for_day)
        r_hat = r_hat + delta_r

        path[day] = r_hat
        legs_by_day[day] = [desc for _, desc in legs_for_day] if legs_for_day else []

    return ReserveProjection(
        base_day=t,
        path=path,
        legs_by_day=legs_by_day,
    )


def compute_cash_tightness(
    projection: ReserveProjection,
    reserve_floor: int,
) -> Decimal:
    """
    Compute the 10-day cash-tightness index L*.

    L* = max{0, R_min - min_{h ∈ {t,...,t+10}} R̂(h)}

    L* measures the worst shortfall of the projected reserve path
    below the floor. Zero means the path never touches the floor.

    Args:
        projection: 10-day reserve projection
        reserve_floor: R_min (system-level minimum)

    Returns:
        Cash-tightness L* as a Decimal (normalized by reserve_floor)
    """
    min_reserves = projection.min_reserves

    # Raw shortfall
    shortfall = max(0, reserve_floor - min_reserves)

    # Normalize by floor to get a ratio
    if reserve_floor > 0:
        return Decimal(shortfall) / Decimal(reserve_floor)
    else:
        return Decimal(shortfall)


def compute_long_side_headroom(
    projection: ReserveProjection,
    reserve_floor: int,
) -> int:
    """
    Compute long-side headroom at end of horizon.

    X^long = max{0, R̂(t+10) - R_min}

    Measures how much room is left on the long side at the end of the
    10-day horizon relative to R_min.

    Args:
        projection: 10-day reserve projection
        reserve_floor: R_min

    Returns:
        Long-side headroom
    """
    end_reserves = projection.at_horizon(projection.base_day + 10)
    return max(0, end_reserves - reserve_floor)


def effective_reserves_after_withdrawals(
    current_reserves: int,
    remaining_withdrawals: int,
) -> int:
    """
    Compute effective reserves after remaining expected withdrawals.

    R^eff_t = R^mid_t - W^rem_t

    This is the starting point for pricing calculations.
    """
    return current_reserves - remaining_withdrawals


# =============================================================================
# Intraday projection updates
# =============================================================================


def update_projection_after_ticket(
    old_projection: ReserveProjection,
    reserve_delta: int,
    withdrawal_delta: int,
    new_scheduled_legs: list[tuple[int, int, str]] | None = None,
) -> ReserveProjection:
    """
    Quickly update projection after a ticket without full rebuild.

    This is an optimization for intraday processing. After each ticket:
    - Reserve position may have changed
    - Remaining withdrawals may have changed
    - New scheduled legs may have been added (e.g., new loan)

    Args:
        old_projection: Previous projection
        reserve_delta: Change in reserves from ticket
        withdrawal_delta: Change in remaining expected withdrawals
        new_scheduled_legs: List of (day, amount, description) for new legs

    Returns:
        Updated projection
    """
    t = old_projection.base_day

    # Net effect on starting point: reserve_delta + withdrawal_delta
    # (if withdrawals realized, withdrawal_delta is negative, which increases R^eff)
    starting_delta = reserve_delta + withdrawal_delta

    # Update path
    new_path = {}
    new_legs = dict(old_projection.legs_by_day)

    for day, r_hat in old_projection.path.items():
        new_path[day] = r_hat + starting_delta

    # Add any new scheduled legs
    if new_scheduled_legs:
        for day, amount, desc in new_scheduled_legs:
            if day in new_path:
                # Add to all days >= this day
                for d in new_path:
                    if d >= day:
                        new_path[d] += amount
                # Record the leg
                if day not in new_legs:
                    new_legs[day] = []
                new_legs[day].append(desc)

    return ReserveProjection(
        base_day=t,
        path=new_path,
        legs_by_day=new_legs,
    )
