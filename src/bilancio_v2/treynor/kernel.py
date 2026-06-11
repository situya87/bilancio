"""The Treynor corridor kernel — shared mathematics of all three chapters.

A dealer lives on a bounded inventory grid and prices around a midline
anchored to outside quotes at the ghost points one step beyond each
boundary. The kernel computes, from six observable inputs, the complete
pricing schedule:

    grid [lower_bound, upper_bound], ticket S
    width L = upper − lower                    (Ch2: L_act = X⁺ + X⁻ = V)
    rungs N = L/S + 1
    layoff probability λ = S / (L + S) = 1/N   (uniform stationary dist.)
    inside spread I = λ × (C₊ + C₋)            (zero-profit identity)
    slope k = O / (L + 2S)                     (ghost-point anchoring)
    center μ = (lower + upper) / 2
    midline(x) = M ∓ k × (x − μ)               (price: −, rate: +)
    quotes(x) = midline(x) ± I/2

Chapter mapping:
- **Ch1 spot dealer**: one-sided price grid {0…X*}; C₊ = C₋ = O/2 so
  I = λO; μ = X*/2.
- **Ch2 speculative Treynor**: asymmetric two-sided rate grid
  [−X⁻, +X⁺] with X⁺ = V − x_spot·M, X⁻ = x_spot·M; O = wholesale ON
  spread; upward-sloping midline; same anchors M ± O/2 at the ghosts.
- **Ch3 bank**: settlement layer is a one-sided rate grid with the CB
  corridor; the speculative layer uses split tolls C₊ ≠ C₋ with
  I₂ = λ₂(C₊ + C₋) and an anchor width O₂ that is *conceptually separate*
  from the tolls (Ch3 Part 6, Step 5) — hence the kernel keeps
  ``anchor_width`` and the tolls as independent inputs.

The zero-profit identity is exact under the dealer's internal model (the
50/50 zero-information prior). ``simulate_walk`` verifies it empirically
and doubles as the diagnostic engine for measuring how far *realized*
flow departs from that prior inside the ring.

All arithmetic is Decimal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

ZERO = Decimal("0")
TWO = Decimal("2")


class Orientation(str, Enum):
    """Whether the corridor quotes prices or rates.

    Price midlines slope *down* in inventory (long dealer quotes low to
    sell); rate midlines slope *up* (cash-short dealer quotes high rates
    to attract funding) — Ch2 Part 6, Step 8.
    """

    PRICE = "price"
    RATE = "rate"


@dataclass(frozen=True)
class TreynorCorridor:
    """A fully-specified Treynor corridor: grid, anchors, tolls.

    Attributes:
        ticket: S — the grid step (face units for price space, notional
            for rate space). Must be positive.
        lower_bound: lower grid edge (0 for one-sided corridors; −X⁻ for
            two-sided).
        upper_bound: upper grid edge (X* or +X⁺).
        anchor_mid: M — the outside mid (price or rate) the midline
            passes through at the corridor center.
        anchor_width: O — the outside corridor width that pins the ghost
            points at M ± O/2 and hence the slope k.
        upper_toll: C₊ — the cost of a layoff at the upper boundary.
        lower_toll: C₋ — the cost of a layoff at the lower boundary.
            Chapter 1: C₊ = C₋ = O/2. Chapter 3 speculative: independent.
        orientation: price (downward midline) or rate (upward midline).
    """

    ticket: Decimal
    lower_bound: Decimal
    upper_bound: Decimal
    anchor_mid: Decimal
    anchor_width: Decimal
    upper_toll: Decimal
    lower_toll: Decimal
    orientation: Orientation

    def __post_init__(self) -> None:
        if self.ticket <= ZERO:
            raise ValueError(f"ticket must be positive, got {self.ticket}")
        if self.upper_bound < self.lower_bound:
            raise ValueError(f"upper_bound {self.upper_bound} < lower_bound {self.lower_bound}")
        width = self.upper_bound - self.lower_bound
        if width % self.ticket != ZERO:
            raise ValueError(
                f"corridor width {width} is not a multiple of ticket {self.ticket}; "
                "snap the bounds to the grid first (chapters: 'replace each boundary "
                "by the nearest feasible grid boundary')"
            )
        if self.anchor_width < ZERO:
            raise ValueError(f"anchor_width must be non-negative, got {self.anchor_width}")

    # -- constructors for the chapter configurations --------------------------

    @classmethod
    def one_sided_price(cls, *, ticket: Decimal, capacity: Decimal, mid: Decimal, outside_spread: Decimal) -> TreynorCorridor:
        """Chapter 1 spot corridor: grid {0…X*}, tolls O/2 per boundary."""
        half = outside_spread / TWO
        return cls(
            ticket=ticket,
            lower_bound=ZERO,
            upper_bound=capacity,
            anchor_mid=mid,
            anchor_width=outside_spread,
            upper_toll=half,
            lower_toll=half,
            orientation=Orientation.PRICE,
        )

    @classmethod
    def two_sided_rate(
        cls,
        *,
        ticket: Decimal,
        upper_capacity: Decimal,
        lower_capacity: Decimal,
        mid_rate: Decimal,
        outside_spread: Decimal,
        upper_toll: Decimal | None = None,
        lower_toll: Decimal | None = None,
    ) -> TreynorCorridor:
        """Ch2/Ch3 speculative corridor: grid [−X⁻, +X⁺], rate space.

        Tolls default to O/2 each (Ch2's symmetric overnight layoff);
        pass them explicitly for Ch3's split C₊/C₋.
        """
        half = outside_spread / TWO
        return cls(
            ticket=ticket,
            lower_bound=-lower_capacity,
            upper_bound=upper_capacity,
            anchor_mid=mid_rate,
            anchor_width=outside_spread,
            upper_toll=half if upper_toll is None else upper_toll,
            lower_toll=half if lower_toll is None else lower_toll,
            orientation=Orientation.RATE,
        )

    @classmethod
    def one_sided_rate(cls, *, ticket: Decimal, capacity: Decimal, mid_rate: Decimal, outside_spread: Decimal) -> TreynorCorridor:
        """Ch3 settlement corridor: reserves grid {0…X₁*} in rate space.

        The CB corridor (floor r̲, ceiling r̄) gives Ω₁ = r̄ − r̲ and
        M₁ = (r̄ + r̲)/2; layoffs cost Ω₁/2 at either boundary (borrow at
        the ceiling / warehouse at the floor).
        """
        half = outside_spread / TWO
        return cls(
            ticket=ticket,
            lower_bound=ZERO,
            upper_bound=capacity,
            anchor_mid=mid_rate,
            anchor_width=outside_spread,
            upper_toll=half,
            lower_toll=half,
            orientation=Orientation.RATE,
        )

    # -- derived quantities -----------------------------------------------------

    @property
    def width(self) -> Decimal:
        """L = upper − lower (Ch2: L_act = X⁺ + X⁻)."""
        return self.upper_bound - self.lower_bound

    @property
    def n_rungs(self) -> int:
        """N = L/S + 1."""
        return int(self.width / self.ticket) + 1

    @property
    def layoff_probability(self) -> Decimal:
        """λ = S / (L + S) = 1/N — uniform stationary distribution."""
        return self.ticket / (self.width + self.ticket)

    @property
    def inside_spread(self) -> Decimal:
        """I = λ × (C₊ + C₋) — the zero-profit identity.

        Chapter 1's I = λO is the special case C₊ + C₋ = O.
        """
        return self.layoff_probability * (self.upper_toll + self.lower_toll)

    @property
    def slope(self) -> Decimal:
        """k = O / (L + 2S) — ghost-point anchoring."""
        return self.anchor_width / (self.width + TWO * self.ticket)

    @property
    def center(self) -> Decimal:
        """μ — the corridor center (X*/2 one-sided; (X⁺ − X⁻)/2 two-sided)."""
        return (self.lower_bound + self.upper_bound) / TWO

    def _check_on_grid(self, x: Decimal) -> None:
        if x < self.lower_bound or x > self.upper_bound:
            raise ValueError(f"inventory {x} outside corridor [{self.lower_bound}, {self.upper_bound}]")

    def midline(self, x: Decimal) -> Decimal:
        """The reference price/rate at inventory x.

        Price space: p(x) = M − k(x − μ) (long inventory ⇒ lower price).
        Rate space:  r(x) = M + k(x − μ) (cash pressure ⇒ higher rate).
        """
        self._check_on_grid(x)
        tilt = self.slope * (x - self.center)
        if self.orientation is Orientation.PRICE:
            return self.anchor_mid - tilt
        return self.anchor_mid + tilt

    def ghost_value(self, *, upper: bool) -> Decimal:
        """Midline extended to the ghost point one step beyond a boundary.

        By construction this equals the outside quote: M − O/2 at the
        heavy boundary in price space (the VBT bid), M + O/2 at the
        upper ghost in rate space, etc.
        """
        x = self.upper_bound + self.ticket if upper else self.lower_bound - self.ticket
        tilt = self.slope * (x - self.center)
        if self.orientation is Orientation.PRICE:
            return self.anchor_mid - tilt
        return self.anchor_mid + tilt

    def ask(self, x: Decimal) -> Decimal:
        """Posted ask (price the dealer sells at / rate the dealer charges)."""
        return self.midline(x) + self.inside_spread / TWO

    def bid(self, x: Decimal) -> Decimal:
        """Posted bid (price the dealer buys at / rate the dealer pays)."""
        return self.midline(x) - self.inside_spread / TWO

    def at_upper_bound(self, x: Decimal) -> bool:
        return x >= self.upper_bound

    def at_lower_bound(self, x: Decimal) -> bool:
        return x <= self.lower_bound


def snap_down(value: Decimal, ticket: Decimal) -> Decimal:
    """Snap a capacity to the nearest feasible grid boundary below it."""
    return (value // ticket) * ticket


# ---------------------------------------------------------------------------
# Random-walk engine: verification of the zero-profit identity, and the
# diagnostic machinery for realized flow vs the 50/50 prior.
# ---------------------------------------------------------------------------


@dataclass
class WalkStats:
    """Accumulated outcomes of inventory steps against a corridor.

    Used two ways: under simulated 50/50 flow it verifies the chapter
    identities (occupancy uniform, layoff frequency → λ, P&L → 0); fed
    with *realized* ring flow it measures the departure of reality from
    the dealer's internal model.
    """

    trades: int = 0
    up_steps: int = 0
    down_steps: int = 0
    upper_layoffs: int = 0
    lower_layoffs: int = 0
    spread_income: Decimal = ZERO
    layoff_costs: Decimal = ZERO
    occupancy: dict[Decimal, int] = field(default_factory=dict)

    def record(
        self,
        corridor: TreynorCorridor,
        x: Decimal,
        *,
        toward_upper: bool,
    ) -> Decimal:
        """Record one customer arrival at inventory ``x``; return new inventory.

        Every trade collects I/2 in spread income (interior and boundary
        alike — Ch1 Part 3, Step 5). A push beyond a boundary triggers a
        layoff: inventory self-loops and the boundary toll accrues.
        """
        self.trades += 1
        self.occupancy[x] = self.occupancy.get(x, 0) + 1
        self.spread_income += corridor.inside_spread / TWO
        if toward_upper:
            self.up_steps += 1
            if corridor.at_upper_bound(x):
                self.upper_layoffs += 1
                self.layoff_costs += corridor.upper_toll
                return x
            return x + corridor.ticket
        self.down_steps += 1
        if corridor.at_lower_bound(x):
            self.lower_layoffs += 1
            self.layoff_costs += corridor.lower_toll
            return x
        return x - corridor.ticket

    @property
    def layoffs(self) -> int:
        return self.upper_layoffs + self.lower_layoffs

    @property
    def realized_layoff_frequency(self) -> Decimal:
        if self.trades == 0:
            return ZERO
        return Decimal(self.layoffs) / Decimal(self.trades)

    @property
    def realized_flow_ratio(self) -> Decimal:
        """Fraction of arrivals pushing toward the upper bound (prior: 1/2)."""
        if self.trades == 0:
            return Decimal("0.5")
        return Decimal(self.up_steps) / Decimal(self.trades)

    @property
    def net_pnl(self) -> Decimal:
        """Spread income minus layoff costs (zero-profit prediction: → 0)."""
        return self.spread_income - self.layoff_costs

    @property
    def pnl_per_trade(self) -> Decimal:
        if self.trades == 0:
            return ZERO
        return self.net_pnl / Decimal(self.trades)


def simulate_walk(
    corridor: TreynorCorridor,
    n_steps: int,
    rng: random.Random,
    *,
    start: Decimal | None = None,
) -> WalkStats:
    """Run the dealer's internal model: 50/50 flow against the corridor."""
    stats = WalkStats()
    x = corridor.center if start is None else start
    # Start on a grid rung.
    offset = (x - corridor.lower_bound) % corridor.ticket
    x = x - offset
    for _ in range(n_steps):
        x = stats.record(corridor, x, toward_upper=rng.random() < 0.5)
    return stats
