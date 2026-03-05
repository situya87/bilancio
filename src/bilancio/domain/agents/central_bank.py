from dataclasses import dataclass, field
from decimal import Decimal

from bilancio.domain.agent import Agent, AgentKind


@dataclass
class CentralBank(Agent):
    """
    Central Bank agent.

    The CB issues two types of liabilities:
    1. Cash - bearer liability, can be held by anyone
    2. ReserveDeposit - deposit liability, held by banks/treasury

    The CB also has a lending facility for banks:
    - CBLoan - asset of CB, liability of borrowing bank

    The corridor is defined by two rates:
    - reserve_remuneration_rate (floor, i_R): what banks earn on reserves
    - cb_lending_rate (ceiling, i_B): what banks pay when borrowing

    Both are 2-day effective rates as per the Banks-as-Dealers specification.
    """

    # Override kind with default to satisfy dataclass field ordering
    # (fields with defaults must come after fields without defaults)
    kind: str = field(default=AgentKind.CENTRAL_BANK)

    # === Corridor Rates (2-day effective rates) ===

    # Floor rate: interest paid on reserves held at CB
    # Default 1% per 2-day period
    reserve_remuneration_rate: Decimal = field(default=Decimal("0.01"))

    # Ceiling rate: interest charged on CB loans to banks
    # Default 3% per 2-day period
    cb_lending_rate: Decimal = field(default=Decimal("0.03"))

    # === Configuration ===

    # Whether to issue cash (can be disabled for reserves-only simulations)
    issues_cash: bool = field(default=True)

    # Whether reserves accrue interest (can be disabled for simpler models)
    reserves_accrue_interest: bool = field(default=True)

    # Rate escalation: effective = base + slope × (outstanding / base_amount)
    rate_escalation_slope: Decimal = field(default=Decimal("0"))  # 0 = static (backward compat)
    escalation_base_amount: int = field(default=0)  # Q_total, set at init

    # CB lending cap: max outstanding as fraction of escalation_base_amount
    max_outstanding_ratio: Decimal = field(default=Decimal("0"))  # 0 = no cap (backward compat)

    # === κ-Informed Corridor (Plan 041) ===
    # P_0 = kappa_informed_prior(κ), set at scenario compilation
    kappa_prior: Decimal = field(default=Decimal("0"))  # 0 = disabled (backward compat)

    # Adaptive flags (Plan 050)
    adaptive_betas: bool = field(default=False)       # [PRE] beta_scale = 1/(1+5*expected)
    adaptive_early_warning: bool = field(default=False)  # [RUN] bank stress signal

    # Dynamic corridor adjustment: deviation from expectation
    # surprise_t = max(0, P_realized_t - P_0)
    # mid_t = mid_0 + beta_mid × surprise_t
    # width_t = width_0 + beta_width × surprise_t
    beta_mid: Decimal = field(default=Decimal("0.50"))
    beta_width: Decimal = field(default=Decimal("0.30"))

    @property
    def corridor_width(self) -> Decimal:
        """
        Ω^(2) = ceiling - floor.

        The outside spread between CB lending rate and reserve remuneration rate.
        Banks place their inside spread within this corridor.
        """
        return self.cb_lending_rate - self.reserve_remuneration_rate

    @property
    def corridor_mid(self) -> Decimal:
        """
        M^(2) = (floor + ceiling) / 2.

        Midpoint of the corridor.
        """
        return (self.reserve_remuneration_rate + self.cb_lending_rate) / 2

    def validate_corridor(self) -> None:
        """Validate corridor parameters."""
        assert self.reserve_remuneration_rate >= 0, "Floor rate must be non-negative"
        assert self.cb_lending_rate >= self.reserve_remuneration_rate, (
            "Ceiling rate must be >= floor rate"
        )

    def effective_lending_rate(self, outstanding: int) -> Decimal:
        """CB lending rate with escalation: r = base + slope × (outstanding / Q_total).

        When rate_escalation_slope == 0 or escalation_base_amount <= 0,
        returns the static cb_lending_rate (backward compatible).
        """
        if self.rate_escalation_slope == 0 or self.escalation_base_amount <= 0:
            return self.cb_lending_rate
        utilization = Decimal(outstanding) / Decimal(self.escalation_base_amount)
        return self.cb_lending_rate + self.rate_escalation_slope * utilization

    def can_lend(self, outstanding: int, amount: int) -> bool:
        """Check if CB can lend given its outstanding cap.

        When max_outstanding_ratio <= 0 or escalation_base_amount <= 0,
        always returns True (no cap, backward compatible).
        """
        if self.max_outstanding_ratio <= 0 or self.escalation_base_amount <= 0:
            return True
        cap = int(self.max_outstanding_ratio * self.escalation_base_amount)
        return outstanding + amount <= cap

    def compute_corridor(
        self, n_defaulted: int, n_total: int, *, bank_stress: Decimal = Decimal("0")
    ) -> tuple[Decimal, Decimal]:
        """Compute dynamic corridor based on deviation from expectation.

        When kappa_prior > 0, the corridor adjusts based on how much the
        realized default rate exceeds the initial expectation (P_0).

        surprise_t = max(0, P_realized_t - P_0)
        mid_t   = base_mid + beta_mid × surprise_t
        width_t = base_width + beta_width × surprise_t
        r_floor = mid_t - width_t / 2
        r_ceiling = mid_t + width_t / 2

        When kappa_prior == 0, returns the static corridor (backward compat).

        Args:
            n_defaulted: Number of agents that have defaulted.
            n_total: Total number of agents in the system.

        Returns:
            (r_floor, r_ceiling) tuple.
        """
        if self.kappa_prior <= 0:
            return self.reserve_remuneration_rate, self.cb_lending_rate

        # Base corridor mid and width from initial static rates
        base_mid = self.corridor_mid
        base_width = self.corridor_width

        # Deviation from expectation
        p_realized = Decimal(n_defaulted) / Decimal(max(n_total, 1))
        surprise = max(Decimal(0), p_realized - self.kappa_prior)

        # When adaptive_betas, scale betas inversely with expected default rate
        effective_beta_mid = self.beta_mid
        effective_beta_width = self.beta_width
        if self.adaptive_betas and self.kappa_prior > 0:
            beta_scale = Decimal(1) / (Decimal(1) + Decimal(5) * self.kappa_prior)
            effective_beta_mid = self.beta_mid * beta_scale
            effective_beta_width = self.beta_width * beta_scale

        # When adaptive_early_warning, blend bank stress into surprise
        combined = surprise + Decimal("0.3") * bank_stress if self.adaptive_early_warning else surprise

        # Adjust mid and width
        mid = base_mid + effective_beta_mid * combined
        width = base_width + effective_beta_width * combined

        r_floor = max(Decimal(0), mid - width / 2)
        r_ceiling = mid + width / 2
        return r_floor, r_ceiling
