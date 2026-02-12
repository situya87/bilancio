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
        assert self.cb_lending_rate >= self.reserve_remuneration_rate, \
            "Ceiling rate must be >= floor rate"
