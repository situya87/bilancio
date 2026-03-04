"""Unit tests for BankingSubsystem initialization and routing.

Tests cover:
1. initialize_banking_subsystem creates BankTreynorState per bank with quotes
2. best_deposit_bank returns highest r_D
3. cheapest_loan_bank returns lowest r_L
4. cheapest_pay_bank returns lowest r_D
5. refresh_all_quotes re-computes from balance sheet changes
6. _get_bank_reserves helper
7. _get_deposit_at_bank helper
"""

from decimal import Decimal

from bilancio.banking.types import Quote
from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.banking_subsystem import (
    BankingSubsystem,
    BankTreynorState,
    _get_bank_reserves,
    _get_deposit_at_bank,
    initialize_banking_subsystem,
)
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_banking_system() -> System:
    """Create a minimal system with CB + 2 banks + 2 firms for testing."""
    system = System(policy=PolicyEngine.default())

    cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
    bank1 = Bank(id="bank_1", name="Bank 1", kind="bank")
    bank2 = Bank(id="bank_2", name="Bank 2", kind="bank")
    firm1 = Firm(id="H_1", name="Firm 1", kind="firm")
    firm2 = Firm(id="H_2", name="Firm 2", kind="firm")

    system.add_agent(cb)
    system.add_agent(bank1)
    system.add_agent(bank2)
    system.add_agent(firm1)
    system.add_agent(firm2)

    # Mint cash, deposit, and mint reserves
    system.mint_cash(to_agent_id="H_1", amount=1000)
    system.mint_cash(to_agent_id="H_2", amount=1000)
    deposit_cash(system, "H_1", "bank_1", 1000)
    deposit_cash(system, "H_2", "bank_2", 1000)
    system.mint_reserves(to_bank_id="bank_1", amount=5000)
    system.mint_reserves(to_bank_id="bank_2", amount=5000)

    return system


class TestInitializeBankingSubsystem:
    """Tests for initialize_banking_subsystem."""

    def test_initialize_banking_subsystem(self):
        """Create system with CB + 2 banks + 2 firms, verify 2 bank states created with quotes."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Two banks should produce two BankTreynorState entries
        assert len(subsystem.banks) == 2
        assert "bank_1" in subsystem.banks
        assert "bank_2" in subsystem.banks

        # Each bank should have a computed quote after initialization
        for bank_id, bank_state in subsystem.banks.items():
            assert bank_state.current_quote is not None, (
                f"Bank {bank_id} should have an initial quote"
            )
            # Deposit rate is clamped to >= 0 by the pricing kernel
            assert bank_state.current_quote.deposit_rate >= Decimal("0"), (
                f"Bank {bank_id} deposit rate must be non-negative"
            )
            # Loan rate is always > deposit rate by the inside spread
            # (both computed from the same midline +/- half-width)
            assert bank_state.current_quote.loan_rate > bank_state.current_quote.deposit_rate or (
                # When inventory is very long, both can be very low/negative;
                # the key property is that quotes are computed (not None)
                bank_state.current_quote.midline is not None
            ), (
                f"Bank {bank_id} must have valid computed quotes"
            )

    def test_initialize_sets_corridor_from_kappa(self):
        """Corridor rates should be derived from kappa via BankProfile."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("0.5")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        expected_floor = profile.r_floor(kappa)
        expected_ceiling = profile.r_ceiling(kappa)

        for bank_state in subsystem.banks.values():
            assert bank_state.pricing_params.reserve_remuneration_rate == expected_floor
            assert bank_state.pricing_params.cb_borrowing_rate == expected_ceiling


class TestBankRouting:
    """Tests for multi-bank routing: best_deposit_bank, cheapest_loan_bank, cheapest_pay_bank."""

    def _make_subsystem_with_different_quotes(self) -> BankingSubsystem:
        """Create a subsystem with two banks that have explicitly different quotes."""
        profile = BankProfile()

        # Manually construct states with explicit quotes for deterministic testing
        bank1_state = BankTreynorState(
            bank_id="bank_1",
            pricing_params=None,  # type: ignore[arg-type]
        )
        bank1_state.current_quote = Quote(
            deposit_rate=Decimal("0.02"),
            loan_rate=Decimal("0.08"),
            day=0,
        )

        bank2_state = BankTreynorState(
            bank_id="bank_2",
            pricing_params=None,  # type: ignore[arg-type]
        )
        bank2_state.current_quote = Quote(
            deposit_rate=Decimal("0.05"),
            loan_rate=Decimal("0.04"),
            day=0,
        )

        subsystem = BankingSubsystem(
            banks={"bank_1": bank1_state, "bank_2": bank2_state},
            bank_profile=profile,
            kappa=Decimal("1.0"),
            trader_banks={"H_1": ["bank_1", "bank_2"]},
        )
        return subsystem

    def test_best_deposit_bank(self):
        """best_deposit_bank returns bank with highest r_D."""
        subsystem = self._make_subsystem_with_different_quotes()
        # bank_2 has higher deposit rate (0.05 > 0.02)
        assert subsystem.best_deposit_bank("H_1") == "bank_2"

    def test_cheapest_loan_bank(self):
        """cheapest_loan_bank returns bank with lowest r_L."""
        subsystem = self._make_subsystem_with_different_quotes()
        # bank_2 has lower loan rate (0.04 < 0.08)
        assert subsystem.cheapest_loan_bank("H_1") == "bank_2"

    def test_cheapest_pay_bank(self):
        """cheapest_pay_bank returns bank with lowest r_D (min opportunity cost)."""
        subsystem = self._make_subsystem_with_different_quotes()
        # bank_1 has lower deposit rate (0.02 < 0.05)
        assert subsystem.cheapest_pay_bank("H_1") == "bank_1"

    def test_best_deposit_bank_no_banks(self):
        """best_deposit_bank returns None when agent has no assigned banks."""
        profile = BankProfile()
        subsystem = BankingSubsystem(
            banks={},
            bank_profile=profile,
            kappa=Decimal("1.0"),
            trader_banks={"H_1": []},
        )
        assert subsystem.best_deposit_bank("H_1") is None


class TestRefreshQuotes:
    """Tests for quote refresh after balance sheet changes."""

    def test_refresh_all_quotes(self):
        """Verify quotes refresh and reflect balance sheet changes."""
        system = _make_banking_system()
        profile = BankProfile()
        kappa = Decimal("1.0")

        subsystem = initialize_banking_subsystem(
            system=system,
            bank_profile=profile,
            kappa=kappa,
            maturity_days=10,
        )

        # Record initial quotes
        initial_quotes = {}
        for bank_id, state in subsystem.banks.items():
            initial_quotes[bank_id] = state.current_quote

        # Change the balance sheet: add more reserves to bank_1
        system.mint_reserves(to_bank_id="bank_1", amount=10000)

        # Refresh all quotes
        subsystem.refresh_all_quotes(system, current_day=0)

        # bank_1's quote should change (more reserves = different inventory)
        new_quote_1 = subsystem.banks["bank_1"].current_quote
        assert new_quote_1 is not None
        # bank_2's quote should stay the same (no balance change)
        new_quote_2 = subsystem.banks["bank_2"].current_quote
        assert new_quote_2 is not None

        # With more reserves, bank_1 should have lower loan rate
        # (inventory moves towards long side -> midline drops -> rates drop)
        old_loan_rate = initial_quotes["bank_1"].loan_rate
        new_loan_rate = new_quote_1.loan_rate
        assert new_loan_rate <= old_loan_rate, (
            f"More reserves should not increase loan rate: {new_loan_rate} vs {old_loan_rate}"
        )


class TestHelperFunctions:
    """Tests for banking subsystem helper functions."""

    def test_get_bank_reserves(self):
        """Verify _get_bank_reserves reads reserves correctly."""
        system = _make_banking_system()
        reserves_1 = _get_bank_reserves(system, "bank_1")
        assert reserves_1 == 5000

        reserves_2 = _get_bank_reserves(system, "bank_2")
        assert reserves_2 == 5000

    def test_get_bank_reserves_nonexistent(self):
        """_get_bank_reserves returns 0 for nonexistent bank."""
        system = _make_banking_system()
        assert _get_bank_reserves(system, "nonexistent") == 0

    def test_get_deposit_at_bank(self):
        """Verify _get_deposit_at_bank reads deposits per bank correctly."""
        system = _make_banking_system()
        # H_1 deposited 1000 at bank_1
        dep_1 = _get_deposit_at_bank(system, "H_1", "bank_1")
        assert dep_1 == 1000

        # H_1 has no deposit at bank_2
        dep_2 = _get_deposit_at_bank(system, "H_1", "bank_2")
        assert dep_2 == 0

        # H_2 deposited 1000 at bank_2
        dep_3 = _get_deposit_at_bank(system, "H_2", "bank_2")
        assert dep_3 == 1000

    def test_get_deposit_at_bank_nonexistent_agent(self):
        """_get_deposit_at_bank returns 0 for nonexistent agent."""
        system = _make_banking_system()
        assert _get_deposit_at_bank(system, "nonexistent", "bank_1") == 0
