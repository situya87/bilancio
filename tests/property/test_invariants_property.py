"""Property-based tests for financial system invariants using Hypothesis."""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from bilancio.engines.system import System
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.firm import Firm
from bilancio.core.invariants import (
    assert_cb_cash_matches_outstanding,
    assert_no_negative_balances,
    assert_cb_reserves_match,
    assert_double_entry_numeric,
)
from bilancio.ops.banking import deposit_cash, withdraw_cash


def _fresh_system():
    """Create a fresh System with a central bank."""
    system = System()
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    system.bootstrap_cb(cb)
    return system


# Strategy for realistic financial amounts
amounts = st.integers(min_value=1, max_value=1_000_000)
small_amounts = st.integers(min_value=1, max_value=100_000)


class TestCashInvariantsProperty:
    """Property: minting and transferring cash preserves system invariants."""

    @given(amount=amounts)
    @settings(max_examples=50)
    def test_mint_cash_preserves_double_entry(self, amount):
        """Minting cash always preserves double-entry balance."""
        system = _fresh_system()
        hh = Household(id="HH1", name="HH1", kind="household")
        system.add_agent(hh)

        system.mint_cash("HH1", amount)

        assert_cb_cash_matches_outstanding(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)

    @given(amount=amounts, transfer=small_amounts)
    @settings(max_examples=50)
    def test_cash_transfer_preserves_invariants(self, amount, transfer):
        """Transferring cash between agents preserves all invariants."""
        assume(transfer <= amount)
        system = _fresh_system()
        hh1 = Household(id="HH1", name="HH1", kind="household")
        hh2 = Household(id="HH2", name="HH2", kind="household")
        system.add_agent(hh1)
        system.add_agent(hh2)

        system.mint_cash("HH1", amount)
        system.transfer_cash("HH1", "HH2", transfer)

        assert_cb_cash_matches_outstanding(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)

    @given(amounts_list=st.lists(amounts, min_size=1, max_size=5))
    @settings(max_examples=30)
    def test_multiple_mints_preserve_invariants(self, amounts_list):
        """Multiple mint operations always preserve invariants."""
        system = _fresh_system()
        for i, amt in enumerate(amounts_list):
            hh = Household(id=f"HH{i}", name=f"HH{i}", kind="household")
            system.add_agent(hh)
            system.mint_cash(f"HH{i}", amt)

        assert_cb_cash_matches_outstanding(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)


class TestReserveInvariantsProperty:
    """Property: reserve operations preserve system invariants."""

    @given(amount=amounts)
    @settings(max_examples=50)
    def test_mint_reserves_preserves_invariants(self, amount):
        """Minting reserves always preserves invariants."""
        system = _fresh_system()
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        system.add_agent(bank)

        system.mint_reserves("B1", amount)

        assert_cb_reserves_match(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)

    @given(amount=amounts, transfer=small_amounts)
    @settings(max_examples=50)
    def test_reserve_transfer_preserves_invariants(self, amount, transfer):
        """Transferring reserves between banks preserves invariants."""
        assume(transfer <= amount)
        system = _fresh_system()
        b1 = Bank(id="B1", name="Bank 1", kind="bank")
        b2 = Bank(id="B2", name="Bank 2", kind="bank")
        system.add_agent(b1)
        system.add_agent(b2)

        system.mint_reserves("B1", amount)
        system.transfer_reserves("B1", "B2", transfer)

        assert_cb_reserves_match(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)


class TestBankingInvariantsProperty:
    """Property: deposit/withdrawal operations preserve invariants."""

    @given(cash_amount=amounts, deposit_amount=small_amounts)
    @settings(max_examples=50)
    def test_deposit_preserves_invariants(self, cash_amount, deposit_amount):
        """Depositing cash preserves all system invariants."""
        assume(deposit_amount <= cash_amount)
        system = _fresh_system()
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        hh = Household(id="HH1", name="HH1", kind="household")
        system.add_agent(bank)
        system.add_agent(hh)

        system.mint_reserves("B1", 1_000_000)
        system.mint_cash("HH1", cash_amount)
        deposit_cash(system, "HH1", "B1", deposit_amount)

        assert_cb_cash_matches_outstanding(system)
        assert_cb_reserves_match(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)

    @given(cash_amount=amounts, deposit_amount=small_amounts, withdraw_amount=small_amounts)
    @settings(max_examples=50)
    def test_deposit_then_withdraw_preserves_invariants(self, cash_amount, deposit_amount, withdraw_amount):
        """Deposit then withdraw preserves all invariants."""
        assume(deposit_amount <= cash_amount)
        assume(withdraw_amount <= deposit_amount)
        system = _fresh_system()
        bank = Bank(id="B1", name="Bank 1", kind="bank")
        hh = Household(id="HH1", name="HH1", kind="household")
        system.add_agent(bank)
        system.add_agent(hh)

        system.mint_reserves("B1", 1_000_000)
        system.mint_cash("HH1", cash_amount)
        deposit_cash(system, "HH1", "B1", deposit_amount)
        withdraw_cash(system, "HH1", "B1", withdraw_amount)

        assert_cb_cash_matches_outstanding(system)
        assert_cb_reserves_match(system)
        assert_double_entry_numeric(system)
        assert_no_negative_balances(system)
