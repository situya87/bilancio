"""Comprehensive tests for the balance analytics module."""

from decimal import Decimal

from bilancio.analysis.balances import agent_balance, system_trial_balance
from bilancio.domain.agents import Bank, CentralBank, Household
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


class TestBalanceAnalytics:
    """Test comprehensive balance analytics scenarios."""

    def test_agent_balance_simple_deposit(self):
        """Test agent balance with a simple cash deposit scenario.

        Creates System with CentralBank, Bank, and Household.
        Mints cash to household, then deposits at bank.
        Verifies household has deposit asset, no liabilities.
        Verifies bank has deposit liability and cash asset.
        Verifies trial balance totals match.
        """
        # Create system and agents
        system = System()
        cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
        bank = Bank(id="BK01", name="Test Bank", kind="bank")
        household = Household(id="HH01", name="Test Household", kind="household")

        system.bootstrap_cb(cb)
        system.add_agent(bank)
        system.add_agent(household)

        # Mint cash to household
        cash_amount = 1000
        system.mint_cash("HH01", cash_amount)

        # Deposit cash at bank
        deposit_amount = 600
        deposit_cash(system, "HH01", "BK01", deposit_amount)

        # Verify household balance
        hh_balance = agent_balance(system, "HH01")
        assert hh_balance.agent_id == "HH01"
        assert hh_balance.total_financial_assets == deposit_amount + (cash_amount - deposit_amount)
        assert hh_balance.total_financial_liabilities == 0
        assert hh_balance.net_financial == cash_amount  # Still has same total wealth
        assert hh_balance.assets_by_kind.get("bank_deposit", 0) == deposit_amount
        assert hh_balance.assets_by_kind.get("cash", 0) == cash_amount - deposit_amount
        assert hh_balance.liabilities_by_kind == {}

        # Verify bank balance
        bank_balance = agent_balance(system, "BK01")
        assert bank_balance.agent_id == "BK01"
        assert bank_balance.total_financial_assets == deposit_amount  # Cash from household
        assert bank_balance.total_financial_liabilities == deposit_amount  # Deposit liability
        assert bank_balance.net_financial == 0  # Bank's net position is zero
        assert bank_balance.assets_by_kind.get("cash", 0) == deposit_amount
        assert bank_balance.liabilities_by_kind.get("bank_deposit", 0) == deposit_amount

        # Verify trial balance
        trial = system_trial_balance(system)
        assert trial.total_financial_assets == trial.total_financial_liabilities
        expected_total = cash_amount + deposit_amount  # Cash + deposit instruments
        assert trial.total_financial_assets == expected_total
        assert trial.assets_by_kind.get("cash", 0) == cash_amount
        assert trial.assets_by_kind.get("bank_deposit", 0) == deposit_amount
        assert trial.liabilities_by_kind.get("cash", 0) == cash_amount
        assert trial.liabilities_by_kind.get("bank_deposit", 0) == deposit_amount

        # Ensure all invariants pass
        system.assert_invariants()

    def test_trial_balance_reserves_counters_match(self):
        """Test trial balance with reserves.

        Creates System with CentralBank and two Banks.
        Mints reserves to both banks.
        Verifies trial balance shows correct reserve amounts.
        Verifies assets equal liabilities system-wide.
        """
        # Create system and agents
        system = System()
        cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
        bank1 = Bank(id="BK01", name="Bank One", kind="bank")
        bank2 = Bank(id="BK02", name="Bank Two", kind="bank")

        system.bootstrap_cb(cb)
        system.add_agent(bank1)
        system.add_agent(bank2)

        # Mint reserves to both banks
        reserves1_amount = 5000
        reserves2_amount = 3000
        system.mint_reserves("BK01", reserves1_amount)
        system.mint_reserves("BK02", reserves2_amount)

        # Verify individual bank balances
        bank1_balance = agent_balance(system, "BK01")
        assert bank1_balance.total_financial_assets == reserves1_amount
        assert bank1_balance.total_financial_liabilities == 0
        assert bank1_balance.net_financial == reserves1_amount
        assert bank1_balance.assets_by_kind.get("reserve_deposit", 0) == reserves1_amount
        assert bank1_balance.liabilities_by_kind == {}

        bank2_balance = agent_balance(system, "BK02")
        assert bank2_balance.total_financial_assets == reserves2_amount
        assert bank2_balance.total_financial_liabilities == 0
        assert bank2_balance.net_financial == reserves2_amount
        assert bank2_balance.assets_by_kind.get("reserve_deposit", 0) == reserves2_amount
        assert bank2_balance.liabilities_by_kind == {}

        # Verify central bank balance
        cb_balance = agent_balance(system, "CB01")
        assert cb_balance.total_financial_assets == 0
        assert cb_balance.total_financial_liabilities == reserves1_amount + reserves2_amount
        assert cb_balance.net_financial == -(reserves1_amount + reserves2_amount)
        assert cb_balance.assets_by_kind == {}
        assert (
            cb_balance.liabilities_by_kind.get("reserve_deposit", 0)
            == reserves1_amount + reserves2_amount
        )

        # Verify trial balance
        trial = system_trial_balance(system)
        total_reserves = reserves1_amount + reserves2_amount
        assert trial.total_financial_assets == trial.total_financial_liabilities == total_reserves
        assert trial.assets_by_kind.get("reserve_deposit", 0) == total_reserves
        assert trial.liabilities_by_kind.get("reserve_deposit", 0) == total_reserves

        # Verify system tracking counters match
        assert system.state.cb_reserves_outstanding == total_reserves

        # Ensure all invariants pass
        system.assert_invariants()

    def test_duplicate_ref_impossible_with_sets(self):
        """Verify that set-based asset_ids/liability_ids prevent duplicates structurally.

        With the list→set migration, duplicate references are impossible by construction.
        This test verifies that .add() on an existing element is a no-op.
        """
        system = System()
        cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
        household = Household(id="HH01", name="Test Household", kind="household")

        system.bootstrap_cb(cb)
        system.add_agent(household)

        cash_id = system.mint_cash("HH01", 1000)
        system.assert_invariants()

        # Adding the same ID again is a no-op for sets
        size_before = len(household.asset_ids)
        household.asset_ids.add(cash_id)
        assert len(household.asset_ids) == size_before

        # Invariants still hold — no duplicates possible
        system.assert_invariants()

    def test_comprehensive_multi_agent_scenario(self):
        """Test a complex scenario with multiple agents and instrument types."""
        # Create system with multiple agents
        system = System()
        cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
        bank1 = Bank(id="BK01", name="Bank One", kind="bank")
        bank2 = Bank(id="BK02", name="Bank Two", kind="bank")
        hh1 = Household(id="HH01", name="Household One", kind="household")
        hh2 = Household(id="HH02", name="Household Two", kind="household")

        system.bootstrap_cb(cb)
        system.add_agent(bank1)
        system.add_agent(bank2)
        system.add_agent(hh1)
        system.add_agent(hh2)

        # Complex operations
        # 1. Mint reserves to banks
        system.mint_reserves("BK01", 10000)
        system.mint_reserves("BK02", 8000)

        # 2. Mint cash to households
        system.mint_cash("HH01", 2000)
        system.mint_cash("HH02", 1500)

        # 3. Make deposits
        deposit_cash(system, "HH01", "BK01", 1200)
        deposit_cash(system, "HH02", "BK02", 800)

        # 4. Create some non-financial instruments
        # Create stock for HH01
        system.create_stock("HH01", "INVENTORY", 50, Decimal("0"))
        # Create delivery obligation from HH01 to HH02
        system.create_delivery_obligation("HH01", "HH02", "WIDGETS", 25, Decimal("0"), due_day=1)

        # Verify system-wide balance
        trial = system_trial_balance(system)

        # Calculate expected totals
        total_reserves = 18000  # 10000 + 8000
        total_cash = 3500  # 2000 + 1500 original cash
        total_deposits = 2000  # 1200 + 800
        total_delivery_obligations = 25

        expected_financial_total = total_reserves + total_cash + total_deposits

        assert (
            trial.total_financial_assets
            == trial.total_financial_liabilities
            == expected_financial_total
        )
        assert trial.assets_by_kind.get("reserve_deposit", 0) == total_reserves
        assert trial.assets_by_kind.get("cash", 0) == total_cash
        assert trial.assets_by_kind.get("bank_deposit", 0) == total_deposits
        assert trial.assets_by_kind.get("delivery_obligation", 0) == total_delivery_obligations

        # Verify individual balances make sense
        hh1_balance = agent_balance(system, "HH01")
        # HH01 should have: remaining cash + deposit + delivery obligation liability
        expected_hh1_cash = 2000 - 1200  # 800
        assert hh1_balance.assets_by_kind.get("cash", 0) == expected_hh1_cash
        assert hh1_balance.assets_by_kind.get("bank_deposit", 0) == 1200
        assert hh1_balance.liabilities_by_kind.get("delivery_obligation", 0) == 25

        # Bank1 should have: reserves + cash from deposit, deposit liability
        bank1_balance = agent_balance(system, "BK01")
        assert bank1_balance.assets_by_kind.get("reserve_deposit", 0) == 10000
        assert bank1_balance.assets_by_kind.get("cash", 0) == 1200  # From HH01 deposit
        assert bank1_balance.liabilities_by_kind.get("bank_deposit", 0) == 1200

        # Ensure all invariants pass
        system.assert_invariants()
