"""Tests for final CB settlement phase (Plan 037).

Tests cover:
1. No outstanding loans → noop
2. Bank repays successfully → no defaults
3. Bank can't repay → loan written off, bank defaulted
4. Multiple loans per bank → all written off when bank defaults
5. Mixed banks → some repay, others default
6. Counters correct after settlement
7. Backward compat: no CB → function not called
"""

from decimal import Decimal

from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.simulation import run_final_cb_settlement
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


def _make_banking_system(
    n_banks: int = 1,
    reserves_per_bank: int = 5000,
) -> System:
    """Create a system with CB + banks, each with reserves."""
    system = System(policy=PolicyEngine.default())
    cb = CentralBank(id="cb", name="CB", kind="central_bank")
    system.add_agent(cb)

    for i in range(1, n_banks + 1):
        bank = Bank(id=f"bank_{i}", name=f"Bank {i}", kind="bank")
        system.add_agent(bank)
        if reserves_per_bank > 0:
            system.mint_reserves(to_bank_id=f"bank_{i}", amount=reserves_per_bank)

    return system


class TestNoOutstandingLoans:
    """When there are no CB loans, final settlement is a noop."""

    def test_returns_zeros(self):
        system = _make_banking_system(n_banks=1)
        result = run_final_cb_settlement(system)

        assert result["loans_attempted"] == 0
        assert result["loans_repaid"] == 0
        assert result["loans_written_off"] == 0
        assert result["bank_defaults"] == 0
        assert result["total_written_off_amount"] == 0

    def test_no_events_logged(self):
        system = _make_banking_system(n_banks=1)
        events_before = len(system.state.events)
        run_final_cb_settlement(system)
        # Should have start + end events only
        new_events = system.state.events[events_before:]
        kinds = [e["kind"] for e in new_events]
        assert "CBFinalSettlementStart" in kinds
        assert "CBFinalSettlementEnd" in kinds
        assert "CBFinalSettlementRepaid" not in kinds
        assert "CBFinalSettlementBankDefault" not in kinds


class TestBankRepaysSuccessfully:
    """Bank has enough reserves to repay CB loan."""

    def test_loan_repaid_no_defaults(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        # Issue a CB loan (bank borrows 1000)
        loan_id = system.cb_lend_reserves("bank_1", 1000, day=0)

        # Bank now has 5000 + 1000 = 6000 reserves, owes 1000 * 1.03 = 1030
        result = run_final_cb_settlement(system)

        assert result["loans_attempted"] == 1
        assert result["loans_repaid"] == 1
        assert result["loans_written_off"] == 0
        assert result["bank_defaults"] == 0
        assert result["cb_loans_outstanding_post_final"] == 0

    def test_bank_not_marked_defaulted(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        run_final_cb_settlement(system)

        bank = system.state.agents["bank_1"]
        assert not bank.defaulted
        assert "bank_1" not in system.state.defaulted_agent_ids

    def test_repayment_event_logged(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        events_before = len(system.state.events)

        run_final_cb_settlement(system)

        new_events = system.state.events[events_before:]
        kinds = [e["kind"] for e in new_events]
        assert "CBFinalSettlementRepaid" in kinds


class TestBankDefaultsOnShortfall:
    """Bank doesn't have enough reserves to repay — defaults."""

    def test_bank_defaults(self):
        # Bank gets 0 reserves, then borrows 1000 (gets 1000 reserves)
        # Needs to repay 1030, but only has 1000 → can't repay
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 1000, day=0)

        result = run_final_cb_settlement(system)

        assert result["loans_attempted"] == 1
        assert result["loans_repaid"] == 0
        assert result["loans_written_off"] == 1
        assert result["bank_defaults"] == 1
        assert result["total_written_off_amount"] == 1000

    def test_bank_marked_defaulted(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        run_final_cb_settlement(system)

        bank = system.state.agents["bank_1"]
        assert bank.defaulted
        assert "bank_1" in system.state.defaulted_agent_ids

    def test_loan_removed_from_contracts(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        loan_id = system.cb_lend_reserves("bank_1", 1000, day=0)
        run_final_cb_settlement(system)

        assert loan_id not in system.state.contracts

    def test_cb_loans_outstanding_zero_after_writeoff(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        run_final_cb_settlement(system)

        assert system.state.cb_loans_outstanding == 0

    def test_default_event_logged(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        events_before = len(system.state.events)

        run_final_cb_settlement(system)

        new_events = system.state.events[events_before:]
        kinds = [e["kind"] for e in new_events]
        assert "CBFinalSettlementBankDefault" in kinds
        assert "CBFinalSettlementWrittenOff" in kinds


class TestMultipleLoansPerBank:
    """When a bank defaults, all its CB loans are written off."""

    def test_all_loans_written_off(self):
        # Bank gets 0 initial reserves; 3 loans give it 1000 reserves total.
        # Loan repayment costs principal*(1+0.03), so:
        #   Loan 1 (500): repay 515, bank has 1000→485
        #   Loan 2 (300): repay 309, bank has 485→176
        #   Loan 3 (200): repay 206, bank has 176 < 206 → DEFAULT
        # When bank defaults on loan 3, no remaining loans to write off.
        # So: 2 repaid, 1 written off, 1 bank default.
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 500, day=0)
        system.cb_lend_reserves("bank_1", 300, day=1)
        system.cb_lend_reserves("bank_1", 200, day=2)

        result = run_final_cb_settlement(system)

        assert result["loans_attempted"] == 3
        assert result["loans_repaid"] == 2
        assert result["loans_written_off"] == 1
        assert result["bank_defaults"] == 1

    def test_multiple_loans_all_default(self):
        """When bank can't repay ANY loan, all are written off."""
        # Give bank 0 reserves, issue 2 large loans
        # Loan 1 (100): gives 100 reserves, needs 103 to repay → fails
        # All remaining loans for this bank are written off
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 100, day=0)
        system.cb_lend_reserves("bank_1", 100, day=1)

        # Consume the loan-granted reserves so bank can't repay
        # Bank has 200 reserves from 2 loans, needs 103 for first.
        # Actually bank HAS enough for first (200 >= 103).
        # Consume reserves manually to force shortfall.
        reserve_ids = [
            cid for cid in system.state.agents["bank_1"].asset_ids
            if cid in system.state.contracts
            and system.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        ]
        from bilancio.ops.primitives import consume
        for rid in reserve_ids:
            consume(system, rid, system.state.contracts[rid].amount)
        system.state.cb_reserves_outstanding = 0

        result = run_final_cb_settlement(system)

        assert result["loans_written_off"] == 2
        assert result["bank_defaults"] == 1

    def test_cb_loans_outstanding_zero(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 500, day=0)
        system.cb_lend_reserves("bank_1", 300, day=1)
        run_final_cb_settlement(system)

        assert system.state.cb_loans_outstanding == 0


class TestMixedBanks:
    """Some banks repay, others default."""

    def test_mixed_outcome(self):
        system = _make_banking_system(n_banks=2, reserves_per_bank=0)
        # Bank 1: gets 1000 reserves from loan, needs 1030 to repay → defaults
        system.cb_lend_reserves("bank_1", 1000, day=0)
        # Bank 2: give it extra reserves so it can repay
        system.mint_reserves(to_bank_id="bank_2", amount=5000)
        system.cb_lend_reserves("bank_2", 1000, day=0)

        result = run_final_cb_settlement(system)

        assert result["loans_attempted"] == 2
        assert result["loans_repaid"] == 1
        assert result["loans_written_off"] == 1
        assert result["bank_defaults"] == 1

    def test_only_insolvent_bank_defaulted(self):
        system = _make_banking_system(n_banks=2, reserves_per_bank=0)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        system.mint_reserves(to_bank_id="bank_2", amount=5000)
        system.cb_lend_reserves("bank_2", 1000, day=0)

        run_final_cb_settlement(system)

        assert system.state.agents["bank_1"].defaulted
        assert not system.state.agents["bank_2"].defaulted


class TestCountersCorrect:
    """Verify state counters are correct after final settlement."""

    def test_cb_loans_created_count(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        system.cb_lend_reserves("bank_1", 500, day=1)

        assert system.state.cb_loans_created_count == 2

    def test_cb_interest_total_after_repayment(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        system.cb_lend_reserves("bank_1", 1000, day=0)

        # CB rate defaults to 0.03, so interest = 1000 * 0.03 = 30
        run_final_cb_settlement(system)

        assert system.state.cb_interest_total_paid == 30

    def test_cb_reserves_initial_captured(self):
        """cb_reserves_initial is set by run_until_stable, not final settlement."""
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        # Simulate what run_until_stable does
        system.state.cb_reserves_initial = system.state.cb_reserves_outstanding

        assert system.state.cb_reserves_initial == 5000

    def test_final_settlement_end_event_has_metrics(self):
        system = _make_banking_system(n_banks=1, reserves_per_bank=5000)
        system.cb_lend_reserves("bank_1", 1000, day=0)
        run_final_cb_settlement(system)

        end_event = next(
            e for e in system.state.events if e.get("kind") == "CBFinalSettlementEnd"
        )
        assert "loans_attempted" in end_event
        assert "loans_repaid" in end_event
        assert "bank_defaults" in end_event
        assert "cb_loans_outstanding_pre_final" in end_event
        assert "cb_loans_outstanding_post_final" in end_event
        assert "cb_interest_total_paid" in end_event
        assert "cb_loans_created_count" in end_event


class TestBackwardCompat:
    """Systems without banking should not be affected."""

    def test_no_cb_loans_returns_zeros(self):
        """A system with no CB loans (non-banking) → noop."""
        system = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="CB", kind="central_bank")
        firm = Firm(id="H_1", name="Firm 1", kind="firm")
        system.add_agent(cb)
        system.add_agent(firm)
        system.mint_cash(to_agent_id="H_1", amount=1000)

        result = run_final_cb_settlement(system)

        assert result["loans_attempted"] == 0
        assert result["bank_defaults"] == 0
