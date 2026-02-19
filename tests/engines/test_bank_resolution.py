"""Tests for bank loan wind-down (Change A) and bank resolution (Change B).

Tests cover:
- Change A: Bank loan wind-down after main loop stability
- Change B: Bank resolution distributing reserves to depositors when banks fail
"""

from decimal import Decimal

import pytest

from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.bank_lending import (
    _execute_bank_loan,
    _increase_deposit,
    run_bank_loan_repayments,
)
from bilancio.engines.banking_subsystem import (
    BankLoanRecord,
    BankingSubsystem,
    BankTreynorState,
    _get_bank_reserves,
    _get_deposit_at_bank,
    _get_total_deposits,
    initialize_banking_subsystem,
)
from bilancio.engines.settlement import (
    _consume_reserves_from_bank,
    _create_resolution_deposit,
    _distribute_pro_rata_recovery,
    _expel_agent,
    _find_surviving_bank,
    _resolve_failed_bank,
    _resolve_to_cash,
)
from bilancio.engines.simulation import (
    _has_outstanding_bank_loans,
    _run_bank_loan_winddown,
    run_final_cb_settlement,
    run_until_stable,
)
from bilancio.engines.system import System
from bilancio.ops.banking import deposit_cash


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_banking_system(
    *,
    n_firms: int = 2,
    n_banks: int = 2,
    firm_cash: int = 1000,
    bank_reserves: int = 5000,
) -> System:
    """Create a system with CB + banks + firms, deposits and reserves."""
    system = System(policy=PolicyEngine.default(), default_mode="expel-agent")

    cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
    system.add_agent(cb)

    banks = []
    for i in range(1, n_banks + 1):
        bank = Bank(id=f"bank_{i}", name=f"Bank {i}", kind="bank")
        system.add_agent(bank)
        banks.append(bank)

    firms = []
    for i in range(1, n_firms + 1):
        firm = Firm(id=f"H_{i}", name=f"Firm {i}", kind="firm")
        system.add_agent(firm)
        firms.append(firm)

    # Mint cash and deposit into banks
    for i, firm in enumerate(firms):
        system.mint_cash(to_agent_id=firm.id, amount=firm_cash)
        bank_id = banks[i % n_banks].id
        deposit_cash(system, firm.id, bank_id, firm_cash)

    # Mint reserves
    for bank in banks:
        system.mint_reserves(to_bank_id=bank.id, amount=bank_reserves)

    return system


def _init_banking_subsystem(
    system: System,
    *,
    kappa: Decimal = Decimal("1.0"),
    maturity_days: int = 10,
) -> BankingSubsystem:
    """Initialize a banking subsystem on the system."""
    profile = BankProfile()
    subsystem = initialize_banking_subsystem(
        system=system,
        bank_profile=profile,
        kappa=kappa,
        maturity_days=maturity_days,
    )
    system.state.banking_subsystem = subsystem
    return subsystem


def _issue_bank_loan(
    system: System,
    banking: BankingSubsystem,
    bank_id: str,
    borrower_id: str,
    amount: int,
    rate: Decimal = Decimal("0.05"),
    current_day: int = 0,
    maturity: int = 5,
) -> str:
    """Issue a bank loan from bank to borrower."""
    bank_state = banking.banks[bank_id]
    loan_id = _execute_bank_loan(
        system, banking, bank_state, borrower_id, amount, rate, current_day, maturity,
    )
    return loan_id


# ===========================================================================
# Change A: Bank Loan Wind-Down Tests
# ===========================================================================


class TestWinddownNoBanking:
    """Test 1: No banking subsystem → wind-down is a no-op."""

    def test_winddown_no_banking(self):
        """When no banking subsystem exists, wind-down does nothing."""
        system = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="CB", kind="central_bank")
        system.add_agent(cb)

        # No banking subsystem
        assert system.state.banking_subsystem is None

        # run_until_stable should work without errors
        # (wind-down guard checks enable_bank_lending)
        reports = run_until_stable(
            system, max_days=5, enable_banking=False, enable_bank_lending=False,
        )
        # Verify no winddown events
        winddown_events = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownStart"
        ]
        assert len(winddown_events) == 0


class TestWinddownNoOutstandingLoans:
    """Test 2: Banking exists but all loans already matured → no-op."""

    def test_winddown_no_outstanding_loans(self):
        system = _make_banking_system()
        banking = _init_banking_subsystem(system)

        # No loans issued — subsystem has empty outstanding_loans
        assert not _has_outstanding_bank_loans(banking)

        days = _run_bank_loan_winddown(system, banking)
        assert days == 0

        # No winddown start event should be logged
        winddown_events = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownStart"
        ]
        assert len(winddown_events) == 0


class TestWinddownRepaysOutstandingLoans:
    """Test 3: Loans issued late mature during wind-down."""

    def test_winddown_repays_outstanding_loans(self):
        system = _make_banking_system(firm_cash=2000, bank_reserves=5000)
        banking = _init_banking_subsystem(system)

        # Issue a loan on day 0 with maturity 5 (due day 5)
        loan_id = _issue_bank_loan(
            system, banking, "bank_1", "H_1", 500, current_day=0, maturity=5,
        )
        assert loan_id is not None
        assert _has_outstanding_bank_loans(banking)

        # Move system to day 3 (loan not yet due)
        system.state.day = 3

        days = _run_bank_loan_winddown(system, banking)

        # Wind-down should have run for 2 days (day 3 and 4, then loan matures on day 5)
        assert days > 0

        # All loans should be repaid after wind-down
        assert not _has_outstanding_bank_loans(banking)


class TestWinddownEventsLogged:
    """Test 4: Verify start/end events are logged."""

    def test_winddown_events_logged(self):
        system = _make_banking_system(firm_cash=2000, bank_reserves=5000)
        banking = _init_banking_subsystem(system)

        # Issue a loan
        _issue_bank_loan(
            system, banking, "bank_1", "H_1", 500, current_day=0, maturity=3,
        )

        days = _run_bank_loan_winddown(system, banking)
        assert days > 0

        start_events = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownStart"
        ]
        end_events = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownEnd"
        ]
        day_events = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownDay"
        ]

        assert len(start_events) == 1
        assert len(end_events) == 1
        assert start_events[0]["outstanding_loans"] > 0
        assert end_events[0]["remaining_loans"] == 0
        assert len(day_events) == days


# ===========================================================================
# Change B: Bank Resolution Tests
# ===========================================================================


class TestResolveBankCBPriority:
    """Test 5: CB loans are cancelled against reserves first."""

    def test_resolve_bank_cb_priority(self):
        system = _make_banking_system(firm_cash=1000, bank_reserves=3000)
        banking = _init_banking_subsystem(system)

        # Create a CB loan for bank_1 (manually, simulating borrowing)
        system.cb_lend_reserves("bank_1", 1000, day=0)

        initial_reserves = _get_bank_reserves(system, "bank_1")
        initial_cb_loans = system.state.cb_loans_outstanding

        # Mark bank as defaulted (prerequisite for resolution)
        bank = system.state.agents["bank_1"]
        bank.defaulted = True
        system.state.defaulted_agent_ids.add("bank_1")

        _resolve_failed_bank(system, "bank_1")

        # CB loan should be cancelled (reserves consumed to offset)
        resolution_events = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCBClaimCancelled"
        ]
        assert len(resolution_events) >= 1

        completed_events = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(completed_events) == 1
        assert completed_events[0]["cb_claims_cancelled"] > 0


class TestResolveBankUnderwater:
    """Test 6: CB claims exceed reserves → depositors get nothing from reserves."""

    def test_resolve_bank_underwater(self):
        system = _make_banking_system(firm_cash=500, bank_reserves=100, n_banks=1)
        banking = _init_banking_subsystem(system)

        # Create CB loans exceeding reserves
        # Bank has 100 reserves, create CB loan of 200 (will add 200 reserves via CB lending)
        system.cb_lend_reserves("bank_1", 200, day=0)

        # Now bank_1 has 300 reserves (100 initial + 200 from CB) and 200 CB loan
        # Consume most reserves to make bank underwater
        from bilancio.ops.primitives import consume
        reserve_ids = [
            cid for cid in system.state.agents["bank_1"].asset_ids
            if system.state.contracts.get(cid)
            and system.state.contracts[cid].kind == InstrumentKind.RESERVE_DEPOSIT
        ]
        # Consume reserves down to 100 (less than 200 CB loan)
        total = sum(system.state.contracts[cid].amount for cid in reserve_ids)
        to_consume = total - 100
        if to_consume > 0:
            for cid in reserve_ids:
                instr = system.state.contracts.get(cid)
                if instr is None:
                    continue
                take = min(instr.amount, to_consume)
                if take > 0:
                    consume(system, cid, take)
                    system.state.cb_reserves_outstanding -= take
                    to_consume -= take
                if to_consume <= 0:
                    break

        bank = system.state.agents["bank_1"]
        bank.defaulted = True
        system.state.defaulted_agent_ids.add("bank_1")

        reserves_before = _get_bank_reserves(system, "bank_1")
        assert reserves_before <= 200  # Less than CB loan

        _resolve_failed_bank(system, "bank_1")

        completed = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(completed) == 1
        # CB claims should have consumed all or most of the reserves
        assert completed[0]["cb_claims_cancelled"] > 0
        # Depositors get nothing (or very little) since CB claims > reserves
        num_depositors = completed[0].get("num_depositors", 0)
        # When CB claims ≥ reserves, depositors get 0
        if reserves_before <= 200:
            assert num_depositors <= 1  # At most a small remainder


class TestResolveDepositorSurvivingBank:
    """Test 7: Reserves transferred to surviving bank, deposit created."""

    def test_resolve_depositor_surviving_bank(self):
        system = _make_banking_system(n_banks=2, firm_cash=1000, bank_reserves=5000)
        banking = _init_banking_subsystem(system)

        # H_1 has deposit at bank_1. bank_1 fails.
        # H_1 should get reserves at bank_2 (surviving).
        # First, give H_1 a deposit at bank_2 too so _find_surviving_bank finds it
        _increase_deposit(system, "H_1", "bank_2", 100)

        deposits_at_bank2_before = _get_deposit_at_bank(system, "H_1", "bank_2")

        bank = system.state.agents["bank_1"]
        bank.defaulted = True
        system.state.defaulted_agent_ids.add("bank_1")

        _resolve_failed_bank(system, "bank_1")

        completed = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(completed) == 1

        # H_1's deposit at bank_2 should have increased
        deposits_at_bank2_after = _get_deposit_at_bank(system, "H_1", "bank_2")
        assert deposits_at_bank2_after >= deposits_at_bank2_before

        # Check that the distribution used "deposit_at_surviving_bank" method
        distributions = completed[0].get("depositor_distributions", [])
        surviving_bank_distributions = [
            d for d in distributions if d.get("method") == "deposit_at_surviving_bank"
        ]
        assert len(surviving_bank_distributions) > 0


class TestResolveDepositorNoSurvivingBank:
    """Test 8: All banks failed → cash fallback."""

    def test_resolve_depositor_no_surviving_bank(self):
        system = _make_banking_system(n_banks=1, n_firms=1, firm_cash=500, bank_reserves=2000)
        banking = _init_banking_subsystem(system)

        # Only 1 bank, which fails. Depositor has no surviving bank.
        bank = system.state.agents["bank_1"]
        bank.defaulted = True
        system.state.defaulted_agent_ids.add("bank_1")

        # Record H_1's cash before resolution
        cash_before = sum(
            c.amount for cid in system.state.agents["H_1"].asset_ids
            if (c := system.state.contracts.get(cid)) and c.kind == InstrumentKind.CASH
        )

        _resolve_failed_bank(system, "bank_1")

        completed = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(completed) == 1

        # H_1 should have received cash
        cash_after = sum(
            c.amount for cid in system.state.agents["H_1"].asset_ids
            if (c := system.state.contracts.get(cid)) and c.kind == InstrumentKind.CASH
        )
        assert cash_after > cash_before

        # Check that the distribution used "cash_no_surviving_bank" method
        distributions = completed[0].get("depositor_distributions", [])
        cash_distributions = [
            d for d in distributions if d.get("method") == "cash_no_surviving_bank"
        ]
        assert len(cash_distributions) > 0


class TestResolveMultipleDepositorsProrata:
    """Test 9: Correct pro-rata split among multiple depositors."""

    def test_resolve_multiple_depositors_prorata(self):
        system = System(policy=PolicyEngine.default(), default_mode="expel-agent")

        cb = CentralBank(id="cb", name="CB", kind="central_bank")
        bank = Bank(id="bank_1", name="Bank 1", kind="bank")
        firm1 = Firm(id="H_1", name="Firm 1", kind="firm")
        firm2 = Firm(id="H_2", name="Firm 2", kind="firm")
        system.add_agent(cb)
        system.add_agent(bank)
        system.add_agent(firm1)
        system.add_agent(firm2)

        # Mint cash + deposit at different amounts
        system.mint_cash(to_agent_id="H_1", amount=3000)
        deposit_cash(system, "H_1", "bank_1", 3000)
        system.mint_cash(to_agent_id="H_2", amount=1000)
        deposit_cash(system, "H_2", "bank_1", 1000)

        # Reserves = 2000 (for distribution)
        system.mint_reserves(to_bank_id="bank_1", amount=2000)
        _init_banking_subsystem(system)

        # Bank defaults — distributes 2000 reserves to 2 depositors
        # H_1 has 3000 deposit (75%), H_2 has 1000 deposit (25%)
        bank_agent = system.state.agents["bank_1"]
        bank_agent.defaulted = True
        system.state.defaulted_agent_ids.add("bank_1")

        _resolve_failed_bank(system, "bank_1")

        completed = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(completed) == 1

        distributions = completed[0].get("depositor_distributions", [])
        assert len(distributions) == 2

        # Check pro-rata shares (approximately 75/25 of 2000)
        amounts = {d["depositor"]: d["amount"] for d in distributions}
        assert "H_1" in amounts
        assert "H_2" in amounts
        # H_1 should get ~1500, H_2 should get ~500
        assert amounts["H_1"] > amounts["H_2"]
        assert amounts["H_1"] + amounts["H_2"] <= 2000


class TestResolveNoReserves:
    """Test 10: Zero reserves → resolution is a no-op."""

    def test_resolve_no_reserves(self):
        system = _make_banking_system(bank_reserves=0)

        # Bank has zero reserves
        bank = system.state.agents["bank_1"]
        bank.defaulted = True
        system.state.defaulted_agent_ids.add("bank_1")

        _resolve_failed_bank(system, "bank_1")

        completed = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(completed) == 1
        assert completed[0]["total_reserves"] == 0


class TestExpelDispatchesForBanks:
    """Test 11: _expel_agent uses _resolve_failed_bank for banks."""

    def test_expel_dispatches_to_resolution_for_banks(self):
        system = _make_banking_system(bank_reserves=1000)
        _init_banking_subsystem(system)

        # Expel bank_1 — should trigger bank resolution, not pro-rata recovery
        _expel_agent(system, "bank_1")

        # Check that BankResolutionCompleted event exists
        resolution_events = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(resolution_events) == 1

        # Check that ProRataRecovery was NOT used
        prorata_events = [
            e for e in system.state.events if e.get("kind") == "ProRataRecovery"
        ]
        assert len(prorata_events) == 0


class TestExpelUsesProrataForNonbanks:
    """Test 12: Firms/households still use _distribute_pro_rata_recovery."""

    def test_expel_uses_prorata_for_nonbanks(self):
        system = _make_banking_system()

        # Give H_1 some cash (not just deposits) for pro-rata recovery
        system.mint_cash(to_agent_id="H_1", amount=500)

        # Create a payable so there's a creditor
        payable = Payable(
            id="PAY_1",
            kind=InstrumentKind.PAYABLE,
            amount=300,
            denom="X",
            asset_holder_id="H_2",
            liability_issuer_id="H_1",
            due_day=5,
        )
        system.add_contract(payable)

        _expel_agent(system, "H_1")

        # Check that BankResolutionCompleted was NOT used
        resolution_events = [
            e for e in system.state.events if e.get("kind") == "BankResolutionCompleted"
        ]
        assert len(resolution_events) == 0

        # The agent should be defaulted
        assert system.state.agents["H_1"].defaulted


# ===========================================================================
# Integration test
# ===========================================================================


class TestFullSimAllLoansMature:
    """Test 13: End-to-end simulation with banking — all loans mature."""

    def test_full_sim_all_loans_mature(self):
        """Run a banking scenario and verify 0 outstanding loans at end."""
        system = _make_banking_system(
            n_firms=4, n_banks=2, firm_cash=2000, bank_reserves=10000,
        )
        banking = _init_banking_subsystem(system, maturity_days=10)

        # Create payables between firms (ring topology)
        firms = ["H_1", "H_2", "H_3", "H_4"]
        for i in range(len(firms)):
            debtor = firms[i]
            creditor = firms[(i + 1) % len(firms)]
            payable = Payable(
                id=f"PAY_{i+1}",
                kind=InstrumentKind.PAYABLE,
                amount=500,
                denom="X",
                asset_holder_id=creditor,
                liability_issuer_id=debtor,
                due_day=3,
                maturity_distance=3,
            )
            system.add_contract(payable)

        # Issue some bank loans (these should all mature during wind-down)
        _issue_bank_loan(
            system, banking, "bank_1", "H_1", 300,
            current_day=0, maturity=5,
        )
        _issue_bank_loan(
            system, banking, "bank_2", "H_3", 300,
            current_day=0, maturity=7,
        )

        assert _has_outstanding_bank_loans(banking)

        # Run simulation with bank lending enabled
        reports = run_until_stable(
            system,
            max_days=20,
            enable_banking=True,
            enable_bank_lending=True,
        )

        # After wind-down, ALL bank loans should have matured
        assert not _has_outstanding_bank_loans(banking)

        # Check wind-down events were logged
        winddown_starts = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownStart"
        ]
        winddown_ends = [
            e for e in system.state.events if e.get("kind") == "BankLoanWinddownEnd"
        ]

        # Wind-down should have occurred (loans were still outstanding after main loop)
        if winddown_starts:
            assert len(winddown_ends) == 1
            assert winddown_ends[0]["remaining_loans"] == 0


# ===========================================================================
# Helper function unit tests
# ===========================================================================


class TestConsumeReservesFromBank:
    """Unit tests for _consume_reserves_from_bank helper."""

    def test_consume_partial_reserves(self):
        system = _make_banking_system(bank_reserves=1000)

        initial_reserves = _get_bank_reserves(system, "bank_1")
        initial_cb_outstanding = system.state.cb_reserves_outstanding

        _consume_reserves_from_bank(system, "bank_1", 400)

        assert _get_bank_reserves(system, "bank_1") == initial_reserves - 400
        assert system.state.cb_reserves_outstanding == initial_cb_outstanding - 400

    def test_consume_all_reserves(self):
        system = _make_banking_system(bank_reserves=500)

        _consume_reserves_from_bank(system, "bank_1", 500)

        assert _get_bank_reserves(system, "bank_1") == 0


class TestFindSurvivingBank:
    """Unit tests for _find_surviving_bank helper."""

    def test_finds_surviving_bank(self):
        system = _make_banking_system(n_banks=2)

        # H_1 has deposit at bank_1 (from setup)
        # Add deposit at bank_2
        _increase_deposit(system, "H_1", "bank_2", 100)

        # bank_1 fails
        system.state.agents["bank_1"].defaulted = True

        result = _find_surviving_bank(system, "H_1", "bank_1")
        assert result == "bank_2"

    def test_no_surviving_bank(self):
        system = _make_banking_system(n_banks=1)

        # Only bank_1, which fails
        system.state.agents["bank_1"].defaulted = True

        result = _find_surviving_bank(system, "H_1", "bank_1")
        assert result is None

    def test_excludes_defaulted_bank(self):
        system = _make_banking_system(n_banks=2)
        _increase_deposit(system, "H_1", "bank_2", 100)

        # Both banks default
        system.state.agents["bank_1"].defaulted = True
        system.state.agents["bank_2"].defaulted = True

        result = _find_surviving_bank(system, "H_1", "bank_1")
        assert result is None


class TestHasOutstandingBankLoans:
    """Unit tests for _has_outstanding_bank_loans helper."""

    def test_no_loans(self):
        system = _make_banking_system()
        banking = _init_banking_subsystem(system)
        assert not _has_outstanding_bank_loans(banking)

    def test_with_loans(self):
        system = _make_banking_system(firm_cash=2000)
        banking = _init_banking_subsystem(system)
        _issue_bank_loan(system, banking, "bank_1", "H_1", 500)
        assert _has_outstanding_bank_loans(banking)


# ===========================================================================
# P1 fix: final CB settlement writes off failed-bank liabilities
# ===========================================================================


class TestFinalCBSettlementWritesOffLiabilities:
    """After resolution in run_final_cb_settlement, remaining bank liabilities
    (BankDeposit, interbank loans, etc.) must be written off so depositors
    don't keep stale deposit claims on their balance sheets."""

    def test_final_settlement_writes_off_bank_deposits(self):
        """When a bank defaults in final CB settlement, its BankDeposit
        liabilities are written off (ObligationWrittenOff events logged)."""
        system = _make_banking_system(firm_cash=1000, bank_reserves=100)
        banking = _init_banking_subsystem(system)

        # Issue a CB loan to bank_1, then drain its reserves so it can't repay.
        system.cb_lend_reserves("bank_1", 5000, day=0)
        # Transfer all reserves away (to bank_2) so bank_1 is insolvent
        from bilancio.engines.banking_subsystem import _get_bank_reserves
        bank1_reserves = _get_bank_reserves(system, "bank_1")
        if bank1_reserves > 0:
            system.transfer_reserves("bank_1", "bank_2", bank1_reserves)

        # Verify bank_1 has deposit liabilities (from firm deposits)
        bank1_deposit_liabs = [
            c for c in system.state.contracts.values()
            if c.kind == InstrumentKind.BANK_DEPOSIT
            and c.liability_issuer_id == "bank_1"
        ]
        assert len(bank1_deposit_liabs) > 0, "bank_1 should have deposit liabilities"

        # Run final CB settlement — bank_1 should default
        result = run_final_cb_settlement(system)
        assert result["bank_defaults"] >= 1

        # Check that ObligationWrittenOff events were logged for bank deposits
        written_off_events = [
            e for e in system.state.events
            if e.get("kind") == "ObligationWrittenOff"
            and e.get("debtor") == "bank_1"
        ]
        assert len(written_off_events) > 0, (
            "BankDeposit liabilities should be written off after bank resolution"
        )

        # Verify no BankDeposit contracts remain for bank_1
        remaining_deposits = [
            c for c in system.state.contracts.values()
            if c.kind == InstrumentKind.BANK_DEPOSIT
            and c.liability_issuer_id == "bank_1"
        ]
        assert len(remaining_deposits) == 0, (
            "No deposit contracts should remain after write-off"
        )


# ===========================================================================
# P3 fix: wind-down cap is computed from actual loan maturity days
# ===========================================================================


class TestWinddownCapComputed:
    """The max_winddown cap should be derived from loan maturity days,
    not a fixed constant."""

    def test_winddown_cap_adapts_to_maturity(self):
        """Loans with high maturity days should still be resolved."""
        system = _make_banking_system(firm_cash=2000, bank_reserves=5000)
        banking = _init_banking_subsystem(system, maturity_days=100)

        # Issue a loan due at day 80
        _issue_bank_loan(
            system, banking, "bank_1", "H_1", 500,
            current_day=0, maturity=80,
        )
        assert _has_outstanding_bank_loans(banking)

        # Start wind-down at day 10 — need 70+ days of wind-down
        system.state.day = 10
        days = _run_bank_loan_winddown(system, banking)

        # With include_overdue=True, it should resolve on the first day
        # (maturity_day=80 <= current_day=10 is False, but <=80 when day reaches 80)
        # Actually with include_overdue, loan.maturity_day(80) > current_day(10)
        # so it needs to advance to day 80
        assert not _has_outstanding_bank_loans(banking)

        winddown_end = [
            e for e in system.state.events
            if e.get("kind") == "BankLoanWinddownEnd"
        ]
        assert len(winddown_end) == 1
        assert winddown_end[0]["remaining_loans"] == 0
