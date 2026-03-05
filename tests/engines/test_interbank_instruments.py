"""Tests for interbank loans as real balance sheet instruments.

Verifies that interbank auction loans are promoted to real Instrument
subclasses that appear in system.state.contracts, agent asset_ids/liability_ids,
and are cleaned up on repayment and wind-down.
"""

from decimal import Decimal

from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.interbank_loan import InterbankLoanContract
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.banking_subsystem import (
    InterbankLoan,
    _get_bank_reserves,
    initialize_banking_subsystem,
)
from bilancio.engines.interbank import (
    compute_interbank_obligations,
    finalize_interbank_repayments,
    run_interbank_auction,
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

    system.mint_cash(to_agent_id="H_1", amount=1000)
    system.mint_cash(to_agent_id="H_2", amount=1000)
    deposit_cash(system, "H_1", "bank_1", 1000)
    deposit_cash(system, "H_2", "bank_2", 1000)
    # Give bank_1 surplus reserves, bank_2 deficit
    system.mint_reserves(to_bank_id="bank_1", amount=8000)
    system.mint_reserves(to_bank_id="bank_2", amount=2000)

    return system


def _make_initialized_banking(system, target_1=5000, target_2=5000):
    """Initialize banking subsystem with custom reserve targets."""
    profile = BankProfile()
    kappa = Decimal("1.0")
    banking = initialize_banking_subsystem(
        system=system,
        bank_profile=profile,
        kappa=kappa,
        maturity_days=10,
    )
    banking.banks["bank_1"].pricing_params.reserve_target = target_1
    banking.banks["bank_2"].pricing_params.reserve_target = target_2
    return banking


def _run_auction_with_trade(system, banking, day=5):
    """Run an auction that produces at least one trade. Returns events."""
    # bank_1 has 8000 reserves vs 5000 target -> surplus of 3000
    # bank_2 has 2000 reserves vs 5000 target -> deficit of -3000
    events = run_interbank_auction(system, day, banking, net_obligations={})
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInterbankLoanOnBalanceSheet:
    """Test that interbank loans appear as real instruments after auction."""

    def test_contract_exists_after_auction(self):
        """After auction with trades, contract exists in system.state.contracts."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        events = _run_auction_with_trade(system, banking, day=5)

        # Should have at least one trade event
        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) >= 1

        # Check the contract exists
        contract_id = trade_events[0]["contract_id"]
        assert contract_id in system.state.contracts

        contract = system.state.contracts[contract_id]
        assert isinstance(contract, InterbankLoanContract)
        assert contract.kind == InstrumentKind.INTERBANK_LOAN

    def test_contract_on_lender_assets(self):
        """Contract ID appears in lender bank's asset_ids."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        events = _run_auction_with_trade(system, banking, day=5)

        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) >= 1

        contract_id = trade_events[0]["contract_id"]
        lender_id = trade_events[0]["lender"]
        lender = system.state.agents[lender_id]
        assert contract_id in lender.asset_ids

    def test_contract_on_borrower_liabilities(self):
        """Contract ID appears in borrower bank's liability_ids."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        events = _run_auction_with_trade(system, banking, day=5)

        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) >= 1

        contract_id = trade_events[0]["contract_id"]
        borrower_id = trade_events[0]["borrower"]
        borrower = system.state.agents[borrower_id]
        assert contract_id in borrower.liability_ids

    def test_contract_fields_correct(self):
        """Contract has correct kind, amount, rate, due_day."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        events = _run_auction_with_trade(system, banking, day=5)

        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        trade = trade_events[0]
        contract = system.state.contracts[trade["contract_id"]]

        assert contract.amount == trade["amount"]
        assert contract.rate == Decimal(trade["rate"])
        assert contract.issuance_day == 5
        assert contract.due_day == 6  # overnight = day + 1
        assert contract.maturity_day == 6
        assert contract.asset_holder_id == trade["lender"]
        assert contract.liability_issuer_id == trade["borrower"]

    def test_contract_in_due_day_index(self):
        """Contract appears in contracts_by_due_day for day+1."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        _run_auction_with_trade(system, banking, day=5)

        # Should have entries due on day 6
        due_day_6 = system.state.contracts_by_due_day.get(6, [])
        ibl_contracts = [
            cid for cid in due_day_6
            if system.state.contracts[cid].kind == InstrumentKind.INTERBANK_LOAN
        ]
        assert len(ibl_contracts) >= 1


class TestInterbankLoanRemovedAfterRepayment:
    """Test that instruments are cleaned up on repayment."""

    def test_contract_removed_after_finalize(self):
        """Contract gone from system.state.contracts after finalize."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        _run_auction_with_trade(system, banking, day=5)

        # Get the contract IDs before finalization
        ibl_contracts = [
            cid for cid, c in system.state.contracts.items()
            if isinstance(c, InterbankLoanContract)
        ]
        assert len(ibl_contracts) >= 1

        # Compute obligations for day 6 (maturity)
        obligations = compute_interbank_obligations(6, banking)
        assert len(obligations) >= 1

        # Finalize repayments
        repay_events = finalize_interbank_repayments(
            system, 6, banking, obligations,
        )

        # Contract should be gone
        for cid in ibl_contracts:
            assert cid not in system.state.contracts

    def test_agent_ids_cleaned_after_finalize(self):
        """Contract ID removed from both agents' asset/liability sets."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        events = _run_auction_with_trade(system, banking, day=5)

        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        trade = trade_events[0]
        contract_id = trade["contract_id"]
        lender_id = trade["lender"]
        borrower_id = trade["borrower"]

        # Before finalization: in agent id sets
        assert contract_id in system.state.agents[lender_id].asset_ids
        assert contract_id in system.state.agents[borrower_id].liability_ids

        # Finalize
        obligations = compute_interbank_obligations(6, banking)
        finalize_interbank_repayments(system, 6, banking, obligations)

        # After finalization: removed from agent id sets
        assert contract_id not in system.state.agents[lender_id].asset_ids
        assert contract_id not in system.state.agents[borrower_id].liability_ids

    def test_due_day_index_cleaned(self):
        """Contract removed from contracts_by_due_day index."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        _run_auction_with_trade(system, banking, day=5)

        obligations = compute_interbank_obligations(6, banking)
        finalize_interbank_repayments(system, 6, banking, obligations)

        # No interbank loan contracts should remain in day 6 bucket
        due_day_6 = system.state.contracts_by_due_day.get(6, [])
        ibl_in_bucket = [
            cid for cid in due_day_6
            if cid in system.state.contracts
            and system.state.contracts[cid].kind == InstrumentKind.INTERBANK_LOAN
        ]
        assert len(ibl_in_bucket) == 0


class TestBookkeepingListStillMaintained:
    """Verify the parallel bookkeeping list is still populated."""

    def test_bookkeeping_list_populated(self):
        """banking.interbank_loans list is populated alongside instruments."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        _run_auction_with_trade(system, banking, day=5)

        assert len(banking.interbank_loans) >= 1
        loan = banking.interbank_loans[0]
        assert isinstance(loan, InterbankLoan)
        assert loan.maturity_day == 6

    def test_bookkeeping_list_cleared_on_finalize(self):
        """banking.interbank_loans list is cleared after finalize."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        _run_auction_with_trade(system, banking, day=5)

        obligations = compute_interbank_obligations(6, banking)
        finalize_interbank_repayments(system, 6, banking, obligations)

        # All matured loans should be removed
        remaining = [l for l in banking.interbank_loans if l.maturity_day == 6]
        assert len(remaining) == 0


class TestContractIdLinked:
    """Verify InterbankLoan.contract_id matches the real instrument."""

    def test_contract_id_matches(self):
        """InterbankLoan.contract_id points to the correct instrument."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        _run_auction_with_trade(system, banking, day=5)

        for loan in banking.interbank_loans:
            assert loan.contract_id is not None
            contract = system.state.contracts[loan.contract_id]
            assert isinstance(contract, InterbankLoanContract)
            assert contract.amount == loan.amount
            assert contract.asset_holder_id == loan.lender_bank
            assert contract.liability_issuer_id == loan.borrower_bank
            assert contract.rate == loan.rate
            assert contract.issuance_day == loan.issuance_day


class TestFailedRepaymentPreservesContract:
    """Verify that failed repayments do NOT remove contracts from balance sheets."""

    def test_failed_finalize_keeps_contract(self):
        """If obligation is excluded from finalize, contract stays on balance sheet."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        events = _run_auction_with_trade(system, banking, day=5)

        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) >= 1
        contract_id = trade_events[0]["contract_id"]
        lender_id = trade_events[0]["lender"]
        borrower_id = trade_events[0]["borrower"]

        # Compute obligations but finalize with an EMPTY list
        # (simulating wind-down where all transfers failed)
        _obligations = compute_interbank_obligations(6, banking)
        assert len(_obligations) >= 1
        finalize_interbank_repayments(system, 6, banking, [])

        # Contract must still exist on balance sheets
        assert contract_id in system.state.contracts
        assert contract_id in system.state.agents[lender_id].asset_ids
        assert contract_id in system.state.agents[borrower_id].liability_ids

    def test_partial_failure_keeps_unsettled(self):
        """When only some obligations settle, unsettled contracts survive."""
        # Create system with 3 banks: surplus bank_1, deficit bank_2 and bank_3
        system = System(policy=PolicyEngine.default())
        cb = CentralBank(id="cb", name="CB", kind="central_bank")
        bank1 = Bank(id="bank_1", name="Bank 1", kind="bank")
        bank2 = Bank(id="bank_2", name="Bank 2", kind="bank")
        bank3 = Bank(id="bank_3", name="Bank 3", kind="bank")
        firm1 = Firm(id="H_1", name="Firm 1", kind="firm")
        firm2 = Firm(id="H_2", name="Firm 2", kind="firm")
        firm3 = Firm(id="H_3", name="Firm 3", kind="firm")
        for a in [cb, bank1, bank2, bank3, firm1, firm2, firm3]:
            system.add_agent(a)
        for fid, bid, cash, reserves in [
            ("H_1", "bank_1", 500, 10000),
            ("H_2", "bank_2", 500, 1000),
            ("H_3", "bank_3", 500, 1000),
        ]:
            system.mint_cash(to_agent_id=fid, amount=cash)
            deposit_cash(system, fid, bid, cash)
            system.mint_reserves(to_bank_id=bid, amount=reserves)

        profile = BankProfile()
        banking = initialize_banking_subsystem(
            system=system, bank_profile=profile,
            kappa=Decimal("1.0"), maturity_days=10,
        )
        for bid in ["bank_1", "bank_2", "bank_3"]:
            banking.banks[bid].pricing_params.reserve_target = 5000

        # Run auction — bank_1 (surplus) lends to bank_2 and/or bank_3
        auction_events = run_interbank_auction(system, 5, banking, net_obligations={})
        trade_events = [e for e in auction_events if e["kind"] == "InterbankAuctionTrade"]

        if len(trade_events) < 2:
            # Need at least 2 trades to test partial failure
            return

        # Finalize only the first obligation (simulate second failed)
        all_obligations = compute_interbank_obligations(6, banking)
        settled = all_obligations[:1]
        unsettled_loan = all_obligations[1][3]

        finalize_interbank_repayments(system, 6, banking, settled)

        # The settled contract should be removed
        settled_cid = settled[0][3].contract_id
        assert settled_cid not in system.state.contracts

        # The unsettled contract must still exist
        unsettled_cid = unsettled_loan.contract_id
        assert unsettled_cid is not None
        assert unsettled_cid in system.state.contracts


class TestPolicyAllowsBankToBank:
    """Verify PolicyEngine accepts interbank loans between banks."""

    def test_bank_can_issue_interbank_loan(self):
        """Borrower bank can issue (be liability issuer of) interbank loan."""
        system = _make_banking_system()
        bank = system.state.agents["bank_2"]
        contract = InterbankLoanContract(
            id="test_ibl",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="bank_2",
            rate=Decimal("0.01"),
            issuance_day=0,
        )
        assert system.policy.can_issue(bank, contract)

    def test_bank_can_hold_interbank_loan(self):
        """Lender bank can hold (be asset holder of) interbank loan."""
        system = _make_banking_system()
        bank = system.state.agents["bank_1"]
        contract = InterbankLoanContract(
            id="test_ibl",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="bank_2",
            rate=Decimal("0.01"),
            issuance_day=0,
        )
        assert system.policy.can_hold(bank, contract)

    def test_firm_cannot_hold_interbank_loan(self):
        """Non-bank agents cannot hold interbank loans."""
        system = _make_banking_system()
        firm = system.state.agents["H_1"]
        contract = InterbankLoanContract(
            id="test_ibl",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="H_1",
            liability_issuer_id="bank_2",
            rate=Decimal("0.01"),
            issuance_day=0,
        )
        assert not system.policy.can_hold(firm, contract)

    def test_firm_cannot_issue_interbank_loan(self):
        """Non-bank agents cannot issue interbank loans."""
        system = _make_banking_system()
        firm = system.state.agents["H_1"]
        contract = InterbankLoanContract(
            id="test_ibl",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="H_1",
            rate=Decimal("0.01"),
            issuance_day=0,
        )
        assert not system.policy.can_issue(firm, contract)


class TestInterbankLoanContract:
    """Unit tests for InterbankLoanContract instrument properties."""

    def test_maturity_day(self):
        """Overnight = issuance_day + 1."""
        contract = InterbankLoanContract(
            id="ibl_1",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="bank_2",
            issuance_day=5,
        )
        assert contract.maturity_day == 6
        assert contract.due_day == 6

    def test_repayment_amount(self):
        """Repayment = principal * (1 + rate)."""
        contract = InterbankLoanContract(
            id="ibl_1",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="bank_2",
            rate=Decimal("0.05"),
            issuance_day=0,
        )
        assert contract.repayment_amount == 1050
        assert contract.interest_amount == 50
        assert contract.principal == 1000

    def test_is_due(self):
        """is_due returns True on and after maturity day."""
        contract = InterbankLoanContract(
            id="ibl_1",
            kind=InstrumentKind.INTERBANK_LOAN,
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="bank_2",
            issuance_day=3,
        )
        assert not contract.is_due(3)  # issuance day
        assert contract.is_due(4)      # maturity day
        assert contract.is_due(5)      # after maturity

    def test_kind_set_automatically(self):
        """Kind is set to INTERBANK_LOAN in __post_init__."""
        contract = InterbankLoanContract(
            id="ibl_1",
            kind=InstrumentKind.CASH,  # wrong kind, should be overridden
            amount=1000,
            denom="X",
            asset_holder_id="bank_1",
            liability_issuer_id="bank_2",
        )
        assert contract.kind == InstrumentKind.INTERBANK_LOAN
