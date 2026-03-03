"""Unit tests for call-auction interbank reserve market.

Tests cover:
1. compute_reserve_positions — balanced, surplus/deficit, defaulted excluded
2. compute_limit_rates — surplus→below M, deficit→above M, boundaries
3. build_order_book — correct sorting, zero excluded
4. clear_auction — full match, partial, no match, single pair, multiple banks
5. run_interbank_auction — reserve transfers, overnight maturity, events
6. compute_combined_nets — client-only, with interbank, netting offset
7. compute_bank_net_obligations — two and three banks
8. finalize_interbank_repayments — removes matured, keeps non-matured, events
9. Phase C pipeline integration
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
    InterbankLoan,
    _get_bank_reserves,
    initialize_banking_subsystem,
)
from bilancio.engines.clearing import (
    compute_bank_net_obligations,
    compute_combined_nets,
)
from bilancio.engines.interbank import (
    AuctionResult,
    InterbankOrder,
    build_order_book,
    clear_auction,
    compute_interbank_obligations,
    compute_limit_rates,
    compute_reserve_positions,
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
    system.mint_reserves(to_bank_id="bank_1", amount=5000)
    system.mint_reserves(to_bank_id="bank_2", amount=5000)

    return system


def _make_initialized_banking(system, target_1=500, target_2=500):
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


# ---------------------------------------------------------------------------
# TestComputeReservePositions
# ---------------------------------------------------------------------------


class TestComputeReservePositions:
    """Tests for compute_reserve_positions."""

    def test_balanced_positions(self):
        """Both banks at target reserves -> positions near 0."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=5000, target_2=5000)

        positions = compute_reserve_positions(system, banking, {})
        # Both have 5000 reserves, 5000 target -> position = 0
        assert positions["bank_1"] == 0
        assert positions["bank_2"] == 0

    def test_surplus_deficit(self):
        """One bank surplus, other deficit."""
        system = _make_banking_system()
        # Move 3000 reserves from bank_2 to bank_1
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)

        positions = compute_reserve_positions(system, banking, {})
        # bank_1: 8000 reserves - 0 obligations - 2000 target = 6000
        assert positions["bank_1"] == 6000
        # bank_2: 2000 reserves - 0 obligations - 5000 target = -3000
        assert positions["bank_2"] == -3000

    def test_net_obligations_affect_position(self):
        """Net obligations reduce the position."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=2000, target_2=2000)

        net_obligations = {"bank_1": 1000, "bank_2": -500}
        positions = compute_reserve_positions(system, banking, net_obligations)
        # bank_1: 5000 - 1000 - 2000 = 2000
        assert positions["bank_1"] == 2000
        # bank_2: 5000 - (-500) - 2000 = 3500
        assert positions["bank_2"] == 3500

    def test_defaulted_bank_excluded(self):
        """Defaulted bank should not appear in positions."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=2000, target_2=2000)

        # Mark bank_2 as defaulted
        system.state.agents["bank_2"].defaulted = True
        system.state.defaulted_agent_ids.add("bank_2")

        positions = compute_reserve_positions(system, banking, {})
        assert "bank_1" in positions
        assert "bank_2" not in positions


# ---------------------------------------------------------------------------
# TestComputeLimitRates
# ---------------------------------------------------------------------------


class TestComputeLimitRates:
    """Tests for compute_limit_rates."""

    def _setup_corridor(self, banking, r_floor, r_ceiling):
        """Set corridor rates for all banks."""
        for bank_state in banking.banks.values():
            bank_state.pricing_params.reserve_remuneration_rate = r_floor
            bank_state.pricing_params.cb_borrowing_rate = r_ceiling

    def test_surplus_below_midpoint(self):
        """Surplus bank gets rate < M."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=1000, target_2=1000)
        self._setup_corridor(banking, Decimal("0.02"), Decimal("0.08"))

        positions = {"bank_1": 500}  # surplus
        rates = compute_limit_rates(positions, banking)

        M = Decimal("0.05")
        assert rates["bank_1"] < M

    def test_deficit_above_midpoint(self):
        """Deficit bank gets rate > M."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=1000, target_2=1000)
        self._setup_corridor(banking, Decimal("0.02"), Decimal("0.08"))

        positions = {"bank_1": -500}  # deficit
        rates = compute_limit_rates(positions, banking)

        M = Decimal("0.05")
        assert rates["bank_1"] > M

    def test_at_target_equals_midpoint(self):
        """Zero position -> rate == M."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=1000, target_2=1000)
        self._setup_corridor(banking, Decimal("0.02"), Decimal("0.08"))

        positions = {"bank_1": 0}
        rates = compute_limit_rates(positions, banking)

        M = Decimal("0.05")
        assert rates["bank_1"] == M

    def test_max_surplus_equals_floor(self):
        """normalized=+1 -> rate == r_floor."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=1000, target_2=1000)
        self._setup_corridor(banking, Decimal("0.02"), Decimal("0.08"))

        # position >= target -> normalized clamped to +1
        positions = {"bank_1": 2000}  # 2x target
        rates = compute_limit_rates(positions, banking)

        assert rates["bank_1"] == Decimal("0.02")

    def test_max_deficit_equals_ceiling(self):
        """normalized=-1 -> rate == r_ceiling."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=1000, target_2=1000)
        self._setup_corridor(banking, Decimal("0.02"), Decimal("0.08"))

        # position <= -target -> normalized clamped to -1
        positions = {"bank_1": -2000}  # -2x target
        rates = compute_limit_rates(positions, banking)

        assert rates["bank_1"] == Decimal("0.08")


# ---------------------------------------------------------------------------
# TestBuildOrderBook
# ---------------------------------------------------------------------------


class TestBuildOrderBook:
    """Tests for build_order_book."""

    def test_correct_sorting(self):
        """Asks ascending, bids descending."""
        positions = {"a": 100, "b": 200, "c": -150, "d": -50}
        rates = {
            "a": Decimal("0.04"),
            "b": Decimal("0.03"),
            "c": Decimal("0.07"),
            "d": Decimal("0.06"),
        }

        asks, bids = build_order_book(positions, rates)

        assert len(asks) == 2
        assert asks[0].bank_id == "b"  # cheapest ask first
        assert asks[1].bank_id == "a"
        assert len(bids) == 2
        assert bids[0].bank_id == "c"  # highest bid first
        assert bids[1].bank_id == "d"

    def test_zero_positions_excluded(self):
        """Banks with zero position don't appear in book."""
        positions = {"a": 100, "b": 0, "c": -50}
        rates = {
            "a": Decimal("0.04"),
            "b": Decimal("0.05"),
            "c": Decimal("0.06"),
        }

        asks, bids = build_order_book(positions, rates)

        assert len(asks) == 1
        assert asks[0].bank_id == "a"
        assert len(bids) == 1
        assert bids[0].bank_id == "c"


# ---------------------------------------------------------------------------
# TestClearAuction
# ---------------------------------------------------------------------------


class TestClearAuction:
    """Tests for clear_auction."""

    def test_full_match(self):
        """Supply == demand -> all filled, no unfilled."""
        asks = [InterbankOrder("lender", "lend", 100, Decimal("0.03"))]
        bids = [InterbankOrder("borrower", "borrow", 100, Decimal("0.07"))]

        result = clear_auction(asks, bids)

        assert result.total_volume == 100
        assert len(result.trades) == 1
        assert result.trades[0]["lender"] == "lender"
        assert result.trades[0]["borrower"] == "borrower"
        assert result.trades[0]["amount"] == 100
        assert result.clearing_rate == Decimal("0.05")  # midpoint
        assert result.unfilled_borrowers == []

    def test_partial_match(self):
        """Supply < demand -> some borrowers unfilled."""
        asks = [InterbankOrder("lender", "lend", 50, Decimal("0.03"))]
        bids = [InterbankOrder("borrower", "borrow", 100, Decimal("0.07"))]

        result = clear_auction(asks, bids)

        assert result.total_volume == 50
        assert len(result.unfilled_borrowers) == 1
        assert result.unfilled_borrowers[0] == ("borrower", 50)

    def test_no_match(self):
        """Ask > bid -> no trades."""
        asks = [InterbankOrder("lender", "lend", 100, Decimal("0.08"))]
        bids = [InterbankOrder("borrower", "borrow", 100, Decimal("0.03"))]

        result = clear_auction(asks, bids)

        assert result.total_volume == 0
        assert result.clearing_rate is None
        assert len(result.trades) == 0
        assert len(result.unfilled_borrowers) == 1

    def test_single_pair(self):
        """One lender, one borrower."""
        asks = [InterbankOrder("A", "lend", 200, Decimal("0.04"))]
        bids = [InterbankOrder("B", "borrow", 200, Decimal("0.06"))]

        result = clear_auction(asks, bids)

        assert result.total_volume == 200
        assert result.clearing_rate == Decimal("0.05")
        assert len(result.trades) == 1

    def test_multiple_banks(self):
        """Three lenders, two borrowers with varying sizes."""
        asks = [
            InterbankOrder("L1", "lend", 100, Decimal("0.03")),
            InterbankOrder("L2", "lend", 150, Decimal("0.04")),
            InterbankOrder("L3", "lend", 200, Decimal("0.05")),
        ]
        bids = [
            InterbankOrder("B1", "borrow", 200, Decimal("0.07")),
            InterbankOrder("B2", "borrow", 100, Decimal("0.06")),
        ]

        result = clear_auction(asks, bids)

        # Total demand = 300, total supply at prices <= 0.07 = 450
        # All demand should be filled
        assert result.total_volume == 300
        assert len(result.unfilled_borrowers) == 0

    def test_clearing_rate_midpoint(self):
        """r* = midpoint of marginal ask and bid."""
        asks = [InterbankOrder("L", "lend", 100, Decimal("0.03"))]
        bids = [InterbankOrder("B", "borrow", 100, Decimal("0.09"))]

        result = clear_auction(asks, bids)

        assert result.clearing_rate == Decimal("0.06")  # (0.03 + 0.09) / 2

    def test_empty_asks(self):
        """No lenders -> no trades, all borrowers unfilled."""
        bids = [InterbankOrder("B", "borrow", 100, Decimal("0.07"))]

        result = clear_auction([], bids)

        assert result.total_volume == 0
        assert result.clearing_rate is None
        assert len(result.unfilled_borrowers) == 1

    def test_empty_bids(self):
        """No borrowers -> no trades, no unfilled."""
        asks = [InterbankOrder("L", "lend", 100, Decimal("0.03"))]

        result = clear_auction(asks, [])

        assert result.total_volume == 0
        assert result.clearing_rate is None
        assert result.unfilled_borrowers == []


# ---------------------------------------------------------------------------
# TestRunInterbankAuction
# ---------------------------------------------------------------------------


class TestRunInterbankAuction:
    """Tests for run_interbank_auction (integration)."""

    def test_reserve_transfers(self):
        """Reserves should move from lender to borrower."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        r1_before = _get_bank_reserves(system, "bank_1")
        r2_before = _get_bank_reserves(system, "bank_2")

        # net_obligations empty: compute_bank_net_obligations not called yet
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})

        r1_after = _get_bank_reserves(system, "bank_1")
        r2_after = _get_bank_reserves(system, "bank_2")

        # bank_1 had surplus, bank_2 had deficit -> reserves should move
        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        if trade_events:
            total_transferred = sum(e["amount"] for e in trade_events)
            assert r1_after == r1_before - total_transferred
            assert r2_after == r2_before + total_transferred

    def test_overnight_maturity(self):
        """Loan maturity should be current_day + 1."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        run_interbank_auction(system, current_day=3, banking=banking, net_obligations={})

        for loan in banking.interbank_loans:
            assert loan.maturity_day == 4  # current_day + 1

    def test_events_emitted(self):
        """Should emit InterbankAuction + trade events."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})

        # Should always have the summary event
        summary_events = [e for e in events if e["kind"] == "InterbankAuction"]
        assert len(summary_events) == 1

        # Should have trade events
        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) >= 1

    def test_unfilled_events(self):
        """InterbankUnfilled event for unfilled borrowers."""
        system = _make_banking_system()
        # Give bank_2 a deficit but bank_1 no surplus (both at target)
        banking = _make_initialized_banking(system, target_1=5000, target_2=8000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})

        unfilled_events = [e for e in events if e["kind"] == "InterbankUnfilled"]
        # bank_2 has deficit (5000 - 8000 = -3000), bank_1 at target (0 surplus)
        # So bank_2 should be unfilled
        assert len(unfilled_events) >= 1
        assert unfilled_events[0]["borrower"] == "bank_2"

    def test_no_trades_when_balanced(self):
        """Balanced banks -> no auction trades."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=5000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})

        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) == 0


# ---------------------------------------------------------------------------
# TestComputeCombinedNets
# ---------------------------------------------------------------------------


class TestComputeCombinedNets:
    """Tests for compute_combined_nets."""

    def test_client_only(self):
        """No interbank obligations -> same as compute_intraday_nets."""
        system = _make_banking_system()
        nets = compute_combined_nets(system, day=0, interbank_obligations=[])
        # No ClientPayment events -> empty
        assert nets == {} or all(v == 0 for v in nets.values())

    def test_with_interbank_obligations(self):
        """Interbank repayment adds to netting."""
        system = _make_banking_system()

        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=1,
        )
        obligations = [("bank_2", "bank_1", 105, loan)]

        nets = compute_combined_nets(system, day=1, interbank_obligations=obligations)

        # bank_1 < bank_2 lexically, so nets[("bank_1", "bank_2")] convention:
        # bank_2 owes bank_1 -> since bank_1 < bank_2, this is negative
        # (bank_1 is owed, not owing)
        assert ("bank_1", "bank_2") in nets
        assert nets[("bank_1", "bank_2")] == -105  # bank_2 owes bank_1

    def test_netting_offset(self):
        """Client flow and interbank flow in opposite directions cancel."""
        system = _make_banking_system()

        # Simulate a ClientPayment event: bank_1 owes bank_2 some amount
        system.state.events_by_day.setdefault(1, []).append({
            "kind": "ClientPayment",
            "payer_bank": "bank_1",
            "payee_bank": "bank_2",
            "amount": 105,
            "day": 1,
        })

        # Interbank obligation: bank_2 owes bank_1 105 (repayment)
        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=1,
        )
        obligations = [("bank_2", "bank_1", 105, loan)]

        nets = compute_combined_nets(system, day=1, interbank_obligations=obligations)

        # bank_1 owes bank_2 105 (client) + bank_2 owes bank_1 105 (interbank)
        # Net should be 0
        assert nets.get(("bank_1", "bank_2"), 0) == 0


# ---------------------------------------------------------------------------
# TestComputeBankNetObligations
# ---------------------------------------------------------------------------


class TestComputeBankNetObligations:
    """Tests for compute_bank_net_obligations."""

    def test_two_banks(self):
        """Simple bilateral: one positive, one negative."""
        nets = {("A", "B"): 100}  # A owes B 100
        obligations = compute_bank_net_obligations(nets)

        assert obligations["A"] == 100   # owes
        assert obligations["B"] == -100  # is owed

    def test_three_banks(self):
        """Multilateral netting."""
        nets = {
            ("A", "B"): 100,   # A owes B 100
            ("A", "C"): -50,   # C owes A 50
            ("B", "C"): 200,   # B owes C 200
        }
        obligations = compute_bank_net_obligations(nets)

        # A: owes B 100, owed by C 50 -> net = 100 - 50 = 50
        assert obligations["A"] == 50
        # B: owed by A 100, owes C 200 -> net = -100 + 200 = 100
        assert obligations["B"] == 100
        # C: owes A 50, owed by B 200 -> net = 50 - 200 = -150
        assert obligations["C"] == -150


# ---------------------------------------------------------------------------
# TestFinalizeInterbankRepayments
# ---------------------------------------------------------------------------


class TestFinalizeInterbankRepayments:
    """Tests for finalize_interbank_repayments."""

    def test_removes_matured_loans(self):
        """Matured loans are removed from the book."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)

        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=1,
        )
        banking.interbank_loans.append(loan)

        obligations = [("bank_2", "bank_1", 105, loan)]

        events = finalize_interbank_repayments(1, banking, obligations)

        assert len(banking.interbank_loans) == 0
        assert len(events) == 1
        assert events[0]["kind"] == "InterbankRepaid"

    def test_keeps_non_matured(self):
        """Non-matured loans stay in the book."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)

        loan_mature = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=1,
        )
        loan_future = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=200,
            rate=Decimal("0.03"),
            issuance_day=1,
            maturity_day=2,
        )
        banking.interbank_loans.extend([loan_mature, loan_future])

        obligations = [("bank_2", "bank_1", 105, loan_mature)]

        finalize_interbank_repayments(1, banking, obligations)

        assert len(banking.interbank_loans) == 1
        assert banking.interbank_loans[0] is loan_future

    def test_event_fields(self):
        """InterbankRepaid event has correct fields."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)

        loan = InterbankLoan(
            lender_bank="bank_1",
            borrower_bank="bank_2",
            amount=100,
            rate=Decimal("0.05"),
            issuance_day=0,
            maturity_day=2,
        )
        banking.interbank_loans.append(loan)

        obligations = [("bank_2", "bank_1", 105, loan)]
        events = finalize_interbank_repayments(2, banking, obligations)

        e = events[0]
        assert e["kind"] == "InterbankRepaid"
        assert e["day"] == 2
        assert e["lender"] == "bank_1"
        assert e["borrower"] == "bank_2"
        assert e["principal"] == 100
        assert e["repayment"] == 105
        assert e["interest"] == 5


# ---------------------------------------------------------------------------
# TestOvernightRollover
# ---------------------------------------------------------------------------


class TestOvernightRollover:
    """Test that day 0 auction -> day 1 repayment works end-to-end."""

    def test_auction_then_repayment(self):
        """Day 0: auction creates overnight loan. Day 1: loan identified as obligation."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        # Day 0: run auction
        events_d0 = run_interbank_auction(
            system, current_day=0, banking=banking, net_obligations={},
        )
        trade_events = [e for e in events_d0 if e["kind"] == "InterbankAuctionTrade"]

        if trade_events:
            # Day 1: identify maturing obligations
            obligations = compute_interbank_obligations(1, banking)
            assert len(obligations) == len(banking.interbank_loans)

            for borrower, lender, repayment, loan in obligations:
                assert loan.maturity_day == 1
                assert repayment == loan.repayment_amount
