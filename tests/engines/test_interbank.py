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

from bilancio.decision.profiles import BankProfile
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.firm import Firm
from bilancio.domain.policy import PolicyEngine
from bilancio.engines.banking_subsystem import (
    InterbankLoan,
    _get_bank_reserves,
    initialize_banking_subsystem,
)
from bilancio.engines.clearing import (
    compute_bank_net_obligations,
    compute_combined_nets,
)
from bilancio.engines.interbank import (
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


def _make_n_bank_system(
    n_banks: int, reserves_per_bank: int = 5000, deposit_per_firm: int = 1000,
) -> System:
    """Create a system with CB + n banks + n firms."""
    system = System(policy=PolicyEngine.default())
    cb = CentralBank(id="cb", name="Central Bank", kind="central_bank")
    system.add_agent(cb)
    for i in range(1, n_banks + 1):
        bank = Bank(id=f"bank_{i}", name=f"Bank {i}", kind="bank")
        firm = Firm(id=f"H_{i}", name=f"Firm {i}", kind="firm")
        system.add_agent(bank)
        system.add_agent(firm)
        system.mint_cash(to_agent_id=f"H_{i}", amount=deposit_per_firm)
        deposit_cash(system, f"H_{i}", f"bank_{i}", deposit_per_firm)
        system.mint_reserves(to_bank_id=f"bank_{i}", amount=reserves_per_bank)
    return system


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

    def test_summary_includes_market_state(self):
        """Auction summary should include positions and the pre-clear order book."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})

        summary = next(e for e in events if e["kind"] == "InterbankAuction")
        market_state = summary["market_state"]
        positions = {row["bank_id"]: row for row in market_state["positions"]}

        assert positions["bank_1"]["position"] > 0
        assert positions["bank_1"]["side"] == "lend"
        assert positions["bank_2"]["position"] < 0
        assert positions["bank_2"]["side"] == "borrow"
        assert market_state["lender_asks"][0]["bank_id"] == "bank_1"
        assert market_state["borrower_bids"][0]["bank_id"] == "bank_2"

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

    def test_summary_counts_only_executed_trades(self, monkeypatch):
        """Summary n_trades must exclude trades that failed during reserve transfer."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)

        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")

        # Force transfer failure so auction has proposed trades but zero executions.
        def _raise_transfer(_from: str, _to: str, _amount: int) -> None:
            raise ValueError("forced transfer failure")

        monkeypatch.setattr(system, "transfer_reserves", _raise_transfer)
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})

        summary = next(e for e in events if e["kind"] == "InterbankAuction")
        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        unfilled_events = [e for e in events if e["kind"] == "InterbankUnfilled"]

        assert len(trade_events) == 0
        assert summary["n_trades"] == 0
        assert summary["total_volume"] == 0
        assert len(unfilled_events) >= 1


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

        events = finalize_interbank_repayments(system, 1, banking, obligations)

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

        finalize_interbank_repayments(system, 1, banking, obligations)

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
        events = finalize_interbank_repayments(system, 2, banking, obligations)

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

            for _borrower, _lender, repayment, loan in obligations:
                assert loan.maturity_day == 1
                assert repayment == loan.repayment_amount


# ---------------------------------------------------------------------------
# TestCorridorStressSensitivity
# ---------------------------------------------------------------------------


class TestCorridorStressSensitivity:
    """Tests that corridor rates respond to liquidity stress (kappa)."""

    def _make_corridor_system(self, kappa):
        """Set up a 2-bank system with corridor derived from kappa."""
        system = _make_banking_system()
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal(str(kappa)), maturity_days=10)
        return system, banking, profile

    def test_low_kappa_wider_corridor(self):
        """Low kappa -> wider corridor than high kappa."""
        _, banking_low, _ = self._make_corridor_system("0.25")
        _, banking_high, _ = self._make_corridor_system("2.0")
        low_b1 = banking_low.banks["bank_1"].pricing_params
        high_b1 = banking_high.banks["bank_1"].pricing_params
        omega_low = low_b1.cb_borrowing_rate - low_b1.reserve_remuneration_rate
        omega_high = high_b1.cb_borrowing_rate - high_b1.reserve_remuneration_rate
        assert omega_low > omega_high

    def test_high_kappa_narrow_corridor(self):
        """kappa >= 1 -> corridor width = omega_base only."""
        _, banking, profile = self._make_corridor_system("2.0")
        b1 = banking.banks["bank_1"].pricing_params
        omega = b1.cb_borrowing_rate - b1.reserve_remuneration_rate
        assert omega == profile.omega_base

    def test_limit_rates_within_corridor(self):
        """All computed limit rates stay within [r_floor, r_ceiling]."""
        system, banking, _ = self._make_corridor_system("0.25")
        system.transfer_reserves("bank_2", "bank_1", 3000)
        banking.banks["bank_1"].pricing_params.reserve_target = 2000
        banking.banks["bank_2"].pricing_params.reserve_target = 5000
        positions = compute_reserve_positions(system, banking, {})
        rates = compute_limit_rates(positions, banking)
        for bank_id, rate in rates.items():
            pp = banking.banks[bank_id].pricing_params
            assert rate >= pp.reserve_remuneration_rate
            assert rate <= pp.cb_borrowing_rate

    def test_kappa_one_corridor_equals_base(self):
        """kappa=1 -> stress_factor=0 -> corridor = base values."""
        _, banking, profile = self._make_corridor_system("1.0")
        b1 = banking.banks["bank_1"].pricing_params
        M = (b1.cb_borrowing_rate + b1.reserve_remuneration_rate) / 2
        expected_M = profile.r_base  # stress_factor=0
        assert M == expected_M


# ---------------------------------------------------------------------------
# TestLargeImbalances
# ---------------------------------------------------------------------------


class TestLargeImbalances:
    """Tests for extreme reserve positions (clamping to +/-1)."""

    def _make_system_with_targets(self, target):
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=target, target_2=target)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        return system, banking

    def test_huge_surplus_clamps_to_floor(self):
        """Position 10x target -> clamped to +1 -> rate = r_floor."""
        system, banking = self._make_system_with_targets(500)
        # bank_1 has 5000 reserves, target 500 -> position 4500 = 9x target
        positions = {"bank_1": 5000}  # 10x target
        rates = compute_limit_rates(positions, banking)
        assert rates["bank_1"] == Decimal("0.02")

    def test_huge_deficit_clamps_to_ceiling(self):
        """Position -10x target -> clamped to -1 -> rate = r_ceiling."""
        system, banking = self._make_system_with_targets(500)
        positions = {"bank_1": -5000}  # -10x target
        rates = compute_limit_rates(positions, banking)
        assert rates["bank_1"] == Decimal("0.08")

    def test_fractional_position_not_clamped(self):
        """Position at 0.5x target -> normalized 0.5 -> rate between M and floor."""
        system, banking = self._make_system_with_targets(1000)
        positions = {"bank_1": 500}  # 0.5x target
        rates = compute_limit_rates(positions, banking)
        M = Decimal("0.05")
        r_floor = Decimal("0.02")
        assert r_floor < rates["bank_1"] < M

    def test_boundary_at_exact_target(self):
        """Position exactly equal to target -> normalized=+1 -> rate = r_floor."""
        system, banking = self._make_system_with_targets(1000)
        positions = {"bank_1": 1000}  # exactly target
        rates = compute_limit_rates(positions, banking)
        assert rates["bank_1"] == Decimal("0.02")


# ---------------------------------------------------------------------------
# TestPartialTransferFailure
# ---------------------------------------------------------------------------


class TestPartialTransferFailure:
    """Tests for failed reserve transfers during auction execution."""

    def _setup_auction_system(self):
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)
        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        return system, banking

    def test_failed_transfer_produces_unfilled(self, monkeypatch):
        """Monkeypatched transfer -> trade fails -> unfilled event emitted."""
        system, banking = self._setup_auction_system()

        def _fail(*args, **kwargs):
            raise ValueError("transfer blocked")

        monkeypatch.setattr(system, "transfer_reserves", _fail)
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        unfilled = [e for e in events if e["kind"] == "InterbankUnfilled"]
        assert len(unfilled) >= 1

    def test_failed_transfer_no_loan_created(self, monkeypatch):
        """Failed transfer means no InterbankLoan recorded."""
        system, banking = self._setup_auction_system()

        def _fail(*args, **kwargs):
            raise ValueError("transfer blocked")

        monkeypatch.setattr(system, "transfer_reserves", _fail)
        run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        assert len(banking.interbank_loans) == 0

    def test_reserve_conservation_on_failure(self, monkeypatch):
        """Reserves unchanged when all transfers fail."""
        system, banking = self._setup_auction_system()
        r1_before = _get_bank_reserves(system, "bank_1")
        r2_before = _get_bank_reserves(system, "bank_2")

        def _fail(*args, **kwargs):
            raise ValueError("transfer blocked")

        monkeypatch.setattr(system, "transfer_reserves", _fail)
        run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        assert _get_bank_reserves(system, "bank_1") == r1_before
        assert _get_bank_reserves(system, "bank_2") == r2_before


# ---------------------------------------------------------------------------
# TestReserveDepletion
# ---------------------------------------------------------------------------


class TestReserveDepletion:
    """Tests for lender exhaustion with multiple borrowers."""

    def test_lender_cant_fill_all_borrowers(self):
        """One lender with 100 surplus, two borrowers needing 80 each -> partial fill."""
        asks = [InterbankOrder("L1", "lend", 100, Decimal("0.03"))]
        bids = [
            InterbankOrder("B1", "borrow", 80, Decimal("0.07")),
            InterbankOrder("B2", "borrow", 80, Decimal("0.06")),
        ]
        result = clear_auction(asks, bids)
        assert result.total_volume == 100  # Lender's full capacity
        assert len(result.unfilled_borrowers) >= 1

    def test_progressive_exhaustion(self):
        """Lender fills first borrower fully, second partially."""
        asks = [InterbankOrder("L1", "lend", 120, Decimal("0.03"))]
        bids = [
            InterbankOrder("B1", "borrow", 80, Decimal("0.07")),
            InterbankOrder("B2", "borrow", 80, Decimal("0.06")),
        ]
        result = clear_auction(asks, bids)
        assert result.total_volume == 120
        # B1 filled (80), B2 partially (40 of 80)
        assert len(result.trades) == 2
        amounts = [t["amount"] for t in result.trades]
        assert 80 in amounts
        assert 40 in amounts

    def test_just_enough_no_unfilled(self):
        """Supply exactly meets demand -> no unfilled."""
        asks = [InterbankOrder("L1", "lend", 160, Decimal("0.03"))]
        bids = [
            InterbankOrder("B1", "borrow", 80, Decimal("0.07")),
            InterbankOrder("B2", "borrow", 80, Decimal("0.06")),
        ]
        result = clear_auction(asks, bids)
        assert result.total_volume == 160
        assert len(result.unfilled_borrowers) == 0


# ---------------------------------------------------------------------------
# TestCircularObligations
# ---------------------------------------------------------------------------


class TestCircularObligations:
    """Tests for circular client payment netting and interbank overlay."""

    def test_circular_client_payments_net_to_zero(self):
        """A->B, B->C, C->A each 100 -> bilateral nets cancel -> positions at target."""
        system = _make_n_bank_system(3, reserves_per_bank=5000, deposit_per_firm=1000)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("1.0"), maturity_days=10)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Circular: each bank owes the next 100 in reserves
        # With equal flows, net obligations for each bank = 0
        net_obligations = {"bank_1": 0, "bank_2": 0, "bank_3": 0}
        positions = compute_reserve_positions(system, banking, net_obligations)
        # All at same reserves and target -> all positions equal
        assert positions["bank_1"] == positions["bank_2"] == positions["bank_3"]

    def test_interbank_overlay_on_netting(self):
        """Maturing interbank loan creates additional obligation."""
        system = _make_n_bank_system(3, reserves_per_bank=5000, deposit_per_firm=1000)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("1.0"), maturity_days=10)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # bank_2 borrowed from bank_1 yesterday, owes repayment today
        net_obligations = {"bank_1": -100, "bank_2": 100, "bank_3": 0}
        positions = compute_reserve_positions(system, banking, net_obligations)
        # bank_1 is owed 100 -> position higher
        # bank_2 owes 100 -> position lower
        assert positions["bank_1"] > positions["bank_3"]
        assert positions["bank_2"] < positions["bank_3"]

    def test_full_auction_from_circular_positions(self):
        """Full auction pipeline with 3 banks at different positions."""
        system = _make_n_bank_system(3, reserves_per_bank=5000, deposit_per_firm=1000)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("1.0"), maturity_days=10)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Create asymmetry: transfer reserves
        system.transfer_reserves("bank_3", "bank_1", 2000)
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        summary = [e for e in events if e["kind"] == "InterbankAuction"]
        assert len(summary) == 1


# ---------------------------------------------------------------------------
# TestRateBoundaryConditions
# ---------------------------------------------------------------------------


class TestRateBoundaryConditions:
    """Tests for normalized position boundary values in limit rate calculation."""

    def _make_corridor_banking(self, target=1000):
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=target, target_2=target)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        return system, banking

    def test_exactly_plus_target_gives_floor(self):
        """x = +R* -> normalized = +1 -> rate = r_floor."""
        _, banking = self._make_corridor_banking(1000)
        rates = compute_limit_rates({"bank_1": 1000}, banking)
        assert rates["bank_1"] == Decimal("0.02")

    def test_exactly_minus_target_gives_ceiling(self):
        """x = -R* -> normalized = -1 -> rate = r_ceiling."""
        _, banking = self._make_corridor_banking(1000)
        rates = compute_limit_rates({"bank_1": -1000}, banking)
        assert rates["bank_1"] == Decimal("0.08")

    def test_slightly_inside_target_not_clamped(self):
        """x = 0.99 * R* -> normalized ~ 0.99 -> rate slightly above floor."""
        _, banking = self._make_corridor_banking(1000)
        rates = compute_limit_rates({"bank_1": 990}, banking)
        # Should be slightly above floor (0.02) but below midpoint (0.05)
        assert Decimal("0.02") < rates["bank_1"] < Decimal("0.05")

    def test_zero_target_fallback(self):
        """Zero target -> falls back to target=1 (logger warning)."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=0, target_2=1000)
        banking.banks["bank_1"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_1"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        banking.banks["bank_2"].pricing_params.reserve_remuneration_rate = Decimal("0.02")
        banking.banks["bank_2"].pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Should not crash - fallback to target=1
        rates = compute_limit_rates({"bank_1": 100}, banking)
        assert "bank_1" in rates
        # With target=1 and position=100, normalized clamped to +1 -> r_floor
        assert rates["bank_1"] == Decimal("0.02")


# ---------------------------------------------------------------------------
# TestManyBankAuction
# ---------------------------------------------------------------------------


class TestManyBankAuction:
    """Tests for auction with many banks (10+)."""

    def test_20_bank_auction_runs(self):
        """20-bank system can run full auction pipeline."""
        system = _make_n_bank_system(20, reserves_per_bank=5000, deposit_per_firm=1000)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("1.0"), maturity_days=10)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Create imbalances: odd banks get extra, even banks lose
        for i in range(1, 21):
            if i % 2 == 0 and i > 2:
                system.transfer_reserves(f"bank_{i}", f"bank_{i-1}", 1000)
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        summary = next(e for e in events if e["kind"] == "InterbankAuction")
        assert summary["n_trades"] >= 0  # at least didn't crash

    def test_asymmetric_large_auction(self):
        """One rich bank, many poor banks."""
        system = _make_n_bank_system(10, reserves_per_bank=1000, deposit_per_firm=500)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("0.5"), maturity_days=10)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Give bank_1 massive reserves
        system.mint_reserves(to_bank_id="bank_1", amount=50000)
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        summary = next(e for e in events if e["kind"] == "InterbankAuction")
        assert isinstance(summary["total_volume"], int)

    def test_reserve_conservation_many_banks(self):
        """Total reserves across all banks unchanged after auction."""
        system = _make_n_bank_system(10, reserves_per_bank=5000, deposit_per_firm=1000)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("1.0"), maturity_days=10)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Create imbalances
        system.transfer_reserves("bank_1", "bank_5", 2000)
        system.transfer_reserves("bank_2", "bank_6", 2000)
        total_before = sum(_get_bank_reserves(system, f"bank_{i}") for i in range(1, 11))
        run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        total_after = sum(_get_bank_reserves(system, f"bank_{i}") for i in range(1, 11))
        assert total_before == total_after


# ---------------------------------------------------------------------------
# TestDayWrapAround
# ---------------------------------------------------------------------------


class TestDayWrapAround:
    """Tests for loan maturity boundary conditions."""

    def test_loan_beyond_simulation_not_in_obligations(self):
        """Loan maturing on day 100 not due on day 5."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        loan = InterbankLoan(
            lender_bank="bank_1", borrower_bank="bank_2",
            amount=100, rate=Decimal("0.05"), issuance_day=0, maturity_day=100,
        )
        banking.interbank_loans.append(loan)
        obligations = compute_interbank_obligations(5, banking)
        assert len(obligations) == 0

    def test_exact_maturity_day_included(self):
        """Loan maturing on day 5 IS included on day 5."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        loan = InterbankLoan(
            lender_bank="bank_1", borrower_bank="bank_2",
            amount=100, rate=Decimal("0.05"), issuance_day=4, maturity_day=5,
        )
        banking.interbank_loans.append(loan)
        obligations = compute_interbank_obligations(5, banking)
        assert len(obligations) == 1
        assert obligations[0][3] is loan

    def test_empty_obligations_no_crash(self):
        """No loans at all -> empty obligations, no crash."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system)
        obligations = compute_interbank_obligations(0, banking)
        assert obligations == []


# ---------------------------------------------------------------------------
# TestDefaultDuringAuction
# ---------------------------------------------------------------------------


class TestDefaultDuringAuction:
    """Tests for bank default during/before auction."""

    def test_defaulted_bank_excluded_from_auction(self):
        """Defaulted bank doesn't participate in auction."""
        system = _make_banking_system()
        system.transfer_reserves("bank_2", "bank_1", 3000)
        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Default bank_2 (the deficit bank)
        system.state.agents["bank_2"].defaulted = True
        system.state.defaulted_agent_ids.add("bank_2")
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        # With only bank_1 (surplus) and bank_2 defaulted, no trade possible
        assert len(trade_events) == 0

    def test_transfer_fail_mid_auction_handled(self, monkeypatch):
        """If transfer fails mid-auction, borrower goes to unfilled."""
        system = _make_n_bank_system(4, reserves_per_bank=5000, deposit_per_firm=1000)
        profile = BankProfile()
        banking = initialize_banking_subsystem(system, profile, Decimal("1.0"), maturity_days=10)
        # Set targets to create surplus/deficit banks
        banking.banks["bank_1"].pricing_params.reserve_target = 2000
        banking.banks["bank_2"].pricing_params.reserve_target = 2000
        banking.banks["bank_3"].pricing_params.reserve_target = 8000
        banking.banks["bank_4"].pricing_params.reserve_target = 8000
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")

        call_count = 0
        original_transfer = system.transfer_reserves

        def _sometimes_fail(from_id, to_id, amount):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first transfer fails")
            return original_transfer(from_id, to_id, amount)

        monkeypatch.setattr(system, "transfer_reserves", _sometimes_fail)
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        unfilled = [e for e in events if e["kind"] == "InterbankUnfilled"]
        # At least one unfilled due to the first transfer failing
        assert len(unfilled) >= 1

    def test_all_defaulted_no_crash(self):
        """All banks defaulted -> empty auction, no crash."""
        system = _make_banking_system()
        banking = _make_initialized_banking(system, target_1=2000, target_2=5000)
        for bs in banking.banks.values():
            bs.pricing_params.reserve_remuneration_rate = Decimal("0.02")
            bs.pricing_params.cb_borrowing_rate = Decimal("0.08")
        # Default both banks
        system.state.agents["bank_1"].defaulted = True
        system.state.agents["bank_2"].defaulted = True
        system.state.defaulted_agent_ids.update({"bank_1", "bank_2"})
        events = run_interbank_auction(system, current_day=0, banking=banking, net_obligations={})
        # Should return empty events (no positions -> early return)
        trade_events = [e for e in events if e["kind"] == "InterbankAuctionTrade"]
        assert len(trade_events) == 0
