"""Tests for dealer metrics (Section 8 of the functional dealer specification).

Tests verify the metrics tracking infrastructure for:
- Trade log with pre/post state (8.1)
- Dealer P&L and profitability (8.2)
- Trader investment returns and liquidity use (8.3)
- Repayment-priority diagnostics (8.4)
"""

from decimal import Decimal

import pytest

from bilancio.dealer.metrics import (
    DealerSnapshot,
    RunMetrics,
    TraderSnapshot,
)
from bilancio.dealer.models import DEFAULT_BUCKETS
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.domain.agents import CentralBank, Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.dealer_integration import (
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
)
from bilancio.engines.system import System


def create_test_system_with_ring():
    """Create a test system with a Kalecki ring of debts.

    Creates 5 households with:
    - 100 cash each
    - Ring of debts: h0->h1->h2->h3->h4->h0 (50 each, due day 5)
    """
    sys = System()

    # Add central bank
    cb = CentralBank(id="CB", name="Central Bank", kind="central_bank")
    sys.add_agent(cb)

    # Create households
    households = []
    for i in range(5):
        h = Household(id=f"h{i}", name=f"Household {i}", kind="household")
        sys.add_agent(h)
        sys.mint_cash(h.id, 100)
        households.append(h)

    # Create ring of debt (each owes 50 to the next)
    for i in range(5):
        creditor_id = households[(i + 1) % 5].id
        debtor_id = households[i].id

        p_id = sys.new_contract_id("P")
        payable = Payable(
            id=p_id,
            kind=InstrumentKind.PAYABLE,
            amount=50,
            denom="USD",
            asset_holder_id=creditor_id,
            liability_issuer_id=debtor_id,
            due_day=5,
        )
        sys.add_contract(payable)

    return sys, households


def create_dealer_config():
    """Create a test dealer configuration."""
    return DealerRingConfig(
        ticket_size=Decimal(1),
        buckets=list(DEFAULT_BUCKETS),
        dealer_share=Decimal("0.25"),
        vbt_share=Decimal("0.50"),
        vbt_anchors={
            "short": (Decimal("1.0"), Decimal("0.20")),
            "mid": (Decimal("1.0"), Decimal("0.30")),
            "long": (Decimal("1.0"), Decimal("0.40")),
        },
        phi_M=Decimal("0.1"),
        phi_O=Decimal("0.1"),
        clip_nonneg_B=True,
        seed=42,
    )


class TestMetricsInitialization:
    """Test metrics are properly initialized."""

    def test_initial_equity_captured(self):
        """Test that initial equity is captured for each bucket."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Should have initial equity for each bucket
        assert len(subsystem.metrics.initial_equity_by_bucket) == 3
        assert "short" in subsystem.metrics.initial_equity_by_bucket
        assert "mid" in subsystem.metrics.initial_equity_by_bucket
        assert "long" in subsystem.metrics.initial_equity_by_bucket

        # Initial equity should be positive (dealer has cash)
        for bucket_id, equity in subsystem.metrics.initial_equity_by_bucket.items():
            assert equity > 0, f"Initial equity for {bucket_id} should be positive"

    def test_metrics_object_created(self):
        """Test that RunMetrics object is created."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        assert subsystem.metrics is not None
        assert isinstance(subsystem.metrics, RunMetrics)
        assert len(subsystem.metrics.trades) == 0  # No trades yet
        assert len(subsystem.metrics.dealer_snapshots) == 0  # No snapshots yet


class TestTradeLogging:
    """Test trade log captures (Section 8.1)."""

    def test_dealer_snapshots_captured_each_day(self):
        """Test dealer snapshots are captured each trading day."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading for 3 days
        for day in range(1, 4):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        # Should have 3 snapshots per bucket (3 buckets * 3 days = 9)
        assert len(subsystem.metrics.dealer_snapshots) == 9

        # Each snapshot should have correct structure
        for snapshot in subsystem.metrics.dealer_snapshots:
            assert isinstance(snapshot, DealerSnapshot)
            assert snapshot.day >= 1
            assert snapshot.bucket in ["short", "mid", "long"]
            assert snapshot.inventory >= 0
            assert snapshot.cash >= 0

    def test_trader_snapshots_captured_each_day(self):
        """Test trader snapshots are captured each trading day."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading for 2 days
        for day in range(1, 3):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        # Should have snapshots for each trader each day (5 traders * 2 days = 10)
        assert len(subsystem.metrics.trader_snapshots) == 10

        # Each snapshot should have correct structure
        for snapshot in subsystem.metrics.trader_snapshots:
            assert isinstance(snapshot, TraderSnapshot)
            assert snapshot.trader_id.startswith("h")
            assert snapshot.cash >= 0


class TestDealerProfitability:
    """Test dealer profitability metrics (Section 8.2)."""

    def test_pnl_computation(self):
        """Test P&L computation after trading."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run several trading days
        for day in range(1, 6):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        # Get P&L by bucket
        pnl = subsystem.metrics.dealer_pnl_by_bucket()

        assert isinstance(pnl, dict)
        assert "short" in pnl or "mid" in pnl or "long" in pnl

    def test_return_computation(self):
        """Test return computation after trading."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading
        for day in range(1, 6):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        # Get returns by bucket
        returns = subsystem.metrics.dealer_return_by_bucket()

        assert isinstance(returns, dict)
        # Returns should be in reasonable range (-1 to +1)
        for bucket_id, ret in returns.items():
            assert ret >= -1 and ret <= 1, f"Return for {bucket_id} should be reasonable"

    def test_summary_includes_profitability(self):
        """Test summary includes profitability metrics."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        summary = subsystem.metrics.summary()

        assert "dealer_total_pnl" in summary
        assert "dealer_total_return" in summary
        assert "dealer_profitable" in summary
        assert "dealer_pnl_by_bucket" in summary


class TestSafetyMargin:
    """Test repayment-priority diagnostics (Section 8.4)."""

    def test_safety_margin_computation(self):
        """Test safety margin computation for a trader."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading to generate snapshots
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        # Check trader snapshots have safety margins
        for snapshot in subsystem.metrics.trader_snapshots:
            # Safety margin should be computed
            assert snapshot.safety_margin is not None

            # For our setup: cash=100, obligations=50
            # Safety margin = cash + saleable_value - obligations
            # Should be positive initially since cash > obligations
            assert snapshot.safety_margin > Decimal(-200)  # Reasonable lower bound

    def test_summary_includes_safety_metrics(self):
        """Test summary includes safety margin metrics."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        summary = subsystem.metrics.summary()

        assert "unsafe_buy_count" in summary
        assert "fraction_unsafe_buys" in summary


class TestLiquidityMetrics:
    """Test liquidity-driven sales metrics (Section 8.3)."""

    def test_liquidity_driven_flag_tracked(self):
        """Test that liquidity-driven flag is tracked on trades."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run multiple days to potentially generate trades
        for day in range(1, 6):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        # If there are trades, check they have the flag
        for trade in subsystem.metrics.trades:
            assert hasattr(trade, "is_liquidity_driven")

    def test_summary_includes_liquidity_metrics(self):
        """Test summary includes liquidity metrics."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        summary = subsystem.metrics.summary()

        assert "liquidity_driven_sales" in summary
        assert "rescue_events" in summary
        assert "total_trades" in summary


class TestSummaryStatistics:
    """Test experiment-level summary statistics (Section 8.5)."""

    def test_summary_completeness(self):
        """Test that summary includes all required metrics."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run a few days
        for day in range(1, 4):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        summary = subsystem.metrics.summary()

        # Section 8.2: Dealer profitability
        assert "dealer_total_pnl" in summary
        assert "dealer_total_return" in summary
        assert "dealer_profitable" in summary
        assert "dealer_pnl_by_bucket" in summary
        assert "dealer_return_by_bucket" in summary
        assert "spread_income_total" in summary
        assert "interior_trades" in summary
        assert "passthrough_trades" in summary

        # Section 8.3: Trader returns
        assert "mean_trader_return" in summary
        assert "fraction_profitable_traders" in summary
        assert "liquidity_driven_sales" in summary
        assert "rescue_events" in summary

        # Section 8.4: Repayment priority
        assert "unsafe_buy_count" in summary
        assert "fraction_unsafe_buys" in summary
        assert "mean_margin_at_default" in summary

        # Trade counts
        assert "total_trades" in summary
        assert "total_sell_trades" in summary
        assert "total_buy_trades" in summary

    def test_summary_values_valid(self):
        """Test that summary values are valid types."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        summary = subsystem.metrics.summary()

        # All values should be JSON-serializable
        import json

        try:
            json.dumps(summary)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Summary should be JSON serializable: {e}")


class TestMarkToMidEquity:
    """Test mark-to-mid equity calculation."""

    def test_equity_calculation(self):
        """Test mark-to-mid equity is computed correctly."""
        snapshot = DealerSnapshot(
            day=1,
            bucket="short",
            inventory=10,
            cash=Decimal(100),
            bid=Decimal("0.9"),
            ask=Decimal("1.1"),
            midline=Decimal(1),
            vbt_mid=Decimal(1),
            vbt_spread=Decimal("0.2"),
            ticket_size=Decimal(1),
        )

        # E = C + M * a * S = 100 + 1 * 10 * 1 = 110
        assert snapshot.mark_to_mid_equity == Decimal(110)

    def test_equity_with_zero_inventory(self):
        """Test equity equals cash when no inventory."""
        snapshot = DealerSnapshot(
            day=1,
            bucket="short",
            inventory=0,
            cash=Decimal(100),
            bid=Decimal("0.9"),
            ask=Decimal("1.1"),
            midline=Decimal(1),
            vbt_mid=Decimal(1),
            vbt_spread=Decimal("0.2"),
            ticket_size=Decimal(1),
        )

        # E = C + M * a * S = 100 + 1 * 0 * 1 = 100
        assert snapshot.mark_to_mid_equity == Decimal(100)


class TestDealerPremiumMetrics:
    """Test dealer and VBT premium/discount metrics (Plan 020 - Phase A.3)."""

    @pytest.mark.parametrize(
        "midline,vbt_mid,bid,ask,expected_dealer_premium,expected_dealer_pct,expected_vbt_premium,expected_vbt_pct",
        [
            # At par: premium is 0
            (
                Decimal("1.0"),
                Decimal("1.0"),
                Decimal("0.95"),
                Decimal("1.05"),
                Decimal(0),
                Decimal(0),
                Decimal(0),
                Decimal(0),
            ),
            # Above par: positive premium
            (
                Decimal("1.05"),
                Decimal("1.02"),
                Decimal("1.00"),
                Decimal("1.10"),
                Decimal("0.05"),
                Decimal("5"),
                Decimal("0.02"),
                Decimal("2"),
            ),
            # Below par: negative premium (discount)
            (
                Decimal("0.85"),
                Decimal("0.90"),
                Decimal("0.80"),
                Decimal("0.90"),
                Decimal("-0.15"),
                Decimal("-15"),
                Decimal("-0.10"),
                Decimal("-10"),
            ),
        ],
        ids=["at_par", "above_par", "below_par"],
    )
    def test_dealer_premium_vs_face(
        self,
        midline,
        vbt_mid,
        bid,
        ask,
        expected_dealer_premium,
        expected_dealer_pct,
        expected_vbt_premium,
        expected_vbt_pct,
    ):
        """Test dealer/VBT premium relative to face value."""
        snapshot = DealerSnapshot(
            day=1,
            bucket="short",
            inventory=0,
            cash=Decimal(100),
            bid=bid,
            ask=ask,
            midline=midline,
            vbt_mid=vbt_mid,
            vbt_spread=Decimal("0.10"),
            ticket_size=Decimal(1),
        )

        assert snapshot.dealer_premium_vs_face == expected_dealer_premium
        assert snapshot.dealer_premium_pct == expected_dealer_pct
        assert snapshot.vbt_premium_vs_face == expected_vbt_premium
        assert snapshot.vbt_premium_pct == expected_vbt_pct

    def test_to_dict_includes_premium_fields(self):
        """Test that to_dict includes premium fields."""
        snapshot = DealerSnapshot(
            day=1,
            bucket="short",
            inventory=0,
            cash=Decimal(100),
            bid=Decimal("0.90"),
            ask=Decimal("1.10"),
            midline=Decimal("1.0"),
            vbt_mid=Decimal("0.98"),
            vbt_spread=Decimal("0.20"),
            ticket_size=Decimal(1),
        )

        d = snapshot.to_dict()

        assert "dealer_premium_pct" in d
        assert "vbt_premium_pct" in d
        # Compare as floats to handle Decimal formatting differences
        assert float(d["dealer_premium_pct"]) == 0.0
        assert float(d["vbt_premium_pct"]) == -2.0


class TestDebtToMoneyRatio:
    """Test debt-to-money ratio metric (Plan 020 - Phase B)."""

    @pytest.mark.parametrize(
        "debt,money,expected_ratio",
        [
            (Decimal(10000), Decimal(5000), Decimal(2)),
            (Decimal(10000), Decimal(0), Decimal(0)),
            (Decimal(0), Decimal(5000), Decimal(0)),
        ],
        ids=["normal", "zero_money", "zero_debt"],
    )
    def test_debt_to_money_ratio_computation(self, debt, money, expected_ratio):
        """Test debt-to-money ratio is computed correctly."""
        metrics = RunMetrics()
        metrics.initial_total_debt = debt
        metrics.initial_total_money = money

        assert metrics.debt_to_money_ratio == expected_ratio

    def test_debt_to_money_captured_on_init(self):
        """Test that debt and money are captured at initialization."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Should have captured initial debt and money
        assert subsystem.metrics.initial_total_debt > 0
        assert subsystem.metrics.initial_total_money > 0

        # For our test setup: 5 payables * 50 = 250 total debt
        # 5 households * 100 cash = 500 total money
        assert subsystem.metrics.initial_total_debt == Decimal(250)
        assert subsystem.metrics.initial_total_money == Decimal(500)

        # Ratio should be 250/500 = 0.5
        assert subsystem.metrics.debt_to_money_ratio == Decimal("0.5")

    def test_summary_includes_debt_to_money(self):
        """Test summary includes debt-to-money metrics."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        summary = subsystem.metrics.summary()

        assert "initial_total_debt" in summary
        assert "initial_total_money" in summary
        assert "debt_to_money_ratio" in summary
        assert summary["initial_total_debt"] == 250.0
        assert summary["initial_total_money"] == 500.0
        assert summary["debt_to_money_ratio"] == 0.5


class TestMidPriceTimeSeries:
    """Test mid price time series methods (Plan 020 - Phase A.2)."""

    def test_dealer_mid_timeseries(self):
        """Test dealer mid time series extraction."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading for 3 days
        for day in range(1, 4):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        timeseries = subsystem.metrics.dealer_mid_timeseries()

        # Should have entries for each day
        assert len(timeseries) == 3
        assert 1 in timeseries
        assert 2 in timeseries
        assert 3 in timeseries

        # Each day should have entries for each bucket
        for _day, buckets in timeseries.items():
            assert len(buckets) == 3
            assert "short" in buckets or "mid" in buckets or "long" in buckets

    def test_vbt_mid_timeseries(self):
        """Test VBT mid time series extraction."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading for 2 days
        for day in range(1, 3):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        timeseries = subsystem.metrics.vbt_mid_timeseries()

        # Should have entries for each day
        assert len(timeseries) == 2

    def test_premium_timeseries(self):
        """Test premium time series extraction."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)
        run_dealer_trading_phase(subsystem, sys, current_day=1)

        dealer_premium = subsystem.metrics.dealer_premium_timeseries()
        vbt_premium = subsystem.metrics.vbt_premium_timeseries()

        assert len(dealer_premium) >= 1
        assert len(vbt_premium) >= 1

    def test_final_mids_in_summary(self):
        """Test summary includes final mid prices."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading
        for day in range(1, 4):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        summary = subsystem.metrics.summary()

        assert "dealer_mid_final" in summary
        assert "vbt_mid_final" in summary
        assert "dealer_premium_final_pct" in summary
        assert "vbt_premium_final_pct" in summary

        # These should be dicts with bucket keys
        assert isinstance(summary["dealer_mid_final"], dict)
        assert isinstance(summary["vbt_mid_final"], dict)


class TestMidTimeseriesCSVExport:
    """Test CSV export of mid price time series."""

    def test_mid_timeseries_csv_export(self, tmp_path):
        """Test exporting mid time series to CSV."""
        sys, _ = create_test_system_with_ring()
        config = create_dealer_config()

        subsystem = initialize_dealer_subsystem(sys, config, current_day=1)

        # Run trading for 2 days
        for day in range(1, 3):
            run_dealer_trading_phase(subsystem, sys, current_day=day)

        # Export to CSV
        csv_path = tmp_path / "mid_timeseries.csv"
        subsystem.metrics.to_mid_timeseries_csv(str(csv_path))

        # Verify file exists
        assert csv_path.exists()

        # Read and verify contents
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have rows (2 days * 3 buckets = 6 rows)
        assert len(rows) == 6

        # Check header fields
        assert "day" in rows[0]
        assert "bucket" in rows[0]
        assert "dealer_midline" in rows[0]
        assert "vbt_mid" in rows[0]
        assert "dealer_premium_pct" in rows[0]
        assert "vbt_premium_pct" in rows[0]
