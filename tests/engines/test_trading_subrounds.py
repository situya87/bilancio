"""Tests for multiple trading sub-rounds per day and matched-pairs-first ordering.

Validates that:
- trading_rounds=1 preserves existing behaviour (one sell + one buy per trader)
- trading_rounds>1 allows multiple sell/buy cycles in a single day
- Early termination: rounds stop when no intentions exist
- Dealer quotes are recomputed between rounds
- Matching engine processes paired sell+buy first, then residuals
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch, call

from bilancio.engines.dealer_integration import DealerSubsystem


class TestTradingRoundsDefault:
    """DealerSubsystem defaults and attribute tests."""

    def test_trading_rounds_default_is_one(self):
        """DealerSubsystem defaults to trading_rounds=1."""
        subsystem = DealerSubsystem()
        assert subsystem.trading_rounds == 1

    def test_trading_rounds_configurable(self):
        """Can set trading_rounds at construction."""
        subsystem = DealerSubsystem(trading_rounds=3)
        assert subsystem.trading_rounds == 3


class TestTradingSubroundsLoop:
    """Test the sub-round looping logic in run_dealer_trading_phase.

    These tests mock the internal functions to verify the loop structure
    without needing a fully wired System.
    """

    @patch("bilancio.engines.dealer_integration._capture_system_state_snapshot")
    @patch("bilancio.engines.dealer_integration._capture_trader_snapshots")
    @patch("bilancio.engines.dealer_integration._capture_dealer_snapshots")
    @patch("bilancio.engines.dealer_integration._update_vbt_credit_mids")
    @patch("bilancio.engines.dealer_integration._pool_desk_cash")
    @patch("bilancio.engines.dealer_integration._update_ticket_maturities")
    @patch("bilancio.engines.dealer_integration._ingest_new_payables")
    @patch("bilancio.engines.dealer_integration._cleanup_orphaned_tickets")
    @patch("bilancio.engines.dealer_integration._sync_dealer_vbt_cash_from_system")
    @patch("bilancio.engines.dealer_integration._sync_trader_cash_from_system")
    @patch("bilancio.engines.dealer_integration.recompute_dealer_state")
    @patch("bilancio.engines.dealer_integration.DealerMatchingEngine")
    @patch("bilancio.engines.dealer_integration.collect_buy_intentions")
    @patch("bilancio.engines.dealer_integration.collect_sell_intentions")
    def test_single_round_calls_matching_once(
        self,
        mock_collect_sell,
        mock_collect_buy,
        mock_engine_cls,
        mock_recompute,
        mock_sync_trader,
        mock_sync_dv,
        mock_cleanup,
        mock_ingest,
        mock_update_mat,
        mock_pool,
        mock_update_vbt,
        mock_snap_dealer,
        mock_snap_trader,
        mock_snap_system,
    ):
        """With trading_rounds=1, matching engine executes exactly once."""
        from bilancio.engines.dealer_integration import run_dealer_trading_phase

        subsystem = DealerSubsystem(enabled=True, trading_rounds=1)
        system = MagicMock()

        mock_collect_sell.return_value = [MagicMock()]  # one sell intention
        mock_collect_buy.return_value = [MagicMock()]   # one buy intention
        engine_instance = MagicMock()
        mock_engine_cls.return_value = engine_instance

        run_dealer_trading_phase(subsystem, system, current_day=1)

        assert engine_instance.execute.call_count == 1

    @patch("bilancio.engines.dealer_integration._capture_system_state_snapshot")
    @patch("bilancio.engines.dealer_integration._capture_trader_snapshots")
    @patch("bilancio.engines.dealer_integration._capture_dealer_snapshots")
    @patch("bilancio.engines.dealer_integration._update_vbt_credit_mids")
    @patch("bilancio.engines.dealer_integration._pool_desk_cash")
    @patch("bilancio.engines.dealer_integration._update_ticket_maturities")
    @patch("bilancio.engines.dealer_integration._ingest_new_payables")
    @patch("bilancio.engines.dealer_integration._cleanup_orphaned_tickets")
    @patch("bilancio.engines.dealer_integration._sync_dealer_vbt_cash_from_system")
    @patch("bilancio.engines.dealer_integration._sync_trader_cash_from_system")
    @patch("bilancio.engines.dealer_integration.recompute_dealer_state")
    @patch("bilancio.engines.dealer_integration.DealerMatchingEngine")
    @patch("bilancio.engines.dealer_integration.collect_buy_intentions")
    @patch("bilancio.engines.dealer_integration.collect_sell_intentions")
    def test_three_rounds_calls_matching_three_times(
        self,
        mock_collect_sell,
        mock_collect_buy,
        mock_engine_cls,
        mock_recompute,
        mock_sync_trader,
        mock_sync_dv,
        mock_cleanup,
        mock_ingest,
        mock_update_mat,
        mock_pool,
        mock_update_vbt,
        mock_snap_dealer,
        mock_snap_trader,
        mock_snap_system,
    ):
        """With trading_rounds=3 and active intentions, matching runs 3 times."""
        from bilancio.engines.dealer_integration import run_dealer_trading_phase

        subsystem = DealerSubsystem(enabled=True, trading_rounds=3)
        # Need dealers/vbts for recompute between rounds
        mock_dealer = MagicMock()
        mock_vbt = MagicMock()
        subsystem.dealers = {"short": mock_dealer}
        subsystem.vbts = {"short": mock_vbt}

        system = MagicMock()

        # Always return some intentions so all 3 rounds execute
        mock_collect_sell.return_value = [MagicMock()]
        mock_collect_buy.return_value = [MagicMock()]
        engine_instance = MagicMock()
        mock_engine_cls.return_value = engine_instance

        run_dealer_trading_phase(subsystem, system, current_day=1)

        assert engine_instance.execute.call_count == 3
        # recompute_dealer_state called between rounds (2 times for 3 rounds)
        assert mock_recompute.call_count >= 2

    @patch("bilancio.engines.dealer_integration._capture_system_state_snapshot")
    @patch("bilancio.engines.dealer_integration._capture_trader_snapshots")
    @patch("bilancio.engines.dealer_integration._capture_dealer_snapshots")
    @patch("bilancio.engines.dealer_integration._update_vbt_credit_mids")
    @patch("bilancio.engines.dealer_integration._pool_desk_cash")
    @patch("bilancio.engines.dealer_integration._update_ticket_maturities")
    @patch("bilancio.engines.dealer_integration._ingest_new_payables")
    @patch("bilancio.engines.dealer_integration._cleanup_orphaned_tickets")
    @patch("bilancio.engines.dealer_integration._sync_dealer_vbt_cash_from_system")
    @patch("bilancio.engines.dealer_integration._sync_trader_cash_from_system")
    @patch("bilancio.engines.dealer_integration.recompute_dealer_state")
    @patch("bilancio.engines.dealer_integration.DealerMatchingEngine")
    @patch("bilancio.engines.dealer_integration.collect_buy_intentions")
    @patch("bilancio.engines.dealer_integration.collect_sell_intentions")
    def test_early_termination_no_intentions(
        self,
        mock_collect_sell,
        mock_collect_buy,
        mock_engine_cls,
        mock_recompute,
        mock_sync_trader,
        mock_sync_dv,
        mock_cleanup,
        mock_ingest,
        mock_update_mat,
        mock_pool,
        mock_update_vbt,
        mock_snap_dealer,
        mock_snap_trader,
        mock_snap_system,
    ):
        """With no intentions, rounds terminate early even with trading_rounds=5."""
        from bilancio.engines.dealer_integration import run_dealer_trading_phase

        subsystem = DealerSubsystem(enabled=True, trading_rounds=5)
        system = MagicMock()

        # Return empty intentions — should break immediately
        mock_collect_sell.return_value = []
        mock_collect_buy.return_value = []
        engine_instance = MagicMock()
        mock_engine_cls.return_value = engine_instance

        run_dealer_trading_phase(subsystem, system, current_day=1)

        # Engine should never execute (no intentions to match)
        assert engine_instance.execute.call_count == 0

    @patch("bilancio.engines.dealer_integration._capture_system_state_snapshot")
    @patch("bilancio.engines.dealer_integration._capture_trader_snapshots")
    @patch("bilancio.engines.dealer_integration._capture_dealer_snapshots")
    @patch("bilancio.engines.dealer_integration._update_vbt_credit_mids")
    @patch("bilancio.engines.dealer_integration._pool_desk_cash")
    @patch("bilancio.engines.dealer_integration._update_ticket_maturities")
    @patch("bilancio.engines.dealer_integration._ingest_new_payables")
    @patch("bilancio.engines.dealer_integration._cleanup_orphaned_tickets")
    @patch("bilancio.engines.dealer_integration._sync_dealer_vbt_cash_from_system")
    @patch("bilancio.engines.dealer_integration._sync_trader_cash_from_system")
    @patch("bilancio.engines.dealer_integration.recompute_dealer_state")
    @patch("bilancio.engines.dealer_integration.DealerMatchingEngine")
    @patch("bilancio.engines.dealer_integration.collect_buy_intentions")
    @patch("bilancio.engines.dealer_integration.collect_sell_intentions")
    def test_partial_termination(
        self,
        mock_collect_sell,
        mock_collect_buy,
        mock_engine_cls,
        mock_recompute,
        mock_sync_trader,
        mock_sync_dv,
        mock_cleanup,
        mock_ingest,
        mock_update_mat,
        mock_pool,
        mock_update_vbt,
        mock_snap_dealer,
        mock_snap_trader,
        mock_snap_system,
    ):
        """Intentions dry up after round 2 — engine runs only twice."""
        from bilancio.engines.dealer_integration import run_dealer_trading_phase

        subsystem = DealerSubsystem(enabled=True, trading_rounds=5)
        mock_dealer = MagicMock()
        mock_vbt = MagicMock()
        subsystem.dealers = {"short": mock_dealer}
        subsystem.vbts = {"short": mock_vbt}
        system = MagicMock()

        # Round 1: sell + buy intentions, round 2: sell only, round 3: empty
        mock_collect_sell.side_effect = [
            [MagicMock()],   # round 1
            [MagicMock()],   # round 2
            [],              # round 3 - no sells
        ]
        mock_collect_buy.side_effect = [
            [MagicMock()],   # round 1
            [],              # round 2 - only sells
            [],              # round 3 - no buys either
        ]
        engine_instance = MagicMock()
        mock_engine_cls.return_value = engine_instance

        run_dealer_trading_phase(subsystem, system, current_day=1)

        # Rounds 1 and 2 execute (both have at least one intention)
        # Round 3 has no intentions, so early break
        assert engine_instance.execute.call_count == 2

    @patch("bilancio.engines.dealer_integration._capture_system_state_snapshot")
    @patch("bilancio.engines.dealer_integration._capture_trader_snapshots")
    @patch("bilancio.engines.dealer_integration._capture_dealer_snapshots")
    @patch("bilancio.engines.dealer_integration._update_vbt_credit_mids")
    @patch("bilancio.engines.dealer_integration._pool_desk_cash")
    @patch("bilancio.engines.dealer_integration._update_ticket_maturities")
    @patch("bilancio.engines.dealer_integration._ingest_new_payables")
    @patch("bilancio.engines.dealer_integration._cleanup_orphaned_tickets")
    @patch("bilancio.engines.dealer_integration._sync_dealer_vbt_cash_from_system")
    @patch("bilancio.engines.dealer_integration._sync_trader_cash_from_system")
    @patch("bilancio.engines.dealer_integration.recompute_dealer_state")
    @patch("bilancio.engines.dealer_integration.DealerMatchingEngine")
    @patch("bilancio.engines.dealer_integration.collect_buy_intentions")
    @patch("bilancio.engines.dealer_integration.collect_sell_intentions")
    def test_recompute_not_called_after_last_round(
        self,
        mock_collect_sell,
        mock_collect_buy,
        mock_engine_cls,
        mock_recompute,
        mock_sync_trader,
        mock_sync_dv,
        mock_cleanup,
        mock_ingest,
        mock_update_mat,
        mock_pool,
        mock_update_vbt,
        mock_snap_dealer,
        mock_snap_trader,
        mock_snap_system,
    ):
        """recompute_dealer_state is NOT called after the final round."""
        from bilancio.engines.dealer_integration import run_dealer_trading_phase

        subsystem = DealerSubsystem(enabled=True, trading_rounds=2)
        mock_dealer = MagicMock()
        mock_vbt = MagicMock()
        subsystem.dealers = {"short": mock_dealer}
        subsystem.vbts = {"short": mock_vbt}
        system = MagicMock()

        mock_collect_sell.return_value = [MagicMock()]
        mock_collect_buy.return_value = [MagicMock()]
        engine_instance = MagicMock()
        mock_engine_cls.return_value = engine_instance

        # Reset mock to count only the inter-round recompute calls
        # (Phase 2 also calls recompute, so we need to track after that)
        mock_recompute.reset_mock()

        run_dealer_trading_phase(subsystem, system, current_day=1)

        # Phase 2 calls recompute once (for the "short" bucket)
        # Inter-round recompute: once (between round 0 and round 1)
        # NOT after round 1 (the last round)
        # Total: 1 (phase 2) + 1 (inter-round) = 2
        phase2_calls = 1  # one bucket
        inter_round_calls = 1  # between round 0 and round 1 only
        assert mock_recompute.call_count == phase2_calls + inter_round_calls


class TestMatchedPairsFirst:
    """Test that the matching engine processes paired sell+buy before residuals.

    The engine should interleave sell-then-buy for the min(sells, buys)
    paired portion, then process leftover sells, then leftover buys.
    This minimises dealer peak inventory by offsetting immediately.
    """

    def test_equal_sells_and_buys_all_paired(self):
        """When #sells == #buys, every trade is paired (sell, buy, sell, buy, ...)."""
        from bilancio.engines.matching import DealerMatchingEngine

        call_log: list[str] = []

        def mock_sell(subsystem, trader_id, day, events, budgets):
            call_log.append(f"sell:{trader_id}")

        def mock_buy(subsystem, trader_id, day, events, budgets, max_spend=None):
            call_log.append(f"buy:{trader_id}")

        with patch("bilancio.engines.dealer_trades._execute_sell_trade", side_effect=mock_sell), \
             patch("bilancio.engines.dealer_trades._execute_buy_trade", side_effect=mock_buy):

            subsystem = MagicMock()
            subsystem.dealers = {}
            subsystem.vbts = {}
            # Disable shuffle so order is deterministic
            subsystem.rng.shuffle = lambda x: None

            sell_intentions = [MagicMock(trader_id=f"S{i}") for i in range(3)]
            buy_intentions = [MagicMock(trader_id=f"B{i}", max_spend=Decimal(100)) for i in range(3)]

            engine = DealerMatchingEngine()
            engine.execute(subsystem, MagicMock(), 1, sell_intentions, buy_intentions, [])

        # All 3 are paired: sell, buy, sell, buy, sell, buy
        assert call_log == [
            "sell:S0", "buy:B0",
            "sell:S1", "buy:B1",
            "sell:S2", "buy:B2",
        ]

    def test_more_sells_than_buys(self):
        """Excess sells are processed after all pairs."""
        from bilancio.engines.matching import DealerMatchingEngine

        call_log: list[str] = []

        def mock_sell(subsystem, trader_id, day, events, budgets):
            call_log.append(f"sell:{trader_id}")

        def mock_buy(subsystem, trader_id, day, events, budgets, max_spend=None):
            call_log.append(f"buy:{trader_id}")

        with patch("bilancio.engines.dealer_trades._execute_sell_trade", side_effect=mock_sell), \
             patch("bilancio.engines.dealer_trades._execute_buy_trade", side_effect=mock_buy):

            subsystem = MagicMock()
            subsystem.dealers = {}
            subsystem.vbts = {}
            subsystem.rng.shuffle = lambda x: None

            sell_intentions = [MagicMock(trader_id=f"S{i}") for i in range(5)]
            buy_intentions = [MagicMock(trader_id=f"B{i}", max_spend=Decimal(100)) for i in range(2)]

            engine = DealerMatchingEngine()
            engine.execute(subsystem, MagicMock(), 1, sell_intentions, buy_intentions, [])

        # 2 paired, then 3 residual sells
        assert call_log == [
            "sell:S0", "buy:B0",
            "sell:S1", "buy:B1",
            "sell:S2", "sell:S3", "sell:S4",
        ]

    def test_more_buys_than_sells(self):
        """Excess buys are processed after all pairs."""
        from bilancio.engines.matching import DealerMatchingEngine

        call_log: list[str] = []

        def mock_sell(subsystem, trader_id, day, events, budgets):
            call_log.append(f"sell:{trader_id}")

        def mock_buy(subsystem, trader_id, day, events, budgets, max_spend=None):
            call_log.append(f"buy:{trader_id}")

        with patch("bilancio.engines.dealer_trades._execute_sell_trade", side_effect=mock_sell), \
             patch("bilancio.engines.dealer_trades._execute_buy_trade", side_effect=mock_buy):

            subsystem = MagicMock()
            subsystem.dealers = {}
            subsystem.vbts = {}
            subsystem.rng.shuffle = lambda x: None

            sell_intentions = [MagicMock(trader_id="S0")]
            buy_intentions = [MagicMock(trader_id=f"B{i}", max_spend=Decimal(100)) for i in range(4)]

            engine = DealerMatchingEngine()
            engine.execute(subsystem, MagicMock(), 1, sell_intentions, buy_intentions, [])

        # 1 paired, then 3 residual buys
        assert call_log == [
            "sell:S0", "buy:B0",
            "buy:B1", "buy:B2", "buy:B3",
        ]

    def test_only_sells(self):
        """All sells, no buys — no pairing phase, straight to residuals."""
        from bilancio.engines.matching import DealerMatchingEngine

        call_log: list[str] = []

        def mock_sell(subsystem, trader_id, day, events, budgets):
            call_log.append(f"sell:{trader_id}")

        def mock_buy(subsystem, trader_id, day, events, budgets, max_spend=None):
            call_log.append(f"buy:{trader_id}")

        with patch("bilancio.engines.dealer_trades._execute_sell_trade", side_effect=mock_sell), \
             patch("bilancio.engines.dealer_trades._execute_buy_trade", side_effect=mock_buy):

            subsystem = MagicMock()
            subsystem.dealers = {}
            subsystem.vbts = {}
            subsystem.rng.shuffle = lambda x: None

            sell_intentions = [MagicMock(trader_id=f"S{i}") for i in range(3)]

            engine = DealerMatchingEngine()
            engine.execute(subsystem, MagicMock(), 1, sell_intentions, [], [])

        assert call_log == ["sell:S0", "sell:S1", "sell:S2"]

    def test_only_buys(self):
        """All buys, no sells — no pairing phase, straight to residuals."""
        from bilancio.engines.matching import DealerMatchingEngine

        call_log: list[str] = []

        def mock_sell(subsystem, trader_id, day, events, budgets):
            call_log.append(f"sell:{trader_id}")

        def mock_buy(subsystem, trader_id, day, events, budgets, max_spend=None):
            call_log.append(f"buy:{trader_id}")

        with patch("bilancio.engines.dealer_trades._execute_sell_trade", side_effect=mock_sell), \
             patch("bilancio.engines.dealer_trades._execute_buy_trade", side_effect=mock_buy):

            subsystem = MagicMock()
            subsystem.dealers = {}
            subsystem.vbts = {}
            subsystem.rng.shuffle = lambda x: None

            buy_intentions = [MagicMock(trader_id=f"B{i}", max_spend=Decimal(100)) for i in range(3)]

            engine = DealerMatchingEngine()
            engine.execute(subsystem, MagicMock(), 1, [], buy_intentions, [])

        assert call_log == ["buy:B0", "buy:B1", "buy:B2"]
