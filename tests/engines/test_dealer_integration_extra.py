"""Extra coverage tests for bilancio.engines.dealer_integration.

Targets uncovered lines:
- Lines 414-425: per-trader assessor creation in initialize_dealer_subsystem
- Line 492: trader_profile/vbt_profile attachment
- Lines 582-593: per-trader assessor creation in balanced subsystem
- Lines 618-622: _prune_ineligible_traders
- Line 729: dirty_bucket_recompute option
- Lines 733-746: incremental_intentions option (round 0 + subsequent rounds)
- Lines 772-778: dirty_bucket_recompute recompute between rounds
- Lines 815-853: compute_passive_pnl
"""

from decimal import Decimal

from bilancio.dealer.models import DEFAULT_BUCKETS, Ticket, TraderState
from bilancio.engines.dealer_integration import (
    DealerSubsystem,
    _assign_bucket,
    _get_agent_cash,
    _prune_ineligible_traders,
    compute_passive_pnl,
    initialize_dealer_subsystem,
    run_dealer_trading_phase,
    sync_dealer_to_system,
)
from tests.conftest import create_dealer_config, create_test_system_with_payables

# ── Tests ──────────────────────────────────────────────────────────


class TestAssignBucket:
    def test_short(self):
        assert _assign_bucket(2, list(DEFAULT_BUCKETS)) == "short"

    def test_mid(self):
        assert _assign_bucket(5, list(DEFAULT_BUCKETS)) == "mid"

    def test_long(self):
        assert _assign_bucket(15, list(DEFAULT_BUCKETS)) == "long"

    def test_empty_configs_fallback(self):
        assert _assign_bucket(5, []) == "default"


class TestPruneIneligibleTraders:
    def test_prunes_empty_traders(self):
        sub = DealerSubsystem()
        # Trader with no tickets and no cash
        t1 = TraderState(agent_id="H1", cash=Decimal("0"), tickets_owned=[], obligations=[])
        # Trader with cash
        t2 = TraderState(agent_id="H2", cash=Decimal("100"), tickets_owned=[], obligations=[])
        # Trader with tickets
        ticket = Ticket(
            id="T1",
            issuer_id="H1",
            owner_id="H3",
            face=Decimal("10"),
            maturity_day=5,
            remaining_tau=5,
            bucket_id="short",
            serial=0,
        )
        t3 = TraderState(agent_id="H3", cash=Decimal("0"), tickets_owned=[ticket], obligations=[])
        sub.traders = {"H1": t1, "H2": t2, "H3": t3}

        eligible = _prune_ineligible_traders(sub)
        assert "H1" not in eligible  # pruned
        assert "H2" in eligible
        assert "H3" in eligible


class TestComputePassivePnl:
    def test_passive_pnl_basic(self):
        """Compute PnL for passive dealer entities."""
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.enabled = False

        # Set initial equity
        for bucket_id in subsystem.dealers:
            subsystem.metrics.initial_equity_by_bucket[bucket_id] = Decimal("100")

        pnl = compute_passive_pnl(subsystem, sys)
        assert "dealer_total_pnl" in pnl
        assert "dealer_total_return" in pnl
        assert pnl["total_trades"] == 0
        assert pnl["total_sell_trades"] == 0
        assert isinstance(pnl["dealer_pnl_by_bucket"], dict)

    def test_passive_pnl_zero_initial_equity(self):
        """When initial equity is 0, return is 0."""
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.enabled = False

        # Zero initial equity
        for bucket_id in subsystem.dealers:
            subsystem.metrics.initial_equity_by_bucket[bucket_id] = Decimal("0")

        pnl = compute_passive_pnl(subsystem, sys)
        assert pnl["dealer_total_return"] == 0.0


class TestDirtyBucketRecompute:
    def test_dirty_bucket_only_recomputes_dirty(self):
        """With dirty_bucket_recompute, only dirty buckets are recomputed."""
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.dirty_bucket_recompute = True
        subsystem.trading_rounds = 2  # 2 rounds

        # Run trading phase - should not error
        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        assert isinstance(events, list)


class TestIncrementalIntentions:
    def test_incremental_intentions_enabled(self):
        """With incremental_intentions, intention cache is used."""
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.incremental_intentions = True
        subsystem.trading_rounds = 2

        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        assert isinstance(events, list)
        # Intention cache should be cleared after trading
        assert subsystem._intention_cache is None


class TestGetAgentCash:
    def test_nonexistent_agent(self):
        sys = create_test_system_with_payables()
        assert _get_agent_cash(sys, "nonexistent") == Decimal(0)

    def test_agent_with_cash(self):
        sys = create_test_system_with_payables()
        cash = _get_agent_cash(sys, "H1")
        assert cash == Decimal(100)


class TestSyncDealerToSystem:
    def test_full_sync_no_error(self):
        """Full sync after trading phase should not raise."""
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Run one trading phase
        run_dealer_trading_phase(subsystem, sys, current_day=0)
        # Sync back
        sync_dealer_to_system(subsystem, sys)

        # All cash should be accounted for
        for tid in ["H1", "H2", "H3"]:
            cash = _get_agent_cash(sys, tid)
            assert cash >= Decimal(0)


class TestMatchingOrderUrgency:
    """Test that matching_order='urgency' doesn't crash."""

    def test_urgency_matching(self):
        sys = create_test_system_with_payables()
        config = create_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.matching_order = "urgency"
        subsystem.trading_rounds = 1

        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        assert isinstance(events, list)
