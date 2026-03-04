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

import random
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
from bilancio.dealer.metrics import RunMetrics
from bilancio.dealer.models import (
    DEFAULT_BUCKETS,
    BucketConfig,
    DealerState,
    Ticket,
    TraderState,
    VBTState,
)
from bilancio.dealer.simulation import DealerRingConfig
from bilancio.decision.profiles import TraderProfile, VBTProfile
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
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
from bilancio.engines.system import System


# ── Helpers ────────────────────────────────────────────────────────


def _make_system_with_payables() -> System:
    """Create a system with 3 households, cash, and payables."""
    sys = System()
    cb = CentralBank(id="CB1", name="CB", kind="central_bank")
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    h3 = Household(id="H3", name="H3", kind="household")
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.add_agent(h3)
    sys.mint_cash("H1", 100)
    sys.mint_cash("H2", 100)
    sys.mint_cash("H3", 100)

    p1 = Payable(
        id=sys.new_contract_id("P"), kind=InstrumentKind.PAYABLE,
        amount=50, denom="X", asset_holder_id="H2",
        liability_issuer_id="H1", due_day=sys.state.day + 2,
    )
    sys.add_contract(p1)
    p2 = Payable(
        id=sys.new_contract_id("P"), kind=InstrumentKind.PAYABLE,
        amount=30, denom="X", asset_holder_id="H3",
        liability_issuer_id="H2", due_day=sys.state.day + 5,
    )
    sys.add_contract(p2)
    p3 = Payable(
        id=sys.new_contract_id("P"), kind=InstrumentKind.PAYABLE,
        amount=20, denom="X", asset_holder_id="H1",
        liability_issuer_id="H3", due_day=sys.state.day + 10,
    )
    sys.add_contract(p3)
    return sys


def _make_dealer_config() -> DealerRingConfig:
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
            id="T1", issuer_id="H1", owner_id="H3", face=Decimal("10"),
            maturity_day=5, remaining_tau=5, bucket_id="short", serial=0,
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
        sys = _make_system_with_payables()
        config = _make_dealer_config()
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
        sys = _make_system_with_payables()
        config = _make_dealer_config()
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
        sys = _make_system_with_payables()
        config = _make_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.dirty_bucket_recompute = True
        subsystem.trading_rounds = 2  # 2 rounds

        # Run trading phase - should not error
        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        assert isinstance(events, list)


class TestIncrementalIntentions:
    def test_incremental_intentions_enabled(self):
        """With incremental_intentions, intention cache is used."""
        sys = _make_system_with_payables()
        config = _make_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.incremental_intentions = True
        subsystem.trading_rounds = 2

        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        assert isinstance(events, list)
        # Intention cache should be cleared after trading
        assert subsystem._intention_cache is None


class TestGetAgentCash:
    def test_nonexistent_agent(self):
        sys = _make_system_with_payables()
        assert _get_agent_cash(sys, "nonexistent") == Decimal(0)

    def test_agent_with_cash(self):
        sys = _make_system_with_payables()
        cash = _get_agent_cash(sys, "H1")
        assert cash == Decimal(100)


class TestSyncDealerToSystem:
    def test_full_sync_no_error(self):
        """Full sync after trading phase should not raise."""
        sys = _make_system_with_payables()
        config = _make_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)

        # Run one trading phase
        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        # Sync back
        sync_dealer_to_system(subsystem, sys)

        # All cash should be accounted for
        for tid in ["H1", "H2", "H3"]:
            cash = _get_agent_cash(sys, tid)
            assert cash >= Decimal(0)


class TestMatchingOrderUrgency:
    """Test that matching_order='urgency' doesn't crash."""

    def test_urgency_matching(self):
        sys = _make_system_with_payables()
        config = _make_dealer_config()
        subsystem = initialize_dealer_subsystem(sys, config, current_day=0)
        subsystem.matching_order = "urgency"
        subsystem.trading_rounds = 1

        events = run_dealer_trading_phase(subsystem, sys, current_day=0)
        assert isinstance(events, list)
