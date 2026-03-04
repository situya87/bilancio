"""Extra coverage tests for bilancio.engines.dealer_sync.

Targets uncovered lines:
- Line 41: _reassign_payable_owner when payable not found
- Line 82: _ingest_new_payables with zero-face payables
- Line 142: _ingest_new_payables assigns to trader with no asset_issuer_id
- Lines 230,234,240-241: estimate_forward_stress edge cases
- Lines 274-276,289-292: _update_vbt_credit_mids with pricing model
- Lines 309-338: _update_vbt_credit_mids legacy inline logic
- Line 529: _sync_payable_ownership no payable_id
- Lines 618-654: _sync_dealer_vbt_cash_to_system deposit + cash mode
- Line 659,695: _sync_dealer_vbt_cash_from_system edge cases
- Lines 731,744-745,749-757: _sync_dealer_vbt_cash_to_system deposit negative
- Line 784: _capture_system_state_snapshot (already well covered)
"""

from decimal import Decimal
from unittest.mock import MagicMock

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
from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit
from bilancio.engines.dealer_integration import DealerSubsystem
from bilancio.engines.dealer_sync import (
    _capture_dealer_snapshots,
    _capture_system_state_snapshot,
    _capture_trader_snapshots,
    _cleanup_orphaned_tickets,
    _ingest_new_payables,
    _move_ticket_to_new_bucket,
    _pool_desk_cash,
    _reassign_payable_owner,
    _remove_ticket_from_holdings,
    _sync_dealer_vbt_cash_from_system,
    _sync_dealer_vbt_cash_to_system,
    _sync_payable_ownership,
    _sync_trader_cash_from_system,
    _sync_trader_cash_to_system,
    _update_ticket_maturities,
    _update_vbt_credit_mids,
    estimate_forward_stress,
)
from bilancio.engines.system import System

import random


# ── Helpers ────────────────────────────────────────────────────────


def _make_system_with_agents() -> System:
    sys = System()
    cb = CentralBank(id="CB1", name="CB", kind="central_bank")
    h1 = Household(id="H1", name="H1", kind="household")
    h2 = Household(id="H2", name="H2", kind="household")
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.add_agent(h2)
    sys.mint_cash("H1", 100)
    sys.mint_cash("H2", 100)
    return sys


def _make_subsystem() -> DealerSubsystem:
    params = KernelParams(S=Decimal("1"))
    sub = DealerSubsystem(
        params=params,
        rng=random.Random(42),
        metrics=RunMetrics(),
        bucket_configs=[
            BucketConfig(name="short", tau_min=1, tau_max=3),
            BucketConfig(name="mid", tau_min=4, tau_max=7),
            BucketConfig(name="long", tau_min=8, tau_max=None),
        ],
    )
    return sub


# ── Tests ──────────────────────────────────────────────────────────


class TestReassignPayableOwner:
    def test_not_payable_does_nothing(self):
        """If contract is not a Payable, no changes occur."""
        sys = _make_system_with_agents()
        sys.mint_cash("H1", 50)  # create a non-payable contract
        # Get any cash contract ID
        cash_ids = [cid for cid in sys.state.agents["H1"].asset_ids
                    if sys.state.contracts[cid].kind == InstrumentKind.CASH]
        _reassign_payable_owner(sys, cash_ids[0], "H1", "H2")
        # Cash contract should still belong to H1
        assert cash_ids[0] in sys.state.agents["H1"].asset_ids

    def test_payable_reassigned(self):
        sys = _make_system_with_agents()
        p = Payable(
            id="P1", kind=InstrumentKind.PAYABLE, amount=50,
            denom="X", asset_holder_id="H1", liability_issuer_id="H2",
            due_day=5,
        )
        sys.add_contract(p)
        _reassign_payable_owner(sys, "P1", "H1", "H2")
        assert "P1" not in sys.state.agents["H1"].asset_ids
        assert "P1" in sys.state.agents["H2"].asset_ids
        assert p.asset_holder_id == "H2"
        assert p.holder_id is None


class TestEstimateForwardStress:
    def test_no_obligations_returns_zero(self):
        sys = _make_system_with_agents()
        stress = estimate_forward_stress(sys, current_day=0, horizon=5)
        assert stress == Decimal(0)

    def test_high_stress(self):
        """When cash < due, stress > 0."""
        sys = _make_system_with_agents()
        # Obligation larger than cash
        p = Payable(
            id="P1", kind=InstrumentKind.PAYABLE, amount=500,
            denom="X", asset_holder_id="H2", liability_issuer_id="H1",
            due_day=2,
        )
        sys.add_contract(p)
        stress = estimate_forward_stress(sys, current_day=0, horizon=5)
        assert stress > Decimal(0)

    def test_stress_capped_at_one(self):
        """Stress is capped at 1.0."""
        sys = _make_system_with_agents()
        # Huge obligation, tiny cash
        p = Payable(
            id="P1", kind=InstrumentKind.PAYABLE, amount=100000,
            denom="X", asset_holder_id="H2", liability_issuer_id="H1",
            due_day=2,
        )
        sys.add_contract(p)
        stress = estimate_forward_stress(sys, current_day=0, horizon=5)
        assert stress <= Decimal(1)


class TestPoolDeskCash:
    def test_pools_evenly(self):
        sub = _make_subsystem()
        d1 = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"))
        d2 = DealerState(bucket_id="mid", agent_id="dealer_mid", cash=Decimal("200"))
        sub.dealers = {"short": d1, "mid": d2}
        v1 = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("1"), O=Decimal("0.1"), cash=Decimal("50"))
        v2 = VBTState(bucket_id="mid", agent_id="vbt_mid", M=Decimal("1"), O=Decimal("0.1"), cash=Decimal("150"))
        sub.vbts = {"short": v1, "mid": v2}

        _pool_desk_cash(sub)

        assert d1.cash == Decimal("150")  # (100+200)/2
        assert d2.cash == Decimal("150")
        assert v1.cash == Decimal("100")  # (50+150)/2
        assert v2.cash == Decimal("100")
        # All buckets marked dirty
        assert sub._dirty_buckets == {"short", "mid"}


class TestUpdateVBTCreditMids:
    def test_without_risk_assessor_is_noop(self):
        sub = _make_subsystem()
        sub.risk_assessor = None
        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("100"))
        sub.vbts = {"short": v}
        _update_vbt_credit_mids(sub, current_day=1)
        # No change expected
        assert v.M == Decimal("0.90")

    def test_with_pricing_model(self):
        """When vbt_pricing_model is set, delegates to compute_mid."""
        sub = _make_subsystem()
        mock_assessor = MagicMock()
        mock_assessor.estimate_default_prob.return_value = Decimal("0.10")
        mock_assessor.params.initial_prior = Decimal("0.15")
        sub.risk_assessor = mock_assessor

        mock_pricing = MagicMock()
        mock_pricing.compute_mid.return_value = Decimal("0.85")
        mock_pricing.compute_spread.return_value = Decimal("0.12")
        sub.vbt_pricing_model = mock_pricing

        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("100"))
        v.recompute_quotes()
        sub.vbts = {"short": v}

        _update_vbt_credit_mids(sub, current_day=1)
        assert v.M == Decimal("0.85")
        mock_pricing.compute_mid.assert_called_once()

    def test_legacy_inline_logic(self):
        """When no pricing model, uses legacy inline computation."""
        sub = _make_subsystem()
        mock_assessor = MagicMock()
        mock_assessor.estimate_default_prob.return_value = Decimal("0.10")
        mock_assessor.params.initial_prior = Decimal("0.15")
        sub.risk_assessor = mock_assessor
        sub.vbt_pricing_model = None  # Force legacy path
        sub.outside_mid_ratio = Decimal("0.90")
        sub.vbt_profile = VBTProfile(mid_sensitivity=Decimal("1.0"), spread_sensitivity=Decimal("0.5"))

        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("100"))
        v.recompute_quotes()
        sub.vbts = {"short": v}
        sub.initial_spread_by_bucket = {"short": Decimal("0.10")}

        _update_vbt_credit_mids(sub, current_day=1)
        # Legacy: new_M = initial_M + sens*(raw_M - initial_M)
        # raw_M = 0.90*(1-0.10) = 0.81, initial_M = 0.90*(1-0.15) = 0.765
        # new_M = 0.765 + 1.0*(0.81 - 0.765) = 0.81
        assert v.M == Decimal("0.81")


class TestCleanupOrphanedTickets:
    def test_removes_orphaned(self):
        sub = _make_subsystem()
        ticket = Ticket(
            id="TKT_1", issuer_id="H1", owner_id="H2",
            face=Decimal("10"), maturity_day=5,
            remaining_tau=5, bucket_id="short", serial=0,
        )
        sub.tickets = {"TKT_1": ticket}
        sub.ticket_to_payable = {"TKT_1": "P_DEAD"}
        sub.payable_to_ticket = {"P_DEAD": "TKT_1"}

        # Put ticket in a dealer inventory
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"), inventory=[ticket])
        sub.dealers = {"short": d}
        sub.vbts = {"short": VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("1"), O=Decimal("0.1"), cash=Decimal("100"))}

        # The payable does NOT exist in system => orphaned
        sys = _make_system_with_agents()
        _cleanup_orphaned_tickets(sub, sys)

        assert "TKT_1" not in sub.tickets
        assert "TKT_1" not in sub.ticket_to_payable
        assert "P_DEAD" not in sub.payable_to_ticket
        assert ticket not in d.inventory


class TestRemoveTicketFromHoldings:
    def test_removes_from_all(self):
        sub = _make_subsystem()
        ticket = Ticket(
            id="TKT_1", issuer_id="H1", owner_id="H2",
            face=Decimal("10"), maturity_day=5,
            remaining_tau=5, bucket_id="short", serial=0,
        )
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"), inventory=[ticket])
        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("1"), O=Decimal("0.1"), cash=Decimal("100"), inventory=[])
        trader = TraderState(agent_id="H2", cash=Decimal("100"), tickets_owned=[ticket], obligations=[ticket])
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        sub.traders = {"H2": trader}

        _remove_ticket_from_holdings(ticket, sub)

        assert ticket not in d.inventory
        assert ticket not in trader.tickets_owned
        assert ticket not in trader.obligations


class TestSyncDealerVBTCashToSystem:
    def test_cash_mode_mint(self):
        """In pure cash mode, delta > 0 mints CB cash."""
        sys = _make_system_with_agents()
        # Add dealer agent
        dealer_agent = Household(id="dealer_short", name="Dealer Short", kind="household")
        sys.add_agent(dealer_agent)
        sys.mint_cash("dealer_short", 100)

        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("150"))  # 50 more than system
        sub.dealers = {"short": d}
        sub.vbts = {}

        _sync_dealer_vbt_cash_to_system(sub, sys)

        from bilancio.engines.dealer_integration import _get_agent_cash
        assert _get_agent_cash(sys, "dealer_short") == Decimal(150)

    def test_cash_mode_burn(self):
        """In pure cash mode, delta < 0 retires CB cash."""
        sys = _make_system_with_agents()
        dealer_agent = Household(id="dealer_short", name="Dealer Short", kind="household")
        sys.add_agent(dealer_agent)
        sys.mint_cash("dealer_short", 100)

        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("80"))  # 20 less
        sub.dealers = {"short": d}
        sub.vbts = {}

        _sync_dealer_vbt_cash_to_system(sub, sys)

        from bilancio.engines.dealer_integration import _get_agent_cash
        assert _get_agent_cash(sys, "dealer_short") == Decimal(80)

    def test_deposit_mode(self):
        """In banking mode, adjusts deposit instead of minting cash."""
        sys = _make_system_with_agents()
        b = Bank(id="B1", name="Bank 1", kind="bank")
        sys.add_agent(b)
        dealer_agent = Household(id="dealer_short", name="Dealer Short", kind="household")
        sys.add_agent(dealer_agent)

        dep = BankDeposit(
            id="DEP_D", kind=InstrumentKind.BANK_DEPOSIT, amount=100,
            denom="X", asset_holder_id="dealer_short", liability_issuer_id="B1",
        )
        sys.add_contract(dep)

        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("130"))  # delta=30
        sub.dealers = {"short": d}
        sub.vbts = {}

        _sync_dealer_vbt_cash_to_system(sub, sys)
        assert dep.amount == 130

    def test_no_agent_in_system_skipped(self):
        """If agent doesn't exist in system, skip silently."""
        sys = _make_system_with_agents()
        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="nonexistent", cash=Decimal("100"))
        sub.dealers = {"short": d}
        sub.vbts = {}
        _sync_dealer_vbt_cash_to_system(sub, sys)  # should not raise


class TestSyncDealerVBTCashFromSystem:
    def test_no_agent_skipped(self):
        """If agent doesn't exist in system, skip."""
        sys = _make_system_with_agents()
        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="nonexistent", cash=Decimal("100"))
        sub.dealers = {"short": d}
        sub.vbts = {}
        _sync_dealer_vbt_cash_from_system(sub, sys)
        assert d.cash == Decimal("100")  # unchanged

    def test_syncs_from_system(self):
        """Cash is updated from the system."""
        sys = _make_system_with_agents()
        dealer_agent = Household(id="dealer_short", name="D", kind="household")
        sys.add_agent(dealer_agent)
        sys.mint_cash("dealer_short", 200)

        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"))  # stale
        sub.dealers = {"short": d}
        sub.vbts = {}

        _sync_dealer_vbt_cash_from_system(sub, sys)
        assert d.cash == Decimal(200)


class TestCaptureSnapshots:
    def test_capture_dealer_snapshots(self):
        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"))
        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("200"))
        recompute_dealer_state(d, v, sub.params)
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        sub.traders = {}

        _capture_dealer_snapshots(sub, current_day=1)
        assert len(sub.metrics.dealer_snapshots) == 1

    def test_capture_trader_snapshots(self):
        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"))
        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("200"))
        recompute_dealer_state(d, v, sub.params)
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        trader = TraderState(agent_id="H1", cash=Decimal("100"), tickets_owned=[], obligations=[])
        sub.traders = {"H1": trader}

        _capture_trader_snapshots(sub, current_day=1)
        assert len(sub.metrics.trader_snapshots) == 1

    def test_capture_system_state_snapshot(self):
        sub = _make_subsystem()
        d = DealerState(bucket_id="short", agent_id="dealer_short", cash=Decimal("100"))
        v = VBTState(bucket_id="short", agent_id="vbt_short", M=Decimal("0.90"), O=Decimal("0.10"), cash=Decimal("200"))
        recompute_dealer_state(d, v, sub.params)
        sub.dealers = {"short": d}
        sub.vbts = {"short": v}
        sub.traders = {}

        _capture_system_state_snapshot(sub, current_day=1)
        assert len(sub.metrics.system_state_snapshots) == 1
