"""Tests for bilancio.engines.termination — Plan 051.

Verifies:
- StopReason enum members
- StabilitySnapshot construction and immutability
- compute_stability_snapshot helper
- LegacyTerminationPolicy parity with pre-051 inline logic
- Helper re-exports from simulation.py still work
"""

from decimal import Decimal

import pytest

from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.engines.system import System
from bilancio.engines.termination import (
    DEFAULT_EVENTS,
    IMPACT_EVENTS,
    LegacyTerminationPolicy,
    StabilitySnapshot,
    StopReason,
    TerminationPolicy,
    _defaults_today,
    _has_open_obligations,
    _impacted_today,
    compute_stability_snapshot,
)


# ---------------------------------------------------------------------------
# StopReason
# ---------------------------------------------------------------------------


class TestStopReason:
    def test_members(self):
        assert StopReason.STABILITY_REACHED.value == "stability_reached"
        assert StopReason.MAX_DAYS_REACHED.value == "max_days_reached"
        assert StopReason.FATAL_ERROR.value == "fatal_error"
        assert StopReason.USER_STOP.value == "user_stop"


# ---------------------------------------------------------------------------
# StabilitySnapshot
# ---------------------------------------------------------------------------


class TestStabilitySnapshot:
    def test_construction(self):
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=2,
            consecutive_no_defaults=3,
            has_open_obligations=True,
            impacted_count=1,
            default_count=0,
        )
        assert snap.day == 5
        assert snap.consecutive_quiet == 2
        assert snap.consecutive_no_defaults == 3
        assert snap.has_open_obligations is True
        assert snap.impacted_count == 1
        assert snap.default_count == 0

    def test_frozen(self):
        snap = StabilitySnapshot(
            day=0,
            consecutive_quiet=0,
            consecutive_no_defaults=0,
            has_open_obligations=False,
            impacted_count=0,
            default_count=0,
        )
        with pytest.raises(AttributeError):
            snap.day = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_stability_snapshot
# ---------------------------------------------------------------------------


class TestComputeStabilitySnapshot:
    def test_basic(self):
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        sys.add_agent(cb)
        snap = compute_stability_snapshot(sys, day=0, consecutive_quiet=1, consecutive_no_defaults=1)
        assert snap.day == 0
        assert snap.consecutive_quiet == 1
        assert snap.consecutive_no_defaults == 1
        assert snap.has_open_obligations is False
        assert snap.impacted_count == 0
        assert snap.default_count == 0

    def test_with_open_obligations(self):
        sys = System()
        cb = CentralBank(id="CB", name="CB")
        h1 = Household(id="H1", name="H1", kind="household")
        h2 = Household(id="H2", name="H2", kind="household")
        sys.add_agent(cb)
        sys.add_agent(h1)
        sys.add_agent(h2)
        sys.mint_cash("H1", 100)
        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            denom="USD",
            liability_issuer_id="H1",
            asset_holder_id="H2",
            amount=50,
            due_day=5,
        )
        sys.add_contract(payable)

        snap = compute_stability_snapshot(sys, day=0, consecutive_quiet=0, consecutive_no_defaults=0)
        assert snap.has_open_obligations is True


# ---------------------------------------------------------------------------
# LegacyTerminationPolicy
# ---------------------------------------------------------------------------


class TestLegacyTerminationPolicy:
    """Parity tests: every case that the inline code handled."""

    def setup_method(self):
        self.policy = LegacyTerminationPolicy()

    def test_protocol_compliance(self):
        assert isinstance(self.policy, TerminationPolicy)

    # -- Non-rollover mode --

    def test_non_rollover_stable(self):
        """Quiet >= quiet_days AND no open obligations → STABILITY_REACHED."""
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=2,
            consecutive_no_defaults=2,
            has_open_obligations=False,
            impacted_count=0,
            default_count=0,
        )
        result = self.policy.evaluate(snap, quiet_days=2, rollover_enabled=False)
        assert result == StopReason.STABILITY_REACHED

    def test_non_rollover_quiet_but_open_obligations(self):
        """Quiet >= quiet_days but open obligations → continue."""
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=3,
            consecutive_no_defaults=3,
            has_open_obligations=True,
            impacted_count=0,
            default_count=0,
        )
        result = self.policy.evaluate(snap, quiet_days=2, rollover_enabled=False)
        assert result is None

    def test_non_rollover_not_quiet_enough(self):
        """consecutive_quiet < quiet_days → continue."""
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=1,
            consecutive_no_defaults=5,
            has_open_obligations=False,
            impacted_count=0,
            default_count=0,
        )
        result = self.policy.evaluate(snap, quiet_days=2, rollover_enabled=False)
        assert result is None

    def test_non_rollover_both_fail(self):
        """Not quiet and has obligations → continue."""
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=0,
            consecutive_no_defaults=0,
            has_open_obligations=True,
            impacted_count=3,
            default_count=1,
        )
        result = self.policy.evaluate(snap, quiet_days=2, rollover_enabled=False)
        assert result is None

    # -- Rollover mode --

    def test_rollover_stable(self):
        """Rollover: no_defaults >= quiet_days → STABILITY_REACHED."""
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=0,  # Doesn't matter in rollover
            consecutive_no_defaults=2,
            has_open_obligations=True,  # Doesn't matter in rollover
            impacted_count=5,
            default_count=0,
        )
        result = self.policy.evaluate(snap, quiet_days=2, rollover_enabled=True)
        assert result == StopReason.STABILITY_REACHED

    def test_rollover_not_enough_no_defaults(self):
        """Rollover: no_defaults < quiet_days → continue."""
        snap = StabilitySnapshot(
            day=5,
            consecutive_quiet=10,
            consecutive_no_defaults=1,
            has_open_obligations=False,
            impacted_count=0,
            default_count=0,
        )
        result = self.policy.evaluate(snap, quiet_days=2, rollover_enabled=True)
        assert result is None

    def test_rollover_zero_quiet_days(self):
        """Rollover with quiet_days=0: always stable immediately."""
        snap = StabilitySnapshot(
            day=0,
            consecutive_quiet=0,
            consecutive_no_defaults=0,
            has_open_obligations=True,
            impacted_count=5,
            default_count=3,
        )
        result = self.policy.evaluate(snap, quiet_days=0, rollover_enabled=True)
        assert result == StopReason.STABILITY_REACHED


# ---------------------------------------------------------------------------
# Re-exports from simulation.py
# ---------------------------------------------------------------------------


class TestReExports:
    """Verify that importing from simulation.py still works."""

    def test_import_impact_events(self):
        from bilancio.engines.simulation import IMPACT_EVENTS as sim_ie
        assert sim_ie is IMPACT_EVENTS

    def test_import_default_events(self):
        from bilancio.engines.simulation import DEFAULT_EVENTS as sim_de
        assert sim_de is DEFAULT_EVENTS

    def test_import_helpers(self):
        from bilancio.engines.simulation import (
            _defaults_today as sim_dt,
            _has_open_obligations as sim_ho,
            _impacted_today as sim_it,
        )
        assert sim_it is _impacted_today
        assert sim_dt is _defaults_today
        assert sim_ho is _has_open_obligations
