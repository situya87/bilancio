"""Tests for flow_sensitivity config round-trip (Feature 2).

Verifies that flow_sensitivity threads correctly from:
  BalancedDealerConfig → VBTProfile → VBTState
"""

from decimal import Decimal

import pytest

from bilancio.config.models import BalancedDealerConfig
from bilancio.dealer.models import VBTState
from bilancio.decision.profiles import VBTProfile


class TestFlowSensitivityConfig:
    """Config round-trip tests for flow_sensitivity."""

    def test_config_default_disabled(self) -> None:
        """Default BalancedDealerConfig has flow_sensitivity=0 (disabled)."""
        cfg = BalancedDealerConfig(enabled=True)
        assert cfg.flow_sensitivity == Decimal("0.0")

    def test_config_accepts_value(self) -> None:
        """BalancedDealerConfig accepts a valid flow_sensitivity."""
        cfg = BalancedDealerConfig(enabled=True, flow_sensitivity=Decimal("0.5"))
        assert cfg.flow_sensitivity == Decimal("0.5")

    def test_config_rejects_negative(self) -> None:
        """flow_sensitivity must be >= 0."""
        with pytest.raises(ValueError):
            BalancedDealerConfig(enabled=True, flow_sensitivity=Decimal("-0.1"))

    def test_config_rejects_above_one(self) -> None:
        """flow_sensitivity must be <= 1."""
        with pytest.raises(ValueError):
            BalancedDealerConfig(enabled=True, flow_sensitivity=Decimal("1.1"))

    def test_vbt_profile_receives_value(self) -> None:
        """VBTProfile stores flow_sensitivity."""
        profile = VBTProfile(flow_sensitivity=Decimal("0.7"))
        assert profile.flow_sensitivity == Decimal("0.7")

    def test_vbt_profile_default(self) -> None:
        """VBTProfile defaults to flow_sensitivity=0."""
        profile = VBTProfile()
        assert profile.flow_sensitivity == Decimal("0.0")

    def test_vbt_state_receives_value(self) -> None:
        """VBTState stores flow_sensitivity."""
        state = VBTState(
            bucket_id="short",
            flow_sensitivity=Decimal("0.3"),
        )
        assert state.flow_sensitivity == Decimal("0.3")

    def test_config_to_profile_threading(self) -> None:
        """flow_sensitivity from config creates correct VBTProfile."""
        cfg = BalancedDealerConfig(enabled=True, flow_sensitivity=Decimal("0.8"))
        profile = VBTProfile(
            mid_sensitivity=cfg.vbt_mid_sensitivity,
            spread_sensitivity=cfg.vbt_spread_sensitivity,
            spread_scale=cfg.spread_scale,
            flow_sensitivity=cfg.flow_sensitivity,
        )
        assert profile.flow_sensitivity == Decimal("0.8")

    def test_disabled_flow_sensitivity_no_ask_change(self) -> None:
        """With flow_sensitivity=0, ask is not affected by outflow."""
        vbt = VBTState(
            bucket_id="short",
            M=Decimal("0.80"),
            O=Decimal("0.10"),
            flow_sensitivity=Decimal("0"),
        )
        vbt.cumulative_outflow = Decimal("100")
        vbt.cumulative_inflow = Decimal("0")
        vbt.recompute_quotes()

        vbt_base = VBTState(
            bucket_id="short",
            M=Decimal("0.80"),
            O=Decimal("0.10"),
            flow_sensitivity=Decimal("0"),
        )
        vbt_base.recompute_quotes()

        assert vbt.A == vbt_base.A

    def test_enabled_flow_sensitivity_widens_ask(self) -> None:
        """With flow_sensitivity>0 and net outflow, ask is wider."""
        from bilancio.dealer.models import Ticket

        vbt = VBTState(
            bucket_id="short",
            M=Decimal("0.80"),
            O=Decimal("0.10"),
            flow_sensitivity=Decimal("0.5"),
        )
        # Simulate outflow
        vbt.cumulative_outflow = Decimal("100")
        vbt.cumulative_inflow = Decimal("0")
        vbt.inventory = [
            Ticket(id=f"T{i}", serial=i, issuer_id="X", owner_id="vbt",
                   face=Decimal("20"), maturity_day=10)
            for i in range(5)
        ]  # 100 face remaining
        vbt.recompute_quotes()

        vbt_base = VBTState(
            bucket_id="short",
            M=Decimal("0.80"),
            O=Decimal("0.10"),
            flow_sensitivity=Decimal("0"),
        )
        vbt_base.recompute_quotes()

        assert vbt.A > vbt_base.A
