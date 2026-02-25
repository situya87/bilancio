"""Tests for Phase 3 (Plan 040): System-Aware Buy Controls.

Covers:
  - VBT flow tracking (cumulative_outflow / cumulative_inflow)
  - VBT ask widening with flow_sensitivity
  - Earning-motive premium in TradeGate.should_buy()
  - Backward compatibility when both controls are zero
"""

from decimal import Decimal

import pytest

from bilancio.dealer.kernel import KernelParams, recompute_dealer_state
from bilancio.dealer.models import DealerState, Ticket, VBTState
from bilancio.dealer.trading import TradeExecutor
from bilancio.decision.risk_assessment import (
    BeliefTracker,
    EVValuer,
    PositionAssessor,
    RiskAssessmentParams,
    RiskAssessor,
    TradeGate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticket(
    ticket_id: str = "TKT_1",
    issuer_id: str = "issuer_A",
    owner_id: str = "trader_1",
    face: Decimal = Decimal("20"),
    maturity_day: int = 10,
    remaining_tau: int = 5,
    bucket_id: str = "short",
    serial: int = 0,
) -> Ticket:
    return Ticket(
        id=ticket_id,
        issuer_id=issuer_id,
        owner_id=owner_id,
        face=face,
        maturity_day=maturity_day,
        remaining_tau=remaining_tau,
        bucket_id=bucket_id,
        serial=serial,
    )


def _make_vbt(
    bucket_id: str = "short",
    M: Decimal = Decimal("0.80"),
    O: Decimal = Decimal("0.10"),
    flow_sensitivity: Decimal = Decimal("0"),
    inventory: list | None = None,
    cash: Decimal = Decimal("1000"),
) -> VBTState:
    vbt = VBTState(
        bucket_id=bucket_id,
        agent_id=f"vbt_{bucket_id}",
        M=M,
        O=O,
        inventory=list(inventory) if inventory else [],
        cash=cash,
        flow_sensitivity=flow_sensitivity,
    )
    vbt.recompute_quotes()
    return vbt


def _make_dealer(
    bucket_id: str = "short",
    cash: Decimal = Decimal("1000"),
    inventory: list | None = None,
) -> DealerState:
    return DealerState(
        bucket_id=bucket_id,
        agent_id=f"dealer_{bucket_id}",
        cash=cash,
        inventory=list(inventory) if inventory else [],
    )


def _make_trade_gate(
    buy_risk_premium: Decimal = Decimal("0.01"),
    earning_motive_premium: Decimal = Decimal("0.0"),
    initial_prior: Decimal = Decimal("0.15"),
) -> TradeGate:
    """Create a TradeGate with a simple BeliefTracker/EVValuer."""
    belief = BeliefTracker(initial_prior=initial_prior)
    valuer = EVValuer(belief)
    position_assessor = PositionAssessor()
    return TradeGate(
        valuer=valuer,
        position_assessor=position_assessor,
        buy_risk_premium=buy_risk_premium,
        earning_motive_premium=earning_motive_premium,
        initial_prior=initial_prior,
    )


# ---------------------------------------------------------------------------
# Solution A: VBT flow tracking & ask widening
# ---------------------------------------------------------------------------


class TestVBTFlowTracking:
    """Test that VBT cumulative flow counters update correctly.

    Flow counters are incremented in the dealer_trades layer (after all
    rejection gates), NOT in the low-level TradeExecutor.  This prevents
    rejected trades from inflating net_outflow and widening future asks.
    """

    def test_vbt_outflow_not_incremented_by_executor(self):
        """TradeExecutor does NOT increment outflow — that happens after acceptance."""
        face = Decimal("20")
        tickets = [_make_ticket(f"TKT_{i}", serial=i, face=face) for i in range(5)]
        vbt = _make_vbt(inventory=tickets, cash=Decimal("500"))
        dealer = _make_dealer(cash=Decimal("0"), inventory=[])

        params = KernelParams(S=face)
        recompute_dealer_state(dealer, vbt, params)
        executor = TradeExecutor(params)

        # Passthrough buy — executor runs but does NOT update counter
        result = executor.execute_customer_buy(
            dealer, vbt, buyer_id="buyer_1", check_assertions=False
        )
        assert result.executed
        assert result.is_passthrough
        # Counter stays at zero — caller is responsible for incrementing
        assert vbt.cumulative_outflow == Decimal(0)

    def test_vbt_inflow_not_incremented_by_executor(self):
        """TradeExecutor does NOT increment inflow — that happens after acceptance."""
        face = Decimal("20")
        ticket = _make_ticket(face=face)
        vbt = _make_vbt(cash=Decimal("500"))
        dealer = _make_dealer(cash=Decimal("0"), inventory=[])

        params = KernelParams(S=face)
        recompute_dealer_state(dealer, vbt, params)
        executor = TradeExecutor(params)

        result = executor.execute_customer_sell(
            dealer, vbt, ticket, check_assertions=False
        )
        assert result.executed
        assert result.is_passthrough
        assert vbt.cumulative_inflow == Decimal(0)

    def test_manual_flow_tracking_after_acceptance(self):
        """Counters increment correctly when caller updates them post-acceptance."""
        face = Decimal("20")
        tickets = [_make_ticket(f"TKT_{i}", serial=i, face=face) for i in range(5)]
        vbt = _make_vbt(inventory=tickets, cash=Decimal("500"))
        dealer = _make_dealer(cash=Decimal("0"), inventory=[])

        params = KernelParams(S=face)
        recompute_dealer_state(dealer, vbt, params)
        executor = TradeExecutor(params)

        # Simulate the acceptance path in dealer_trades.py
        result = executor.execute_customer_buy(
            dealer, vbt, buyer_id="buyer_1", check_assertions=False
        )
        assert result.executed and result.is_passthrough
        # Caller increments after final acceptance (as dealer_trades.py does)
        vbt.cumulative_outflow += result.ticket.face
        assert vbt.cumulative_outflow == face

        # Second buy
        recompute_dealer_state(dealer, vbt, params)
        result2 = executor.execute_customer_buy(
            dealer, vbt, buyer_id="buyer_2", check_assertions=False
        )
        assert result2.executed
        vbt.cumulative_outflow += result2.ticket.face
        assert vbt.cumulative_outflow == face * 2


class TestVBTAskWidening:
    """Test that flow_sensitivity widens the ask when VBT is drained."""

    def test_vbt_ask_widens_with_outflow(self):
        """With flow_sensitivity > 0 and net outflow, ask should be higher."""
        M = Decimal("0.80")
        O = Decimal("0.10")

        # Baseline: no flow sensitivity
        vbt_base = _make_vbt(M=M, O=O, flow_sensitivity=Decimal("0"))
        ask_base = vbt_base.A

        # With flow sensitivity and accumulated outflow (in face-value units)
        face = Decimal("20")
        vbt_flow = _make_vbt(M=M, O=O, flow_sensitivity=Decimal("0.5"))
        # Simulate: VBT started with 10 tickets (face=20 each → 200 total), sold 5 → 100 face
        vbt_flow.cumulative_outflow = face * 5  # 100 face-value units sold
        vbt_flow.cumulative_inflow = Decimal("0")
        # Current inventory: 5 tickets remain (100 face-value)
        vbt_flow.inventory = [_make_ticket(f"TKT_{i}", serial=i, face=face) for i in range(5)]
        vbt_flow.recompute_quotes()

        # Ask should be higher than baseline
        assert vbt_flow.A > ask_base, (
            f"Expected ask with flow premium ({vbt_flow.A}) > baseline ({ask_base})"
        )

    def test_vbt_ask_no_change_without_outflow(self):
        """With flow_sensitivity > 0 but no outflow, ask unchanged."""
        M = Decimal("0.80")
        O = Decimal("0.10")

        vbt_base = _make_vbt(M=M, O=O, flow_sensitivity=Decimal("0"))
        vbt_flow = _make_vbt(M=M, O=O, flow_sensitivity=Decimal("0.5"))
        # No outflow
        vbt_flow.cumulative_outflow = Decimal("0")
        vbt_flow.cumulative_inflow = Decimal("0")
        vbt_flow.recompute_quotes()

        assert vbt_flow.A == vbt_base.A

    def test_vbt_ask_capped_at_par(self):
        """Even with large flow premium, ask cannot exceed 1.0."""
        vbt = _make_vbt(
            M=Decimal("0.95"),
            O=Decimal("0.10"),
            flow_sensitivity=Decimal("5.0"),
        )
        vbt.cumulative_outflow = Decimal("100")
        vbt.cumulative_inflow = Decimal("0")
        vbt.inventory = [_make_ticket()]  # 1 remaining
        vbt.recompute_quotes()

        assert vbt.A <= Decimal("1")

    def test_vbt_ask_net_outflow_only(self):
        """Ask premium applies only when outflow > inflow."""
        M = Decimal("0.80")
        O = Decimal("0.10")
        sensitivity = Decimal("0.5")

        vbt = _make_vbt(M=M, O=O, flow_sensitivity=sensitivity)
        # More inflow than outflow -> net inflow, no premium
        vbt.cumulative_outflow = Decimal("3")
        vbt.cumulative_inflow = Decimal("5")
        vbt.recompute_quotes()

        vbt_base = _make_vbt(M=M, O=O, flow_sensitivity=Decimal("0"))
        assert vbt.A == vbt_base.A


# ---------------------------------------------------------------------------
# Solution B: Earning-motive premium in TradeGate
# ---------------------------------------------------------------------------


class TestEarningMotivePremium:
    """Test that earning_motive_premium raises buy threshold for speculative buys."""

    def test_earning_premium_raises_threshold_no_shortfall(self):
        """When shortfall=0 and earning_motive_premium > 0, buy is harder."""
        ticket = _make_ticket(face=Decimal("20"))

        # Without earning premium
        gate_base = _make_trade_gate(
            buy_risk_premium=Decimal("0.01"),
            earning_motive_premium=Decimal("0.0"),
        )
        # With earning premium
        gate_premium = _make_trade_gate(
            buy_risk_premium=Decimal("0.01"),
            earning_motive_premium=Decimal("0.05"),
        )

        # Construct a price where the base gate would accept but premium gate would reject.
        # EV = (1 - 0.15) * 20 = 17.0
        # Base threshold ~ 0.01 + liquidity_adj (small)
        # Premium adds 0.05 more
        # Find a dealer_ask that base accepts but premium rejects.
        #
        # We need: EV >= cost + threshold_base (accept)
        #      and: EV < cost + threshold_premium (reject)
        #
        # With shortfall=0, no urgency, cash-rich trader:
        # buy_threshold_base = 0.01 + p_blended * liq_factor
        # buy_threshold_premium = 0.01 + 0.05 + p_blended * liq_factor
        # p_blended = (0.15 + 0.15)/2 = 0.15
        # liq_factor = max(0.75, 1 - cash_ratio)
        # cash=100, asset_value=50 -> cash_ratio=100/150~0.667 -> liq_factor=0.75
        # buy_threshold_base = 0.01 + 0.15*0.75 = 0.01 + 0.1125 = 0.1225
        # buy_threshold_premium = 0.01 + 0.05 + 0.1125 = 0.1725
        # EV = 17.0
        # For base accept: cost < 17.0 - 0.1225*20 = 17.0 - 2.45 = 14.55
        # For premium reject: cost > 17.0 - 0.1725*20 = 17.0 - 3.45 = 13.55
        # So dealer_ask in (13.55/20, 14.55/20) = (0.6775, 0.7275)
        dealer_ask = Decimal("0.70")  # unit price

        base_accepts = gate_base.should_buy(
            ticket=ticket,
            dealer_ask=dealer_ask,
            current_day=0,
            trader_cash=Decimal("100"),
            trader_shortfall=Decimal("0"),
            trader_asset_value=Decimal("50"),
        )
        premium_accepts = gate_premium.should_buy(
            ticket=ticket,
            dealer_ask=dealer_ask,
            current_day=0,
            trader_cash=Decimal("100"),
            trader_shortfall=Decimal("0"),
            trader_asset_value=Decimal("50"),
        )

        assert base_accepts is True, "Base gate should accept at this price"
        assert premium_accepts is False, "Premium gate should reject speculative buy"

    def test_earning_premium_no_effect_with_shortfall(self):
        """When shortfall > 0, earning_motive_premium has no effect."""
        ticket = _make_ticket(face=Decimal("20"))

        gate_base = _make_trade_gate(
            buy_risk_premium=Decimal("0.01"),
            earning_motive_premium=Decimal("0.0"),
        )
        gate_premium = _make_trade_gate(
            buy_risk_premium=Decimal("0.01"),
            earning_motive_premium=Decimal("0.05"),
        )

        # With shortfall > 0, earning premium should NOT apply
        # Both gates should produce the same decision
        dealer_ask = Decimal("0.70")

        base_result = gate_base.should_buy(
            ticket=ticket,
            dealer_ask=dealer_ask,
            current_day=0,
            trader_cash=Decimal("100"),
            trader_shortfall=Decimal("10"),  # Non-zero shortfall
            trader_asset_value=Decimal("50"),
        )
        premium_result = gate_premium.should_buy(
            ticket=ticket,
            dealer_ask=dealer_ask,
            current_day=0,
            trader_cash=Decimal("100"),
            trader_shortfall=Decimal("10"),  # Non-zero shortfall
            trader_asset_value=Decimal("50"),
        )

        assert base_result == premium_result, (
            "With shortfall > 0, earning premium should not change outcome"
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Both params = 0 -> existing behavior preserved."""

    def test_backward_compat_both_zero(self):
        """With flow_sensitivity=0 and earning_motive_premium=0, no change."""
        # VBT: flow_sensitivity=0 -> no ask widening even with outflow
        vbt = _make_vbt(
            M=Decimal("0.80"),
            O=Decimal("0.10"),
            flow_sensitivity=Decimal("0"),
        )
        vbt.cumulative_outflow = Decimal("100")
        vbt.cumulative_inflow = Decimal("0")
        vbt.recompute_quotes()

        expected_ask = Decimal("0.80") + Decimal("0.05")  # M + O/2
        assert vbt.A == expected_ask, (
            f"With flow_sensitivity=0, ask should be {expected_ask}, got {vbt.A}"
        )

        # TradeGate: earning_motive_premium=0 -> no extra premium
        params = RiskAssessmentParams(
            earning_motive_premium=Decimal("0.0"),
        )
        assessor = RiskAssessor(params)
        # Just verify it constructs without error and trade_gate has the field
        assert assessor.trade_gate.earning_motive_premium == Decimal("0.0")

    def test_vbt_state_defaults(self):
        """New VBTState fields default to zero."""
        vbt = VBTState(bucket_id="test")
        assert vbt.cumulative_outflow == Decimal(0)
        assert vbt.cumulative_inflow == Decimal(0)
        assert vbt.flow_sensitivity == Decimal(0)

    def test_risk_params_default(self):
        """RiskAssessmentParams.earning_motive_premium defaults to 0."""
        params = RiskAssessmentParams()
        assert params.earning_motive_premium == Decimal("0.0")
