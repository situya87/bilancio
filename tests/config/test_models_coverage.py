"""Comprehensive tests for configuration models to maximize coverage.

Covers all dataclass validators, default values, property methods, and edge cases
that are not already covered by test_models.py.
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from bilancio.config.models import (
    AgentSpec,
    BalancedDealerConfig,
    ClientPayment,
    CreateDeliveryObligation,
    CreatePayable,
    CreateStock,
    DealerBucketConfig,
    DealerConfig,
    DealerOrderFlowConfig,
    DealerTraderPolicyConfig,
    DepositCash,
    ExportConfig,
    GeneratorCompileConfig,
    MintCash,
    MintReserves,
    PolicyOverrides,
    RingExplorerGeneratorConfig,
    RingExplorerInequalityConfig,
    RingExplorerLiquidityAllocation,
    RingExplorerLiquidityConfig,
    RingExplorerMaturityConfig,
    RingExplorerParamsModel,
    RiskAssessmentConfig,
    RunConfig,
    ScenarioConfig,
    ScheduledAction,
    ShowConfig,
    TransferCash,
    TransferClaim,
    TransferReserves,
    TransferStock,
    WithdrawCash,
)

# ──────────────────────────────────────────────────────────────────────
# PolicyOverrides
# ──────────────────────────────────────────────────────────────────────


class TestPolicyOverrides:
    def test_default_none(self):
        po = PolicyOverrides()
        assert po.mop_rank is None

    def test_with_mop_rank(self):
        po = PolicyOverrides(mop_rank={"household": ["bank_deposit", "cash"]})
        assert po.mop_rank["household"] == ["bank_deposit", "cash"]


# ──────────────────────────────────────────────────────────────────────
# AgentSpec
# ──────────────────────────────────────────────────────────────────────


class TestAgentSpecCoverage:
    @pytest.mark.parametrize(
        "kind",
        [
            "central_bank",
            "bank",
            "household",
            "firm",
            "treasury",
        ],
    )
    def test_all_valid_kinds(self, kind):
        agent = AgentSpec(id="A1", kind=kind, name="test")
        assert agent.kind == kind

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AgentSpec(id="A1", kind="bank")  # name missing


# ──────────────────────────────────────────────────────────────────────
# MintReserves
# ──────────────────────────────────────────────────────────────────────


class TestMintReserves:
    def test_valid(self):
        mr = MintReserves(to="B1", amount=Decimal("500"))
        assert mr.action == "mint_reserves"
        assert mr.to == "B1"
        assert mr.amount == Decimal("500")
        assert mr.alias is None

    def test_with_alias(self):
        mr = MintReserves(to="B1", amount=Decimal("500"), alias="res1")
        assert mr.alias == "res1"

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            MintReserves(to="B1", amount=Decimal("0"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            MintReserves(to="B1", amount=Decimal("-100"))


# ──────────────────────────────────────────────────────────────────────
# MintCash
# ──────────────────────────────────────────────────────────────────────


class TestMintCash:
    def test_with_alias(self):
        mc = MintCash(to="H1", amount=Decimal("100"), alias="c1")
        assert mc.alias == "c1"

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            MintCash(to="H1", amount=Decimal("0"))


# ──────────────────────────────────────────────────────────────────────
# TransferReserves
# ──────────────────────────────────────────────────────────────────────


class TestTransferReserves:
    def test_valid(self):
        tr = TransferReserves(from_bank="B1", to_bank="B2", amount=Decimal("200"))
        assert tr.action == "transfer_reserves"
        assert tr.from_bank == "B1"
        assert tr.to_bank == "B2"
        assert tr.amount == Decimal("200")

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            TransferReserves(from_bank="B1", to_bank="B2", amount=Decimal("0"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            TransferReserves(from_bank="B1", to_bank="B2", amount=Decimal("-1"))


# ──────────────────────────────────────────────────────────────────────
# TransferCash
# ──────────────────────────────────────────────────────────────────────


class TestTransferCash:
    def test_valid(self):
        tc = TransferCash(from_agent="A1", to_agent="A2", amount=Decimal("50"))
        assert tc.action == "transfer_cash"
        assert tc.from_agent == "A1"
        assert tc.to_agent == "A2"

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            TransferCash(from_agent="A1", to_agent="A2", amount=Decimal("0"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            TransferCash(from_agent="A1", to_agent="A2", amount=Decimal("-10"))


# ──────────────────────────────────────────────────────────────────────
# DepositCash
# ──────────────────────────────────────────────────────────────────────


class TestDepositCash:
    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DepositCash(customer="H1", bank="B1", amount=Decimal("0"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DepositCash(customer="H1", bank="B1", amount=Decimal("-50"))


# ──────────────────────────────────────────────────────────────────────
# WithdrawCash
# ──────────────────────────────────────────────────────────────────────


class TestWithdrawCash:
    def test_valid(self):
        wc = WithdrawCash(customer="H1", bank="B1", amount=Decimal("200"))
        assert wc.action == "withdraw_cash"
        assert wc.customer == "H1"
        assert wc.bank == "B1"
        assert wc.amount == Decimal("200")

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            WithdrawCash(customer="H1", bank="B1", amount=Decimal("0"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            WithdrawCash(customer="H1", bank="B1", amount=Decimal("-1"))


# ──────────────────────────────────────────────────────────────────────
# ClientPayment
# ──────────────────────────────────────────────────────────────────────


class TestClientPayment:
    def test_valid(self):
        cp = ClientPayment(payer="H1", payee="H2", amount=Decimal("100"))
        assert cp.action == "client_payment"
        assert cp.payer == "H1"
        assert cp.payee == "H2"

    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            ClientPayment(payer="H1", payee="H2", amount=Decimal("0"))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            ClientPayment(payer="H1", payee="H2", amount=Decimal("-5"))


# ──────────────────────────────────────────────────────────────────────
# CreateStock (additional coverage)
# ──────────────────────────────────────────────────────────────────────


class TestCreateStockCoverage:
    def test_negative_quantity_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            CreateStock(owner="F1", sku="W", quantity=-1, unit_price=Decimal("10"))

    def test_negative_unit_price_rejected(self):
        with pytest.raises(ValidationError, match="negative"):
            CreateStock(owner="F1", sku="W", quantity=5, unit_price=Decimal("-1"))

    def test_zero_unit_price_allowed(self):
        cs = CreateStock(owner="F1", sku="W", quantity=5, unit_price=Decimal("0"))
        assert cs.unit_price == Decimal("0")


# ──────────────────────────────────────────────────────────────────────
# TransferStock
# ──────────────────────────────────────────────────────────────────────


class TestTransferStock:
    def test_valid(self):
        ts = TransferStock(from_agent="F1", to_agent="F2", sku="W", quantity=10)
        assert ts.action == "transfer_stock"
        assert ts.from_agent == "F1"
        assert ts.to_agent == "F2"
        assert ts.sku == "W"
        assert ts.quantity == 10

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            TransferStock(from_agent="F1", to_agent="F2", sku="W", quantity=0)

    def test_negative_quantity_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            TransferStock(from_agent="F1", to_agent="F2", sku="W", quantity=-5)


# ──────────────────────────────────────────────────────────────────────
# CreateDeliveryObligation (additional coverage)
# ──────────────────────────────────────────────────────────────────────


class TestCreateDeliveryObligationCoverage:
    def test_negative_quantity_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            CreateDeliveryObligation(
                **{
                    "from": "F1",
                    "to": "H1",
                    "sku": "W",
                    "quantity": -1,
                    "unit_price": Decimal("10"),
                    "due_day": 1,
                }
            )

    def test_negative_unit_price_rejected(self):
        with pytest.raises(ValidationError, match="negative"):
            CreateDeliveryObligation(
                **{
                    "from": "F1",
                    "to": "H1",
                    "sku": "W",
                    "quantity": 5,
                    "unit_price": Decimal("-1"),
                    "due_day": 1,
                }
            )

    def test_zero_unit_price_allowed(self):
        d = CreateDeliveryObligation(
            **{
                "from": "F1",
                "to": "H1",
                "sku": "W",
                "quantity": 5,
                "unit_price": Decimal("0"),
                "due_day": 1,
            }
        )
        assert d.unit_price == Decimal("0")

    def test_negative_due_day_rejected(self):
        with pytest.raises(ValidationError, match="negative"):
            CreateDeliveryObligation(
                **{
                    "from": "F1",
                    "to": "H1",
                    "sku": "W",
                    "quantity": 5,
                    "unit_price": Decimal("10"),
                    "due_day": -1,
                }
            )

    def test_zero_due_day_allowed(self):
        d = CreateDeliveryObligation(
            **{
                "from": "F1",
                "to": "H1",
                "sku": "W",
                "quantity": 5,
                "unit_price": Decimal("10"),
                "due_day": 0,
            }
        )
        assert d.due_day == 0

    def test_with_alias(self):
        d = CreateDeliveryObligation(
            **{
                "from": "F1",
                "to": "H1",
                "sku": "W",
                "quantity": 5,
                "unit_price": Decimal("10"),
                "due_day": 3,
                "alias": "del1",
            }
        )
        assert d.alias == "del1"


# ──────────────────────────────────────────────────────────────────────
# CreatePayable (additional coverage)
# ──────────────────────────────────────────────────────────────────────


class TestCreatePayableCoverage:
    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            CreatePayable(
                **{
                    "from": "H1",
                    "to": "H2",
                    "amount": Decimal("0"),
                    "due_day": 1,
                }
            )

    def test_with_alias_and_maturity_distance(self):
        p = CreatePayable(
            **{
                "from": "H1",
                "to": "H2",
                "amount": Decimal("100"),
                "due_day": 5,
                "alias": "pay1",
                "maturity_distance": 3,
            }
        )
        assert p.alias == "pay1"
        assert p.maturity_distance == 3

    def test_maturity_distance_defaults_none(self):
        p = CreatePayable(
            **{
                "from": "H1",
                "to": "H2",
                "amount": Decimal("100"),
                "due_day": 5,
            }
        )
        assert p.maturity_distance is None

    def test_zero_due_day_allowed(self):
        p = CreatePayable(
            **{
                "from": "H1",
                "to": "H2",
                "amount": Decimal("100"),
                "due_day": 0,
            }
        )
        assert p.due_day == 0


# ──────────────────────────────────────────────────────────────────────
# TransferClaim
# ──────────────────────────────────────────────────────────────────────


class TestTransferClaim:
    def test_valid_with_alias(self):
        tc = TransferClaim(contract_alias="pay1", to_agent="H2")
        assert tc.action == "transfer_claim"
        assert tc.contract_alias == "pay1"
        assert tc.contract_id is None
        assert tc.to_agent == "H2"

    def test_valid_with_contract_id(self):
        tc = TransferClaim(contract_id="abc-123", to_agent="H2")
        assert tc.contract_id == "abc-123"
        assert tc.contract_alias is None

    def test_valid_with_both_alias_and_id(self):
        tc = TransferClaim(contract_alias="pay1", contract_id="abc-123", to_agent="H2")
        assert tc.contract_alias == "pay1"
        assert tc.contract_id == "abc-123"

    def test_neither_alias_nor_id_rejected(self):
        with pytest.raises(
            ValidationError, match="contract_alias.*contract_id|contract_id.*contract_alias"
        ):
            TransferClaim(to_agent="H2")

    def test_empty_to_agent_rejected(self):
        with pytest.raises(ValidationError):
            TransferClaim(contract_alias="pay1", to_agent="")


# ──────────────────────────────────────────────────────────────────────
# ScheduledAction
# ──────────────────────────────────────────────────────────────────────


class TestScheduledAction:
    def test_valid(self):
        sa = ScheduledAction(day=1, action={"action": "mint_cash", "to": "H1", "amount": 100})
        assert sa.day == 1
        assert sa.action["action"] == "mint_cash"

    def test_day_zero_rejected(self):
        with pytest.raises(ValidationError, match=">="):
            ScheduledAction(day=0, action={"action": "mint_cash"})

    def test_negative_day_rejected(self):
        with pytest.raises(ValidationError, match=">="):
            ScheduledAction(day=-1, action={"action": "mint_cash"})

    def test_day_1_is_minimum(self):
        sa = ScheduledAction(day=1, action={"action": "mint_cash"})
        assert sa.day == 1


# ──────────────────────────────────────────────────────────────────────
# ShowConfig
# ──────────────────────────────────────────────────────────────────────


class TestShowConfig:
    def test_defaults(self):
        sc = ShowConfig()
        assert sc.balances is None
        assert sc.events == "detailed"

    def test_summary_events(self):
        sc = ShowConfig(events="summary")
        assert sc.events == "summary"

    def test_table_events(self):
        sc = ShowConfig(events="table")
        assert sc.events == "table"

    def test_invalid_events_mode(self):
        with pytest.raises(ValidationError):
            ShowConfig(events="invalid")

    def test_with_balances(self):
        sc = ShowConfig(balances=["A1", "A2"])
        assert sc.balances == ["A1", "A2"]


# ──────────────────────────────────────────────────────────────────────
# ExportConfig
# ──────────────────────────────────────────────────────────────────────


class TestExportConfig:
    def test_defaults(self):
        ec = ExportConfig()
        assert ec.balances_csv is None
        assert ec.events_jsonl is None

    def test_with_paths(self):
        ec = ExportConfig(balances_csv="out/b.csv", events_jsonl="out/e.jsonl")
        assert ec.balances_csv == "out/b.csv"
        assert ec.events_jsonl == "out/e.jsonl"


# ──────────────────────────────────────────────────────────────────────
# RunConfig (additional coverage)
# ──────────────────────────────────────────────────────────────────────


class TestRunConfigCoverage:
    def test_zero_max_days_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RunConfig(max_days=0)

    def test_zero_quiet_days_allowed(self):
        rc = RunConfig(quiet_days=0)
        assert rc.quiet_days == 0

    def test_rollover_enabled_default_false(self):
        rc = RunConfig()
        assert rc.rollover_enabled is False

    def test_rollover_enabled_true(self):
        rc = RunConfig(rollover_enabled=True)
        assert rc.rollover_enabled is True

    def test_step_mode(self):
        rc = RunConfig(mode="step")
        assert rc.mode == "step"

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            RunConfig(mode="continuous")

    def test_invalid_default_handling(self):
        with pytest.raises(ValidationError):
            RunConfig(default_handling="ignore")


# ──────────────────────────────────────────────────────────────────────
# DealerBucketConfig
# ──────────────────────────────────────────────────────────────────────


class TestDealerBucketConfig:
    def test_valid(self):
        bc = DealerBucketConfig(tau_min=1, tau_max=5)
        assert bc.tau_min == 1
        assert bc.tau_max == 5
        assert bc.M == Decimal("1.0")
        assert bc.O == Decimal("0.30")

    def test_custom_mid_and_spread(self):
        bc = DealerBucketConfig(
            tau_min=1,
            tau_max=3,
            M=Decimal("0.95"),
            O=Decimal("0.10"),
        )
        assert bc.M == Decimal("0.95")
        assert bc.O == Decimal("0.10")

    def test_tau_min_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=0, tau_max=5)

    def test_tau_max_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=1, tau_max=0)

    def test_tau_min_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=-1, tau_max=5)

    def test_mid_price_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=1, tau_max=5, M=Decimal("0"))

    def test_mid_price_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=1, tau_max=5, M=Decimal("-1"))

    def test_spread_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=1, tau_max=5, O=Decimal("0"))

    def test_spread_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerBucketConfig(tau_min=1, tau_max=5, O=Decimal("-0.1"))

    def test_tau_min_greater_than_tau_max_rejected(self):
        with pytest.raises(ValidationError, match="tau_min.*tau_max"):
            DealerBucketConfig(tau_min=10, tau_max=5)

    def test_tau_min_equals_tau_max_allowed(self):
        bc = DealerBucketConfig(tau_min=5, tau_max=5)
        assert bc.tau_min == bc.tau_max


# ──────────────────────────────────────────────────────────────────────
# DealerOrderFlowConfig
# ──────────────────────────────────────────────────────────────────────


class TestDealerOrderFlowConfig:
    def test_defaults(self):
        ofc = DealerOrderFlowConfig()
        assert ofc.pi_sell == Decimal("0.5")
        assert ofc.N_max == 3

    def test_pi_sell_boundary_zero(self):
        ofc = DealerOrderFlowConfig(pi_sell=Decimal("0"))
        assert ofc.pi_sell == Decimal("0")

    def test_pi_sell_boundary_one(self):
        ofc = DealerOrderFlowConfig(pi_sell=Decimal("1"))
        assert ofc.pi_sell == Decimal("1")

    def test_pi_sell_above_one_rejected(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DealerOrderFlowConfig(pi_sell=Decimal("1.1"))

    def test_pi_sell_negative_rejected(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DealerOrderFlowConfig(pi_sell=Decimal("-0.1"))

    def test_n_max_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerOrderFlowConfig(N_max=0)

    def test_n_max_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerOrderFlowConfig(N_max=-1)

    def test_n_max_one_allowed(self):
        ofc = DealerOrderFlowConfig(N_max=1)
        assert ofc.N_max == 1


# ──────────────────────────────────────────────────────────────────────
# DealerTraderPolicyConfig
# ──────────────────────────────────────────────────────────────────────


class TestDealerTraderPolicyConfig:
    def test_defaults(self):
        tp = DealerTraderPolicyConfig()
        assert tp.horizon_H == 3
        assert tp.buffer_B == Decimal("1.0")

    def test_horizon_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerTraderPolicyConfig(horizon_H=0)

    def test_horizon_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerTraderPolicyConfig(horizon_H=-1)

    def test_buffer_negative_rejected(self):
        with pytest.raises(ValidationError, match="negative"):
            DealerTraderPolicyConfig(buffer_B=Decimal("-0.1"))

    def test_buffer_zero_allowed(self):
        tp = DealerTraderPolicyConfig(buffer_B=Decimal("0"))
        assert tp.buffer_B == Decimal("0")

    def test_custom_horizon_valid(self):
        tp = DealerTraderPolicyConfig(horizon_H=5)
        assert tp.horizon_H == 5


# ──────────────────────────────────────────────────────────────────────
# RiskAssessmentConfig
# ──────────────────────────────────────────────────────────────────────


class TestRiskAssessmentConfig:
    def test_defaults(self):
        ra = RiskAssessmentConfig()
        assert ra.enabled is True
        assert ra.lookback_window == 5
        assert ra.smoothing_alpha == Decimal("1.0")
        assert ra.base_risk_premium == Decimal("0.02")
        assert ra.urgency_sensitivity == Decimal("0.10")
        assert ra.use_issuer_specific is False
        assert ra.buy_premium_multiplier == Decimal("1.0")

    def test_lookback_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RiskAssessmentConfig(lookback_window=0)

    def test_lookback_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RiskAssessmentConfig(lookback_window=-1)

    def test_smoothing_alpha_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RiskAssessmentConfig(smoothing_alpha=Decimal("0"))

    def test_smoothing_alpha_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RiskAssessmentConfig(smoothing_alpha=Decimal("-0.5"))

    def test_base_risk_premium_negative_rejected(self):
        with pytest.raises(ValidationError, match="negative"):
            RiskAssessmentConfig(base_risk_premium=Decimal("-0.01"))

    def test_urgency_sensitivity_negative_rejected(self):
        with pytest.raises(ValidationError, match="negative"):
            RiskAssessmentConfig(urgency_sensitivity=Decimal("-0.01"))

    def test_base_risk_premium_zero_allowed(self):
        ra = RiskAssessmentConfig(base_risk_premium=Decimal("0"))
        assert ra.base_risk_premium == Decimal("0")

    def test_urgency_sensitivity_zero_allowed(self):
        ra = RiskAssessmentConfig(urgency_sensitivity=Decimal("0"))
        assert ra.urgency_sensitivity == Decimal("0")

    def test_buy_premium_multiplier_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RiskAssessmentConfig(buy_premium_multiplier=Decimal("0"))

    def test_buy_premium_multiplier_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RiskAssessmentConfig(buy_premium_multiplier=Decimal("-1"))

    def test_enabled_false(self):
        ra = RiskAssessmentConfig(enabled=False)
        assert ra.enabled is False

    def test_use_issuer_specific_true(self):
        ra = RiskAssessmentConfig(use_issuer_specific=True)
        assert ra.use_issuer_specific is True

    def test_custom_lookback_valid(self):
        ra = RiskAssessmentConfig(lookback_window=10)
        assert ra.lookback_window == 10

    def test_custom_smoothing_alpha_valid(self):
        ra = RiskAssessmentConfig(smoothing_alpha=Decimal("2.5"))
        assert ra.smoothing_alpha == Decimal("2.5")

    def test_custom_buy_premium_multiplier_valid(self):
        ra = RiskAssessmentConfig(buy_premium_multiplier=Decimal("1.5"))
        assert ra.buy_premium_multiplier == Decimal("1.5")


# ──────────────────────────────────────────────────────────────────────
# DealerConfig
# ──────────────────────────────────────────────────────────────────────


class TestDealerConfig:
    def test_defaults(self):
        dc = DealerConfig()
        assert dc.enabled is False
        assert dc.ticket_size == Decimal("1")
        assert dc.dealer_share == Decimal("0.25")
        assert dc.vbt_share == Decimal("0.50")
        # Default buckets should be set by model_validator
        assert dc.buckets is not None
        assert "short" in dc.buckets
        assert "mid" in dc.buckets
        assert "long" in dc.buckets

    def test_default_bucket_short(self):
        dc = DealerConfig()
        short = dc.buckets["short"]
        assert short.tau_min == 1
        assert short.tau_max == 3
        assert short.M == Decimal("1.0")
        assert short.O == Decimal("0.20")

    def test_default_bucket_mid(self):
        dc = DealerConfig()
        mid = dc.buckets["mid"]
        assert mid.tau_min == 4
        assert mid.tau_max == 8
        assert mid.M == Decimal("1.0")
        assert mid.O == Decimal("0.30")

    def test_default_bucket_long(self):
        dc = DealerConfig()
        lng = dc.buckets["long"]
        assert lng.tau_min == 9
        assert lng.tau_max == 999
        assert lng.M == Decimal("1.0")
        assert lng.O == Decimal("0.40")

    def test_custom_buckets_not_overridden(self):
        custom_buckets = {
            "only": DealerBucketConfig(
                tau_min=1,
                tau_max=999,
                M=Decimal("0.90"),
                O=Decimal("0.50"),
            ),
        }
        dc = DealerConfig(buckets=custom_buckets)
        assert len(dc.buckets) == 1
        assert "only" in dc.buckets
        assert dc.buckets["only"].M == Decimal("0.90")

    def test_enabled_true(self):
        dc = DealerConfig(enabled=True)
        assert dc.enabled is True

    def test_ticket_size_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerConfig(ticket_size=Decimal("0"))

    def test_ticket_size_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            DealerConfig(ticket_size=Decimal("-1"))

    def test_dealer_share_negative_rejected(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DealerConfig(dealer_share=Decimal("-0.1"))

    def test_dealer_share_above_one_rejected(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DealerConfig(dealer_share=Decimal("1.1"))

    def test_dealer_share_zero_allowed(self):
        dc = DealerConfig(dealer_share=Decimal("0"))
        assert dc.dealer_share == Decimal("0")

    def test_dealer_share_one_allowed(self):
        dc = DealerConfig(dealer_share=Decimal("1"))
        assert dc.dealer_share == Decimal("1")

    def test_vbt_share_negative_rejected(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DealerConfig(vbt_share=Decimal("-0.1"))

    def test_vbt_share_above_one_rejected(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DealerConfig(vbt_share=Decimal("1.1"))

    def test_order_flow_defaults(self):
        dc = DealerConfig()
        assert dc.order_flow.pi_sell == Decimal("0.5")
        assert dc.order_flow.N_max == 3

    def test_trader_policy_defaults(self):
        dc = DealerConfig()
        assert dc.trader_policy.horizon_H == 3
        assert dc.trader_policy.buffer_B == Decimal("1.0")

    def test_risk_assessment_defaults(self):
        dc = DealerConfig()
        assert dc.risk_assessment.enabled is True
        assert dc.risk_assessment.lookback_window == 5

    def test_custom_ticket_size_valid(self):
        dc = DealerConfig(ticket_size=Decimal("5"))
        assert dc.ticket_size == Decimal("5")


# ──────────────────────────────────────────────────────────────────────
# BalancedDealerConfig
# ──────────────────────────────────────────────────────────────────────


class TestBalancedDealerConfig:
    def test_defaults(self):
        bdc = BalancedDealerConfig()
        assert bdc.enabled is False
        assert bdc.face_value == Decimal("20")
        assert bdc.outside_mid_ratio == Decimal("0.75")
        assert bdc.big_entity_share == Decimal("0.25")
        assert bdc.vbt_share_per_bucket == Decimal("0.20")
        assert bdc.dealer_share_per_bucket == Decimal("0.05")
        assert bdc.rollover_enabled is True
        assert bdc.mode == "active"

    def test_passive_mode(self):
        bdc = BalancedDealerConfig(mode="passive")
        assert bdc.mode == "passive"

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            BalancedDealerConfig(mode="hybrid")

    def test_face_value_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            BalancedDealerConfig(face_value=Decimal("0"))

    def test_face_value_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            BalancedDealerConfig(face_value=Decimal("-10"))

    def test_outside_mid_ratio_zero_rejected(self):
        with pytest.raises(ValidationError, match="outside_mid_ratio"):
            BalancedDealerConfig(outside_mid_ratio=Decimal("0"))

    def test_outside_mid_ratio_above_one_rejected(self):
        with pytest.raises(ValidationError, match="outside_mid_ratio"):
            BalancedDealerConfig(outside_mid_ratio=Decimal("1.1"))

    def test_outside_mid_ratio_one_allowed(self):
        bdc = BalancedDealerConfig(outside_mid_ratio=Decimal("1"))
        assert bdc.outside_mid_ratio == Decimal("1")

    def test_outside_mid_ratio_negative_rejected(self):
        with pytest.raises(ValidationError, match="outside_mid_ratio"):
            BalancedDealerConfig(outside_mid_ratio=Decimal("-0.5"))

    def test_big_entity_share_negative_rejected(self):
        with pytest.raises(ValidationError, match="big_entity_share"):
            BalancedDealerConfig(big_entity_share=Decimal("-0.1"))

    def test_big_entity_share_one_rejected(self):
        with pytest.raises(ValidationError, match="big_entity_share"):
            BalancedDealerConfig(big_entity_share=Decimal("1"))

    def test_big_entity_share_zero_allowed(self):
        bdc = BalancedDealerConfig(big_entity_share=Decimal("0"))
        assert bdc.big_entity_share == Decimal("0")

    def test_vbt_share_zero_rejected(self):
        with pytest.raises(ValidationError, match="vbt_share_per_bucket"):
            BalancedDealerConfig(vbt_share_per_bucket=Decimal("0"))

    def test_vbt_share_one_rejected(self):
        with pytest.raises(ValidationError, match="vbt_share_per_bucket"):
            BalancedDealerConfig(vbt_share_per_bucket=Decimal("1"))

    def test_vbt_share_negative_rejected(self):
        with pytest.raises(ValidationError, match="vbt_share_per_bucket"):
            BalancedDealerConfig(vbt_share_per_bucket=Decimal("-0.1"))

    def test_dealer_share_zero_rejected(self):
        with pytest.raises(ValidationError, match="dealer_share_per_bucket"):
            BalancedDealerConfig(dealer_share_per_bucket=Decimal("0"))

    def test_dealer_share_one_rejected(self):
        with pytest.raises(ValidationError, match="dealer_share_per_bucket"):
            BalancedDealerConfig(dealer_share_per_bucket=Decimal("1"))

    def test_dealer_share_negative_rejected(self):
        with pytest.raises(ValidationError, match="dealer_share_per_bucket"):
            BalancedDealerConfig(dealer_share_per_bucket=Decimal("-0.1"))

    def test_enabled_true(self):
        bdc = BalancedDealerConfig(enabled=True)
        assert bdc.enabled is True

    def test_rollover_disabled(self):
        bdc = BalancedDealerConfig(rollover_enabled=False)
        assert bdc.rollover_enabled is False

    def test_custom_face_value_valid(self):
        bdc = BalancedDealerConfig(face_value=Decimal("50"))
        assert bdc.face_value == Decimal("50")

    def test_custom_vbt_share_valid(self):
        bdc = BalancedDealerConfig(vbt_share_per_bucket=Decimal("0.5"))
        assert bdc.vbt_share_per_bucket == Decimal("0.5")

    def test_custom_dealer_share_valid(self):
        bdc = BalancedDealerConfig(dealer_share_per_bucket=Decimal("0.3"))
        assert bdc.dealer_share_per_bucket == Decimal("0.3")


# ──────────────────────────────────────────────────────────────────────
# ScenarioConfig (additional coverage)
# ──────────────────────────────────────────────────────────────────────


class TestScenarioConfigCoverage:
    def _minimal_agents(self):
        return [{"id": "CB", "kind": "central_bank", "name": "CB"}]

    def test_with_dealer_config(self):
        sc = ScenarioConfig(
            name="Dealer Scenario",
            agents=self._minimal_agents(),
            dealer={"enabled": True},
        )
        assert sc.dealer is not None
        assert sc.dealer.enabled is True

    def test_with_balanced_dealer_config(self):
        sc = ScenarioConfig(
            name="Balanced",
            agents=self._minimal_agents(),
            balanced_dealer={"enabled": True, "mode": "passive"},
        )
        assert sc.balanced_dealer is not None
        assert sc.balanced_dealer.mode == "passive"

    def test_with_scheduled_actions(self):
        sc = ScenarioConfig(
            name="Scheduled",
            agents=self._minimal_agents(),
            scheduled_actions=[
                {"day": 1, "action": {"action": "mint_cash", "to": "CB", "amount": 100}},
            ],
        )
        assert len(sc.scheduled_actions) == 1
        assert sc.scheduled_actions[0].day == 1

    def test_defaults_for_optional_fields(self):
        sc = ScenarioConfig(name="Minimal", agents=self._minimal_agents())
        assert sc.description is None
        assert sc.policy_overrides is None
        assert sc.dealer is None
        assert sc.balanced_dealer is None
        assert sc.initial_actions == []
        assert sc.scheduled_actions == []
        assert sc.run.mode == "until_stable"

    def test_empty_agents_allowed(self):
        # Empty list passes the uniqueness check (no duplicates)
        sc = ScenarioConfig(name="Empty", agents=[])
        assert len(sc.agents) == 0


# ──────────────────────────────────────────────────────────────────────
# RingExplorerLiquidityAllocation
# ──────────────────────────────────────────────────────────────────────


class TestRingExplorerLiquidityAllocation:
    def test_default_uniform(self):
        la = RingExplorerLiquidityAllocation()
        assert la.mode == "uniform"
        assert la.agent is None
        assert la.vector is None

    def test_uniform_mode(self):
        la = RingExplorerLiquidityAllocation(mode="uniform")
        assert la.mode == "uniform"

    def test_single_at_mode_valid(self):
        la = RingExplorerLiquidityAllocation(mode="single_at", agent="F0")
        assert la.mode == "single_at"
        assert la.agent == "F0"

    def test_single_at_mode_missing_agent_rejected(self):
        with pytest.raises(ValidationError, match="single_at"):
            RingExplorerLiquidityAllocation(mode="single_at")

    def test_vector_mode_valid(self):
        la = RingExplorerLiquidityAllocation(
            mode="vector",
            vector=[Decimal("1"), Decimal("2"), Decimal("3")],
        )
        assert la.mode == "vector"
        assert len(la.vector) == 3

    def test_vector_mode_missing_vector_rejected(self):
        with pytest.raises(ValidationError, match="vector"):
            RingExplorerLiquidityAllocation(mode="vector")

    def test_vector_mode_empty_vector_rejected(self):
        with pytest.raises(ValidationError, match="vector"):
            RingExplorerLiquidityAllocation(mode="vector", vector=[])

    def test_vector_mode_non_positive_values_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RingExplorerLiquidityAllocation(
                mode="vector",
                vector=[Decimal("1"), Decimal("0"), Decimal("3")],
            )

    def test_vector_mode_negative_values_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RingExplorerLiquidityAllocation(
                mode="vector",
                vector=[Decimal("1"), Decimal("-1")],
            )


# ──────────────────────────────────────────────────────────────────────
# RingExplorerLiquidityConfig
# ──────────────────────────────────────────────────────────────────────


class TestRingExplorerLiquidityConfig:
    def test_defaults(self):
        lc = RingExplorerLiquidityConfig()
        assert lc.total is None
        assert lc.allocation.mode == "uniform"

    def test_with_total(self):
        lc = RingExplorerLiquidityConfig(total=Decimal("1000"))
        assert lc.total == Decimal("1000")

    def test_zero_total_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RingExplorerLiquidityConfig(total=Decimal("0"))

    def test_negative_total_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RingExplorerLiquidityConfig(total=Decimal("-100"))

    def test_none_total_allowed(self):
        lc = RingExplorerLiquidityConfig(total=None)
        assert lc.total is None


# ──────────────────────────────────────────────────────────────────────
# RingExplorerInequalityConfig
# ──────────────────────────────────────────────────────────────────────


class TestRingExplorerInequalityConfig:
    def test_defaults(self):
        ic = RingExplorerInequalityConfig()
        assert ic.scheme == "dirichlet"
        assert ic.concentration == Decimal("1")
        assert ic.monotonicity == Decimal("0")

    def test_concentration_zero_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RingExplorerInequalityConfig(concentration=Decimal("0"))

    def test_concentration_negative_rejected(self):
        with pytest.raises(ValidationError, match="positive"):
            RingExplorerInequalityConfig(concentration=Decimal("-1"))

    def test_monotonicity_bounds(self):
        ic_neg = RingExplorerInequalityConfig(monotonicity=Decimal("-1"))
        assert ic_neg.monotonicity == Decimal("-1")
        ic_pos = RingExplorerInequalityConfig(monotonicity=Decimal("1"))
        assert ic_pos.monotonicity == Decimal("1")

    def test_monotonicity_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerInequalityConfig(monotonicity=Decimal("1.1"))
        with pytest.raises(ValidationError):
            RingExplorerInequalityConfig(monotonicity=Decimal("-1.1"))

    def test_custom_concentration_valid(self):
        ic = RingExplorerInequalityConfig(concentration=Decimal("5"))
        assert ic.concentration == Decimal("5")


# ──────────────────────────────────────────────────────────────────────
# RingExplorerMaturityConfig
# ──────────────────────────────────────────────────────────────────────


class TestRingExplorerMaturityConfig:
    def test_defaults(self):
        mc = RingExplorerMaturityConfig()
        assert mc.days == 1
        assert mc.mode == "lead_lag"
        assert mc.mu == Decimal("0")

    def test_custom_values(self):
        mc = RingExplorerMaturityConfig(days=10, mu=Decimal("0.5"))
        assert mc.days == 10
        assert mc.mu == Decimal("0.5")

    def test_days_zero_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerMaturityConfig(days=0)

    def test_days_negative_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerMaturityConfig(days=-1)

    def test_mu_boundary_zero(self):
        mc = RingExplorerMaturityConfig(mu=Decimal("0"))
        assert mc.mu == Decimal("0")

    def test_mu_boundary_one(self):
        mc = RingExplorerMaturityConfig(mu=Decimal("1"))
        assert mc.mu == Decimal("1")

    def test_mu_above_one_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerMaturityConfig(mu=Decimal("1.1"))

    def test_mu_negative_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerMaturityConfig(mu=Decimal("-0.1"))


# ──────────────────────────────────────────────────────────────────────
# RingExplorerParamsModel
# ──────────────────────────────────────────────────────────────────────


class TestRingExplorerParamsModel:
    def test_minimal(self):
        p = RingExplorerParamsModel(kappa=Decimal("1"))
        assert p.n_agents == 5
        assert p.seed == 42
        assert p.kappa == Decimal("1")
        assert p.currency == "USD"
        assert p.Q_total is None
        assert p.policy_overrides is None

    def test_custom_values(self):
        p = RingExplorerParamsModel(
            n_agents=10,
            seed=123,
            kappa=Decimal("0.5"),
            currency="EUR",
            Q_total=Decimal("500"),
        )
        assert p.n_agents == 10
        assert p.seed == 123
        assert p.currency == "EUR"
        assert p.Q_total == Decimal("500")

    def test_n_agents_too_small_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerParamsModel(n_agents=2, kappa=Decimal("1"))

    def test_n_agents_minimum_three(self):
        p = RingExplorerParamsModel(n_agents=3, kappa=Decimal("1"))
        assert p.n_agents == 3

    def test_kappa_zero_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerParamsModel(kappa=Decimal("0"))

    def test_kappa_negative_rejected(self):
        with pytest.raises(ValidationError):
            RingExplorerParamsModel(kappa=Decimal("-1"))

    def test_with_policy_overrides(self):
        p = RingExplorerParamsModel(
            kappa=Decimal("1"),
            policy_overrides=PolicyOverrides(mop_rank={"firm": ["cash"]}),
        )
        assert p.policy_overrides.mop_rank["firm"] == ["cash"]

    def test_nested_defaults(self):
        p = RingExplorerParamsModel(kappa=Decimal("1"))
        assert p.liquidity.total is None
        assert p.liquidity.allocation.mode == "uniform"
        assert p.inequality.scheme == "dirichlet"
        assert p.inequality.concentration == Decimal("1")
        assert p.maturity.days == 1
        assert p.maturity.mode == "lead_lag"


# ──────────────────────────────────────────────────────────────────────
# GeneratorCompileConfig
# ──────────────────────────────────────────────────────────────────────


class TestGeneratorCompileConfig:
    def test_defaults(self):
        gc = GeneratorCompileConfig()
        assert gc.out_dir is None
        assert gc.emit_yaml is True

    def test_custom(self):
        gc = GeneratorCompileConfig(out_dir="/tmp/out", emit_yaml=False)
        assert gc.out_dir == "/tmp/out"
        assert gc.emit_yaml is False


# ──────────────────────────────────────────────────────────────────────
# RingExplorerGeneratorConfig
# ──────────────────────────────────────────────────────────────────────


class TestRingExplorerGeneratorConfig:
    def test_valid(self):
        gc = RingExplorerGeneratorConfig(
            name_prefix="test",
            params={"kappa": "1"},
        )
        assert gc.version == 1
        assert gc.generator == "ring_explorer_v1"
        assert gc.name_prefix == "test"
        assert gc.params.kappa == Decimal("1")

    def test_unsupported_version_rejected(self):
        with pytest.raises(ValidationError, match="version"):
            RingExplorerGeneratorConfig(
                version=2,
                name_prefix="test",
                params={"kappa": "1"},
            )

    def test_version_1_accepted(self):
        gc = RingExplorerGeneratorConfig(
            version=1,
            name_prefix="test",
            params={"kappa": "1"},
        )
        assert gc.version == 1

    def test_default_compile_config(self):
        gc = RingExplorerGeneratorConfig(
            name_prefix="test",
            params={"kappa": "1"},
        )
        assert gc.compile.out_dir is None
        assert gc.compile.emit_yaml is True

    def test_custom_compile_config(self):
        gc = RingExplorerGeneratorConfig(
            name_prefix="test",
            params={"kappa": "1"},
            compile={"out_dir": "/tmp/out", "emit_yaml": False},
        )
        assert gc.compile.out_dir == "/tmp/out"
        assert gc.compile.emit_yaml is False
