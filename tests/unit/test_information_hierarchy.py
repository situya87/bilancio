"""Tests for the hierarchical information/decision architecture (Phase 1).

Covers:
1. Sub-profile defaults and construction
2. Profile ↔ sub-profile round-trips
3. from_hierarchy() classmethod
4. Preset equivalence (LENDER_REALISTIC == LENDER_REALISTIC_V2)
5. Contextual view queries (system, counterparty, instrument, transaction)
6. View object immutability
7. Decision protocol defaults
8. Regression — existing test suite unaffected
"""

import dataclasses
from decimal import Decimal

import pytest

from bilancio.decision import (
    CounterpartyScreener,
    FixedMaturitySelector,
    FixedPortfolioStrategy,
    InstrumentSelector,
    LinearPricer,
    PortfolioStrategy,
    ThresholdScreener,
    TransactionPricer,
)
from bilancio.information import (
    LENDER_REALISTIC,
    LENDER_REALISTIC_V2,
    OMNISCIENT,
    AccessLevel,
    CategoryAccess,
    CounterpartyAccess,
    CounterpartyView,
    EstimationNoise,
    InformationProfile,
    InformationService,
    InstrumentAccess,
    InstrumentView,
    SystemAccess,
    SystemView,
    TransactionAccess,
    TransactionView,
)

# ═══════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════


def _build_system():
    """Build a minimal system for view query tests."""
    from bilancio.domain.agents.bank import Bank
    from bilancio.domain.agents.central_bank import CentralBank
    from bilancio.domain.agents.firm import Firm
    from bilancio.domain.agents.non_bank_lender import NonBankLender
    from bilancio.domain.instruments.base import InstrumentKind
    from bilancio.domain.instruments.credit import Payable
    from bilancio.engines.system import System

    system = System()
    cb = CentralBank(id="CB01", name="CB", kind="central_bank")
    bank = Bank(id="B01", name="Bank", kind="bank")
    lender = NonBankLender(id="NBL01", name="Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)

    system.mint_cash("NBL01", 10000)
    system.mint_cash("F01", 500)

    payable = Payable(
        id=system.new_contract_id("PAY"),
        kind=InstrumentKind.PAYABLE,
        amount=1000,
        denom="X",
        asset_holder_id="F02",
        liability_issuer_id="F01",
        due_day=3,
    )
    system.add_contract(payable)

    return system


# ═══════════════════════════════════════════════════════════════════════
# 1. Sub-profile defaults
# ═══════════════════════════════════════════════════════════════════════


class TestSubProfileDefaults:
    def test_system_access_defaults_all_perfect(self):
        s = SystemAccess()
        for f in dataclasses.fields(s):
            ca = getattr(s, f.name)
            assert ca.level == AccessLevel.PERFECT, f"{f.name} not PERFECT"

    def test_counterparty_access_defaults_all_perfect(self):
        c = CounterpartyAccess()
        for f in dataclasses.fields(c):
            ca = getattr(c, f.name)
            assert ca.level == AccessLevel.PERFECT, f"{f.name} not PERFECT"

    def test_instrument_access_defaults_all_perfect(self):
        i = InstrumentAccess()
        for f in dataclasses.fields(i):
            ca = getattr(i, f.name)
            assert ca.level == AccessLevel.PERFECT, f"{f.name} not PERFECT"

    def test_transaction_access_defaults_all_perfect(self):
        t = TransactionAccess()
        for f in dataclasses.fields(t):
            ca = getattr(t, f.name)
            assert ca.level == AccessLevel.PERFECT, f"{f.name} not PERFECT"

    def test_field_counts(self):
        assert len(dataclasses.fields(SystemAccess)) == 6
        assert len(dataclasses.fields(CounterpartyAccess)) == 11
        assert len(dataclasses.fields(InstrumentAccess)) == 4
        assert len(dataclasses.fields(TransactionAccess)) == 7
        # Total should be 28
        total = (
            len(dataclasses.fields(SystemAccess))
            + len(dataclasses.fields(CounterpartyAccess))
            + len(dataclasses.fields(InstrumentAccess))
            + len(dataclasses.fields(TransactionAccess))
        )
        assert total == 28

    def test_sub_profiles_are_frozen(self):
        s = SystemAccess()
        with pytest.raises(AttributeError):
            s.aggregate_default_rate = CategoryAccess(AccessLevel.NONE)  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 2. Profile → sub-profile properties
# ═══════════════════════════════════════════════════════════════════════


class TestProfileToSubProfile:
    def test_counterparty_cash_maps_correctly(self):
        p = InformationProfile(counterparty_cash=CategoryAccess(AccessLevel.NONE))
        assert p.counterparty.cash.level == AccessLevel.NONE

    def test_system_liquidity_maps_correctly(self):
        p = InformationProfile(system_liquidity=CategoryAccess(AccessLevel.NONE))
        assert p.system.system_liquidity.level == AccessLevel.NONE

    def test_dealer_quotes_maps_correctly(self):
        p = InformationProfile(dealer_quotes=CategoryAccess(AccessLevel.NONE))
        assert p.instrument.dealer_quotes.level == AccessLevel.NONE

    def test_bilateral_history_maps_correctly(self):
        p = InformationProfile(bilateral_history=CategoryAccess(AccessLevel.NONE))
        assert p.transaction.bilateral_history.level == AccessLevel.NONE

    def test_omniscient_all_sub_profiles_perfect(self):
        p = OMNISCIENT
        for f in dataclasses.fields(p.system):
            assert getattr(p.system, f.name).level == AccessLevel.PERFECT
        for f in dataclasses.fields(p.counterparty):
            assert getattr(p.counterparty, f.name).level == AccessLevel.PERFECT
        for f in dataclasses.fields(p.instrument):
            assert getattr(p.instrument, f.name).level == AccessLevel.PERFECT
        for f in dataclasses.fields(p.transaction):
            assert getattr(p.transaction, f.name).level == AccessLevel.PERFECT

    def test_lender_realistic_counterparty_cash_noisy(self):
        p = LENDER_REALISTIC
        assert p.counterparty.cash.level == AccessLevel.NOISY
        assert isinstance(p.counterparty.cash.noise, EstimationNoise)

    def test_lender_realistic_instrument_none(self):
        p = LENDER_REALISTIC
        assert p.instrument.dealer_quotes.level == AccessLevel.NONE
        assert p.instrument.vbt_anchors.level == AccessLevel.NONE


# ═══════════════════════════════════════════════════════════════════════
# 3. from_hierarchy() classmethod
# ═══════════════════════════════════════════════════════════════════════


class TestFromHierarchy:
    def test_no_args_equals_omniscient(self):
        p = InformationProfile.from_hierarchy()
        for f in dataclasses.fields(InformationProfile):
            v1 = getattr(p, f.name)
            v2 = getattr(OMNISCIENT, f.name)
            assert v1 == v2, f"{f.name}: {v1} != {v2}"

    def test_counterparty_cash_none_round_trip(self):
        p = InformationProfile.from_hierarchy(
            counterparty=CounterpartyAccess(cash=CategoryAccess(AccessLevel.NONE))
        )
        assert p.counterparty_cash.level == AccessLevel.NONE
        # Other counterparty fields remain PERFECT
        assert p.counterparty_assets.level == AccessLevel.PERFECT

    def test_system_and_transaction_together(self):
        p = InformationProfile.from_hierarchy(
            system=SystemAccess(system_liquidity=CategoryAccess(AccessLevel.NONE)),
            transaction=TransactionAccess(obligation_graph=CategoryAccess(AccessLevel.NONE)),
        )
        assert p.system_liquidity.level == AccessLevel.NONE
        assert p.obligation_graph.level == AccessLevel.NONE
        # Untouched fields stay PERFECT
        assert p.aggregate_default_rate.level == AccessLevel.PERFECT
        assert p.bilateral_history.level == AccessLevel.PERFECT

    def test_round_trip_with_noise(self):
        noise = EstimationNoise(Decimal("0.25"))
        p = InformationProfile.from_hierarchy(
            counterparty=CounterpartyAccess(cash=CategoryAccess(AccessLevel.NOISY, noise))
        )
        assert p.counterparty_cash.level == AccessLevel.NOISY
        assert p.counterparty_cash.noise == noise


# ═══════════════════════════════════════════════════════════════════════
# 4. Preset equivalence
# ═══════════════════════════════════════════════════════════════════════


class TestPresetEquivalence:
    def test_lender_realistic_v2_equals_v1(self):
        """LENDER_REALISTIC_V2 (from_hierarchy) must match LENDER_REALISTIC (flat)."""
        for f in dataclasses.fields(InformationProfile):
            v1 = getattr(LENDER_REALISTIC, f.name)
            v2 = getattr(LENDER_REALISTIC_V2, f.name)
            assert v1 == v2, f"Field {f.name}: {v1} != {v2}"

    def test_v2_is_informationprofile(self):
        assert isinstance(LENDER_REALISTIC_V2, InformationProfile)


# ═══════════════════════════════════════════════════════════════════════
# 5. Contextual view queries
# ═══════════════════════════════════════════════════════════════════════


class TestSystemView:
    def test_omniscient_returns_values(self):
        system = _build_system()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        view = info.system_view(day=0)
        assert isinstance(view, SystemView)
        assert view.day == 0
        assert view.aggregate_default_rate is not None
        assert view.aggregate_default_rate == Decimal("0")
        assert view.system_liquidity is not None
        assert view.system_liquidity == 10500

    def test_none_access_returns_none_fields(self):
        system = _build_system()
        profile = InformationProfile(
            aggregate_default_rate=CategoryAccess(AccessLevel.NONE),
            system_liquidity=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        view = info.system_view(day=0)
        assert view.aggregate_default_rate is None
        assert view.system_liquidity is None


class TestCounterpartyView:
    def test_omniscient_returns_values(self):
        system = _build_system()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        view = info.counterparty_view("F01", day=0, horizon=5)
        assert isinstance(view, CounterpartyView)
        assert view.agent_id == "F01"
        assert view.day == 0
        assert view.cash == 500
        assert view.obligations == 1000
        assert view.net_worth is not None
        assert view.default_probability is not None

    def test_none_counterparty_cash_returns_none(self):
        system = _build_system()
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        view = info.counterparty_view("F01", day=0)
        assert view.cash is None

    def test_self_access_always_perfect(self):
        system = _build_system()
        profile = InformationProfile(
            counterparty_cash=CategoryAccess(AccessLevel.NONE),
            counterparty_liabilities=CategoryAccess(AccessLevel.NONE),
            counterparty_net_worth=CategoryAccess(AccessLevel.NONE),
            counterparty_default_history=CategoryAccess(AccessLevel.NONE),
        )
        info = InformationService(system, profile, observer_id="NBL01")
        # Self-query bypasses all NONE restrictions
        view = info.counterparty_view("NBL01", day=0)
        assert view.cash == 10000
        assert view.net_worth is not None
        assert view.default_probability is not None

    def test_default_horizon_is_5(self):
        system = _build_system()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        # F01's payable is due day 3, horizon=5 should include it
        view = info.counterparty_view("F01", day=0)
        assert view.obligations == 1000


class TestInstrumentView:
    def test_returns_instrument_view(self):
        system = _build_system()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        view = info.instrument_view(day=0)
        assert isinstance(view, InstrumentView)
        assert view.day == 0


class TestTransactionView:
    def test_returns_transaction_view(self):
        system = _build_system()
        # Create a loan to have exposure
        system.nonbank_lend_cash("NBL01", "F01", 2000, Decimal("0.05"), 0)
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        view = info.transaction_view("F01", day=0)
        assert isinstance(view, TransactionView)
        assert view.agent_id == "F01"
        assert view.day == 0
        assert view.bilateral_exposure == 2000
        assert view.total_exposure is not None
        assert view.total_exposure >= 2000

    def test_no_exposure_returns_zero(self):
        system = _build_system()
        info = InformationService(system, OMNISCIENT, observer_id="NBL01")
        view = info.transaction_view("F01", day=0)
        assert view.bilateral_exposure == 0
        assert view.total_exposure == 0


# ═══════════════════════════════════════════════════════════════════════
# 6. View objects are frozen
# ═══════════════════════════════════════════════════════════════════════


class TestViewsFrozen:
    def test_system_view_frozen(self):
        v = SystemView(day=0, aggregate_default_rate=Decimal("0.1"))
        with pytest.raises(AttributeError):
            v.day = 1  # type: ignore[misc]

    def test_counterparty_view_frozen(self):
        v = CounterpartyView(agent_id="X", day=0, cash=100)
        with pytest.raises(AttributeError):
            v.cash = 200  # type: ignore[misc]

    def test_instrument_view_frozen(self):
        v = InstrumentView(day=0)
        with pytest.raises(AttributeError):
            v.day = 1  # type: ignore[misc]

    def test_transaction_view_frozen(self):
        v = TransactionView(agent_id="X", day=0)
        with pytest.raises(AttributeError):
            v.agent_id = "Y"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 7. Decision protocol defaults
# ═══════════════════════════════════════════════════════════════════════


class TestDecisionProtocols:
    def test_threshold_screener_eligible(self):
        s = ThresholdScreener(max_default_prob=Decimal("0.5"))
        assert s.is_eligible(Decimal("0.3")) is True

    def test_threshold_screener_ineligible(self):
        s = ThresholdScreener(max_default_prob=Decimal("0.2"))
        assert s.is_eligible(Decimal("0.3")) is False

    def test_threshold_screener_boundary(self):
        s = ThresholdScreener(max_default_prob=Decimal("0.3"))
        assert s.is_eligible(Decimal("0.3")) is True

    def test_linear_pricer_calculation(self):
        p = LinearPricer(risk_premium_scale=Decimal("0.5"))
        rate = p.price(Decimal("0.05"), Decimal("0.2"))
        assert rate == Decimal("0.15")

    def test_linear_pricer_zero_default_prob(self):
        p = LinearPricer(risk_premium_scale=Decimal("0.5"))
        rate = p.price(Decimal("0.05"), Decimal("0"))
        assert rate == Decimal("0.05")

    def test_fixed_portfolio_strategy(self):
        s = FixedPortfolioStrategy(
            max_exposure_fraction=Decimal("0.5"),
            base_return=Decimal("0.08"),
        )
        assert s.max_exposure(10000) == 5000
        assert s.target_return() == Decimal("0.08")

    def test_fixed_maturity_selector(self):
        s = FixedMaturitySelector(maturity_days=15)
        assert s.select_maturity() == 15

    def test_protocol_isinstance_checks(self):
        """Default implementations satisfy their protocol interfaces."""
        assert isinstance(FixedPortfolioStrategy(), PortfolioStrategy)
        assert isinstance(ThresholdScreener(), CounterpartyScreener)
        assert isinstance(FixedMaturitySelector(), InstrumentSelector)
        assert isinstance(LinearPricer(), TransactionPricer)
