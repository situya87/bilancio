"""Tests for jurisdiction domain models, config models, and ops functions."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from bilancio.domain.jurisdiction import (
    BankingRules,
    CapitalControlAction,
    CapitalControlRule,
    CapitalControls,
    CapitalFlowPurpose,
    ExchangeRatePair,
    FXMarket,
    InterbankSettlementMode,
    Jurisdiction,
)

# ── Domain model tests ───────────────────────────────────────────────


class TestJurisdictionInstantiation:
    def test_defaults(self):
        j = Jurisdiction(id="US", name="United States", domestic_currency="USD")
        assert j.id == "US"
        assert j.name == "United States"
        assert j.domestic_currency == "USD"
        assert j.institutional_agent_ids == []
        assert j.banking_rules.reserve_requirement_ratio == Decimal("0")
        assert j.capital_controls.default_action == CapitalControlAction.ALLOW

    def test_with_institutional_agents(self):
        j = Jurisdiction(
            id="EU",
            name="EU",
            domestic_currency="EUR",
            institutional_agent_ids=["CB_EU", "T_EU"],
        )
        assert j.institutional_agent_ids == ["CB_EU", "T_EU"]


class TestBankingRules:
    def test_defaults(self):
        rules = BankingRules()
        assert rules.reserve_requirement_ratio == Decimal("0")
        assert rules.interbank_settlement_mode == InterbankSettlementMode.RTGS
        assert rules.deposit_convertibility is True
        assert rules.cb_lending_enabled is True

    def test_custom(self):
        rules = BankingRules(
            reserve_requirement_ratio=Decimal("0.10"),
            interbank_settlement_mode=InterbankSettlementMode.DNS,
            deposit_convertibility=False,
            cb_lending_enabled=False,
        )
        assert rules.reserve_requirement_ratio == Decimal("0.10")
        assert rules.interbank_settlement_mode == InterbankSettlementMode.DNS
        assert rules.deposit_convertibility is False
        assert rules.cb_lending_enabled is False


class TestCapitalControls:
    def test_default_action(self):
        cc = CapitalControls()
        action, rate = cc.evaluate(CapitalFlowPurpose.TRADE, "inflow")
        assert action == CapitalControlAction.ALLOW
        assert rate == Decimal("0")

    def test_first_match_wins(self):
        rule1 = CapitalControlRule(
            purpose=CapitalFlowPurpose.PORTFOLIO,
            direction="inflow",
            action=CapitalControlAction.TAX,
            tax_rate=Decimal("0.01"),
        )
        rule2 = CapitalControlRule(
            purpose=CapitalFlowPurpose.PORTFOLIO,
            direction="inflow",
            action=CapitalControlAction.BLOCK,
        )
        cc = CapitalControls(rules=[rule1, rule2])
        action, rate = cc.evaluate(CapitalFlowPurpose.PORTFOLIO, "inflow")
        assert action == CapitalControlAction.TAX
        assert rate == Decimal("0.01")

    def test_both_direction(self):
        rule = CapitalControlRule(
            purpose=CapitalFlowPurpose.FDI,
            direction="both",
            action=CapitalControlAction.BLOCK,
        )
        cc = CapitalControls(rules=[rule])
        action_in, _ = cc.evaluate(CapitalFlowPurpose.FDI, "inflow")
        action_out, _ = cc.evaluate(CapitalFlowPurpose.FDI, "outflow")
        assert action_in == CapitalControlAction.BLOCK
        assert action_out == CapitalControlAction.BLOCK

    def test_default_fallback(self):
        cc = CapitalControls(default_action=CapitalControlAction.BLOCK)
        action, rate = cc.evaluate(CapitalFlowPurpose.REMITTANCE, "outflow")
        assert action == CapitalControlAction.BLOCK
        assert rate == Decimal("0")

    def test_no_match_returns_default(self):
        rule = CapitalControlRule(
            purpose=CapitalFlowPurpose.TRADE,
            direction="inflow",
            action=CapitalControlAction.BLOCK,
        )
        cc = CapitalControls(rules=[rule], default_action=CapitalControlAction.ALLOW)
        # Different purpose → no match → default
        action, rate = cc.evaluate(CapitalFlowPurpose.OTHER, "inflow")
        assert action == CapitalControlAction.ALLOW


class TestExchangeRatePair:
    def test_bid_ask(self):
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
            spread=Decimal("0.02"),
        )
        assert pair.bid == Decimal("1.09")
        assert pair.ask == Decimal("1.11")

    def test_convert_base_to_quote(self):
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        # 100 EUR -> 110 USD
        result = pair.convert(100, "EUR")
        assert result == 110

    def test_convert_quote_to_base(self):
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        # 110 USD -> 100 EUR
        result = pair.convert(110, "USD")
        assert result == 100

    def test_convert_rounding(self):
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        # 1 USD -> 0.909... EUR -> rounds to 1
        result = pair.convert(1, "USD")
        assert result == 1

    def test_convert_invalid_currency(self):
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        with pytest.raises(ValueError, match="neither base"):
            pair.convert(100, "GBP")

    def test_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            ExchangeRatePair(
                base_currency="EUR",
                quote_currency="USD",
                rate=Decimal("0"),
            )
        with pytest.raises(ValueError, match="positive"):
            ExchangeRatePair(
                base_currency="EUR",
                quote_currency="USD",
                rate=Decimal("-1"),
            )


class TestFXMarket:
    def test_add_and_get_rate(self):
        market = FXMarket()
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        market.add_rate(pair)
        result = market.get_rate("EUR", "USD")
        assert result.rate == Decimal("1.10")

    def test_auto_inversion(self):
        market = FXMarket()
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        market.add_rate(pair)
        inverse = market.get_rate("USD", "EUR")
        assert inverse.base_currency == "USD"
        assert inverse.quote_currency == "EUR"
        # 1/1.10 ≈ 0.909...
        expected = Decimal("1") / Decimal("1.10")
        assert inverse.rate == expected

    def test_auto_inversion_spread(self):
        """Test that spread inversion formula produces accurate results."""
        market = FXMarket()
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
            spread=Decimal("0.02"),
        )
        market.add_rate(pair)
        inverse = market.get_rate("USD", "EUR")
        # Inverse spread formula: spread / rate^2
        expected_spread = Decimal("0.02") / (Decimal("1.10") ** 2)
        assert inverse.spread == expected_spread
        # Verify bid < rate < ask
        assert inverse.bid < inverse.rate < inverse.ask

    def test_same_currency_identity(self):
        market = FXMarket()
        pair = market.get_rate("USD", "USD")
        assert pair.rate == Decimal("1")
        assert pair.spread == Decimal("0")

    def test_missing_rate_raises(self):
        market = FXMarket()
        with pytest.raises(KeyError, match="No exchange rate"):
            market.get_rate("GBP", "JPY")

    def test_convert(self):
        market = FXMarket()
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        market.add_rate(pair)
        assert market.convert(100, "EUR", "USD") == 110
        assert market.convert(100, "USD", "USD") == 100


# ── Config model tests ───────────────────────────────────────────────


class TestConfigModels:
    def test_jurisdiction_config_basic(self):
        from bilancio.config.models import JurisdictionConfig

        jc = JurisdictionConfig(
            id="US",
            name="United States",
            domestic_currency="USD",
        )
        assert jc.id == "US"
        assert jc.institutional_agents == []
        assert jc.banking_rules.reserve_requirement_ratio == Decimal("0")

    def test_exchange_rate_pair_config_validation(self):
        from bilancio.config.models import ExchangeRatePairConfig

        with pytest.raises(ValueError):
            ExchangeRatePairConfig(
                base_currency="USD",
                quote_currency="USD",
                rate=Decimal("1.0"),
            )

    def test_exchange_rate_pair_config_positive_rate(self):
        from bilancio.config.models import ExchangeRatePairConfig

        with pytest.raises(ValueError):
            ExchangeRatePairConfig(
                base_currency="EUR",
                quote_currency="USD",
                rate=Decimal("-1.0"),
            )

    def test_scenario_config_with_jurisdictions(self):
        from bilancio.config.models import (
            AgentSpec,
            JurisdictionConfig,
            ScenarioConfig,
        )

        config = ScenarioConfig(
            name="test",
            agents=[
                AgentSpec(id="CB", kind="central_bank", name="CB", jurisdiction="US"),
                AgentSpec(id="B1", kind="bank", name="B1"),
            ],
            jurisdictions=[
                JurisdictionConfig(
                    id="US",
                    name="US",
                    domestic_currency="USD",
                    institutional_agents=["CB"],
                ),
            ],
        )
        assert len(config.jurisdictions) == 1
        assert config.agents[0].jurisdiction == "US"
        assert config.agents[1].jurisdiction is None

    def test_scenario_config_rejects_bad_jurisdiction_ref(self):
        from bilancio.config.models import (
            AgentSpec,
            JurisdictionConfig,
            ScenarioConfig,
        )

        with pytest.raises(ValidationError, match="unknown jurisdiction"):
            ScenarioConfig(
                name="test",
                agents=[
                    AgentSpec(id="CB", kind="central_bank", name="CB", jurisdiction="INVALID"),
                ],
                jurisdictions=[
                    JurisdictionConfig(id="US", name="US", domestic_currency="USD"),
                ],
            )

    def test_scenario_config_rejects_bad_institutional_agent_ref(self):
        from bilancio.config.models import (
            AgentSpec,
            JurisdictionConfig,
            ScenarioConfig,
        )

        with pytest.raises(ValidationError, match="unknown institutional agent"):
            ScenarioConfig(
                name="test",
                agents=[
                    AgentSpec(id="CB", kind="central_bank", name="CB"),
                ],
                jurisdictions=[
                    JurisdictionConfig(
                        id="US",
                        name="US",
                        domestic_currency="USD",
                        institutional_agents=["NONEXISTENT"],
                    ),
                ],
            )


# ── Agent field tests ─────────────────────────────────────────────────


class TestAgentJurisdiction:
    def test_default_none(self):
        from bilancio.domain.agent import Agent

        a = Agent(id="A1", name="Test", kind="firm")
        assert a.jurisdiction_id is None

    def test_can_set(self):
        from bilancio.domain.agents import Bank

        b = Bank(id="B1", name="Bank", kind="bank")
        b.jurisdiction_id = "US"
        assert b.jurisdiction_id == "US"


# ── Ops function tests ───────────────────────────────────────────────


class TestOps:
    def _build_system(self):
        """Build a simple system with two jurisdictions."""
        from bilancio.domain.agents import Bank, CentralBank, Firm
        from bilancio.engines.system import System

        system = System()
        cb = CentralBank(id="CB", name="CB", kind="central_bank")
        b1 = Bank(id="B1", name="B1", kind="bank")
        b1.jurisdiction_id = "US"
        f1 = Firm(id="F1", name="F1", kind="firm")
        f1.jurisdiction_id = "US"
        f2 = Firm(id="F2", name="F2", kind="firm")
        f2.jurisdiction_id = "EU"
        f3 = Firm(id="F3", name="F3", kind="firm")
        # f3 has no jurisdiction

        system.add_agent(cb)
        system.add_agent(b1)
        system.add_agent(f1)
        system.add_agent(f2)
        system.add_agent(f3)

        # Add jurisdictions to state
        us = Jurisdiction(
            id="US",
            name="US",
            domestic_currency="USD",
            banking_rules=BankingRules(
                reserve_requirement_ratio=Decimal("0.10"),
            ),
        )
        eu = Jurisdiction(
            id="EU",
            name="EU",
            domestic_currency="EUR",
        )
        system.state.jurisdictions["US"] = us
        system.state.jurisdictions["EU"] = eu

        return system

    def test_are_same_jurisdiction(self):
        from bilancio.ops.jurisdiction import are_same_jurisdiction

        system = self._build_system()
        assert are_same_jurisdiction(system, "B1", "F1") is True
        assert are_same_jurisdiction(system, "B1", "F2") is False
        # Both None → True
        assert are_same_jurisdiction(system, "CB", "F3") is True

    def test_get_agent_domestic_currency(self):
        from bilancio.ops.jurisdiction import get_agent_domestic_currency

        system = self._build_system()
        assert get_agent_domestic_currency(system, "F1") == "USD"
        assert get_agent_domestic_currency(system, "F2") == "EUR"
        assert get_agent_domestic_currency(system, "F3") == "X"

    def test_validate_same_denomination(self):
        from bilancio.domain.instruments.base import InstrumentKind
        from bilancio.domain.instruments.means_of_payment import Cash
        from bilancio.ops.jurisdiction import validate_same_denomination

        system = self._build_system()
        c1 = Cash(
            id="C1",
            kind=InstrumentKind.CASH,
            amount=100,
            denom="X",
            asset_holder_id="F1",
            liability_issuer_id="CB",
        )
        c2 = Cash(
            id="C2",
            kind=InstrumentKind.CASH,
            amount=200,
            denom="X",
            asset_holder_id="F2",
            liability_issuer_id="CB",
        )
        system.add_contract(c1)
        system.add_contract(c2)
        assert validate_same_denomination(system, "C1", "C2") is True
        assert validate_same_denomination(system, "C1", "NOPE") is False

    def test_check_reserve_requirement(self):
        from bilancio.domain.instruments.base import InstrumentKind
        from bilancio.domain.instruments.means_of_payment import BankDeposit
        from bilancio.ops.jurisdiction import check_reserve_requirement

        system = self._build_system()

        # Mint reserves to bank
        system.mint_reserves("B1", 1000)

        # Add a bank deposit liability
        deposit = BankDeposit(
            id="BD1",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=5000,
            denom="X",
            asset_holder_id="F1",
            liability_issuer_id="B1",
        )
        system.add_contract(deposit)

        compliant, actual, required = check_reserve_requirement(system, "B1")
        assert actual == 1000
        assert required == 500  # 10% of 5000
        assert compliant is True

    def test_check_reserve_requirement_no_jurisdiction(self):
        from bilancio.ops.jurisdiction import check_reserve_requirement

        system = self._build_system()
        compliant, actual, required = check_reserve_requirement(system, "F3")
        assert compliant is True
        assert actual == 0
        assert required == 0

    def test_fx_convert(self):
        from bilancio.ops.jurisdiction import fx_convert

        market = FXMarket()
        pair = ExchangeRatePair(
            base_currency="EUR",
            quote_currency="USD",
            rate=Decimal("1.10"),
        )
        market.add_rate(pair)
        assert fx_convert(market, 100, "EUR", "USD") == 110
        assert fx_convert(market, 100, "USD", "USD") == 100

    def test_check_capital_controls(self):
        from bilancio.ops.jurisdiction import check_capital_controls

        j = Jurisdiction(
            id="EU",
            name="EU",
            domestic_currency="EUR",
            capital_controls=CapitalControls(
                rules=[
                    CapitalControlRule(
                        purpose=CapitalFlowPurpose.PORTFOLIO,
                        direction="inflow",
                        action=CapitalControlAction.TAX,
                        tax_rate=Decimal("0.001"),
                    ),
                ],
            ),
        )
        action, rate = check_capital_controls(j, "PORTFOLIO", "inflow")
        assert action == "TAX"
        assert rate == Decimal("0.001")

        action2, rate2 = check_capital_controls(j, "TRADE", "inflow")
        assert action2 == "ALLOW"
        assert rate2 == Decimal("0")


# ── Integration test: load example YAML ──────────────────────────────


class TestIntegration:
    def test_load_two_jurisdictions_yaml(self):
        from bilancio.config.loaders import load_yaml

        config = load_yaml("examples/scenarios/two_jurisdictions.yaml")
        assert config.name == "Two-Jurisdiction Banking System (US + EU)"
        assert len(config.jurisdictions) == 2
        assert len(config.fx_rates) == 1
        assert config.jurisdictions[0].id == "US"
        assert config.jurisdictions[1].id == "EU"
        assert config.fx_rates[0].base_currency == "EUR"
        assert config.fx_rates[0].rate == Decimal("1.10")

    def test_apply_two_jurisdictions_to_system(self):
        from bilancio.config.apply import apply_to_system
        from bilancio.config.loaders import load_yaml
        from bilancio.engines.system import System

        config = load_yaml("examples/scenarios/two_jurisdictions.yaml")
        system = System()
        apply_to_system(config, system)

        # Verify jurisdictions hydrated
        assert "US" in system.state.jurisdictions
        assert "EU" in system.state.jurisdictions
        us = system.state.jurisdictions["US"]
        eu = system.state.jurisdictions["EU"]
        assert us.domestic_currency == "USD"
        assert eu.domestic_currency == "EUR"
        assert us.banking_rules.reserve_requirement_ratio == Decimal("0.10")
        assert eu.banking_rules.reserve_requirement_ratio == Decimal("0.01")
        assert len(eu.capital_controls.rules) == 1

        # Verify FX market
        assert system.state.fx_market is not None
        pair = system.state.fx_market.get_rate("EUR", "USD")
        assert pair.rate == Decimal("1.10")

        # Verify agent jurisdiction_id was set
        assert system.state.agents["CB_US"].jurisdiction_id == "US"
        assert system.state.agents["F_EU"].jurisdiction_id == "EU"
        assert system.state.agents["B_US"].jurisdiction_id == "US"


# ── Event kind tests ─────────────────────────────────────────────────


class TestEventKinds:
    def test_jurisdiction_event_kinds_exist(self):
        from bilancio.core.events import EventKind

        assert EventKind.FX_CONVERSION == "FXConversion"
        assert EventKind.CAPITAL_CONTROL_BLOCKED == "CapitalControlBlocked"
        assert EventKind.CAPITAL_CONTROL_TAXED == "CapitalControlTaxed"
        assert EventKind.RESERVE_REQUIREMENT_BREACH == "ReserveRequirementBreach"
