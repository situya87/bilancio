"""Tests for decision protocol integration with the lending engine.

Covers:
1. Default protocol implementations (scaffold tests)
2. Protocol resolution from LendingConfig
3. Custom protocol injection
4. Behavioral equivalence with pre-refactor lending
5. Protocol interface compliance
"""

from decimal import Decimal

from bilancio.decision.protocols import (
    CounterpartyScreener,
    FixedMaturitySelector,
    FixedPortfolioStrategy,
    InstrumentSelector,
    LinearPricer,
    PortfolioStrategy,
    ThresholdScreener,
    TransactionPricer,
)
from bilancio.engines.lending import LendingConfig, _resolve_protocols, run_lending_phase

# ═══════════════════════════════════════════════════════════════════════
# 1. Default Protocol Implementation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFixedPortfolioStrategy:
    """Tests for FixedPortfolioStrategy default implementation."""

    def test_max_exposure_default(self):
        """Default max_exposure_fraction=0.80 gives 80% of assets."""
        strategy = FixedPortfolioStrategy()
        assert strategy.max_exposure(10000) == 8000

    def test_max_exposure_custom_fraction(self):
        """Custom fraction is applied correctly."""
        strategy = FixedPortfolioStrategy(max_exposure_fraction=Decimal("0.80"))
        assert strategy.max_exposure(10000) == 8000

    def test_max_exposure_truncates(self):
        """max_exposure truncates to int (floor)."""
        strategy = FixedPortfolioStrategy(max_exposure_fraction=Decimal("0.33"))
        assert strategy.max_exposure(100) == 33

    def test_max_exposure_zero_assets(self):
        """Zero assets gives zero exposure."""
        strategy = FixedPortfolioStrategy()
        assert strategy.max_exposure(0) == 0

    def test_target_return_default(self):
        """Default base_return is 0.05."""
        strategy = FixedPortfolioStrategy()
        assert strategy.target_return() == Decimal("0.05")

    def test_target_return_custom(self):
        """Custom base_return is returned."""
        strategy = FixedPortfolioStrategy(base_return=Decimal("0.10"))
        assert strategy.target_return() == Decimal("0.10")


class TestThresholdScreener:
    """Tests for ThresholdScreener default implementation."""

    def test_eligible_below_threshold(self):
        """Default probability below threshold is eligible."""
        screener = ThresholdScreener()
        assert screener.is_eligible(Decimal("0.30")) is True

    def test_eligible_at_threshold(self):
        """Probability exactly at threshold is eligible (<=)."""
        screener = ThresholdScreener(max_default_prob=Decimal("0.50"))
        assert screener.is_eligible(Decimal("0.50")) is True

    def test_ineligible_above_threshold(self):
        """Probability above threshold is ineligible."""
        screener = ThresholdScreener(max_default_prob=Decimal("0.50"))
        assert screener.is_eligible(Decimal("0.51")) is False

    def test_custom_threshold(self):
        """Custom threshold changes eligibility boundary."""
        screener = ThresholdScreener(max_default_prob=Decimal("0.10"))
        assert screener.is_eligible(Decimal("0.09")) is True
        assert screener.is_eligible(Decimal("0.10")) is True
        assert screener.is_eligible(Decimal("0.11")) is False

    def test_zero_threshold_rejects_all_positive(self):
        """Zero threshold rejects any positive default probability."""
        screener = ThresholdScreener(max_default_prob=Decimal("0"))
        assert screener.is_eligible(Decimal("0")) is True
        assert screener.is_eligible(Decimal("0.01")) is False


class TestFixedMaturitySelector:
    """Tests for FixedMaturitySelector default implementation."""

    def test_default_maturity(self):
        """Default maturity is 2 days."""
        selector = FixedMaturitySelector()
        assert selector.select_maturity() == 2

    def test_custom_maturity(self):
        """Custom maturity is returned."""
        selector = FixedMaturitySelector(maturity_days=5)
        assert selector.select_maturity() == 5

    def test_consistent_returns(self):
        """Calling select_maturity multiple times returns same value."""
        selector = FixedMaturitySelector(maturity_days=7)
        assert selector.select_maturity() == selector.select_maturity()


class TestLinearPricer:
    """Tests for LinearPricer default implementation."""

    def test_price_formula(self):
        """Price = base_rate + risk_premium_scale * default_probability."""
        pricer = LinearPricer(risk_premium_scale=Decimal("0.20"))
        rate = pricer.price(Decimal("0.05"), Decimal("0.30"))
        assert rate == Decimal("0.05") + Decimal("0.20") * Decimal("0.30")
        assert rate == Decimal("0.11")

    def test_price_zero_default_prob(self):
        """Zero default probability means rate = base_rate."""
        pricer = LinearPricer(risk_premium_scale=Decimal("0.50"))
        rate = pricer.price(Decimal("0.05"), Decimal("0"))
        assert rate == Decimal("0.05")

    def test_price_high_default_prob(self):
        """High default probability increases rate."""
        pricer = LinearPricer(risk_premium_scale=Decimal("1.0"))
        rate = pricer.price(Decimal("0.05"), Decimal("0.50"))
        assert rate == Decimal("0.55")

    def test_custom_scale(self):
        """Custom risk_premium_scale changes pricing."""
        pricer = LinearPricer(risk_premium_scale=Decimal("0.10"))
        rate = pricer.price(Decimal("0.05"), Decimal("0.30"))
        assert rate == Decimal("0.08")


# ═══════════════════════════════════════════════════════════════════════
# 2. Protocol Resolution Tests
# ═══════════════════════════════════════════════════════════════════════


class TestResolveProtocols:
    """Tests for _resolve_protocols helper."""

    def test_defaults_from_scalar_params(self):
        """With no explicit protocols, defaults are constructed from LendingConfig scalars."""
        cfg = LendingConfig()
        portfolio, screener, selector, pricer = _resolve_protocols(cfg)

        assert isinstance(portfolio, FixedPortfolioStrategy)
        assert isinstance(screener, ThresholdScreener)
        assert isinstance(selector, FixedMaturitySelector)
        assert isinstance(pricer, LinearPricer)

    def test_default_portfolio_uses_config_values(self):
        """Default portfolio uses max_total_exposure and base_rate from config."""
        cfg = LendingConfig(
            max_total_exposure=Decimal("0.80"),
            base_rate=Decimal("0.05"),
        )
        portfolio, _, _, _ = _resolve_protocols(cfg)

        assert portfolio.max_exposure(10000) == 8000
        assert portfolio.target_return() == Decimal("0.05")

    def test_default_screener_uses_config_values(self):
        """Default screener uses max_default_prob from config."""
        cfg = LendingConfig(max_default_prob=Decimal("0.50"))
        _, screener, _, _ = _resolve_protocols(cfg)

        assert screener.is_eligible(Decimal("0.50")) is True
        assert screener.is_eligible(Decimal("0.51")) is False

    def test_default_selector_uses_config_values(self):
        """Default selector uses maturity_days from config."""
        cfg = LendingConfig(maturity_days=2)
        _, _, selector, _ = _resolve_protocols(cfg)

        assert selector.select_maturity() == 2

    def test_default_pricer_uses_config_values(self):
        """Default pricer uses risk_premium_scale from config."""
        cfg = LendingConfig(risk_premium_scale=Decimal("0.20"))
        _, _, _, pricer = _resolve_protocols(cfg)

        rate = pricer.price(Decimal("0.05"), Decimal("0.30"))
        expected = Decimal("0.05") + Decimal("0.20") * Decimal("0.30")
        assert rate == expected

    def test_explicit_portfolio_override(self):
        """Explicit portfolio_strategy takes precedence over scalar params."""
        custom = FixedPortfolioStrategy(
            max_exposure_fraction=Decimal("0.99"),
            base_return=Decimal("0.20"),
        )
        cfg = LendingConfig(
            max_total_exposure=Decimal("0.50"),  # should be ignored
            portfolio_strategy=custom,
        )
        portfolio, _, _, _ = _resolve_protocols(cfg)

        assert portfolio is custom
        assert portfolio.max_exposure(10000) == 9900

    def test_explicit_screener_override(self):
        """Explicit counterparty_screener takes precedence."""
        custom = ThresholdScreener(max_default_prob=Decimal("0.01"))
        cfg = LendingConfig(
            max_default_prob=Decimal("0.50"),  # should be ignored
            counterparty_screener=custom,
        )
        _, screener, _, _ = _resolve_protocols(cfg)

        assert screener is custom
        assert screener.is_eligible(Decimal("0.02")) is False

    def test_explicit_selector_override(self):
        """Explicit instrument_selector takes precedence."""
        custom = FixedMaturitySelector(maturity_days=30)
        cfg = LendingConfig(
            maturity_days=2,  # should be ignored
            instrument_selector=custom,
        )
        _, _, selector, _ = _resolve_protocols(cfg)

        assert selector is custom
        assert selector.select_maturity() == 30

    def test_explicit_pricer_override(self):
        """Explicit transaction_pricer takes precedence."""
        custom = LinearPricer(risk_premium_scale=Decimal("5.0"))
        cfg = LendingConfig(
            risk_premium_scale=Decimal("0.20"),  # should be ignored
            transaction_pricer=custom,
        )
        _, _, _, pricer = _resolve_protocols(cfg)

        assert pricer is custom

    def test_mixed_overrides(self):
        """Some protocols explicit, others defaulted from scalars."""
        custom_pricer = LinearPricer(risk_premium_scale=Decimal("1.0"))
        cfg = LendingConfig(
            max_total_exposure=Decimal("0.60"),
            transaction_pricer=custom_pricer,
        )
        portfolio, screener, selector, pricer = _resolve_protocols(cfg)

        # Portfolio from scalars
        assert isinstance(portfolio, FixedPortfolioStrategy)
        assert portfolio.max_exposure(10000) == 6000
        # Pricer is custom
        assert pricer is custom_pricer


# ═══════════════════════════════════════════════════════════════════════
# 3. Protocol Interface Compliance Tests
# ═══════════════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """Tests that default implementations satisfy Protocol interfaces."""

    def test_fixed_portfolio_is_portfolio_strategy(self):
        assert isinstance(FixedPortfolioStrategy(), PortfolioStrategy)

    def test_threshold_screener_is_counterparty_screener(self):
        assert isinstance(ThresholdScreener(), CounterpartyScreener)

    def test_fixed_maturity_is_instrument_selector(self):
        assert isinstance(FixedMaturitySelector(), InstrumentSelector)

    def test_linear_pricer_is_transaction_pricer(self):
        assert isinstance(LinearPricer(), TransactionPricer)

    def test_custom_portfolio_satisfies_protocol(self):
        """A custom class with the right methods satisfies PortfolioStrategy."""

        class MyPortfolio:
            def max_exposure(self, total_assets: int) -> int:
                return total_assets  # 100% exposure

            def target_return(self) -> Decimal:
                return Decimal("0.15")

        assert isinstance(MyPortfolio(), PortfolioStrategy)

    def test_custom_screener_satisfies_protocol(self):
        """A custom class with is_eligible satisfies CounterpartyScreener."""

        class MyScreener:
            def is_eligible(self, default_probability: Decimal) -> bool:
                return True  # Accept everyone

        assert isinstance(MyScreener(), CounterpartyScreener)

    def test_custom_pricer_satisfies_protocol(self):
        """A custom class with price satisfies TransactionPricer."""

        class MyPricer:
            def price(self, base_rate: Decimal, default_probability: Decimal) -> Decimal:
                return base_rate * 2  # Double the base rate

        assert isinstance(MyPricer(), TransactionPricer)


# ═══════════════════════════════════════════════════════════════════════
# 4. Behavioral Equivalence Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBehavioralEquivalence:
    """Verify protocol-based lending produces identical results to scalar-based."""

    def test_max_exposure_equivalence(self):
        """portfolio.max_exposure matches int(config.max_total_exposure * capital)."""
        cfg = LendingConfig()
        portfolio, _, _, _ = _resolve_protocols(cfg)

        initial_capital = 10000
        old_way = int(cfg.max_total_exposure * initial_capital)
        new_way = portfolio.max_exposure(initial_capital)
        assert old_way == new_way

    def test_screening_equivalence(self):
        """screener.is_eligible matches p_default <= config.max_default_prob."""
        cfg = LendingConfig()
        _, screener, _, _ = _resolve_protocols(cfg)

        test_probs = [
            Decimal("0"),
            Decimal("0.25"),
            Decimal("0.50"),
            Decimal("0.51"),
            Decimal("0.99"),
        ]
        for p in test_probs:
            old_passes = p <= cfg.max_default_prob
            new_passes = screener.is_eligible(p)
            assert old_passes == new_passes, f"Mismatch at p={p}"

    def test_pricing_equivalence(self):
        """pricer.price matches config.base_rate + config.risk_premium_scale * p."""
        cfg = LendingConfig()
        portfolio, _, _, pricer = _resolve_protocols(cfg)
        base_rate = portfolio.target_return()

        test_probs = [Decimal("0"), Decimal("0.10"), Decimal("0.25"), Decimal("0.50")]
        for p in test_probs:
            old_rate = cfg.base_rate + cfg.risk_premium_scale * p
            new_rate = pricer.price(base_rate, p)
            assert old_rate == new_rate, f"Mismatch at p={p}: {old_rate} != {new_rate}"

    def test_maturity_equivalence(self):
        """selector.select_maturity matches config.maturity_days."""
        cfg = LendingConfig()
        _, _, selector, _ = _resolve_protocols(cfg)

        assert selector.select_maturity() == cfg.maturity_days


# ═══════════════════════════════════════════════════════════════════════
# 5. Custom Protocol Injection Tests (via run_lending_phase)
# ═══════════════════════════════════════════════════════════════════════


def _build_lending_system(
    lender_cash: int = 10000,
    firm_cash: int = 500,
    firm_payable_amount: int = 1000,
    payable_due_day: int = 2,
):
    """Build a minimal system for lending tests."""
    from bilancio.domain.agents.bank import Bank
    from bilancio.domain.agents.central_bank import CentralBank
    from bilancio.domain.agents.firm import Firm
    from bilancio.domain.agents.non_bank_lender import NonBankLender
    from bilancio.domain.instruments.base import InstrumentKind
    from bilancio.domain.instruments.credit import Payable
    from bilancio.engines.system import System

    system = System()
    cb = CentralBank(id="CB01", name="Central Bank", kind="central_bank")
    bank = Bank(id="B01", name="Bank 1", kind="bank")
    lender = NonBankLender(id="NBL01", name="Non-Bank Lender")
    firm1 = Firm(id="F01", name="Firm 1", kind="firm")
    firm2 = Firm(id="F02", name="Firm 2", kind="firm")

    system.bootstrap_cb(cb)
    system.add_agent(bank)
    system.add_agent(lender)
    system.add_agent(firm1)
    system.add_agent(firm2)

    if lender_cash > 0:
        system.mint_cash("NBL01", lender_cash)
    if firm_cash > 0:
        system.mint_cash("F01", firm_cash)

    if firm_payable_amount > 0:
        payable = Payable(
            id=system.new_contract_id("PAY"),
            kind=InstrumentKind.PAYABLE,
            amount=firm_payable_amount,
            denom="X",
            asset_holder_id="F02",
            liability_issuer_id="F01",
            due_day=payable_due_day,
        )
        system.add_contract(payable)

    return system


class TestCustomProtocolInjection:
    """Tests that custom protocols change run_lending_phase behavior."""

    def test_custom_pricer_changes_rates(self):
        """Injecting a custom pricer changes loan rates."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )

        # Default config
        default_cfg = LendingConfig(horizon=3, min_shortfall=1)
        default_events = run_lending_phase(system, current_day=0, lending_config=default_cfg)

        # Rebuild same system
        system2 = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )

        # Custom pricer with much higher scale
        custom_pricer = LinearPricer(risk_premium_scale=Decimal("2.0"))
        custom_cfg = LendingConfig(
            horizon=3,
            min_shortfall=1,
            transaction_pricer=custom_pricer,
        )
        custom_events = run_lending_phase(system2, current_day=0, lending_config=custom_cfg)

        # Both should create loans (if borrower eligible)
        if default_events and custom_events:
            default_rate = Decimal(default_events[0]["rate"])
            custom_rate = Decimal(custom_events[0]["rate"])
            assert custom_rate > default_rate

    def test_strict_screener_rejects_all(self):
        """A very strict screener rejects all borrowers."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )

        strict_screener = ThresholdScreener(max_default_prob=Decimal("0.001"))
        cfg = LendingConfig(
            horizon=3,
            min_shortfall=1,
            counterparty_screener=strict_screener,
        )
        events = run_lending_phase(system, current_day=0, lending_config=cfg)
        assert events == []

    def test_custom_maturity_used_in_loan(self):
        """Custom maturity selector changes loan maturity."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=200,
            firm_payable_amount=1000,
            payable_due_day=2,
        )

        custom_selector = FixedMaturitySelector(maturity_days=15)
        cfg = LendingConfig(
            horizon=3,
            min_shortfall=1,
            instrument_selector=custom_selector,
        )
        events = run_lending_phase(system, current_day=0, lending_config=cfg)

        if events:
            loan_id = events[0]["loan_id"]
            loan = system.state.contracts[loan_id]
            assert loan.maturity_days == 15

    def test_custom_portfolio_limits_exposure(self):
        """Custom portfolio with low exposure fraction limits lending."""
        system = _build_lending_system(
            lender_cash=10000,
            firm_cash=0,
            firm_payable_amount=5000,
            payable_due_day=2,
        )

        # Very restrictive: only 10% exposure
        custom_portfolio = FixedPortfolioStrategy(
            max_exposure_fraction=Decimal("0.10"),
            base_return=Decimal("0.05"),
        )
        cfg = LendingConfig(
            horizon=3,
            min_shortfall=1,
            portfolio_strategy=custom_portfolio,
        )
        events = run_lending_phase(system, current_day=0, lending_config=cfg)

        if events:
            # 10% of 10000 = 1000, so loan should be at most 1000
            assert events[0]["amount"] <= 1000
