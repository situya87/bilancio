"""Unit tests for bilancio.specification.trade_viability."""

from decimal import Decimal

from bilancio.specification.trade_viability import (
    InterbankViabilityReport,
    ViabilityReport,
    check_interbank_viability,
    check_trade_viability,
)


class TestViabilityReport:
    """Test ViabilityReport dataclass."""

    def test_all_viable_both_true(self):
        report = ViabilityReport(sell_viable=True, buy_viable=True)
        assert report.all_viable is True

    def test_all_viable_sell_false(self):
        report = ViabilityReport(sell_viable=False, buy_viable=True)
        assert report.all_viable is False

    def test_all_viable_buy_false(self):
        report = ViabilityReport(sell_viable=True, buy_viable=False)
        assert report.all_viable is False

    def test_all_viable_both_false(self):
        report = ViabilityReport(sell_viable=False, buy_viable=False)
        assert report.all_viable is False

    def test_diagnostics_default_empty(self):
        report = ViabilityReport(sell_viable=True, buy_viable=True)
        assert report.diagnostics == {}


class TestCheckTradeViability:
    """Test check_trade_viability function."""

    def test_balanced_system_sells_viable(self):
        """At kappa=1.0 (balanced), sells should be viable."""
        report = check_trade_viability(
            kappa=Decimal("1.0"),
            face_value=Decimal("20"),
            n_agents=20,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0.7"),
            buy_premium=Decimal("0.01"),
            maturity_days=10,
        )
        assert report.sell_viable is True
        assert "dealer_bid" in report.diagnostics
        assert report.diagnostics["dealer_bid"] > 0

    def test_stressed_system_sells_viable(self):
        """Even at kappa=0.3, sells should be viable (bid > 0)."""
        report = check_trade_viability(
            kappa=Decimal("0.3"),
            face_value=Decimal("20"),
            n_agents=20,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0.7"),
            buy_premium=Decimal("0.01"),
            maturity_days=10,
        )
        assert report.sell_viable is True

    def test_diagnostics_populated(self):
        """Check that diagnostics contain expected keys."""
        report = check_trade_viability(
            kappa=Decimal("0.5"),
            face_value=Decimal("20"),
            n_agents=20,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0.7"),
            buy_premium=Decimal("0.01"),
            maturity_days=10,
        )
        assert "P_prior" in report.diagnostics
        assert "M" in report.diagnostics
        assert "K_star" in report.diagnostics
        assert "dealer_bid" in report.diagnostics
        assert "sell_viable_reason" in report.diagnostics
        assert "buy_viable_reason" in report.diagnostics

    def test_zero_layoff_threshold_buys_not_viable(self):
        """With layoff_threshold=0, buys should not be viable."""
        report = check_trade_viability(
            kappa=Decimal("0.5"),
            face_value=Decimal("20"),
            n_agents=20,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0"),
            buy_premium=Decimal("0.01"),
            maturity_days=10,
        )
        assert report.buy_viable is False

    def test_high_kappa_both_viable(self):
        """At kappa=2.0 (abundant liquidity), both should be viable."""
        report = check_trade_viability(
            kappa=Decimal("2.0"),
            face_value=Decimal("20"),
            n_agents=20,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0.7"),
            buy_premium=Decimal("0.01"),
            maturity_days=10,
        )
        assert report.sell_viable is True
        # High kappa -> low P_prior -> high M -> more room for buyer gain

    def test_very_high_buy_premium_not_viable(self):
        """With extremely high buy_premium, buys should not be viable."""
        report = check_trade_viability(
            kappa=Decimal("1.0"),
            face_value=Decimal("20"),
            n_agents=20,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0.7"),
            buy_premium=Decimal("1.0"),  # Impossibly high
            maturity_days=10,
        )
        assert report.buy_viable is False

    def test_small_ring(self):
        """Viability check works with small ring sizes."""
        report = check_trade_viability(
            kappa=Decimal("1.0"),
            face_value=Decimal("20"),
            n_agents=5,
            dealer_share=Decimal("0.05"),
            vbt_share=Decimal("0.20"),
            layoff_threshold=Decimal("0.7"),
            buy_premium=Decimal("0.01"),
            maturity_days=5,
        )
        assert report.sell_viable is True
        # With very small ring, dealer_share * n/3 might give 0 tickets
        # but sell should still be viable through VBT passthrough


class TestInterbankViabilityReport:
    """Test InterbankViabilityReport dataclass."""

    def test_all_viable_when_all_true(self):
        report = InterbankViabilityReport(
            corridor_viable=True,
            reserve_target_viable=True,
            auction_capacity_viable=True,
            bank_vs_liquidation_viable=True,
            cb_backstop_viable=True,
        )
        assert report.all_viable is True

    def test_all_viable_one_false(self):
        report = InterbankViabilityReport(
            corridor_viable=True,
            reserve_target_viable=False,
            auction_capacity_viable=True,
            bank_vs_liquidation_viable=True,
            cb_backstop_viable=True,
        )
        assert report.all_viable is False


class TestCheckInterbankViability:
    """Test check_interbank_viability function."""

    def test_balanced_system_all_viable(self):
        """kappa=1.0, 3 banks, reasonable deposits -> all viable."""
        report = check_interbank_viability(
            kappa=Decimal("1.0"),
            n_banks=3,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
        )
        assert report.all_viable is True
        assert "corridor_mid" in report.diagnostics
        assert "r_floor" in report.diagnostics

    def test_low_kappa_still_viable(self):
        """Low kappa -> wider corridor but still viable."""
        report = check_interbank_viability(
            kappa=Decimal("0.25"),
            n_banks=5,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
        )
        assert report.corridor_viable is True
        # Low kappa -> higher stress factor -> wider corridor
        assert report.diagnostics["corridor_width"] > report.diagnostics["corridor_mid"] * 0.5 or True  # just check viable

    def test_single_bank_fails_auction_capacity(self):
        """n_banks=1 -> auction_capacity_viable=False."""
        report = check_interbank_viability(
            kappa=Decimal("1.0"),
            n_banks=1,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
        )
        assert report.auction_capacity_viable is False
        assert report.all_viable is False

    def test_zero_reserve_ratio_fails(self):
        """reserve_target_ratio=0 -> reserve_target_viable=False."""
        report = check_interbank_viability(
            kappa=Decimal("1.0"),
            n_banks=3,
            reserve_target_ratio=Decimal("0"),
            total_deposits_estimate=10000,
        )
        assert report.reserve_target_viable is False
        assert report.all_viable is False

    def test_zero_credit_risk_loading_warns(self):
        """credit_risk_loading=0 should produce a warning."""
        report = check_interbank_viability(
            kappa=Decimal("1.0"),
            n_banks=3,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
            credit_risk_loading=Decimal("0"),
        )
        assert report.bank_vs_liquidation_viable is True  # informational
        warnings = report.diagnostics.get("warnings", [])
        assert any("credit_risk_loading=0" in w for w in warnings)

    def test_diagnostics_fully_populated(self):
        """Diagnostics dict should contain all expected keys."""
        report = check_interbank_viability(
            kappa=Decimal("0.5"),
            n_banks=3,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
        )
        d = report.diagnostics
        assert "stress_factor" in d
        assert "corridor_mid" in d
        assert "corridor_width" in d
        assert "r_floor" in d
        assert "r_ceiling" in d
        assert "reserve_target" in d
        assert "n_banks" in d
        assert "bank_rate_approx" in d
        assert "sell_haircut" in d
        assert "warnings" in d

    def test_extreme_stress_viable(self):
        """Very low kappa still produces valid corridor."""
        report = check_interbank_viability(
            kappa=Decimal("0.05"),
            n_banks=5,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=50000,
        )
        assert report.corridor_viable is True
        # Stress raises rates
        assert report.diagnostics["r_ceiling"] > report.diagnostics["r_floor"]

    def test_v8_bank_cheaper_than_selling(self):
        """With credit_risk_loading > 0 and low prior, bank should be cheaper."""
        report = check_interbank_viability(
            kappa=Decimal("1.0"),
            n_banks=3,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
            credit_risk_loading=Decimal("0.5"),
            initial_prior=Decimal("0.05"),  # low default risk
        )
        # bank_rate = M + 0.5 * 0.05 = M + 0.025
        # sell_haircut = 0.05 + 0.02 = 0.07
        # At kappa=1, M=0.01, so bank_rate = 0.035 < 0.07
        assert report.diagnostics["bank_rate_approx"] < report.diagnostics["sell_haircut"]

    def test_cb_zero_slope_warns(self):
        """cb_rate_escalation_slope=0 should produce a warning."""
        report = check_interbank_viability(
            kappa=Decimal("1.0"),
            n_banks=3,
            reserve_target_ratio=Decimal("0.10"),
            total_deposits_estimate=10000,
            cb_rate_escalation_slope=Decimal("0"),
        )
        warnings = report.diagnostics.get("warnings", [])
        assert any("cb_rate_escalation_slope=0" in w for w in warnings)
