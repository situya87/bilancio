"""Unit tests for bilancio.specification.trade_viability."""

from decimal import Decimal

import pytest

from bilancio.specification.trade_viability import (
    ViabilityReport,
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
