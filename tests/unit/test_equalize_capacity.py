"""Tests for bank capacity equalization (equalize_capacity flag)."""

from decimal import Decimal

import pytest

from bilancio.config.models import RingExplorerGeneratorConfig
from bilancio.scenarios.ring.compiler import compile_ring_explorer_balanced


def _make_config() -> RingExplorerGeneratorConfig:
    """Create a standard test config."""
    return RingExplorerGeneratorConfig.model_validate({
        "name_prefix": "test",
        "params": {
            "n_agents": 20,
            "maturity": {"days": 10},
            "inequality": {"concentration": "1", "monotonicity": "0"},
            "Q_total": "10000",
            "seed": 42,
            "kappa": "1.0",
        },
        "compile": {"emit_yaml": False},
    })


def _sum_intermediary_capitals(scenario: dict) -> dict:
    """Extract intermediary capital totals from a compiled scenario."""
    dealer_cash = vbt_cash = lender_cash = bank_reserves = 0.0
    for action in scenario["initial_actions"]:
        if "mint_cash" in action:
            to = action["mint_cash"]["to"]
            amt = float(action["mint_cash"]["amount"])
            if to.startswith("dealer_"):
                dealer_cash += amt
            elif to.startswith("vbt_"):
                vbt_cash += amt
            elif to == "lender":
                lender_cash += amt
        elif "mint_reserves" in action:
            bank_reserves += float(action["mint_reserves"]["amount"])
    return {
        "dealer_cash": dealer_cash,
        "vbt_cash": vbt_cash,
        "lender_cash": lender_cash,
        "bank_reserves": bank_reserves,
        "total": dealer_cash + vbt_cash + lender_cash + bank_reserves,
    }


class TestEqualizeCapacity:
    """Tests for equalize_capacity=True in compile_ring_explorer_balanced."""

    def test_nonbank_arms_unaffected(self):
        """equalize_capacity should not change non-banking arms."""
        config = _make_config()
        for mode in ("passive", "active", "nbfi", "nbfi_dealer"):
            without = compile_ring_explorer_balanced(
                config, mode=mode, kappa=Decimal("1"), equalize_capacity=False,
            )
            with_eq = compile_ring_explorer_balanced(
                config, mode=mode, kappa=Decimal("1"), equalize_capacity=True,
            )
            caps_without = _sum_intermediary_capitals(without)
            caps_with = _sum_intermediary_capitals(with_eq)
            assert caps_without["total"] == pytest.approx(caps_with["total"], rel=0.01), (
                f"Mode {mode}: equalize_capacity changed non-bank arm total"
            )

    def test_banking_arms_equalized_to_nonbank(self):
        """Banking arms should have same total intermediary capital as non-bank arms."""
        config = _make_config()
        kappa = Decimal("1")
        # Reference: passive arm total
        passive = compile_ring_explorer_balanced(config, mode="passive", kappa=kappa)
        reference_total = _sum_intermediary_capitals(passive)["total"]

        for mode in ("banking", "bank_dealer", "bank_dealer_nbfi"):
            scenario = compile_ring_explorer_balanced(
                config, mode=mode, kappa=kappa, n_banks=3,
                equalize_capacity=True,
            )
            caps = _sum_intermediary_capitals(scenario)
            # Allow small int truncation error (up to n_banks)
            assert caps["total"] == pytest.approx(reference_total, abs=5), (
                f"Mode {mode}: total {caps['total']:.1f} != reference {reference_total:.1f}"
            )

    def test_banking_arms_not_equalized_by_default(self):
        """Without equalize_capacity, banking arms have different totals."""
        config = _make_config()
        kappa = Decimal("1")
        passive = compile_ring_explorer_balanced(config, mode="passive", kappa=kappa)
        reference_total = _sum_intermediary_capitals(passive)["total"]

        scenario = compile_ring_explorer_balanced(
            config, mode="banking", kappa=kappa, n_banks=3,
            reserve_multiplier=10.0, equalize_capacity=False,
        )
        caps = _sum_intermediary_capitals(scenario)
        # With reserve_multiplier=10, bank reserves should be much larger
        assert caps["total"] > reference_total * 1.5, (
            f"Expected banking arm total ({caps['total']:.1f}) to be much larger "
            f"than reference ({reference_total:.1f})"
        )

    def test_bank_share_proportional_to_mode(self):
        """Bank reserves should be proportional: banking > bank_dealer > bank_dealer_nbfi."""
        config = _make_config()
        kappa = Decimal("1")
        banking = compile_ring_explorer_balanced(
            config, mode="banking", kappa=kappa, n_banks=3, equalize_capacity=True,
        )
        bank_dealer = compile_ring_explorer_balanced(
            config, mode="bank_dealer", kappa=kappa, n_banks=3, equalize_capacity=True,
        )
        bank_dealer_nbfi = compile_ring_explorer_balanced(
            config, mode="bank_dealer_nbfi", kappa=kappa, n_banks=3, equalize_capacity=True,
        )
        b_res = _sum_intermediary_capitals(banking)["bank_reserves"]
        bd_res = _sum_intermediary_capitals(bank_dealer)["bank_reserves"]
        bdn_res = _sum_intermediary_capitals(bank_dealer_nbfi)["bank_reserves"]
        # Banking gets 100%, bank_dealer gets 50%, bank_dealer_nbfi gets 33%
        assert b_res > bd_res > bdn_res
        assert bd_res == pytest.approx(b_res * 0.5, abs=5)
        assert bdn_res == pytest.approx(b_res / 3, abs=5)

    def test_equalize_capacity_stored_in_config(self):
        """The equalize_capacity flag should be stored in _balanced_config."""
        config = _make_config()
        scenario = compile_ring_explorer_balanced(
            config, mode="banking", kappa=Decimal("1"), n_banks=3,
            equalize_capacity=True,
        )
        assert scenario["_balanced_config"]["equalize_capacity"] is True

        scenario2 = compile_ring_explorer_balanced(
            config, mode="banking", kappa=Decimal("1"), n_banks=3,
            equalize_capacity=False,
        )
        assert scenario2["_balanced_config"]["equalize_capacity"] is False

    def test_equalize_capacity_across_kappas(self):
        """Equalization should hold across different kappa values."""
        config = _make_config()
        for kappa_val in ("0.3", "0.5", "1.0", "2.0"):
            kappa = Decimal(kappa_val)
            passive = compile_ring_explorer_balanced(config, mode="passive", kappa=kappa)
            ref = _sum_intermediary_capitals(passive)["total"]

            banking = compile_ring_explorer_balanced(
                config, mode="banking", kappa=kappa, n_banks=3, equalize_capacity=True,
            )
            caps = _sum_intermediary_capitals(banking)
            assert caps["total"] == pytest.approx(ref, abs=5), (
                f"kappa={kappa_val}: banking total {caps['total']:.1f} != ref {ref:.1f}"
            )
