"""
Tests for balanced dealer wiring across models.py, apply.py, and ring.py.

Verifies that:
- BalancedDealerConfig validates correctly
- apply_to_system dispatches to initialize_balanced_dealer_subsystem when enabled
- RingSweepRunner._prepare_run injects the balanced_dealer section into scenario dicts
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from bilancio.config.apply import apply_to_system
from bilancio.config.models import BalancedDealerConfig, ScenarioConfig
from bilancio.engines.system import System

# ---------------------------------------------------------------------------
# Group 1: TestBalancedDealerConfig — Model validation (models.py)
# ---------------------------------------------------------------------------


class TestBalancedDealerConfig:
    def test_big_entity_share_allows_zero(self):
        """big_entity_share=0 should be valid (0 <= v < 1)."""
        cfg = BalancedDealerConfig(big_entity_share=Decimal("0"))
        assert cfg.big_entity_share == Decimal("0")

    def test_big_entity_share_rejects_negative(self):
        """big_entity_share=-0.1 must raise ValidationError."""
        with pytest.raises(ValidationError, match="big_entity_share"):
            BalancedDealerConfig(big_entity_share=Decimal("-0.1"))

    def test_balanced_dealer_config_defaults(self):
        """Default values match the specification."""
        cfg = BalancedDealerConfig()
        assert cfg.enabled is False
        assert cfg.face_value == Decimal("20")
        assert cfg.outside_mid_ratio == Decimal("0.75")
        assert cfg.big_entity_share == Decimal("0.25")
        assert cfg.vbt_share_per_bucket == Decimal("0.20")
        assert cfg.dealer_share_per_bucket == Decimal("0.05")
        assert cfg.rollover_enabled is True
        assert cfg.mode == "active"

    def test_scenario_config_accepts_balanced_dealer(self):
        """ScenarioConfig with balanced_dealer={...} parses correctly."""
        data = {
            "name": "test-scenario",
            "agents": [
                {"id": "CB", "kind": "central_bank", "name": "CB"},
                {"id": "HH1", "kind": "household", "name": "HH1"},
            ],
            "balanced_dealer": {
                "enabled": True,
                "face_value": "20",
                "outside_mid_ratio": "0.75",
                "mode": "active",
            },
        }
        cfg = ScenarioConfig(**data)
        assert cfg.balanced_dealer is not None
        assert cfg.balanced_dealer.enabled is True
        assert cfg.balanced_dealer.face_value == Decimal("20")
        assert cfg.balanced_dealer.outside_mid_ratio == Decimal("0.75")
        assert cfg.balanced_dealer.mode == "active"


# ---------------------------------------------------------------------------
# Group 2: TestApplyBalancedDealer — Config apply wiring (apply.py)
# ---------------------------------------------------------------------------


def _balanced_scenario_config(*, enabled=True, mode="active"):
    """Build a minimal ScenarioConfig dict for balanced dealer testing.

    Creates: CB + 2 traders + vbt_short + dealer_short, payables to each, cash minted.
    VBT/Dealer agents use kind="household" (same as compile_ring_explorer_balanced).
    The dealer section is always included (balanced_dealer requires dealer to be present).
    """
    agents = [
        {"id": "CB", "kind": "central_bank", "name": "Central Bank"},
        {"id": "T1", "kind": "household", "name": "Trader 1"},
        {"id": "T2", "kind": "household", "name": "Trader 2"},
        {"id": "vbt_short", "kind": "household", "name": "VBT (short)"},
        {"id": "dealer_short", "kind": "household", "name": "Dealer (short)"},
    ]

    initial_actions = [
        # Mint cash to traders and big entities
        {"mint_cash": {"to": "T1", "amount": 100}},
        {"mint_cash": {"to": "T2", "amount": 100}},
        {"mint_cash": {"to": "vbt_short", "amount": 50}},
        {"mint_cash": {"to": "dealer_short", "amount": 30}},
        # Payables: T1 owes T2, T2 owes T1 (ring)
        {
            "create_payable": {
                "from": "T1",
                "to": "T2",
                "amount": 20,
                "due_day": 3,
            }
        },
        {
            "create_payable": {
                "from": "T2",
                "to": "T1",
                "amount": 20,
                "due_day": 3,
            }
        },
        # Payable held by VBT
        {
            "create_payable": {
                "from": "T1",
                "to": "vbt_short",
                "amount": 20,
                "due_day": 2,
            }
        },
        # Payable held by Dealer
        {
            "create_payable": {
                "from": "T2",
                "to": "dealer_short",
                "amount": 20,
                "due_day": 2,
            }
        },
    ]

    dealer = {
        "enabled": True,
        "ticket_size": 1,
        "dealer_share": "0.25",
        "vbt_share": "0.50",
    }

    balanced_dealer = {
        "enabled": enabled,
        "face_value": "20",
        "outside_mid_ratio": "0.75",
        "mode": mode,
    }

    return {
        "name": "balanced-test",
        "agents": agents,
        "initial_actions": initial_actions,
        "dealer": dealer,
        "balanced_dealer": balanced_dealer,
    }


class TestApplyBalancedDealer:
    def test_balanced_dealer_uses_correct_vbt_anchors(self):
        """With balanced_dealer.enabled=True: VBT anchor M = ρ × (1 - P_default_prior).

        With no kappa, default prior=0.15, ρ=0.75: M = 0.75 × 0.85 = 0.6375.
        The VBT is credit-aware and discounts by estimated default probability,
        scaled by the outside_mid_ratio (ρ).
        """
        data = _balanced_scenario_config(enabled=True, mode="active")
        config = ScenarioConfig(**data)
        system = System()
        apply_to_system(config, system)

        subsystem = system.state.dealer_subsystem
        assert subsystem is not None

        # M = ρ × (1 - shared_prior) = 0.75 × 0.85 = 0.6375
        rho = Decimal("0.75")
        expected_M = rho * (Decimal(1) - Decimal("0.15"))
        for bucket_id, vbt in subsystem.vbts.items():
            assert vbt.M == expected_M, (
                f"VBT bucket '{bucket_id}' M={vbt.M}, expected {expected_M} "
                f"(ρ × (1 - 0.15) credit-adjusted)"
            )

    def test_balanced_dealer_gives_inventory(self):
        """Dealer buckets have non-empty inventory (tickets from payables to dealer_* agents)."""
        data = _balanced_scenario_config(enabled=True, mode="active")
        config = ScenarioConfig(**data)
        system = System()
        apply_to_system(config, system)

        subsystem = system.state.dealer_subsystem
        assert subsystem is not None

        # dealer_short should have inventory from P_DLR
        dealer_short = subsystem.dealers.get("short")
        assert dealer_short is not None, "dealer_short bucket must exist"
        assert len(dealer_short.inventory) > 0, (
            "Dealer 'short' bucket should have inventory from payable P_DLR"
        )

    def test_no_balanced_dealer_uses_generic_init(self):
        """Without balanced_dealer: dealers start with empty inventory, VBT anchors default M=1.0."""
        data = _balanced_scenario_config(enabled=False, mode="active")
        config = ScenarioConfig(**data)
        system = System()
        apply_to_system(config, system)

        subsystem = system.state.dealer_subsystem
        assert subsystem is not None

        # Generic init uses default VBT anchors (M=1.0 from DealerBucketConfig defaults)
        for bucket_id, vbt in subsystem.vbts.items():
            assert vbt.M == Decimal("1.0"), (
                f"Generic init VBT bucket '{bucket_id}' M={vbt.M}, expected 1.0"
            )

    def test_passive_mode_disables_subsystem(self):
        """mode='passive' → subsystem.enabled == False."""
        data = _balanced_scenario_config(enabled=True, mode="passive")
        config = ScenarioConfig(**data)
        system = System()
        apply_to_system(config, system)

        subsystem = system.state.dealer_subsystem
        assert subsystem is not None
        assert subsystem.enabled is False, "Passive mode should set subsystem.enabled=False"


# ---------------------------------------------------------------------------
# Group 3: TestRingScenarioDictInjection — Scenario dict construction (ring.py)
# ---------------------------------------------------------------------------


class TestRingScenarioDictInjection:
    def _make_runner(self, tmp_path, *, balanced_mode, dealer_enabled):
        """Create a RingSweepRunner with minimal config."""
        from bilancio.experiments.ring import RingSweepRunner

        return RingSweepRunner(
            out_dir=tmp_path / "experiment",
            name_prefix="test",
            n_agents=4,
            maturity_days=5,
            Q_total=Decimal("100"),
            liquidity_mode="uniform",
            liquidity_agent=None,
            base_seed=42,
            dealer_enabled=dealer_enabled,
            balanced_mode=balanced_mode,
            face_value=Decimal("20"),
            outside_mid_ratio=Decimal("0.75"),
            vbt_share_per_bucket=Decimal("0.25"),
            dealer_share_per_bucket=Decimal("0.125"),
            rollover_enabled=True,
        )

    def test_balanced_mode_adds_balanced_dealer_section(self, tmp_path):
        """Scenario dict has 'balanced_dealer' key with expected fields."""
        runner = self._make_runner(tmp_path, balanced_mode=True, dealer_enabled=True)
        prepared = runner._prepare_run(
            phase="active",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0.5"),
            monotonicity=Decimal("0"),
            seed=42,
        )

        sc = prepared.scenario_config
        assert "balanced_dealer" in sc, "balanced_mode=True should add 'balanced_dealer' section"

        bd = sc["balanced_dealer"]
        assert bd["enabled"] is True
        assert "face_value" in bd
        assert "outside_mid_ratio" in bd
        assert "mode" in bd
        assert "rollover_enabled" in bd

    def test_non_balanced_mode_no_balanced_dealer(self, tmp_path):
        """balanced_mode=False → no 'balanced_dealer' key in scenario dict."""
        runner = self._make_runner(tmp_path, balanced_mode=False, dealer_enabled=True)
        prepared = runner._prepare_run(
            phase="passive",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0.5"),
            monotonicity=Decimal("0"),
            seed=42,
        )

        sc = prepared.scenario_config
        assert "balanced_dealer" not in sc, (
            "balanced_mode=False should NOT add 'balanced_dealer' section"
        )

    def test_balanced_dealer_values_match_runner(self, tmp_path):
        """Field values in scenario['balanced_dealer'] match the runner's attributes."""
        runner = self._make_runner(tmp_path, balanced_mode=True, dealer_enabled=True)
        prepared = runner._prepare_run(
            phase="active",
            kappa=Decimal("1"),
            concentration=Decimal("1"),
            mu=Decimal("0.5"),
            monotonicity=Decimal("0"),
            seed=42,
        )

        bd = prepared.scenario_config["balanced_dealer"]
        assert bd["face_value"] == str(runner.face_value)
        assert bd["outside_mid_ratio"] == str(runner.outside_mid_ratio)
        assert bd["vbt_share_per_bucket"] == str(runner.vbt_share_per_bucket)
        assert bd["dealer_share_per_bucket"] == str(runner.dealer_share_per_bucket)
        assert bd["rollover_enabled"] == runner.rollover_enabled
        assert bd["mode"] == "active"
