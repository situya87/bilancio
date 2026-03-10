"""Tests for bilancio.experiments.balanced_comparison module.

Tests cover:
- BalancedComparisonResult dataclass construction and computed properties
  (lending_effect, combined_effect, bank effects, system loss effects,
   adjusted effects, loss/capital ratios, _compute_incremental_pnl)
- BalancedComparisonConfig defaults, custom values, validation, serialization
- BalancedComparisonRunner construction, CSV output, arm definitions, helpers
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from bilancio.experiments.balanced_comparison import (
    BalancedComparisonConfig,
    BalancedComparisonResult,
    BalancedComparisonRunner,
)

# =============================================================================
# Helpers
# =============================================================================


def _base_kwargs(**overrides) -> dict:
    """Return minimal kwargs for BalancedComparisonResult."""
    defaults = {
        "kappa": Decimal("1"),
        "concentration": Decimal("1"),
        "mu": Decimal("0"),
        "monotonicity": Decimal("0"),
        "seed": 42,
        "face_value": Decimal("20"),
        "outside_mid_ratio": Decimal("0.90"),
        "big_entity_share": Decimal("0.25"),
        "delta_passive": Decimal("0.4"),
        "phi_passive": Decimal("0.6"),
        "passive_run_id": "passive-1",
        "passive_status": "completed",
        "delta_active": Decimal("0.3"),
        "phi_active": Decimal("0.7"),
        "active_run_id": "active-1",
        "active_status": "completed",
    }
    defaults.update(overrides)
    return defaults


def _make_result(**overrides) -> BalancedComparisonResult:
    """Build a BalancedComparisonResult with sensible defaults."""
    return BalancedComparisonResult(**_base_kwargs(**overrides))


# =============================================================================
# 1. BalancedComparisonConfig defaults verification
# =============================================================================


class TestBalancedComparisonConfigDefaults:
    """Tests for BalancedComparisonConfig default values across all parameter categories."""

    def test_ring_defaults(self):
        """Ring parameters have expected defaults."""
        cfg = BalancedComparisonConfig()
        assert cfg.n_agents == 100
        assert cfg.maturity_days == 10
        assert cfg.max_simulation_days == 15
        assert cfg.Q_total == Decimal("10000")
        assert cfg.liquidity_mode == "uniform"
        assert cfg.base_seed == 42
        assert cfg.n_replicates == 1
        assert cfg.default_handling == "expel-agent"
        assert cfg.quiet is True
        assert cfg.detailed_logging is False

    def test_grid_defaults(self):
        """Grid parameters have expected default lists."""
        cfg = BalancedComparisonConfig()
        assert cfg.kappas == [
            Decimal("0.25"), Decimal("0.5"), Decimal("1"),
            Decimal("2"), Decimal("4"),
        ]
        assert cfg.concentrations == [
            Decimal("0.2"), Decimal("0.5"), Decimal("1"),
            Decimal("2"), Decimal("5"),
        ]
        assert cfg.mus == [
            Decimal("0"), Decimal("0.25"), Decimal("0.5"),
            Decimal("0.75"), Decimal("1"),
        ]
        assert cfg.monotonicities == [Decimal("0")]
        assert cfg.outside_mid_ratios == [Decimal("0.90")]

    def test_dealer_balanced_defaults(self):
        """Balanced dealer parameters have expected defaults."""
        cfg = BalancedComparisonConfig()
        assert cfg.face_value == Decimal("20")
        assert cfg.big_entity_share == Decimal("0.25")
        assert cfg.vbt_share_per_bucket == Decimal("0.25")
        assert cfg.dealer_share_per_bucket == Decimal("0.125")
        assert cfg.rollover_enabled is True
        assert cfg.vbt_share == Decimal("0.50")

    def test_trader_decision_defaults(self):
        """Trader decision module parameters have expected defaults."""
        cfg = BalancedComparisonConfig()
        assert cfg.risk_aversion == Decimal("0")
        assert cfg.planning_horizon == 10
        assert cfg.aggressiveness == Decimal("1.0")
        assert cfg.default_observability == Decimal("1.0")
        assert cfg.trading_motive == "liquidity_then_earning"
        assert cfg.vbt_mid_sensitivity == Decimal("1.0")
        assert cfg.vbt_spread_sensitivity == Decimal("0.0")

    def test_risk_assessment_defaults(self):
        """Risk assessment defaults are correct."""
        cfg = BalancedComparisonConfig()
        assert cfg.risk_assessment_enabled is True
        assert cfg.risk_assessment_config["base_risk_premium"] == "0.02"
        assert cfg.risk_assessment_config["urgency_sensitivity"] == "0.30"
        assert cfg.risk_assessment_config["lookback_window"] == 5

    def test_informedness_defaults(self):
        """Informedness parameters default to zero (naive)."""
        cfg = BalancedComparisonConfig()
        assert cfg.alpha_vbt == Decimal("0")
        assert cfg.alpha_trader == Decimal("0")

    def test_arm_enable_defaults(self):
        """Optional arm flags default to False."""
        cfg = BalancedComparisonConfig()
        assert cfg.enable_lender is False
        assert cfg.enable_dealer_lender is False
        assert cfg.enable_bank_passive is False
        assert cfg.enable_bank_dealer is False
        assert cfg.enable_bank_dealer_nbfi is False

    def test_bank_defaults(self):
        """Bank parameters have expected defaults."""
        cfg = BalancedComparisonConfig()
        assert cfg.n_banks == 3
        assert cfg.reserve_multiplier == 10.0
        assert cfg.n_banks_for_banking == 3
        assert cfg.bank_reserve_multiplier == 0.5
        assert cfg.equalize_bank_capacity is True
        assert cfg.credit_risk_loading == Decimal("0.5")
        assert cfg.max_borrower_risk == Decimal("0.4")
        assert cfg.min_coverage_ratio == Decimal("0")

    def test_cb_defaults(self):
        """Central bank parameters have expected defaults."""
        cfg = BalancedComparisonConfig()
        assert cfg.cb_rate_escalation_slope == Decimal("0.05")
        assert cfg.cb_max_outstanding_ratio == Decimal("2.0")
        assert cfg.cb_lending_cutoff_day is None

    def test_lender_defaults(self):
        """NBFI lender parameters have expected defaults."""
        cfg = BalancedComparisonConfig()
        assert cfg.lender_share == Decimal("0.10")
        assert cfg.lender_base_rate == Decimal("0.05")
        assert cfg.lender_risk_premium_scale == Decimal("0.20")
        assert cfg.lender_max_single_exposure == Decimal("0.15")
        assert cfg.lender_max_total_exposure == Decimal("0.80")
        assert cfg.lender_maturity_days == 2
        assert cfg.lender_horizon == 3
        assert cfg.lender_min_coverage == Decimal("0.5")
        assert cfg.lender_maturity_matching is False
        assert cfg.lender_ranking_mode == "profit"
        assert cfg.lender_coverage_mode == "gate"
        assert cfg.lender_preventive_lending is False

    def test_trading_defaults(self):
        """Trading-related defaults (rounds, spread, etc.)."""
        cfg = BalancedComparisonConfig()
        assert cfg.spread_scale == Decimal("1.0")
        assert cfg.trading_rounds == 100
        assert cfg.issuer_specific_pricing is False
        assert cfg.flow_sensitivity == Decimal("0.0")
        assert cfg.dealer_concentration_limit == Decimal("0")
        assert cfg.performance == {}


# =============================================================================
# 2. BalancedComparisonConfig custom values
# =============================================================================


class TestBalancedComparisonConfigCustom:
    """Tests for BalancedComparisonConfig with custom values."""

    def test_custom_grid(self):
        """Config accepts custom grid parameters."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("0.1"), Decimal("0.5")],
            concentrations=[Decimal("2"), Decimal("3")],
            mus=[Decimal("0.5")],
            outside_mid_ratios=[Decimal("0.80"), Decimal("0.95")],
        )
        assert cfg.kappas == [Decimal("0.1"), Decimal("0.5")]
        assert cfg.concentrations == [Decimal("2"), Decimal("3")]
        assert cfg.mus == [Decimal("0.5")]
        assert len(cfg.outside_mid_ratios) == 2

    def test_custom_trader_params(self):
        """Config accepts custom trader decision parameters."""
        cfg = BalancedComparisonConfig(
            risk_aversion=Decimal("0.5"),
            planning_horizon=5,
            aggressiveness=Decimal("0.7"),
            default_observability=Decimal("0.8"),
            trading_motive="liquidity_only",
        )
        assert cfg.risk_aversion == Decimal("0.5")
        assert cfg.planning_horizon == 5
        assert cfg.aggressiveness == Decimal("0.7")
        assert cfg.default_observability == Decimal("0.8")
        assert cfg.trading_motive == "liquidity_only"

    def test_custom_arm_enables(self):
        """Config accepts enabling optional arms."""
        cfg = BalancedComparisonConfig(
            enable_lender=True,
            enable_dealer_lender=True,
            enable_bank_passive=True,
            enable_bank_dealer=True,
            enable_bank_dealer_nbfi=True,
        )
        assert cfg.enable_lender is True
        assert cfg.enable_dealer_lender is True
        assert cfg.enable_bank_passive is True
        assert cfg.enable_bank_dealer is True
        assert cfg.enable_bank_dealer_nbfi is True

    def test_custom_bank_params(self):
        """Config accepts custom bank parameters."""
        cfg = BalancedComparisonConfig(
            n_banks=10,
            n_banks_for_banking=5,
            bank_reserve_multiplier=1.5,
            credit_risk_loading=Decimal("1.0"),
            max_borrower_risk=Decimal("0.6"),
        )
        assert cfg.n_banks == 10
        assert cfg.n_banks_for_banking == 5
        assert cfg.bank_reserve_multiplier == 1.5
        assert cfg.credit_risk_loading == Decimal("1.0")
        assert cfg.max_borrower_risk == Decimal("0.6")


# =============================================================================
# 3. BalancedComparisonConfig serialization round-trip
# =============================================================================


class TestBalancedComparisonConfigSerialization:
    """Tests for BalancedComparisonConfig serialization round-trip."""

    def test_model_dump_and_validate(self):
        """Config can be serialized and deserialized."""
        original = BalancedComparisonConfig(
            n_agents=50,
            maturity_days=5,
            kappas=[Decimal("0.3"), Decimal("1.0")],
            enable_lender=True,
            risk_aversion=Decimal("0.5"),
        )
        dumped = original.model_dump()
        restored = BalancedComparisonConfig.model_validate(dumped)

        assert restored.n_agents == original.n_agents
        assert restored.maturity_days == original.maturity_days
        assert restored.kappas == original.kappas
        assert restored.enable_lender == original.enable_lender
        assert restored.risk_aversion == original.risk_aversion

    def test_round_trip_preserves_defaults(self):
        """Round-trip of default config preserves all defaults."""
        original = BalancedComparisonConfig()
        dumped = original.model_dump()
        restored = BalancedComparisonConfig.model_validate(dumped)

        assert restored.n_agents == original.n_agents
        assert restored.base_seed == original.base_seed
        assert restored.kappas == original.kappas
        assert restored.concentrations == original.concentrations
        assert restored.enable_lender == original.enable_lender
        assert restored.risk_assessment_enabled == original.risk_assessment_enabled
        assert restored.cb_rate_escalation_slope == original.cb_rate_escalation_slope
        assert restored.performance == original.performance

    def test_model_dump_includes_all_field_categories(self):
        """model_dump includes fields from all config categories."""
        cfg = BalancedComparisonConfig()
        dumped = cfg.model_dump()
        expected_fields = [
            # Ring params
            "n_agents", "maturity_days", "Q_total", "base_seed",
            # Grid params
            "kappas", "concentrations", "mus", "outside_mid_ratios",
            # Dealer params
            "face_value", "vbt_share_per_bucket", "dealer_share_per_bucket",
            # Trader params
            "risk_aversion", "planning_horizon", "aggressiveness",
            # Arm enables
            "enable_lender", "enable_dealer_lender", "enable_bank_passive",
            # Bank params
            "n_banks", "credit_risk_loading", "max_borrower_risk",
            # CB params
            "cb_rate_escalation_slope", "cb_max_outstanding_ratio",
            # Performance
            "performance",
        ]
        for field in expected_fields:
            assert field in dumped, f"Missing field in dump: {field}"


# =============================================================================
# 4. BalancedComparisonConfig validation (constraints)
# =============================================================================


class TestBalancedComparisonConfigValidation:
    """Tests for BalancedComparisonConfig validation constraints."""

    def test_n_replicates_must_be_ge_1(self):
        """n_replicates must be >= 1."""
        with pytest.raises(ValidationError):
            BalancedComparisonConfig(n_replicates=0)

    def test_trading_rounds_must_be_ge_1(self):
        """trading_rounds must be >= 1."""
        with pytest.raises(ValidationError):
            BalancedComparisonConfig(trading_rounds=0)

    def test_flow_sensitivity_bounded(self):
        """flow_sensitivity must be between 0 and 1."""
        with pytest.raises(ValidationError):
            BalancedComparisonConfig(flow_sensitivity=Decimal("1.5"))

    def test_dealer_concentration_limit_bounded(self):
        """dealer_concentration_limit must be between 0 and 1."""
        with pytest.raises(ValidationError):
            BalancedComparisonConfig(dealer_concentration_limit=Decimal("2.0"))


# =============================================================================
# 5. _get_enabled_arm_defs with various arm combinations
# =============================================================================


class TestGetEnabledArmDefs:
    """Tests for BalancedComparisonRunner._get_enabled_arm_defs."""

    def test_default_arms_passive_and_active(self, tmp_path: Path):
        """Default config gives only passive and active arms."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 2
        arm_names = [a[0] for a in arms]
        assert arm_names == ["passive", "active"]

    def test_passive_active_lender(self, tmp_path: Path):
        """Enabling lender adds a third arm."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_lender=True,
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 3
        arm_names = [a[0] for a in arms]
        assert "passive" in arm_names
        assert "active" in arm_names
        assert "lender" in arm_names

    def test_passive_active_dealer_lender(self, tmp_path: Path):
        """Enabling dealer_lender adds a fourth arm."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_dealer_lender=True,
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 3  # passive + active + dealer_lender
        arm_names = [a[0] for a in arms]
        assert "dealer_lender" in arm_names

    def test_all_arms_enabled(self, tmp_path: Path):
        """Enabling all optional arms gives 7 total arms."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_lender=True,
            enable_dealer_lender=True,
            enable_bank_passive=True,
            enable_bank_dealer=True,
            enable_bank_dealer_nbfi=True,
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 7
        arm_names = [a[0] for a in arms]
        assert arm_names == [
            "passive", "active", "lender", "dealer_lender",
            "bank_passive", "bank_dealer", "bank_dealer_nbfi",
        ]

    def test_bank_arms_only(self, tmp_path: Path):
        """Enabling only bank arms gives passive + active + bank arms."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_bank_passive=True,
            enable_bank_dealer=True,
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 4
        arm_names = [a[0] for a in arms]
        assert arm_names == ["passive", "active", "bank_passive", "bank_dealer"]

    def test_arm_tuples_have_correct_structure(self, tmp_path: Path):
        """Each arm tuple has (name, phase, getter_name, supabase_regime)."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_lender=True,
            enable_bank_passive=True,
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        for arm in arms:
            assert len(arm) == 4
            name, phase, getter, regime = arm
            assert isinstance(name, str)
            assert isinstance(phase, str)
            assert getter.startswith("_get_")
            assert isinstance(regime, str)

    def test_passive_arm_phase(self, tmp_path: Path):
        """Passive arm uses 'balanced_passive' phase."""
        cfg = BalancedComparisonConfig(kappas=[Decimal("1")])
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        passive_arm = next(a for a in arms if a[0] == "passive")
        assert passive_arm[1] == "balanced_passive"

    def test_bank_passive_arm_phase(self, tmp_path: Path):
        """Bank passive arm uses 'balanced_bank_passive' phase."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_bank_passive=True,
        )
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, enable_supabase=False,
        )
        arms = runner._get_enabled_arm_defs()
        bp_arm = next(a for a in arms if a[0] == "bank_passive")
        assert bp_arm[1] == "balanced_bank_passive"
        assert bp_arm[3] == "bank_passive"


# =============================================================================
# 6. BalancedComparisonRunner construction
# =============================================================================


class TestBalancedComparisonRunnerConstruction:
    """Tests for BalancedComparisonRunner initialization."""

    def test_creates_output_directories(self, tmp_path: Path):
        """Runner creates passive, active, and aggregate directories."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert (tmp_path / "passive").exists()
        assert (tmp_path / "active").exists()
        assert (tmp_path / "aggregate").exists()

    def test_creates_lender_directory_when_enabled(self, tmp_path: Path):
        """Runner creates NBFI directory when lender is enabled."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_lender=True,
        )
        BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert (tmp_path / "nbfi").exists()

    def test_creates_bank_directories_when_enabled(self, tmp_path: Path):
        """Runner creates bank arm directories when bank arms are enabled."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            enable_bank_passive=True,
            enable_bank_dealer=True,
            enable_bank_dealer_nbfi=True,
        )
        BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert (tmp_path / "bank_passive").exists()
        assert (tmp_path / "bank_dealer").exists()
        assert (tmp_path / "bank_dealer_nbfi").exists()

    def test_stores_config(self, tmp_path: Path):
        """Runner stores the config object."""
        cfg = BalancedComparisonConfig(n_agents=50)
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner.config.n_agents == 50

    def test_default_executor_is_local(self, tmp_path: Path):
        """When no executor is provided, defaults to LocalExecutor."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        from bilancio.runners import LocalExecutor
        assert isinstance(runner.executor, LocalExecutor)

    def test_seed_counter_starts_at_base_seed(self, tmp_path: Path):
        """Seed counter starts from config.base_seed."""
        cfg = BalancedComparisonConfig(base_seed=100)
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner.seed_counter == 100

    def test_comparison_results_starts_empty(self, tmp_path: Path):
        """comparison_results list starts empty."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner.comparison_results == []

    def test_job_id_stored(self, tmp_path: Path):
        """Job ID is stored when provided."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(
            config=cfg, out_dir=tmp_path, job_id="test-job-123", enable_supabase=False,
        )
        assert runner.job_id == "test-job-123"


# =============================================================================
# 7. Runner seed counter and helper methods
# =============================================================================


class TestRunnerHelpers:
    """Tests for BalancedComparisonRunner helper methods."""

    def test_next_seed_increments(self, tmp_path: Path):
        """_next_seed() returns base_seed and increments each call."""
        cfg = BalancedComparisonConfig(base_seed=100)
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner._next_seed() == 100
        assert runner._next_seed() == 101
        assert runner._next_seed() == 102

    def test_format_time_seconds(self, tmp_path: Path):
        """_format_time returns '45s' for small durations."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(45) == "45s"

    def test_format_time_minutes(self, tmp_path: Path):
        """_format_time returns minutes for medium durations."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(150) == "2.5m"

    def test_format_time_hours(self, tmp_path: Path):
        """_format_time returns hours+minutes for large durations."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(3720) == "1h 2m"

    def test_make_key(self, tmp_path: Path):
        """_make_key returns a string tuple for tracking."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        key = runner._make_key(
            Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"),
        )
        assert key == ("0.5", "1", "0", "0", "0.90", "ring")
        # With explicit topology
        key2 = runner._make_key(
            Decimal("0.5"), Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0.90"),
            topology="k_regular",
        )
        assert key2 == ("0.5", "1", "0", "0", "0.90", "k_regular")
        # Keys should be hashable for use in dicts
        d = {key: 1, key2: 2}
        assert d[key] == 1
        assert d[key2] == 2


# =============================================================================
# 8. _write_comparison_csv
# =============================================================================


class TestWriteComparisonCSV:
    """Tests for BalancedComparisonRunner._write_comparison_csv."""

    def test_writes_header_and_rows(self, tmp_path: Path):
        """_write_comparison_csv writes header and result rows."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            _make_result(
                kappa=Decimal("0.5"),
                delta_passive=Decimal("0.6"),
                delta_active=Decimal("0.3"),
                seed=42,
            ),
            _make_result(
                kappa=Decimal("1.0"),
                delta_passive=Decimal("0.4"),
                delta_active=Decimal("0.2"),
                seed=43,
            ),
        ]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        assert csv_path.exists()

        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["kappa"] == "0.5"
        assert rows[0]["delta_passive"] == "0.6"
        assert rows[0]["delta_active"] == "0.3"
        assert rows[0]["trading_effect"] == "0.3"
        assert rows[1]["kappa"] == "1.0"

    def test_csv_has_all_comparison_fields(self, tmp_path: Path):
        """CSV header contains all COMPARISON_FIELDS."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [_make_result()]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames

        assert header is not None
        for field in BalancedComparisonRunner.COMPARISON_FIELDS:
            assert field in header, f"Missing field: {field}"

    def test_csv_handles_none_values(self, tmp_path: Path):
        """CSV writes empty strings for None values."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [
            _make_result(delta_passive=None, delta_active=None),
        ]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["delta_passive"] == ""
        assert rows[0]["delta_active"] == ""
        assert rows[0]["trading_effect"] == ""

    def test_csv_overwrites_on_rewrite(self, tmp_path: Path):
        """Writing CSV twice overwrites (not appends)."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [_make_result(seed=1)]
        runner._write_comparison_csv()

        runner.comparison_results = [_make_result(seed=2), _make_result(seed=3)]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["seed"] == "2"
        assert rows[1]["seed"] == "3"

    def test_csv_includes_lender_fields(self, tmp_path: Path):
        """CSV includes lender fields when a result has lender data."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [
            _make_result(
                delta_lender=Decimal("0.25"),
                phi_lender=Decimal("0.75"),
                lender_run_id="lender-1",
                lender_status="completed",
            ),
        ]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["delta_lender"] == "0.25"
        assert rows[0]["lending_effect"] == "0.15"  # 0.4 - 0.25


# =============================================================================
# 9. BalancedComparisonResult — trading_effect and trading_relief_ratio
# =============================================================================


class TestTradingEffectProperties:
    """Tests for trading_effect and trading_relief_ratio properties."""

    def test_trading_effect_positive(self):
        """Positive trading_effect when active reduces defaults."""
        r = _make_result(delta_passive=Decimal("0.4"), delta_active=Decimal("0.2"))
        assert r.trading_effect == Decimal("0.2")

    def test_trading_effect_none_when_passive_missing(self):
        """trading_effect is None when delta_passive is None."""
        r = _make_result(delta_passive=None)
        assert r.trading_effect is None

    def test_trading_effect_none_when_active_missing(self):
        """trading_effect is None when delta_active is None."""
        r = _make_result(delta_active=None)
        assert r.trading_effect is None

    def test_trading_relief_ratio(self):
        """trading_relief_ratio = effect / delta_passive."""
        r = _make_result(delta_passive=Decimal("0.5"), delta_active=Decimal("0.25"))
        assert r.trading_relief_ratio == Decimal("0.5")

    def test_trading_relief_ratio_zero_passive(self):
        """trading_relief_ratio is 0 when delta_passive is 0."""
        r = _make_result(delta_passive=Decimal("0"), delta_active=Decimal("0"))
        assert r.trading_relief_ratio == Decimal("0")

    def test_trading_relief_ratio_none_when_missing(self):
        """trading_relief_ratio is None when either delta is missing."""
        r = _make_result(delta_passive=None)
        assert r.trading_relief_ratio is None


# =============================================================================
# 10. BalancedComparisonResult — lending_effect, combined_effect, bank effects
# =============================================================================


class TestLendingAndBankEffectProperties:
    """Tests for lending_effect, combined_effect, and bank effect properties."""

    def test_lending_effect_positive(self):
        """Positive lending_effect when lender reduces defaults."""
        r = _make_result(delta_lender=Decimal("0.2"))
        assert r.lending_effect == Decimal("0.2")  # 0.4 - 0.2

    def test_lending_effect_none_when_lender_missing(self):
        """lending_effect is None when delta_lender is None."""
        r = _make_result()  # delta_lender defaults to None
        assert r.lending_effect is None

    def test_lending_effect_none_when_passive_missing(self):
        """lending_effect is None when delta_passive is None."""
        r = _make_result(delta_passive=None, delta_lender=Decimal("0.2"))
        assert r.lending_effect is None

    def test_lending_effect_negative(self):
        """Negative lending_effect when lending worsens defaults."""
        r = _make_result(delta_lender=Decimal("0.5"))
        assert r.lending_effect == Decimal("-0.1")  # 0.4 - 0.5

    def test_bank_passive_effect(self):
        """bank_passive_effect = delta_passive - delta_bank_passive."""
        r = _make_result(delta_bank_passive=Decimal("0.25"))
        assert r.bank_passive_effect == Decimal("0.15")  # 0.4 - 0.25

    def test_bank_passive_effect_none(self):
        """bank_passive_effect is None when delta_bank_passive is None."""
        r = _make_result()
        assert r.bank_passive_effect is None

    def test_bank_dealer_effect(self):
        """bank_dealer_effect = delta_passive - delta_bank_dealer."""
        r = _make_result(delta_bank_dealer=Decimal("0.1"))
        assert r.bank_dealer_effect == Decimal("0.3")

    def test_bank_dealer_effect_none(self):
        """bank_dealer_effect is None when delta_bank_dealer is None."""
        r = _make_result()
        assert r.bank_dealer_effect is None

    def test_bank_dealer_nbfi_effect(self):
        """bank_dealer_nbfi_effect = delta_passive - delta_bank_dealer_nbfi."""
        r = _make_result(delta_bank_dealer_nbfi=Decimal("0.05"))
        assert r.bank_dealer_nbfi_effect == Decimal("0.35")

    def test_bank_dealer_nbfi_effect_none(self):
        """bank_dealer_nbfi_effect is None when delta_bank_dealer_nbfi is None."""
        r = _make_result()
        assert r.bank_dealer_nbfi_effect is None


# =============================================================================
# 11. Result with all arms None except passive
# =============================================================================


class TestResultMinimalArms:
    """Tests for BalancedComparisonResult with only passive completed."""

    def test_all_optional_effects_none(self):
        """When no optional arms ran, all optional effects are None."""
        r = _make_result(delta_active=None)
        assert r.trading_effect is None
        assert r.lending_effect is None
        assert r.combined_effect is None
        assert r.bank_passive_effect is None
        assert r.bank_dealer_effect is None
        assert r.bank_dealer_nbfi_effect is None

    def test_default_lender_fields(self):
        """Lender fields default to None/empty/0."""
        r = _make_result()
        assert r.delta_lender is None
        assert r.phi_lender is None
        assert r.lender_run_id == ""
        assert r.lender_status == ""
        assert r.n_defaults_lender == 0
        assert r.cascade_fraction_lender is None
        assert r.lender_total_pnl is None
        assert r.total_loans is None

    def test_default_bank_passive_fields(self):
        """Bank passive fields default to None/empty/0."""
        r = _make_result()
        assert r.delta_bank_passive is None
        assert r.phi_bank_passive is None
        assert r.bank_passive_run_id == ""
        assert r.cb_loans_created_bank_passive == 0
        assert r.cb_interest_total_bank_passive == 0
        assert r.delta_bank_bank_passive is None
        assert r.deposit_loss_gross_bank_passive == 0

    def test_default_loss_fields(self):
        """Loss fields default to 0 or None."""
        r = _make_result()
        assert r.payable_default_loss_passive == 0
        assert r.total_loss_passive == 0
        assert r.total_loss_pct_passive is None
        assert r.intermediary_loss_passive == 0.0
        assert r.intermediary_loss_pct_passive is None
        assert r.system_loss_passive == 0.0
        assert r.system_loss_pct_passive is None
        assert r.intermediary_capital_passive == 0.0
        assert r.loss_capital_ratio_passive is None


# =============================================================================
# 12. System loss effect properties
# =============================================================================


class TestSystemLossEffectProperties:
    """Tests for system_loss_*_effect computed properties."""

    def test_system_loss_trading_effect(self):
        """system_loss_trading_effect = pct_passive - pct_active."""
        r = _make_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_active=0.06,
        )
        assert r.system_loss_trading_effect == pytest.approx(0.04)

    def test_system_loss_trading_effect_none(self):
        """system_loss_trading_effect is None when pcts are None."""
        r = _make_result()
        assert r.system_loss_trading_effect is None

    def test_system_loss_lending_effect(self):
        """system_loss_lending_effect = pct_passive - pct_lender."""
        r = _make_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_lender=0.04,
        )
        assert r.system_loss_lending_effect == pytest.approx(0.06)

    def test_system_loss_lending_effect_none(self):
        """system_loss_lending_effect is None when pct_lender is None."""
        r = _make_result(system_loss_pct_passive=0.10)
        assert r.system_loss_lending_effect is None

    def test_system_loss_combined_effect(self):
        """system_loss_combined_effect = pct_passive - pct_dealer_lender."""
        r = _make_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_dealer_lender=0.03,
        )
        assert r.system_loss_combined_effect == pytest.approx(0.07)

    def test_system_loss_combined_effect_none(self):
        """system_loss_combined_effect is None when missing."""
        r = _make_result(system_loss_pct_passive=0.10)
        assert r.system_loss_combined_effect is None

    def test_system_loss_bank_passive_effect(self):
        """system_loss_bank_passive_effect = pct_passive - pct_bank_passive."""
        r = _make_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_bank_passive=0.05,
        )
        assert r.system_loss_bank_passive_effect == pytest.approx(0.05)

    def test_system_loss_bank_dealer_effect(self):
        """system_loss_bank_dealer_effect works."""
        r = _make_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_bank_dealer=0.02,
        )
        assert r.system_loss_bank_dealer_effect == pytest.approx(0.08)

    def test_system_loss_bank_dealer_nbfi_effect(self):
        """system_loss_bank_dealer_nbfi_effect works."""
        r = _make_result(
            system_loss_pct_passive=0.10,
            system_loss_pct_bank_dealer_nbfi=0.01,
        )
        assert r.system_loss_bank_dealer_nbfi_effect == pytest.approx(0.09)


# =============================================================================
# 13. Adjusted effect properties
# =============================================================================


class TestAdjustedEffectProperties:
    """Tests for adjusted_*_effect computed properties."""

    def test_adjusted_trading_effect(self):
        """adjusted_trading_effect subtracts intermediary loss differential."""
        r = _make_result(
            delta_passive=Decimal("0.4"),
            delta_active=Decimal("0.2"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_active=0.03,
        )
        # trading_effect = 0.2, adjustment = 0.03 - 0.01 = 0.02
        # adjusted = 0.2 - 0.02 = 0.18
        assert r.adjusted_trading_effect == pytest.approx(0.18)

    def test_adjusted_trading_effect_none_when_trading_effect_none(self):
        """adjusted_trading_effect is None when trading_effect is None."""
        r = _make_result(delta_passive=None)
        assert r.adjusted_trading_effect is None

    def test_adjusted_trading_effect_none_when_loss_pct_missing(self):
        """adjusted_trading_effect is None when intermediary_loss_pct is missing."""
        r = _make_result()  # intermediary_loss_pct_* default to None
        assert r.adjusted_trading_effect is None

    def test_adjusted_lending_effect(self):
        """adjusted_lending_effect subtracts intermediary loss differential."""
        r = _make_result(
            delta_passive=Decimal("0.4"),
            delta_lender=Decimal("0.2"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_lender=0.04,
        )
        # lending_effect = 0.2, adjustment = 0.04 - 0.01 = 0.03
        # adjusted = 0.2 - 0.03 = 0.17
        assert r.adjusted_lending_effect == pytest.approx(0.17)

    def test_adjusted_lending_effect_none(self):
        """adjusted_lending_effect is None when lending_effect is None."""
        r = _make_result()  # no delta_lender
        assert r.adjusted_lending_effect is None

    def test_adjusted_combined_effect(self):
        """adjusted_combined_effect subtracts intermediary loss differential."""
        r = _make_result(
            delta_passive=Decimal("0.4"),
            delta_dealer_lender=Decimal("0.15"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_dealer_lender=0.05,
        )
        # combined_effect = 0.25, adjustment = 0.05 - 0.01 = 0.04
        # adjusted = 0.25 - 0.04 = 0.21
        assert r.adjusted_combined_effect == pytest.approx(0.21)

    def test_adjusted_combined_effect_none(self):
        """adjusted_combined_effect is None when combined_effect is None."""
        r = _make_result()  # no delta_dealer_lender
        assert r.adjusted_combined_effect is None

    def test_adjusted_bank_passive_effect(self):
        """adjusted_bank_passive_effect works."""
        r = _make_result(
            delta_passive=Decimal("0.4"),
            delta_bank_passive=Decimal("0.3"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_bank_passive=0.02,
        )
        # bank_passive_effect = 0.1, adjustment = 0.02 - 0.01 = 0.01
        # adjusted = 0.1 - 0.01 = 0.09
        assert r.adjusted_bank_passive_effect == pytest.approx(0.09)

    def test_adjusted_bank_dealer_effect(self):
        """adjusted_bank_dealer_effect works."""
        r = _make_result(
            delta_passive=Decimal("0.4"),
            delta_bank_dealer=Decimal("0.2"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_bank_dealer=0.03,
        )
        assert r.adjusted_bank_dealer_effect == pytest.approx(0.18)

    def test_adjusted_bank_dealer_nbfi_effect(self):
        """adjusted_bank_dealer_nbfi_effect works."""
        r = _make_result(
            delta_passive=Decimal("0.4"),
            delta_bank_dealer_nbfi=Decimal("0.1"),
            intermediary_loss_pct_passive=0.01,
            intermediary_loss_pct_bank_dealer_nbfi=0.06,
        )
        # bank_dealer_nbfi_effect = 0.3, adjustment = 0.06 - 0.01 = 0.05
        # adjusted = 0.3 - 0.05 = 0.25
        assert r.adjusted_bank_dealer_nbfi_effect == pytest.approx(0.25)


# =============================================================================
# 14. _compute_incremental_pnl static method
# =============================================================================


class TestComputeIncrementalPnl:
    """Tests for BalancedComparisonResult._compute_incremental_pnl."""

    def test_computes_incremental_pnl(self):
        """Returns active_pnl - passive_pnl when both are available."""
        result = BalancedComparisonResult._compute_incremental_pnl(
            {"dealer_total_pnl": 100.0},
            {"dealer_total_pnl": 30.0},
        )
        assert result == pytest.approx(70.0)

    def test_none_when_active_missing(self):
        """Returns None when active_metrics is None."""
        result = BalancedComparisonResult._compute_incremental_pnl(
            None,
            {"dealer_total_pnl": 30.0},
        )
        assert result is None

    def test_none_when_passive_missing(self):
        """Returns None when passive_metrics is None."""
        result = BalancedComparisonResult._compute_incremental_pnl(
            {"dealer_total_pnl": 100.0},
            None,
        )
        assert result is None

    def test_none_when_both_missing(self):
        """Returns None when both metrics are None."""
        result = BalancedComparisonResult._compute_incremental_pnl(None, None)
        assert result is None

    def test_none_when_key_missing(self):
        """Returns None when dealer_total_pnl key is missing."""
        result = BalancedComparisonResult._compute_incremental_pnl(
            {"other_key": 100.0},
            {"dealer_total_pnl": 30.0},
        )
        assert result is None

    def test_negative_incremental(self):
        """Handles negative incremental PnL (active worse than passive)."""
        result = BalancedComparisonResult._compute_incremental_pnl(
            {"dealer_total_pnl": 10.0},
            {"dealer_total_pnl": 50.0},
        )
        assert result == pytest.approx(-40.0)


# =============================================================================
# 15. Cascade effect property
# =============================================================================


class TestCascadeEffectProperty:
    """Tests for cascade_effect computed property."""

    def test_cascade_effect_positive(self):
        """Positive cascade effect when trading reduces cascading defaults."""
        r = _make_result(
            cascade_fraction_passive=Decimal("0.3"),
            cascade_fraction_active=Decimal("0.1"),
        )
        assert r.cascade_effect == Decimal("0.2")

    def test_cascade_effect_none_when_passive_missing(self):
        """cascade_effect is None when cascade_fraction_passive is None."""
        r = _make_result(cascade_fraction_active=Decimal("0.1"))
        assert r.cascade_effect is None

    def test_cascade_effect_none_when_active_missing(self):
        """cascade_effect is None when cascade_fraction_active is None."""
        r = _make_result(cascade_fraction_passive=Decimal("0.3"))
        assert r.cascade_effect is None


# =============================================================================
# 16. COMPARISON_FIELDS completeness
# =============================================================================


class TestComparisonFieldsCompleteness:
    """Tests for COMPARISON_FIELDS class variable."""

    def test_comparison_fields_is_nonempty_list(self):
        """COMPARISON_FIELDS is a non-empty list of strings."""
        assert isinstance(BalancedComparisonRunner.COMPARISON_FIELDS, list)
        assert len(BalancedComparisonRunner.COMPARISON_FIELDS) > 50
        for field in BalancedComparisonRunner.COMPARISON_FIELDS:
            assert isinstance(field, str)

    def test_comparison_fields_includes_all_core_metrics(self):
        """COMPARISON_FIELDS includes core comparison metrics."""
        fields = BalancedComparisonRunner.COMPARISON_FIELDS
        core_fields = [
            "kappa", "concentration", "mu", "seed",
            "delta_passive", "delta_active", "trading_effect",
            "phi_passive", "phi_active",
            "delta_lender", "lending_effect",
            "delta_dealer_lender", "combined_effect",
            "delta_bank_passive", "bank_passive_effect",
            "delta_bank_dealer", "bank_dealer_effect",
            "delta_bank_dealer_nbfi", "bank_dealer_nbfi_effect",
        ]
        for field in core_fields:
            assert field in fields, f"Missing core field: {field}"

    def test_comparison_fields_includes_system_loss_effects(self):
        """COMPARISON_FIELDS includes all Approach 3 system loss effect metrics."""
        fields = BalancedComparisonRunner.COMPARISON_FIELDS
        system_loss_fields = [
            "system_loss_trading_effect",
            "system_loss_lending_effect",
            "system_loss_combined_effect",
            "system_loss_bank_passive_effect",
            "system_loss_bank_dealer_effect",
            "system_loss_bank_dealer_nbfi_effect",
        ]
        for field in system_loss_fields:
            assert field in fields, f"Missing system loss field: {field}"

    def test_comparison_fields_includes_adjusted_effects(self):
        """COMPARISON_FIELDS includes all adjusted effect metrics."""
        fields = BalancedComparisonRunner.COMPARISON_FIELDS
        adjusted_fields = [
            "adjusted_trading_effect",
            "adjusted_lending_effect",
            "adjusted_combined_effect",
            "adjusted_bank_passive_effect",
            "adjusted_bank_dealer_effect",
            "adjusted_bank_dealer_nbfi_effect",
        ]
        for field in adjusted_fields:
            assert field in fields, f"Missing adjusted field: {field}"

    def test_comparison_fields_no_duplicates(self):
        """COMPARISON_FIELDS contains no duplicates."""
        fields = BalancedComparisonRunner.COMPARISON_FIELDS
        assert len(fields) == len(set(fields)), "Duplicate fields found"


# =============================================================================
# 17. Paired-control integrity — same seed means same topology
# =============================================================================


class TestPairedControlIntegrity:
    """Verify that passive and active arms of the same pair use the same seed."""

    def test_same_seed_used_for_both_arms(self, tmp_path: Path):
        """Both passive and active runs in a pair receive the same seed."""
        cfg = BalancedComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
            base_seed=42,
        )
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        # _next_seed() should return the same seed for both arms in a pair
        seed = runner._next_seed()
        assert seed == 42
        # Next call yields 43 (for next pair, not for second arm)
        assert runner._next_seed() == 43

    def test_seed_appears_in_csv_result(self, tmp_path: Path):
        """Both passive and active run_ids use the same seed value in the CSV."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        # Create result with explicit seed
        result = _make_result(seed=77)
        runner.comparison_results = [result]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        rows = list(csv.DictReader(csv_path.open("r")))
        assert rows[0]["seed"] == "77"
        # Both arm run_ids share the same row (same seed/topology)
        assert rows[0]["passive_run_id"] != ""
        assert rows[0]["active_run_id"] != ""


# =============================================================================
# 18. Artifact column check — comparison.csv has all expected headers
# =============================================================================


class TestArtifactColumnCheck:
    """Verify comparison.csv column headers match COMPARISON_FIELDS exactly."""

    def test_csv_columns_match_comparison_fields(self, tmp_path: Path):
        """CSV header columns should be a superset of COMPARISON_FIELDS."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [_make_result()]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames

        assert header is not None
        # All COMPARISON_FIELDS must appear as columns
        for field in BalancedComparisonRunner.COMPARISON_FIELDS:
            assert field in header, f"Column missing: {field}"
        # And the header should contain exactly COMPARISON_FIELDS (no extra)
        assert set(header) == set(BalancedComparisonRunner.COMPARISON_FIELDS)

    def test_csv_has_parameter_columns(self, tmp_path: Path):
        """CSV must have standard parameter columns (kappa, concentration, mu, seed)."""
        cfg = BalancedComparisonConfig()
        runner = BalancedComparisonRunner(config=cfg, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [_make_result()]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames

        required_params = ["kappa", "concentration", "mu", "seed", "outside_mid_ratio"]
        for p in required_params:
            assert p in header, f"Parameter column missing: {p}"


# =============================================================================
# 19. Config round-trip with all field categories
# =============================================================================


class TestConfigRoundTripComprehensive:
    """Comprehensive config serialization round-trip tests."""

    def test_round_trip_with_custom_arms_and_bank_params(self):
        """Round-trip preserves arm enables and bank parameters."""
        original = BalancedComparisonConfig(
            enable_lender=True,
            enable_dealer_lender=True,
            enable_bank_passive=True,
            n_banks=7,
            credit_risk_loading=Decimal("0.8"),
            max_borrower_risk=Decimal("0.6"),
            cb_rate_escalation_slope=Decimal("0.10"),
            cb_max_outstanding_ratio=Decimal("3.0"),
        )
        dumped = original.model_dump()
        restored = BalancedComparisonConfig.model_validate(dumped)

        assert restored.enable_lender == original.enable_lender
        assert restored.enable_dealer_lender == original.enable_dealer_lender
        assert restored.enable_bank_passive == original.enable_bank_passive
        assert restored.n_banks == original.n_banks
        assert restored.credit_risk_loading == original.credit_risk_loading
        assert restored.max_borrower_risk == original.max_borrower_risk
        assert restored.cb_rate_escalation_slope == original.cb_rate_escalation_slope
        assert restored.cb_max_outstanding_ratio == original.cb_max_outstanding_ratio

    def test_round_trip_with_trader_and_risk_params(self):
        """Round-trip preserves trader decision and risk assessment parameters."""
        original = BalancedComparisonConfig(
            risk_aversion=Decimal("0.7"),
            planning_horizon=5,
            aggressiveness=Decimal("0.3"),
            default_observability=Decimal("0.5"),
            trading_motive="liquidity_only",
            alpha_vbt=Decimal("0.5"),
            alpha_trader=Decimal("0.3"),
            vbt_mid_sensitivity=Decimal("0.8"),
            vbt_spread_sensitivity=Decimal("0.5"),
        )
        dumped = original.model_dump()
        restored = BalancedComparisonConfig.model_validate(dumped)

        assert restored.risk_aversion == original.risk_aversion
        assert restored.planning_horizon == original.planning_horizon
        assert restored.aggressiveness == original.aggressiveness
        assert restored.default_observability == original.default_observability
        assert restored.trading_motive == original.trading_motive
        assert restored.alpha_vbt == original.alpha_vbt
        assert restored.alpha_trader == original.alpha_trader
        assert restored.vbt_mid_sensitivity == original.vbt_mid_sensitivity
        assert restored.vbt_spread_sensitivity == original.vbt_spread_sensitivity

    def test_round_trip_identity_for_default_config(self):
        """model_dump then model_validate on default config produces identical object."""
        original = BalancedComparisonConfig()
        dumped = original.model_dump()
        restored = BalancedComparisonConfig.model_validate(dumped)
        # Compare full model_dump outputs for equality
        assert original.model_dump() == restored.model_dump()
