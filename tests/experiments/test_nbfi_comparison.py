"""Tests for bilancio.experiments.nbfi_comparison module.

Tests cover:
- NBFIComparisonResult dataclass construction and computed properties
- NBFIComparisonConfig defaults, validation, custom values, serialization
- NBFIComparisonRunner construction, CSV output, and helper methods
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from bilancio.experiments.nbfi_comparison import (
    NBFIComparisonConfig,
    NBFIComparisonResult,
    NBFIComparisonRunner,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_result(
    *,
    kappa: Decimal = Decimal("1"),
    concentration: Decimal = Decimal("1"),
    mu: Decimal = Decimal("0"),
    monotonicity: Decimal = Decimal("0"),
    seed: int = 42,
    outside_mid_ratio: Decimal = Decimal("0.90"),
    delta_idle: Decimal | None = Decimal("0.5"),
    phi_idle: Decimal | None = Decimal("0.8"),
    idle_run_id: str = "idle_1",
    idle_status: str = "completed",
    delta_lend: Decimal | None = Decimal("0.3"),
    phi_lend: Decimal | None = Decimal("0.9"),
    lend_run_id: str = "lend_1",
    lend_status: str = "completed",
    **kwargs,
) -> NBFIComparisonResult:
    """Build an NBFIComparisonResult with sensible defaults."""
    return NBFIComparisonResult(
        kappa=kappa,
        concentration=concentration,
        mu=mu,
        monotonicity=monotonicity,
        seed=seed,
        outside_mid_ratio=outside_mid_ratio,
        delta_idle=delta_idle,
        phi_idle=phi_idle,
        idle_run_id=idle_run_id,
        idle_status=idle_status,
        delta_lend=delta_lend,
        phi_lend=phi_lend,
        lend_run_id=lend_run_id,
        lend_status=lend_status,
        **kwargs,
    )


# =============================================================================
# 1. NBFIComparisonResult construction
# =============================================================================


class TestNBFIComparisonResultConstruction:
    """Tests for NBFIComparisonResult dataclass construction."""

    def test_required_fields(self):
        """Result can be constructed with only required fields."""
        result = _make_result()
        assert result.kappa == Decimal("1")
        assert result.concentration == Decimal("1")
        assert result.mu == Decimal("0")
        assert result.monotonicity == Decimal("0")
        assert result.seed == 42
        assert result.outside_mid_ratio == Decimal("0.90")
        assert result.delta_idle == Decimal("0.5")
        assert result.phi_idle == Decimal("0.8")
        assert result.idle_run_id == "idle_1"
        assert result.idle_status == "completed"
        assert result.delta_lend == Decimal("0.3")
        assert result.phi_lend == Decimal("0.9")
        assert result.lend_run_id == "lend_1"
        assert result.lend_status == "completed"

    def test_default_optional_fields(self):
        """Optional fields have expected defaults (zero or None)."""
        result = _make_result()
        assert result.n_defaults_idle == 0
        assert result.cascade_fraction_idle is None
        assert result.idle_modal_call_id is None
        assert result.n_defaults_lend == 0
        assert result.cascade_fraction_lend is None
        assert result.lend_modal_call_id is None
        # Loss metric defaults
        assert result.total_loss_idle == 0
        assert result.total_loss_lend == 0
        assert result.total_loss_pct_idle is None
        assert result.total_loss_pct_lend is None
        assert result.intermediary_loss_idle == 0.0
        assert result.intermediary_loss_lend == 0.0
        assert result.intermediary_loss_pct_idle is None
        assert result.intermediary_loss_pct_lend is None
        assert result.system_loss_idle == 0.0
        assert result.system_loss_lend == 0.0
        assert result.system_loss_pct_idle is None
        assert result.system_loss_pct_lend is None


# =============================================================================
# 2. NBFIComparisonResult computed properties
# =============================================================================


class TestNBFIComparisonResultProperties:
    """Tests for NBFIComparisonResult computed properties."""

    def test_lending_effect_positive(self):
        """lending_effect = delta_idle - delta_lend (positive means lending helped)."""
        result = _make_result(delta_idle=Decimal("0.5"), delta_lend=Decimal("0.3"))
        assert result.lending_effect == Decimal("0.2")

    def test_lending_effect_zero(self):
        """lending_effect is zero when both deltas are equal."""
        result = _make_result(delta_idle=Decimal("0.4"), delta_lend=Decimal("0.4"))
        assert result.lending_effect == Decimal("0")

    def test_lending_effect_negative(self):
        """lending_effect is negative when lending worsened defaults."""
        result = _make_result(delta_idle=Decimal("0.2"), delta_lend=Decimal("0.5"))
        assert result.lending_effect == Decimal("-0.3")

    def test_lending_effect_none_when_idle_missing(self):
        """lending_effect is None when delta_idle is None."""
        result = _make_result(delta_idle=None)
        assert result.lending_effect is None

    def test_lending_effect_none_when_lend_missing(self):
        """lending_effect is None when delta_lend is None."""
        result = _make_result(delta_lend=None)
        assert result.lending_effect is None

    def test_lending_relief_ratio(self):
        """lending_relief_ratio = effect / delta_idle."""
        result = _make_result(delta_idle=Decimal("0.5"), delta_lend=Decimal("0.25"))
        assert result.lending_relief_ratio == Decimal("0.5")

    def test_lending_relief_ratio_zero_idle_delta(self):
        """lending_relief_ratio is 0 when delta_idle is 0 (no defaults to reduce)."""
        result = _make_result(delta_idle=Decimal("0"), delta_lend=Decimal("0"))
        assert result.lending_relief_ratio == Decimal("0")

    def test_lending_relief_ratio_none_when_missing(self):
        """lending_relief_ratio is None when either delta is missing."""
        result = _make_result(delta_idle=None, delta_lend=Decimal("0.3"))
        assert result.lending_relief_ratio is None

    def test_system_loss_lending_effect(self):
        """system_loss_lending_effect = system_loss_pct_idle - system_loss_pct_lend."""
        result = _make_result(
            system_loss_pct_idle=0.10,
            system_loss_pct_lend=0.06,
        )
        assert result.system_loss_lending_effect == pytest.approx(0.04)

    def test_system_loss_lending_effect_none(self):
        """system_loss_lending_effect is None when pcts are missing."""
        result = _make_result()  # pcts default to None
        assert result.system_loss_lending_effect is None


# =============================================================================
# 3. NBFIComparisonConfig default values
# =============================================================================


class TestNBFIComparisonConfigDefaults:
    """Tests for NBFIComparisonConfig default values."""

    def test_ring_defaults(self):
        """Ring parameters have expected defaults."""
        config = NBFIComparisonConfig()
        assert config.n_agents == 100
        assert config.maturity_days == 10
        assert config.Q_total == Decimal("10000")
        assert config.liquidity_mode == "uniform"
        assert config.base_seed == 42
        assert config.n_replicates == 1
        assert config.default_handling == "expel-agent"
        assert config.quiet is True
        assert config.rollover_enabled is True

    def test_grid_defaults(self):
        """Grid parameters have expected defaults."""
        config = NBFIComparisonConfig()
        assert config.kappas == [
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("1"),
            Decimal("2"),
            Decimal("4"),
        ]
        assert config.concentrations == [Decimal("1")]
        assert config.mus == [Decimal("0")]
        assert config.monotonicities == [Decimal("0")]
        assert config.outside_mid_ratios == [Decimal("0.90")]

    def test_nbfi_lender_defaults(self):
        """NBFI lender parameters have expected defaults."""
        config = NBFIComparisonConfig()
        assert config.lender_share == Decimal("0.10")
        assert config.lender_base_rate == Decimal("0.05")
        assert config.lender_risk_premium_scale == Decimal("0.20")
        assert config.lender_max_single_exposure == Decimal("0.15")
        assert config.lender_max_total_exposure == Decimal("0.80")
        assert config.lender_maturity_days == 2
        assert config.lender_horizon == 3

    def test_risk_assessment_defaults(self):
        """Risk assessment defaults are correct."""
        config = NBFIComparisonConfig()
        assert config.risk_assessment_enabled is True
        assert config.risk_assessment_config["base_risk_premium"] == "0.02"
        assert config.risk_assessment_config["urgency_sensitivity"] == "0.30"
        assert config.risk_assessment_config["buy_premium_multiplier"] == "1.0"
        assert config.risk_assessment_config["lookback_window"] == 5

    def test_trader_defaults(self):
        """Trader decision parameter defaults are correct."""
        config = NBFIComparisonConfig()
        assert config.risk_aversion == Decimal("0")
        assert config.planning_horizon == 10
        assert config.aggressiveness == Decimal("1.0")
        assert config.default_observability == Decimal("1.0")
        assert config.trading_motive == "liquidity_then_earning"

    def test_vbt_defaults(self):
        """VBT/dealer defaults are correct."""
        config = NBFIComparisonConfig()
        assert config.alpha_vbt == Decimal("0")
        assert config.alpha_trader == Decimal("0")
        assert config.vbt_mid_sensitivity == Decimal("1.0")
        assert config.vbt_spread_sensitivity == Decimal("0.0")
        assert config.vbt_share_per_bucket == Decimal("0.20")
        assert config.dealer_share_per_bucket == Decimal("0.05")

    def test_other_defaults(self):
        """Other defaults (spread_scale, trading_rounds, bank params, etc.)."""
        config = NBFIComparisonConfig()
        assert config.spread_scale == Decimal("1.0")
        assert config.trading_rounds == 100
        assert config.detailed_logging is False
        assert config.face_value == Decimal("20")
        assert config.n_banks == 3
        assert config.reserve_multiplier == 10.0


# =============================================================================
# 4. NBFIComparisonConfig with custom NBFI lender values
# =============================================================================


class TestNBFIComparisonConfigCustom:
    """Tests for NBFIComparisonConfig with custom values."""

    def test_custom_grid(self):
        """Config accepts custom grid parameters."""
        config = NBFIComparisonConfig(
            kappas=[Decimal("0.1"), Decimal("0.5")],
            concentrations=[Decimal("2"), Decimal("3")],
            mus=[Decimal("0.5")],
            outside_mid_ratios=[Decimal("0.80"), Decimal("0.95")],
        )
        assert config.kappas == [Decimal("0.1"), Decimal("0.5")]
        assert config.concentrations == [Decimal("2"), Decimal("3")]
        assert len(config.outside_mid_ratios) == 2

    def test_custom_nbfi_lender_params(self):
        """Config accepts custom NBFI lender parameters."""
        config = NBFIComparisonConfig(
            lender_share=Decimal("0.20"),
            lender_base_rate=Decimal("0.08"),
            lender_risk_premium_scale=Decimal("0.30"),
            lender_max_single_exposure=Decimal("0.25"),
            lender_max_total_exposure=Decimal("0.90"),
            lender_maturity_days=5,
            lender_horizon=7,
        )
        assert config.lender_share == Decimal("0.20")
        assert config.lender_base_rate == Decimal("0.08")
        assert config.lender_risk_premium_scale == Decimal("0.30")
        assert config.lender_max_single_exposure == Decimal("0.25")
        assert config.lender_max_total_exposure == Decimal("0.90")
        assert config.lender_maturity_days == 5
        assert config.lender_horizon == 7

    def test_custom_bank_params(self):
        """Config accepts custom bank parameters."""
        config = NBFIComparisonConfig(
            n_banks=5,
            reserve_multiplier=15.0,
        )
        assert config.n_banks == 5
        assert config.reserve_multiplier == 15.0


# =============================================================================
# 5. NBFIComparisonConfig validation
# =============================================================================


class TestNBFIComparisonConfigValidation:
    """Tests for NBFIComparisonConfig validation rules."""

    def test_n_replicates_must_be_ge_1(self):
        """n_replicates must be >= 1."""
        with pytest.raises(ValidationError):
            NBFIComparisonConfig(n_replicates=0)

    def test_trading_rounds_must_be_ge_1(self):
        """trading_rounds must be >= 1."""
        with pytest.raises(ValidationError):
            NBFIComparisonConfig(trading_rounds=0)


# =============================================================================
# 6. NBFIComparisonRunner construction
# =============================================================================


class TestNBFIComparisonRunnerConstruction:
    """Tests for NBFIComparisonRunner initialization."""

    def test_creates_output_directories(self, tmp_path: Path):
        """Runner creates nbfi_idle, nbfi_lend, and aggregate directories."""
        config = NBFIComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        assert (tmp_path / "nbfi_idle").exists()
        assert (tmp_path / "nbfi_lend").exists()
        assert (tmp_path / "aggregate").exists()

    def test_stores_config(self, tmp_path: Path):
        """Runner stores the config object."""
        config = NBFIComparisonConfig(n_agents=50)
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner.config.n_agents == 50

    def test_default_executor_is_local(self, tmp_path: Path):
        """When no executor is provided, defaults to LocalExecutor."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        from bilancio.runners import LocalExecutor

        assert isinstance(runner.executor, LocalExecutor)

    def test_seed_counter_starts_at_base_seed(self, tmp_path: Path):
        """Seed counter starts from config.base_seed."""
        config = NBFIComparisonConfig(base_seed=100)
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner.seed_counter == 100

    def test_seed_increments(self, tmp_path: Path):
        """_next_seed() returns base_seed and increments."""
        config = NBFIComparisonConfig(base_seed=100)
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._next_seed() == 100
        assert runner._next_seed() == 101
        assert runner._next_seed() == 102

    def test_comparison_results_starts_empty(self, tmp_path: Path):
        """comparison_results list starts empty."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner.comparison_results == []

    def test_job_id_stored(self, tmp_path: Path):
        """Job ID is stored when provided."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(
            config=config, out_dir=tmp_path, job_id="test-job-123", enable_supabase=False
        )
        assert runner.job_id == "test-job-123"


# =============================================================================
# 7. _write_comparison_csv
# =============================================================================


class TestWriteComparisonCSV:
    """Tests for NBFIComparisonRunner._write_comparison_csv."""

    def test_writes_header_and_rows(self, tmp_path: Path):
        """_write_comparison_csv writes header and result rows."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            _make_result(
                kappa=Decimal("0.5"),
                delta_idle=Decimal("0.6"),
                delta_lend=Decimal("0.3"),
                seed=42,
            ),
            _make_result(
                kappa=Decimal("1.0"),
                delta_idle=Decimal("0.4"),
                delta_lend=Decimal("0.2"),
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
        assert rows[0]["delta_idle"] == "0.6"
        assert rows[0]["delta_lend"] == "0.3"
        assert rows[0]["lending_effect"] == "0.3"
        assert rows[1]["kappa"] == "1.0"

    def test_csv_has_all_comparison_fields(self, tmp_path: Path):
        """CSV header contains all COMPARISON_FIELDS."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [_make_result()]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames

        assert header is not None
        for field in NBFIComparisonRunner.COMPARISON_FIELDS:
            assert field in header, f"Missing field: {field}"

    def test_csv_handles_none_values(self, tmp_path: Path):
        """CSV writes empty strings for None values."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [
            _make_result(delta_idle=None, delta_lend=None)
        ]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["delta_idle"] == ""
        assert rows[0]["delta_lend"] == ""
        assert rows[0]["lending_effect"] == ""

    def test_csv_overwrites_on_rewrite(self, tmp_path: Path):
        """Writing CSV twice overwrites (not appends)."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [_make_result(seed=1)]
        runner._write_comparison_csv()

        runner.comparison_results = [_make_result(seed=2), _make_result(seed=3)]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Should have exactly 2 rows (the second write), not 3
        assert len(rows) == 2
        assert rows[0]["seed"] == "2"
        assert rows[1]["seed"] == "3"


# =============================================================================
# 8. Config serialization round-trip
# =============================================================================


class TestConfigSerialization:
    """Tests for NBFIComparisonConfig serialization round-trip."""

    def test_model_dump_and_validate(self):
        """Config can be serialized and deserialized."""
        original = NBFIComparisonConfig(
            n_agents=50,
            maturity_days=5,
            kappas=[Decimal("0.3"), Decimal("1.0")],
            lender_share=Decimal("0.15"),
            lender_base_rate=Decimal("0.08"),
            lender_risk_premium_scale=Decimal("0.25"),
        )
        dumped = original.model_dump()
        restored = NBFIComparisonConfig.model_validate(dumped)

        assert restored.n_agents == original.n_agents
        assert restored.maturity_days == original.maturity_days
        assert restored.kappas == original.kappas
        assert restored.lender_share == original.lender_share
        assert restored.lender_base_rate == original.lender_base_rate
        assert restored.lender_risk_premium_scale == original.lender_risk_premium_scale

    def test_round_trip_preserves_defaults(self):
        """Round-trip of default config preserves all defaults."""
        original = NBFIComparisonConfig()
        dumped = original.model_dump()
        restored = NBFIComparisonConfig.model_validate(dumped)

        assert restored.n_agents == original.n_agents
        assert restored.base_seed == original.base_seed
        assert restored.kappas == original.kappas
        assert restored.concentrations == original.concentrations
        assert restored.lender_share == original.lender_share
        assert restored.lender_base_rate == original.lender_base_rate
        assert restored.risk_assessment_enabled == original.risk_assessment_enabled
        assert restored.n_banks == original.n_banks
        assert restored.reserve_multiplier == original.reserve_multiplier

    def test_model_dump_includes_all_fields(self):
        """model_dump includes all config fields."""
        config = NBFIComparisonConfig()
        dumped = config.model_dump()
        # Check a representative set of field names
        expected_fields = [
            "n_agents",
            "maturity_days",
            "Q_total",
            "kappas",
            "lender_share",
            "lender_base_rate",
            "lender_risk_premium_scale",
            "lender_max_single_exposure",
            "lender_max_total_exposure",
            "lender_maturity_days",
            "lender_horizon",
            "risk_assessment_enabled",
            "risk_aversion",
            "trading_rounds",
            "n_banks",
            "reserve_multiplier",
        ]
        for field in expected_fields:
            assert field in dumped, f"Missing field in dump: {field}"


# =============================================================================
# 9. NBFI-specific config fields (lender parameters)
# =============================================================================


class TestNBFISpecificConfigFields:
    """Tests for NBFI-specific lender configuration fields."""

    def test_lender_share_customizable(self):
        """lender_share can be set to a custom value."""
        config = NBFIComparisonConfig(lender_share=Decimal("0.25"))
        assert config.lender_share == Decimal("0.25")

    def test_lender_base_rate_customizable(self):
        """lender_base_rate can be set to a custom value."""
        config = NBFIComparisonConfig(lender_base_rate=Decimal("0.10"))
        assert config.lender_base_rate == Decimal("0.10")

    def test_lender_risk_premium_scale_customizable(self):
        """lender_risk_premium_scale can be set to a custom value."""
        config = NBFIComparisonConfig(lender_risk_premium_scale=Decimal("0.50"))
        assert config.lender_risk_premium_scale == Decimal("0.50")

    def test_lender_max_single_exposure_customizable(self):
        """lender_max_single_exposure can be set to a custom value."""
        config = NBFIComparisonConfig(lender_max_single_exposure=Decimal("0.30"))
        assert config.lender_max_single_exposure == Decimal("0.30")

    def test_lender_max_total_exposure_customizable(self):
        """lender_max_total_exposure can be set to a custom value."""
        config = NBFIComparisonConfig(lender_max_total_exposure=Decimal("0.95"))
        assert config.lender_max_total_exposure == Decimal("0.95")

    def test_lender_maturity_days_customizable(self):
        """lender_maturity_days can be set to a custom value."""
        config = NBFIComparisonConfig(lender_maturity_days=7)
        assert config.lender_maturity_days == 7

    def test_lender_horizon_customizable(self):
        """lender_horizon can be set to a custom value."""
        config = NBFIComparisonConfig(lender_horizon=10)
        assert config.lender_horizon == 10


# =============================================================================
# 10. Result with None optional fields
# =============================================================================


class TestResultHandlesNoneValues:
    """Tests for NBFIComparisonResult handling None optional fields."""

    def test_both_deltas_none(self):
        """All computed properties are None when both deltas are None."""
        result = _make_result(delta_idle=None, delta_lend=None)
        assert result.lending_effect is None
        assert result.lending_relief_ratio is None

    def test_all_loss_pcts_none(self):
        """Loss effect properties return None when pcts are None."""
        result = _make_result()
        # system_loss_pct defaults to None
        assert result.system_loss_lending_effect is None

    def test_explicit_none_for_optional_metrics(self):
        """Explicitly setting optional fields to None works."""
        result = _make_result(
            delta_idle=None,
            phi_idle=None,
            delta_lend=None,
            phi_lend=None,
            idle_status="failed",
            lend_status="failed",
        )
        assert result.delta_idle is None
        assert result.phi_idle is None
        assert result.delta_lend is None
        assert result.phi_lend is None
        assert result.lending_effect is None


# =============================================================================
# Additional tests: arm definitions and format_time
# =============================================================================


class TestNBFIComparisonRunnerArms:
    """Tests for arm definition logic."""

    def test_get_enabled_arm_defs_returns_two_arms(self, tmp_path: Path):
        """_get_enabled_arm_defs returns idle and lend arms."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 2
        arm_names = [a[0] for a in arms]
        assert "idle" in arm_names
        assert "lend" in arm_names

    def test_arm_phases(self, tmp_path: Path):
        """Arms use nbfi_idle and nbfi_lend phase names."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        phases = {a[0]: a[1] for a in arms}
        assert phases["idle"] == "nbfi_idle"
        assert phases["lend"] == "nbfi_lend"

    def test_arm_supabase_regimes(self, tmp_path: Path):
        """Arms have correct supabase regime names."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        regimes = {a[0]: a[3] for a in arms}
        assert regimes["idle"] == "nbfi_idle"
        assert regimes["lend"] == "nbfi_lend"


class TestFormatTime:
    """Tests for _format_time helper."""

    def test_seconds(self, tmp_path: Path):
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(45) == "45s"

    def test_minutes(self, tmp_path: Path):
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(150) == "2.5m"

    def test_hours(self, tmp_path: Path):
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(3720) == "1h 2m"


# =============================================================================
# 11. Missing lender arm graceful handling
# =============================================================================


class TestMissingLenderArmHandling:
    """Tests for NBFIComparisonResult when lender arm fails but idle succeeds."""

    def test_lending_effect_none_when_lend_failed(self):
        """lending_effect is None when lend arm failed (delta_lend=None)."""
        result = _make_result(
            delta_idle=Decimal("0.5"),
            delta_lend=None,
            lend_status="failed",
        )
        assert result.lending_effect is None
        assert result.lending_relief_ratio is None

    def test_idle_metrics_still_available_when_lend_failed(self):
        """Idle arm metrics should still be accessible when lend arm failed."""
        result = _make_result(
            delta_idle=Decimal("0.5"),
            phi_idle=Decimal("0.8"),
            delta_lend=None,
            phi_lend=None,
            lend_status="failed",
        )
        assert result.delta_idle == Decimal("0.5")
        assert result.phi_idle == Decimal("0.8")

    def test_csv_handles_failed_lend_arm(self, tmp_path: Path):
        """CSV writes properly when lend arm failed (None values become empty)."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [
            _make_result(
                delta_idle=Decimal("0.5"),
                delta_lend=None,
                phi_lend=None,
                lend_status="failed",
            )
        ]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["delta_idle"] == "0.5"
        assert rows[0]["delta_lend"] == ""
        assert rows[0]["lending_effect"] == ""
        assert rows[0]["lend_status"] == "failed"


# =============================================================================
# 12. Config round-trip comprehensive
# =============================================================================


class TestNBFIConfigRoundTripComprehensive:
    """Comprehensive config serialization round-trip tests for NBFI."""

    def test_round_trip_identity_for_default_config(self):
        """model_dump then model_validate on default config produces identical object."""
        original = NBFIComparisonConfig()
        dumped = original.model_dump()
        restored = NBFIComparisonConfig.model_validate(dumped)
        assert original.model_dump() == restored.model_dump()

    def test_round_trip_with_all_lender_params_customized(self):
        """Round-trip preserves all custom lender parameters."""
        original = NBFIComparisonConfig(
            lender_share=Decimal("0.25"),
            lender_base_rate=Decimal("0.10"),
            lender_risk_premium_scale=Decimal("0.40"),
            lender_max_single_exposure=Decimal("0.30"),
            lender_max_total_exposure=Decimal("0.95"),
            lender_maturity_days=5,
            lender_horizon=10,
            risk_aversion=Decimal("0.7"),
            planning_horizon=5,
            n_banks=7,
            reserve_multiplier=20.0,
        )
        dumped = original.model_dump()
        restored = NBFIComparisonConfig.model_validate(dumped)

        assert restored.lender_share == original.lender_share
        assert restored.lender_base_rate == original.lender_base_rate
        assert restored.lender_risk_premium_scale == original.lender_risk_premium_scale
        assert restored.lender_max_single_exposure == original.lender_max_single_exposure
        assert restored.lender_max_total_exposure == original.lender_max_total_exposure
        assert restored.lender_maturity_days == original.lender_maturity_days
        assert restored.lender_horizon == original.lender_horizon
        assert restored.risk_aversion == original.risk_aversion
        assert restored.planning_horizon == original.planning_horizon
        assert restored.n_banks == original.n_banks
        assert restored.reserve_multiplier == original.reserve_multiplier


# =============================================================================
# 13. Comparison CSV header validation
# =============================================================================


class TestNBFIComparisonCSVHeaders:
    """Verify comparison.csv column headers match COMPARISON_FIELDS."""

    def test_csv_columns_match_comparison_fields_exactly(self, tmp_path: Path):
        """CSV header set should equal COMPARISON_FIELDS set."""
        config = NBFIComparisonConfig()
        runner = NBFIComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        runner.comparison_results = [_make_result()]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames

        assert header is not None
        assert set(header) == set(NBFIComparisonRunner.COMPARISON_FIELDS)
