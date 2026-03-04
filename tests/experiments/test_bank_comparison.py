"""Tests for bilancio.experiments.bank_comparison module.

Tests cover:
- BankComparisonResult dataclass construction and computed properties
- BankComparisonConfig defaults, validation, custom values, serialization
- BankComparisonRunner construction, CSV output, and helper methods
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bilancio.experiments.bank_comparison import (
    BankComparisonConfig,
    BankComparisonResult,
    BankComparisonRunner,
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
) -> BankComparisonResult:
    """Build a BankComparisonResult with sensible defaults."""
    return BankComparisonResult(
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
# 1. BankComparisonResult construction
# =============================================================================


class TestBankComparisonResultConstruction:
    """Tests for BankComparisonResult dataclass construction."""

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
        # CB stress defaults
        assert result.cb_loans_created_idle == 0
        assert result.cb_interest_total_idle == 0
        assert result.cb_loans_outstanding_pre_final_idle == 0
        assert result.bank_defaults_final_idle == 0
        assert result.cb_reserve_destruction_pct_idle == 0.0
        # Banking-specific defaults
        assert result.delta_bank_idle is None
        assert result.deposit_loss_gross_idle == 0
        assert result.deposit_loss_pct_idle is None
        # Loss defaults
        assert result.total_loss_idle == 0
        assert result.total_loss_lend == 0
        assert result.intermediary_loss_idle == 0.0
        assert result.system_loss_idle == 0.0
        assert result.system_loss_pct_idle is None


# =============================================================================
# 2. BankComparisonResult computed properties
# =============================================================================


class TestBankComparisonResultProperties:
    """Tests for BankComparisonResult computed properties."""

    def test_bank_lending_effect_positive(self):
        """bank_lending_effect = delta_idle - delta_lend (positive means lending helped)."""
        result = _make_result(delta_idle=Decimal("0.5"), delta_lend=Decimal("0.3"))
        assert result.bank_lending_effect == Decimal("0.2")

    def test_bank_lending_effect_zero(self):
        """bank_lending_effect is zero when both deltas are equal."""
        result = _make_result(delta_idle=Decimal("0.4"), delta_lend=Decimal("0.4"))
        assert result.bank_lending_effect == Decimal("0")

    def test_bank_lending_effect_negative(self):
        """bank_lending_effect is negative when lending worsened defaults."""
        result = _make_result(delta_idle=Decimal("0.2"), delta_lend=Decimal("0.5"))
        assert result.bank_lending_effect == Decimal("-0.3")

    def test_bank_lending_effect_none_when_idle_missing(self):
        """bank_lending_effect is None when delta_idle is None."""
        result = _make_result(delta_idle=None)
        assert result.bank_lending_effect is None

    def test_bank_lending_effect_none_when_lend_missing(self):
        """bank_lending_effect is None when delta_lend is None."""
        result = _make_result(delta_lend=None)
        assert result.bank_lending_effect is None

    def test_bank_lending_relief_ratio(self):
        """bank_lending_relief_ratio = effect / delta_idle."""
        result = _make_result(delta_idle=Decimal("0.5"), delta_lend=Decimal("0.25"))
        assert result.bank_lending_relief_ratio == Decimal("0.5")

    def test_bank_lending_relief_ratio_zero_idle_delta(self):
        """bank_lending_relief_ratio is 0 when delta_idle is 0 (no defaults to reduce)."""
        result = _make_result(delta_idle=Decimal("0"), delta_lend=Decimal("0"))
        assert result.bank_lending_relief_ratio == Decimal("0")

    def test_bank_lending_relief_ratio_none_when_missing(self):
        """bank_lending_relief_ratio is None when either delta is missing."""
        result = _make_result(delta_idle=None, delta_lend=Decimal("0.3"))
        assert result.bank_lending_relief_ratio is None

    def test_system_loss_bank_lending_effect(self):
        """system_loss_bank_lending_effect = system_loss_pct_idle - system_loss_pct_lend."""
        result = _make_result(
            system_loss_pct_idle=0.10,
            system_loss_pct_lend=0.06,
        )
        assert result.system_loss_bank_lending_effect == pytest.approx(0.04)

    def test_system_loss_bank_lending_effect_none(self):
        """system_loss_bank_lending_effect is None when pcts are missing."""
        result = _make_result()  # pcts default to None
        assert result.system_loss_bank_lending_effect is None

    def test_deposit_loss_effect(self):
        """deposit_loss_effect = deposit_loss_pct_idle - deposit_loss_pct_lend."""
        result = _make_result(
            deposit_loss_pct_idle=0.05,
            deposit_loss_pct_lend=0.08,
        )
        # Negative because lending increased deposit losses
        assert result.deposit_loss_effect == pytest.approx(-0.03)

    def test_deposit_loss_effect_none(self):
        """deposit_loss_effect is None when pcts are missing."""
        result = _make_result()
        assert result.deposit_loss_effect is None


# =============================================================================
# 3. BankComparisonConfig defaults
# =============================================================================


class TestBankComparisonConfigDefaults:
    """Tests for BankComparisonConfig default values."""

    def test_ring_defaults(self):
        """Ring parameters have expected defaults."""
        config = BankComparisonConfig()
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
        config = BankComparisonConfig()
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

    def test_bank_defaults(self):
        """Bank-specific parameters have expected defaults."""
        config = BankComparisonConfig()
        assert config.n_banks == 5
        assert config.reserve_ratio == Decimal("0.50")
        assert config.credit_risk_loading == Decimal("0.5")
        assert config.max_borrower_risk == Decimal("0.4")
        assert config.min_coverage_ratio == Decimal("0")

    def test_cb_defaults(self):
        """Central bank parameters have expected defaults."""
        config = BankComparisonConfig()
        assert config.cb_rate_escalation_slope == Decimal("0.05")
        assert config.cb_max_outstanding_ratio == Decimal("2.0")
        assert config.cb_lending_cutoff_day is None

    def test_risk_assessment_defaults(self):
        """Risk assessment defaults are correct."""
        config = BankComparisonConfig()
        assert config.risk_assessment_enabled is True
        assert config.risk_assessment_config["base_risk_premium"] == "0.02"
        assert config.risk_assessment_config["urgency_sensitivity"] == "0.30"
        assert config.risk_assessment_config["lookback_window"] == 5

    def test_trader_defaults(self):
        """Trader decision parameter defaults are correct."""
        config = BankComparisonConfig()
        assert config.risk_aversion == Decimal("0")
        assert config.planning_horizon == 10
        assert config.aggressiveness == Decimal("1.0")
        assert config.default_observability == Decimal("1.0")
        assert config.trading_motive == "liquidity_then_earning"

    def test_other_defaults(self):
        """Other defaults (spread_scale, trading_rounds, etc.)."""
        config = BankComparisonConfig()
        assert config.spread_scale == Decimal("1.0")
        assert config.trading_rounds == 100
        assert config.detailed_logging is False
        assert config.face_value == Decimal("20")
        assert config.performance == {}


# =============================================================================
# 4. BankComparisonConfig validation
# =============================================================================


class TestBankComparisonConfigValidation:
    """Tests for BankComparisonConfig validation rules."""

    def test_n_replicates_must_be_ge_1(self):
        """n_replicates must be >= 1."""
        with pytest.raises(Exception):
            BankComparisonConfig(n_replicates=0)

    def test_trading_rounds_must_be_ge_1(self):
        """trading_rounds must be >= 1."""
        with pytest.raises(Exception):
            BankComparisonConfig(trading_rounds=0)


# =============================================================================
# 5. BankComparisonConfig with custom values
# =============================================================================


class TestBankComparisonConfigCustom:
    """Tests for BankComparisonConfig with custom values."""

    def test_custom_grid(self):
        """Config accepts custom grid parameters."""
        config = BankComparisonConfig(
            kappas=[Decimal("0.1"), Decimal("0.5")],
            concentrations=[Decimal("2"), Decimal("3")],
            mus=[Decimal("0.5")],
            outside_mid_ratios=[Decimal("0.80"), Decimal("0.95")],
        )
        assert config.kappas == [Decimal("0.1"), Decimal("0.5")]
        assert config.concentrations == [Decimal("2"), Decimal("3")]
        assert len(config.outside_mid_ratios) == 2

    def test_custom_bank_params(self):
        """Config accepts custom bank parameters."""
        config = BankComparisonConfig(
            n_banks=10,
            reserve_ratio=Decimal("0.30"),
            credit_risk_loading=Decimal("1.0"),
            max_borrower_risk=Decimal("0.6"),
        )
        assert config.n_banks == 10
        assert config.reserve_ratio == Decimal("0.30")
        assert config.credit_risk_loading == Decimal("1.0")
        assert config.max_borrower_risk == Decimal("0.6")

    def test_custom_cb_cutoff(self):
        """Config accepts a custom CB lending cutoff day."""
        config = BankComparisonConfig(cb_lending_cutoff_day=7)
        assert config.cb_lending_cutoff_day == 7


# =============================================================================
# 6. _common_runner_kwargs
# =============================================================================


class TestCommonRunnerKwargs:
    """Tests for BankComparisonRunner._common_runner_kwargs output."""

    def test_has_expected_keys(self, tmp_path: Path):
        """_common_runner_kwargs returns dict with all expected keys."""
        config = BankComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        with patch(
            "bilancio.experiments.bank_comparison.BankComparisonRunner._load_existing_results"
        ):
            runner = BankComparisonRunner(
                config=config,
                out_dir=tmp_path,
                enable_supabase=False,
            )

        kwargs = runner._common_runner_kwargs(outside_mid_ratio=Decimal("0.90"))

        expected_keys = {
            "n_agents",
            "maturity_days",
            "Q_total",
            "liquidity_mode",
            "liquidity_agent",
            "base_seed",
            "default_handling",
            "dealer_enabled",
            "dealer_config",
            "balanced_mode",
            "face_value",
            "outside_mid_ratio",
            "vbt_share_per_bucket",
            "dealer_share_per_bucket",
            "rollover_enabled",
            "detailed_dealer_logging",
            "executor",
            "quiet",
            "risk_assessment_enabled",
            "risk_assessment_config",
            "risk_aversion",
            "planning_horizon",
            "aggressiveness",
            "default_observability",
            "trading_motive",
            "lender_mode",
            "n_banks",
            "reserve_ratio",
            "credit_risk_loading",
            "max_borrower_risk",
            "min_coverage_ratio",
            "cb_rate_escalation_slope",
            "cb_max_outstanding_ratio",
            "spread_scale",
            "cb_lending_cutoff_day",
            "trading_rounds",
            "alpha_vbt",
            "alpha_trader",
            "vbt_mid_sensitivity",
            "vbt_spread_sensitivity",
            "performance",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_dealer_disabled(self, tmp_path: Path):
        """Bank comparison disables dealer and VBT."""
        config = BankComparisonConfig()
        with patch(
            "bilancio.experiments.bank_comparison.BankComparisonRunner._load_existing_results"
        ):
            runner = BankComparisonRunner(
                config=config,
                out_dir=tmp_path,
                enable_supabase=False,
            )

        kwargs = runner._common_runner_kwargs(outside_mid_ratio=Decimal("0.90"))
        assert kwargs["dealer_enabled"] is False
        assert kwargs["vbt_share_per_bucket"] == Decimal("0")
        assert kwargs["dealer_share_per_bucket"] == Decimal("0")
        assert kwargs["lender_mode"] is False

    def test_cb_cutoff_defaults_to_maturity_days(self, tmp_path: Path):
        """When cb_lending_cutoff_day is None, defaults to maturity_days."""
        config = BankComparisonConfig(maturity_days=15, cb_lending_cutoff_day=None)
        with patch(
            "bilancio.experiments.bank_comparison.BankComparisonRunner._load_existing_results"
        ):
            runner = BankComparisonRunner(
                config=config,
                out_dir=tmp_path,
                enable_supabase=False,
            )

        kwargs = runner._common_runner_kwargs(outside_mid_ratio=Decimal("0.90"))
        assert kwargs["cb_lending_cutoff_day"] == 15

    def test_cb_cutoff_explicit(self, tmp_path: Path):
        """When cb_lending_cutoff_day is set, uses that value."""
        config = BankComparisonConfig(maturity_days=15, cb_lending_cutoff_day=7)
        with patch(
            "bilancio.experiments.bank_comparison.BankComparisonRunner._load_existing_results"
        ):
            runner = BankComparisonRunner(
                config=config,
                out_dir=tmp_path,
                enable_supabase=False,
            )

        kwargs = runner._common_runner_kwargs(outside_mid_ratio=Decimal("0.90"))
        assert kwargs["cb_lending_cutoff_day"] == 7


# =============================================================================
# 7. BankComparisonRunner construction
# =============================================================================


class TestBankComparisonRunnerConstruction:
    """Tests for BankComparisonRunner initialization."""

    def test_creates_output_directories(self, tmp_path: Path):
        """Runner creates bank_idle, bank_lend, and aggregate directories."""
        config = BankComparisonConfig(
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
        )
        BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        assert (tmp_path / "bank_idle").exists()
        assert (tmp_path / "bank_lend").exists()
        assert (tmp_path / "aggregate").exists()

    def test_stores_config(self, tmp_path: Path):
        """Runner stores the config object."""
        config = BankComparisonConfig(n_agents=50)
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner.config.n_agents == 50

    def test_default_executor_is_local(self, tmp_path: Path):
        """When no executor is provided, defaults to LocalExecutor."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        from bilancio.runners import LocalExecutor

        assert isinstance(runner.executor, LocalExecutor)

    def test_seed_counter_starts_at_base_seed(self, tmp_path: Path):
        """Seed counter starts from config.base_seed."""
        config = BankComparisonConfig(base_seed=100)
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner.seed_counter == 100

    def test_seed_increments(self, tmp_path: Path):
        """_next_seed() returns base_seed and increments."""
        config = BankComparisonConfig(base_seed=100)
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._next_seed() == 100
        assert runner._next_seed() == 101
        assert runner._next_seed() == 102

    def test_comparison_results_starts_empty(self, tmp_path: Path):
        """comparison_results list starts empty."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner.comparison_results == []

    def test_job_id_stored(self, tmp_path: Path):
        """Job ID is stored when provided."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(
            config=config, out_dir=tmp_path, job_id="test-job-123", enable_supabase=False
        )
        assert runner.job_id == "test-job-123"


# =============================================================================
# 8. _write_comparison_csv
# =============================================================================


class TestWriteComparisonCSV:
    """Tests for BankComparisonRunner._write_comparison_csv."""

    def test_writes_header_and_rows(self, tmp_path: Path):
        """_write_comparison_csv writes header and result rows."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

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
        assert rows[0]["bank_lending_effect"] == "0.3"
        assert rows[1]["kappa"] == "1.0"

    def test_csv_has_all_comparison_fields(self, tmp_path: Path):
        """CSV header contains all COMPARISON_FIELDS."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        runner.comparison_results = [_make_result()]
        runner._write_comparison_csv()

        csv_path = tmp_path / "aggregate" / "comparison.csv"
        with csv_path.open("r") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames

        assert header is not None
        for field in BankComparisonRunner.COMPARISON_FIELDS:
            assert field in header, f"Missing field: {field}"

    def test_csv_handles_none_values(self, tmp_path: Path):
        """CSV writes empty strings for None values."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

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
        assert rows[0]["bank_lending_effect"] == ""

    def test_csv_overwrites_on_rewrite(self, tmp_path: Path):
        """Writing CSV twice overwrites (not appends)."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

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
# 9. _get_performance
# =============================================================================


class TestGetPerformance:
    """Tests for BankComparisonRunner._get_performance."""

    def test_returns_none_when_empty(self, tmp_path: Path):
        """_get_performance returns None when config.performance is empty."""
        config = BankComparisonConfig(performance={})
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._get_performance() is None

    def test_returns_performance_config_when_populated(self, tmp_path: Path):
        """_get_performance returns PerformanceConfig when flags are set."""
        config = BankComparisonConfig(performance={"fast_atomic": True})
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        from bilancio.core.performance import PerformanceConfig

        perf = runner._get_performance()
        assert isinstance(perf, PerformanceConfig)
        assert perf.fast_atomic is True


# =============================================================================
# 10. Result with None optional fields
# =============================================================================


class TestResultHandlesNoneValues:
    """Tests for BankComparisonResult handling None optional fields."""

    def test_both_deltas_none(self):
        """All computed properties are None when both deltas are None."""
        result = _make_result(delta_idle=None, delta_lend=None)
        assert result.bank_lending_effect is None
        assert result.bank_lending_relief_ratio is None

    def test_all_loss_pcts_none(self):
        """Loss effect properties return None when pcts are None."""
        result = _make_result()
        # deposit_loss_pct and system_loss_pct default to None
        assert result.deposit_loss_effect is None
        assert result.system_loss_bank_lending_effect is None

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
        assert result.bank_lending_effect is None


# =============================================================================
# 11. Config serialization round-trip
# =============================================================================


class TestConfigSerialization:
    """Tests for BankComparisonConfig serialization round-trip."""

    def test_model_dump_and_validate(self):
        """Config can be serialized and deserialized."""
        original = BankComparisonConfig(
            n_agents=50,
            maturity_days=5,
            kappas=[Decimal("0.3"), Decimal("1.0")],
            n_banks=3,
            reserve_ratio=Decimal("0.40"),
            credit_risk_loading=Decimal("0.8"),
        )
        dumped = original.model_dump()
        restored = BankComparisonConfig.model_validate(dumped)

        assert restored.n_agents == original.n_agents
        assert restored.maturity_days == original.maturity_days
        assert restored.kappas == original.kappas
        assert restored.n_banks == original.n_banks
        assert restored.reserve_ratio == original.reserve_ratio
        assert restored.credit_risk_loading == original.credit_risk_loading

    def test_round_trip_preserves_defaults(self):
        """Round-trip of default config preserves all defaults."""
        original = BankComparisonConfig()
        dumped = original.model_dump()
        restored = BankComparisonConfig.model_validate(dumped)

        assert restored.n_agents == original.n_agents
        assert restored.base_seed == original.base_seed
        assert restored.kappas == original.kappas
        assert restored.concentrations == original.concentrations
        assert restored.n_banks == original.n_banks
        assert restored.cb_rate_escalation_slope == original.cb_rate_escalation_slope
        assert restored.risk_assessment_enabled == original.risk_assessment_enabled
        assert restored.performance == original.performance

    def test_model_dump_includes_all_fields(self):
        """model_dump includes all config fields."""
        config = BankComparisonConfig()
        dumped = config.model_dump()
        # Check a representative set of field names
        expected_fields = [
            "n_agents",
            "maturity_days",
            "Q_total",
            "kappas",
            "n_banks",
            "reserve_ratio",
            "credit_risk_loading",
            "max_borrower_risk",
            "cb_rate_escalation_slope",
            "risk_assessment_enabled",
            "risk_aversion",
            "trading_rounds",
            "performance",
        ]
        for field in expected_fields:
            assert field in dumped, f"Missing field in dump: {field}"


# =============================================================================
# 12. Bank-specific config fields
# =============================================================================


class TestBankSpecificConfigFields:
    """Tests for bank-specific configuration fields."""

    def test_n_banks_customizable(self):
        """n_banks can be set to a custom value."""
        config = BankComparisonConfig(n_banks=10)
        assert config.n_banks == 10

    def test_reserve_ratio_customizable(self):
        """reserve_ratio can be set to a custom value."""
        config = BankComparisonConfig(reserve_ratio=Decimal("0.25"))
        assert config.reserve_ratio == Decimal("0.25")

    def test_credit_risk_loading_customizable(self):
        """credit_risk_loading can be set to a custom value."""
        config = BankComparisonConfig(credit_risk_loading=Decimal("1.0"))
        assert config.credit_risk_loading == Decimal("1.0")

    def test_max_borrower_risk_customizable(self):
        """max_borrower_risk can be set to a custom value."""
        config = BankComparisonConfig(max_borrower_risk=Decimal("0.8"))
        assert config.max_borrower_risk == Decimal("0.8")

    def test_min_coverage_ratio_customizable(self):
        """min_coverage_ratio can be set to a custom value."""
        config = BankComparisonConfig(min_coverage_ratio=Decimal("1.5"))
        assert config.min_coverage_ratio == Decimal("1.5")

    def test_cb_parameters_wired_to_common_kwargs(self, tmp_path: Path):
        """CB parameters appear in _common_runner_kwargs."""
        config = BankComparisonConfig(
            cb_rate_escalation_slope=Decimal("0.10"),
            cb_max_outstanding_ratio=Decimal("3.0"),
        )
        with patch(
            "bilancio.experiments.bank_comparison.BankComparisonRunner._load_existing_results"
        ):
            runner = BankComparisonRunner(
                config=config, out_dir=tmp_path, enable_supabase=False
            )

        kwargs = runner._common_runner_kwargs(outside_mid_ratio=Decimal("0.90"))
        assert kwargs["cb_rate_escalation_slope"] == Decimal("0.10")
        assert kwargs["cb_max_outstanding_ratio"] == Decimal("3.0")

    def test_bank_params_wired_to_common_kwargs(self, tmp_path: Path):
        """Bank parameters appear in _common_runner_kwargs."""
        config = BankComparisonConfig(
            n_banks=8,
            reserve_ratio=Decimal("0.60"),
            credit_risk_loading=Decimal("0.75"),
            max_borrower_risk=Decimal("0.5"),
            min_coverage_ratio=Decimal("1.2"),
        )
        with patch(
            "bilancio.experiments.bank_comparison.BankComparisonRunner._load_existing_results"
        ):
            runner = BankComparisonRunner(
                config=config, out_dir=tmp_path, enable_supabase=False
            )

        kwargs = runner._common_runner_kwargs(outside_mid_ratio=Decimal("0.90"))
        assert kwargs["n_banks"] == 8
        assert kwargs["reserve_ratio"] == Decimal("0.60")
        assert kwargs["credit_risk_loading"] == Decimal("0.75")
        assert kwargs["max_borrower_risk"] == Decimal("0.5")
        assert kwargs["min_coverage_ratio"] == Decimal("1.2")


# =============================================================================
# Additional edge-case tests
# =============================================================================


class TestBankComparisonRunnerArms:
    """Tests for arm definition logic."""

    def test_get_enabled_arm_defs_returns_two_arms(self, tmp_path: Path):
        """_get_enabled_arm_defs returns idle and lend arms."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        assert len(arms) == 2
        arm_names = [a[0] for a in arms]
        assert "idle" in arm_names
        assert "lend" in arm_names

    def test_arm_phases(self, tmp_path: Path):
        """Arms use bank_idle and bank_lend phase names."""
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)

        arms = runner._get_enabled_arm_defs()
        phases = {a[0]: a[1] for a in arms}
        assert phases["idle"] == "bank_idle"
        assert phases["lend"] == "bank_lend"


class TestFormatTime:
    """Tests for _format_time helper."""

    def test_seconds(self, tmp_path: Path):
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(45) == "45s"

    def test_minutes(self, tmp_path: Path):
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(150) == "2.5m"

    def test_hours(self, tmp_path: Path):
        config = BankComparisonConfig()
        runner = BankComparisonRunner(config=config, out_dir=tmp_path, enable_supabase=False)
        assert runner._format_time(3720) == "1h 2m"
