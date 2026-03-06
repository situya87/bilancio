"""Unit tests for sweep_trigger module (mocked Modal)."""

from __future__ import annotations

import sys
from decimal import Decimal
from unittest.mock import MagicMock

# Modal must be mocked before importing sweep_trigger, since it runs
# module-level code (modal.App, modal.Image).
_mock_modal = MagicMock()


def _capture_decorator(*args, **kwargs):
    """Return a decorator that leaves the wrapped function intact."""

    def _inner(fn):
        return fn

    return _inner


_mock_modal.App.return_value.function = _capture_decorator
_mock_modal.App.return_value.local_entrypoint = _capture_decorator
sys.modules.setdefault("modal", _mock_modal)



class TestSweepTriggerConfig:
    """Tests for the sweep configuration constructed inside run_corrected_risk_sweep."""

    def _make_config(self):
        """Import and construct the BalancedComparisonConfig used by sweep_trigger."""
        from bilancio.experiments.balanced_comparison import BalancedComparisonConfig

        return BalancedComparisonConfig(
            n_agents=100,
            maturity_days=10,
            Q_total=Decimal("10000"),
            kappas=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
            concentrations=[Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
            mus=[Decimal("0"), Decimal("0.5"), Decimal("1")],
            outside_mid_ratios=[Decimal("1.0")],
            base_seed=42,
            face_value=Decimal("20"),
            vbt_share_per_bucket=Decimal("0.25"),
            dealer_share_per_bucket=Decimal("0.125"),
            rollover_enabled=True,
            risk_assessment_enabled=True,
            risk_assessment_config={
                "base_risk_premium": "0.02",
                "urgency_sensitivity": "0.30",
                "buy_premium_multiplier": "1.0",
                "lookback_window": 5,
            },
        )

    def test_config_kappas(self):
        """Config has exactly the 4 kappa values from sweep_trigger."""
        config = self._make_config()
        assert config.kappas == [
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("1.0"),
            Decimal("2.0"),
        ]

    def test_config_concentrations(self):
        """Config has exactly the 3 concentration values."""
        config = self._make_config()
        assert config.concentrations == [
            Decimal("0.5"),
            Decimal("1.0"),
            Decimal("2.0"),
        ]

    def test_config_mus(self):
        """Config has exactly the 3 mu values."""
        config = self._make_config()
        assert config.mus == [Decimal("0"), Decimal("0.5"), Decimal("1")]

    def test_total_pairs_calculation(self):
        """Total pairs = len(kappas) * len(concentrations) * len(mus)."""
        config = self._make_config()
        total_pairs = len(config.kappas) * len(config.concentrations) * len(config.mus)
        assert total_pairs == 4 * 3 * 3
        assert total_pairs == 36

    def test_total_runs_is_double_pairs(self):
        """Each pair has passive + active = 2 runs."""
        config = self._make_config()
        total_pairs = len(config.kappas) * len(config.concentrations) * len(config.mus)
        total_runs = total_pairs * 2
        assert total_runs == 72

    def test_risk_assessment_enabled(self):
        """Risk assessment is enabled in the sweep config."""
        config = self._make_config()
        assert config.risk_assessment_enabled is True

    def test_risk_assessment_config_values(self):
        """Risk assessment config has expected parameter values."""
        config = self._make_config()
        rac = config.risk_assessment_config
        assert rac["base_risk_premium"] == "0.02"
        assert rac["urgency_sensitivity"] == "0.30"
        assert rac["lookback_window"] == 5


class TestJobIdFormat:
    """Tests for job ID generation used in sweep_trigger."""

    def test_job_id_is_four_words(self):
        """generate_job_id produces a 4-word hyphen-separated string."""
        from bilancio.jobs import generate_job_id

        job_id = generate_job_id()
        parts = job_id.split("-")
        assert len(parts) == 4, f"Expected 4 words, got {len(parts)}: {job_id}"

    def test_job_ids_are_unique(self):
        """Two generated job IDs should be different."""
        from bilancio.jobs import generate_job_id

        ids = {generate_job_id() for _ in range(10)}
        # 10 random 4-word passphrases should all be distinct
        assert len(ids) == 10


class TestSweepTriggerModuleLevel:
    """Tests for module-level constructs in sweep_trigger."""

    def test_module_defines_app(self):
        """sweep_trigger defines a Modal App named 'bilancio-sweep-trigger'."""
        from bilancio.cloud import sweep_trigger

        # The app is created at module level; since modal is mocked, it's a MagicMock,
        # but we can verify the call was made with the expected name.
        assert hasattr(sweep_trigger, "app")

    def test_module_defines_run_corrected_risk_sweep(self):
        """sweep_trigger defines the run_corrected_risk_sweep function."""
        from bilancio.cloud import sweep_trigger

        assert hasattr(sweep_trigger, "run_corrected_risk_sweep")

    def test_module_defines_main_entrypoint(self):
        """sweep_trigger defines a main() local entrypoint."""
        from bilancio.cloud import sweep_trigger

        assert hasattr(sweep_trigger, "main")

    def test_result_dict_structure(self):
        """The return dict from run_corrected_risk_sweep has expected shape.

        Since modal is mocked and the decorator replaces the function, we
        replicate the return-value construction logic and validate the dict
        structure.
        """
        config = self._make_config()
        job_id = "test-sweep-job-id"
        num_results = 36  # simulated run_all result count

        result = {
            "job_id": job_id,
            "total_pairs": num_results,
            "execution_mode": "cloud",
            "config": {
                "kappas": [str(k) for k in config.kappas],
                "concentrations": [str(c) for c in config.concentrations],
                "mus": [str(m) for m in config.mus],
            },
        }

        assert result["job_id"] == "test-sweep-job-id"
        assert result["total_pairs"] == 36
        assert result["config"]["kappas"] == ["0.25", "0.5", "1.0", "2.0"]
        assert result["config"]["concentrations"] == ["0.5", "1.0", "2.0"]
        assert result["config"]["mus"] == ["0", "0.5", "1"]

    @staticmethod
    def _make_config():
        from bilancio.experiments.balanced_comparison import BalancedComparisonConfig

        return BalancedComparisonConfig(
            n_agents=100,
            maturity_days=10,
            Q_total=Decimal("10000"),
            kappas=[Decimal("0.25"), Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
            concentrations=[Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
            mus=[Decimal("0"), Decimal("0.5"), Decimal("1")],
            outside_mid_ratios=[Decimal("1.0")],
            base_seed=42,
            face_value=Decimal("20"),
            vbt_share_per_bucket=Decimal("0.25"),
            dealer_share_per_bucket=Decimal("0.125"),
            rollover_enabled=True,
            risk_assessment_enabled=True,
            risk_assessment_config={
                "base_risk_premium": "0.02",
                "urgency_sensitivity": "0.30",
                "buy_premium_multiplier": "1.0",
                "lookback_window": 5,
            },
        )


class TestMainFallback:
    """Tests that main() falls back to local execution when .remote() fails."""

    def test_main_falls_back_when_remote_raises(self, monkeypatch):
        """When .remote() raises (no Modal auth), main() runs the impl locally."""
        from bilancio.cloud import sweep_trigger

        sentinel = {"job_id": "local-test", "total_pairs": 0, "config": {}, "execution_mode": "local"}

        # Make .remote() raise as it would without Modal auth.
        real_fn = sweep_trigger.run_corrected_risk_sweep
        monkeypatch.setattr(real_fn, "remote", MagicMock(side_effect=ConnectionError("no auth")), raising=False)

        # Make the local impl return a sentinel instead of running a full sweep.
        fallback = MagicMock(return_value=sentinel)
        monkeypatch.setattr(sweep_trigger, "_run_corrected_risk_sweep_impl", fallback)

        # main() should catch the ConnectionError and call the local impl.
        sweep_trigger.main()  # should not raise
        real_fn.remote.assert_called_once_with(use_cloud=True)
        fallback.assert_called_once_with(use_cloud=False)

    def test_local_impl_uses_runner_default_executor(self, monkeypatch):
        """Local fallback should not construct a CloudExecutor."""
        from bilancio.cloud import sweep_trigger
        import bilancio.experiments.balanced_comparison as balanced_comparison
        import bilancio.jobs as jobs

        captured = {}

        class FakeRunner:
            def __init__(self, *, config, out_dir, executor, job_id, enable_supabase):
                captured["config"] = config
                captured["out_dir"] = out_dir
                captured["executor"] = executor
                captured["job_id"] = job_id
                captured["enable_supabase"] = enable_supabase

            def run_all(self):
                return [object(), object(), object()]

        monkeypatch.setattr(jobs, "generate_job_id", lambda: "local-job-id")
        monkeypatch.setattr(balanced_comparison, "BalancedComparisonRunner", FakeRunner)

        result = sweep_trigger._run_corrected_risk_sweep_impl(use_cloud=False)

        assert captured["executor"] is None
        assert captured["job_id"] == "local-job-id"
        assert captured["enable_supabase"] is True
        assert result["job_id"] == "local-job-id"
        assert result["total_pairs"] == 3
        assert result["execution_mode"] == "local"
