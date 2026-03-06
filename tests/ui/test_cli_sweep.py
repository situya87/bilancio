"""Tests for CLI sweep commands.

Tests parameter parsing, help text, and command invocation via Click's CliRunner.
Mocks actual runner classes to avoid executing real simulations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bilancio.ui.cli import cli


class TestSweepHelp:
    """Test help text for sweep commands."""

    def test_sweep_help(self):
        """sweep --help shows the sweep group help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "--help"])
        assert result.exit_code == 0
        assert "Experiment sweeps" in result.output

    def test_sweep_ring_help(self):
        """sweep ring --help shows ring-specific options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "ring", "--help"])
        assert result.exit_code == 0
        assert "--kappas" in result.output
        assert "--concentrations" in result.output
        assert "--n-agents" in result.output
        assert "--maturity-days" in result.output
        assert "--cloud" in result.output
        assert "--out-dir" in result.output
        assert "--lhs" in result.output
        assert "--frontier" in result.output

    def test_sweep_balanced_help(self):
        """sweep balanced --help shows balanced-specific options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "balanced", "--help"])
        assert result.exit_code == 0
        assert "--kappas" in result.output
        assert "--out-dir" in result.output
        assert "--enable-lender" in result.output
        assert "--risk-premium" in result.output
        assert "--trading-motive" in result.output
        assert "--rollover" in result.output
        assert "--enable-bank-passive" in result.output
        assert "--post-analysis" in result.output

    def test_sweep_bank_help(self):
        """sweep bank --help shows bank-specific options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "bank", "--help"])
        assert result.exit_code == 0
        assert "--n-banks" in result.output
        assert "--reserve-ratio" in result.output
        assert "--credit-risk-loading" in result.output
        assert "--max-borrower-risk" in result.output
        assert "--cb-rate-escalation-slope" in result.output
        assert "--cb-max-outstanding-ratio" in result.output

    def test_sweep_nbfi_help(self):
        """sweep nbfi --help shows NBFI-specific options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "nbfi", "--help"])
        assert result.exit_code == 0
        assert "--nbfi-share" in result.output
        assert "--n-agents" in result.output
        assert "--kappas" in result.output
        assert "--out-dir" in result.output


class TestSweepList:
    """Test sweep list command."""

    def test_sweep_list_shows_plugins(self):
        """sweep list shows available scenario plugins."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "list"])
        # Should succeed even if no plugins are registered
        assert result.exit_code == 0


class TestSweepRingExecution:
    """Test sweep ring command invocation with mocked runners."""

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ring-job-id")
    def test_sweep_ring_minimal(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring with minimal args creates a runner and executes."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_runner_cls.assert_called_once()
        mock_runner.run_grid.assert_called_once()

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ring-params")
    def test_sweep_ring_parameter_parsing(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring parses --kappas, --concentrations, --mus correctly."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--kappas",
                "0.3,1.0",
                "--concentrations",
                "0.5,2.0",
                "--mus",
                "0,0.5",
                "--n-agents",
                "20",
                "--maturity-days",
                "5",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        # Verify the runner was constructed with the parsed parameters
        call_kwargs = mock_runner_cls.call_args
        assert call_kwargs[1]["n_agents"] == 20
        assert call_kwargs[1]["maturity_days"] == 5

        # Verify run_grid was called (grid runs by default)
        mock_runner.run_grid.assert_called_once()
        grid_call_args = mock_runner.run_grid.call_args[0]
        # grid_kappas should be [Decimal('0.3'), Decimal('1.0')]
        assert len(grid_call_args[0]) == 2  # kappas
        assert len(grid_call_args[1]) == 2  # concentrations
        assert len(grid_call_args[2]) == 2  # mus


class TestSweepBalancedExecution:
    """Test sweep balanced command invocation with mocked runners."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-job")
    def test_sweep_balanced_minimal(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced with --out-dir creates a runner and executes."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_runner_init.assert_called_once()
        mock_run_all.assert_called_once()

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-lender")
    def test_sweep_balanced_with_enable_lender(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced --enable-lender passes lender flag to config."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--enable-lender",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        # Verify BalancedComparisonRunner.__init__ received a config with enable_lender=True
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.enable_lender is True


class TestSweepBankExecution:
    """Test sweep bank command invocation with mocked runners."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-bank-job")
    def test_sweep_bank_minimal(
        self,
        mock_gen_id,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep bank with --out-dir creates a runner and executes."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "bank",
                "--out-dir",
                str(tmp_path / "bank_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_runner_init.assert_called_once()
        mock_run_all.assert_called_once()
        assert "Job ID: test-bank-job" in result.output


class TestSweepNBFIExecution:
    """Test sweep nbfi command invocation with mocked runners."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.nbfi_comparison.NBFIComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.nbfi_comparison.NBFIComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-nbfi-job")
    def test_sweep_nbfi_minimal(
        self,
        mock_gen_id,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep nbfi with --out-dir creates a runner and executes."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "nbfi",
                "--out-dir",
                str(tmp_path / "nbfi_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_runner_init.assert_called_once()
        mock_run_all.assert_called_once()
        assert "Job ID: test-nbfi-job" in result.output


class TestSweepPerformanceFlags:
    """Test performance flag parsing for sweep commands."""

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-perf-flags")
    def test_ring_performance_flags(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring passes performance flags through to runner."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--kappas",
                "1.0",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--fast-atomic",
                "--prune-ineligible",
                "--perf-preset",
                "fast",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        # The runner should have been constructed with a performance config
        call_kwargs = mock_runner_cls.call_args[1]
        perf = call_kwargs.get("performance")
        # Performance object should exist since we passed flags
        assert perf is not None

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-perf")
    def test_balanced_performance_flags(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced passes performance flags through to config."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--fast-atomic",
                "--preview-buy",
                "--dealer-backend",
                "native",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        # The config should have a performance dict with our flags
        assert config_arg.performance is not None
        perf = config_arg.performance
        assert perf.get("fast_atomic") is True or getattr(perf, "fast_atomic", None) is True


class TestSweepBalancedMissingRequired:
    """Test that required arguments are enforced."""

    def test_balanced_missing_out_dir(self):
        """sweep balanced without --out-dir fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "balanced"])
        assert result.exit_code != 0
        assert "out-dir" in result.output.lower() or "required" in result.output.lower()

    def test_bank_missing_out_dir(self):
        """sweep bank without --out-dir fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "bank"])
        assert result.exit_code != 0
        assert "out-dir" in result.output.lower() or "required" in result.output.lower()

    def test_nbfi_missing_out_dir(self):
        """sweep nbfi without --out-dir fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "nbfi"])
        assert result.exit_code != 0
        assert "out-dir" in result.output.lower() or "required" in result.output.lower()


# ── _offer_post_sweep_analysis tests ─────────────────────────────────────────


class TestOfferPostSweepAnalysis:
    """Test _offer_post_sweep_analysis helper function."""

    def test_no_csv_local_skips(self, tmp_path):
        """When comparison.csv is missing locally, prints skip message."""
        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        # No aggregate/comparison.csv exists
        _offer_post_sweep_analysis(tmp_path, "dealer", None, cloud=False)
        # No exception; function returns silently

    def test_no_csv_cloud_returns_silently(self, tmp_path):
        """When cloud=True and no csv, returns without output."""
        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        _offer_post_sweep_analysis(tmp_path, "bank", None, cloud=True)

    def test_post_analysis_none_skips(self, tmp_path):
        """post_analysis='none' skips analysis."""
        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text("kappa,delta\n0.5,0.1\n")

        _offer_post_sweep_analysis(tmp_path, "dealer", "none", cloud=False)

    @patch("bilancio.analysis.post_sweep.run_post_sweep_analysis")
    def test_post_analysis_all_runs_all(self, mock_run, tmp_path):
        """post_analysis='all' runs all four analyses."""
        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text("kappa,delta\n0.5,0.1\n")
        mock_run.return_value = {"drilldowns": tmp_path / "drill.html"}

        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        _offer_post_sweep_analysis(tmp_path, "dealer", "all", cloud=False)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert set(call_kwargs["analyses"]) == {"drilldowns", "deltas", "dynamics", "narrative"}

    @patch("bilancio.analysis.post_sweep.run_post_sweep_analysis")
    def test_post_analysis_comma_list(self, mock_run, tmp_path):
        """post_analysis='drilldowns,deltas' runs specified analyses."""
        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text("kappa,delta\n")
        mock_run.return_value = {}

        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        _offer_post_sweep_analysis(tmp_path, "dealer", "drilldowns,deltas", cloud=False)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert "drilldowns" in call_kwargs["analyses"]
        assert "deltas" in call_kwargs["analyses"]

    @patch("bilancio.analysis.post_sweep.run_post_sweep_analysis")
    def test_post_analysis_unknown_entry_warns(self, mock_run, tmp_path):
        """Unknown analysis names in comma-separated list are filtered out."""
        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text("kappa,delta\n")
        mock_run.return_value = {}

        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        _offer_post_sweep_analysis(tmp_path, "dealer", "drilldowns,bogus", cloud=False)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert "drilldowns" in call_kwargs["analyses"]
        assert "bogus" not in call_kwargs["analyses"]

    @patch("bilancio.analysis.post_sweep.run_post_sweep_analysis")
    def test_post_analysis_empty_results(self, mock_run, tmp_path):
        """When analysis returns empty results, prints appropriate message."""
        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text("kappa,delta\n")
        mock_run.return_value = {}  # empty

        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        _offer_post_sweep_analysis(tmp_path, "dealer", "all", cloud=False)

    @patch("bilancio.analysis.post_sweep.run_post_sweep_analysis")
    def test_post_analysis_exception_handled(self, mock_run, tmp_path):
        """Exceptions from run_post_sweep_analysis are caught gracefully."""
        agg = tmp_path / "aggregate"
        agg.mkdir()
        (agg / "comparison.csv").write_text("kappa,delta\n")
        mock_run.side_effect = ValueError("broken analysis")

        from bilancio.ui.cli.sweep import _offer_post_sweep_analysis

        # Should not raise
        _offer_post_sweep_analysis(tmp_path, "dealer", "all", cloud=False)


# ── sweep ring: LHS and frontier paths ──────────────────────────────────────


class TestSweepRingLHS:
    """Test sweep ring with --lhs flag for Latin Hypercube sampling."""

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ring-lhs")
    def test_sweep_ring_lhs(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring --lhs 5 invokes run_lhs with correct count."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--no-grid",
                "--lhs",
                "5",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_runner.run_lhs.assert_called_once()
        lhs_args = mock_runner.run_lhs.call_args
        assert lhs_args[0][0] == 5  # count

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ring-frontier")
    def test_sweep_ring_frontier(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring --frontier invokes run_frontier."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--no-grid",
                "--frontier",
                "--frontier-low",
                "0.1",
                "--frontier-high",
                "5.0",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_runner.run_frontier.assert_called_once()


class TestSweepRingJobId:
    """Test sweep ring custom job ID."""

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    def test_sweep_ring_custom_job_id(
        self,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring --job-id custom-id uses the provided job ID."""
        mock_manager = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "my-custom-ring-id"
        mock_manager.create_job.return_value = mock_job
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
                "--job-id",
                "my-custom-ring-id",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "my-custom-ring-id" in result.output


class TestSweepRingNoOutDir:
    """Test sweep ring with auto-generated output directory."""

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ring-auto")
    def test_sweep_ring_auto_out_dir(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring without --out-dir auto-generates output directory."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "ring"])

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"


class TestSweepRingErrorHandling:
    """Test sweep ring error handling paths."""

    @patch("bilancio.ui.cli.sweep.aggregate_runs")
    @patch("bilancio.ui.cli.sweep.render_dashboard")
    @patch("bilancio.ui.cli.sweep.RingSweepRunner")
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-ring-fail")
    def test_sweep_ring_job_manager_failure(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_cls,
        mock_render,
        mock_agg,
        tmp_path,
    ):
        """sweep ring continues even when job manager creation fails."""
        mock_create_jm.side_effect = RuntimeError("supabase down")
        mock_runner = MagicMock()
        mock_runner.registry_dir = tmp_path / "registry"
        mock_runner.aggregate_dir = tmp_path / "aggregate"
        (tmp_path / "registry").mkdir()
        (tmp_path / "aggregate").mkdir()
        mock_runner_cls.return_value = mock_runner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "ring",
                "--out-dir",
                str(tmp_path / "ring_out"),
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Warning" in result.output


# ── sweep balanced: additional parameters ─────────────────────────────────────


class TestSweepBalancedAdditionalParams:
    """Test sweep balanced with various option combinations."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-risk")
    def test_sweep_balanced_risk_params(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced passes risk assessment params to config."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--risk-premium",
                "0.05",
                "--risk-urgency",
                "0.20",
                "--alpha-vbt",
                "0.5",
                "--alpha-trader",
                "0.3",
                "--risk-aversion",
                "0.4",
                "--planning-horizon",
                "5",
                "--aggressiveness",
                "0.8",
                "--default-observability",
                "0.7",
                "--trading-motive",
                "liquidity_only",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.trading_motive == "liquidity_only"
        assert config_arg.planning_horizon == 5

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-bank-arms")
    def test_sweep_balanced_bank_arms(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced with --enable-bank-passive passes to config."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--enable-bank-passive",
                "--enable-bank-dealer",
                "--n-banks-for-banking",
                "5",
                "--bank-reserve-multiplier",
                "0.8",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.enable_bank_passive is True
        assert config_arg.enable_bank_dealer is True
        assert config_arg.n_banks_for_banking == 5

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-lender-opts")
    def test_sweep_balanced_lender_options(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced with lender-specific options passes them to config."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--enable-lender",
                "--lender-share",
                "0.15",
                "--lender-min-coverage",
                "0.7",
                "--lender-maturity-matching",
                "--lender-ranking-mode",
                "cascade",
                "--lender-coverage-mode",
                "graduated",
                "--lender-preventive-lending",
                "--lender-prevention-threshold",
                "0.4",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.enable_lender is True
        assert config_arg.lender_maturity_matching is True
        assert config_arg.lender_ranking_mode == "cascade"
        assert config_arg.lender_coverage_mode == "graduated"
        assert config_arg.lender_preventive_lending is True

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-custom-id")
    def test_sweep_balanced_custom_job_id(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced --job-id custom-id uses provided ID."""
        mock_manager = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "my-balanced-run"
        mock_manager.create_job.return_value = mock_job
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--job-id",
                "my-balanced-run",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "my-balanced-run" in result.output

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-vbt")
    def test_sweep_balanced_vbt_and_flow_params(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced with VBT and flow sensitivity parameters."""
        mock_manager = MagicMock()
        mock_create_jm.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--vbt-mid-sensitivity",
                "0.5",
                "--vbt-spread-sensitivity",
                "0.3",
                "--flow-sensitivity",
                "0.2",
                "--trading-rounds",
                "50",
                "--issuer-specific-pricing",
                "--dealer-concentration-limit",
                "0.1",
                "--no-rollover",
                "--no-risk-assessment",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.rollover_enabled is False
        assert config_arg.risk_assessment_enabled is False
        assert config_arg.trading_rounds == 50
        assert config_arg.issuer_specific_pricing is True


class TestSweepBalancedJobTracking:
    """Test sweep balanced job tracking error handling."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-balanced-jm-fail")
    def test_sweep_balanced_job_manager_failure(
        self,
        mock_gen_id,
        mock_create_jm,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep balanced continues when job manager initialization fails."""
        mock_create_jm.side_effect = RuntimeError("cannot reach supabase")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "balanced",
                "--out-dir",
                str(tmp_path / "balanced_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "Warning" in result.output


# ── sweep bank: additional parameters ──────────────────────────────────────────


class TestSweepBankAdditionalParams:
    """Test sweep bank with various option combinations."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-bank-full")
    def test_sweep_bank_full_params(
        self,
        mock_gen_id,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep bank with all bank-specific parameters passes them to config."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "bank",
                "--out-dir",
                str(tmp_path / "bank_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--n-banks",
                "3",
                "--reserve-ratio",
                "0.30",
                "--credit-risk-loading",
                "0.8",
                "--max-borrower-risk",
                "0.5",
                "--min-coverage-ratio",
                "0.2",
                "--cb-rate-escalation-slope",
                "0.10",
                "--cb-max-outstanding-ratio",
                "3.0",
                "--fast-atomic",
                "--n-agents",
                "50",
                "--maturity-days",
                "5",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.n_banks == 3
        assert config_arg.n_agents == 50
        assert config_arg.maturity_days == 5

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-bank-custom")
    def test_sweep_bank_custom_job_id(
        self,
        mock_gen_id,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep bank --job-id custom-id uses provided ID."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "bank",
                "--out-dir",
                str(tmp_path / "bank_out"),
                "--kappas",
                "0.5",
                "--concentrations",
                "1",
                "--mus",
                "0",
                "--job-id",
                "my-bank-run",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "my-bank-run" in result.output


# ── sweep nbfi: additional parameters ─────────────────────────────────────────


class TestSweepNBFIAdditionalParams:
    """Test sweep nbfi with various option combinations."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.nbfi_comparison.NBFIComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.nbfi_comparison.NBFIComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-nbfi-full")
    def test_sweep_nbfi_full_params(
        self,
        mock_gen_id,
        mock_runner_init,
        mock_run_all,
        mock_post,
        tmp_path,
    ):
        """sweep nbfi with all params passes them to config."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "nbfi",
                "--out-dir",
                str(tmp_path / "nbfi_out"),
                "--kappas",
                "0.3,0.5",
                "--concentrations",
                "0.5,1",
                "--mus",
                "0,0.5",
                "--n-agents",
                "50",
                "--maturity-days",
                "5",
                "--nbfi-share",
                "0.15",
                "--no-rollover",
                "--no-quiet",
                "--n-replicates",
                "2",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        init_call = mock_runner_init.call_args
        config_arg = init_call[1].get("config") or init_call[0][0]
        assert config_arg.n_agents == 50
        assert config_arg.rollover_enabled is False
        assert config_arg.quiet is False


# ── sweep strategy-outcomes and dealer-usage ──────────────────────────────────


class TestSweepStrategyOutcomes:
    """Test sweep strategy-outcomes command."""

    @patch("bilancio.analysis.strategy_outcomes.run_strategy_analysis")
    def test_strategy_outcomes_success(self, mock_run, tmp_path):
        """strategy-outcomes with valid experiment dir runs analysis."""
        by_run = tmp_path / "aggregate" / "strategy_outcomes_by_run.csv"
        overall = tmp_path / "aggregate" / "strategy_outcomes_overall.csv"
        (tmp_path / "aggregate").mkdir()
        by_run.write_text("data\n")
        overall.write_text("data\n")
        mock_run.return_value = (by_run, overall)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sweep", "strategy-outcomes", "--experiment", str(tmp_path)],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "OK" in result.output

    @patch("bilancio.analysis.strategy_outcomes.run_strategy_analysis")
    def test_strategy_outcomes_no_output(self, mock_run, tmp_path):
        """strategy-outcomes with no output prints warning."""
        mock_run.return_value = (None, None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sweep", "strategy-outcomes", "--experiment", str(tmp_path)],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "No output generated" in result.output

    @patch("bilancio.analysis.strategy_outcomes.run_strategy_analysis")
    def test_strategy_outcomes_verbose(self, mock_run, tmp_path):
        """strategy-outcomes -v enables verbose logging."""
        mock_run.return_value = (None, None)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sweep", "strategy-outcomes", "--experiment", str(tmp_path), "-v"],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"


class TestSweepDealerUsage:
    """Test sweep dealer-usage command."""

    @patch("bilancio.analysis.dealer_usage_summary.run_dealer_usage_analysis")
    def test_dealer_usage_success(self, mock_run, tmp_path):
        """dealer-usage with valid experiment dir runs analysis."""
        output = tmp_path / "aggregate" / "dealer_usage_by_run.csv"
        (tmp_path / "aggregate").mkdir()
        output.write_text("data\n")
        mock_run.return_value = output

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sweep", "dealer-usage", "--experiment", str(tmp_path)],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "OK" in result.output

    @patch("bilancio.analysis.dealer_usage_summary.run_dealer_usage_analysis")
    def test_dealer_usage_no_output(self, mock_run, tmp_path):
        """dealer-usage with no output prints warning."""
        mock_run.return_value = None

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["sweep", "dealer-usage", "--experiment", str(tmp_path)],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        assert "No output generated" in result.output


# ── sweep analyze ─────────────────────────────────────────────────────────────


class TestSweepAnalyze:
    """Test sweep analyze command."""

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    def test_sweep_analyze_calls_post_analysis(self, mock_offer, tmp_path):
        """sweep analyze --experiment --sweep-type calls _offer_post_sweep_analysis."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "analyze",
                "--experiment",
                str(tmp_path),
                "--sweep-type",
                "dealer",
                "--post-analysis",
                "drilldowns",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_offer.assert_called_once()

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    def test_sweep_analyze_default_post_analysis(self, mock_offer, tmp_path):
        """sweep analyze without --post-analysis defaults to 'all'."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sweep",
                "analyze",
                "--experiment",
                str(tmp_path),
                "--sweep-type",
                "bank",
            ],
        )

        assert result.exit_code == 0, f"Failed with output:\n{result.output}"
        mock_offer.assert_called_once_with(tmp_path, "bank", "all")


# ── sweep list with plugins ──────────────────────────────────────────────────


class TestSweepListWithPlugins:
    """Test sweep list when plugins are registered."""

    @patch("bilancio.scenarios.registry.get_registry")
    def test_sweep_list_empty_registry(self, mock_get_reg):
        """sweep list with empty registry shows message."""
        mock_get_reg.return_value = {}
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "list"])
        assert result.exit_code == 0
        assert "No scenario plugins" in result.output

    @patch("bilancio.scenarios.registry.get_registry")
    def test_sweep_list_with_plugins(self, mock_get_reg):
        """sweep list shows registered plugin metadata."""
        mock_meta = MagicMock()
        mock_meta.display_name = "Test Plugin"
        mock_meta.description = "A test scenario"
        mock_meta.version = "1.0"
        mock_meta.instruments_used = ["payable"]
        mock_meta.agent_types = ["firm"]
        mock_meta.supports_dealer = True
        mock_meta.supports_lender = False

        mock_plugin = MagicMock()
        mock_plugin.metadata = mock_meta
        mock_dim = MagicMock()
        mock_dim.name = "kappa"
        mock_dim.display_name = "Liquidity Ratio"
        mock_dim.description = "L0/S1 ratio"
        mock_dim.default_values = [0.5, 1.0]
        mock_plugin.parameter_dimensions.return_value = [mock_dim]

        mock_get_reg.return_value = {"test": mock_plugin}

        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "list"])
        assert result.exit_code == 0
        assert "Test Plugin" in result.output
        assert "A test scenario" in result.output


# ── build_cli_args → CLI end-to-end validation ─────────────────────────────


class TestBuildCliArgsCLIAcceptance:
    """P3: Verify build_cli_args output is accepted by each sweep command's Click parser.

    These tests catch flag mismatches where build_cli_args emits a flag
    the target command doesn't accept (exit_code=2, 'No such option').
    """

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-e2e-balanced")
    def test_balanced_with_full_advanced_params(
        self, mock_gen, mock_jm, mock_init, mock_run, mock_post, tmp_path,
    ):
        """build_cli_args for balanced with all advanced params produces valid CLI."""
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args

        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "n_agents": 50,
                "maturity_days": 5,
                "kappas": "0.5",
                "concentrations": "1",
                "mus": "0",
                "outside_mid_ratios": "0.90",
                "risk_aversion": "0.5",
                "planning_horizon": 10,
                "aggressiveness": "0.8",
                "default_observability": "1.0",
                "trading_motive": "liquidity_then_earning",
                "risk_premium": "0.02",
                "risk_urgency": "0.30",
                "risk_assessment": True,
                "enable_lender": True,
                "lender_share": "0.10",
                "lender_min_coverage": "0.5",
                "lender_ranking_mode": "profit",
                "lender_coverage_mode": "gate",
                "rollover": True,
            },
            out_dir=tmp_path / "balanced_out",
        )

        mock_jm.return_value = MagicMock()
        cli_args = build_cli_args(result)

        runner = CliRunner()
        res = runner.invoke(cli, ["sweep", "balanced"] + cli_args)
        assert res.exit_code == 0, f"balanced rejected args: {res.output}"

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.bank_comparison.BankComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-e2e-bank")
    def test_bank_with_advanced_params_no_crash(
        self, mock_gen, mock_init, mock_run, mock_post, tmp_path,
    ):
        """build_cli_args for bank with trader/risk params must NOT emit unsupported flags."""
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args

        result = SweepSetupResult(
            sweep_type="bank",
            cloud=False,
            params={
                "n_agents": 50,
                "maturity_days": 5,
                "kappas": "0.5",
                "concentrations": "1",
                "mus": "0",
                "outside_mid_ratios": "0.90",
                # These are collected by setup but must NOT appear in bank CLI args
                "risk_aversion": "0.5",
                "planning_horizon": 10,
                "risk_premium": "0.02",
                "risk_urgency": "0.30",
                "trading_motive": "liquidity_then_earning",
                # Bank-specific
                "n_banks": 5,
                "reserve_ratio": "0.50",
                "credit_risk_loading": "0.5",
                "rollover": True,
            },
            out_dir=tmp_path / "bank_out",
        )

        cli_args = build_cli_args(result)

        runner = CliRunner()
        res = runner.invoke(cli, ["sweep", "bank"] + cli_args)
        assert res.exit_code == 0, f"bank rejected args: {res.output}"

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.nbfi_comparison.NBFIComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.nbfi_comparison.NBFIComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-e2e-nbfi")
    def test_nbfi_with_advanced_params_no_crash(
        self, mock_gen, mock_init, mock_run, mock_post, tmp_path,
    ):
        """build_cli_args for nbfi with trader/risk params must NOT emit unsupported flags."""
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args

        result = SweepSetupResult(
            sweep_type="nbfi",
            cloud=False,
            params={
                "n_agents": 50,
                "maturity_days": 5,
                "kappas": "0.5",
                "concentrations": "1",
                "mus": "0",
                "outside_mid_ratios": "0.90",
                # These are collected by setup but must NOT appear in nbfi CLI args
                "risk_aversion": "0.5",
                "planning_horizon": 10,
                "risk_premium": "0.02",
                # NBFI-specific
                "lender_share": "0.10",
                "rollover": True,
            },
            out_dir=tmp_path / "nbfi_out",
        )

        cli_args = build_cli_args(result)

        runner = CliRunner()
        res = runner.invoke(cli, ["sweep", "nbfi"] + cli_args)
        assert res.exit_code == 0, f"nbfi rejected args: {res.output}"

    @patch("bilancio.ui.cli.sweep._offer_post_sweep_analysis")
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.run_all",
        return_value=[],
    )
    @patch(
        "bilancio.experiments.balanced_comparison.BalancedComparisonRunner.__init__",
        return_value=None,
    )
    @patch("bilancio.ui.cli.sweep.create_job_manager")
    @patch("bilancio.ui.cli.sweep.generate_job_id", return_value="test-e2e-banking")
    def test_balanced_with_enable_banking(
        self, mock_gen, mock_jm, mock_init, mock_run, mock_post, tmp_path,
    ):
        """build_cli_args for balanced with enable_banking emits valid bank arm flags."""
        from bilancio.ui.sweep_setup import SweepSetupResult, build_cli_args

        result = SweepSetupResult(
            sweep_type="balanced",
            cloud=False,
            params={
                "n_agents": 50,
                "kappas": "0.5",
                "concentrations": "1",
                "mus": "0",
                "outside_mid_ratios": "0.90",
                "enable_banking": True,
                "rollover": True,
            },
            out_dir=tmp_path / "balanced_out",
        )

        mock_jm.return_value = MagicMock()
        cli_args = build_cli_args(result)

        # Verify bank arm flags are present
        assert "--enable-bank-passive" in cli_args
        assert "--enable-bank-dealer" in cli_args
        assert "--enable-bank-dealer-nbfi" in cli_args

        runner = CliRunner()
        res = runner.invoke(cli, ["sweep", "balanced"] + cli_args)
        assert res.exit_code == 0, f"balanced+banking rejected args: {res.output}"
