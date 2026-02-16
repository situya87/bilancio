"""Tests for MetricsComputer."""

from pathlib import Path

import pytest

from bilancio.analysis.metrics_computer import MetricsBundle, MetricsComputer
from bilancio.storage.artifact_loaders import LocalArtifactLoader

# Sample events in JSONL format for testing
SAMPLE_EVENTS_JSONL = """{"type": "setup", "event": "mint_reserves", "day": 0, "amount": 10000, "to": "Bank1"}
{"type": "setup", "event": "create_payable", "day": 0, "from": "Firm1", "to": "Firm2", "amount": 500, "due_day": 1}
{"type": "phase", "day": 1, "phase": "B"}
{"type": "settlement", "day": 1, "from": "Firm1", "to": "Firm2", "amount": 500, "event": "settled"}
{"type": "phase", "day": 1, "phase": "C"}
{"type": "end_day", "day": 1}
{"type": "phase", "day": 2, "phase": "B"}
{"type": "phase", "day": 2, "phase": "C"}
{"type": "end_day", "day": 2}
"""

# Sample balances CSV content
SAMPLE_BALANCES_CSV = """day,agent,account,balance
0,Bank1,reserves,10000
0,Firm1,deposits,2000
0,Firm2,deposits,1000
1,Bank1,reserves,10000
1,Firm1,deposits,1500
1,Firm2,deposits,1500
"""


@pytest.fixture
def sample_events_file(tmp_path: Path) -> Path:
    """Create a sample events.jsonl file."""
    events_file = tmp_path / "events.jsonl"
    events_file.write_text(SAMPLE_EVENTS_JSONL)
    return events_file


@pytest.fixture
def sample_balances_file(tmp_path: Path) -> Path:
    """Create a sample balances.csv file."""
    balances_file = tmp_path / "balances.csv"
    balances_file.write_text(SAMPLE_BALANCES_CSV)
    return balances_file


@pytest.fixture
def loader(tmp_path: Path) -> LocalArtifactLoader:
    """Create a LocalArtifactLoader for temp directory."""
    return LocalArtifactLoader(tmp_path)


@pytest.fixture
def computer(loader: LocalArtifactLoader) -> MetricsComputer:
    """Create a MetricsComputer instance."""
    return MetricsComputer(loader)


class TestMetricsComputerCompute:
    """Tests for MetricsComputer.compute()."""

    def test_compute_with_events_only(self, computer: MetricsComputer, sample_events_file: Path):
        """compute() works with only events_jsonl artifact."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        assert isinstance(bundle, MetricsBundle)
        assert isinstance(bundle.day_metrics, list)
        assert isinstance(bundle.debtor_shares, list)
        assert isinstance(bundle.intraday, list)
        assert isinstance(bundle.summary, dict)

    def test_compute_with_events_and_balances(
        self,
        computer: MetricsComputer,
        sample_events_file: Path,
        sample_balances_file: Path,
    ):
        """compute() works with both events_jsonl and balances_csv."""
        artifacts = {
            "events_jsonl": "events.jsonl",
            "balances_csv": "balances.csv",
        }
        bundle = computer.compute(artifacts)

        assert isinstance(bundle, MetricsBundle)
        # With balances, M_t and G_t should potentially be computed
        assert bundle.day_metrics is not None

    def test_compute_raises_keyerror_when_events_missing(self, computer: MetricsComputer):
        """compute() raises KeyError when events_jsonl is missing."""
        artifacts: dict[str, str] = {}

        with pytest.raises(KeyError) as exc_info:
            computer.compute(artifacts)

        assert "events_jsonl" in str(exc_info.value)

    def test_compute_raises_keyerror_when_events_none(self, computer: MetricsComputer):
        """compute() raises KeyError when events_jsonl is None."""
        artifacts = {"events_jsonl": None}

        with pytest.raises(KeyError):
            computer.compute(artifacts)

    def test_compute_raises_filenotfound_for_missing_events_file(
        self, computer: MetricsComputer, tmp_path: Path
    ):
        """compute() raises FileNotFoundError when events file doesn't exist."""
        artifacts = {"events_jsonl": "nonexistent.jsonl"}

        with pytest.raises(FileNotFoundError):
            computer.compute(artifacts)

    def test_compute_with_day_list(self, computer: MetricsComputer, sample_events_file: Path):
        """compute() accepts optional day_list parameter."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts, day_list=[1])

        assert isinstance(bundle, MetricsBundle)

    def test_compute_ignores_missing_balances_file(
        self, computer: MetricsComputer, sample_events_file: Path
    ):
        """compute() ignores balances_csv if file doesn't exist."""
        artifacts = {
            "events_jsonl": "events.jsonl",
            "balances_csv": "nonexistent_balances.csv",
        }

        # Should not raise, just skip balances
        bundle = computer.compute(artifacts)
        assert isinstance(bundle, MetricsBundle)

    def test_compute_returns_metrics_bundle(
        self, computer: MetricsComputer, sample_events_file: Path
    ):
        """compute() returns a MetricsBundle instance."""
        artifacts = {"events_jsonl": "events.jsonl"}
        result = computer.compute(artifacts)

        assert isinstance(result, MetricsBundle)


class TestMetricsComputerWriteOutputs:
    """Tests for MetricsComputer.write_outputs()."""

    def test_write_outputs_creates_metrics_csv(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates metrics.csv file."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(bundle, output_dir)

        assert "metrics_csv" in paths
        assert paths["metrics_csv"].exists()
        assert paths["metrics_csv"].name == "metrics.csv"

    def test_write_outputs_creates_metrics_json(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates metrics.json file."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(bundle, output_dir)

        assert "metrics_json" in paths
        assert paths["metrics_json"].exists()
        assert paths["metrics_json"].name == "metrics.json"

    def test_write_outputs_creates_debtor_shares_csv(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates debtor_shares.csv file."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(bundle, output_dir)

        assert "debtor_shares_csv" in paths
        assert paths["debtor_shares_csv"].exists()
        assert paths["debtor_shares_csv"].name == "debtor_shares.csv"

    def test_write_outputs_creates_intraday_csv(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates intraday.csv file."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(bundle, output_dir)

        assert "intraday_csv" in paths
        assert paths["intraday_csv"].exists()
        assert paths["intraday_csv"].name == "intraday.csv"

    def test_write_outputs_creates_metrics_html(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates metrics.html file."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(bundle, output_dir)

        assert "metrics_html" in paths
        assert paths["metrics_html"].exists()
        assert paths["metrics_html"].name == "metrics.html"

    def test_write_outputs_creates_all_expected_files(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates all 5 expected output files."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(bundle, output_dir)

        expected_files = {
            "metrics_csv",
            "metrics_json",
            "debtor_shares_csv",
            "intraday_csv",
            "metrics_html",
        }
        assert set(paths.keys()) == expected_files

    def test_write_outputs_creates_output_directory(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() creates output directory if it doesn't exist."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        computer.write_outputs(bundle, output_dir)

        assert output_dir.exists()

    def test_write_outputs_accepts_title_and_subtitle(
        self, computer: MetricsComputer, sample_events_file: Path, tmp_path: Path
    ):
        """write_outputs() accepts optional title and subtitle."""
        artifacts = {"events_jsonl": "events.jsonl"}
        bundle = computer.compute(artifacts)

        output_dir = tmp_path / "output"
        paths = computer.write_outputs(
            bundle, output_dir, title="Test Report", subtitle="Test Subtitle"
        )

        # Just verify it doesn't raise and creates the files
        assert paths["metrics_html"].exists()


class TestMetricsBundle:
    """Tests for MetricsBundle dataclass."""

    def test_metrics_bundle_has_day_metrics_attribute(self):
        """MetricsBundle has day_metrics attribute."""
        bundle = MetricsBundle(
            day_metrics=[],
            debtor_shares=[],
            intraday=[],
            summary={},
        )
        assert hasattr(bundle, "day_metrics")

    def test_metrics_bundle_has_debtor_shares_attribute(self):
        """MetricsBundle has debtor_shares attribute."""
        bundle = MetricsBundle(
            day_metrics=[],
            debtor_shares=[],
            intraday=[],
            summary={},
        )
        assert hasattr(bundle, "debtor_shares")

    def test_metrics_bundle_has_intraday_attribute(self):
        """MetricsBundle has intraday attribute."""
        bundle = MetricsBundle(
            day_metrics=[],
            debtor_shares=[],
            intraday=[],
            summary={},
        )
        assert hasattr(bundle, "intraday")

    def test_metrics_bundle_has_summary_attribute(self):
        """MetricsBundle has summary attribute."""
        bundle = MetricsBundle(
            day_metrics=[],
            debtor_shares=[],
            intraday=[],
            summary={},
        )
        assert hasattr(bundle, "summary")

    def test_metrics_bundle_stores_values(self):
        """MetricsBundle stores provided values correctly."""
        day_metrics = [{"day": 1, "S_t": 100}]
        debtor_shares = [{"day": 1, "agent": "A", "DS_t": 0.5}]
        intraday = [{"day": 1, "step": 0, "P_prefix": 50}]
        summary = {"phi_total": 100.0, "delta_total": 50.0}

        bundle = MetricsBundle(
            day_metrics=day_metrics,
            debtor_shares=debtor_shares,
            intraday=intraday,
            summary=summary,
        )

        assert bundle.day_metrics == day_metrics
        assert bundle.debtor_shares == debtor_shares
        assert bundle.intraday == intraday
        assert bundle.summary == summary


class TestMetricsComputerInit:
    """Tests for MetricsComputer initialization."""

    def test_init_stores_loader(self, loader: LocalArtifactLoader):
        """__init__ stores the loader."""
        computer = MetricsComputer(loader)
        assert computer.loader is loader

    def test_accepts_artifact_loader_protocol(self, tmp_path: Path):
        """__init__ accepts any ArtifactLoader implementation."""
        loader = LocalArtifactLoader(tmp_path)
        computer = MetricsComputer(loader)
        assert computer.loader is not None
