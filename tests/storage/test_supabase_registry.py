"""Contract tests for SupabaseRegistryStore.

Tests verify that SupabaseRegistryStore correctly:
- Maps RegistryEntry fields to runs/metrics table columns
- Handles Decimal-to-float conversion for storage
- Parses database rows back into RegistryEntry objects
- Reconstructs artifact paths from modal_volume_path
- Handles metrics as both list (join) and dict forms
- Filters queries correctly
- Gracefully handles Supabase errors
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.storage.models import RegistryEntry, RunStatus
from bilancio.storage.supabase_registry import SupabaseRegistryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Create a mock Supabase client with fluent table API."""
    return MagicMock()


def _make_entry(
    run_id: str = "balanced_passive_abc123",
    experiment_id: str = "test-experiment",
    status: RunStatus = RunStatus.COMPLETED,
    with_metrics: bool = True,
    with_artifacts: bool = True,
    error: str | None = None,
) -> RegistryEntry:
    """Create a sample RegistryEntry for testing."""
    parameters = {
        "kappa": Decimal("0.5"),
        "concentration": Decimal("1.0"),
        "mu": Decimal("0"),
        "seed": 42,
        "regime": "passive",
        "outside_mid_ratio": Decimal("0.9"),
        "extra_param": "ignored_by_runs_table",
    }
    metrics = {}
    if with_metrics:
        metrics = {
            "delta_total": Decimal("0.15"),
            "phi_total": Decimal("0.85"),
            "n_defaults": 3,
            "n_clears": 47,
            "time_to_stability": 7,
            "trading_effect": 0.0,
            "total_trades": 0,
            "total_trade_volume": Decimal("0"),
            "custom_metric": "extra_value",
        }
    artifact_paths = {}
    if with_artifacts:
        artifact_paths = {
            "scenario_yaml": "test-experiment/runs/balanced_passive_abc123/scenario.yaml",
            "events_jsonl": "test-experiment/runs/balanced_passive_abc123/out/events.jsonl",
        }
    return RegistryEntry(
        run_id=run_id,
        experiment_id=experiment_id,
        status=status,
        parameters=parameters,
        metrics=metrics,
        artifact_paths=artifact_paths,
        error=error,
    )


def _make_db_row(**overrides: object) -> dict:
    """Create a database row dict mimicking a Supabase runs+metrics join."""
    row: dict = {
        "run_id": "balanced_passive_abc123",
        "job_id": "test-experiment",
        "status": "completed",
        "kappa": 0.5,
        "concentration": 1.0,
        "mu": 0,
        "seed": 42,
        "regime": "passive",
        "outside_mid_ratio": 0.9,
        "error": None,
        "modal_volume_path": "test-experiment/runs/balanced_passive_abc123",
        "metrics": [
            {
                "run_id": "balanced_passive_abc123",
                "job_id": "test-experiment",
                "delta_total": 0.15,
                "phi_total": 0.85,
                "n_defaults": 3,
                "n_clears": 47,
                "time_to_stability": 7,
                "raw_metrics": {
                    "delta_total": 0.15,
                    "phi_total": 0.85,
                    "n_defaults": 3,
                    "custom": "value",
                },
            }
        ],
    }
    row.update(overrides)
    return row


# ===========================================================================
# upsert tests
# ===========================================================================


class TestUpsert:
    """Tests for SupabaseRegistryStore.upsert()."""

    def test_upsert_creates_runs_and_metrics_rows(self) -> None:
        """upsert writes both runs and metrics table rows."""
        client = _make_mock_client()
        runs_table = MagicMock()
        metrics_table = MagicMock()

        def table_router(name: str) -> MagicMock:
            if name == "runs":
                return runs_table
            elif name == "metrics":
                return metrics_table
            return MagicMock()

        client.table.side_effect = table_router

        # runs upsert
        runs_table.upsert.return_value.execute.return_value = MagicMock()
        # metrics select (check existing)
        metrics_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        # metrics insert
        metrics_table.insert.return_value.execute.return_value = MagicMock()

        store = SupabaseRegistryStore(client=client)
        entry = _make_entry()
        store.upsert(entry)

        # Verify runs upsert was called
        runs_table.upsert.assert_called_once()
        runs_data = runs_table.upsert.call_args[0][0]
        assert runs_data["run_id"] == "balanced_passive_abc123"
        assert runs_data["job_id"] == "test-experiment"
        assert runs_data["status"] == "completed"
        assert runs_data["kappa"] == float(Decimal("0.5"))
        assert runs_data["seed"] == 42

        # Verify metrics insert was called (no existing row)
        metrics_table.insert.assert_called_once()
        metrics_data = metrics_table.insert.call_args[0][0]
        assert metrics_data["run_id"] == "balanced_passive_abc123"
        assert metrics_data["delta_total"] == float(Decimal("0.15"))

    def test_upsert_updates_existing_metrics(self) -> None:
        """upsert updates metrics when a row already exists."""
        client = _make_mock_client()
        runs_table = MagicMock()
        metrics_table = MagicMock()

        def table_router(name: str) -> MagicMock:
            if name == "runs":
                return runs_table
            elif name == "metrics":
                return metrics_table
            return MagicMock()

        client.table.side_effect = table_router

        runs_table.upsert.return_value.execute.return_value = MagicMock()
        # Existing metrics row found
        metrics_table.select.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )
        metrics_table.update.return_value.eq.return_value.execute.return_value = MagicMock()

        store = SupabaseRegistryStore(client=client)
        store.upsert(_make_entry())

        # Verify update was called instead of insert
        metrics_table.update.assert_called_once()
        metrics_table.insert.assert_not_called()

    def test_upsert_entry_without_metrics(self) -> None:
        """upsert only writes runs row when entry has no metrics."""
        client = _make_mock_client()
        runs_table = MagicMock()
        metrics_table = MagicMock()

        def table_router(name: str) -> MagicMock:
            if name == "runs":
                return runs_table
            elif name == "metrics":
                return metrics_table
            return MagicMock()

        client.table.side_effect = table_router
        runs_table.upsert.return_value.execute.return_value = MagicMock()

        store = SupabaseRegistryStore(client=client)
        entry = _make_entry(with_metrics=False)
        store.upsert(entry)

        runs_table.upsert.assert_called_once()
        # metrics table should not be touched at all for select/insert/update
        metrics_table.select.assert_not_called()
        metrics_table.insert.assert_not_called()

    def test_upsert_handles_supabase_error(self) -> None:
        """upsert catches SUPABASE_OPERATION_ERRORS gracefully."""
        client = _make_mock_client()
        client.table.return_value.upsert.return_value.execute.side_effect = RuntimeError("fail")

        store = SupabaseRegistryStore(client=client)
        # Should not raise
        store.upsert(_make_entry())


# ===========================================================================
# get tests
# ===========================================================================


class TestGet:
    """Tests for SupabaseRegistryStore.get()."""

    def test_get_found(self) -> None:
        """get returns a RegistryEntry when the run exists."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.return_value = MagicMock(data=[_make_db_row()])

        store = SupabaseRegistryStore(client=client)
        entry = store.get("test-experiment", "balanced_passive_abc123")

        assert entry is not None
        assert entry.run_id == "balanced_passive_abc123"
        assert entry.experiment_id == "test-experiment"
        assert entry.status == RunStatus.COMPLETED
        # Verify metrics were extracted from raw_metrics
        assert entry.metrics.get("delta_total") == 0.15
        assert entry.metrics.get("custom") == "value"

    def test_get_not_found(self) -> None:
        """get returns None when the run does not exist."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.return_value = MagicMock(data=[])

        store = SupabaseRegistryStore(client=client)
        assert store.get("test-experiment", "nonexistent") is None

    def test_get_handles_supabase_error(self) -> None:
        """get catches SUPABASE_OPERATION_ERRORS and returns None."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.side_effect = ConnectionError("timeout")

        store = SupabaseRegistryStore(client=client)
        assert store.get("test-experiment", "run-id") is None


# ===========================================================================
# list_runs tests
# ===========================================================================


class TestListRuns:
    """Tests for SupabaseRegistryStore.list_runs()."""

    def test_list_runs_returns_run_ids(self) -> None:
        """list_runs returns a list of run_id strings."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value
        chain.execute.return_value = MagicMock(
            data=[{"run_id": "run-a"}, {"run_id": "run-b"}, {"run_id": "run-c"}]
        )

        store = SupabaseRegistryStore(client=client)
        result = store.list_runs("test-experiment")

        assert result == ["run-a", "run-b", "run-c"]

    def test_list_runs_empty(self) -> None:
        """list_runs returns empty list when no runs exist."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value
        chain.execute.return_value = MagicMock(data=[])

        store = SupabaseRegistryStore(client=client)
        assert store.list_runs("nonexistent") == []

    def test_list_runs_handles_error(self) -> None:
        """list_runs catches errors and returns empty list."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value
        chain.execute.side_effect = RuntimeError("fail")

        store = SupabaseRegistryStore(client=client)
        assert store.list_runs("test-experiment") == []


# ===========================================================================
# query tests
# ===========================================================================


class TestQuery:
    """Tests for SupabaseRegistryStore.query()."""

    def test_query_without_filters(self) -> None:
        """query with no filters returns all entries for the experiment."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value
        chain.execute.return_value = MagicMock(data=[_make_db_row()])

        store = SupabaseRegistryStore(client=client)
        results = store.query("test-experiment")

        assert len(results) == 1
        assert results[0].run_id == "balanced_passive_abc123"

    def test_query_with_filters(self) -> None:
        """query applies field=value filters to the database query."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        # Build a chain: .select().eq("job_id", ...).eq("regime", ...).execute()
        eq1 = table_mock.select.return_value.eq.return_value
        eq2 = eq1.eq.return_value
        eq2.execute.return_value = MagicMock(data=[_make_db_row()])

        store = SupabaseRegistryStore(client=client)
        results = store.query("test-experiment", filters={"regime": "passive"})

        assert len(results) == 1
        # Verify the filter was applied
        eq1.eq.assert_called_once_with("regime", "passive")

    def test_query_handles_error(self) -> None:
        """query catches errors and returns empty list."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value
        chain.execute.side_effect = ValueError("bad query")

        store = SupabaseRegistryStore(client=client)
        assert store.query("test-experiment") == []


# ===========================================================================
# get_completed_keys tests
# ===========================================================================


class TestGetCompletedKeys:
    """Tests for SupabaseRegistryStore.get_completed_keys()."""

    def test_get_completed_keys_returns_tuples(self) -> None:
        """get_completed_keys returns a set of parameter tuples."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.return_value = MagicMock(
            data=[
                {"seed": 42, "kappa": "0.5", "concentration": "1.0"},
                {"seed": 99, "kappa": "0.3", "concentration": "2.0"},
            ]
        )

        store = SupabaseRegistryStore(client=client)
        keys = store.get_completed_keys("test-experiment")

        assert len(keys) == 2
        assert (42, 0.5, 1.0) in keys
        assert (99, 0.3, 2.0) in keys

    def test_get_completed_keys_decimal_conversion(self) -> None:
        """get_completed_keys converts Decimal values to float for hashing."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.return_value = MagicMock(
            data=[
                {"seed": 42, "kappa": Decimal("0.5"), "concentration": Decimal("1.0")},
            ]
        )

        store = SupabaseRegistryStore(client=client)
        keys = store.get_completed_keys("test-experiment")

        assert (42, 0.5, 1.0) in keys

    def test_get_completed_keys_custom_fields(self) -> None:
        """get_completed_keys uses custom key_fields when provided."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.return_value = MagicMock(
            data=[{"kappa": "0.5", "mu": "0"}]
        )

        store = SupabaseRegistryStore(client=client)
        keys = store.get_completed_keys("test-experiment", key_fields=["kappa", "mu"])

        assert (0.5, 0) in keys

    def test_get_completed_keys_handles_none_values(self) -> None:
        """get_completed_keys handles None values in key fields."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.return_value = MagicMock(
            data=[{"seed": 42, "kappa": None, "concentration": "1.0"}]
        )

        store = SupabaseRegistryStore(client=client)
        keys = store.get_completed_keys("test-experiment")

        assert (42, None, 1.0) in keys

    def test_get_completed_keys_handles_error(self) -> None:
        """get_completed_keys returns empty set on error."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.eq.return_value
        chain.execute.side_effect = RuntimeError("fail")

        store = SupabaseRegistryStore(client=client)
        assert store.get_completed_keys("test-experiment") == set()


# ===========================================================================
# _build_runs_row tests
# ===========================================================================


class TestBuildRunsRow:
    """Tests for SupabaseRegistryStore._build_runs_row()."""

    def test_build_runs_row_column_mapping(self) -> None:
        """_build_runs_row maps entry fields to correct columns."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry()

        row = store._build_runs_row(entry)

        assert row["run_id"] == "balanced_passive_abc123"
        assert row["job_id"] == "test-experiment"
        assert row["status"] == "completed"
        # Parameters from RUNS_PARAM_COLUMNS
        assert row["kappa"] == float(Decimal("0.5"))
        assert row["concentration"] == float(Decimal("1.0"))
        assert row["mu"] == float(Decimal("0"))
        assert row["seed"] == 42
        assert row["regime"] == "passive"
        # Non-param-column parameters should NOT appear
        assert "extra_param" not in row

    def test_build_runs_row_decimal_conversion(self) -> None:
        """_build_runs_row converts Decimal values to float."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry()

        row = store._build_runs_row(entry)

        assert isinstance(row["kappa"], float)
        assert isinstance(row["concentration"], float)

    def test_build_runs_row_volume_path_extraction(self) -> None:
        """_build_runs_row extracts modal_volume_path from artifact_paths."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry(with_artifacts=True)

        row = store._build_runs_row(entry)

        assert row.get("modal_volume_path") == "test-experiment/runs/balanced_passive_abc123"

    def test_build_runs_row_with_error(self) -> None:
        """_build_runs_row includes error field when present."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry(error="something failed")

        row = store._build_runs_row(entry)

        assert row["error"] == "something failed"

    def test_build_runs_row_completed_status_sets_timestamp(self) -> None:
        """_build_runs_row sets completed_at for completed runs."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry(status=RunStatus.COMPLETED)

        row = store._build_runs_row(entry)

        assert "completed_at" in row

    def test_build_runs_row_failed_status_sets_timestamp(self) -> None:
        """_build_runs_row sets completed_at for failed runs."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry(status=RunStatus.FAILED)

        row = store._build_runs_row(entry)

        assert "completed_at" in row

    def test_build_runs_row_pending_status_no_timestamp(self) -> None:
        """_build_runs_row does not set completed_at for pending runs."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry(status=RunStatus.PENDING)

        row = store._build_runs_row(entry)

        assert "completed_at" not in row


# ===========================================================================
# _build_metrics_row tests
# ===========================================================================


class TestBuildMetricsRow:
    """Tests for SupabaseRegistryStore._build_metrics_row()."""

    def test_build_metrics_row_decimal_handling(self) -> None:
        """_build_metrics_row converts Decimal metric values to float."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry()

        row = store._build_metrics_row(entry)

        assert row["run_id"] == "balanced_passive_abc123"
        assert row["job_id"] == "test-experiment"
        assert row["delta_total"] == float(Decimal("0.15"))
        assert row["phi_total"] == float(Decimal("0.85"))
        assert row["n_defaults"] == 3
        # raw_metrics stores the full dict
        assert row["raw_metrics"] == entry.metrics

    def test_build_metrics_row_only_known_columns(self) -> None:
        """_build_metrics_row only maps known METRICS_COLUMNS to top-level columns."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        entry = _make_entry()

        row = store._build_metrics_row(entry)

        # custom_metric is in entry.metrics but not in METRICS_COLUMNS
        assert "custom_metric" not in row
        # But it's still in raw_metrics
        assert row["raw_metrics"].get("custom_metric") == "extra_value"


# ===========================================================================
# _row_to_entry tests
# ===========================================================================


class TestRowToEntry:
    """Tests for SupabaseRegistryStore._row_to_entry()."""

    def test_row_to_entry_with_metrics_list(self) -> None:
        """_row_to_entry handles metrics as a list (from Supabase join)."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row()

        entry = store._row_to_entry(row)

        assert entry.run_id == "balanced_passive_abc123"
        assert entry.experiment_id == "test-experiment"
        assert entry.status == RunStatus.COMPLETED
        # Parameters extracted from runs columns
        assert entry.parameters["kappa"] == 0.5
        assert entry.parameters["seed"] == 42
        assert entry.parameters["regime"] == "passive"
        # Metrics from raw_metrics
        assert entry.metrics["delta_total"] == 0.15
        assert entry.metrics["custom"] == "value"

    def test_row_to_entry_with_metrics_dict(self) -> None:
        """_row_to_entry handles metrics as a dict (alternative format)."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row(
            metrics={
                "raw_metrics": {"delta_total": 0.25, "phi_total": 0.75},
            }
        )

        entry = store._row_to_entry(row)

        assert entry.metrics["delta_total"] == 0.25
        assert entry.metrics["phi_total"] == 0.75

    def test_row_to_entry_metrics_fallback_to_columns(self) -> None:
        """_row_to_entry falls back to individual columns when raw_metrics is absent."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row(
            metrics=[
                {
                    "run_id": "balanced_passive_abc123",
                    "delta_total": 0.10,
                    "phi_total": 0.90,
                    "n_defaults": 2,
                    "n_clears": 48,
                    "time_to_stability": None,
                    "raw_metrics": None,
                }
            ]
        )

        entry = store._row_to_entry(row)

        assert entry.metrics["delta_total"] == 0.10
        assert entry.metrics["phi_total"] == 0.90
        assert entry.metrics["n_defaults"] == 2
        assert entry.metrics["n_clears"] == 48
        assert "time_to_stability" not in entry.metrics

    def test_row_to_entry_no_metrics(self) -> None:
        """_row_to_entry handles rows with no metrics data."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row(metrics=None)

        entry = store._row_to_entry(row)

        assert entry.metrics == {}

    def test_row_to_entry_artifact_paths_from_volume(self) -> None:
        """_row_to_entry reconstructs artifact paths from modal_volume_path."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row()

        entry = store._row_to_entry(row)

        assert entry.artifact_paths["scenario_yaml"].endswith("scenario.yaml")
        assert entry.artifact_paths["events_jsonl"].endswith("out/events.jsonl")
        assert entry.artifact_paths["balances_csv"].endswith("out/balances.csv")
        assert entry.artifact_paths["metrics_csv"].endswith("out/metrics.csv")
        assert entry.artifact_paths["run_html"].endswith("run.html")

    def test_row_to_entry_no_volume_path(self) -> None:
        """_row_to_entry returns empty artifact_paths when no volume path."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row(modal_volume_path=None)

        entry = store._row_to_entry(row)

        assert entry.artifact_paths == {}

    def test_row_to_entry_error_field(self) -> None:
        """_row_to_entry preserves error field from row."""
        store = SupabaseRegistryStore(client=None)
        store._initialized = True
        row = _make_db_row(status="failed", error="timeout exceeded")

        entry = store._row_to_entry(row)

        assert entry.status == RunStatus.FAILED
        assert entry.error == "timeout exceeded"


# ===========================================================================
# _convert_value / _parse_value tests
# ===========================================================================


class TestValueConversion:
    """Tests for _convert_value and _parse_value static methods."""

    def test_convert_decimal_to_float(self) -> None:
        """_convert_value converts Decimal to float."""
        assert SupabaseRegistryStore._convert_value(Decimal("0.5")) == 0.5
        assert isinstance(SupabaseRegistryStore._convert_value(Decimal("1")), float)

    def test_convert_preserves_bool(self) -> None:
        """_convert_value preserves bool values (not converting to int)."""
        assert SupabaseRegistryStore._convert_value(True) is True
        assert SupabaseRegistryStore._convert_value(False) is False

    def test_convert_passthrough(self) -> None:
        """_convert_value passes through other types unchanged."""
        assert SupabaseRegistryStore._convert_value("hello") == "hello"
        assert SupabaseRegistryStore._convert_value(42) == 42
        assert SupabaseRegistryStore._convert_value(None) is None

    def test_parse_string_float(self) -> None:
        """_parse_value parses string with '.' as float."""
        assert SupabaseRegistryStore._parse_value("0.5") == 0.5
        assert isinstance(SupabaseRegistryStore._parse_value("0.5"), float)

    def test_parse_string_int(self) -> None:
        """_parse_value parses string without '.' as int."""
        assert SupabaseRegistryStore._parse_value("42") == 42
        assert isinstance(SupabaseRegistryStore._parse_value("42"), int)

    def test_parse_non_numeric_string(self) -> None:
        """_parse_value returns non-numeric strings as-is."""
        assert SupabaseRegistryStore._parse_value("passive") == "passive"

    def test_parse_decimal_to_float(self) -> None:
        """_parse_value converts Decimal to float."""
        assert SupabaseRegistryStore._parse_value(Decimal("0.5")) == 0.5

    def test_parse_none(self) -> None:
        """_parse_value returns None as-is."""
        assert SupabaseRegistryStore._parse_value(None) is None

    def test_parse_int_passthrough(self) -> None:
        """_parse_value passes through int values unchanged."""
        assert SupabaseRegistryStore._parse_value(42) == 42


# ===========================================================================
# Client property tests
# ===========================================================================


class TestClientProperty:
    """Tests for the lazy client property."""

    def test_client_returns_injected_client(self) -> None:
        """When a client is injected, the property returns it."""
        mock_client = _make_mock_client()
        store = SupabaseRegistryStore(client=mock_client)

        assert store.client is mock_client

    def test_client_raises_when_not_configured(self) -> None:
        """client property raises RuntimeError when Supabase is not configured."""
        store = SupabaseRegistryStore(client=None)

        # The imports happen inside the property body via
        # `from bilancio.storage.supabase_client import ...`, so we patch
        # at the source module where they are defined.
        with patch(
            "bilancio.storage.supabase_client.is_supabase_configured",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="Supabase is not configured"):
                _ = store.client
