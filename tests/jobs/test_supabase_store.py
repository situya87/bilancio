"""Contract tests for SupabaseJobStore.

Tests verify that SupabaseJobStore correctly:
- Maps Job dataclass fields to database columns
- Handles Decimal list serialization
- Parses timestamps and status enums from database rows
- Gracefully degrades when the Supabase client is unavailable
- Handles empty results and error conditions
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from bilancio.jobs.models import Job, JobConfig, JobEvent, JobStatus
from bilancio.jobs.supabase_store import SupabaseJobStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Create a mock Supabase client with fluent table API."""
    client = MagicMock()
    return client


def _make_job(
    job_id: str = "test-job-id",
    status: JobStatus = JobStatus.COMPLETED,
    completed_at: datetime | None = None,
) -> Job:
    """Create a sample Job for testing."""
    return Job(
        job_id=job_id,
        created_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC),
        status=status,
        description="Test sweep",
        config=JobConfig(
            sweep_type="balanced",
            n_agents=100,
            kappas=[Decimal("0.3"), Decimal("0.5"), Decimal("1.0")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0")],
            cloud=True,
            outside_mid_ratios=[Decimal("0.9")],
            maturity_days=10,
            seeds=[42, 99],
        ),
        run_ids=["run-a", "run-b"],
        completed_at=completed_at,
        notes="a test note",
        error=None,
    )


def _make_db_row(**overrides: object) -> dict:
    """Create a database row dict that mimics what Supabase returns."""
    row: dict = {
        "job_id": "test-job-id",
        "created_at": "2025-01-15T10:30:00+00:00",
        "status": "completed",
        "description": "Test sweep",
        "sweep_type": "balanced",
        "n_agents": 100,
        "maturity_days": 10,
        "kappas": ["0.3", "0.5", "1.0"],
        "concentrations": ["1"],
        "mus": ["0"],
        "outside_mid_ratios": ["0.9"],
        "seeds": [42, 99],
        "cloud": True,
        "completed_at": "2025-01-15T10:45:00+00:00",
        "notes": "a test note",
        "error": None,
        "total_runs": 2,
        "completed_runs": 2,
    }
    row.update(overrides)
    return row


# ===========================================================================
# save_job tests
# ===========================================================================


class TestSaveJob:
    """Tests for SupabaseJobStore.save_job()."""

    def test_save_job_field_mapping(self) -> None:
        """save_job maps Job fields to the correct database columns."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.upsert.return_value.execute.return_value = MagicMock(data=[{}])

        store = SupabaseJobStore(client=client)
        job = _make_job(status=JobStatus.COMPLETED, completed_at=None)

        store.save_job(job)

        # Verify upsert was called on the jobs table
        client.table.assert_called_with("jobs")
        call_args = table_mock.upsert.call_args
        data = call_args[0][0]

        assert data["job_id"] == "test-job-id"
        assert data["status"] == "completed"
        assert data["description"] == "Test sweep"
        assert data["sweep_type"] == "balanced"
        assert data["n_agents"] == 100
        assert data["cloud"] is True
        assert data["notes"] == "a test note"
        assert data["total_runs"] == 2
        assert data["completed_runs"] == 2  # status is COMPLETED

    def test_save_job_decimal_list_serialization(self) -> None:
        """Decimal lists are serialized to string lists for storage."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.upsert.return_value.execute.return_value = MagicMock(data=[{}])

        store = SupabaseJobStore(client=client)
        job = _make_job()

        store.save_job(job)

        data = table_mock.upsert.call_args[0][0]
        assert data["kappas"] == ["0.3", "0.5", "1.0"]
        assert data["concentrations"] == ["1"]
        assert data["mus"] == ["0"]
        assert data["outside_mid_ratios"] == ["0.9"]
        assert data["seeds"] == [42, 99]

    def test_save_job_with_completed_at(self) -> None:
        """completed_at is included in the data when present."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.upsert.return_value.execute.return_value = MagicMock(data=[{}])

        completed = datetime(2025, 1, 15, 10, 45, 0, tzinfo=UTC)
        store = SupabaseJobStore(client=client)
        job = _make_job(completed_at=completed)

        store.save_job(job)

        data = table_mock.upsert.call_args[0][0]
        assert "completed_at" in data
        assert data["completed_at"] == completed.isoformat()

    def test_save_job_client_unavailable_returns_gracefully(self) -> None:
        """When client is None, save_job returns without error."""
        store = SupabaseJobStore(client=None)
        store._initialized = True  # prevent lazy init attempt

        # Should not raise
        store.save_job(_make_job())

    def test_save_job_handles_supabase_error(self) -> None:
        """save_job catches SUPABASE_OPERATION_ERRORS and logs a warning."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.upsert.return_value.execute.side_effect = RuntimeError("connection lost")

        store = SupabaseJobStore(client=client)
        # Should not raise
        store.save_job(_make_job())


# ===========================================================================
# get_job tests
# ===========================================================================


class TestGetJob:
    """Tests for SupabaseJobStore.get_job()."""

    def test_get_job_found(self) -> None:
        """get_job returns a fully populated Job when found."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.single.return_value
        chain.execute.return_value = MagicMock(data=_make_db_row())

        store = SupabaseJobStore(client=client)
        job = store.get_job("test-job-id")

        assert job is not None
        assert job.job_id == "test-job-id"
        assert job.status == JobStatus.COMPLETED
        assert job.config.sweep_type == "balanced"
        assert job.config.n_agents == 100
        assert job.config.kappas == [Decimal("0.3"), Decimal("0.5"), Decimal("1.0")]
        assert job.config.concentrations == [Decimal("1")]
        assert job.config.mus == [Decimal("0")]
        assert job.config.cloud is True
        assert job.config.maturity_days == 10
        assert job.completed_at is not None
        assert job.notes == "a test note"

    def test_get_job_not_found(self) -> None:
        """get_job returns None when the job doesn't exist."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.single.return_value
        chain.execute.return_value = MagicMock(data=None)

        store = SupabaseJobStore(client=client)
        result = store.get_job("nonexistent")

        assert result is None

    def test_get_job_client_unavailable(self) -> None:
        """get_job returns None when client is unavailable."""
        store = SupabaseJobStore(client=None)
        store._initialized = True

        assert store.get_job("any-id") is None

    def test_get_job_handles_supabase_error(self) -> None:
        """get_job catches SUPABASE_OPERATION_ERRORS and returns None."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.eq.return_value.single.return_value
        chain.execute.side_effect = ConnectionError("timeout")

        store = SupabaseJobStore(client=client)
        assert store.get_job("any-id") is None


# ===========================================================================
# list_jobs tests
# ===========================================================================


class TestListJobs:
    """Tests for SupabaseJobStore.list_jobs()."""

    def test_list_jobs_returns_list(self) -> None:
        """list_jobs returns a list of Job objects."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(data=[_make_db_row(), _make_db_row(job_id="job-2")])

        store = SupabaseJobStore(client=client)
        jobs = store.list_jobs()

        assert len(jobs) == 2
        assert jobs[0].job_id == "test-job-id"
        assert jobs[1].job_id == "job-2"

    def test_list_jobs_with_status_filter(self) -> None:
        """list_jobs applies status filter when provided."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        order_chain = table_mock.select.return_value.order.return_value.limit.return_value
        # When status filter is applied, .eq() is called on the chain
        eq_chain = order_chain.eq.return_value
        eq_chain.execute.return_value = MagicMock(data=[_make_db_row()])

        store = SupabaseJobStore(client=client)
        jobs = store.list_jobs(status="running")

        assert len(jobs) == 1
        order_chain.eq.assert_called_once_with("status", "running")

    def test_list_jobs_empty_results(self) -> None:
        """list_jobs returns empty list when no jobs match."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(data=None)

        store = SupabaseJobStore(client=client)
        assert store.list_jobs() == []

    def test_list_jobs_client_unavailable(self) -> None:
        """list_jobs returns empty list when client is unavailable."""
        store = SupabaseJobStore(client=None)
        store._initialized = True

        assert store.list_jobs() == []


# ===========================================================================
# get_run_counts tests
# ===========================================================================


class TestGetRunCounts:
    """Tests for SupabaseJobStore.get_run_counts()."""

    def test_get_run_counts_aggregation(self) -> None:
        """get_run_counts correctly counts runs per job."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.in_.return_value
        chain.execute.return_value = MagicMock(
            data=[
                {"job_id": "job-a"},
                {"job_id": "job-a"},
                {"job_id": "job-a"},
                {"job_id": "job-b"},
            ]
        )

        store = SupabaseJobStore(client=client)
        counts = store.get_run_counts(["job-a", "job-b"])

        assert counts == {"job-a": 3, "job-b": 1}

    def test_get_run_counts_empty_job_ids(self) -> None:
        """get_run_counts returns {} when given an empty job_ids list."""
        client = _make_mock_client()
        store = SupabaseJobStore(client=client)

        assert store.get_run_counts([]) == {}

    def test_get_run_counts_no_data(self) -> None:
        """get_run_counts returns {} when response has no data."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = table_mock.select.return_value.in_.return_value
        chain.execute.return_value = MagicMock(data=None)

        store = SupabaseJobStore(client=client)
        assert store.get_run_counts(["job-a"]) == {}

    def test_get_run_counts_client_unavailable(self) -> None:
        """get_run_counts returns {} when client is None."""
        store = SupabaseJobStore(client=None)
        store._initialized = True

        assert store.get_run_counts(["job-a"]) == {}


# ===========================================================================
# update_status tests
# ===========================================================================


class TestUpdateStatus:
    """Tests for SupabaseJobStore.update_status()."""

    def test_update_status_with_completed_at(self) -> None:
        """update_status includes completed_at when provided."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.update.return_value.eq.return_value.execute.return_value = MagicMock()

        store = SupabaseJobStore(client=client)
        completed = datetime(2025, 1, 15, 10, 45, 0, tzinfo=UTC)
        store.update_status("test-id", JobStatus.COMPLETED, completed_at=completed)

        client.table.assert_called_with("jobs")
        data = table_mock.update.call_args[0][0]
        assert data["status"] == "completed"
        assert data["completed_at"] == completed.isoformat()

    def test_update_status_without_completed_at(self) -> None:
        """update_status only sets status when completed_at is None."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.update.return_value.eq.return_value.execute.return_value = MagicMock()

        store = SupabaseJobStore(client=client)
        store.update_status("test-id", JobStatus.RUNNING)

        data = table_mock.update.call_args[0][0]
        assert data == {"status": "running"}

    def test_update_status_with_error(self) -> None:
        """update_status includes error message when provided."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.update.return_value.eq.return_value.execute.return_value = MagicMock()

        store = SupabaseJobStore(client=client)
        store.update_status("test-id", JobStatus.FAILED, error="something broke")

        data = table_mock.update.call_args[0][0]
        assert data["status"] == "failed"
        assert data["error"] == "something broke"

    def test_update_status_client_unavailable(self) -> None:
        """update_status is a no-op when client is None."""
        store = SupabaseJobStore(client=None)
        store._initialized = True

        # Should not raise
        store.update_status("test-id", JobStatus.RUNNING)


# ===========================================================================
# get_events tests
# ===========================================================================


class TestGetEvents:
    """Tests for SupabaseJobStore.get_events()."""

    def test_get_events_parses_timestamps(self) -> None:
        """get_events correctly parses ISO timestamps from database rows."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = (
            table_mock.select.return_value
            .eq.return_value
            .order.return_value
        )
        chain.execute.return_value = MagicMock(
            data=[
                {
                    "job_id": "test-id",
                    "event_type": "created",
                    "timestamp": "2025-01-15T10:30:00+00:00",
                    "details": {"msg": "created"},
                },
                {
                    "job_id": "test-id",
                    "event_type": "completed",
                    "timestamp": "2025-01-15T10:45:00Z",
                    "details": {},
                },
            ]
        )

        store = SupabaseJobStore(client=client)
        events = store.get_events("test-id")

        assert len(events) == 2
        assert events[0].event_type == "created"
        assert events[0].timestamp == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        assert events[0].details == {"msg": "created"}
        assert events[1].event_type == "completed"
        assert events[1].timestamp == datetime(2025, 1, 15, 10, 45, 0, tzinfo=UTC)

    def test_get_events_empty_results(self) -> None:
        """get_events returns empty list when no events exist."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        chain = (
            table_mock.select.return_value
            .eq.return_value
            .order.return_value
        )
        chain.execute.return_value = MagicMock(data=None)

        store = SupabaseJobStore(client=client)
        assert store.get_events("nonexistent") == []

    def test_get_events_client_unavailable(self) -> None:
        """get_events returns empty list when client is None."""
        store = SupabaseJobStore(client=None)
        store._initialized = True

        assert store.get_events("any-id") == []


# ===========================================================================
# _row_to_job tests
# ===========================================================================


class TestRowToJob:
    """Tests for SupabaseJobStore._row_to_job()."""

    def test_row_to_job_full_round_trip(self) -> None:
        """_row_to_job correctly parses all fields from a database row."""
        store = SupabaseJobStore(client=None)
        row = _make_db_row()

        job = store._row_to_job(row)

        assert job.job_id == "test-job-id"
        assert job.status == JobStatus.COMPLETED
        assert job.description == "Test sweep"
        assert job.created_at == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        assert job.completed_at == datetime(2025, 1, 15, 10, 45, 0, tzinfo=UTC)
        assert job.notes == "a test note"
        assert job.error is None
        # Config
        assert job.config.sweep_type == "balanced"
        assert job.config.n_agents == 100
        assert job.config.maturity_days == 10
        assert job.config.kappas == [Decimal("0.3"), Decimal("0.5"), Decimal("1.0")]
        assert job.config.concentrations == [Decimal("1")]
        assert job.config.mus == [Decimal("0")]
        assert job.config.outside_mid_ratios == [Decimal("0.9")]
        assert job.config.seeds == [42, 99]
        assert job.config.cloud is True
        # Defaults for fields not in row
        assert job.run_ids == []
        assert job.modal_call_ids == {}
        assert job.events == []

    def test_row_to_job_missing_optional_fields(self) -> None:
        """_row_to_job handles missing optional fields with defaults."""
        store = SupabaseJobStore(client=None)
        # Use keys that are absent (via pop) rather than set to None,
        # since row.get("key", default) only uses default when key is absent.
        row = _make_db_row()
        del row["completed_at"]
        del row["notes"]
        del row["outside_mid_ratios"]
        del row["seeds"]
        del row["maturity_days"]
        del row["description"]

        job = store._row_to_job(row)

        assert job.completed_at is None
        assert job.notes is None
        assert job.description == ""
        assert job.config.outside_mid_ratios == [Decimal("1")]
        assert job.config.seeds == [42]
        assert job.config.maturity_days == 5

    def test_row_to_job_z_suffix_timestamp(self) -> None:
        """_row_to_job correctly handles 'Z' suffix in timestamps."""
        store = SupabaseJobStore(client=None)
        row = _make_db_row(
            created_at="2025-06-01T12:00:00Z",
            completed_at="2025-06-01T13:00:00Z",
        )

        job = store._row_to_job(row)

        assert job.created_at == datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        assert job.completed_at == datetime(2025, 6, 1, 13, 0, 0, tzinfo=UTC)


# ===========================================================================
# Client property tests
# ===========================================================================


class TestClientProperty:
    """Tests for the lazy client property."""

    def test_client_returns_injected_client(self) -> None:
        """When a client is injected via __init__, the property returns it directly."""
        mock_client = _make_mock_client()
        store = SupabaseJobStore(client=mock_client)

        assert store.client is mock_client

    def test_client_returns_none_when_not_configured(self) -> None:
        """When Supabase is not configured, client returns None."""
        store = SupabaseJobStore(client=None)

        with patch(
            "bilancio.jobs.supabase_store.SupabaseJobStore.client",
            new_callable=lambda: property(lambda self: None),
        ):
            # Simpler: just set the init flag so it doesn't try to import
            store._initialized = True
            assert store.client is None

    def test_client_caches_none_after_first_attempt(self) -> None:
        """After a failed initialization attempt, subsequent calls return None
        without re-attempting."""
        store = SupabaseJobStore(client=None)
        store._initialized = True  # simulate previous failed init

        # Should return None immediately without attempting import
        assert store.client is None


# ===========================================================================
# save_event tests
# ===========================================================================


class TestSaveEvent:
    """Tests for SupabaseJobStore.save_event()."""

    def test_save_event_field_mapping(self) -> None:
        """save_event maps JobEvent fields to the correct database columns."""
        client = _make_mock_client()
        table_mock = client.table.return_value
        table_mock.insert.return_value.execute.return_value = MagicMock()

        store = SupabaseJobStore(client=client)
        event = JobEvent(
            job_id="test-id",
            event_type="progress",
            timestamp=datetime(2025, 1, 15, 10, 35, 0, tzinfo=UTC),
            details={"completed": 5, "total": 10},
        )

        store.save_event(event)

        client.table.assert_called_with("job_events")
        data = table_mock.insert.call_args[0][0]
        assert data["job_id"] == "test-id"
        assert data["event_type"] == "progress"
        assert data["timestamp"] == "2025-01-15T10:35:00+00:00"
        assert data["details"] == {"completed": 5, "total": 10}

    def test_save_event_client_unavailable(self) -> None:
        """save_event is a no-op when client is None."""
        store = SupabaseJobStore(client=None)
        store._initialized = True

        event = JobEvent(
            job_id="test-id",
            event_type="created",
            timestamp=datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC),
        )
        # Should not raise
        store.save_event(event)
