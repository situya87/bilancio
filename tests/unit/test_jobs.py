"""Unit tests for the job system."""

import json
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from bilancio.jobs import (
    Job,
    JobConfig,
    JobEvent,
    JobManager,
    JobStatus,
    generate_job_id,
    validate_job_id,
)


class TestJobId:
    """Tests for job ID generation and validation."""

    def test_generate_job_id_format(self):
        """Test generate_job_id returns correct format (4 words, hyphen-separated)."""
        job_id = generate_job_id()
        parts = job_id.split("-")
        assert len(parts) == 4
        assert all(part.isalpha() and part.islower() for part in parts)

    def test_validate_job_id_accepts_valid_ids(self):
        """Test validate_job_id accepts valid IDs."""
        assert validate_job_id("castle-river-mountain-forest") is True
        assert validate_job_id("bright-ocean-swift-tiger") is True
        assert validate_job_id("alpha-beta") is True  # Minimum 2 words

    def test_validate_job_id_rejects_empty(self):
        """Test validate_job_id rejects empty strings."""
        assert validate_job_id("") is False

    def test_validate_job_id_rejects_single_word(self):
        """Test validate_job_id rejects single word IDs."""
        assert validate_job_id("castle") is False
        assert validate_job_id("river") is False

    def test_validate_job_id_rejects_uppercase(self):
        """Test validate_job_id rejects IDs with uppercase letters."""
        assert validate_job_id("Castle-river-mountain-forest") is False
        assert validate_job_id("castle-RIVER-mountain-forest") is False
        assert validate_job_id("CASTLE-RIVER-MOUNTAIN-FOREST") is False

    def test_validate_job_id_rejects_numbers(self):
        """Test validate_job_id rejects IDs with numbers."""
        assert validate_job_id("castle-river-123-forest") is False
        assert validate_job_id("castle1-river-mountain-forest") is False
        assert validate_job_id("1castle-river-mountain-forest") is False

    def test_validate_job_id_rejects_special_characters(self):
        """Test validate_job_id rejects IDs with special characters."""
        assert validate_job_id("castle_river_mountain_forest") is False
        assert validate_job_id("castle.river.mountain.forest") is False
        assert validate_job_id("castle river mountain forest") is False

    def test_uniqueness_over_multiple_generations(self):
        """Test uniqueness over multiple generations."""
        # Generate many IDs and check for uniqueness
        job_ids = [generate_job_id() for _ in range(100)]
        # With 597 words choosing 4, there are many combinations
        # 100 IDs should all be unique (collision probability is very low)
        assert len(set(job_ids)) == len(job_ids)

    def test_generated_ids_are_valid(self):
        """Test that all generated IDs pass validation."""
        for _ in range(20):
            job_id = generate_job_id()
            assert validate_job_id(job_id) is True


class TestJobConfig:
    """Tests for JobConfig serialization."""

    def test_to_dict_from_dict_round_trip(self):
        """Test to_dict/from_dict round-trip preserves data."""
        config = JobConfig(
            sweep_type="balanced",
            n_agents=100,
            kappas=[Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
            concentrations=[Decimal("0.5"), Decimal("1.0")],
            mus=[Decimal("0"), Decimal("0.5"), Decimal("1.0")],
            cloud=True,
            outside_mid_ratios=[Decimal("0.8"), Decimal("1.0")],
            maturity_days=10,
            seeds=[42, 123, 456],
        )

        data = config.to_dict()
        restored = JobConfig.from_dict(data)

        assert restored.sweep_type == config.sweep_type
        assert restored.n_agents == config.n_agents
        assert restored.kappas == config.kappas
        assert restored.concentrations == config.concentrations
        assert restored.mus == config.mus
        assert restored.cloud == config.cloud
        assert restored.outside_mid_ratios == config.outside_mid_ratios
        assert restored.maturity_days == config.maturity_days
        assert restored.seeds == config.seeds

    def test_defaults(self):
        """Test JobConfig defaults are applied correctly."""
        config = JobConfig(
            sweep_type="ring",
            n_agents=50,
            kappas=[Decimal("1")],
            concentrations=[Decimal("1")],
            mus=[Decimal("0.5")],
        )

        assert config.cloud is False
        assert config.outside_mid_ratios == [Decimal("1")]
        assert config.maturity_days == 5
        assert config.seeds == [42]

    def test_from_dict_uses_defaults_for_missing_keys(self):
        """Test from_dict uses defaults for missing optional keys."""
        data = {
            "sweep_type": "ring",
            "n_agents": 50,
            "kappas": ["1"],
            "concentrations": ["1"],
            "mus": ["0.5"],
        }

        config = JobConfig.from_dict(data)

        assert config.cloud is False
        assert config.outside_mid_ratios == [Decimal("1")]
        assert config.maturity_days == 5
        assert config.seeds == [42]

    def test_to_dict_serializes_decimals_as_strings(self):
        """Test to_dict converts Decimals to strings for JSON compatibility."""
        config = JobConfig(
            sweep_type="ring",
            n_agents=50,
            kappas=[Decimal("0.25")],
            concentrations=[Decimal("0.5")],
            mus=[Decimal("0.75")],
            outside_mid_ratios=[Decimal("0.9")],
        )

        data = config.to_dict()

        assert data["kappas"] == ["0.25"]
        assert data["concentrations"] == ["0.5"]
        assert data["mus"] == ["0.75"]
        assert data["outside_mid_ratios"] == ["0.9"]


class TestJobEvent:
    """Tests for JobEvent serialization."""

    def test_to_dict_from_dict_round_trip(self):
        """Test to_dict/from_dict round-trip preserves data."""
        timestamp = datetime(2025, 1, 12, 10, 30, 45)
        event = JobEvent(
            job_id="castle-river-mountain-forest",
            event_type="progress",
            timestamp=timestamp,
            details={"run_id": "run_001", "metrics": {"delta_total": 0.05}},
        )

        data = event.to_dict()
        restored = JobEvent.from_dict(data)

        assert restored.job_id == event.job_id
        assert restored.event_type == event.event_type
        assert restored.timestamp == event.timestamp
        assert restored.details == event.details

    def test_to_dict_serializes_timestamp_as_iso(self):
        """Test to_dict converts timestamp to ISO format string."""
        timestamp = datetime(2025, 1, 12, 10, 30, 45)
        event = JobEvent(
            job_id="test-job",
            event_type="created",
            timestamp=timestamp,
        )

        data = event.to_dict()

        assert data["timestamp"] == "2025-01-12T10:30:45"

    def test_from_dict_uses_empty_dict_for_missing_details(self):
        """Test from_dict uses empty dict when details is missing."""
        data = {
            "job_id": "test-job",
            "event_type": "started",
            "timestamp": "2025-01-12T10:30:45",
        }

        event = JobEvent.from_dict(data)

        assert event.details == {}


class TestJob:
    """Tests for Job serialization."""

    def _create_sample_config(self) -> JobConfig:
        """Create a sample JobConfig for testing."""
        return JobConfig(
            sweep_type="balanced",
            n_agents=100,
            kappas=[Decimal("0.5"), Decimal("1.0")],
            concentrations=[Decimal("1.0")],
            mus=[Decimal("0.5")],
            cloud=True,
        )

    def test_to_dict_from_dict_round_trip(self):
        """Test to_dict/from_dict round-trip preserves data."""
        config = self._create_sample_config()
        created_at = datetime(2025, 1, 12, 10, 0, 0)
        completed_at = datetime(2025, 1, 12, 11, 30, 0)

        event = JobEvent(
            job_id="test-id",
            event_type="created",
            timestamp=created_at,
        )

        job = Job(
            job_id="test-id",
            created_at=created_at,
            status=JobStatus.COMPLETED,
            description="Test simulation",
            config=config,
            run_ids=["run_001", "run_002"],
            completed_at=completed_at,
            error=None,
            notes="Some notes",
            events=[event],
        )

        data = job.to_dict()
        restored = Job.from_dict(data)

        assert restored.job_id == job.job_id
        assert restored.created_at == job.created_at
        assert restored.status == job.status
        assert restored.description == job.description
        assert restored.config.sweep_type == job.config.sweep_type
        assert restored.config.n_agents == job.config.n_agents
        assert restored.run_ids == job.run_ids
        assert restored.completed_at == job.completed_at
        assert restored.error == job.error
        assert restored.notes == job.notes
        assert len(restored.events) == len(job.events)
        assert restored.events[0].event_type == job.events[0].event_type

    def test_to_dict_handles_none_values(self):
        """Test to_dict correctly handles None for optional fields."""
        config = self._create_sample_config()
        job = Job(
            job_id="test-id",
            created_at=datetime(2025, 1, 12, 10, 0, 0),
            status=JobStatus.PENDING,
            description="Test",
            config=config,
        )

        data = job.to_dict()

        assert data["completed_at"] is None
        assert data["error"] is None
        assert data["notes"] is None
        assert data["run_ids"] == []
        assert data["events"] == []

    def test_from_dict_handles_missing_optional_fields(self):
        """Test from_dict handles missing optional fields."""
        config = self._create_sample_config()
        data = {
            "job_id": "test-id",
            "created_at": "2025-01-12T10:00:00",
            "status": "pending",
            "description": "Test",
            "config": config.to_dict(),
        }

        job = Job.from_dict(data)

        assert job.run_ids == []
        assert job.completed_at is None
        assert job.error is None
        assert job.notes is None
        assert job.events == []

    def test_status_serialization(self):
        """Test JobStatus is serialized/deserialized correctly."""
        config = self._create_sample_config()

        for status in JobStatus:
            job = Job(
                job_id="test-id",
                created_at=datetime(2025, 1, 12, 10, 0, 0),
                status=status,
                description="Test",
                config=config,
            )

            data = job.to_dict()
            assert data["status"] == status.value

            restored = Job.from_dict(data)
            assert restored.status == status


class TestJobManager:
    """Tests for JobManager functionality."""

    def _create_sample_config(self) -> JobConfig:
        """Create a sample JobConfig for testing."""
        return JobConfig(
            sweep_type="ring",
            n_agents=50,
            kappas=[Decimal("1.0")],
            concentrations=[Decimal("1.0")],
            mus=[Decimal("0.5")],
        )

    def test_create_job_auto_generates_id(self):
        """Test create_job auto-generates ID when not provided."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test simulation",
            config=config,
        )

        # Should have a valid auto-generated ID
        assert validate_job_id(job.job_id) is True
        assert job.status == JobStatus.PENDING
        assert job.description == "Test simulation"

    def test_create_job_with_custom_id(self):
        """Test create_job with custom ID."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test simulation",
            config=config,
            job_id="custom-test-job-id",
        )

        assert job.job_id == "custom-test-job-id"

    def test_create_job_adds_created_event(self):
        """Test create_job adds a 'created' event."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test simulation",
            config=config,
        )

        assert len(job.events) == 1
        assert job.events[0].event_type == "created"
        assert job.events[0].job_id == job.job_id

    def test_start_job_updates_status(self):
        """Test start_job updates status to RUNNING."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test",
            config=config,
        )
        assert job.status == JobStatus.PENDING

        manager.start_job(job.job_id)

        updated_job = manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.RUNNING
        assert len(updated_job.events) == 2
        assert updated_job.events[1].event_type == "started"

    def test_start_job_raises_for_unknown_id(self):
        """Test start_job raises KeyError for unknown job ID."""
        manager = JobManager()

        with pytest.raises(KeyError, match="Job not found"):
            manager.start_job("nonexistent-job-id")

    def test_record_progress_adds_run_id(self):
        """Test record_progress adds run_id to the job."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test",
            config=config,
        )
        manager.start_job(job.job_id)

        manager.record_progress(job.job_id, "run_001", {"delta_total": 0.05})
        manager.record_progress(job.job_id, "run_002", {"delta_total": 0.03})

        updated_job = manager.get_job(job.job_id)
        assert updated_job.run_ids == ["run_001", "run_002"]
        assert len(updated_job.events) == 4  # created, started, progress, progress
        assert updated_job.events[2].event_type == "progress"
        assert updated_job.events[2].details["run_id"] == "run_001"

    def test_record_progress_raises_for_unknown_id(self):
        """Test record_progress raises KeyError for unknown job ID."""
        manager = JobManager()

        with pytest.raises(KeyError, match="Job not found"):
            manager.record_progress("nonexistent-job-id", "run_001")

    def test_complete_job_sets_completed_at(self):
        """Test complete_job sets completed_at timestamp."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test",
            config=config,
        )
        manager.start_job(job.job_id)

        manager.complete_job(job.job_id, {"total_runs": 10})

        updated_job = manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.completed_at is not None
        assert updated_job.events[-1].event_type == "completed"
        assert updated_job.events[-1].details["summary"] == {"total_runs": 10}

    def test_complete_job_raises_for_unknown_id(self):
        """Test complete_job raises KeyError for unknown job ID."""
        manager = JobManager()

        with pytest.raises(KeyError, match="Job not found"):
            manager.complete_job("nonexistent-job-id")

    def test_fail_job_sets_error(self):
        """Test fail_job sets error message and status."""
        manager = JobManager()
        config = self._create_sample_config()

        job = manager.create_job(
            description="Test",
            config=config,
        )
        manager.start_job(job.job_id)

        manager.fail_job(job.job_id, "Connection timeout")

        updated_job = manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error == "Connection timeout"
        assert updated_job.completed_at is not None
        assert updated_job.events[-1].event_type == "failed"
        assert updated_job.events[-1].details["error"] == "Connection timeout"

    def test_fail_job_raises_for_unknown_id(self):
        """Test fail_job raises KeyError for unknown job ID."""
        manager = JobManager()

        with pytest.raises(KeyError, match="Job not found"):
            manager.fail_job("nonexistent-job-id", "Some error")

    def test_get_job_returns_none_for_unknown_id(self):
        """Test get_job returns None for unknown ID."""
        manager = JobManager()

        result = manager.get_job("nonexistent-job-id")

        assert result is None

    def test_get_job_returns_job_by_id(self):
        """Test get_job returns the correct job by ID."""
        manager = JobManager()
        config = self._create_sample_config()

        job1 = manager.create_job(description="Job 1", config=config)
        job2 = manager.create_job(description="Job 2", config=config)

        retrieved = manager.get_job(job1.job_id)
        assert retrieved.job_id == job1.job_id
        assert retrieved.description == "Job 1"

        retrieved = manager.get_job(job2.job_id)
        assert retrieved.job_id == job2.job_id
        assert retrieved.description == "Job 2"

    def test_list_jobs_returns_all_jobs(self):
        """Test list_jobs returns all created jobs."""
        manager = JobManager()
        config = self._create_sample_config()

        job1 = manager.create_job(description="Job 1", config=config)
        job2 = manager.create_job(description="Job 2", config=config)
        job3 = manager.create_job(description="Job 3", config=config)

        jobs = manager.list_jobs()

        assert len(jobs) == 3
        job_ids = {j.job_id for j in jobs}
        assert job1.job_id in job_ids
        assert job2.job_id in job_ids
        assert job3.job_id in job_ids


class TestJobManagerPersistence:
    """Tests for JobManager persistence to disk."""

    def _create_sample_config(self) -> JobConfig:
        """Create a sample JobConfig for testing."""
        return JobConfig(
            sweep_type="ring",
            n_agents=50,
            kappas=[Decimal("1.0")],
            concentrations=[Decimal("1.0")],
            mus=[Decimal("0.5")],
        )

    def test_save_and_load_job_from_disk(self):
        """Test persistence: save job and load from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            config = self._create_sample_config()

            # Create job with first manager
            manager1 = JobManager(jobs_dir=jobs_dir)
            job = manager1.create_job(
                description="Persistent job",
                config=config,
                job_id="test-persistent-job-id",
            )
            manager1.start_job(job.job_id)
            manager1.record_progress(job.job_id, "run_001")
            manager1.complete_job(job.job_id)

            # Verify manifest file was created
            manifest_path = jobs_dir / "test-persistent-job-id" / "job_manifest.json"
            assert manifest_path.exists()

            # Load with new manager instance
            manager2 = JobManager(jobs_dir=jobs_dir)
            loaded_job = manager2.get_job("test-persistent-job-id")

            assert loaded_job is not None
            assert loaded_job.job_id == "test-persistent-job-id"
            assert loaded_job.description == "Persistent job"
            assert loaded_job.status == JobStatus.COMPLETED
            assert loaded_job.run_ids == ["run_001"]
            assert len(loaded_job.events) == 4  # created, started, progress, completed

    def test_list_jobs_loads_from_disk(self):
        """Test list_jobs loads jobs from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            config = self._create_sample_config()

            # Create jobs with first manager
            manager1 = JobManager(jobs_dir=jobs_dir)
            manager1.create_job(description="Job A", config=config, job_id="job-a-id")
            manager1.create_job(description="Job B", config=config, job_id="job-b-id")

            # List with new manager instance (should load from disk)
            manager2 = JobManager(jobs_dir=jobs_dir)
            jobs = manager2.list_jobs()

            assert len(jobs) == 2
            job_ids = {j.job_id for j in jobs}
            assert "job-a-id" in job_ids
            assert "job-b-id" in job_ids

    def test_manifest_json_structure(self):
        """Test manifest JSON file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            config = self._create_sample_config()

            manager = JobManager(jobs_dir=jobs_dir)
            manager.create_job(
                description="Test job",
                config=config,
                job_id="test-structure-job",
                notes="Test notes",
            )

            manifest_path = jobs_dir / "test-structure-job" / "job_manifest.json"
            with open(manifest_path) as f:
                data = json.load(f)

            assert data["job_id"] == "test-structure-job"
            assert data["description"] == "Test job"
            assert data["status"] == "pending"
            assert data["notes"] == "Test notes"
            assert "config" in data
            assert data["config"]["sweep_type"] == "ring"
            assert data["config"]["n_agents"] == 50
            assert "created_at" in data
            assert "events" in data

    def test_in_memory_only_when_no_jobs_dir(self):
        """Test jobs are in-memory only when jobs_dir is None."""
        config = self._create_sample_config()

        manager = JobManager(jobs_dir=None)
        job = manager.create_job(
            description="In-memory job",
            config=config,
        )

        # Should be retrievable from in-memory cache
        retrieved = manager.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.description == "In-memory job"

    def test_updates_persist_to_disk(self):
        """Test that job updates are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            config = self._create_sample_config()

            manager = JobManager(jobs_dir=jobs_dir)
            job = manager.create_job(
                description="Test",
                config=config,
                job_id="test-updates-id",
            )

            # Initial state
            manifest_path = jobs_dir / "test-updates-id" / "job_manifest.json"
            with open(manifest_path) as f:
                data = json.load(f)
            assert data["status"] == "pending"

            # After start
            manager.start_job(job.job_id)
            with open(manifest_path) as f:
                data = json.load(f)
            assert data["status"] == "running"

            # After completion
            manager.complete_job(job.job_id)
            with open(manifest_path) as f:
                data = json.load(f)
            assert data["status"] == "completed"
            assert data["completed_at"] is not None
