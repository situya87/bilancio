"""Coverage tests for bilancio.jobs.manager.

Focuses on uncovered branches:
- create_job_manager factory with cloud=True (import paths, Supabase not configured)
- JobManager with cloud_store error handling
- list_jobs from disk
- get_job from disk
- start_job / fail_job / complete_job / record_progress on missing jobs
- _save_job / _save_event cloud error paths
- _load_job from manifest on disk
"""

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bilancio.jobs.manager import JobManager, create_job_manager
from bilancio.jobs.models import Job, JobConfig, JobEvent, JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> JobConfig:
    return JobConfig(
        sweep_type="ring",
        n_agents=10,
        kappas=[Decimal("0.5")],
        concentrations=[Decimal("1")],
        mus=[Decimal("0")],
    )


# ---------------------------------------------------------------------------
# create_job_manager factory
# ---------------------------------------------------------------------------


class TestCreateJobManager:
    """Tests for the create_job_manager factory function."""

    def test_local_only_default(self):
        """Default call returns a manager with no cloud store."""
        mgr = create_job_manager()
        assert mgr.cloud_store is None
        assert mgr.jobs_dir is None

    def test_local_with_jobs_dir(self, tmp_path):
        """Passing jobs_dir creates the directory."""
        d = tmp_path / "jobs"
        mgr = create_job_manager(jobs_dir=d)
        assert mgr.jobs_dir == d
        assert d.exists()

    def test_cloud_true_but_not_configured(self):
        """cloud=True but Supabase not configured logs warning."""
        with patch(
            "bilancio.storage.supabase_client.is_supabase_configured",
            return_value=False,
        ):
            mgr = create_job_manager(cloud=True, local=False)
        assert mgr.cloud_store is None

    def test_cloud_true_import_error(self, caplog):
        """cloud=True with ImportError logs warning gracefully."""
        with patch(
            "bilancio.jobs.manager.is_supabase_configured",
            side_effect=ImportError("no supabase"),
            create=True,
        ):
            # The actual code catches ImportError at the outer level
            with patch.dict("sys.modules", {"bilancio.storage.supabase_client": None}):
                mgr = create_job_manager(cloud=True, local=False)
        assert mgr.cloud_store is None

    def test_local_false_skips_jobs_dir(self, tmp_path):
        """local=False means jobs_dir is not passed through."""
        d = tmp_path / "jobs"
        mgr = create_job_manager(jobs_dir=d, local=False)
        assert mgr.jobs_dir is None


# ---------------------------------------------------------------------------
# JobManager lifecycle
# ---------------------------------------------------------------------------


class TestJobManagerLifecycle:
    """Tests for create / start / progress / complete / fail."""

    def test_create_job_custom_id(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), job_id="my-custom-id")
        assert job.job_id == "my-custom-id"
        assert job.status == JobStatus.PENDING
        assert len(job.events) == 1
        assert job.events[0].event_type == "created"

    def test_create_job_auto_id(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config())
        assert job.job_id  # auto-generated, non-empty
        assert "-" in job.job_id  # xkcdpass format

    def test_create_job_with_notes(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), notes="some note")
        assert job.notes == "some note"

    def test_start_job(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), job_id="start-test")
        mgr.start_job("start-test")
        assert job.status == JobStatus.RUNNING
        assert len(job.events) == 2
        assert job.events[1].event_type == "started"

    def test_start_job_not_found(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        with pytest.raises(KeyError, match="Job not found"):
            mgr.start_job("nonexistent")

    def test_record_progress(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), job_id="prog-test")
        mgr.start_job("prog-test")
        mgr.record_progress("prog-test", "run-1", metrics={"delta": 0.5})
        assert "run-1" in job.run_ids
        assert job.events[-1].event_type == "progress"
        assert job.events[-1].details["metrics"] == {"delta": 0.5}

    def test_record_progress_with_modal_call_id(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), job_id="modal-test")
        mgr.record_progress("modal-test", "run-1", modal_call_id="mc-123")
        assert job.modal_call_ids["run-1"] == "mc-123"

    def test_record_progress_not_found(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        with pytest.raises(KeyError, match="Job not found"):
            mgr.record_progress("nonexistent", "run-1")

    def test_complete_job(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), job_id="comp-test")
        mgr.start_job("comp-test")
        mgr.complete_job("comp-test", summary={"total_runs": 5})
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.events[-1].event_type == "completed"
        assert job.events[-1].details["summary"]["total_runs"] == 5

    def test_complete_job_not_found(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        with pytest.raises(KeyError, match="Job not found"):
            mgr.complete_job("nonexistent")

    def test_fail_job(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job("test", _make_config(), job_id="fail-test")
        mgr.start_job("fail-test")
        mgr.fail_job("fail-test", error="something broke")
        assert job.status == JobStatus.FAILED
        assert job.error == "something broke"
        assert job.completed_at is not None
        assert job.events[-1].event_type == "failed"

    def test_fail_job_not_found(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        with pytest.raises(KeyError, match="Job not found"):
            mgr.fail_job("nonexistent", error="err")


# ---------------------------------------------------------------------------
# Persistence: disk load/save
# ---------------------------------------------------------------------------


class TestJobManagerPersistence:
    """Tests for saving to and loading from disk."""

    def test_get_job_from_disk(self, tmp_path):
        """get_job loads from disk when not in memory."""
        mgr1 = JobManager(jobs_dir=tmp_path)
        job = mgr1.create_job("persist test", _make_config(), job_id="disk-test")
        mgr1.start_job("disk-test")

        # Create a second manager pointing to same dir (empty in-memory cache)
        mgr2 = JobManager(jobs_dir=tmp_path)
        loaded = mgr2.get_job("disk-test")
        assert loaded is not None
        assert loaded.job_id == "disk-test"
        assert loaded.status == JobStatus.RUNNING

    def test_get_job_returns_none_not_on_disk(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        assert mgr.get_job("does-not-exist") is None

    def test_get_job_in_memory_only(self):
        """Without jobs_dir, only in-memory works."""
        mgr = JobManager()
        job = mgr.create_job("mem test", _make_config(), job_id="mem-only")
        assert mgr.get_job("mem-only") is not None
        assert mgr.get_job("something-else") is None

    def test_list_jobs_from_disk(self, tmp_path):
        """list_jobs discovers jobs from disk."""
        mgr1 = JobManager(jobs_dir=tmp_path)
        mgr1.create_job("job A", _make_config(), job_id="list-a")
        mgr1.create_job("job B", _make_config(), job_id="list-b")

        # New manager, empty cache
        mgr2 = JobManager(jobs_dir=tmp_path)
        jobs = mgr2.list_jobs()
        ids = {j.job_id for j in jobs}
        assert "list-a" in ids
        assert "list-b" in ids

    def test_list_jobs_skips_non_dirs(self, tmp_path):
        """list_jobs ignores non-directory entries."""
        mgr = JobManager(jobs_dir=tmp_path)
        # Create a plain file in the jobs dir
        (tmp_path / "not-a-job.txt").write_text("noise")
        jobs = mgr.list_jobs()
        assert len(jobs) == 0

    def test_list_jobs_empty(self, tmp_path):
        mgr = JobManager(jobs_dir=tmp_path)
        assert mgr.list_jobs() == []

    def test_list_jobs_no_dir(self):
        mgr = JobManager()
        assert mgr.list_jobs() == []

    def test_manifest_round_trip(self, tmp_path):
        """Job saved to disk can be loaded and matches original."""
        mgr = JobManager(jobs_dir=tmp_path)
        job = mgr.create_job(
            "round-trip",
            _make_config(),
            job_id="rt-test",
            notes="my note",
        )
        mgr.start_job("rt-test")
        mgr.record_progress("rt-test", "run-1", metrics={"delta": 0.1})
        mgr.complete_job("rt-test", summary={"done": True})

        # Read the manifest file back
        manifest_path = tmp_path / "rt-test" / "job_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        loaded = Job.from_dict(data)
        assert loaded.job_id == "rt-test"
        assert loaded.status == JobStatus.COMPLETED
        assert loaded.notes == "my note"
        assert "run-1" in loaded.run_ids
        assert len(loaded.events) == 4  # created, started, progress, completed


# ---------------------------------------------------------------------------
# Cloud store error handling
# ---------------------------------------------------------------------------


class TestCloudStoreErrors:
    """Tests for cloud store error paths in _save_job and _save_event."""

    def test_save_job_cloud_error_logged(self, tmp_path, caplog):
        """Cloud store failure in _save_job is caught and logged."""
        cloud = MagicMock()
        cloud.save_job.side_effect = ConnectionError("cloud down")

        mgr = JobManager(jobs_dir=tmp_path, cloud_store=cloud)
        with caplog.at_level(logging.WARNING):
            job = mgr.create_job("cloud-err", _make_config(), job_id="cloud-err")

        # Job should still be created locally
        assert job.job_id == "cloud-err"
        assert "Failed to save job to cloud" in caplog.text

    def test_save_event_cloud_error_logged(self, tmp_path, caplog):
        """Cloud store failure in _save_event is caught and logged."""
        cloud = MagicMock()
        cloud.save_job.return_value = None  # save_job succeeds
        cloud.save_event.side_effect = TimeoutError("timeout")

        mgr = JobManager(jobs_dir=tmp_path, cloud_store=cloud)
        with caplog.at_level(logging.WARNING):
            mgr.create_job("event-err", _make_config(), job_id="event-err")

        assert "Failed to save event to cloud" in caplog.text

    def test_save_job_cloud_runtime_error(self, tmp_path, caplog):
        """RuntimeError from cloud store is caught."""
        cloud = MagicMock()
        cloud.save_job.side_effect = RuntimeError("unexpected")

        mgr = JobManager(jobs_dir=tmp_path, cloud_store=cloud)
        with caplog.at_level(logging.WARNING):
            mgr.create_job("rt-err", _make_config(), job_id="rt-err")
        assert "Failed to save job to cloud" in caplog.text

    def test_save_event_cloud_value_error(self, tmp_path, caplog):
        """ValueError from cloud event store is caught."""
        cloud = MagicMock()
        cloud.save_job.return_value = None
        cloud.save_event.side_effect = ValueError("bad value")

        mgr = JobManager(jobs_dir=tmp_path, cloud_store=cloud)
        with caplog.at_level(logging.WARNING):
            mgr.create_job("val-err", _make_config(), job_id="val-err")
        assert "Failed to save event to cloud" in caplog.text

    def test_cloud_store_save_job_success(self, tmp_path):
        """Cloud store called successfully on job save."""
        cloud = MagicMock()
        mgr = JobManager(jobs_dir=tmp_path, cloud_store=cloud)
        mgr.create_job("ok-cloud", _make_config(), job_id="ok-cloud")
        cloud.save_job.assert_called_once()
        cloud.save_event.assert_called_once()

    def test_cloud_store_receives_all_events(self, tmp_path):
        """Cloud store receives events for each lifecycle step."""
        cloud = MagicMock()
        mgr = JobManager(jobs_dir=tmp_path, cloud_store=cloud)
        mgr.create_job("all-events", _make_config(), job_id="all-events")
        mgr.start_job("all-events")
        mgr.record_progress("all-events", "run-1")
        mgr.complete_job("all-events")

        # 4 saves: create, start, progress, complete
        assert cloud.save_job.call_count == 4
        assert cloud.save_event.call_count == 4
