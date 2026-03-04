"""Job lifecycle management."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from bilancio.provenance import collect_provenance

from .job_id import generate_job_id
from .models import Job, JobConfig, JobEvent, JobStatus

if TYPE_CHECKING:
    from .supabase_store import SupabaseJobStore

logger = logging.getLogger(__name__)
EXTERNAL_STORE_ERRORS = (
    FileNotFoundError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
)


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp for job lifecycle events."""
    return datetime.now(UTC)


def create_job_manager(
    jobs_dir: Path | None = None,
    cloud: bool = False,
    local: bool = True,
) -> "JobManager":
    """Factory function to create a JobManager with optional cloud storage.

    Args:
        jobs_dir: Directory to store job manifests. If None and local=True,
                  jobs are only in-memory.
        cloud: If True, enable Supabase cloud storage (requires BILANCIO_SUPABASE_*
               environment variables to be set).
        local: If False, skip local file storage (useful for memory-constrained VMs).
               When cloud=True and local=False, jobs are only stored in Supabase.

    Returns:
        Configured JobManager instance.

    Example:
        >>> # Cloud-only (no local files)
        >>> manager = create_job_manager(cloud=True, local=False)
        >>> job = manager.create_job("Test job", config)
        >>>
        >>> # Both local and cloud
        >>> manager = create_job_manager(Path("./jobs"), cloud=True, local=True)
    """
    cloud_store = None
    if cloud:
        try:
            from bilancio.storage.supabase_client import is_supabase_configured

            if is_supabase_configured():
                from .supabase_store import SupabaseJobStore

                cloud_store = SupabaseJobStore()
                logger.info("Supabase cloud storage enabled for jobs")
            else:
                logger.warning(
                    "Cloud storage requested but Supabase not configured. "
                    "Set BILANCIO_SUPABASE_URL and BILANCIO_SUPABASE_ANON_KEY."
                )
        except ImportError as e:
            logger.warning(f"Failed to import Supabase: {e}")

    # If local=False, don't pass jobs_dir to avoid creating local files
    effective_jobs_dir = jobs_dir if local else None

    return JobManager(jobs_dir=effective_jobs_dir, cloud_store=cloud_store)


class JobManager:
    """Manages job lifecycle and persistence.

    Supports both local file storage and optional cloud storage via Supabase.
    When cloud_store is provided, jobs are persisted to both local and cloud.
    """

    def __init__(
        self,
        jobs_dir: Path | None = None,
        cloud_store: Optional["SupabaseJobStore"] = None,
    ):
        """Initialize the job manager.

        Args:
            jobs_dir: Directory to store job manifests. If None, jobs are only in-memory.
            cloud_store: Optional Supabase store for cloud persistence. Jobs will be
                        saved to both local and cloud when provided.
        """
        self.jobs_dir = jobs_dir
        self.cloud_store = cloud_store
        self._jobs: dict[str, Job] = {}

        if jobs_dir:
            jobs_dir.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        description: str,
        config: JobConfig,
        job_id: str | None = None,
        notes: str | None = None,
    ) -> Job:
        """Create a new job.

        Args:
            description: User's description of the simulation request
            config: Job configuration
            job_id: Optional custom job ID (auto-generated if not provided)
            notes: Optional additional notes

        Returns:
            The created Job instance
        """
        if job_id is None:
            job_id = generate_job_id()

        now = _utc_now()
        job = Job(
            job_id=job_id,
            created_at=now,
            status=JobStatus.PENDING,
            description=description,
            config=config,
            notes=notes,
            provenance=collect_provenance(),
        )

        # Record creation event
        event = JobEvent(
            job_id=job_id,
            event_type="created",
            timestamp=now,
            details={"config": config.to_dict()},
        )
        job.events.append(event)

        self._jobs[job_id] = job
        self._save_job(job)
        self._save_event(event)

        return job

    def start_job(self, job_id: str) -> None:
        """Mark a job as started.

        Args:
            job_id: The job ID to start

        Raises:
            KeyError: If job not found
        """
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")

        job.status = JobStatus.RUNNING
        event = JobEvent(
            job_id=job_id,
            event_type="started",
            timestamp=_utc_now(),
        )
        job.events.append(event)
        self._save_job(job)
        self._save_event(event)

    def record_progress(
        self,
        job_id: str,
        run_id: str,
        metrics: dict[str, Any] | None = None,
        modal_call_id: str | None = None,
    ) -> None:
        """Record progress on a job (a run completed).

        Args:
            job_id: The job ID
            run_id: The completed run ID
            metrics: Optional metrics from the run
            modal_call_id: Optional Modal function call ID for debugging

        Raises:
            KeyError: If job not found
        """
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")

        job.run_ids.append(run_id)
        if modal_call_id:
            job.modal_call_ids[run_id] = modal_call_id
        event = JobEvent(
            job_id=job_id,
            event_type="progress",
            timestamp=_utc_now(),
            details={"run_id": run_id, "metrics": metrics or {}, "modal_call_id": modal_call_id},
        )
        job.events.append(event)
        self._save_job(job)
        self._save_event(event)

    def complete_job(self, job_id: str, summary: dict[str, Any] | None = None) -> None:
        """Mark a job as completed.

        Args:
            job_id: The job ID
            summary: Optional summary statistics

        Raises:
            KeyError: If job not found
        """
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")

        now = _utc_now()
        job.status = JobStatus.COMPLETED
        job.completed_at = now

        event = JobEvent(
            job_id=job_id,
            event_type="completed",
            timestamp=now,
            details={"summary": summary or {}},
        )
        job.events.append(event)
        self._save_job(job)
        self._save_event(event)

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: The job ID
            error: Error message

        Raises:
            KeyError: If job not found
        """
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")

        now = _utc_now()
        job.status = JobStatus.FAILED
        job.completed_at = now
        job.error = error

        event = JobEvent(
            job_id=job_id,
            event_type="failed",
            timestamp=now,
            details={"error": error},
        )
        job.events.append(event)
        self._save_job(job)
        self._save_event(event)

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID.

        Args:
            job_id: The job ID

        Returns:
            The Job instance or None if not found
        """
        # Check in-memory cache first
        if job_id in self._jobs:
            return self._jobs[job_id]

        # Try to load from disk
        if self.jobs_dir:
            manifest_path = self.jobs_dir / job_id / "job_manifest.json"
            if manifest_path.exists():
                job = self._load_job(manifest_path)
                self._jobs[job_id] = job
                return job

        return None

    def list_jobs(self) -> list[Job]:
        """List all known jobs.

        Returns:
            List of Job instances
        """
        # Load any jobs from disk that aren't in memory
        if self.jobs_dir and self.jobs_dir.exists():
            for job_dir in self.jobs_dir.iterdir():
                if job_dir.is_dir():
                    job_id = job_dir.name
                    if job_id not in self._jobs:
                        manifest_path = job_dir / "job_manifest.json"
                        if manifest_path.exists():
                            job = self._load_job(manifest_path)
                            self._jobs[job_id] = job

        return list(self._jobs.values())

    def _save_job(self, job: Job) -> None:
        """Save job manifest to disk and optionally to cloud.

        Args:
            job: The job to save
        """
        # Save to local filesystem
        if self.jobs_dir is not None:
            job_dir = self.jobs_dir / job.job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = job_dir / "job_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(job.to_dict(), f, indent=2)

        # Save to cloud store if configured
        if self.cloud_store is not None:
            try:
                self.cloud_store.save_job(job)
            except EXTERNAL_STORE_ERRORS as e:
                logger.warning(f"Failed to save job to cloud: {e}")

    def _save_event(self, event: JobEvent) -> None:
        """Save a job event to cloud storage.

        Events are saved to Supabase for queryable history.
        Local events are stored in the job manifest itself.

        Args:
            event: The event to save
        """
        if self.cloud_store is not None:
            try:
                self.cloud_store.save_event(event)
            except EXTERNAL_STORE_ERRORS as e:
                logger.warning(f"Failed to save event to cloud: {e}")

    def _load_job(self, manifest_path: Path) -> Job:
        """Load a job from a manifest file.

        Args:
            manifest_path: Path to job_manifest.json

        Returns:
            The loaded Job instance
        """
        with open(manifest_path) as f:
            data = json.load(f)
        return Job.from_dict(data)
