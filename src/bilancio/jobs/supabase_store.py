"""Supabase storage backend for job persistence.

This module provides a SupabaseJobStore class that persists jobs and events
to a Supabase PostgreSQL database for durable, queryable storage.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from .models import Job, JobConfig, JobEvent, JobStatus

logger = logging.getLogger(__name__)
SUPABASE_OPERATION_ERRORS = (
    FileNotFoundError,
    ImportError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
)


class SupabaseJobStore:
    """Persists jobs and events to Supabase.

    This store provides durable storage for job metadata, allowing jobs to be
    queried across sessions and from different environments (local, Modal, etc).

    If Supabase is unavailable, operations fail gracefully with warnings rather
    than raising exceptions, allowing the application to continue with local
    storage fallback.
    """

    def __init__(self, client: Any | None = None) -> None:
        """Initialize the Supabase job store.

        Args:
            client: Optional Supabase client instance. If not provided,
                   attempts to get one from the supabase_client module.
        """
        self._client = client
        self._initialized = False

    @property
    def client(self) -> Any | None:
        """Lazily initialize and return the Supabase client."""
        if self._client is not None:
            return self._client

        if self._initialized:
            return None

        self._initialized = True

        try:
            from bilancio.storage.supabase_client import (
                get_supabase_client,
                is_supabase_configured,
            )

            if not is_supabase_configured():
                logger.debug("Supabase is not configured")
                return None

            self._client = get_supabase_client()
            return self._client
        except ImportError:
            logger.warning("supabase_client module not found")
            return None
        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to initialize Supabase client: {e}")
            return None

    def save_job(self, job: Job) -> None:
        """Save or update a job to Supabase.

        Maps the Job dataclass to the database schema, handling type conversions
        for arrays, decimals, and timestamps.

        Args:
            job: The Job instance to save.
        """
        if self.client is None:
            logger.debug("Supabase unavailable, skipping job save")
            return

        try:
            # Map Job fields to database columns
            data = {
                "job_id": job.job_id,
                "created_at": job.created_at.isoformat(),
                "status": job.status.value,
                "description": job.description,
                # Config fields (flattened into job table)
                "sweep_type": job.config.sweep_type,
                "n_agents": job.config.n_agents,
                "maturity_days": job.config.maturity_days,
                "kappas": [str(k) for k in job.config.kappas],
                "concentrations": [str(c) for c in job.config.concentrations],
                "mus": [str(m) for m in job.config.mus],
                "outside_mid_ratios": [str(r) for r in job.config.outside_mid_ratios],
                "seeds": job.config.seeds,
                "cloud": job.config.cloud,
                # Optional fields
                "notes": job.notes,
                "error": job.error,
                "total_runs": len(job.run_ids) if job.run_ids else None,
                "completed_runs": len(job.run_ids) if job.status == JobStatus.COMPLETED else 0,
            }

            # Add completed_at if present
            if job.completed_at:
                data["completed_at"] = job.completed_at.isoformat()

            # Upsert the job (insert or update on conflict)
            self.client.table("jobs").upsert(data, on_conflict="job_id").execute()

            logger.debug(f"Saved job {job.job_id} to Supabase")

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to save job to Supabase: {e}")

    def save_event(self, event: JobEvent) -> None:
        """Save a job event to Supabase.

        Args:
            event: The JobEvent instance to save.
        """
        if self.client is None:
            logger.debug("Supabase unavailable, skipping event save")
            return

        try:
            data = {
                "job_id": event.job_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
            }

            self.client.table("job_events").insert(data).execute()

            logger.debug(f"Saved event {event.event_type} for job {event.job_id}")

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to save event to Supabase: {e}")

    def get_job(self, job_id: str) -> Job | None:
        """Load a job from Supabase by ID.

        Args:
            job_id: The unique job identifier.

        Returns:
            The Job instance if found, None otherwise.
        """
        if self.client is None:
            logger.debug("Supabase unavailable, cannot get job")
            return None

        try:
            response = self.client.table("jobs").select("*").eq("job_id", job_id).single().execute()

            if not response.data:
                return None

            return self._row_to_job(response.data)

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to get job from Supabase: {e}")
            return None

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs from Supabase with optional filtering.

        Args:
            status: Optional status filter (e.g., "running", "completed").
            limit: Maximum number of jobs to return (default 100).

        Returns:
            List of Job instances matching the criteria.
        """
        if self.client is None:
            logger.debug("Supabase unavailable, returning empty list")
            return []

        try:
            query = self.client.table("jobs").select("*").order("created_at", desc=True).limit(limit)

            if status:
                query = query.eq("status", status)

            response = query.execute()

            if not response.data:
                return []

            return [self._row_to_job(row) for row in response.data]

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to list jobs from Supabase: {e}")
            return []

    def get_run_counts(self, job_ids: list[str]) -> dict[str, int]:
        """Get run counts for multiple jobs.

        Args:
            job_ids: List of job IDs to count runs for.

        Returns:
            Dict mapping job_id to run count.
        """
        if self.client is None or not job_ids:
            return {}

        try:
            # Query runs table and count by job_id
            response = self.client.table("runs").select("job_id").in_("job_id", job_ids).execute()

            if not response.data:
                return {}

            # Count runs per job
            counts: dict[str, int] = {}
            for row in response.data:
                job_id = row["job_id"]
                counts[job_id] = counts.get(job_id, 0) + 1

            return counts

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to get run counts from Supabase: {e}")
            return {}

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        completed_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status of a job.

        This is a lightweight update method that only modifies status-related
        fields without requiring a full Job object.

        Args:
            job_id: The job ID to update.
            status: The new status.
            completed_at: Optional completion timestamp.
            error: Optional error message (for failed status).
        """
        if self.client is None:
            logger.debug("Supabase unavailable, skipping status update")
            return

        try:
            data: dict[str, Any] = {"status": status.value}

            if completed_at:
                data["completed_at"] = completed_at.isoformat()

            if error:
                data["error"] = error

            self.client.table("jobs").update(data).eq("job_id", job_id).execute()

            logger.debug(f"Updated job {job_id} status to {status.value}")

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to update job status in Supabase: {e}")

    def _row_to_job(self, row: dict[str, Any]) -> Job:
        """Convert a database row to a Job instance.

        Args:
            row: Dictionary containing database row data.

        Returns:
            A Job instance populated from the row data.
        """
        # Parse arrays - Supabase returns them as lists already
        kappas = [Decimal(str(k)) for k in (row.get("kappas") or [])]
        concentrations = [Decimal(str(c)) for c in (row.get("concentrations") or [])]
        mus = [Decimal(str(m)) for m in (row.get("mus") or [])]
        outside_mid_ratios = [Decimal(str(r)) for r in (row.get("outside_mid_ratios") or ["1"])]
        seeds = row.get("seeds") or [42]

        # Build JobConfig
        config = JobConfig(
            sweep_type=row["sweep_type"],
            n_agents=row["n_agents"],
            kappas=kappas,
            concentrations=concentrations,
            mus=mus,
            cloud=row.get("cloud", False),
            outside_mid_ratios=outside_mid_ratios,
            maturity_days=row.get("maturity_days") or 5,
            seeds=seeds,
        )

        # Parse timestamps
        created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        completed_at = None
        if row.get("completed_at"):
            completed_at = datetime.fromisoformat(row["completed_at"].replace("Z", "+00:00"))

        # Build Job (events are loaded separately if needed)
        return Job(
            job_id=row["job_id"],
            created_at=created_at,
            status=JobStatus(row["status"]),
            description=row.get("description") or "",
            config=config,
            run_ids=[],  # Run IDs would need separate query to runs table
            modal_call_ids={},  # Modal call IDs stored per-run, not in jobs table
            completed_at=completed_at,
            error=row.get("error"),
            notes=row.get("notes"),
            events=[],  # Events loaded separately if needed
        )

    def get_events(self, job_id: str) -> list[JobEvent]:
        """Load all events for a job from Supabase.

        Args:
            job_id: The job ID to get events for.

        Returns:
            List of JobEvent instances, ordered by timestamp.
        """
        if self.client is None:
            logger.debug("Supabase unavailable, returning empty events list")
            return []

        try:
            response = self.client.table("job_events").select("*").eq("job_id", job_id).order("timestamp", desc=False).execute()

            if not response.data:
                return []

            events = []
            for row in response.data:
                timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                events.append(
                    JobEvent(
                        job_id=row["job_id"],
                        event_type=row["event_type"],
                        timestamp=timestamp,
                        details=row.get("details") or {},
                    )
                )

            return events

        except SUPABASE_OPERATION_ERRORS as e:
            logger.warning(f"Failed to get events from Supabase: {e}")
            return []
