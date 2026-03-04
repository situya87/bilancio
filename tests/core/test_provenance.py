"""Tests for provenance collection (WI-6A)."""

from __future__ import annotations

from bilancio.provenance import (
    _compute_dep_fingerprint,
    _get_git_sha,
    collect_provenance,
)


class TestCollectProvenance:
    """Verify provenance collection returns expected structure."""

    def test_collect_provenance_keys(self) -> None:
        prov = collect_provenance()
        required_keys = {
            "git_sha",
            "git_dirty",
            "python_version",
            "platform",
            "cpu_count",
            "bilancio_version",
            "dep_fingerprint",
            "timestamp_utc",
        }
        assert required_keys.issubset(prov.keys())

    def test_dep_fingerprint_deterministic(self) -> None:
        fp1 = _compute_dep_fingerprint()
        fp2 = _compute_dep_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex digest

    def test_git_sha_format(self) -> None:
        sha = _get_git_sha()
        if sha is not None:
            assert len(sha) == 40
            assert all(c in "0123456789abcdef" for c in sha)

    def test_provenance_in_job(self) -> None:
        from bilancio.jobs import JobConfig, create_job_manager

        config = JobConfig(
            sweep_type="test",
            n_agents=10,
            kappas=[],
            concentrations=[],
            mus=[],
        )
        manager = create_job_manager()
        job = manager.create_job("test provenance", config)
        assert "git_sha" in job.provenance
        assert "python_version" in job.provenance
        assert "dep_fingerprint" in job.provenance
