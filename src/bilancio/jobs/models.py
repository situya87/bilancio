"""Job system data models."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(Enum):
    """Status of a simulation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobConfig:
    """Configuration for a simulation job."""

    sweep_type: str  # "ring", "balanced", "single"
    n_agents: int
    kappas: List[Decimal]
    concentrations: List[Decimal]
    mus: List[Decimal]
    cloud: bool = False
    outside_mid_ratios: List[Decimal] = field(default_factory=lambda: [Decimal("1")])
    maturity_days: int = 5
    seeds: List[int] = field(default_factory=lambda: [42])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sweep_type": self.sweep_type,
            "n_agents": self.n_agents,
            "kappas": [str(k) for k in self.kappas],
            "concentrations": [str(c) for c in self.concentrations],
            "mus": [str(m) for m in self.mus],
            "cloud": self.cloud,
            "outside_mid_ratios": [str(r) for r in self.outside_mid_ratios],
            "maturity_days": self.maturity_days,
            "seeds": self.seeds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobConfig":
        """Create from dictionary."""
        return cls(
            sweep_type=data["sweep_type"],
            n_agents=data["n_agents"],
            kappas=[Decimal(k) for k in data["kappas"]],
            concentrations=[Decimal(c) for c in data["concentrations"]],
            mus=[Decimal(m) for m in data["mus"]],
            cloud=data.get("cloud", False),
            outside_mid_ratios=[
                Decimal(r) for r in data.get("outside_mid_ratios", ["1"])
            ],
            maturity_days=data.get("maturity_days", 5),
            seeds=data.get("seeds", [42]),
        )


@dataclass
class JobEvent:
    """An event in the job lifecycle."""

    job_id: str
    event_type: str  # "created", "started", "progress", "completed", "failed"
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobEvent":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data.get("details", {}),
        )


@dataclass
class Job:
    """A simulation job with lifecycle tracking."""

    job_id: str
    created_at: datetime
    status: JobStatus
    description: str
    config: JobConfig
    run_ids: List[str] = field(default_factory=list)
    modal_call_ids: Dict[str, str] = field(default_factory=dict)  # run_id -> modal_call_id
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    notes: Optional[str] = None
    events: List[JobEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "description": self.description,
            "config": self.config.to_dict(),
            "run_ids": self.run_ids,
            "modal_call_ids": self.modal_call_ids,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "notes": self.notes,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=JobStatus(data["status"]),
            description=data["description"],
            config=JobConfig.from_dict(data["config"]),
            run_ids=data.get("run_ids", []),
            modal_call_ids=data.get("modal_call_ids", {}),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            error=data.get("error"),
            notes=data.get("notes"),
            events=[JobEvent.from_dict(e) for e in data.get("events", [])],
        )
