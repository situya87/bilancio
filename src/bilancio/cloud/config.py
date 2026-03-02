"""Configuration for cloud execution."""

import os
from dataclasses import dataclass, field


@dataclass
class CloudConfig:
    """Configuration for cloud execution.

    Attributes:
        volume_name: Name of the Modal Volume for storing results.
            Note: This must match the volume name in modal_app.py (hardcoded as
            "bilancio-results"). Changing this without also updating modal_app.py
            will cause download failures.
        timeout_seconds: Maximum execution time per simulation.
        memory_mb: Memory allocation in MB.
        max_parallel: Maximum concurrent Modal function calls.
        gpu: Optional GPU type (e.g., "T4", "A10G", "A100").
    """

    # Note: This must match the volume name hardcoded in modal_app.py
    volume_name: str = field(
        default_factory=lambda: os.getenv("BILANCIO_MODAL_VOLUME", "bilancio-results")
    )
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("BILANCIO_CLOUD_TIMEOUT", "600"))
    )
    memory_mb: int = field(default_factory=lambda: int(os.getenv("BILANCIO_CLOUD_MEMORY", "2048")))
    max_parallel: int = field(
        default_factory=lambda: int(os.getenv("BILANCIO_CLOUD_MAX_PARALLEL", "10"))
    )

    # GPU configuration (for future ML workloads)
    gpu: str | None = None
