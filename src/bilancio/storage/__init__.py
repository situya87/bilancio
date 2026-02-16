"""Storage abstractions for experiment results."""

from .artifact_loaders import ArtifactLoader, LocalArtifactLoader
from .file_store import FileRegistryStore, FileResultStore
from .modal_artifact_loader import ModalVolumeArtifactLoader
from .models import (
    RegistryEntry,
    RunArtifacts,
    RunResult,
    RunStatus,
)
from .protocols import RegistryStore, ResultStore
from .supabase_client import (
    SupabaseConfigError,
    get_supabase_client,
    is_supabase_configured,
)
from .supabase_registry import SupabaseRegistryStore

__all__ = [
    "RunStatus",
    "RunArtifacts",
    "RunResult",
    "RegistryEntry",
    "ResultStore",
    "RegistryStore",
    "FileResultStore",
    "FileRegistryStore",
    "ArtifactLoader",
    "LocalArtifactLoader",
    "ModalVolumeArtifactLoader",
    # Supabase
    "get_supabase_client",
    "is_supabase_configured",
    "SupabaseConfigError",
    "SupabaseRegistryStore",
]
