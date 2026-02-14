from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Any

if TYPE_CHECKING:
    from bilancio.engines.system import System


@contextmanager
def atomic(system: System) -> Generator[None, None, None]:
    """Context manager for atomic operations — rollback on failure.

    Creates a ``copy.deepcopy`` snapshot of ``system.state`` before the
    block executes and restores it if any exception is raised inside.
    The deepcopy is O(n) in the size of the state (agents, contracts,
    events) so it dominates wall-clock time in tight settlement loops.

    Future optimisation paths include copy-on-write proxies, journal-based
    undo logs, or structural sharing (only copy the mutated sub-trees).
    """
    snapshot = copy.deepcopy(system.state)
    try:
        yield
    except Exception:  # Intentionally broad: must rollback on any failure
        system.state = snapshot
        raise
