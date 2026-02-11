from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Any

if TYPE_CHECKING:
    from bilancio.engines.system import System


@contextmanager
def atomic(system: System) -> Generator[None, None, None]:
    """Context manager for atomic operations - rollback on failure"""
    snapshot = copy.deepcopy(system.state)
    try:
        yield
    except Exception:
        system.state = snapshot
        raise
