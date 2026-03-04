from __future__ import annotations

import copy
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System


@contextmanager
def atomic(system: System) -> Generator[None, None, None]:
    """Context manager for atomic operations — rollback on failure.

    Creates a ``copy.deepcopy`` snapshot of ``system.state`` before the
    block executes and restores it if any exception is raised inside.
    The deepcopy is O(n) in the size of the state (agents, contracts,
    events) so it dominates wall-clock time in tight settlement loops.

    When ``system._atomic_disabled`` is set (by the settlement engine
    in expel-agent mode), the deepcopy is skipped entirely — no rollback
    is possible, but the O(n) cost per operation is eliminated.
    """
    if getattr(system, "_atomic_disabled", False):
        yield
        return
    snapshot = copy.deepcopy(system.state)
    try:
        yield
    except BaseException:  # Roll back on any failure, including interrupts.
        system.state = snapshot
        raise
