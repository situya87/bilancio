from __future__ import annotations

import copy
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilancio.engines.system import System


@contextmanager
def atomic(system: System, *, use_undo_log: bool = False) -> Generator[None, None, None]:
    """Context manager for atomic operations -- rollback on failure.

    Creates a ``copy.deepcopy`` snapshot of ``system.state`` before the
    block executes and restores it if any exception is raised inside.
    The deepcopy is O(n) in the size of the state (agents, contracts,
    events) so it dominates wall-clock time in tight settlement loops.

    When ``system._atomic_disabled`` is set (by the settlement engine
    in expel-agent mode), the deepcopy is skipped entirely -- no rollback
    is possible, but the O(n) cost per operation is eliminated.

    When *use_undo_log* is ``True``, an :class:`UndoLog` is installed on
    the system instead of taking a deepcopy snapshot.  Instrumented
    mutation methods record their changes into the log and, on failure,
    the log replays in reverse -- O(K) where K is the number of actual
    mutations (typically 2-5) rather than O(N) for the full state.
    **Scaffold**: only ``mint_cash`` and ``retire_cash`` are instrumented.
    """
    if getattr(system, "_atomic_disabled", False):
        yield
        return

    if use_undo_log:
        from bilancio.core.undo_log import UndoLog

        log = UndoLog()
        prev_log = getattr(system, "_undo_log", None)
        system._undo_log = log  # type: ignore[attr-defined]
        try:
            yield
        except BaseException:
            log.rollback()
            raise
        finally:
            log.clear()
            system._undo_log = prev_log  # type: ignore[attr-defined]
        return

    # Original deepcopy path
    snapshot = copy.deepcopy(system.state)
    try:
        yield
    except BaseException:  # Roll back on any failure, including interrupts.
        system.state = snapshot
        raise
