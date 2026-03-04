"""Tests for bilancio.core.atomic_tx module.

Covers:
- atomic() context manager with deepcopy (default path)
- atomic() with _atomic_disabled flag (no-op path)
- atomic() with use_undo_log=True (undo log path)
- Rollback on exception in each path
- Successful commit in each path
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bilancio.core.atomic_tx import atomic
from bilancio.engines.system import System


class TestAtomicDeepCopy:
    """Test the default deepcopy path."""

    def test_successful_block_preserves_changes(self):
        """When block succeeds, state changes persist."""
        system = System()
        system.state.day = 0
        with atomic(system):
            system.state.day = 5
        assert system.state.day == 5

    def test_exception_rolls_back_state(self):
        """When block raises, state is rolled back to snapshot."""
        system = System()
        system.state.day = 0
        with pytest.raises(ValueError):
            with atomic(system):
                system.state.day = 99
                raise ValueError("test error")
        assert system.state.day == 0

    def test_keyboard_interrupt_rolls_back(self):
        """BaseException (like KeyboardInterrupt) also triggers rollback."""
        system = System()
        system.state.day = 0
        with pytest.raises(KeyboardInterrupt):
            with atomic(system):
                system.state.day = 42
                raise KeyboardInterrupt()
        assert system.state.day == 0


class TestAtomicDisabled:
    """Test the _atomic_disabled path (no snapshot, no rollback)."""

    def test_disabled_skips_deepcopy(self):
        """When _atomic_disabled is True, atomic is a no-op wrapper."""
        system = System()
        system._atomic_disabled = True  # type: ignore[attr-defined]
        system.state.day = 0
        with atomic(system):
            system.state.day = 10
        assert system.state.day == 10

    def test_disabled_no_rollback_on_exception(self):
        """When _atomic_disabled is True, no rollback occurs."""
        system = System()
        system._atomic_disabled = True  # type: ignore[attr-defined]
        system.state.day = 0
        with pytest.raises(ValueError):
            with atomic(system):
                system.state.day = 77
                raise ValueError("boom")
        # State is NOT rolled back because atomic is disabled
        assert system.state.day == 77


class TestAtomicUndoLog:
    """Test the use_undo_log=True path."""

    def test_undo_log_successful_block(self):
        """When undo log block succeeds, no rollback occurs, log is cleared."""
        system = System()
        system.state.day = 0

        with atomic(system, use_undo_log=True):
            system.state.day = 15

        assert system.state.day == 15
        # _undo_log should be restored to None (prev_log)
        assert getattr(system, "_undo_log", None) is None

    def test_undo_log_exception_triggers_rollback(self):
        """When undo log block raises, rollback() is called and exception re-raised."""
        from bilancio.core.undo_log import UndoLog

        system = System()
        system.state.day = 0

        with pytest.raises(RuntimeError):
            with atomic(system, use_undo_log=True):
                # Manually record state change in the undo log
                log = system._undo_log  # type: ignore[attr-defined]
                log.record_setattr(system.state, "day", system.state.day)
                system.state.day = 99
                raise RuntimeError("fail")

        # The undo log should have rolled back
        assert system.state.day == 0
        # _undo_log should be restored to None (prev_log)
        assert getattr(system, "_undo_log", None) is None

    def test_undo_log_preserves_previous_log(self):
        """atomic(use_undo_log=True) preserves a previously set _undo_log."""
        from bilancio.core.undo_log import UndoLog

        system = System()
        prev_log = UndoLog()
        system._undo_log = prev_log  # type: ignore[attr-defined]

        with atomic(system, use_undo_log=True):
            # Inside, a new log is installed
            assert system._undo_log is not prev_log  # type: ignore[attr-defined]

        # After, previous log is restored
        assert system._undo_log is prev_log  # type: ignore[attr-defined]

    def test_undo_log_clears_on_success(self):
        """After successful block, the undo log's clear() is called."""
        from bilancio.core.undo_log import UndoLog

        system = System()

        with atomic(system, use_undo_log=True):
            log = system._undo_log  # type: ignore[attr-defined]
            log.record_setattr(system.state, "day", system.state.day)
            system.state.day = 50

        # Log was cleared, and state persists
        assert system.state.day == 50
