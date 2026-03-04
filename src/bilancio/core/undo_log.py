"""Field-level undo log for targeted rollback.

Instead of ``copy.deepcopy(system.state)`` before each atomic
operation, mutations are recorded one-by-one and replayed in reverse
on rollback.  This turns the O(N) snapshot into an O(K) replay where
K is the number of actual mutations in the block (typically 2-5 for a
single settlement operation, vs N = thousands for a full state copy).

**Status: scaffold** -- only ``mint_cash`` and ``retire_cash`` are
instrumented.  Full instrumentation of all System mutation methods is
a follow-up project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UndoEntry:
    """A single reversible mutation."""

    kind: str  # "setattr" | "set_add" | "set_remove" | "dict_set" | "dict_del" | "list_append"
    obj_id: int  # id() of the mutated object
    obj_ref: Any  # weak-ish ref (kept alive by the caller's scope)
    key: str  # field name, dict key, or ""
    old_value: Any = None  # previous value (for setattr / dict_set)


class UndoLog:
    """Accumulates undo entries and can replay them in reverse.

    Usage::

        log = UndoLog()
        log.record_setattr(agent, "cash", agent.cash)
        agent.cash += 100
        ...
        log.rollback()  # agent.cash back to original
    """

    __slots__ = ("_entries", "_active")

    def __init__(self) -> None:
        self._entries: list[UndoEntry] = []
        self._active: bool = True

    # -- Recording helpers --------------------------------------------------

    def record_setattr(self, obj: Any, attr: str, old_value: Any) -> None:
        """Record that *obj.attr* is about to change from *old_value*."""
        if not self._active:
            return
        self._entries.append(
            UndoEntry(kind="setattr", obj_id=id(obj), obj_ref=obj, key=attr, old_value=old_value)
        )

    def record_set_add(self, the_set: set, item: Any) -> None:
        """Record that *item* is about to be added to *the_set*."""
        if not self._active:
            return
        self._entries.append(
            UndoEntry(kind="set_add", obj_id=id(the_set), obj_ref=the_set, key="", old_value=item)
        )

    def record_set_remove(self, the_set: set, item: Any) -> None:
        """Record that *item* is about to be removed from *the_set*."""
        if not self._active:
            return
        self._entries.append(
            UndoEntry(
                kind="set_remove", obj_id=id(the_set), obj_ref=the_set, key="", old_value=item
            )
        )

    def record_dict_set(self, the_dict: dict, key: str, old_value: Any) -> None:
        """Record that *the_dict[key]* is about to be set (old_value is the previous value or _MISSING)."""
        if not self._active:
            return
        self._entries.append(
            UndoEntry(
                kind="dict_set", obj_id=id(the_dict), obj_ref=the_dict, key=key, old_value=old_value
            )
        )

    def record_dict_del(self, the_dict: dict, key: str, old_value: Any) -> None:
        """Record that *the_dict[key]* (with value *old_value*) is about to be deleted."""
        if not self._active:
            return
        self._entries.append(
            UndoEntry(
                kind="dict_del", obj_id=id(the_dict), obj_ref=the_dict, key=key, old_value=old_value
            )
        )

    def record_list_append(self, the_list: list, item: Any) -> None:
        """Record that *item* is about to be appended to *the_list*."""
        if not self._active:
            return
        self._entries.append(
            UndoEntry(
                kind="list_append", obj_id=id(the_list), obj_ref=the_list, key="", old_value=item
            )
        )

    # -- Rollback -----------------------------------------------------------

    def rollback(self) -> int:
        """Replay entries in reverse, undoing each mutation.

        Returns the number of entries replayed.
        """
        self._active = False  # prevent recording during rollback
        count = 0
        for entry in reversed(self._entries):
            _undo_one(entry)
            count += 1
        self._entries.clear()
        self._active = True
        return count

    def clear(self) -> None:
        """Discard all entries (commit succeeded, no rollback needed)."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# Sentinel for "key didn't exist in dict before"
_MISSING = object()


def _undo_one(entry: UndoEntry) -> None:
    """Undo a single mutation entry."""
    if entry.kind == "setattr":
        setattr(entry.obj_ref, entry.key, entry.old_value)
    elif entry.kind == "set_add":
        entry.obj_ref.discard(entry.old_value)
    elif entry.kind == "set_remove":
        entry.obj_ref.add(entry.old_value)
    elif entry.kind == "dict_set":
        if entry.old_value is _MISSING:
            entry.obj_ref.pop(entry.key, None)
        else:
            entry.obj_ref[entry.key] = entry.old_value
    elif entry.kind == "dict_del":
        entry.obj_ref[entry.key] = entry.old_value
    elif entry.kind == "list_append":
        # Remove the last element (assumes append was the last op on this list)
        if entry.obj_ref and entry.obj_ref[-1] is entry.old_value:
            entry.obj_ref.pop()
