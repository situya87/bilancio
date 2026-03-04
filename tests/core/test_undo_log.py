"""Tests for bilancio.core.undo_log -- field-level undo log for targeted rollback."""

from dataclasses import dataclass

from bilancio.core.undo_log import UndoLog, _MISSING


@dataclass
class Obj:
    """Minimal mutable object for testing setattr rollback."""

    cash: float = 0.0
    name: str = ""


# -- 1. record_setattr + rollback restores original value --------------------


def test_setattr_rollback_restores_value():
    obj = Obj(cash=100.0)
    log = UndoLog()

    log.record_setattr(obj, "cash", obj.cash)
    obj.cash = 999.0

    assert obj.cash == 999.0
    count = log.rollback()
    assert obj.cash == 100.0
    assert count == 1


# -- 2. record_set_add + rollback removes the item --------------------------


def test_set_add_rollback_removes_item():
    s = {1, 2, 3}
    log = UndoLog()

    log.record_set_add(s, 4)
    s.add(4)

    assert 4 in s
    log.rollback()
    assert 4 not in s
    assert s == {1, 2, 3}


# -- 3. record_set_remove + rollback re-adds the item -----------------------


def test_set_remove_rollback_readds_item():
    s = {1, 2, 3}
    log = UndoLog()

    log.record_set_remove(s, 2)
    s.remove(2)

    assert 2 not in s
    log.rollback()
    assert 2 in s
    assert s == {1, 2, 3}


# -- 4. record_dict_set (existing key) + rollback restores old value ---------


def test_dict_set_existing_key_rollback():
    d = {"a": 10, "b": 20}
    log = UndoLog()

    log.record_dict_set(d, "a", d["a"])
    d["a"] = 999

    assert d["a"] == 999
    log.rollback()
    assert d["a"] == 10


# -- 5. record_dict_set (new key, _MISSING) + rollback removes key ----------


def test_dict_set_new_key_rollback_removes():
    d = {"a": 10}
    log = UndoLog()

    log.record_dict_set(d, "new_key", _MISSING)
    d["new_key"] = 42

    assert "new_key" in d
    log.rollback()
    assert "new_key" not in d
    assert d == {"a": 10}


# -- 6. record_dict_del + rollback restores key ------------------------------


def test_dict_del_rollback_restores_key():
    d = {"a": 10, "b": 20}
    log = UndoLog()

    log.record_dict_del(d, "b", d["b"])
    del d["b"]

    assert "b" not in d
    log.rollback()
    assert d["b"] == 20


# -- 7. record_list_append + rollback pops last element ---------------------


def test_list_append_rollback_pops():
    lst = [1, 2, 3]
    log = UndoLog()

    item = 99
    log.record_list_append(lst, item)
    lst.append(item)

    assert lst == [1, 2, 3, 99]
    log.rollback()
    assert lst == [1, 2, 3]


# -- 8. Multi-mutation rollback (several ops, all reversed) ------------------


def test_multi_mutation_rollback():
    obj = Obj(cash=50.0, name="alice")
    s = {"x"}
    d = {"k": 1}
    lst = [10]
    log = UndoLog()

    # 1) setattr cash
    log.record_setattr(obj, "cash", obj.cash)
    obj.cash = 200.0

    # 2) setattr name
    log.record_setattr(obj, "name", obj.name)
    obj.name = "bob"

    # 3) set add
    log.record_set_add(s, "y")
    s.add("y")

    # 4) dict set new key
    log.record_dict_set(d, "k2", _MISSING)
    d["k2"] = 2

    # 5) list append
    item = 20
    log.record_list_append(lst, item)
    lst.append(item)

    assert len(log) == 5
    count = log.rollback()
    assert count == 5

    # Everything restored
    assert obj.cash == 50.0
    assert obj.name == "alice"
    assert s == {"x"}
    assert d == {"k": 1}
    assert lst == [10]


# -- 9. Empty log rollback returns 0 ----------------------------------------


def test_empty_rollback_returns_zero():
    log = UndoLog()
    assert log.rollback() == 0


# -- 10. clear() discards entries, subsequent rollback does nothing ----------


def test_clear_discards_entries():
    obj = Obj(cash=100.0)
    log = UndoLog()

    log.record_setattr(obj, "cash", obj.cash)
    obj.cash = 0.0

    assert len(log) == 1
    log.clear()
    assert len(log) == 0

    count = log.rollback()
    assert count == 0
    # The mutation sticks because the log was cleared before rollback.
    assert obj.cash == 0.0


# -- 11. __len__ tracks entry count ------------------------------------------


def test_len_tracks_entries():
    log = UndoLog()
    assert len(log) == 0

    obj = Obj()
    log.record_setattr(obj, "cash", 0.0)
    assert len(log) == 1

    s = set()
    log.record_set_add(s, 1)
    assert len(log) == 2

    log.record_set_remove(s, 1)
    assert len(log) == 3

    d: dict = {}
    log.record_dict_set(d, "k", _MISSING)
    assert len(log) == 4

    log.record_dict_del(d, "k", 1)
    assert len(log) == 5

    lst: list = []
    log.record_list_append(lst, "item")
    assert len(log) == 6


# -- 12. Inactive log does not record new entries during rollback ------------


def test_inactive_during_rollback():
    """Entries recorded inside a rollback callback are silently dropped.

    We test this indirectly: during rollback, _active is False.
    If something called record_setattr at that point it would be ignored.
    We simulate this by manually deactivating the log.
    """
    log = UndoLog()
    obj = Obj(cash=100.0)

    # Deactivate to simulate the state during rollback
    log._active = False
    log.record_setattr(obj, "cash", obj.cash)
    assert len(log) == 0  # nothing recorded

    # Same for every other record method
    log.record_set_add(set(), 1)
    log.record_set_remove(set(), 1)
    log.record_dict_set({}, "k", _MISSING)
    log.record_dict_del({}, "k", 1)
    log.record_list_append([], "x")
    assert len(log) == 0


# -- 13. rollback re-activates recording ------------------------------------


def test_rollback_reactivates_recording():
    log = UndoLog()
    obj = Obj(cash=50.0)

    log.record_setattr(obj, "cash", obj.cash)
    obj.cash = 200.0
    log.rollback()

    assert obj.cash == 50.0

    # After rollback, recording is active again
    log.record_setattr(obj, "cash", obj.cash)
    obj.cash = 300.0
    assert len(log) == 1

    log.rollback()
    assert obj.cash == 50.0
