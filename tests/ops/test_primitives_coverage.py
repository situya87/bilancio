"""Coverage tests for ops/primitives.py.

Targets uncovered lines: fungible_key, is_divisible edge cases,
split with extra fields, merge non-fungible error, consume with undo_log.
"""

from __future__ import annotations

import pytest

from bilancio.config.apply import create_agent
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.engines.system import System
from bilancio.ops.primitives import (
    coalesce_deposits,
    consume,
    fungible_key,
    is_divisible,
    merge,
    split,
)


def _make_system():
    """Create a system with CB and two households."""
    sys = System()
    cb = create_agent(type("S", (), {"id": "CB", "kind": "central_bank", "name": "CB"})())
    h1 = create_agent(type("S", (), {"id": "H1", "kind": "household", "name": "H1"})())
    h2 = create_agent(type("S", (), {"id": "H2", "kind": "household", "name": "H2"})())
    sys.add_agent(cb)
    sys.add_agent(h1)
    sys.add_agent(h2)
    return sys


class TestFungibleKey:
    """Cover fungible_key."""

    def test_returns_tuple(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        instr = sys.state.contracts[cid]
        key = fungible_key(instr)
        assert isinstance(key, tuple)
        assert len(key) == 4
        assert key[0] == InstrumentKind.CASH


class TestIsDivisible:
    """Cover is_divisible for different instrument types."""

    def test_cash_is_divisible(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        instr = sys.state.contracts[cid]
        assert is_divisible(instr) is True

    def test_payable_not_divisible(self):
        from bilancio.domain.instruments.credit import Payable

        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="H2",
            due_day=5,
        )
        assert is_divisible(payable) is False


class TestSplit:
    """Cover split function."""

    def test_basic_split(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        twin_id = split(sys, cid, 30)
        assert sys.state.contracts[cid].amount == 70
        assert sys.state.contracts[twin_id].amount == 30

    def test_split_invalid_amount_zero(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        with pytest.raises(Exception, match="invalid split amount"):
            split(sys, cid, 0)

    def test_split_amount_exceeds(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        with pytest.raises(Exception, match="invalid split amount"):
            split(sys, cid, 200)

    def test_split_non_divisible_raises(self):
        sys = _make_system()
        from bilancio.domain.instruments.credit import Payable

        payable = Payable(
            id="P1",
            kind=InstrumentKind.PAYABLE,
            amount=100,
            denom="X",
            asset_holder_id="H2",
            liability_issuer_id="H1",
            due_day=5,
        )
        sys.add_contract(payable)
        with pytest.raises(Exception, match="not divisible"):
            split(sys, "P1", 50)


class TestMerge:
    """Cover merge function."""

    def test_merge_same_id_returns_same(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        result = merge(sys, cid, cid)
        assert result == cid

    def test_merge_non_fungible_raises(self):
        sys = _make_system()
        c1 = sys.mint_cash("H1", 100)
        c2 = sys.mint_cash("H2", 50)
        with pytest.raises(Exception, match="not fungible"):
            merge(sys, c1, c2)


class TestConsume:
    """Cover consume function."""

    def test_consume_partial(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        consume(sys, cid, 30)
        assert sys.state.contracts[cid].amount == 70

    def test_consume_full_removes_contract(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        consume(sys, cid, 100)
        assert cid not in sys.state.contracts

    def test_consume_invalid_amount(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        with pytest.raises(Exception, match="invalid consume amount"):
            consume(sys, cid, 0)

    def test_consume_exceeds_amount(self):
        sys = _make_system()
        cid = sys.mint_cash("H1", 100)
        with pytest.raises(Exception, match="invalid consume amount"):
            consume(sys, cid, 200)


class TestCoalesceDeposits:
    """Cover coalesce_deposits."""

    def test_no_deposits_creates_zero_balance(self):
        sys = _make_system()
        from bilancio.config.apply import create_agent

        bank = create_agent(type("S", (), {"id": "B1", "kind": "bank", "name": "Bank1"})())
        sys.add_agent(bank)
        dep_id = coalesce_deposits(sys, "H1", "B1")
        assert sys.state.contracts[dep_id].amount == 0
