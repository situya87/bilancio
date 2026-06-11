"""Unit tests for the v2 ledger: audited operations and invariants."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bilancio_v2.ledger import InsufficientFundsError, InvariantViolation, Ledger


def make_ledger() -> Ledger:
    ledger = Ledger()
    ledger.register_agent("CB", "central_bank", "Central Bank")
    ledger.register_agent("BANK", "bank", "Bank")
    ledger.register_agent("ALICE", "household", "Alice")
    ledger.register_agent("BOB", "household", "Bob")
    return ledger


def test_mint_and_transfer_cash_conserves_total() -> None:
    ledger = make_ledger()
    ledger.mint_cash("ALICE", Decimal(1000))
    ledger.transfer_cash("ALICE", "BOB", Decimal(400))
    assert ledger.cash["ALICE"] == Decimal(600)
    assert ledger.cash["BOB"] == Decimal(400)
    ledger.check_invariants()


def test_transfer_more_than_balance_fails() -> None:
    ledger = make_ledger()
    ledger.mint_cash("ALICE", Decimal(100))
    with pytest.raises(InsufficientFundsError):
        ledger.transfer_cash("ALICE", "BOB", Decimal(101))


def test_deposit_and_withdraw_round_trip() -> None:
    ledger = make_ledger()
    ledger.mint_cash("ALICE", Decimal(500))
    ledger.deposit_cash("ALICE", "BANK", Decimal(300))
    assert ledger.deposits[("ALICE", "BANK")] == Decimal(300)
    assert ledger.cash["ALICE"] == Decimal(200)
    assert ledger.cash["BANK"] == Decimal(300)
    ledger.withdraw_cash("ALICE", "BANK", Decimal(300))
    assert ledger.deposits[("ALICE", "BANK")] == Decimal(0)
    assert ledger.cash["ALICE"] == Decimal(500)
    ledger.check_invariants()


def test_every_operation_is_journaled() -> None:
    ledger = make_ledger()
    ledger.mint_cash("ALICE", Decimal(500))
    ledger.deposit_cash("ALICE", "BANK", Decimal(300))
    kinds = [event.kind for event in ledger.journal]
    assert kinds == ["CashMinted", "CashDeposited"]


def test_checkpoint_rolls_back_balances_and_journal() -> None:
    ledger = make_ledger()
    ledger.mint_cash("ALICE", Decimal(500))
    checkpoint = ledger.checkpoint()
    ledger.transfer_cash("ALICE", "BOB", Decimal(200))
    ledger.deposit_cash("ALICE", "BANK", Decimal(100))
    ledger.restore(checkpoint)
    assert ledger.cash["ALICE"] == Decimal(500)
    assert ledger.cash["BOB"] == Decimal(0)
    assert ledger.deposits[("ALICE", "BANK")] == Decimal(0)
    assert [event.kind for event in ledger.journal] == ["CashMinted"]
    ledger.check_invariants()


def test_invariant_catches_unaudited_mutation() -> None:
    ledger = make_ledger()
    ledger.mint_cash("ALICE", Decimal(500))
    ledger.cash["ALICE"] += Decimal(1)  # corrupt state behind the ledger's back
    with pytest.raises(InvariantViolation):
        ledger.check_invariants()


def test_reserves_track_cb_outstanding() -> None:
    ledger = make_ledger()
    ledger.mint_reserves("BANK", Decimal(10_000))
    assert ledger.cb_reserves_outstanding == Decimal(10_000)
    ledger.check_invariants()
    ledger.reserves["BANK"] -= Decimal(5)  # corrupt
    with pytest.raises(InvariantViolation):
        ledger.check_invariants()
